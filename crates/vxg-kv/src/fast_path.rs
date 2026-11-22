//! Zero-allocation, direct-to-wire RESP2 batch dispatcher operating directly on `&[u8]` byte
//! spans inside [`RespBuffer`] (eliminating per-command `Vec<Vec<u8>>`, `String`, and `RespVal`
//! heap allocations on all hot paths).

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::{Arc, RwLock};

use crate::dispatch::{dispatch_command, exec_single, ConnState};
use crate::dispatch_ds::{format_float, WRONGTYPE};
use crate::engine::KvEngine;
use crate::pubsub::PubSubBroker;
use crate::types::Value;

#[inline]
fn push_usize_dec(out: &mut Vec<u8>, mut n: usize) {
    let mut tmp = [0u8; 20];
    let mut i = 20;
    if n == 0 {
        out.push(b'0');
        return;
    }
    while n > 0 {
        i -= 1;
        tmp[i] = b'0' + (n % 10) as u8;
        n /= 10;
    }
    out.extend_from_slice(&tmp[i..]);
}

#[inline]
fn push_i64_dec(out: &mut Vec<u8>, n: i64) {
    if n < 0 {
        out.push(b'-');
        push_usize_dec(out, n.unsigned_abs() as usize);
    } else {
        push_usize_dec(out, n as usize);
    }
}

#[inline]
fn write_bulk(out: &mut Vec<u8>, b: &[u8]) {
    out.push(b'$');
    push_usize_dec(out, b.len());
    out.extend_from_slice(b"\r\n");
    out.extend_from_slice(b);
    out.extend_from_slice(b"\r\n");
}

#[inline]
fn write_int(out: &mut Vec<u8>, n: i64) {
    out.push(b':');
    push_i64_dec(out, n);
    out.extend_from_slice(b"\r\n");
}

#[inline]
fn write_wrongtype(out: &mut Vec<u8>) {
    out.push(b'-');
    out.extend_from_slice(WRONGTYPE.as_bytes());
    out.extend_from_slice(b"\r\n");
}

#[inline]
fn parse_i64_fast(b: &[u8]) -> Option<i64> {
    if b.is_empty() {
        return None;
    }
    let (neg, digits) = if b[0] == b'-' {
        (true, &b[1..])
    } else {
        (false, b)
    };
    if digits.is_empty() {
        return None;
    }
    let mut acc: i64 = 0;
    for &c in digits {
        if !c.is_ascii_digit() {
            return None;
        }
        acc = acc.checked_mul(10)?.checked_add((c - b'0') as i64)?;
    }
    Some(if neg { acc.wrapping_neg() } else { acc })
}

#[inline]
fn is_control_slice(b: &[u8]) -> bool {
    const CTRL: &[&[u8]] = &[
        b"SUBSCRIBE", b"UNSUBSCRIBE", b"PSUBSCRIBE", b"PUNSUBSCRIBE", b"PUBSUB", b"QUIT",
        b"MULTI", b"EXEC", b"DISCARD", b"WATCH", b"UNWATCH",
    ];
    CTRL.iter().any(|k| b.eq_ignore_ascii_case(k))
}

#[inline]
fn is_ro_slice(b: &[u8]) -> bool {
    const RO: &[&[u8]] = &[
        b"PING", b"ECHO", b"GET", b"EXISTS", b"STRLEN", b"HGET", b"LLEN", b"SCARD", b"ZSCORE",
    ];
    RO.iter().any(|k| b.eq_ignore_ascii_case(k))
}

#[inline]
fn arg<'a>(buf: &'a [u8], spans: &[(usize, usize)], idx: usize) -> &'a [u8] {
    let (s, e) = spans[idx];
    &buf[s..e]
}

/// Try zero-alloc read-only execution under `RwLock::read()`. Returns `false` on cold cache miss.
fn try_ro_spans(eng: &KvEngine, buf: &[u8], spans: &[(usize, usize)], out: &mut Vec<u8>) -> bool {
    let argc = spans.len();
    let c = arg(buf, spans, 0);
    if c.eq_ignore_ascii_case(b"PING") {
        if argc == 1 {
            out.extend_from_slice(b"+PONG\r\n");
        } else {
            write_bulk(out, arg(buf, spans, 1));
        }
        return true;
    }
    if c.eq_ignore_ascii_case(b"GET") && argc == 2 {
        let Some(rec_opt) = eng.get_record_ref(arg(buf, spans, 1)) else {
            return false;
        };
        match rec_opt {
            None => out.extend_from_slice(b"$-1\r\n"),
            Some(r) => match r.value.as_ref() {
                Some(Value::Str(b)) => write_bulk(out, b),
                None => out.extend_from_slice(b"$-1\r\n"),
                _ => write_wrongtype(out),
            },
        }
        return true;
    }
    if c.eq_ignore_ascii_case(b"ECHO") && argc == 2 {
        write_bulk(out, arg(buf, spans, 1));
        return true;
    }
    if c.eq_ignore_ascii_case(b"EXISTS") && argc == 2 {
        let Some(rec_opt) = eng.get_record_ref(arg(buf, spans, 1)) else {
            return false;
        };
        write_int(out, i64::from(rec_opt.is_some()));
        return true;
    }
    if c.eq_ignore_ascii_case(b"STRLEN") && argc == 2 {
        let Some(rec_opt) = eng.get_record_ref(arg(buf, spans, 1)) else {
            return false;
        };
        match rec_opt {
            None => write_int(out, 0),
            Some(r) => match r.value.as_ref() {
                Some(Value::Str(b)) => write_int(out, b.len() as i64),
                _ => write_wrongtype(out),
            },
        }
        return true;
    }
    if c.eq_ignore_ascii_case(b"HGET") && argc == 3 {
        let Some(rec_opt) = eng.get_record_ref(arg(buf, spans, 1)) else {
            return false;
        };
        match rec_opt {
            None => out.extend_from_slice(b"$-1\r\n"),
            Some(r) => match r.value.as_ref() {
                Some(Value::Hash(m)) => match m.get(arg(buf, spans, 2)) {
                    Some(v) => write_bulk(out, v),
                    None => out.extend_from_slice(b"$-1\r\n"),
                },
                _ => write_wrongtype(out),
            },
        }
        return true;
    }
    if c.eq_ignore_ascii_case(b"LLEN") && argc == 2 {
        let Some(rec_opt) = eng.get_record_ref(arg(buf, spans, 1)) else {
            return false;
        };
        match rec_opt {
            None => write_int(out, 0),
            Some(r) => match r.value.as_ref() {
                Some(Value::List(l)) => write_int(out, l.len() as i64),
                _ => write_wrongtype(out),
            },
        }
        return true;
    }
    if c.eq_ignore_ascii_case(b"SCARD") && argc == 2 {
        let Some(rec_opt) = eng.get_record_ref(arg(buf, spans, 1)) else {
            return false;
        };
        match rec_opt {
            None => write_int(out, 0),
            Some(r) => match r.value.as_ref() {
                Some(Value::Set(s)) => write_int(out, s.len() as i64),
                _ => write_wrongtype(out),
            },
        }
        return true;
    }
    if c.eq_ignore_ascii_case(b"ZSCORE") && argc == 3 {
        let Some(rec_opt) = eng.get_record_ref(arg(buf, spans, 1)) else {
            return false;
        };
        match rec_opt {
            None => out.extend_from_slice(b"$-1\r\n"),
            Some(r) => match r.value.as_ref() {
                Some(Value::ZSet(z)) => match z.get(arg(buf, spans, 2)) {
                    Some(&sc) => write_bulk(out, format_float(sc).as_bytes()),
                    None => out.extend_from_slice(b"$-1\r\n"),
                },
                _ => write_wrongtype(out),
            },
        }
        return true;
    }
    false
}

/// Try zero-allocation hot write/read execution directly from byte spans under `RwLock::write()`.
/// Returns `false` if command needs the full general dispatcher (options, rare commands, etc.).
fn try_rw_fast_spans(
    eng: &mut KvEngine,
    buf: &[u8],
    spans: &[(usize, usize)],
    out: &mut Vec<u8>,
) -> bool {
    let argc = spans.len();
    let c = arg(buf, spans, 0);

    // Plain `SET key val` (3 args): ZERO heap alloc on overwrites!
    if c.eq_ignore_ascii_case(b"SET") && argc == 3 {
        eng.set_str_in_place(arg(buf, spans, 1), arg(buf, spans, 2));
        out.extend_from_slice(b"+OK\r\n");
        return true;
    }
    if c.eq_ignore_ascii_case(b"GET") && argc == 2 {
        match eng.get_record_or_load(arg(buf, spans, 1)) {
            Ok(None) => out.extend_from_slice(b"$-1\r\n"),
            Ok(Some(r)) => match r.value.as_ref() {
                Some(Value::Str(b)) => write_bulk(out, b),
                None => out.extend_from_slice(b"$-1\r\n"),
                _ => write_wrongtype(out),
            },
            Err(e) => {
                out.push(b'-');
                out.extend_from_slice(e.as_bytes());
                out.extend_from_slice(b"\r\n");
            }
        }
        return true;
    }
    if c.eq_ignore_ascii_case(b"PING") && argc == 1 {
        out.extend_from_slice(b"+PONG\r\n");
        return true;
    }
    if c.eq_ignore_ascii_case(b"MSET") && argc >= 3 && (argc - 1).is_multiple_of(2) {
        let mut i = 1;
        while i < argc {
            eng.set_str_in_place(arg(buf, spans, i), arg(buf, spans, i + 1));
            i += 2;
        }
        out.extend_from_slice(b"+OK\r\n");
        return true;
    }
    if (c.eq_ignore_ascii_case(b"INCR") || c.eq_ignore_ascii_case(b"DECR")) && argc == 2 {
        let delta = if c[0] == b'I' || c[0] == b'i' { 1 } else { -1 };
        let Ok(rec) = eng.ensure_entry_mut(arg(buf, spans, 1), || Value::Str(b"0".to_vec())) else {
            return false;
        };
        let Some(Value::Str(vbuf)) = rec.value.as_mut() else {
            write_wrongtype(out);
            return true;
        };
        let Some(cur) = parse_i64_fast(vbuf) else {
            return false;
        };
        let next = cur.saturating_add(delta);
        vbuf.clear();
        push_i64_dec(vbuf, next);
        write_int(out, next);
        return true;
    }
    if (c.eq_ignore_ascii_case(b"LPUSH") || c.eq_ignore_ascii_case(b"RPUSH")) && argc >= 3 {
        let left = c[0] == b'L' || c[0] == b'l';
        let Ok(rec) = eng.ensure_entry_mut(arg(buf, spans, 1), || Value::List(VecDeque::new())) else {
            return false;
        };
        let Some(Value::List(l)) = rec.value.as_mut() else {
            write_wrongtype(out);
            return true;
        };
        for i in 2..argc {
            let item = arg(buf, spans, i).to_vec();
            if left {
                l.push_front(item);
            } else {
                l.push_back(item);
            }
        }
        write_int(out, l.len() as i64);
        return true;
    }
    if (c.eq_ignore_ascii_case(b"LPOP") || c.eq_ignore_ascii_case(b"RPOP")) && argc == 2 {
        let left = c[0] == b'L' || c[0] == b'l';
        let key = arg(buf, spans, 1);
        let (popped, empty) = {
            let Ok(rec_opt) = eng.get_or_promote_mut(key) else {
                return false;
            };
            let Some(rec) = rec_opt else {
                out.extend_from_slice(b"$-1\r\n");
                return true;
            };
            let Some(Value::List(l)) = rec.value.as_mut() else {
                write_wrongtype(out);
                return true;
            };
            let val = if left { l.pop_front() } else { l.pop_back() };
            (val, l.is_empty())
        };
        if empty {
            let _ = eng.del_key(key);
        }
        match popped {
            Some(v) => write_bulk(out, &v),
            None => out.extend_from_slice(b"$-1\r\n"),
        }
        return true;
    }
    if c.eq_ignore_ascii_case(b"SADD") && argc >= 3 {
        let Ok(rec) = eng.ensure_entry_mut(arg(buf, spans, 1), || Value::Set(BTreeSet::new())) else {
            return false;
        };
        let Some(Value::Set(s)) = rec.value.as_mut() else {
            write_wrongtype(out);
            return true;
        };
        let mut added = 0i64;
        for i in 2..argc {
            let m = arg(buf, spans, i);
            if !s.contains(m) {
                s.insert(m.to_vec());
                added += 1;
            }
        }
        write_int(out, added);
        return true;
    }
    if c.eq_ignore_ascii_case(b"HSET") && argc >= 4 && argc.is_multiple_of(2) {
        let Ok(rec) = eng.ensure_entry_mut(arg(buf, spans, 1), || Value::Hash(BTreeMap::new())) else {
            return false;
        };
        let Some(Value::Hash(m)) = rec.value.as_mut() else {
            write_wrongtype(out);
            return true;
        };
        let mut added = 0i64;
        let mut i = 2;
        while i < argc {
            let f = arg(buf, spans, i);
            let v = arg(buf, spans, i + 1);
            if let Some(slot) = m.get_mut(f) {
                slot.clear();
                slot.extend_from_slice(v);
            } else {
                m.insert(f.to_vec(), v.to_vec());
                added += 1;
            }
            i += 2;
        }
        write_int(out, added);
        return true;
    }
    if c.eq_ignore_ascii_case(b"ZADD") && argc == 4 {
        let sc_b = arg(buf, spans, 2);
        let Ok(sc) = std::str::from_utf8(sc_b).ok().and_then(|s| s.parse::<f64>().ok()).ok_or(()) else {
            return false;
        };
        if sc.is_nan() {
            return false;
        }
        let Ok(rec) = eng.ensure_entry_mut(arg(buf, spans, 1), || Value::ZSet(BTreeMap::new())) else {
            return false;
        };
        let Some(Value::ZSet(z)) = rec.value.as_mut() else {
            write_wrongtype(out);
            return true;
        };
        let m = arg(buf, spans, 3);
        let added = if let Some(slot) = z.get_mut(m) {
            *slot = sc;
            0
        } else {
            z.insert(m.to_vec(), sc);
            1
        };
        write_int(out, added);
        return true;
    }
    false
}

fn materialize_args(buf: &[u8], spans: &[(usize, usize)]) -> Vec<Vec<u8>> {
    spans.iter().map(|&(s, e)| buf[s..e].to_vec()).collect()
}

/// Execute an entire zero-allocation parsed batch directly from `buf` spans into `out`.
pub fn dispatch_span_batch(
    engine: &Arc<RwLock<KvEngine>>,
    broker: &Arc<PubSubBroker>,
    conn: &mut ConnState,
    buf: &[u8],
    arg_spans: &[(usize, usize)],
    cmd_ranges: &[(usize, usize)],
    out: &mut Vec<u8>,
) {
    if cmd_ranges.is_empty() {
        return;
    }
    // Fast Path 1: 100% read-only hot batch under 1 shared `RwLock::read()` (parallel across cores!)
    if !conn.in_multi
        && conn.pinned_epoch.is_none()
        && cmd_ranges
            .iter()
            .all(|&(a, b)| a < b && is_ro_slice(arg(buf, &arg_spans[a..b], 0)))
    {
        let saved = out.len();
        let eng = engine.read().unwrap();
        let mut all_ok = true;
        for &(a, b) in cmd_ranges {
            if !try_ro_spans(&eng, buf, &arg_spans[a..b], out) {
                all_ok = false;
                break;
            }
        }
        drop(eng);
        if all_ok {
            return;
        }
        out.truncate(saved);
    }

    // Fast Path 2: single `RwLock::write()` for entire data-only pipeline batch!
    if !conn.in_multi
        && conn.pinned_epoch.is_none()
        && cmd_ranges
            .iter()
            .all(|&(a, b)| a < b && !is_control_slice(arg(buf, &arg_spans[a..b], 0)))
    {
        let mut eng = engine.write().unwrap();
        for &(a, b) in cmd_ranges {
            let spans = &arg_spans[a..b];
            if !try_rw_fast_spans(&mut eng, buf, spans, out) {
                let owned = materialize_args(buf, spans);
                let cmd = String::from_utf8_lossy(&owned[0]).to_ascii_uppercase();
                exec_single(&mut eng, conn, &cmd, &owned).encode(out);
            }
        }
        return;
    }

    // Fallback: control / transaction / pinned Time-Travel / PubSub commands
    for &(a, b) in cmd_ranges {
        let owned = materialize_args(buf, &arg_spans[a..b]);
        for reply in dispatch_command(engine, broker, conn, owned) {
            reply.encode(out);
        }
        if conn.quit {
            break;
        }
    }
}
