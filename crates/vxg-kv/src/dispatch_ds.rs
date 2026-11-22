//! Redis non-deprecated Hash (`H*`) and List (`L*`/`R*`) command dispatchers (zero-clone in-place).

use std::collections::{BTreeMap, VecDeque};

use crate::dispatch_stream::dispatch_stream_cmd;
use crate::dispatch_zset::dispatch_set_zset_cmd;
use crate::engine::KvEngine;
use crate::resp::RespVal;
use crate::types::Value;

pub(crate) const WRONGTYPE: &str =
    "WRONGTYPE Operation against a key holding the wrong kind of value";

pub(crate) fn arg_str(args: &[Vec<u8>], idx: usize) -> Option<&str> {
    args.get(idx).and_then(|b| std::str::from_utf8(b).ok())
}

pub(crate) fn parse_i64(b: &[u8]) -> Result<i64, RespVal> {
    std::str::from_utf8(b)
        .ok()
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| RespVal::err("ERR value is not an integer or out of range"))
}

pub(crate) fn parse_f64(b: &[u8]) -> Result<f64, RespVal> {
    let f: f64 = std::str::from_utf8(b)
        .ok()
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| RespVal::err("ERR value is not a valid float"))?;
    if f.is_nan() {
        return Err(RespVal::err("ERR resulting score is not a number (NaN)"));
    }
    Ok(f)
}

pub(crate) fn format_float(v: f64) -> String {
    if v.fract() == 0.0 && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        format!("{v}")
    }
}

pub(crate) fn norm_range(start: i64, stop: i64, len: usize) -> Option<(usize, usize)> {
    if len == 0 {
        return None;
    }
    let l = len as i64;
    let mut s = if start < 0 { l + start } else { start };
    let mut e = if stop < 0 { l + stop } else { stop };
    if s < 0 {
        s = 0;
    }
    if e >= l {
        e = l - 1;
    }
    if s > e || s >= l {
        return None;
    }
    Some((s as usize, e as usize))
}

type HashRef<'a> = Option<&'a BTreeMap<Vec<u8>, Vec<u8>>>;

fn hash_ref<'a>(eng: &'a mut KvEngine, key: &[u8]) -> Result<HashRef<'a>, RespVal> {
    match eng.get_record_or_load(key).map_err(RespVal::err)? {
        None => Ok(None),
        Some(r) => match r.value.as_ref() {
            Some(Value::Hash(m)) => Ok(Some(m)),
            _ => Err(RespVal::err(WRONGTYPE)),
        },
    }
}

fn list_ref<'a>(eng: &'a mut KvEngine, key: &[u8]) -> Result<Option<&'a VecDeque<Vec<u8>>>, RespVal> {
    match eng.get_record_or_load(key).map_err(RespVal::err)? {
        None => Ok(None),
        Some(r) => match r.value.as_ref() {
            Some(Value::List(l)) => Ok(Some(l)),
            _ => Err(RespVal::err(WRONGTYPE)),
        },
    }
}

/// Dispatch Hash, List, Set, ZSet, Stream, Bitmap, and HLL commands.
pub fn dispatch_ds_cmd(eng: &mut KvEngine, cmd: &str, args: &[Vec<u8>]) -> Option<RespVal> {
    if let Some(r) = dispatch_set_zset_cmd(eng, cmd, args) {
        return Some(r);
    }
    if let Some(r) = dispatch_stream_cmd(eng, cmd, args) {
        return Some(r);
    }
    Some(match cmd {
        // ---- HASH ----
        "HSET" => {
            if args.len() < 4 || !args.len().is_multiple_of(2) {
                return Some(RespVal::err("ERR wrong number of arguments for 'hset'"));
            }
            let rec = match eng.ensure_entry_mut(&args[1], || Value::Hash(BTreeMap::new())) {
                Ok(r) => r,
                Err(e) => return Some(RespVal::err(e)),
            };
            let Some(Value::Hash(m)) = rec.value.as_mut() else {
                return Some(RespVal::err(WRONGTYPE));
            };
            let mut added = 0i64;
            for pair in args[2..].as_chunks::<2>().0 {
                if m.insert(pair[0].clone(), pair[1].clone()).is_none() {
                    added += 1;
                }
            }
            RespVal::int(added)
        }
        "HSETNX" => {
            if args.len() != 4 {
                return Some(RespVal::err("ERR wrong number of arguments for 'hsetnx'"));
            }
            let rec = match eng.ensure_entry_mut(&args[1], || Value::Hash(BTreeMap::new())) {
                Ok(r) => r,
                Err(e) => return Some(RespVal::err(e)),
            };
            let Some(Value::Hash(m)) = rec.value.as_mut() else {
                return Some(RespVal::err(WRONGTYPE));
            };
            if m.contains_key(&args[2]) {
                RespVal::int(0)
            } else {
                m.insert(args[2].clone(), args[3].clone());
                RespVal::int(1)
            }
        }
        "HGET" => {
            if args.len() != 3 {
                return Some(RespVal::err("ERR wrong number of arguments for 'hget'"));
            }
            match hash_ref(eng, &args[1]) {
                Ok(Some(m)) => m.get(&args[2]).map_or_else(RespVal::nil, |v| RespVal::bulk(v.clone())),
                Ok(None) => RespVal::nil(),
                Err(e) => e,
            }
        }
        "HMGET" => {
            if args.len() < 3 {
                return Some(RespVal::err("ERR wrong number of arguments for 'hmget'"));
            }
            match hash_ref(eng, &args[1]) {
                Ok(m_opt) => RespVal::array(
                    args[2..]
                        .iter()
                        .map(|f| {
                            m_opt
                                .and_then(|m| m.get(f))
                                .map_or_else(RespVal::nil, |v| RespVal::bulk(v.clone()))
                        })
                        .collect(),
                ),
                Err(e) => e,
            }
        }
        "HGETALL" => {
            if args.len() != 2 {
                return Some(RespVal::err("ERR wrong number of arguments for 'hgetall'"));
            }
            match hash_ref(eng, &args[1]) {
                Ok(None) => RespVal::array(Vec::new()),
                Ok(Some(m)) => {
                    let mut out = Vec::with_capacity(m.len() * 2);
                    for (k, v) in m {
                        out.push(RespVal::bulk(k.clone()));
                        out.push(RespVal::bulk(v.clone()));
                    }
                    RespVal::array(out)
                }
                Err(e) => e,
            }
        }
        "HDEL" => {
            if args.len() < 3 {
                return Some(RespVal::err("ERR wrong number of arguments for 'hdel'"));
            }
            let (removed, empty) = {
                let Some(rec) = (match eng.get_or_promote_mut(&args[1]) {
                    Ok(v) => v,
                    Err(e) => return Some(RespVal::err(e)),
                }) else {
                    return Some(RespVal::int(0));
                };
                let Some(Value::Hash(m)) = rec.value.as_mut() else {
                    return Some(RespVal::err(WRONGTYPE));
                };
                let mut r = 0i64;
                for f in &args[2..] {
                    if m.remove(f).is_some() {
                        r += 1;
                    }
                }
                (r, m.is_empty())
            };
            if removed > 0 && empty {
                let _ = eng.del_key(&args[1]);
            }
            RespVal::int(removed)
        }
        "HEXISTS" => {
            if args.len() != 3 {
                return Some(RespVal::err("ERR wrong number of arguments for 'hexists'"));
            }
            match hash_ref(eng, &args[1]) {
                Ok(Some(m)) => RespVal::int(i64::from(m.contains_key(&args[2]))),
                Ok(None) => RespVal::int(0),
                Err(e) => e,
            }
        }
        "HLEN" => {
            if args.len() != 2 {
                return Some(RespVal::err("ERR wrong number of arguments for 'hlen'"));
            }
            match hash_ref(eng, &args[1]) {
                Ok(Some(m)) => RespVal::int(m.len() as i64),
                Ok(None) => RespVal::int(0),
                Err(e) => e,
            }
        }
        "HKEYS" | "HVALS" => {
            if args.len() != 2 {
                return Some(RespVal::err("ERR wrong number of arguments"));
            }
            match hash_ref(eng, &args[1]) {
                Ok(None) => RespVal::array(Vec::new()),
                Ok(Some(m)) => {
                    if cmd == "HKEYS" {
                        RespVal::array(m.keys().cloned().map(RespVal::bulk).collect())
                    } else {
                        RespVal::array(m.values().cloned().map(RespVal::bulk).collect())
                    }
                }
                Err(e) => e,
            }
        }
        "HINCRBY" => {
            if args.len() != 4 {
                return Some(RespVal::err("ERR wrong number of arguments for 'hincrby'"));
            }
            let delta = match parse_i64(&args[3]) {
                Ok(v) => v,
                Err(e) => return Some(e),
            };
            let rec = match eng.ensure_entry_mut(&args[1], || Value::Hash(BTreeMap::new())) {
                Ok(r) => r,
                Err(e) => return Some(RespVal::err(e)),
            };
            let Some(Value::Hash(m)) = rec.value.as_mut() else {
                return Some(RespVal::err(WRONGTYPE));
            };
            let cur = match m.get(&args[2]) {
                None => 0i64,
                Some(b) => match parse_i64(b) {
                    Ok(v) => v,
                    Err(e) => return Some(e),
                },
            };
            let next = cur.saturating_add(delta);
            m.insert(args[2].clone(), next.to_string().into_bytes());
            RespVal::int(next)
        }
        "HINCRBYFLOAT" => {
            if args.len() != 4 {
                return Some(RespVal::err("ERR wrong number of arguments"));
            }
            let delta = match parse_f64(&args[3]) {
                Ok(v) => v,
                Err(e) => return Some(e),
            };
            let rec = match eng.ensure_entry_mut(&args[1], || Value::Hash(BTreeMap::new())) {
                Ok(r) => r,
                Err(e) => return Some(RespVal::err(e)),
            };
            let Some(Value::Hash(m)) = rec.value.as_mut() else {
                return Some(RespVal::err(WRONGTYPE));
            };
            let cur = match m.get(&args[2]) {
                None => 0.0,
                Some(b) => match parse_f64(b) {
                    Ok(v) => v,
                    Err(e) => return Some(e),
                },
            };
            let next = format_float(cur + delta);
            m.insert(args[2].clone(), next.as_bytes().to_vec());
            RespVal::bulk(next.into_bytes())
        }

        // ---- LIST (Zero-Clone In-Place O(1)!) ----
        "LPUSH" | "RPUSH" => {
            if args.len() < 3 {
                return Some(RespVal::err("ERR wrong number of arguments"));
            }
            let rec = match eng.ensure_entry_mut(&args[1], || Value::List(VecDeque::new())) {
                Ok(r) => r,
                Err(e) => return Some(RespVal::err(e)),
            };
            let Some(Value::List(l)) = rec.value.as_mut() else {
                return Some(RespVal::err(WRONGTYPE));
            };
            for item in &args[2..] {
                if cmd == "LPUSH" {
                    l.push_front(item.clone());
                } else {
                    l.push_back(item.clone());
                }
            }
            RespVal::int(l.len() as i64)
        }
        "LPOP" | "RPOP" => {
            if args.len() < 2 || args.len() > 3 {
                return Some(RespVal::err("ERR wrong number of arguments"));
            }
            let count_opt = if args.len() == 3 {
                match parse_i64(&args[2]) {
                    Ok(n) if n >= 0 => Some(n as usize),
                    _ => return Some(RespVal::err("ERR value is out of range, must be positive")),
                }
            } else {
                None
            };
            let (resp, empty) = {
                let Some(rec) = (match eng.get_or_promote_mut(&args[1]) {
                    Ok(v) => v,
                    Err(e) => return Some(RespVal::err(e)),
                }) else {
                    return Some(RespVal::nil());
                };
                let Some(Value::List(l)) = rec.value.as_mut() else {
                    return Some(RespVal::err(WRONGTYPE));
                };
                if l.is_empty() {
                    return Some(RespVal::nil());
                }
                let pop_one = |dq: &mut VecDeque<Vec<u8>>| {
                    if cmd == "LPOP" {
                        dq.pop_front()
                    } else {
                        dq.pop_back()
                    }
                };
                let r = if let Some(cnt) = count_opt {
                    let mut out = Vec::with_capacity(cnt.min(l.len()));
                    for _ in 0..cnt {
                        let Some(v) = pop_one(l) else { break };
                        out.push(RespVal::bulk(v));
                    }
                    RespVal::array(out)
                } else {
                    pop_one(l).map_or_else(RespVal::nil, RespVal::bulk)
                };
                (r, l.is_empty())
            };
            if empty {
                let _ = eng.del_key(&args[1]);
            }
            resp
        }
        "LLEN" => {
            if args.len() != 2 {
                return Some(RespVal::err("ERR wrong number of arguments for 'llen'"));
            }
            match list_ref(eng, &args[1]) {
                Ok(Some(l)) => RespVal::int(l.len() as i64),
                Ok(None) => RespVal::int(0),
                Err(e) => e,
            }
        }
        "LRANGE" => {
            if args.len() != 4 {
                return Some(RespVal::err("ERR wrong number of arguments for 'lrange'"));
            }
            let (Ok(s), Ok(e)) = (parse_i64(&args[2]), parse_i64(&args[3])) else {
                return Some(RespVal::err("ERR value is not an integer"));
            };
            match list_ref(eng, &args[1]) {
                Ok(None) => RespVal::array(Vec::new()),
                Ok(Some(l)) => match norm_range(s, e, l.len()) {
                    None => RespVal::array(Vec::new()),
                    Some((a, b)) => {
                        RespVal::array((a..=b).map(|i| RespVal::bulk(l[i].clone())).collect())
                    }
                },
                Err(err) => err,
            }
        }
        "LINDEX" => {
            if args.len() != 3 {
                return Some(RespVal::err("ERR wrong number of arguments for 'lindex'"));
            }
            let idx = match parse_i64(&args[2]) {
                Ok(v) => v,
                Err(e) => return Some(e),
            };
            match list_ref(eng, &args[1]) {
                Ok(None) => RespVal::nil(),
                Ok(Some(l)) => {
                    let i = if idx < 0 { l.len() as i64 + idx } else { idx };
                    if i < 0 || (i as usize) >= l.len() {
                        RespVal::nil()
                    } else {
                        RespVal::bulk(l[i as usize].clone())
                    }
                }
                Err(e) => e,
            }
        }
        "LSET" => {
            if args.len() != 4 {
                return Some(RespVal::err("ERR wrong number of arguments for 'lset'"));
            }
            let idx = match parse_i64(&args[2]) {
                Ok(v) => v,
                Err(e) => return Some(e),
            };
            let Some(rec) = (match eng.get_or_promote_mut(&args[1]) {
                Ok(v) => v,
                Err(e) => return Some(RespVal::err(e)),
            }) else {
                return Some(RespVal::err("ERR no such key"));
            };
            let Some(Value::List(l)) = rec.value.as_mut() else {
                return Some(RespVal::err(WRONGTYPE));
            };
            let i = if idx < 0 { l.len() as i64 + idx } else { idx };
            if i < 0 || (i as usize) >= l.len() {
                return Some(RespVal::err("ERR index out of range"));
            }
            l[i as usize] = args[3].clone();
            RespVal::ok()
        }
        "LTRIM" => {
            if args.len() != 4 {
                return Some(RespVal::err("ERR wrong number of arguments for 'ltrim'"));
            }
            let (Ok(s), Ok(e)) = (parse_i64(&args[2]), parse_i64(&args[3])) else {
                return Some(RespVal::err("ERR value is not an integer"));
            };
            let del = {
                let Some(rec) = (match eng.get_or_promote_mut(&args[1]) {
                    Ok(v) => v,
                    Err(e) => return Some(RespVal::err(e)),
                }) else {
                    return Some(RespVal::ok());
                };
                let Some(Value::List(l)) = rec.value.as_mut() else {
                    return Some(RespVal::err(WRONGTYPE));
                };
                match norm_range(s, e, l.len()) {
                    None => true,
                    Some((a, b)) => {
                        let tail_drop = l.len() - 1 - b;
                        for _ in 0..tail_drop {
                            l.pop_back();
                        }
                        for _ in 0..a {
                            l.pop_front();
                        }
                        false
                    }
                }
            };
            if del {
                let _ = eng.del_key(&args[1]);
            }
            RespVal::ok()
        }
        "LREM" => {
            if args.len() != 4 {
                return Some(RespVal::err("ERR wrong number of arguments for 'lrem'"));
            }
            let count = match parse_i64(&args[2]) {
                Ok(v) => v,
                Err(e) => return Some(e),
            };
            let (removed, empty) = {
                let Some(rec) = (match eng.get_or_promote_mut(&args[1]) {
                    Ok(v) => v,
                    Err(e) => return Some(RespVal::err(e)),
                }) else {
                    return Some(RespVal::int(0));
                };
                let Some(Value::List(l)) = rec.value.as_mut() else {
                    return Some(RespVal::err(WRONGTYPE));
                };
                let target = &args[3];
                let mut rem = 0i64;
                let mut next = VecDeque::with_capacity(l.len());
                if count >= 0 {
                    let limit = if count == 0 { usize::MAX } else { count as usize };
                    for item in l.drain(..) {
                        if &item == target && (rem as usize) < limit {
                            rem += 1;
                        } else {
                            next.push_back(item);
                        }
                    }
                } else {
                    let limit = count.unsigned_abs() as usize;
                    for item in l.drain(..).rev() {
                        if &item == target && (rem as usize) < limit {
                            rem += 1;
                        } else {
                            next.push_front(item);
                        }
                    }
                }
                *l = next;
                (rem, l.is_empty())
            };
            if empty {
                let _ = eng.del_key(&args[1]);
            }
            RespVal::int(removed)
        }
        "LMOVE" => {
            if args.len() != 5 {
                return Some(RespVal::err("ERR wrong number of arguments for 'lmove'"));
            }
            let from_left = arg_str(args, 3).is_some_and(|s| s.eq_ignore_ascii_case("LEFT"));
            let to_left = arg_str(args, 4).is_some_and(|s| s.eq_ignore_ascii_case("LEFT"));
            let (popped, src_empty) = {
                let Some(rec) = (match eng.get_or_promote_mut(&args[1]) {
                    Ok(v) => v,
                    Err(e) => return Some(RespVal::err(e)),
                }) else {
                    return Some(RespVal::nil());
                };
                let Some(Value::List(src)) = rec.value.as_mut() else {
                    return Some(RespVal::err(WRONGTYPE));
                };
                let val = if from_left {
                    src.pop_front()
                } else {
                    src.pop_back()
                };
                (val, src.is_empty())
            };
            let Some(val) = popped else {
                return Some(RespVal::nil());
            };
            if src_empty {
                let _ = eng.del_key(&args[1]);
            }
            let dst_rec = match eng.ensure_entry_mut(&args[2], || Value::List(VecDeque::new())) {
                Ok(r) => r,
                Err(e) => return Some(RespVal::err(e)),
            };
            let Some(Value::List(dst)) = dst_rec.value.as_mut() else {
                return Some(RespVal::err(WRONGTYPE));
            };
            if to_left {
                dst.push_front(val.clone());
            } else {
                dst.push_back(val.clone());
            }
            RespVal::bulk(val)
        }
        _ => return None,
    })
}
