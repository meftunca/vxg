//! Redis non-deprecated Hash (`H*`) and List (`L*`/`R*`) command dispatchers.

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

type HashEntry = (BTreeMap<Vec<u8>, Vec<u8>>, Option<u64>);

fn get_hash_mut(eng: &mut KvEngine, key: &[u8]) -> Result<HashEntry, RespVal> {
    match eng.get_record(key).map_err(RespVal::err)? {
        None => Ok((BTreeMap::new(), None)),
        Some(r) => match r.value {
            Some(Value::Hash(m)) => Ok((m, r.expires_at_ms)),
            _ => Err(RespVal::err(WRONGTYPE)),
        },
    }
}

fn get_list_mut(
    eng: &mut KvEngine,
    key: &[u8],
) -> Result<(VecDeque<Vec<u8>>, Option<u64>), RespVal> {
    match eng.get_record(key).map_err(RespVal::err)? {
        None => Ok((VecDeque::new(), None)),
        Some(r) => match r.value {
            Some(Value::List(l)) => Ok((l, r.expires_at_ms)),
            _ => Err(RespVal::err(WRONGTYPE)),
        },
    }
}

/// Dispatch Hash, List, Set, and ZSet commands. Returns `None` if command belongs elsewhere.
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
            let (mut m, exp) = match get_hash_mut(eng, &args[1]) {
                Ok(v) => v,
                Err(e) => return Some(e),
            };
            let mut added = 0i64;
            for pair in args[2..].as_chunks::<2>().0 {
                if m.insert(pair[0].clone(), pair[1].clone()).is_none() {
                    added += 1;
                }
            }
            eng.set_record(args[1].clone(), Value::Hash(m), exp);
            RespVal::int(added)
        }
        "HSETNX" => {
            if args.len() != 4 {
                return Some(RespVal::err("ERR wrong number of arguments for 'hsetnx'"));
            }
            let (mut m, exp) = match get_hash_mut(eng, &args[1]) {
                Ok(v) => v,
                Err(e) => return Some(e),
            };
            if m.contains_key(&args[2]) {
                RespVal::int(0)
            } else {
                m.insert(args[2].clone(), args[3].clone());
                eng.set_record(args[1].clone(), Value::Hash(m), exp);
                RespVal::int(1)
            }
        }
        "HGET" => {
            if args.len() != 3 {
                return Some(RespVal::err("ERR wrong number of arguments for 'hget'"));
            }
            let (m, _) = match get_hash_mut(eng, &args[1]) {
                Ok(v) => v,
                Err(e) => return Some(e),
            };
            match m.get(&args[2]) {
                Some(v) => RespVal::bulk(v.clone()),
                None => RespVal::nil(),
            }
        }
        "HMGET" => {
            if args.len() < 3 {
                return Some(RespVal::err("ERR wrong number of arguments for 'hmget'"));
            }
            let (m, _) = match get_hash_mut(eng, &args[1]) {
                Ok(v) => v,
                Err(e) => return Some(e),
            };
            let out = args[2..]
                .iter()
                .map(|f| match m.get(f) {
                    Some(v) => RespVal::bulk(v.clone()),
                    None => RespVal::nil(),
                })
                .collect();
            RespVal::array(out)
        }
        "HGETALL" => {
            if args.len() != 2 {
                return Some(RespVal::err("ERR wrong number of arguments for 'hgetall'"));
            }
            let (m, _) = match get_hash_mut(eng, &args[1]) {
                Ok(v) => v,
                Err(e) => return Some(e),
            };
            let mut out = Vec::with_capacity(m.len() * 2);
            for (k, v) in m {
                out.push(RespVal::bulk(k));
                out.push(RespVal::bulk(v));
            }
            RespVal::array(out)
        }
        "HDEL" => {
            if args.len() < 3 {
                return Some(RespVal::err("ERR wrong number of arguments for 'hdel'"));
            }
            let (mut m, exp) = match get_hash_mut(eng, &args[1]) {
                Ok(v) => v,
                Err(e) => return Some(e),
            };
            let mut removed = 0i64;
            for f in &args[2..] {
                if m.remove(f).is_some() {
                    removed += 1;
                }
            }
            if removed > 0 {
                if m.is_empty() {
                    let _ = eng.del_key(&args[1]);
                } else {
                    eng.set_record(args[1].clone(), Value::Hash(m), exp);
                }
            }
            RespVal::int(removed)
        }
        "HEXISTS" => {
            if args.len() != 3 {
                return Some(RespVal::err("ERR wrong number of arguments for 'hexists'"));
            }
            let (m, _) = match get_hash_mut(eng, &args[1]) {
                Ok(v) => v,
                Err(e) => return Some(e),
            };
            RespVal::int(i64::from(m.contains_key(&args[2])))
        }
        "HLEN" => {
            if args.len() != 2 {
                return Some(RespVal::err("ERR wrong number of arguments for 'hlen'"));
            }
            let (m, _) = match get_hash_mut(eng, &args[1]) {
                Ok(v) => v,
                Err(e) => return Some(e),
            };
            RespVal::int(m.len() as i64)
        }
        "HKEYS" | "HVALS" => {
            if args.len() != 2 {
                return Some(RespVal::err("ERR wrong number of arguments"));
            }
            let (m, _) = match get_hash_mut(eng, &args[1]) {
                Ok(v) => v,
                Err(e) => return Some(e),
            };
            let out = if cmd == "HKEYS" {
                m.into_keys().map(RespVal::bulk).collect()
            } else {
                m.into_values().map(RespVal::bulk).collect()
            };
            RespVal::array(out)
        }
        "HINCRBY" => {
            if args.len() != 4 {
                return Some(RespVal::err("ERR wrong number of arguments for 'hincrby'"));
            }
            let delta = match parse_i64(&args[3]) {
                Ok(v) => v,
                Err(e) => return Some(e),
            };
            let (mut m, exp) = match get_hash_mut(eng, &args[1]) {
                Ok(v) => v,
                Err(e) => return Some(e),
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
            eng.set_record(args[1].clone(), Value::Hash(m), exp);
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
            let (mut m, exp) = match get_hash_mut(eng, &args[1]) {
                Ok(v) => v,
                Err(e) => return Some(e),
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
            eng.set_record(args[1].clone(), Value::Hash(m), exp);
            RespVal::bulk(next.into_bytes())
        }

        // ---- LIST ----
        "LPUSH" | "RPUSH" => {
            if args.len() < 3 {
                return Some(RespVal::err("ERR wrong number of arguments"));
            }
            let (mut l, exp) = match get_list_mut(eng, &args[1]) {
                Ok(v) => v,
                Err(e) => return Some(e),
            };
            for item in &args[2..] {
                if cmd == "LPUSH" {
                    l.push_front(item.clone());
                } else {
                    l.push_back(item.clone());
                }
            }
            let len = l.len() as i64;
            eng.set_record(args[1].clone(), Value::List(l), exp);
            RespVal::int(len)
        }
        "LPOP" | "RPOP" => {
            if args.len() < 2 || args.len() > 3 {
                return Some(RespVal::err("ERR wrong number of arguments"));
            }
            let (mut l, exp) = match get_list_mut(eng, &args[1]) {
                Ok(v) => v,
                Err(e) => return Some(e),
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
            let resp = if args.len() == 3 {
                let cnt = match parse_i64(&args[2]) {
                    Ok(n) if n >= 0 => n as usize,
                    _ => return Some(RespVal::err("ERR value is out of range, must be positive")),
                };
                let mut out = Vec::with_capacity(cnt.min(l.len()));
                for _ in 0..cnt {
                    let Some(v) = pop_one(&mut l) else { break };
                    out.push(RespVal::bulk(v));
                }
                RespVal::array(out)
            } else {
                match pop_one(&mut l) {
                    Some(v) => RespVal::bulk(v),
                    None => RespVal::nil(),
                }
            };
            if l.is_empty() {
                let _ = eng.del_key(&args[1]);
            } else {
                eng.set_record(args[1].clone(), Value::List(l), exp);
            }
            resp
        }
        "LLEN" => {
            if args.len() != 2 {
                return Some(RespVal::err("ERR wrong number of arguments for 'llen'"));
            }
            let (l, _) = match get_list_mut(eng, &args[1]) {
                Ok(v) => v,
                Err(e) => return Some(e),
            };
            RespVal::int(l.len() as i64)
        }
        "LRANGE" => {
            if args.len() != 4 {
                return Some(RespVal::err("ERR wrong number of arguments for 'lrange'"));
            }
            let (Ok(s), Ok(e)) = (parse_i64(&args[2]), parse_i64(&args[3])) else {
                return Some(RespVal::err("ERR value is not an integer"));
            };
            let (l, _) = match get_list_mut(eng, &args[1]) {
                Ok(v) => v,
                Err(e) => return Some(e),
            };
            match norm_range(s, e, l.len()) {
                None => RespVal::array(Vec::new()),
                Some((a, b)) => RespVal::array(
                    (a..=b).map(|i| RespVal::bulk(l[i].clone())).collect(),
                ),
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
            let (l, _) = match get_list_mut(eng, &args[1]) {
                Ok(v) => v,
                Err(e) => return Some(e),
            };
            let i = if idx < 0 { l.len() as i64 + idx } else { idx };
            if i < 0 || (i as usize) >= l.len() {
                RespVal::nil()
            } else {
                RespVal::bulk(l[i as usize].clone())
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
            let (mut l, exp) = match get_list_mut(eng, &args[1]) {
                Ok(v) => v,
                Err(e) => return Some(e),
            };
            if l.is_empty() {
                return Some(RespVal::err("ERR no such key"));
            }
            let i = if idx < 0 { l.len() as i64 + idx } else { idx };
            if i < 0 || (i as usize) >= l.len() {
                return Some(RespVal::err("ERR index out of range"));
            }
            l[i as usize] = args[3].clone();
            eng.set_record(args[1].clone(), Value::List(l), exp);
            RespVal::ok()
        }
        "LTRIM" => {
            if args.len() != 4 {
                return Some(RespVal::err("ERR wrong number of arguments for 'ltrim'"));
            }
            let (Ok(s), Ok(e)) = (parse_i64(&args[2]), parse_i64(&args[3])) else {
                return Some(RespVal::err("ERR value is not an integer"));
            };
            let (l, exp) = match get_list_mut(eng, &args[1]) {
                Ok(v) => v,
                Err(e) => return Some(e),
            };
            match norm_range(s, e, l.len()) {
                None => {
                    let _ = eng.del_key(&args[1]);
                }
                Some((a, b)) => {
                    let trimmed: VecDeque<Vec<u8>> = (a..=b).map(|i| l[i].clone()).collect();
                    eng.set_record(args[1].clone(), Value::List(trimmed), exp);
                }
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
            let (l, exp) = match get_list_mut(eng, &args[1]) {
                Ok(v) => v,
                Err(e) => return Some(e),
            };
            let target = &args[3];
            let mut removed = 0i64;
            let mut next = VecDeque::with_capacity(l.len());
            if count >= 0 {
                let limit = if count == 0 { usize::MAX } else { count as usize };
                for item in l {
                    if &item == target && (removed as usize) < limit {
                        removed += 1;
                    } else {
                        next.push_back(item);
                    }
                }
            } else {
                let limit = count.unsigned_abs() as usize;
                for item in l.into_iter().rev() {
                    if &item == target && (removed as usize) < limit {
                        removed += 1;
                    } else {
                        next.push_front(item);
                    }
                }
            }
            if next.is_empty() {
                let _ = eng.del_key(&args[1]);
            } else {
                eng.set_record(args[1].clone(), Value::List(next), exp);
            }
            RespVal::int(removed)
        }
        "LMOVE" => {
            if args.len() != 5 {
                return Some(RespVal::err("ERR wrong number of arguments for 'lmove'"));
            }
            let from_left = arg_str(args, 3).is_some_and(|s| s.eq_ignore_ascii_case("LEFT"));
            let to_left = arg_str(args, 4).is_some_and(|s| s.eq_ignore_ascii_case("LEFT"));
            let (mut src, src_exp) = match get_list_mut(eng, &args[1]) {
                Ok(v) => v,
                Err(e) => return Some(e),
            };
            let Some(val) = (if from_left {
                src.pop_front()
            } else {
                src.pop_back()
            }) else {
                return Some(RespVal::nil());
            };
            if src.is_empty() {
                let _ = eng.del_key(&args[1]);
            } else {
                eng.set_record(args[1].clone(), Value::List(src), src_exp);
            }
            let (mut dst, dst_exp) = match get_list_mut(eng, &args[2]) {
                Ok(v) => v,
                Err(e) => return Some(e),
            };
            if to_left {
                dst.push_front(val.clone());
            } else {
                dst.push_back(val.clone());
            }
            eng.set_record(args[2].clone(), Value::List(dst), dst_exp);
            RespVal::bulk(val)
        }

        _ => return None,
    })
}
