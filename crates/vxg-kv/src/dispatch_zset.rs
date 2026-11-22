//! Redis Set (`S*`) and Sorted Set (`Z*`) command dispatchers (zero-clone in-place).

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use crate::dispatch_ds::{arg_str, format_float, norm_range, parse_f64, parse_i64, WRONGTYPE};
use crate::engine::KvEngine;
use crate::resp::RespVal;
use crate::types::Value;

fn parse_score_bound(b: &[u8]) -> Result<(f64, bool), RespVal> {
    let s = std::str::from_utf8(b).map_err(|_| RespVal::err("ERR min or max is not a float"))?;
    let (exclusive, num) = if let Some(rest) = s.strip_prefix('(') {
        (true, rest)
    } else {
        (false, s)
    };
    let val = match num {
        "-inf" => f64::NEG_INFINITY,
        "+inf" | "inf" => f64::INFINITY,
        other => other
            .parse::<f64>()
            .map_err(|_| RespVal::err("ERR min or max is not a float"))?,
    };
    Ok((val, exclusive))
}

fn set_ref<'a>(eng: &'a mut KvEngine, key: &[u8]) -> Result<Option<&'a BTreeSet<Vec<u8>>>, RespVal> {
    match eng.get_record_or_load(key).map_err(RespVal::err)? {
        None => Ok(None),
        Some(r) => match r.value.as_ref() {
            Some(Value::Set(s)) => Ok(Some(s)),
            _ => Err(RespVal::err(WRONGTYPE)),
        },
    }
}

fn zset_ref<'a>(
    eng: &'a mut KvEngine,
    key: &[u8],
) -> Result<Option<&'a BTreeMap<Vec<u8>, f64>>, RespVal> {
    match eng.get_record_or_load(key).map_err(RespVal::err)? {
        None => Ok(None),
        Some(r) => match r.value.as_ref() {
            Some(Value::ZSet(z)) => Ok(Some(z)),
            _ => Err(RespVal::err(WRONGTYPE)),
        },
    }
}

fn zset_sorted(z: &BTreeMap<Vec<u8>, f64>) -> Vec<(&[u8], f64)> {
    let mut items: Vec<(&[u8], f64)> = z.iter().map(|(k, &v)| (k.as_slice(), v)).collect();
    items.sort_by(|(ka, sa), (kb, sb)| {
        sa.partial_cmp(sb)
            .unwrap_or(Ordering::Equal)
            .then_with(|| ka.cmp(kb))
    });
    items
}

pub fn dispatch_set_zset_cmd(eng: &mut KvEngine, cmd: &str, args: &[Vec<u8>]) -> Option<RespVal> {
    Some(match cmd {
        // ---- SET (Zero-Clone In-Place) ----
        "SADD" => {
            if args.len() < 3 {
                return Some(RespVal::err("ERR wrong number of arguments for 'sadd'"));
            }
            let rec = match eng.ensure_entry_mut(&args[1], || Value::Set(BTreeSet::new())) {
                Ok(r) => r,
                Err(e) => return Some(RespVal::err(e)),
            };
            let Some(Value::Set(s)) = rec.value.as_mut() else {
                return Some(RespVal::err(WRONGTYPE));
            };
            let mut added = 0i64;
            for m in &args[2..] {
                if s.insert(m.clone()) {
                    added += 1;
                }
            }
            RespVal::int(added)
        }
        "SREM" => {
            if args.len() < 3 {
                return Some(RespVal::err("ERR wrong number of arguments for 'srem'"));
            }
            let (removed, empty) = {
                let Some(rec) = (match eng.get_or_promote_mut(&args[1]) {
                    Ok(v) => v,
                    Err(e) => return Some(RespVal::err(e)),
                }) else {
                    return Some(RespVal::int(0));
                };
                let Some(Value::Set(s)) = rec.value.as_mut() else {
                    return Some(RespVal::err(WRONGTYPE));
                };
                let mut r = 0i64;
                for m in &args[2..] {
                    if s.remove(m) {
                        r += 1;
                    }
                }
                (r, s.is_empty())
            };
            if removed > 0 && empty {
                let _ = eng.del_key(&args[1]);
            }
            RespVal::int(removed)
        }
        "SMEMBERS" => {
            if args.len() != 2 {
                return Some(RespVal::err("ERR wrong number of arguments for 'smembers'"));
            }
            match set_ref(eng, &args[1]) {
                Ok(Some(s)) => RespVal::array(s.iter().cloned().map(RespVal::bulk).collect()),
                Ok(None) => RespVal::array(Vec::new()),
                Err(e) => e,
            }
        }
        "SISMEMBER" => {
            if args.len() != 3 {
                return Some(RespVal::err("ERR wrong number of arguments for 'sismember'"));
            }
            match set_ref(eng, &args[1]) {
                Ok(Some(s)) => RespVal::int(i64::from(s.contains(&args[2]))),
                Ok(None) => RespVal::int(0),
                Err(e) => e,
            }
        }
        "SMISMEMBER" => {
            if args.len() < 3 {
                return Some(RespVal::err("ERR wrong number of arguments"));
            }
            match set_ref(eng, &args[1]) {
                Ok(s_opt) => RespVal::array(
                    args[2..]
                        .iter()
                        .map(|m| RespVal::int(i64::from(s_opt.is_some_and(|s| s.contains(m)))))
                        .collect(),
                ),
                Err(e) => e,
            }
        }
        "SCARD" => {
            if args.len() != 2 {
                return Some(RespVal::err("ERR wrong number of arguments for 'scard'"));
            }
            match set_ref(eng, &args[1]) {
                Ok(Some(s)) => RespVal::int(s.len() as i64),
                Ok(None) => RespVal::int(0),
                Err(e) => e,
            }
        }
        "SPOP" => {
            if args.len() < 2 || args.len() > 3 {
                return Some(RespVal::err("ERR wrong number of arguments for 'spop'"));
            }
            let cnt_opt = (args.len() == 3).then(|| parse_i64(&args[2]).unwrap_or(1).max(0) as usize);
            let (resp, empty) = {
                let Some(rec) = (match eng.get_or_promote_mut(&args[1]) {
                    Ok(v) => v,
                    Err(e) => return Some(RespVal::err(e)),
                }) else {
                    return Some(RespVal::nil());
                };
                let Some(Value::Set(s)) = rec.value.as_mut() else {
                    return Some(RespVal::err(WRONGTYPE));
                };
                if s.is_empty() {
                    return Some(RespVal::nil());
                }
                let r = if let Some(cnt) = cnt_opt {
                    let mut out = Vec::new();
                    for _ in 0..cnt {
                        let Some(first) = s.pop_first() else { break };
                        out.push(RespVal::bulk(first));
                    }
                    RespVal::array(out)
                } else {
                    RespVal::bulk(s.pop_first().unwrap())
                };
                (r, s.is_empty())
            };
            if empty {
                let _ = eng.del_key(&args[1]);
            }
            resp
        }
        "SINTER" | "SUNION" | "SDIFF" => {
            if args.len() < 2 {
                return Some(RespVal::err("ERR wrong number of arguments"));
            }
            let mut acc = match set_ref(eng, &args[1]) {
                Ok(Some(s)) => s.clone(),
                Ok(None) => BTreeSet::new(),
                Err(e) => return Some(e),
            };
            for k in &args[2..] {
                let other = match set_ref(eng, k) {
                    Ok(Some(s)) => s.clone(),
                    Ok(None) => BTreeSet::new(),
                    Err(e) => return Some(e),
                };
                match cmd {
                    "SINTER" => acc.retain(|x| other.contains(x)),
                    "SUNION" => acc.extend(other),
                    "SDIFF" => acc.retain(|x| !other.contains(x)),
                    _ => {}
                }
            }
            RespVal::array(acc.into_iter().map(RespVal::bulk).collect())
        }

        // ---- SORTED SET (ZSET, Zero-Clone In-Place) ----
        "ZADD" => dispatch_zadd(eng, args),
        "ZREM" => {
            if args.len() < 3 {
                return Some(RespVal::err("ERR wrong number of arguments for 'zrem'"));
            }
            let (rem, empty) = {
                let Some(rec) = (match eng.get_or_promote_mut(&args[1]) {
                    Ok(v) => v,
                    Err(e) => return Some(RespVal::err(e)),
                }) else {
                    return Some(RespVal::int(0));
                };
                let Some(Value::ZSet(z)) = rec.value.as_mut() else {
                    return Some(RespVal::err(WRONGTYPE));
                };
                let mut r = 0i64;
                for m in &args[2..] {
                    if z.remove(m).is_some() {
                        r += 1;
                    }
                }
                (r, z.is_empty())
            };
            if rem > 0 && empty {
                let _ = eng.del_key(&args[1]);
            }
            RespVal::int(rem)
        }
        "ZSCORE" => {
            if args.len() != 3 {
                return Some(RespVal::err("ERR wrong number of arguments for 'zscore'"));
            }
            match zset_ref(eng, &args[1]) {
                Ok(Some(z)) => match z.get(&args[2]) {
                    Some(&sc) => RespVal::bulk(format_float(sc).into_bytes()),
                    None => RespVal::nil(),
                },
                Ok(None) => RespVal::nil(),
                Err(e) => e,
            }
        }
        "ZCARD" => {
            if args.len() != 2 {
                return Some(RespVal::err("ERR wrong number of arguments for 'zcard'"));
            }
            match zset_ref(eng, &args[1]) {
                Ok(Some(z)) => RespVal::int(z.len() as i64),
                Ok(None) => RespVal::int(0),
                Err(e) => e,
            }
        }
        "ZRANK" | "ZREVRANK" => {
            if args.len() != 3 {
                return Some(RespVal::err("ERR wrong number of arguments"));
            }
            match zset_ref(eng, &args[1]) {
                Ok(None) => RespVal::nil(),
                Ok(Some(z)) => {
                    let mut sorted = zset_sorted(z);
                    if cmd == "ZREVRANK" {
                        sorted.reverse();
                    }
                    match sorted.iter().position(|(m, _)| *m == args[2].as_slice()) {
                        Some(p) => RespVal::int(p as i64),
                        None => RespVal::nil(),
                    }
                }
                Err(e) => e,
            }
        }
        "ZINCRBY" => {
            if args.len() != 4 {
                return Some(RespVal::err("ERR wrong number of arguments for 'zincrby'"));
            }
            let delta = match parse_f64(&args[2]) {
                Ok(v) => v,
                Err(e) => return Some(e),
            };
            let rec = match eng.ensure_entry_mut(&args[1], || Value::ZSet(BTreeMap::new())) {
                Ok(r) => r,
                Err(e) => return Some(RespVal::err(e)),
            };
            let Some(Value::ZSet(z)) = rec.value.as_mut() else {
                return Some(RespVal::err(WRONGTYPE));
            };
            let next = z.get(&args[3]).copied().unwrap_or(0.0) + delta;
            z.insert(args[3].clone(), next);
            RespVal::bulk(format_float(next).into_bytes())
        }
        "ZCOUNT" => {
            if args.len() != 4 {
                return Some(RespVal::err("ERR wrong number of arguments for 'zcount'"));
            }
            let (Ok((lo, lo_ex)), Ok((hi, hi_ex))) =
                (parse_score_bound(&args[2]), parse_score_bound(&args[3]))
            else {
                return Some(RespVal::err("ERR min or max is not a float"));
            };
            match zset_ref(eng, &args[1]) {
                Ok(None) => RespVal::int(0),
                Ok(Some(z)) => {
                    let cnt = z
                        .values()
                        .filter(|&&s| {
                            (if lo_ex { s > lo } else { s >= lo })
                                && (if hi_ex { s < hi } else { s <= hi })
                        })
                        .count();
                    RespVal::int(cnt as i64)
                }
                Err(e) => e,
            }
        }
        "ZRANGE" | "ZRANGEBYSCORE" => dispatch_zrange(eng, cmd, args),
        "ZPOPMIN" | "ZPOPMAX" => {
            if args.len() < 2 || args.len() > 3 {
                return Some(RespVal::err("ERR wrong number of arguments"));
            }
            let cnt = if args.len() == 3 {
                parse_i64(&args[2]).unwrap_or(1).max(0) as usize
            } else {
                1
            };
            let (out, empty) = {
                let Some(rec) = (match eng.get_or_promote_mut(&args[1]) {
                    Ok(v) => v,
                    Err(e) => return Some(RespVal::err(e)),
                }) else {
                    return Some(RespVal::array(Vec::new()));
                };
                let Some(Value::ZSet(z)) = rec.value.as_mut() else {
                    return Some(RespVal::err(WRONGTYPE));
                };
                let mut sorted: Vec<(Vec<u8>, f64)> = zset_sorted(z)
                    .into_iter()
                    .map(|(m, s)| (m.to_vec(), s))
                    .collect();
                if cmd == "ZPOPMAX" {
                    sorted.reverse();
                }
                let mut res = Vec::new();
                for (m, s) in sorted.into_iter().take(cnt) {
                    z.remove(&m);
                    res.push(RespVal::bulk(m));
                    res.push(RespVal::bulk(format_float(s).into_bytes()));
                }
                (res, z.is_empty())
            };
            if empty {
                let _ = eng.del_key(&args[1]);
            }
            RespVal::array(out)
        }
        _ => return None,
    })
}

fn dispatch_zadd(eng: &mut KvEngine, args: &[Vec<u8>]) -> RespVal {
    if args.len() < 4 {
        return RespVal::err("ERR wrong number of arguments for 'zadd'");
    }
    let (mut nx, mut xx, mut gt, mut lt, mut ch) = (false, false, false, false, false);
    let mut i = 2usize;
    while i < args.len() {
        let Some(tok) = arg_str(args, i) else { break };
        if tok.eq_ignore_ascii_case("NX") {
            nx = true;
        } else if tok.eq_ignore_ascii_case("XX") {
            xx = true;
        } else if tok.eq_ignore_ascii_case("GT") {
            gt = true;
        } else if tok.eq_ignore_ascii_case("LT") {
            lt = true;
        } else if tok.eq_ignore_ascii_case("CH") {
            ch = true;
        } else {
            break;
        }
        i += 1;
    }
    if i >= args.len() || !(args.len() - i).is_multiple_of(2) {
        return RespVal::err("ERR syntax error");
    }
    let rec = match eng.ensure_entry_mut(&args[1], || Value::ZSet(BTreeMap::new())) {
        Ok(r) => r,
        Err(e) => return RespVal::err(e),
    };
    let Some(Value::ZSet(z)) = rec.value.as_mut() else {
        return RespVal::err(WRONGTYPE);
    };
    let mut added = 0i64;
    let mut changed = 0i64;
    for pair in args[i..].as_chunks::<2>().0 {
        let sc = match parse_f64(&pair[0]) {
            Ok(v) => v,
            Err(e) => return e,
        };
        let member = &pair[1];
        match z.get(member).copied() {
            None => {
                if !xx {
                    z.insert(member.clone(), sc);
                    added += 1;
                    changed += 1;
                }
            }
            Some(old) => {
                if !nx && (!gt || sc > old) && (!lt || sc < old) && sc != old {
                    z.insert(member.clone(), sc);
                    changed += 1;
                }
            }
        }
    }
    RespVal::int(if ch { changed } else { added })
}

fn dispatch_zrange(eng: &mut KvEngine, cmd: &str, args: &[Vec<u8>]) -> RespVal {
    if args.len() < 4 {
        return RespVal::err("ERR wrong number of arguments for 'zrange'");
    }
    let mut byscore = cmd == "ZRANGEBYSCORE";
    let mut rev = false;
    let mut withscores = false;
    let mut limit: Option<(usize, usize)> = None;
    let mut i = 4usize;
    while i < args.len() {
        let tok = arg_str(args, i).unwrap_or("");
        if tok.eq_ignore_ascii_case("BYSCORE") {
            byscore = true;
        } else if tok.eq_ignore_ascii_case("REV") {
            rev = true;
        } else if tok.eq_ignore_ascii_case("WITHSCORES") {
            withscores = true;
        } else if tok.eq_ignore_ascii_case("LIMIT") && i + 2 < args.len() {
            let off = parse_i64(&args[i + 1]).unwrap_or(0).max(0) as usize;
            let cnt = parse_i64(&args[i + 2]).unwrap_or(i64::MAX).max(0) as usize;
            limit = Some((off, cnt));
            i += 2;
        }
        i += 1;
    }
    let Some(z) = (match zset_ref(eng, &args[1]) {
        Ok(v) => v,
        Err(e) => return e,
    }) else {
        return RespVal::array(Vec::new());
    };
    let mut sorted = zset_sorted(z);
    if rev {
        sorted.reverse();
    }
    let slice: Vec<(&[u8], f64)> = if byscore {
        let (Ok((a, a_ex)), Ok((b, b_ex))) =
            (parse_score_bound(&args[2]), parse_score_bound(&args[3]))
        else {
            return RespVal::err("ERR min or max is not a float");
        };
        let (lo, lo_ex, hi, hi_ex) = if rev {
            (b, b_ex, a, a_ex)
        } else {
            (a, a_ex, b, b_ex)
        };
        let mut f: Vec<_> = sorted
            .into_iter()
            .filter(|&(_, s)| {
                (if lo_ex { s > lo } else { s >= lo })
                    && (if hi_ex { s < hi } else { s <= hi })
            })
            .collect();
        if let Some((off, cnt)) = limit {
            f = f.into_iter().skip(off).take(cnt).collect();
        }
        f
    } else {
        let (Ok(s), Ok(e)) = (parse_i64(&args[2]), parse_i64(&args[3])) else {
            return RespVal::err("ERR value is not an integer");
        };
        match norm_range(s, e, sorted.len()) {
            None => Vec::new(),
            Some((a, b)) => sorted[a..=b].to_vec(),
        }
    };
    let mut out = Vec::with_capacity(if withscores {
        slice.len() * 2
    } else {
        slice.len()
    });
    for (m, s) in slice {
        out.push(RespVal::bulk(m.to_vec()));
        if withscores {
            out.push(RespVal::bulk(format_float(s).into_bytes()));
        }
    }
    RespVal::array(out)
}
