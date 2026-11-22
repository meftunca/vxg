//! Redis Set (`S*`) and Sorted Set (`Z*`) command dispatchers.

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

fn get_set_mut(eng: &mut KvEngine, key: &[u8]) -> Result<(BTreeSet<Vec<u8>>, Option<u64>), RespVal> {
    match eng.get_record(key).map_err(RespVal::err)? {
        None => Ok((BTreeSet::new(), None)),
        Some(r) => match r.value {
            Some(Value::Set(s)) => Ok((s, r.expires_at_ms)),
            _ => Err(RespVal::err(WRONGTYPE)),
        },
    }
}

type ZSetEntry = (BTreeMap<Vec<u8>, f64>, Option<u64>);

fn get_zset_mut(eng: &mut KvEngine, key: &[u8]) -> Result<ZSetEntry, RespVal> {
    match eng.get_record(key).map_err(RespVal::err)? {
        None => Ok((BTreeMap::new(), None)),
        Some(r) => match r.value {
            Some(Value::ZSet(z)) => Ok((z, r.expires_at_ms)),
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
        // ---- SET ----
        "SADD" => {
            if args.len() < 3 {
                return Some(RespVal::err("ERR wrong number of arguments for 'sadd'"));
            }
            let (mut s, exp) = match get_set_mut(eng, &args[1]) {
                Ok(v) => v,
                Err(e) => return Some(e),
            };
            let mut added = 0i64;
            for m in &args[2..] {
                if s.insert(m.clone()) {
                    added += 1;
                }
            }
            eng.set_record(args[1].clone(), Value::Set(s), exp);
            RespVal::int(added)
        }
        "SREM" => {
            if args.len() < 3 {
                return Some(RespVal::err("ERR wrong number of arguments for 'srem'"));
            }
            let (mut s, exp) = match get_set_mut(eng, &args[1]) {
                Ok(v) => v,
                Err(e) => return Some(e),
            };
            let mut removed = 0i64;
            for m in &args[2..] {
                if s.remove(m) {
                    removed += 1;
                }
            }
            if removed > 0 {
                if s.is_empty() {
                    let _ = eng.del_key(&args[1]);
                } else {
                    eng.set_record(args[1].clone(), Value::Set(s), exp);
                }
            }
            RespVal::int(removed)
        }
        "SMEMBERS" => {
            if args.len() != 2 {
                return Some(RespVal::err("ERR wrong number of arguments for 'smembers'"));
            }
            let (s, _) = match get_set_mut(eng, &args[1]) {
                Ok(v) => v,
                Err(e) => return Some(e),
            };
            RespVal::array(s.into_iter().map(RespVal::bulk).collect())
        }
        "SISMEMBER" => {
            if args.len() != 3 {
                return Some(RespVal::err("ERR wrong number of arguments for 'sismember'"));
            }
            let (s, _) = match get_set_mut(eng, &args[1]) {
                Ok(v) => v,
                Err(e) => return Some(e),
            };
            RespVal::int(i64::from(s.contains(&args[2])))
        }
        "SMISMEMBER" => {
            if args.len() < 3 {
                return Some(RespVal::err("ERR wrong number of arguments"));
            }
            let (s, _) = match get_set_mut(eng, &args[1]) {
                Ok(v) => v,
                Err(e) => return Some(e),
            };
            RespVal::array(
                args[2..]
                    .iter()
                    .map(|m| RespVal::int(i64::from(s.contains(m))))
                    .collect(),
            )
        }
        "SCARD" => {
            if args.len() != 2 {
                return Some(RespVal::err("ERR wrong number of arguments for 'scard'"));
            }
            let (s, _) = match get_set_mut(eng, &args[1]) {
                Ok(v) => v,
                Err(e) => return Some(e),
            };
            RespVal::int(s.len() as i64)
        }
        "SPOP" => {
            if args.len() < 2 || args.len() > 3 {
                return Some(RespVal::err("ERR wrong number of arguments for 'spop'"));
            }
            let (mut s, exp) = match get_set_mut(eng, &args[1]) {
                Ok(v) => v,
                Err(e) => return Some(e),
            };
            if s.is_empty() {
                return Some(RespVal::nil());
            }
            let resp = if args.len() == 3 {
                let cnt = parse_i64(&args[2]).unwrap_or(1).max(0) as usize;
                let mut out = Vec::new();
                for _ in 0..cnt {
                    let Some(first) = s.pop_first() else { break };
                    out.push(RespVal::bulk(first));
                }
                RespVal::array(out)
            } else {
                RespVal::bulk(s.pop_first().unwrap())
            };
            if s.is_empty() {
                let _ = eng.del_key(&args[1]);
            } else {
                eng.set_record(args[1].clone(), Value::Set(s), exp);
            }
            resp
        }
        "SINTER" | "SUNION" | "SDIFF" => {
            if args.len() < 2 {
                return Some(RespVal::err("ERR wrong number of arguments"));
            }
            let (mut acc, _) = match get_set_mut(eng, &args[1]) {
                Ok(v) => v,
                Err(e) => return Some(e),
            };
            for k in &args[2..] {
                let (other, _) = match get_set_mut(eng, k) {
                    Ok(v) => v,
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

        // ---- SORTED SET (ZSET) ----
        "ZADD" => dispatch_zadd(eng, args),
        "ZREM" => {
            if args.len() < 3 {
                return Some(RespVal::err("ERR wrong number of arguments for 'zrem'"));
            }
            let (mut z, exp) = match get_zset_mut(eng, &args[1]) {
                Ok(v) => v,
                Err(e) => return Some(e),
            };
            let mut rem = 0i64;
            for m in &args[2..] {
                if z.remove(m).is_some() {
                    rem += 1;
                }
            }
            if rem > 0 {
                if z.is_empty() {
                    let _ = eng.del_key(&args[1]);
                } else {
                    eng.set_record(args[1].clone(), Value::ZSet(z), exp);
                }
            }
            RespVal::int(rem)
        }
        "ZSCORE" => {
            if args.len() != 3 {
                return Some(RespVal::err("ERR wrong number of arguments for 'zscore'"));
            }
            let (z, _) = match get_zset_mut(eng, &args[1]) {
                Ok(v) => v,
                Err(e) => return Some(e),
            };
            match z.get(&args[2]) {
                Some(&sc) => RespVal::bulk(format_float(sc).into_bytes()),
                None => RespVal::nil(),
            }
        }
        "ZCARD" => {
            if args.len() != 2 {
                return Some(RespVal::err("ERR wrong number of arguments for 'zcard'"));
            }
            let (z, _) = match get_zset_mut(eng, &args[1]) {
                Ok(v) => v,
                Err(e) => return Some(e),
            };
            RespVal::int(z.len() as i64)
        }
        "ZRANK" | "ZREVRANK" => {
            if args.len() != 3 {
                return Some(RespVal::err("ERR wrong number of arguments"));
            }
            let (z, _) = match get_zset_mut(eng, &args[1]) {
                Ok(v) => v,
                Err(e) => return Some(e),
            };
            let mut sorted = zset_sorted(&z);
            if cmd == "ZREVRANK" {
                sorted.reverse();
            }
            match sorted.iter().position(|(m, _)| *m == args[2].as_slice()) {
                Some(p) => RespVal::int(p as i64),
                None => RespVal::nil(),
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
            let (mut z, exp) = match get_zset_mut(eng, &args[1]) {
                Ok(v) => v,
                Err(e) => return Some(e),
            };
            let next = z.get(&args[3]).copied().unwrap_or(0.0) + delta;
            z.insert(args[3].clone(), next);
            eng.set_record(args[1].clone(), Value::ZSet(z), exp);
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
            let (z, _) = match get_zset_mut(eng, &args[1]) {
                Ok(v) => v,
                Err(e) => return Some(e),
            };
            let cnt = z
                .values()
                .filter(|&&s| {
                    (if lo_ex { s > lo } else { s >= lo })
                        && (if hi_ex { s < hi } else { s <= hi })
                })
                .count();
            RespVal::int(cnt as i64)
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
            let (mut z, exp) = match get_zset_mut(eng, &args[1]) {
                Ok(v) => v,
                Err(e) => return Some(e),
            };
            let mut sorted: Vec<(Vec<u8>, f64)> = zset_sorted(&z)
                .into_iter()
                .map(|(m, s)| (m.to_vec(), s))
                .collect();
            if cmd == "ZPOPMAX" {
                sorted.reverse();
            }
            let mut out = Vec::new();
            for (m, s) in sorted.into_iter().take(cnt) {
                z.remove(&m);
                out.push(RespVal::bulk(m));
                out.push(RespVal::bulk(format_float(s).into_bytes()));
            }
            if z.is_empty() {
                let _ = eng.del_key(&args[1]);
            } else {
                eng.set_record(args[1].clone(), Value::ZSet(z), exp);
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
    let (mut z, exp) = match get_zset_mut(eng, &args[1]) {
        Ok(v) => v,
        Err(e) => return e,
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
    eng.set_record(args[1].clone(), Value::ZSet(z), exp);
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
    let (z, _) = match get_zset_mut(eng, &args[1]) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let mut sorted = zset_sorted(&z);
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
