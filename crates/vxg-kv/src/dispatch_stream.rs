//! Redis Streams (`X*`), Bitmaps (`SETBIT`/`GETBIT`/`BITCOUNT`/`BITOP`/`BITPOS`),
//! and 16384-register HyperLogLog (`PFADD`/`PFCOUNT`/`PFMERGE`).

use crate::dispatch_ds::{arg_str, parse_i64, WRONGTYPE};
use crate::engine::KvEngine;
use crate::resp::RespVal;
use crate::types::{now_ms, StreamEntry, StreamId, Value};

const HLL_MAGIC: &[u8; 4] = b"HYLL";
const HLL_REGS: usize = 16_384; // p = 14 (2^14 registers, ~0.81% std error)
const HLL_SIZE: usize = 4 + HLL_REGS;

type StreamData = (Vec<StreamEntry>, Option<u64>);
type StrData = (Vec<u8>, Option<u64>);

fn get_stream_mut(eng: &mut KvEngine, key: &[u8]) -> Result<StreamData, RespVal> {
    match eng.get_record(key).map_err(RespVal::err)? {
        None => Ok((Vec::new(), None)),
        Some(r) => {
            let exp = r.expires_opt();
            match r.value {
                Some(Value::Stream(v)) => Ok((v, exp)),
                _ => Err(RespVal::err(WRONGTYPE)),
            }
        }
    }
}

fn get_str_mut(eng: &mut KvEngine, key: &[u8]) -> Result<StrData, RespVal> {
    match eng.get_record(key).map_err(RespVal::err)? {
        None => Ok((Vec::new(), None)),
        Some(r) => {
            let exp = r.expires_opt();
            match r.value {
                Some(Value::Str(b)) => Ok((b, exp)),
                _ => Err(RespVal::err(WRONGTYPE)),
            }
        }
    }
}

fn parse_stream_id(s: &str, default_seq: u64) -> Result<StreamId, RespVal> {
    if s == "-" {
        return Ok(StreamId { ms: 0, seq: 0 });
    }
    if s == "+" {
        return Ok(StreamId {
            ms: u64::MAX,
            seq: u64::MAX,
        });
    }
    if let Some((a, b)) = s.split_once('-') {
        let ms = a
            .parse::<u64>()
            .map_err(|_| RespVal::err("ERR Invalid stream ID specified as stream command argument"))?;
        let seq = b
            .parse::<u64>()
            .map_err(|_| RespVal::err("ERR Invalid stream ID specified as stream command argument"))?;
        Ok(StreamId { ms, seq })
    } else {
        let ms = s
            .parse::<u64>()
            .map_err(|_| RespVal::err("ERR Invalid stream ID specified as stream command argument"))?;
        Ok(StreamId {
            ms,
            seq: default_seq,
        })
    }
}

fn encode_stream_entry(e: &StreamEntry) -> RespVal {
    let mut flat = Vec::with_capacity(e.fields.len() * 2);
    for (k, v) in &e.fields {
        flat.push(RespVal::bulk(k.clone()));
        flat.push(RespVal::bulk(v.clone()));
    }
    RespVal::array(vec![
        RespVal::bulk(e.id.format().into_bytes()),
        RespVal::array(flat),
    ])
}

/// Dispatch Stream, Bitmap, and HyperLogLog commands.
pub fn dispatch_stream_cmd(eng: &mut KvEngine, cmd: &str, args: &[Vec<u8>]) -> Option<RespVal> {
    Some(match cmd {
        // ---- REDIS STREAMS ----
        "XADD" => dispatch_xadd(eng, args),
        "XLEN" => {
            if args.len() != 2 {
                return Some(RespVal::err("ERR wrong number of arguments for 'xlen'"));
            }
            let (entries, _) = match get_stream_mut(eng, &args[1]) {
                Ok(v) => v,
                Err(e) => return Some(e),
            };
            RespVal::int(entries.len() as i64)
        }
        "XRANGE" | "XREVRANGE" => dispatch_xrange(eng, cmd == "XREVRANGE", args),
        "XREAD" => dispatch_xread(eng, args),
        "XDEL" => {
            if args.len() < 3 {
                return Some(RespVal::err("ERR wrong number of arguments for 'xdel'"));
            }
            let (mut entries, exp) = match get_stream_mut(eng, &args[1]) {
                Ok(v) => v,
                Err(e) => return Some(e),
            };
            let mut ids = Vec::new();
            for raw in &args[2..] {
                let s = std::str::from_utf8(raw).unwrap_or("");
                match parse_stream_id(s, 0) {
                    Ok(id) => ids.push(id),
                    Err(e) => return Some(e),
                }
            }
            let before = entries.len();
            entries.retain(|e| !ids.contains(&e.id));
            let removed = (before - entries.len()) as i64;
            if removed > 0 {
                eng.set_record(args[1].clone(), Value::Stream(entries), exp);
            }
            RespVal::int(removed)
        }
        "XTRIM" => dispatch_xtrim(eng, args),

        // ---- BITMAPS ----
        "SETBIT" => {
            if args.len() != 4 {
                return Some(RespVal::err("ERR wrong number of arguments for 'setbit'"));
            }
            let offset = match parse_i64(&args[2]) {
                Ok(n) if n >= 0 => n as usize,
                _ => return Some(RespVal::err("ERR bit offset is not an integer or out of range")),
            };
            let bit = match parse_i64(&args[3]) {
                Ok(0) => 0u8,
                Ok(1) => 1u8,
                _ => return Some(RespVal::err("ERR bit is not an integer or out of range")),
            };
            let rec = match eng.ensure_entry_mut(&args[1], || Value::Str(Vec::new())) {
                Ok(r) => r,
                Err(e) => return Some(RespVal::err(e)),
            };
            let Some(Value::Str(buf)) = rec.value.as_mut() else {
                return Some(RespVal::err(WRONGTYPE));
            };
            let byte_idx = offset >> 3;
            let mask = 0x80u8 >> (offset & 7);
            if buf.len() <= byte_idx {
                buf.resize(byte_idx + 1, 0);
            }
            let prev = i64::from((buf[byte_idx] & mask) != 0);
            if bit == 1 {
                buf[byte_idx] |= mask;
            } else {
                buf[byte_idx] &= !mask;
            }
            RespVal::int(prev)
        }
        "GETBIT" => {
            if args.len() != 3 {
                return Some(RespVal::err("ERR wrong number of arguments for 'getbit'"));
            }
            let offset = match parse_i64(&args[2]) {
                Ok(n) if n >= 0 => n as usize,
                _ => return Some(RespVal::err("ERR bit offset is not an integer or out of range")),
            };
            let (buf, _) = match get_str_mut(eng, &args[1]) {
                Ok(v) => v,
                Err(e) => return Some(e),
            };
            let byte_idx = offset >> 3;
            let mask = 0x80u8 >> (offset & 7);
            let val = buf.get(byte_idx).map_or(0, |&b| i64::from((b & mask) != 0));
            RespVal::int(val)
        }
        "BITCOUNT" => dispatch_bitcount(eng, args),
        "BITOP" => dispatch_bitop(eng, args),
        "BITPOS" => dispatch_bitpos(eng, args),

        // ---- HYPERLOGLOG ----
        "PFADD" => dispatch_pfadd(eng, args),
        "PFCOUNT" => dispatch_pfcount(eng, args),
        "PFMERGE" => dispatch_pfmerge(eng, args),

        _ => return None,
    })
}

fn dispatch_xadd(eng: &mut KvEngine, args: &[Vec<u8>]) -> RespVal {
    if args.len() < 5 {
        return RespVal::err("ERR wrong number of arguments for 'xadd'");
    }
    let mut i = 2usize;
    let mut maxlen: Option<usize> = None;
    if arg_str(args, i).is_some_and(|s| s.eq_ignore_ascii_case("MAXLEN")) {
        i += 1;
        if arg_str(args, i).is_some_and(|s| s == "~" || s == "=") {
            i += 1;
        }
        let Ok(n) = args
            .get(i)
            .ok_or_else(|| RespVal::err("ERR syntax error"))
            .and_then(|b| parse_i64(b))
        else {
            return RespVal::err("ERR syntax error");
        };
        maxlen = Some(n.max(0) as usize);
        i += 1;
    }
    let Some(id_tok) = arg_str(args, i) else {
        return RespVal::err("ERR syntax error");
    };
    i += 1;
    if i >= args.len() || !(args.len() - i).is_multiple_of(2) {
        return RespVal::err("ERR wrong number of arguments for 'xadd'");
    }
    let rec = match eng.ensure_entry_mut(&args[1], || Value::Stream(Vec::new())) {
        Ok(r) => r,
        Err(e) => return RespVal::err(e),
    };
    let Some(Value::Stream(entries)) = rec.value.as_mut() else {
        return RespVal::err(WRONGTYPE);
    };
    let last_id = entries.last().map_or(StreamId { ms: 0, seq: 0 }, |e| e.id);
    let new_id = if id_tok == "*" {
        let now = now_ms();
        if now > last_id.ms {
            StreamId { ms: now, seq: 0 }
        } else {
            StreamId { ms: last_id.ms, seq: last_id.seq + 1 }
        }
    } else {
        match parse_stream_id(id_tok, 0) {
            Ok(id) if id > last_id => id,
            Ok(_) => return RespVal::err("ERR stream ID must be greater than top item"),
            Err(e) => return e,
        }
    };
    let fields = args[i..]
        .as_chunks::<2>()
        .0
        .iter()
        .map(|p| (p[0].clone(), p[1].clone()))
        .collect();
    entries.push(StreamEntry { id: new_id, fields });
    if let Some(max) = maxlen {
        if entries.len() > max {
            entries.drain(..entries.len() - max);
        }
    }
    RespVal::bulk(new_id.format().into_bytes())
}

fn dispatch_xrange(eng: &mut KvEngine, rev: bool, args: &[Vec<u8>]) -> RespVal {
    if args.len() < 4 {
        return RespVal::err("ERR wrong number of arguments for 'xrange'");
    }
    let (a_s, b_s) = (arg_str(args, 2).unwrap_or("-"), arg_str(args, 3).unwrap_or("+"));
    let (lo_s, hi_s) = if rev { (b_s, a_s) } else { (a_s, b_s) };
    let (lo, hi) = match (parse_stream_id(lo_s, 0), parse_stream_id(hi_s, u64::MAX)) {
        (Ok(a), Ok(b)) => (a, b),
        (Err(e), _) | (_, Err(e)) => return e,
    };
    let mut count = usize::MAX;
    if args.len() >= 6 && arg_str(args, 4).is_some_and(|s| s.eq_ignore_ascii_case("COUNT")) {
        count = parse_i64(&args[5]).unwrap_or(0).max(0) as usize;
    }
    let (entries, _) = match get_stream_mut(eng, &args[1]) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let mut out = Vec::new();
    if rev {
        for e in entries.iter().rev() {
            if e.id >= lo && e.id <= hi {
                out.push(encode_stream_entry(e));
                if out.len() >= count {
                    break;
                }
            }
        }
    } else {
        for e in &entries {
            if e.id >= lo && e.id <= hi {
                out.push(encode_stream_entry(e));
                if out.len() >= count {
                    break;
                }
            }
        }
    }
    RespVal::array(out)
}

fn dispatch_xread(eng: &mut KvEngine, args: &[Vec<u8>]) -> RespVal {
    let mut i = 1usize;
    let mut count = usize::MAX;
    while i < args.len() {
        let tok = arg_str(args, i).unwrap_or("");
        if tok.eq_ignore_ascii_case("COUNT") && i + 1 < args.len() {
            count = parse_i64(&args[i + 1]).unwrap_or(0).max(0) as usize;
            i += 2;
        } else if tok.eq_ignore_ascii_case("STREAMS") {
            i += 1;
            break;
        } else {
            return RespVal::err("ERR syntax error");
        }
    }
    let rem = args.len().saturating_sub(i);
    if rem == 0 || !rem.is_multiple_of(2) {
        return RespVal::err("ERR Unbalanced 'xread' list of streams");
    }
    let n_streams = rem / 2;
    let keys = &args[i..i + n_streams];
    let ids = &args[i + n_streams..];
    let mut result = Vec::new();
    for (k, id_b) in keys.iter().zip(ids.iter()) {
        let id_s = std::str::from_utf8(id_b).unwrap_or("0-0");
        let after = match parse_stream_id(id_s, 0) {
            Ok(v) => v,
            Err(e) => return e,
        };
        let (entries, _) = match get_stream_mut(eng, k) {
            Ok(v) => v,
            Err(e) => return e,
        };
        let matched: Vec<RespVal> = entries
            .iter()
            .filter(|e| e.id > after)
            .take(count)
            .map(encode_stream_entry)
            .collect();
        if !matched.is_empty() {
            result.push(RespVal::array(vec![
                RespVal::bulk(k.clone()),
                RespVal::array(matched),
            ]));
        }
    }
    if result.is_empty() {
        RespVal::nil_array()
    } else {
        RespVal::array(result)
    }
}

fn dispatch_xtrim(eng: &mut KvEngine, args: &[Vec<u8>]) -> RespVal {
    if args.len() < 4 || !arg_str(args, 2).is_some_and(|s| s.eq_ignore_ascii_case("MAXLEN")) {
        return RespVal::err("ERR syntax error for 'xtrim'");
    }
    let idx = if arg_str(args, 3).is_some_and(|s| s == "~" || s == "=") {
        4
    } else {
        3
    };
    let Some(raw_n) = args.get(idx) else {
        return RespVal::err("ERR syntax error for 'xtrim'");
    };
    let max = parse_i64(raw_n).unwrap_or(0).max(0) as usize;
    let (mut entries, exp) = match get_stream_mut(eng, &args[1]) {
        Ok(v) => v,
        Err(e) => return e,
    };
    if entries.len() <= max {
        return RespVal::int(0);
    }
    let drop_n = entries.len() - max;
    entries.drain(..drop_n);
    eng.set_record(args[1].clone(), Value::Stream(entries), exp);
    RespVal::int(drop_n as i64)
}

fn dispatch_bitcount(eng: &mut KvEngine, args: &[Vec<u8>]) -> RespVal {
    if args.len() < 2 {
        return RespVal::err("ERR wrong number of arguments for 'bitcount'");
    }
    let (buf, _) = match get_str_mut(eng, &args[1]) {
        Ok(v) => v,
        Err(e) => return e,
    };
    if buf.is_empty() {
        return RespVal::int(0);
    }
    let slice = if args.len() >= 4 {
        let (Ok(s), Ok(e)) = (parse_i64(&args[2]), parse_i64(&args[3])) else {
            return RespVal::err("ERR value is not an integer");
        };
        let l = buf.len() as i64;
        let a = (if s < 0 { l + s } else { s }).max(0);
        let b = (if e < 0 { l + e } else { e }).min(l - 1);
        if a > b || a >= l {
            return RespVal::int(0);
        }
        &buf[a as usize..=b as usize]
    } else {
        &buf[..]
    };
    let ones: u32 = slice.iter().map(|b| b.count_ones()).sum();
    RespVal::int(ones as i64)
}

fn dispatch_bitop(eng: &mut KvEngine, args: &[Vec<u8>]) -> RespVal {
    if args.len() < 4 {
        return RespVal::err("ERR wrong number of arguments for 'bitop'");
    }
    let op = arg_str(args, 1).unwrap_or("").to_ascii_uppercase();
    let mut srcs = Vec::with_capacity(args.len() - 3);
    let mut max_len = 0usize;
    for k in &args[3..] {
        let (b, _) = match get_str_mut(eng, k) {
            Ok(v) => v,
            Err(e) => return e,
        };
        max_len = max_len.max(b.len());
        srcs.push(b);
    }
    let mut out = vec![0u8; max_len];
    match op.as_str() {
        "NOT" => {
            if srcs.len() != 1 {
                return RespVal::err("ERR BITOP NOT must be called with a single source key");
            }
            for (i, &b) in srcs[0].iter().enumerate() {
                out[i] = !b;
            }
        }
        "AND" => {
            for (i, slot) in out.iter_mut().enumerate() {
                let mut v = 0xffu8;
                for s in &srcs {
                    v &= s.get(i).copied().unwrap_or(0);
                }
                *slot = v;
            }
        }
        "OR" => {
            for s in &srcs {
                for (i, &b) in s.iter().enumerate() {
                    out[i] |= b;
                }
            }
        }
        "XOR" => {
            for s in &srcs {
                for (i, &b) in s.iter().enumerate() {
                    out[i] ^= b;
                }
            }
        }
        _ => return RespVal::err("ERR syntax error"),
    }
    let len = out.len() as i64;
    if out.is_empty() {
        let _ = eng.del_key(&args[2]);
    } else {
        eng.set_record(args[2].clone(), Value::Str(out), None);
    }
    RespVal::int(len)
}

fn dispatch_bitpos(eng: &mut KvEngine, args: &[Vec<u8>]) -> RespVal {
    if args.len() < 3 {
        return RespVal::err("ERR wrong number of arguments for 'bitpos'");
    }
    let target = match parse_i64(&args[2]) {
        Ok(0) => false,
        Ok(1) => true,
        _ => return RespVal::err("ERR The bit argument must be 1 or 0."),
    };
    let (buf, _) = match get_str_mut(eng, &args[1]) {
        Ok(v) => v,
        Err(e) => return e,
    };
    for (bi, &byte) in buf.iter().enumerate() {
        for bit in 0..8 {
            let is_one = (byte & (0x80u8 >> bit)) != 0;
            if is_one == target {
                return RespVal::int((bi * 8 + bit) as i64);
            }
        }
    }
    RespVal::int(-1)
}

fn hll_hash(elem: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in elem {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51afd7ed558ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
    h ^ (h >> 33)
}

fn load_hll(eng: &mut KvEngine, key: &[u8]) -> Result<(Vec<u8>, Option<u64>), RespVal> {
    let (buf, exp) = get_str_mut(eng, key)?;
    if buf.is_empty() {
        let mut h = vec![0u8; HLL_SIZE];
        h[..4].copy_from_slice(HLL_MAGIC);
        return Ok((h, exp));
    }
    if buf.len() != HLL_SIZE || &buf[..4] != HLL_MAGIC {
        return Err(RespVal::err(WRONGTYPE));
    }
    Ok((buf, exp))
}

fn hll_estimate(regs: &[u8]) -> i64 {
    let m = HLL_REGS as f64;
    let alpha = 0.7213 / (1.0 + 1.079 / m);
    let mut sum = 0.0f64;
    let mut zeros = 0usize;
    for &r in regs {
        if r == 0 {
            zeros += 1;
        }
        sum += 2.0f64.powi(-(r as i32));
    }
    let raw = alpha * m * m / sum;
    if raw <= 2.5 * m && zeros > 0 {
        (m * (m / zeros as f64).ln()).round() as i64
    } else {
        raw.round() as i64
    }
}

fn dispatch_pfadd(eng: &mut KvEngine, args: &[Vec<u8>]) -> RespVal {
    if args.len() < 2 {
        return RespVal::err("ERR wrong number of arguments for 'pfadd'");
    }
    let (mut hll, exp) = match load_hll(eng, &args[1]) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let mut changed = 0i64;
    for elem in &args[2..] {
        let h = hll_hash(elem);
        let idx = (h & ((HLL_REGS as u64) - 1)) as usize;
        let w = h >> 14;
        let rho = (w.trailing_zeros().min(50) as u8) + 1;
        if rho > hll[4 + idx] {
            hll[4 + idx] = rho;
            changed = 1;
        }
    }
    if changed == 1 || eng.get_record(&args[1]).ok().flatten().is_none() {
        eng.set_record(args[1].clone(), Value::Str(hll), exp);
    }
    RespVal::int(changed)
}

fn dispatch_pfcount(eng: &mut KvEngine, args: &[Vec<u8>]) -> RespVal {
    if args.len() < 2 {
        return RespVal::err("ERR wrong number of arguments for 'pfcount'");
    }
    let mut merged = [0u8; HLL_REGS];
    for k in &args[1..] {
        let (hll, _) = match load_hll(eng, k) {
            Ok(v) => v,
            Err(e) => return e,
        };
        for (dst, &src) in merged.iter_mut().zip(&hll[4..]) {
            if src > *dst {
                *dst = src;
            }
        }
    }
    RespVal::int(hll_estimate(&merged))
}

fn dispatch_pfmerge(eng: &mut KvEngine, args: &[Vec<u8>]) -> RespVal {
    if args.len() < 2 {
        return RespVal::err("ERR wrong number of arguments for 'pfmerge'");
    }
    let (mut dst, exp) = match load_hll(eng, &args[1]) {
        Ok(v) => v,
        Err(e) => return e,
    };
    for k in &args[2..] {
        let (src, _) = match load_hll(eng, k) {
            Ok(v) => v,
            Err(e) => return e,
        };
        for (d, &s) in dst[4..].iter_mut().zip(&src[4..]) {
            if s > *d {
                *d = s;
            }
        }
    }
    eng.set_record(args[1].clone(), Value::Str(dst), exp);
    RespVal::ok()
}
