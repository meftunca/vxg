//! `SET`, `INCR/DECR`, `EXPIRE`, `SCAN`, `VXG.*` superpowers, and `INFO`/`VXG.STATS` handlers.

use vxg::SqlFormat;

use crate::dispatch::ConnState;
use crate::dispatch_ds::{arg_str, parse_i64, WRONGTYPE};
use crate::engine::KvEngine;
use crate::resp::RespVal;
use crate::types::{now_ms, Value};

pub(crate) fn dispatch_vxg_cmd(
    eng: &mut KvEngine,
    conn: &mut ConnState,
    cmd: &str,
    args: &[Vec<u8>],
) -> Option<RespVal> {
    Some(match cmd {
        "INFO" | "VXG.STATS" => build_stats_info(eng, conn),
        "SET" => handle_set(eng, args),
        "INCR" | "DECR" | "INCRBY" | "DECRBY" => handle_incr(eng, cmd, args),
        "EXPIRE" | "PEXPIRE" | "EXPIREAT" | "PEXPIREAT" => handle_expire(eng, cmd, args),
        "SCAN" => handle_scan(eng, args),
        "VXG.COMMIT" => match eng.commit() {
            Ok(ep) => RespVal::int(ep as i64),
            Err(e) => RespVal::err(e),
        },
        "VXG.PULL" => match eng.pull() {
            Ok(n) => RespVal::int(n as i64),
            Err(e) => RespVal::err(e),
        },
        "VXG.COMPACT" => match eng.compact() {
            Ok(snap) => RespVal::Simple(format!(
                "OK epoch={} keys={} raw={}B compressed={}B ratio={:.4}",
                snap.epoch,
                snap.key_count,
                snap.raw_bytes,
                snap.compressed_bytes,
                if snap.raw_bytes > 0 {
                    snap.compressed_bytes as f64 / snap.raw_bytes as f64
                } else {
                    0.0
                }
            )),
            Err(e) => RespVal::err(e),
        },
        "VXG.SNAPSHOTS" => {
            let mut items = Vec::new();
            for h in &eng.manifest.history {
                items.push(RespVal::bulk(
                    format!(
                        "epoch={} keys={} raw={} compressed={} status=history",
                        h.epoch, h.key_count, h.raw_bytes, h.compressed_bytes
                    )
                    .into_bytes(),
                ));
            }
            if let Some(ref b) = eng.manifest.base_snapshot {
                items.push(RespVal::bulk(
                    format!(
                        "epoch={} keys={} raw={} compressed={} status=active",
                        b.epoch, b.key_count, b.raw_bytes, b.compressed_bytes
                    )
                    .into_bytes(),
                ));
            }
            RespVal::array(items)
        }
        "VXG.AT" => {
            if args.len() != 4 || !arg_str(args, 2).is_some_and(|s| s.eq_ignore_ascii_case("GET")) {
                return Some(RespVal::err("ERR usage: VXG.AT <epoch> GET <key>"));
            }
            let Ok(ep) = arg_str(args, 1).unwrap_or("").parse::<u64>() else {
                return Some(RespVal::err("ERR invalid epoch"));
            };
            match eng.get_record_at_epoch(ep, &args[3]) {
                Ok(Some(r)) => match r.value {
                    Some(Value::Str(b)) => RespVal::bulk(b),
                    _ => RespVal::nil(),
                },
                Ok(None) => RespVal::nil(),
                Err(e) => RespVal::err(e),
            }
        }
        "VXG.PIN" => {
            if args.len() != 2 {
                return Some(RespVal::err("ERR usage: VXG.PIN <epoch>"));
            }
            let Ok(ep) = arg_str(args, 1).unwrap_or("").parse::<u64>() else {
                return Some(RespVal::err("ERR invalid epoch"));
            };
            if eng.manifest.find_snapshot(ep).is_none() {
                return Some(RespVal::err(format!("ERR snapshot epoch {ep} not found")));
            }
            conn.pinned_epoch = Some(ep);
            RespVal::ok()
        }
        "VXG.UNPIN" => {
            conn.pinned_epoch = None;
            RespVal::ok()
        }
        "VXG.CAS" => handle_vxg_cas(eng, args),
        "VXG.SQL" => handle_vxg_sql(eng, conn, args),
        _ => return None,
    })
}

fn handle_set(eng: &mut KvEngine, args: &[Vec<u8>]) -> RespVal {
    if args.len() < 3 {
        return RespVal::err("ERR wrong number of arguments for 'set'");
    }
    let key = &args[1];
    let val = &args[2];
    let (mut nx, mut xx, mut get_old, mut keepttl, mut sync) = (false, false, false, false, false);
    let mut exp_ms: Option<u64> = None;
    let mut i = 3usize;
    while i < args.len() {
        let opt = arg_str(args, i).unwrap_or("").to_ascii_uppercase();
        match opt.as_str() {
            "NX" => nx = true,
            "XX" => xx = true,
            "GET" => get_old = true,
            "KEEPTTL" => keepttl = true,
            "SYNC" => sync = true,
            "EX" if i + 1 < args.len() => {
                let Ok(s) = parse_i64(&args[i + 1]) else {
                    return RespVal::err("ERR invalid expire time");
                };
                if s <= 0 {
                    return RespVal::err("ERR invalid expire time");
                }
                exp_ms = Some(now_ms().saturating_add((s as u64) * 1000));
                i += 1;
            }
            "PX" if i + 1 < args.len() => {
                let Ok(ms) = parse_i64(&args[i + 1]) else {
                    return RespVal::err("ERR invalid expire time");
                };
                if ms <= 0 {
                    return RespVal::err("ERR invalid expire time");
                }
                exp_ms = Some(now_ms().saturating_add(ms as u64));
                i += 1;
            }
            "EXAT" if i + 1 < args.len() => {
                let Ok(s) = parse_i64(&args[i + 1]) else {
                    return RespVal::err("ERR invalid expire time");
                };
                exp_ms = Some((s.max(0) as u64) * 1000);
                i += 1;
            }
            "PXAT" if i + 1 < args.len() => {
                let Ok(ms) = parse_i64(&args[i + 1]) else {
                    return RespVal::err("ERR invalid expire time");
                };
                exp_ms = Some(ms.max(0) as u64);
                i += 1;
            }
            _ => return RespVal::err("ERR syntax error"),
        }
        i += 1;
    }
    if !(nx || xx || get_old || keepttl) {
        eng.set_record(key.clone(), Value::Str(val.clone()), exp_ms);
        if sync {
            if let Err(e) = eng.commit() {
                return RespVal::err(e);
            }
        }
        return RespVal::ok();
    }
    let existing = match eng.get_record(key) {
        Ok(v) => v,
        Err(e) => return RespVal::err(e),
    };
    let old_val = existing.as_ref().and_then(|r| match &r.value {
        Some(Value::Str(b)) => Some(b.clone()),
        _ => None,
    });
    if (nx && existing.is_some()) || (xx && existing.is_none()) {
        return if get_old {
            old_val.map_or_else(RespVal::nil, RespVal::bulk)
        } else {
            RespVal::nil()
        };
    }
    let final_exp = if keepttl {
        existing.and_then(|r| r.expires_at_ms)
    } else {
        exp_ms
    };
    eng.set_record(key.clone(), Value::Str(val.clone()), final_exp);
    if sync {
        if let Err(e) = eng.commit() {
            return RespVal::err(e);
        }
    }
    if get_old {
        old_val.map_or_else(RespVal::nil, RespVal::bulk)
    } else {
        RespVal::ok()
    }
}

fn handle_incr(eng: &mut KvEngine, cmd: &str, args: &[Vec<u8>]) -> RespVal {
    let delta = match cmd {
        "INCR" => 1i64,
        "DECR" => -1i64,
        "INCRBY" | "DECRBY" => {
            if args.len() != 3 {
                return RespVal::err("ERR wrong number of arguments");
            }
            let v = match parse_i64(&args[2]) {
                Ok(n) => n,
                Err(e) => return e,
            };
            if cmd == "DECRBY" { -v } else { v }
        }
        _ => 0,
    };
    if args.len() < 2 {
        return RespVal::err("ERR wrong number of arguments");
    }
    let rec = match eng.ensure_entry_mut(&args[1], || Value::Str(b"0".to_vec())) {
        Ok(r) => r,
        Err(e) => return RespVal::err(e),
    };
    let Some(Value::Str(buf)) = rec.value.as_mut() else {
        return RespVal::err(WRONGTYPE);
    };
    let cur = match parse_i64(buf) {
        Ok(n) => n,
        Err(e) => return e,
    };
    let next = cur.saturating_add(delta);
    buf.clear();
    use std::io::Write;
    let _ = write!(buf, "{next}");
    RespVal::int(next)
}

fn handle_expire(eng: &mut KvEngine, cmd: &str, args: &[Vec<u8>]) -> RespVal {
    if args.len() != 3 {
        return RespVal::err("ERR wrong number of arguments");
    }
    let v = match parse_i64(&args[2]) {
        Ok(n) => n,
        Err(e) => return e,
    };
    let exp_ms = match cmd {
        "EXPIRE" => now_ms().saturating_add(v.max(0) as u64 * 1000),
        "PEXPIRE" => now_ms().saturating_add(v.max(0) as u64),
        "EXPIREAT" => v.max(0) as u64 * 1000,
        "PEXPIREAT" => v.max(0) as u64,
        _ => 0,
    };
    if v <= 0 {
        return RespVal::int(i64::from(matches!(eng.del_key(&args[1]), Ok(true))));
    }
    RespVal::int(i64::from(matches!(
        eng.set_expiry(&args[1], Some(exp_ms)),
        Ok(true)
    )))
}

fn handle_scan(eng: &mut KvEngine, args: &[Vec<u8>]) -> RespVal {
    if args.len() < 2 {
        return RespVal::err("ERR wrong number of arguments for 'scan'");
    }
    let cursor = parse_i64(&args[1]).unwrap_or(0).max(0) as usize;
    let mut pat: &[u8] = b"*";
    let mut count = 10usize;
    let mut i = 2;
    while i + 1 < args.len() {
        let opt = arg_str(args, i).unwrap_or("").to_ascii_uppercase();
        if opt == "MATCH" {
            pat = &args[i + 1];
            i += 2;
        } else if opt == "COUNT" {
            count = parse_i64(&args[i + 1]).unwrap_or(10).max(1) as usize;
            i += 2;
        } else {
            i += 1;
        }
    }
    let all = match eng.matching_keys(pat) {
        Ok(v) => v,
        Err(e) => return RespVal::err(e),
    };
    let end = (cursor + count).min(all.len());
    let next_cursor = if end >= all.len() { 0 } else { end };
    let batch = if cursor < all.len() {
        all[cursor..end]
            .iter()
            .cloned()
            .map(RespVal::bulk)
            .collect()
    } else {
        Vec::new()
    };
    RespVal::array(vec![
        RespVal::bulk(next_cursor.to_string().into_bytes()),
        RespVal::array(batch),
    ])
}

fn handle_vxg_cas(eng: &mut KvEngine, args: &[Vec<u8>]) -> RespVal {
    if args.len() < 4 {
        return RespVal::err("ERR usage: VXG.CAS key expected_or_NIL new_val [EX s | PX ms]");
    }
    let expected = if args[2] == b"__NIL__" {
        None
    } else {
        Some(args[2].as_slice())
    };
    let mut exp_ms = None;
    if args.len() == 6 {
        let opt = arg_str(args, 4).unwrap_or("").to_ascii_uppercase();
        if let Ok(v) = parse_i64(&args[5]) {
            if opt == "EX" && v > 0 {
                exp_ms = Some(now_ms().saturating_add(v as u64 * 1000));
            } else if opt == "PX" && v > 0 {
                exp_ms = Some(now_ms().saturating_add(v as u64));
            }
        }
    }
    match eng.cas_string(args[1].clone(), expected, args[3].clone(), exp_ms) {
        Ok(ok) => RespVal::int(i64::from(ok)),
        Err(e) => RespVal::err(e),
    }
}

fn handle_vxg_sql(eng: &mut KvEngine, conn: &ConnState, args: &[Vec<u8>]) -> RespVal {
    if args.len() < 2 {
        return RespVal::err("ERR usage: VXG.SQL \"SELECT ...\" [JSON|CSV|TSV]");
    }
    let Some(sql) = arg_str(args, 1) else {
        return RespVal::err("ERR non-UTF8 SQL query");
    };
    let fmt = match arg_str(args, 2).unwrap_or("TSV").to_ascii_uppercase().as_str() {
        "JSON" => SqlFormat::Json,
        "CSV" => SqlFormat::Csv,
        _ => SqlFormat::Tsv,
    };
    match eng.execute_sql(sql, fmt, conn.pinned_epoch) {
        Ok(bytes) => RespVal::bulk(bytes),
        Err(e) => RespVal::err(e),
    }
}

fn build_stats_info(eng: &KvEngine, conn: &ConnState) -> RespVal {
    let base_info = match &eng.base {
        Some((s, idx)) => format!(
            "base_epoch:{}\r\nbase_keys:{}\r\nbase_blocks:{}\r\nbase_raw_bytes:{}\r\nbase_compressed_bytes:{}\r\nbase_ratio:{:.4}\r\n",
            s.epoch,
            s.key_count,
            idx.blocks.len(),
            s.raw_bytes,
            s.compressed_bytes,
            if s.raw_bytes > 0 {
                s.compressed_bytes as f64 / s.raw_bytes as f64
            } else {
                0.0
            }
        ),
        None => "base_epoch:0\r\n".to_owned(),
    };
    let s = format!(
        "# VXG-KV\r\npod_id:{}\r\nepoch:{}\r\npinned_epoch:{:?}\r\nmem_dirty_keys:{}\r\nwal_overlay_keys:{}\r\nactive_wals:{}\r\nhistory_snapshots:{}\r\n{base_info}cache_blocks:{}\r\ncache_bytes:{}\r\ncache_hits:{}\r\ncache_misses:{}\r\nrange_requests:{}\r\nrange_bytes:{}\r\nbloom_filtered:{}\r\nout_of_range_filtered:{}\r\ncas_commits:{}\r\ncas_retries:{}\r\ncompactions:{}\r\npulls:{}\r\npubsub_remote_delivered:{}\r\n",
        eng.cfg.pod_id,
        eng.manifest.epoch,
        conn.pinned_epoch,
        eng.mem.len(),
        eng.wal_overlay.len(),
        eng.manifest.wals.len(),
        eng.manifest.history.len(),
        eng.cache.cached_blocks(),
        eng.cache.bytes_used(),
        eng.stats.cache_hits.load(std::sync::atomic::Ordering::Relaxed),
        eng.stats.cache_misses,
        eng.stats.range_requests,
        eng.stats.range_bytes,
        eng.stats.bloom_filtered.load(std::sync::atomic::Ordering::Relaxed),
        eng.stats.out_of_range_filtered.load(std::sync::atomic::Ordering::Relaxed),
        eng.stats.cas_commits,
        eng.stats.cas_retries,
        eng.stats.compactions,
        eng.stats.pulls,
        eng.stats.pubsub_remote_delivered
    );
    RespVal::bulk(s.into_bytes())
}

