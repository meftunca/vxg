//! Redis Stream Consumer Groups (`XGROUP CREATE/DESTROY/SETID`, `XREADGROUP`, `XACK`, `XPENDING`).
//!
//! Group offsets and Pending Entry Lists (PEL) are persisted inside a reserved internal key
//! (`\x00__vxg_xg__:<stream>`) so they automatically survive restarts and replicate across pods!

use std::collections::BTreeMap;

use crate::dispatch_ds::{arg_str, parse_i64, WRONGTYPE};
use crate::engine::KvEngine;
use crate::resp::RespVal;
use crate::types::{now_ms, StreamEntry, StreamId, Value};

/// Prefix for internal metadata keys hidden from `DBSIZE`, `KEYS`, and `SCAN`.
pub const INTERNAL_KEY_PREFIX: &[u8] = b"\x00__vxg_";
const XGROUP_PREFIX: &[u8] = b"\x00__vxg_xg__:";

#[derive(Clone, Debug)]
struct PelEntry {
    id: StreamId,
    consumer: Vec<u8>,
    delivered_ms: u64,
    deliveries: u32,
}

#[derive(Clone, Debug)]
struct GroupMeta {
    last_delivered: StreamId,
    pel: Vec<PelEntry>,
}

type GroupTable = BTreeMap<Vec<u8>, GroupMeta>;

fn meta_key(stream: &[u8]) -> Vec<u8> {
    let mut k = Vec::with_capacity(XGROUP_PREFIX.len() + stream.len());
    k.extend_from_slice(XGROUP_PREFIX);
    k.extend_from_slice(stream);
    k
}

fn encode_groups(tbl: &GroupTable) -> Vec<u8> {
    let mut out = Vec::with_capacity(64);
    out.extend_from_slice(&(tbl.len() as u32).to_le_bytes());
    for (name, g) in tbl {
        out.extend_from_slice(&(name.len() as u32).to_le_bytes());
        out.extend_from_slice(name);
        out.extend_from_slice(&g.last_delivered.ms.to_le_bytes());
        out.extend_from_slice(&g.last_delivered.seq.to_le_bytes());
        out.extend_from_slice(&(g.pel.len() as u32).to_le_bytes());
        for p in &g.pel {
            out.extend_from_slice(&p.id.ms.to_le_bytes());
            out.extend_from_slice(&p.id.seq.to_le_bytes());
            out.extend_from_slice(&(p.consumer.len() as u32).to_le_bytes());
            out.extend_from_slice(&p.consumer);
            out.extend_from_slice(&p.delivered_ms.to_le_bytes());
            out.extend_from_slice(&p.deliveries.to_le_bytes());
        }
    }
    out
}

fn decode_groups(raw: &[u8]) -> GroupTable {
    let mut pos = 0usize;
    let r_u32 = |p: &mut usize| -> Option<u32> {
        let s = raw.get(*p..*p + 4)?;
        *p += 4;
        Some(u32::from_le_bytes(s.try_into().ok()?))
    };
    let r_u64 = |p: &mut usize| -> Option<u64> {
        let s = raw.get(*p..*p + 8)?;
        *p += 8;
        Some(u64::from_le_bytes(s.try_into().ok()?))
    };
    let mut tbl = BTreeMap::new();
    let Some(ng) = r_u32(&mut pos) else {
        return tbl;
    };
    for _ in 0..ng {
        let Some(nlen) = r_u32(&mut pos) else { break };
        let Some(name) = raw.get(pos..pos + nlen as usize) else { break };
        pos += nlen as usize;
        let (Some(ms), Some(seq), Some(npel)) = (r_u64(&mut pos), r_u64(&mut pos), r_u32(&mut pos))
        else {
            break;
        };
        let mut pel = Vec::with_capacity(npel as usize);
        for _ in 0..npel {
            let (Some(ims), Some(iseq), Some(clen)) =
                (r_u64(&mut pos), r_u64(&mut pos), r_u32(&mut pos))
            else {
                break;
            };
            let Some(consumer) = raw.get(pos..pos + clen as usize) else { break };
            pos += clen as usize;
            let (Some(dms), Some(deliv)) = (r_u64(&mut pos), r_u32(&mut pos)) else { break };
            pel.push(PelEntry {
                id: StreamId { ms: ims, seq: iseq },
                consumer: consumer.to_vec(),
                delivered_ms: dms,
                deliveries: deliv,
            });
        }
        tbl.insert(
            name.to_vec(),
            GroupMeta {
                last_delivered: StreamId { ms, seq },
                pel,
            },
        );
    }
    tbl
}

fn load_groups(eng: &mut KvEngine, stream: &[u8]) -> Result<GroupTable, RespVal> {
    let mk = meta_key(stream);
    match eng.get_record_or_load(&mk).map_err(RespVal::err)? {
        Some(r) => match r.value.as_ref() {
            Some(Value::Str(b)) => Ok(decode_groups(b)),
            _ => Ok(BTreeMap::new()),
        },
        None => Ok(BTreeMap::new()),
    }
}

fn save_groups(eng: &mut KvEngine, stream: &[u8], tbl: &GroupTable) {
    let mk = meta_key(stream);
    if tbl.is_empty() {
        let _ = eng.del_key(&mk);
    } else {
        eng.set_str_in_place(&mk, &encode_groups(tbl));
    }
}

fn parse_sid(s: &str) -> Result<StreamId, RespVal> {
    if s == "0" || s == "0-0" {
        return Ok(StreamId { ms: 0, seq: 0 });
    }
    if let Some((a, b)) = s.split_once('-') {
        let ms = a
            .parse::<u64>()
            .map_err(|_| RespVal::err("ERR Invalid stream ID"))?;
        let seq = b
            .parse::<u64>()
            .map_err(|_| RespVal::err("ERR Invalid stream ID"))?;
        Ok(StreamId { ms, seq })
    } else {
        let ms = s
            .parse::<u64>()
            .map_err(|_| RespVal::err("ERR Invalid stream ID"))?;
        Ok(StreamId { ms, seq: 0 })
    }
}

fn encode_entry(e: &StreamEntry) -> RespVal {
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

/// Dispatch Stream Consumer Group commands (`XGROUP`, `XREADGROUP`, `XACK`, `XPENDING`).
pub fn dispatch_xgroup_cmd(eng: &mut KvEngine, cmd: &str, args: &[Vec<u8>]) -> Option<RespVal> {
    Some(match cmd {
        "XGROUP" => dispatch_xgroup(eng, args),
        "XREADGROUP" => dispatch_xreadgroup(eng, args),
        "XACK" => dispatch_xack(eng, args),
        "XPENDING" => dispatch_xpending(eng, args),
        _ => return None,
    })
}

fn dispatch_xgroup(eng: &mut KvEngine, args: &[Vec<u8>]) -> RespVal {
    let sub = arg_str(args, 1).unwrap_or("").to_ascii_uppercase();
    match sub.as_str() {
        "CREATE" => {
            if args.len() < 5 {
                return RespVal::err("ERR wrong number of arguments for 'xgroup|create'");
            }
            let stream_key = &args[2];
            let group_name = &args[3];
            let id_str = arg_str(args, 4).unwrap_or("0");
            let mkstream = args[5..]
                .iter()
                .any(|a| a.eq_ignore_ascii_case(b"MKSTREAM"));

            let last_id = match eng.get_record_or_load(stream_key) {
                Ok(None) => {
                    if !mkstream {
                        return RespVal::err(
                            "ERR The XGROUP subcommand requires the key to exist. Use MKSTREAM.",
                        );
                    }
                    eng.set_record(stream_key.clone(), Value::Stream(Vec::new()), None);
                    StreamId { ms: 0, seq: 0 }
                }
                Ok(Some(r)) => match r.value.as_ref() {
                    Some(Value::Stream(v)) => {
                        v.last().map_or(StreamId { ms: 0, seq: 0 }, |e| e.id)
                    }
                    _ => return RespVal::err(WRONGTYPE),
                },
                Err(e) => return RespVal::err(e),
            };
            let start_id = if id_str == "$" {
                last_id
            } else {
                match parse_sid(id_str) {
                    Ok(id) => id,
                    Err(e) => return e,
                }
            };
            let mut tbl = match load_groups(eng, stream_key) {
                Ok(t) => t,
                Err(e) => return e,
            };
            if tbl.contains_key(group_name) {
                return RespVal::err("BUSYGROUP Consumer Group name already exists");
            }
            tbl.insert(
                group_name.clone(),
                GroupMeta {
                    last_delivered: start_id,
                    pel: Vec::new(),
                },
            );
            save_groups(eng, stream_key, &tbl);
            RespVal::ok()
        }
        "DESTROY" => {
            if args.len() != 4 {
                return RespVal::err("ERR wrong number of arguments for 'xgroup|destroy'");
            }
            let mut tbl = match load_groups(eng, &args[2]) {
                Ok(t) => t,
                Err(e) => return e,
            };
            let removed = i64::from(tbl.remove(&args[3]).is_some());
            save_groups(eng, &args[2], &tbl);
            RespVal::int(removed)
        }
        "SETID" => {
            if args.len() < 5 {
                return RespVal::err("ERR wrong number of arguments for 'xgroup|setid'");
            }
            let id_str = arg_str(args, 4).unwrap_or("0");
            let stream_last = match eng.get_record_or_load(&args[2]) {
                Ok(Some(r)) => match r.value.as_ref() {
                    Some(Value::Stream(v)) => {
                        v.last().map_or(StreamId { ms: 0, seq: 0 }, |e| e.id)
                    }
                    _ => return RespVal::err(WRONGTYPE),
                },
                Ok(None) => return RespVal::err("ERR no such key"),
                Err(e) => return RespVal::err(e),
            };
            let new_id = if id_str == "$" {
                stream_last
            } else {
                match parse_sid(id_str) {
                    Ok(id) => id,
                    Err(e) => return e,
                }
            };
            let mut tbl = match load_groups(eng, &args[2]) {
                Ok(t) => t,
                Err(e) => return e,
            };
            let Some(g) = tbl.get_mut(&args[3]) else {
                return RespVal::err("NOGROUP No such consumer group");
            };
            g.last_delivered = new_id;
            save_groups(eng, &args[2], &tbl);
            RespVal::ok()
        }
        _ => RespVal::err("ERR unknown XGROUP subcommand"),
    }
}

fn dispatch_xreadgroup(eng: &mut KvEngine, args: &[Vec<u8>]) -> RespVal {
    if args.len() < 7 || !arg_str(args, 1).is_some_and(|s| s.eq_ignore_ascii_case("GROUP")) {
        return RespVal::err("ERR syntax error in XREADGROUP");
    }
    let group_name = &args[2];
    let consumer = &args[3];
    let mut count = usize::MAX;
    let mut noack = false;
    let mut i = 4usize;
    while i < args.len() {
        let tok = arg_str(args, i).unwrap_or("").to_ascii_uppercase();
        if tok == "COUNT" && i + 1 < args.len() {
            count = parse_i64(&args[i + 1]).unwrap_or(0).max(0) as usize;
            i += 2;
        } else if tok == "BLOCK" && i + 1 < args.len() {
            i += 2;
        } else if tok == "NOACK" {
            noack = true;
            i += 1;
        } else if tok == "STREAMS" {
            i += 1;
            break;
        } else {
            return RespVal::err("ERR syntax error in XREADGROUP");
        }
    }
    let rem = &args[i..];
    if rem.is_empty() || !rem.len().is_multiple_of(2) {
        return RespVal::err("ERR Unbalanced XREADGROUP list of streams");
    }
    let n = rem.len() / 2;
    let now = now_ms();
    let mut out = Vec::new();
    for j in 0..n {
        let stream_key = &rem[j];
        let id_arg = std::str::from_utf8(&rem[n + j]).unwrap_or(">");
        let entries = match eng.get_record(stream_key) {
            Ok(Some(r)) => match r.value {
                Some(Value::Stream(v)) => v,
                _ => return RespVal::err(WRONGTYPE),
            },
            Ok(None) => return RespVal::err("NOGROUP No such key or consumer group"),
            Err(e) => return RespVal::err(e),
        };
        let mut tbl = match load_groups(eng, stream_key) {
            Ok(t) => t,
            Err(e) => return e,
        };
        let Some(g) = tbl.get_mut(group_name) else {
            return RespVal::err("NOGROUP No such consumer group");
        };
        let mut matched = Vec::new();
        if id_arg == ">" {
            for e in &entries {
                if e.id > g.last_delivered {
                    g.last_delivered = e.id;
                    if !noack {
                        g.pel.push(PelEntry {
                            id: e.id,
                            consumer: consumer.clone(),
                            delivered_ms: now,
                            deliveries: 1,
                        });
                    }
                    matched.push(encode_entry(e));
                    if matched.len() >= count {
                        break;
                    }
                }
            }
            if !matched.is_empty() {
                save_groups(eng, stream_key, &tbl);
            }
        } else {
            let after = match parse_sid(id_arg) {
                Ok(id) => id,
                Err(e) => return e,
            };
            let mut changed = false;
            for p in &mut g.pel {
                if p.consumer == *consumer && p.id > after {
                    if let Some(e) = entries.iter().find(|e| e.id == p.id) {
                        p.delivered_ms = now;
                        p.deliveries = p.deliveries.saturating_add(1);
                        changed = true;
                        matched.push(encode_entry(e));
                        if matched.len() >= count {
                            break;
                        }
                    }
                }
            }
            if changed {
                save_groups(eng, stream_key, &tbl);
            }
        }
        if !matched.is_empty() {
            out.push(RespVal::array(vec![
                RespVal::bulk(stream_key.clone()),
                RespVal::array(matched),
            ]));
        }
    }
    if out.is_empty() {
        RespVal::nil_array()
    } else {
        RespVal::array(out)
    }
}

fn dispatch_xack(eng: &mut KvEngine, args: &[Vec<u8>]) -> RespVal {
    if args.len() < 4 {
        return RespVal::err("ERR wrong number of arguments for 'xack'");
    }
    let mut ids = Vec::with_capacity(args.len() - 3);
    for raw in &args[3..] {
        let s = std::str::from_utf8(raw).unwrap_or("");
        match parse_sid(s) {
            Ok(id) => ids.push(id),
            Err(e) => return e,
        }
    }
    let mut tbl = match load_groups(eng, &args[1]) {
        Ok(t) => t,
        Err(e) => return e,
    };
    let Some(g) = tbl.get_mut(&args[2]) else {
        return RespVal::int(0);
    };
    let before = g.pel.len();
    g.pel.retain(|p| !ids.contains(&p.id));
    let acked = (before - g.pel.len()) as i64;
    if acked > 0 {
        save_groups(eng, &args[1], &tbl);
    }
    RespVal::int(acked)
}

fn dispatch_xpending(eng: &mut KvEngine, args: &[Vec<u8>]) -> RespVal {
    if args.len() < 3 {
        return RespVal::err("ERR wrong number of arguments for 'xpending'");
    }
    let tbl = match load_groups(eng, &args[1]) {
        Ok(t) => t,
        Err(e) => return e,
    };
    let Some(g) = tbl.get(&args[2]) else {
        return RespVal::err("NOGROUP No such consumer group");
    };
    if g.pel.is_empty() {
        return RespVal::array(vec![
            RespVal::int(0),
            RespVal::nil(),
            RespVal::nil(),
            RespVal::nil_array(),
        ]);
    }
    let min_id = g.pel.iter().map(|p| p.id).min().unwrap();
    let max_id = g.pel.iter().map(|p| p.id).max().unwrap();
    let mut per_c: BTreeMap<&[u8], i64> = BTreeMap::new();
    for p in &g.pel {
        *per_c.entry(&p.consumer).or_insert(0) += 1;
    }
    let consumers = per_c
        .into_iter()
        .map(|(c, cnt)| {
            RespVal::array(vec![
                RespVal::bulk(c.to_vec()),
                RespVal::bulk(cnt.to_string().into_bytes()),
            ])
        })
        .collect();
    RespVal::array(vec![
        RespVal::int(g.pel.len() as i64),
        RespVal::bulk(min_id.format().into_bytes()),
        RespVal::bulk(max_id.format().into_bytes()),
        RespVal::array(consumers),
    ])
}
