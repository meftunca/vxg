//! Primary RESP command dispatcher: String/Counter/TTL/Key/Scan, Transactions (`MULTI`/`EXEC`/`WATCH`),
//! Pub/Sub (`SUBSCRIBE`/`PSUBSCRIBE`/`PUBLISH`/`PUBSUB`), and `VXG.*` superpowers.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::dispatch_ds::{arg_str as s_arg, dispatch_ds_cmd, parse_f64, parse_i64, WRONGTYPE};
use crate::dispatch_vxg::{dispatch_vxg_cmd, handle_auth, handle_hello};
use crate::engine::KvEngine;
use crate::pubsub::PubSubBroker;
use crate::resp::RespVal;
use crate::types::{now_ms, Value};

/// Per-TCP-connection session state.
pub struct ConnState {
    pub sub_id: u64,
    pub authed: bool,
    pub resp3: bool,
    pub in_multi: bool,
    pub tx_queue: Vec<Vec<Vec<u8>>>,
    pub watched: HashMap<Vec<u8>, u64>,
    pub pinned_epoch: Option<u64>,
    pub quit: bool,
}

impl ConnState {
    pub fn new(sub_id: u64) -> Self {
        Self::new_with_auth(sub_id, true)
    }

    pub fn new_with_auth(sub_id: u64, authed: bool) -> Self {
        Self {
            sub_id,
            authed,
            resp3: false,
            in_multi: false,
            tx_queue: Vec::new(),
            watched: HashMap::new(),
            pinned_epoch: None,
            quit: false,
        }
    }
}

/// Dispatch one parsed command with an already-owned `&mut KvEngine` (used by zero-lock reactor).
pub fn dispatch_command_unlocked(
    eng: &mut KvEngine,
    broker: &Arc<PubSubBroker>,
    conn: &mut ConnState,
    args: Vec<Vec<u8>>,
) -> Vec<RespVal> {
    if args.is_empty() {
        return Vec::new();
    }
    let cmd = String::from_utf8_lossy(&args[0]).to_ascii_uppercase();
    if !conn.authed {
        return match cmd.as_str() {
            "AUTH" => vec![handle_auth(eng, conn, &args)],
            "HELLO" => vec![handle_hello(eng, conn, &args)],
            "QUIT" => {
                conn.quit = true;
                vec![RespVal::ok()]
            }
            _ => vec![RespVal::err("NOAUTH Authentication required.")],
        };
    }
    if let Some(res) = dispatch_pubsub_or_quit(broker, conn, &cmd, &args) {
        return res;
    }
    match cmd.as_str() {
        "MULTI" => {
            if conn.in_multi {
                return vec![RespVal::err("ERR MULTI calls can not be nested")];
            }
            conn.in_multi = true;
            conn.tx_queue.clear();
            return vec![RespVal::ok()];
        }
        "DISCARD" => {
            if !conn.in_multi {
                return vec![RespVal::err("ERR DISCARD without MULTI")];
            }
            conn.in_multi = false;
            conn.tx_queue.clear();
            conn.watched.clear();
            return vec![RespVal::ok()];
        }
        "WATCH" => {
            if conn.in_multi {
                return vec![RespVal::err("ERR WATCH inside MULTI is not allowed")];
            }
            for k in &args[1..] {
                let v = eng.watch_key(k);
                conn.watched.insert(k.clone(), v);
            }
            return vec![RespVal::ok()];
        }
        "UNWATCH" => {
            conn.watched.clear();
            return vec![RespVal::ok()];
        }
        "EXEC" => {
            if !conn.in_multi {
                return vec![RespVal::err("ERR EXEC without MULTI")];
            }
            conn.in_multi = false;
            let queued = std::mem::take(&mut conn.tx_queue);
            let watched = std::mem::take(&mut conn.watched);
            for (k, ver) in watched {
                if eng.key_version(&k) != ver {
                    return vec![RespVal::nil_array()];
                }
            }
            let mut results = Vec::with_capacity(queued.len());
            for qargs in queued {
                let qcmd = String::from_utf8_lossy(&qargs[0]).to_ascii_uppercase();
                results.push(exec_single(eng, conn, &qcmd, &qargs));
            }
            return vec![RespVal::array(results)];
        }
        _ => {}
    }
    if conn.in_multi {
        conn.tx_queue.push(args);
        return vec![RespVal::Simple("QUEUED".to_owned())];
    }
    vec![exec_single(eng, conn, &cmd, &args)]
}

/// Dispatch one parsed command under `RwLock` (used by portable thread-per-client fallback).
pub fn dispatch_command(
    engine: &Arc<RwLock<KvEngine>>,
    broker: &Arc<PubSubBroker>,
    conn: &mut ConnState,
    args: Vec<Vec<u8>>,
) -> Vec<RespVal> {
    if args.is_empty() {
        return Vec::new();
    }
    let cmd = String::from_utf8_lossy(&args[0]).to_ascii_uppercase();
    if conn.authed {
        if let Some(res) = dispatch_pubsub_or_quit(broker, conn, &cmd, &args) {
            return res;
        }
        if !conn.in_multi && (cmd == "BLPOP" || cmd == "BRPOP") {
            return vec![crate::blocking::blocking_pop_rwlock(
                engine,
                &args,
                cmd == "BLPOP",
            )];
        }
    }
    let mut eng = engine.write().unwrap();
    dispatch_command_unlocked(&mut eng, broker, conn, args)
}

fn dispatch_pubsub_or_quit(
    broker: &Arc<PubSubBroker>,
    conn: &mut ConnState,
    cmd: &str,
    args: &[Vec<u8>],
) -> Option<Vec<RespVal>> {
    Some(match cmd {
        "SUBSCRIBE" => {
            if args.len() < 2 {
                return Some(vec![RespVal::err("ERR wrong number of arguments for 'subscribe'")]);
            }
            args[1..]
                .iter()
                .map(|ch| {
                    let n = broker.subscribe(conn.sub_id, ch.clone()) as i64;
                    RespVal::array(vec![
                        RespVal::bulk(b"subscribe".to_vec()),
                        RespVal::bulk(ch.clone()),
                        RespVal::int(n),
                    ])
                })
                .collect()
        }
        "UNSUBSCRIBE" => {
            let items = if args.len() == 1 {
                broker.unsubscribe(conn.sub_id, None)
            } else {
                args[1..]
                    .iter()
                    .flat_map(|ch| broker.unsubscribe(conn.sub_id, Some(ch)))
                    .collect()
            };
            if items.is_empty() {
                return Some(vec![RespVal::array(vec![
                    RespVal::bulk(b"unsubscribe".to_vec()),
                    RespVal::nil(),
                    RespVal::int(broker.subscription_count(conn.sub_id) as i64),
                ])]);
            }
            items
                .into_iter()
                .map(|(ch, rem)| {
                    RespVal::array(vec![
                        RespVal::bulk(b"unsubscribe".to_vec()),
                        RespVal::bulk(ch),
                        RespVal::int(rem as i64),
                    ])
                })
                .collect()
        }
        "PSUBSCRIBE" => {
            if args.len() < 2 {
                return Some(vec![RespVal::err("ERR wrong number of arguments for 'psubscribe'")]);
            }
            args[1..]
                .iter()
                .map(|pat| {
                    let n = broker.psubscribe(conn.sub_id, pat.clone()) as i64;
                    RespVal::array(vec![
                        RespVal::bulk(b"psubscribe".to_vec()),
                        RespVal::bulk(pat.clone()),
                        RespVal::int(n),
                    ])
                })
                .collect()
        }
        "PUNSUBSCRIBE" => {
            let items = if args.len() == 1 {
                broker.punsubscribe(conn.sub_id, None)
            } else {
                args[1..]
                    .iter()
                    .flat_map(|pat| broker.punsubscribe(conn.sub_id, Some(pat)))
                    .collect()
            };
            if items.is_empty() {
                return Some(vec![RespVal::array(vec![
                    RespVal::bulk(b"punsubscribe".to_vec()),
                    RespVal::nil(),
                    RespVal::int(broker.subscription_count(conn.sub_id) as i64),
                ])]);
            }
            items
                .into_iter()
                .map(|(pat, rem)| {
                    RespVal::array(vec![
                        RespVal::bulk(b"punsubscribe".to_vec()),
                        RespVal::bulk(pat),
                        RespVal::int(rem as i64),
                    ])
                })
                .collect()
        }
        "PUBSUB" => vec![handle_pubsub_introspect(broker, args)],
        "QUIT" => {
            conn.quit = true;
            vec![RespVal::ok()]
        }
        _ => return None,
    })
}

fn handle_pubsub_introspect(broker: &PubSubBroker, args: &[Vec<u8>]) -> RespVal {
    let sub = s_arg(args, 1).unwrap_or("").to_ascii_uppercase();
    match sub.as_str() {
        "CHANNELS" => {
            let pat = args.get(2).map(Vec::as_slice);
            RespVal::array(
                broker
                    .active_channels(pat)
                    .into_iter()
                    .map(RespVal::bulk)
                    .collect(),
            )
        }
        "NUMSUB" => {
            let mut out = Vec::new();
            for (ch, cnt) in broker.numsub(&args[2..]) {
                out.push(RespVal::bulk(ch));
                out.push(RespVal::int(cnt as i64));
            }
            RespVal::array(out)
        }
        _ => RespVal::err("ERR unknown PUBSUB subcommand"),
    }
}

pub(crate) fn exec_single(
    eng: &mut KvEngine,
    conn: &mut ConnState,
    cmd: &str,
    args: &[Vec<u8>],
) -> RespVal {
    if let Some(resp) = dispatch_ds_cmd(eng, cmd, args) {
        return resp;
    }
    if let Some(resp) = dispatch_vxg_cmd(eng, conn, cmd, args) {
        return resp;
    }
    match cmd {
        // ---- CONNECTION / SERVER META ----
        "PING" => match args.get(1) {
            Some(m) => RespVal::bulk(m.clone()),
            None => RespVal::pong(),
        },
        "ECHO" => match args.get(1) {
            Some(m) => RespVal::bulk(m.clone()),
            None => RespVal::err("ERR wrong number of arguments for 'echo'"),
        },
        "CLIENT" => RespVal::ok(),
        "COMMAND" => RespVal::array(Vec::new()),
        "TIME" => {
            let ms = now_ms();
            RespVal::array(vec![
                RespVal::bulk((ms / 1000).to_string().into_bytes()),
                RespVal::bulk(((ms % 1000) * 1000).to_string().into_bytes()),
            ])
        }
        "DBSIZE" => match eng.dbsize_fast() {
            Ok(n) => RespVal::int(n as i64),
            Err(e) => RespVal::err(e),
        },
        "FLUSHDB" | "FLUSHALL" => match eng.flush_all() {
            Ok(_) => RespVal::ok(),
            Err(e) => RespVal::err(e),
        },

        // ---- PUBLISH (Local + Cross-Pod) ----
        "PUBLISH" => {
            if args.len() != 3 {
                return RespVal::err("ERR wrong number of arguments for 'publish'");
            }
            let local = eng.publish(args[1].clone(), args[2].clone()) as i64;
            RespVal::int(local)
        }

        // ---- STRING & KV ----
        "GET" => {
            if args.len() != 2 {
                return RespVal::err("ERR wrong number of arguments for 'get'");
            }
            let rec = match conn.pinned_epoch {
                Some(ep) => eng.get_record_at_epoch(ep, &args[1]),
                None => eng.get_record(&args[1]),
            };
            match rec {
                Ok(None) => RespVal::nil(),
                Ok(Some(r)) => match r.value {
                    Some(Value::Str(b)) => RespVal::bulk(b),
                    None => RespVal::nil(),
                    _ => RespVal::err(WRONGTYPE),
                },
                Err(e) => RespVal::err(e),
            }
        }
        "MGET" => {
            if args.len() < 2 {
                return RespVal::err("ERR wrong number of arguments for 'mget'");
            }
            let mut out = Vec::with_capacity(args.len() - 1);
            for k in &args[1..] {
                let rec = match conn.pinned_epoch {
                    Some(ep) => eng.get_record_at_epoch(ep, k),
                    None => eng.get_record(k),
                };
                match rec {
                    Ok(Some(r)) => match r.value {
                        Some(Value::Str(b)) => out.push(RespVal::bulk(b)),
                        _ => out.push(RespVal::nil()),
                    },
                    _ => out.push(RespVal::nil()),
                }
            }
            RespVal::array(out)
        }
        "MSET" | "MSETNX" => {
            if args.len() < 3 || !(args.len() - 1).is_multiple_of(2) {
                return RespVal::err("ERR wrong number of arguments for 'mset'");
            }
            if cmd == "MSETNX" {
                for pair in args[1..].as_chunks::<2>().0 {
                    if matches!(eng.get_record(&pair[0]), Ok(Some(_))) {
                        return RespVal::int(0);
                    }
                }
            }
            for pair in args[1..].as_chunks::<2>().0 {
                eng.set_record(pair[0].clone(), Value::Str(pair[1].clone()), None);
            }
            if cmd == "MSETNX" {
                RespVal::int(1)
            } else {
                RespVal::ok()
            }
        }
        "DEL" | "UNLINK" => {
            let mut n = 0i64;
            for k in &args[1..] {
                if matches!(eng.del_key(k), Ok(true)) {
                    n += 1;
                }
            }
            RespVal::int(n)
        }
        "EXISTS" => {
            let mut n = 0i64;
            for k in &args[1..] {
                if matches!(eng.get_record(k), Ok(Some(_))) {
                    n += 1;
                }
            }
            RespVal::int(n)
        }
        "GETDEL" => {
            if args.len() != 2 {
                return RespVal::err("ERR wrong number of arguments for 'getdel'");
            }
            match eng.get_record(&args[1]) {
                Ok(None) => RespVal::nil(),
                Ok(Some(r)) => match r.value {
                    Some(Value::Str(b)) => {
                        let _ = eng.del_key(&args[1]);
                        RespVal::bulk(b)
                    }
                    _ => RespVal::err(WRONGTYPE),
                },
                Err(e) => RespVal::err(e),
            }
        }
        "APPEND" => {
            if args.len() != 3 {
                return RespVal::err("ERR wrong number of arguments for 'append'");
            }
            let (mut cur, exp) = match eng.get_record(&args[1]) {
                Ok(None) => (Vec::new(), None),
                Ok(Some(r)) => {
                    let exp = r.expires_opt();
                    match r.value {
                        Some(Value::Str(b)) => (b, exp),
                        _ => return RespVal::err(WRONGTYPE),
                    }
                }
                Err(e) => return RespVal::err(e),
            };
            cur.extend_from_slice(&args[2]);
            let len = cur.len() as i64;
            eng.set_record(args[1].clone(), Value::Str(cur), exp);
            RespVal::int(len)
        }
        "STRLEN" => {
            if args.len() != 2 {
                return RespVal::err("ERR wrong number of arguments for 'strlen'");
            }
            match eng.get_record(&args[1]) {
                Ok(None) => RespVal::int(0),
                Ok(Some(r)) => match r.value {
                    Some(Value::Str(b)) => RespVal::int(b.len() as i64),
                    _ => RespVal::err(WRONGTYPE),
                },
                Err(e) => RespVal::err(e),
            }
        }
        "GETRANGE" => {
            if args.len() != 4 {
                return RespVal::err("ERR wrong number of arguments for 'getrange'");
            }
            let (Ok(s), Ok(e)) = (parse_i64(&args[2]), parse_i64(&args[3])) else {
                return RespVal::err("ERR value is not an integer");
            };
            let b = match eng.get_record(&args[1]) {
                Ok(None) => Vec::new(),
                Ok(Some(r)) => match r.value {
                    Some(Value::Str(v)) => v,
                    _ => return RespVal::err(WRONGTYPE),
                },
                Err(err) => return RespVal::err(err),
            };
            if b.is_empty() {
                return RespVal::bulk(Vec::new());
            }
            let l = b.len() as i64;
            let st = (if s < 0 { l + s } else { s }).max(0);
            let en = (if e < 0 { l + e } else { e }).min(l - 1);
            if st > en || st >= l {
                RespVal::bulk(Vec::new())
            } else {
                RespVal::bulk(b[st as usize..=en as usize].to_vec())
            }
        }
        "INCRBYFLOAT" => {
            if args.len() != 3 {
                return RespVal::err("ERR wrong number of arguments");
            }
            let delta = match parse_f64(&args[2]) {
                Ok(v) => v,
                Err(e) => return e,
            };
            let (cur, exp) = match eng.get_record(&args[1]) {
                Ok(None) => (0.0, None),
                Ok(Some(r)) => {
                    let exp = r.expires_opt();
                    match r.value {
                        Some(Value::Str(b)) => match parse_f64(&b) {
                            Ok(v) => (v, exp),
                            Err(e) => return e,
                        },
                        _ => return RespVal::err(WRONGTYPE),
                    }
                }
                Err(e) => return RespVal::err(e),
            };
            let next = format!("{}", cur + delta);
            eng.set_record(args[1].clone(), Value::Str(next.as_bytes().to_vec()), exp);
            RespVal::bulk(next.into_bytes())
        }

        // ---- TTL & KEY OPS ----
        "TTL" | "PTTL" => {
            if args.len() != 2 {
                return RespVal::err("ERR wrong number of arguments");
            }
            match eng.get_record(&args[1]) {
                Ok(None) => RespVal::int(-2),
                Ok(Some(r)) => match r.expires_opt() {
                    None => RespVal::int(-1),
                    Some(exp) => {
                        let rem = exp.saturating_sub(now_ms()) as i64;
                        RespVal::int(if cmd == "TTL" { rem / 1000 } else { rem })
                    }
                },
                Err(e) => RespVal::err(e),
            }
        }
        "PERSIST" => {
            if args.len() != 2 {
                return RespVal::err("ERR wrong number of arguments for 'persist'");
            }
            match eng.get_record(&args[1]) {
                Ok(Some(r)) if r.expires_at_ms > 0 => {
                    let _ = eng.set_expiry(&args[1], None);
                    RespVal::int(1)
                }
                _ => RespVal::int(0),
            }
        }
        "TYPE" => {
            if args.len() != 2 {
                return RespVal::err("ERR wrong number of arguments for 'type'");
            }
            match eng.get_record(&args[1]) {
                Ok(Some(r)) => match r.value {
                    Some(v) => RespVal::Simple(v.type_name().to_owned()),
                    None => RespVal::Simple("none".to_owned()),
                },
                _ => RespVal::Simple("none".to_owned()),
            }
        }
        "RENAME" | "RENAMENX" => {
            if args.len() != 3 {
                return RespVal::err("ERR wrong number of arguments");
            }
            let Some(rec) = (match eng.get_record(&args[1]) {
                Ok(v) => v,
                Err(e) => return RespVal::err(e),
            }) else {
                return RespVal::err("ERR no such key");
            };
            if cmd == "RENAMENX" && matches!(eng.get_record(&args[2]), Ok(Some(_))) {
                return RespVal::int(0);
            }
            let exp = rec.expires_opt();
            if let Some(val) = rec.value {
                let _ = eng.del_key(&args[1]);
                eng.set_record(args[2].clone(), val, exp);
            }
            if cmd == "RENAMENX" {
                RespVal::int(1)
            } else {
                RespVal::ok()
            }
        }
        "KEYS" => {
            let pat = args.get(1).map_or(b"*".as_slice(), Vec::as_slice);
            match eng.matching_keys(pat) {
                Ok(ks) => RespVal::array(ks.into_iter().map(RespVal::bulk).collect()),
                Err(e) => RespVal::err(e),
            }
        }

        _ => RespVal::err(format!("ERR unknown command '{cmd}'")),
    }
}
