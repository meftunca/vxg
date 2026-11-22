//! Non-blocking `BLPOP` / `BRPOP` waiter queue for the single-threaded `kqueue`/`epoll` reactor.
//!
//! Clients blocked on empty lists consume **zero threads** and incur **zero overhead** when the
//! waiter queue is empty (1 predicted-false check in the reactor loop).

use std::time::Duration;

use crate::engine::KvEngine;
use crate::resp::RespVal;
use crate::types::Value;

pub type PopResult = Result<Option<(Vec<u8>, Vec<u8>)>, String>;

/// Pop one element from the first non-empty list in `keys` (`left = true` => `LPOP`, else `RPOP`).
pub fn try_pop_any(eng: &mut KvEngine, keys: &[Vec<u8>], left: bool) -> PopResult {
    for key in keys {
        let popped = {
            let Some(rec) = eng.get_or_promote_mut(key)? else {
                continue;
            };
            let Some(Value::List(l)) = rec.value.as_mut() else {
                return Err(crate::dispatch_ds::WRONGTYPE.to_owned());
            };
            let item = if left { l.pop_front() } else { l.pop_back() };
            (item, l.is_empty())
        };
        if let (Some(val), empty) = popped {
            if empty {
                let _ = eng.del_key(key);
            }
            return Ok(Some((key.clone(), val)));
        }
    }
    Ok(None)
}

/// Parse `BLPOP key [key ...] timeout` / `BRPOP key [key ...] timeout`.
/// Returns `(keys, timeout_opt)` where `timeout == 0` means wait indefinitely (`None`).
pub fn parse_blocking_pop(args: &[Vec<u8>]) -> Result<(Vec<Vec<u8>>, Option<Duration>), RespVal> {
    if args.len() < 3 {
        return Err(RespVal::err("ERR wrong number of arguments for blocking pop"));
    }
    let timeout_str = std::str::from_utf8(args.last().unwrap())
        .map_err(|_| RespVal::err("ERR timeout is not a float or out of range"))?;
    let secs: f64 = timeout_str
        .parse()
        .map_err(|_| RespVal::err("ERR timeout is not a float or out of range"))?;
    if !secs.is_finite() || secs < 0.0 {
        return Err(RespVal::err("ERR timeout is negative or invalid"));
    }
    let dur = if secs == 0.0 {
        None
    } else {
        Some(Duration::from_secs_f64(secs))
    };
    let keys = args[1..args.len() - 1].to_vec();
    Ok((keys, dur))
}

#[cfg(any(target_os = "macos", target_os = "linux"))]
pub mod reactor_queue {
    use std::os::fd::RawFd;
    use std::time::{Duration, Instant};

    use super::{parse_blocking_pop, try_pop_any};
    use crate::dispatch::{dispatch_command_unlocked, ConnState};
    use crate::engine::KvEngine;
    use crate::pubsub::PubSubBroker;
    use crate::resp::RespVal;
    use std::sync::Arc;

    pub struct BlockedWaiter {
        pub fd: RawFd,
        pub keys: Vec<Vec<u8>>,
        pub left: bool,
        pub deadline: Option<Instant>,
    }

    #[derive(Default)]
    pub struct WaiterQueue {
        waiters: Vec<BlockedWaiter>,
    }

    impl WaiterQueue {
        #[inline]
        pub fn is_empty(&self) -> bool {
            self.waiters.is_empty()
        }

        pub fn register(
            &mut self,
            fd: RawFd,
            keys: Vec<Vec<u8>>,
            left: bool,
            timeout: Option<Duration>,
        ) {
            self.waiters.retain(|w| w.fd != fd);
            self.waiters.push(BlockedWaiter {
                fd,
                keys,
                left,
                deadline: timeout.map(|d| Instant::now() + d),
            });
        }

        pub fn remove_fd(&mut self, fd: RawFd) {
            if !self.waiters.is_empty() {
                self.waiters.retain(|w| w.fd != fd);
            }
        }

        /// Intercept `BLPOP`/`BRPOP` on an authenticated non-MULTI socket. Returns `true` if handled.
        pub fn try_handle_blocking(
            &mut self,
            fd: RawFd,
            eng: &mut KvEngine,
            conn: &ConnState,
            args: &[Vec<u8>],
            out: &mut Vec<u8>,
        ) -> bool {
            if !conn.authed || conn.in_multi {
                return false;
            }
            let Some(first) = args.first() else {
                return false;
            };
            let left = if first.eq_ignore_ascii_case(b"BLPOP") {
                true
            } else if first.eq_ignore_ascii_case(b"BRPOP") {
                false
            } else {
                return false;
            };
            let (keys, timeout) = match parse_blocking_pop(args) {
                Ok(v) => v,
                Err(e) => {
                    e.encode(out);
                    return true;
                }
            };
            match try_pop_any(eng, &keys, left) {
                Ok(Some((k, v))) => {
                    RespVal::array(vec![RespVal::bulk(k), RespVal::bulk(v)]).encode(out);
                }
                Ok(None) => self.register(fd, keys, left, timeout),
                Err(e) => RespVal::err(e).encode(out),
            }
            true
        }

        /// Dispatch a parsed span batch that contains at least one `BLPOP`/`BRPOP` command.
        #[allow(clippy::too_many_arguments)]
        pub fn dispatch_blocking_batch(
            &mut self,
            fd: RawFd,
            eng: &mut KvEngine,
            broker: &Arc<PubSubBroker>,
            conn: &mut ConnState,
            raw: &[u8],
            arg_spans: &[(usize, usize)],
            cmd_ranges: &[(usize, usize)],
            out: &mut Vec<u8>,
        ) {
            for &(a, b) in cmd_ranges {
                let owned: Vec<Vec<u8>> = arg_spans[a..b]
                    .iter()
                    .map(|&(s, e)| raw[s..e].to_vec())
                    .collect();
                if !self.try_handle_blocking(fd, eng, conn, &owned, out) {
                    for r in dispatch_command_unlocked(eng, broker, conn, owned) {
                        r.encode(out);
                    }
                }
                if conn.quit {
                    break;
                }
            }
        }

        /// Check all registered waiters in FIFO order: wake any whose list now has data,
        /// and expire any whose deadline has elapsed. Returns `(fd, encoded_resp_reply)` pairs.
        pub fn poll_ready(&mut self, eng: &mut KvEngine) -> Vec<(RawFd, RespVal)> {
            if self.waiters.is_empty() {
                return Vec::new();
            }
            let now = Instant::now();
            let mut ready = Vec::new();
            let mut remaining = Vec::with_capacity(self.waiters.len());
            for w in self.waiters.drain(..) {
                match try_pop_any(eng, &w.keys, w.left) {
                    Ok(Some((k, v))) => {
                        ready.push((
                            w.fd,
                            RespVal::array(vec![RespVal::bulk(k), RespVal::bulk(v)]),
                        ));
                    }
                    Err(e) => {
                        ready.push((w.fd, RespVal::err(e)));
                    }
                    Ok(None) => {
                        if w.deadline.is_some_and(|dl| now >= dl) {
                            ready.push((w.fd, RespVal::nil_array()));
                        } else {
                            remaining.push(w);
                        }
                    }
                }
            }
            self.waiters = remaining;
            ready
        }
    }
}
