//! Local + Cross-Pod Redis Pub/Sub broker (`SUBSCRIBE`, `PSUBSCRIBE`, `PUBLISH`, `PUBSUB`).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::mpsc::Sender;
use std::sync::Mutex;

use crate::types::glob_match;

/// A single message pushed to a subscribed client connection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PushMessage {
    Message {
        channel: Vec<u8>,
        payload: Vec<u8>,
    },
    PMessage {
        pattern: Vec<u8>,
        channel: Vec<u8>,
        payload: Vec<u8>,
    },
}

/// Cross-pod event carried inside a VXG WAL segment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PubSubEvent {
    pub origin_pod: String,
    pub channel: Vec<u8>,
    pub payload: Vec<u8>,
}

struct Subscriber {
    tx: Sender<PushMessage>,
    channels: BTreeSet<Vec<u8>>,
    patterns: BTreeSet<Vec<u8>>,
}

/// Thread-safe Pub/Sub registry shared by all client connections in a pod.
pub struct PubSubBroker {
    inner: Mutex<BrokerInner>,
}

struct BrokerInner {
    next_id: u64,
    subs: BTreeMap<u64, Subscriber>,
}

impl Default for PubSubBroker {
    fn default() -> Self {
        Self::new()
    }
}

impl PubSubBroker {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(BrokerInner {
                next_id: 1,
                subs: BTreeMap::new(),
            }),
        }
    }

    pub fn register(&self, tx: Sender<PushMessage>) -> u64 {
        let mut g = self.inner.lock().unwrap();
        let id = g.next_id;
        g.next_id += 1;
        g.subs.insert(
            id,
            Subscriber {
                tx,
                channels: BTreeSet::new(),
                patterns: BTreeSet::new(),
            },
        );
        id
    }

    pub fn unregister(&self, id: u64) {
        self.inner.lock().unwrap().subs.remove(&id);
    }

    pub fn subscribe(&self, id: u64, channel: Vec<u8>) -> usize {
        let mut g = self.inner.lock().unwrap();
        if let Some(s) = g.subs.get_mut(&id) {
            s.channels.insert(channel);
            s.channels.len() + s.patterns.len()
        } else {
            0
        }
    }

    pub fn unsubscribe(&self, id: u64, channel: Option<&[u8]>) -> Vec<(Vec<u8>, usize)> {
        let mut g = self.inner.lock().unwrap();
        let Some(s) = g.subs.get_mut(&id) else {
            return Vec::new();
        };
        let removed: Vec<Vec<u8>> = match channel {
            Some(ch) => {
                s.channels.remove(ch);
                vec![ch.to_vec()]
            }
            None => {
                let all: Vec<_> = s.channels.iter().cloned().collect();
                s.channels.clear();
                all
            }
        };
        let mut out = Vec::with_capacity(removed.len());
        let mut rem = s.channels.len() + s.patterns.len() + removed.len();
        for ch in removed {
            rem = rem.saturating_sub(1);
            out.push((ch, rem));
        }
        out
    }

    pub fn psubscribe(&self, id: u64, pattern: Vec<u8>) -> usize {
        let mut g = self.inner.lock().unwrap();
        if let Some(s) = g.subs.get_mut(&id) {
            s.patterns.insert(pattern);
            s.channels.len() + s.patterns.len()
        } else {
            0
        }
    }

    pub fn punsubscribe(&self, id: u64, pattern: Option<&[u8]>) -> Vec<(Vec<u8>, usize)> {
        let mut g = self.inner.lock().unwrap();
        let Some(s) = g.subs.get_mut(&id) else {
            return Vec::new();
        };
        let removed: Vec<Vec<u8>> = match pattern {
            Some(pat) => {
                s.patterns.remove(pat);
                vec![pat.to_vec()]
            }
            None => {
                let all: Vec<_> = s.patterns.iter().cloned().collect();
                s.patterns.clear();
                all
            }
        };
        let mut out = Vec::with_capacity(removed.len());
        let mut rem = s.channels.len() + s.patterns.len() + removed.len();
        for pat in removed {
            rem = rem.saturating_sub(1);
            out.push((pat, rem));
        }
        out
    }

    pub fn subscription_count(&self, id: u64) -> usize {
        self.inner
            .lock()
            .unwrap()
            .subs
            .get(&id)
            .map_or(0, |s| s.channels.len() + s.patterns.len())
    }

    /// Deliver `(channel, payload)` to all matching local subscribers; returns recipient count.
    pub fn publish_local(&self, channel: &[u8], payload: &[u8]) -> usize {
        let mut g = self.inner.lock().unwrap();
        let mut delivered = 0usize;
        let mut dead = Vec::new();
        for (&id, sub) in &g.subs {
            let mut failed = false;
            if sub.channels.contains(channel) {
                if sub
                    .tx
                    .send(PushMessage::Message {
                        channel: channel.to_vec(),
                        payload: payload.to_vec(),
                    })
                    .is_ok()
                {
                    delivered += 1;
                } else {
                    failed = true;
                }
            }
            for pat in &sub.patterns {
                if glob_match(pat, channel) {
                    if sub
                        .tx
                        .send(PushMessage::PMessage {
                            pattern: pat.clone(),
                            channel: channel.to_vec(),
                            payload: payload.to_vec(),
                        })
                        .is_ok()
                    {
                        delivered += 1;
                    } else {
                        failed = true;
                    }
                }
            }
            if failed {
                dead.push(id);
            }
        }
        for id in dead {
            g.subs.remove(&id);
        }
        delivered
    }

    pub fn active_channels(&self, pat: Option<&[u8]>) -> Vec<Vec<u8>> {
        let g = self.inner.lock().unwrap();
        let mut set = BTreeSet::new();
        for sub in g.subs.values() {
            for ch in &sub.channels {
                if pat.is_none_or(|p| glob_match(p, ch)) {
                    set.insert(ch.clone());
                }
            }
        }
        set.into_iter().collect()
    }

    pub fn numsub(&self, channels: &[Vec<u8>]) -> Vec<(Vec<u8>, usize)> {
        let g = self.inner.lock().unwrap();
        channels
            .iter()
            .map(|ch| {
                let c = g.subs.values().filter(|s| s.channels.contains(ch)).count();
                (ch.clone(), c)
            })
            .collect()
    }
}
