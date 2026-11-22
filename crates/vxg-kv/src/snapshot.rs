//! VXG v5 sorted-block snapshot builder, single-block HTTP Range reader, WAL delta codec
//! (KV records + cross-pod Pub/Sub events), atomic `Manifest`, and `VXG.SQL` execution bridge.

use std::collections::BTreeMap;
use std::sync::Arc;

use vxg::{DecodeOptions, Frame, Options, RemoteFrame, SqlFormat, SqlQuery};

use crate::index::{BlockIndexEntry, DecodedBlock, SnapshotIndex};
use crate::pubsub::PubSubEvent;
use crate::store::{ObjectStore, StoreRangeReader};
use crate::types::{
    decode_record_line, encode_record_line, hex_decode, hex_encode, now_ms, FastMap, Record,
};

pub const MANIFEST_KEY: &str = "manifest.json";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SnapshotRef {
    pub epoch: u64,
    pub vxg_key: String,
    pub idx_key: String,
    pub raw_bytes: u64,
    pub compressed_bytes: u64,
    pub key_count: u64,
    pub created_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WalRef {
    pub epoch: u64,
    pub key: String,
    pub pod_id: String,
    pub compressed_bytes: u64,
    pub created_ms: u64,
}

/// Atomic source-of-truth catalog stored at `manifest.json` with ETag CAS.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Manifest {
    pub epoch: u64,
    pub base_snapshot: Option<SnapshotRef>,
    pub wals: Vec<WalRef>,
    pub history: Vec<SnapshotRef>,
}

impl Manifest {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut s = format!("{{\"epoch\":{}", self.epoch);
        s.push_str(",\"base\":");
        match &self.base_snapshot {
            Some(r) => push_snap_json(&mut s, r),
            None => s.push_str("null"),
        }
        s.push_str(",\"wals\":[");
        for (i, w) in self.wals.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&format!(
                "{{\"epoch\":{},\"key\":\"{}\",\"pod\":\"{}\",\"cbytes\":{},\"ts\":{}}}",
                w.epoch, w.key, w.pod_id, w.compressed_bytes, w.created_ms
            ));
        }
        s.push_str("],\"history\":[");
        for (i, h) in self.history.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            push_snap_json(&mut s, h);
        }
        s.push_str("]}\n");
        s.into_bytes()
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let s = std::str::from_utf8(bytes).map_err(|_| "non-UTF8 manifest".to_owned())?;
        let epoch = parse_u64_after(s, "\"epoch\":").ok_or("manifest missing epoch")?;
        let base_snapshot = parse_opt_snap(s, "\"base\":")?;
        let wals = parse_wal_array(s, "\"wals\":[")?;
        let history = parse_snap_array(s, "\"history\":[")?;
        Ok(Self {
            epoch,
            base_snapshot,
            wals,
            history,
        })
    }

    pub fn find_snapshot(&self, epoch: u64) -> Option<&SnapshotRef> {
        if let Some(ref b) = self.base_snapshot {
            if b.epoch == epoch {
                return Some(b);
            }
        }
        self.history.iter().find(|h| h.epoch == epoch)
    }
}

fn push_snap_json(s: &mut String, r: &SnapshotRef) {
    s.push_str(&format!(
        "{{\"epoch\":{},\"vxg\":\"{}\",\"idx\":\"{}\",\"raw\":{},\"comp\":{},\"keys\":{},\"ts\":{}}}",
        r.epoch, r.vxg_key, r.idx_key, r.raw_bytes, r.compressed_bytes, r.key_count, r.created_ms
    ));
}

fn parse_u64_after(s: &str, marker: &str) -> Option<u64> {
    let p = s.find(marker)? + marker.len();
    let tail = s[p..].trim_start();
    let end = tail
        .bytes()
        .take_while(|b| b.is_ascii_digit())
        .count();
    tail[..end].parse().ok()
}

fn parse_str_after(s: &str, marker: &str) -> Option<String> {
    let p = s.find(marker)? + marker.len();
    let tail = &s[p..];
    let end = tail.find('"')?;
    Some(tail[..end].to_owned())
}

fn parse_snap_obj(obj: &str) -> Result<SnapshotRef, String> {
    Ok(SnapshotRef {
        epoch: parse_u64_after(obj, "\"epoch\":").ok_or("snap missing epoch")?,
        vxg_key: parse_str_after(obj, "\"vxg\":\"").ok_or("snap missing vxg")?,
        idx_key: parse_str_after(obj, "\"idx\":\"").ok_or("snap missing idx")?,
        raw_bytes: parse_u64_after(obj, "\"raw\":").unwrap_or(0),
        compressed_bytes: parse_u64_after(obj, "\"comp\":").unwrap_or(0),
        key_count: parse_u64_after(obj, "\"keys\":").unwrap_or(0),
        created_ms: parse_u64_after(obj, "\"ts\":").unwrap_or(0),
    })
}

fn parse_opt_snap(s: &str, marker: &str) -> Result<Option<SnapshotRef>, String> {
    let p = s.find(marker).ok_or("manifest missing base")? + marker.len();
    let tail = s[p..].trim_start();
    if tail.starts_with("null") {
        return Ok(None);
    }
    let end = tail.find('}').ok_or("unterminated base snap")?;
    Ok(Some(parse_snap_obj(&tail[..=end])?))
}

fn extract_bracket_slice<'a>(s: &'a str, marker: &str) -> Result<&'a str, String> {
    let p = s.find(marker).ok_or_else(|| format!("missing {marker}"))? + marker.len();
    let tail = &s[p..];
    let end = tail.find(']').ok_or_else(|| format!("unterminated {marker}"))?;
    Ok(&tail[..end])
}

fn parse_wal_array(s: &str, marker: &str) -> Result<Vec<WalRef>, String> {
    let inner = extract_bracket_slice(s, marker)?;
    let mut out = Vec::new();
    let mut rest = inner;
    while let Some(st) = rest.find('{') {
        let en = rest[st..].find('}').ok_or("unterminated wal obj")? + st;
        let obj = &rest[st..=en];
        out.push(WalRef {
            epoch: parse_u64_after(obj, "\"epoch\":").ok_or("wal missing epoch")?,
            key: parse_str_after(obj, "\"key\":\"").ok_or("wal missing key")?,
            pod_id: parse_str_after(obj, "\"pod\":\"").unwrap_or_default(),
            compressed_bytes: parse_u64_after(obj, "\"cbytes\":").unwrap_or(0),
            created_ms: parse_u64_after(obj, "\"ts\":").unwrap_or(0),
        });
        rest = &rest[en + 1..];
    }
    Ok(out)
}

fn parse_snap_array(s: &str, marker: &str) -> Result<Vec<SnapshotRef>, String> {
    let inner = extract_bracket_slice(s, marker)?;
    let mut out = Vec::new();
    let mut rest = inner;
    while let Some(st) = rest.find('{') {
        let en = rest[st..].find('}').ok_or("unterminated history obj")? + st;
        out.push(parse_snap_obj(&rest[st..=en])?);
        rest = &rest[en + 1..];
    }
    Ok(out)
}

/// Build and upload a sorted VXG v5 snapshot (`snap_<epoch>.vxg`) + `VXI1` index (`snap_<epoch>.idx`)
/// from an already key-sorted slice of `(key, record)` pairs (zero JSON re-decode during indexing!).
pub fn write_snapshot_sorted(
    store: &Arc<dyn ObjectStore>,
    epoch: u64,
    sorted: &[(&[u8], &Record)],
    block_size: usize,
) -> Result<(SnapshotRef, SnapshotIndex), String> {
    let now = now_ms();
    let mut raw = Vec::with_capacity(sorted.len() * 64);
    let mut line_keys: Vec<(usize, &[u8])> = Vec::with_capacity(sorted.len());
    for &(k, rec) in sorted {
        if rec.is_live(now) {
            line_keys.push((raw.len(), k));
            encode_record_line(&mut raw, k, rec);
        }
    }
    let live_count = line_keys.len() as u64;
    let enc_opts = Options {
        block_size: block_size.clamp(4096, 262_144),
        record_aligned: true,
        row_index: true,
        checksum: true,
        ..Options::default()
    };
    let compressed = vxg::compress(&raw, &enc_opts).map_err(|e| format!("vxg compress: {e}"))?;
    let frame = Frame::parse(&compressed, usize::MAX).map_err(|e| format!("frame parse: {e}"))?;

    let mut idx_blocks = Vec::with_capacity(frame.block_count());
    let mut cursor = 0usize;
    let mut key_refs: Vec<&[u8]> = Vec::new();
    for (bi, binfo) in frame.blocks().enumerate() {
        let end_off = binfo.raw_offset + binfo.raw_size;
        key_refs.clear();
        while cursor < line_keys.len() && line_keys[cursor].0 < end_off {
            key_refs.push(line_keys[cursor].1);
            cursor += 1;
        }
        idx_blocks.push(BlockIndexEntry::build(bi as u32, &key_refs));
    }
    let index = SnapshotIndex {
        total_keys: live_count,
        blocks: idx_blocks,
    };
    let vxg_key = format!("snapshots/snap_{epoch:08}.vxg");
    let idx_key = format!("snapshots/snap_{epoch:08}.idx");
    store.put(&vxg_key, &compressed)?;
    store.put(&idx_key, &index.encode())?;

    let snap_ref = SnapshotRef {
        epoch,
        vxg_key,
        idx_key,
        raw_bytes: raw.len() as u64,
        compressed_bytes: compressed.len() as u64,
        key_count: live_count,
        created_ms: now,
    };
    Ok((snap_ref, index))
}

pub fn write_snapshot(
    store: &Arc<dyn ObjectStore>,
    epoch: u64,
    records: &BTreeMap<Vec<u8>, Record>,
    block_size: usize,
) -> Result<(SnapshotRef, SnapshotIndex), String> {
    let sorted: Vec<(&[u8], &Record)> = records.iter().map(|(k, r)| (k.as_slice(), r)).collect();
    write_snapshot_sorted(store, epoch, &sorted, block_size)
}

pub fn write_snapshot_fastmap(
    store: &Arc<dyn ObjectStore>,
    epoch: u64,
    records: &FastMap<Vec<u8>, Record>,
    block_size: usize,
) -> Result<(SnapshotRef, SnapshotIndex), String> {
    let mut sorted: Vec<(&[u8], &Record)> = records.iter().map(|(k, r)| (k.as_slice(), r)).collect();
    sorted.sort_unstable_by_key(|(k, _)| *k);
    write_snapshot_sorted(store, epoch, &sorted, block_size)
}

/// Fetch strictly ONE block from a remote `.vxg` snapshot via `vxg::RemoteFrame` Range I/O.
pub fn fetch_snapshot_block(
    store: &Arc<dyn ObjectStore>,
    snap: &SnapshotRef,
    block_idx: usize,
) -> Result<(DecodedBlock, usize, usize, u64), String> {
    let reader = StoreRangeReader::open(Arc::clone(store), &snap.vxg_key, snap.compressed_bytes);
    let mut rf = RemoteFrame::open(reader).map_err(|e| format!("remote open: {e}"))?;
    let raw = rf
        .extract_block(block_idx, &DecodeOptions::default())
        .map_err(|e| format!("remote block {block_idx}: {e}"))?;
    let stats = rf.stats();
    let approx = raw.len();
    let mut map = BTreeMap::new();
    for line in raw.split(|&b| b == b'\n') {
        if line.is_empty() {
            continue;
        }
        if let Some((k, rec)) = decode_record_line(line) {
            map.insert(k, rec);
        }
    }
    Ok((map, approx, stats.requests, stats.bytes_fetched))
}

/// Fetch & decode ALL records of a snapshot (used during compaction & full scans).
pub fn read_full_snapshot(
    store: &Arc<dyn ObjectStore>,
    snap: &SnapshotRef,
) -> Result<BTreeMap<Vec<u8>, Record>, String> {
    let Some((bytes, _)) = store.get(&snap.vxg_key)? else {
        return Err(format!("snapshot missing: {}", snap.vxg_key));
    };
    let raw =
        vxg::decompress(&bytes, &DecodeOptions::default()).map_err(|e| format!("decompress: {e}"))?;
    let mut map = BTreeMap::new();
    for line in raw.split(|&b| b == b'\n') {
        if line.is_empty() {
            continue;
        }
        if let Some((k, rec)) = decode_record_line(line) {
            map.insert(k, rec);
        }
    }
    Ok(map)
}

/// Encode KV delta mutations + cross-pod Pub/Sub events into a compressed `.vxg` WAL segment.
pub fn encode_wal_segment(
    mutations: &FastMap<Vec<u8>, Record>,
    events: &[PubSubEvent],
) -> Result<Vec<u8>, String> {
    let mut raw = Vec::with_capacity(mutations.len() * 64 + events.len() * 64);
    for (k, rec) in mutations {
        encode_record_line(&mut raw, k, rec);
    }
    for ev in events {
        raw.extend_from_slice(
            format!(
                "{{\"_ps\":1,\"pod\":\"{}\",\"ch\":\"{}\",\"msg\":\"{}\"}}\n",
                ev.origin_pod,
                hex_encode(&ev.channel),
                hex_encode(&ev.payload)
            )
            .as_bytes(),
        );
    }
    let opts = Options {
        block_size: 16_384,
        record_aligned: true,
        row_index: true,
        checksum: true,
        ..Options::default()
    };
    vxg::compress(&raw, &opts).map_err(|e| format!("wal compress: {e}"))
}

pub type DecodedWal = (Vec<(Vec<u8>, Record)>, Vec<PubSubEvent>);

/// Decode a `.vxg` WAL segment back into KV records + cross-pod Pub/Sub events.
pub fn decode_wal_segment(bytes: &[u8]) -> Result<DecodedWal, String> {
    let raw =
        vxg::decompress(bytes, &DecodeOptions::default()).map_err(|e| format!("wal decode: {e}"))?;
    let mut recs = Vec::new();
    let mut events = Vec::new();
    for line in raw.split(|&b| b == b'\n') {
        if line.is_empty() {
            continue;
        }
        if line.starts_with(b"{\"_ps\":1,") {
            if let Ok(s) = std::str::from_utf8(line) {
                if let (Some(pod), Some(ch_h), Some(msg_h)) = (
                    parse_str_after(s, "\"pod\":\""),
                    parse_str_after(s, "\"ch\":\""),
                    parse_str_after(s, "\"msg\":\""),
                ) {
                    if let (Some(channel), Some(payload)) =
                        (hex_decode(ch_h.as_bytes()), hex_decode(msg_h.as_bytes()))
                    {
                        events.push(PubSubEvent {
                            origin_pod: pod,
                            channel,
                            payload,
                        });
                    }
                }
            }
        } else if let Some(pair) = decode_record_line(line) {
            recs.push(pair);
        }
    }
    Ok((recs, events))
}

/// Execute a compressed-domain `VXG.SQL` query over an immutable snapshot frame or live state.
pub fn execute_vxg_sql_on_frame(
    frame_bytes: &[u8],
    sql: &str,
    format: SqlFormat,
) -> Result<Vec<u8>, String> {
    let q = SqlQuery::parse(sql).map_err(|e| format!("ERR SQL parse: {e}"))?;
    vxg::execute_sql(frame_bytes, &q, format, false, &DecodeOptions::default())
        .map_err(|e| format!("ERR SQL exec: {e}"))
}
