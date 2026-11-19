//! Streaming `.vxt` (`VXT1`) archive writer for low-RAM packing of large (`27B`/`70B+`) models.
//! Produces 100% bit-for-bit identical `.vxt` files to [`crate::format::write_vxt_model`] while
//! flushing each encoded layer/tensor batch directly to disk and immediately dropping RAM.

use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::format::{serialize_vxt_entry, VxtEntry};
use crate::format_io::{fnv1a64_update, push_u8_slice, FNV1A64_INIT};
use crate::meta_lz::compress_meta_blob;
use crate::turbo_linalg::effective_cpu_threads;
use crate::{Error, Result};

const MAGIC: &[u8; 4] = b"VXT1";
const VERSION: u32 = 1;
const ZERO_PAD64: [u8; 64] = [0u8; 64];

/// Streaming `VXT1` writer that maintains a running `FNV-1a 64` archive checksum and flushes
/// compressed tensor sections straight to `W` without retaining the whole archive in memory.
pub struct VxtStreamWriter<W: Write> {
    out: BufWriter<W>,
    crc: u64,
    bytes_written: u64,
    expected_entries: u32,
    written_entries: u32,
    threads: usize,
}

impl<W: Write> VxtStreamWriter<W> {
    /// Creates a streaming writer and immediately writes the `VXT1` header and compressed metadata KV table.
    pub fn new(
        writer: W,
        expected_entries: usize,
        meta_kv: &[(String, Vec<u8>)],
        threads: usize,
    ) -> Result<Self> {
        let count = u32::try_from(expected_entries)
            .map_err(|_| Error::Format("too many tensors in VXT archive".into()))?;
        let mcount = u32::try_from(meta_kv.len())
            .map_err(|_| Error::Format("too many metadata entries in VXT archive".into()))?;

        let mut hdr = Vec::with_capacity(64 + meta_kv.len() * 64);
        hdr.extend_from_slice(MAGIC);
        hdr.extend_from_slice(&VERSION.to_le_bytes());
        hdr.extend_from_slice(&count.to_le_bytes());
        hdr.extend_from_slice(&mcount.to_le_bytes());
        for (k, v) in meta_kv {
            let kb = k.as_bytes();
            hdr.extend_from_slice(&(kb.len() as u32).to_le_bytes());
            hdr.extend_from_slice(kb);
            let packed_v = compress_meta_blob(v);
            push_u8_slice(&mut hdr, &packed_v);
        }

        let mut sw = Self {
            out: BufWriter::with_capacity(4 * 1024 * 1024, writer),
            crc: FNV1A64_INIT,
            bytes_written: 0,
            expected_entries: count,
            written_entries: 0,
            threads: effective_cpu_threads(threads),
        };
        sw.write_raw(&hdr)?;
        Ok(sw)
    }

    #[inline]
    fn write_raw(&mut self, data: &[u8]) -> Result<()> {
        self.out
            .write_all(data)
            .map_err(|e| Error::Io(format!("VXT stream write failed: {e}")))?;
        self.crc = fnv1a64_update(self.crc, data);
        self.bytes_written += data.len() as u64;
        Ok(())
    }

    #[inline]
    fn pad64(&mut self) -> Result<()> {
        let rem = (self.bytes_written % 64) as usize;
        if rem != 0 {
            self.write_raw(&ZERO_PAD64[..64 - rem])?;
        }
        Ok(())
    }

    /// Serializes a single [`VxtEntry`] (entropy-coded + per-section CRC) and streams it immediately.
    pub fn write_entry(&mut self, entry: VxtEntry) -> Result<()> {
        let sec = serialize_vxt_entry(&entry);
        drop(entry);
        self.write_section(&sec)
    }

    /// Compresses a batch of [`VxtEntry`] tensors in parallel across CPU workers, streams each
    /// section straight to disk, and immediately drops the batch from memory.
    pub fn write_batch_parallel(&mut self, entries: Vec<VxtEntry>) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }
        if entries.len() == 1 || self.threads <= 1 {
            for e in entries {
                self.write_entry(e)?;
            }
            return Ok(());
        }
        let n = entries.len();
        let workers = self.threads.min(n);
        let next = AtomicUsize::new(0);
        let slots: Vec<Mutex<Vec<u8>>> = (0..n).map(|_| Mutex::new(Vec::new())).collect();
        std::thread::scope(|s| {
            for _ in 0..workers {
                let (n_ref, s_ref, e_ref) = (&next, &slots, &entries);
                s.spawn(move || loop {
                    let idx = n_ref.fetch_add(1, Ordering::Relaxed);
                    if idx >= n {
                        break;
                    }
                    let sec = serialize_vxt_entry(&e_ref[idx]);
                    *s_ref[idx].lock().unwrap() = sec;
                });
            }
        });
        drop(entries);
        for slot in slots {
            let sec = slot.into_inner().unwrap();
            self.write_section(&sec)?;
        }
        Ok(())
    }

    fn write_section(&mut self, sec: &[u8]) -> Result<()> {
        if self.written_entries >= self.expected_entries {
            return Err(Error::Format("VxtStreamWriter received more entries than declared".into()));
        }
        self.pad64()?;
        self.write_raw(sec)?;
        self.written_entries += 1;
        Ok(())
    }

    /// Writes the trailing 8-byte archive `FNV-1a 64` checksum, flushes the buffer, and returns total bytes.
    pub fn finish(mut self) -> Result<u64> {
        if self.written_entries != self.expected_entries {
            return Err(Error::Format(format!(
                "VxtStreamWriter entry count mismatch: wrote {} of {}",
                self.written_entries, self.expected_entries
            )));
        }
        let crc_bytes = self.crc.to_le_bytes();
        self.out
            .write_all(&crc_bytes)
            .map_err(|e| Error::Io(format!("VXT stream checksum write failed: {e}")))?;
        self.out
            .flush()
            .map_err(|e| Error::Io(format!("VXT stream flush failed: {e}")))?;
        Ok(self.bytes_written + 8)
    }
}

#[cfg(test)]
mod tests {
    use super::VxtStreamWriter;
    use crate::format::{read_vxt_model, write_vxt_model, PackedTensor, VxtEntry};

    #[test]
    fn stream_writer_is_bit_exact_with_in_memory_writer() {
        let meta = vec![
            ("model.config".to_owned(), b"{\"arch\":\"qwen35\"}".to_vec()),
            ("model.pack_mode".to_owned(), b"turbo-hq".to_vec()),
        ];
        let entries = vec![
            VxtEntry {
                name: "blk.0.attn_norm.weight".into(),
                tensor: PackedTensor::RawF32 {
                    rows: 1,
                    cols: 64,
                    data: (0..64).map(|i| (i as f32) * 0.01).collect(),
                },
            },
            VxtEntry {
                name: "blk.0.ffn_down.weight".into(),
                tensor: PackedTensor::RawF32 {
                    rows: 16,
                    cols: 32,
                    data: (0..512).map(|i| ((i % 19) as f32 - 9.0) * 0.05).collect(),
                },
            },
            VxtEntry {
                name: "output_norm.weight".into(),
                tensor: PackedTensor::RawF32 {
                    rows: 1,
                    cols: 64,
                    data: vec![1.0; 64],
                },
            },
        ];

        let expected = write_vxt_model(&meta, &entries).unwrap();
        let mut streamed = Vec::new();
        {
            let mut sw = VxtStreamWriter::new(&mut streamed, entries.len(), &meta, 2).unwrap();
            sw.write_entry(entries[0].clone()).unwrap();
            sw.write_batch_parallel(entries[1..].to_vec()).unwrap();
            let total = sw.finish().unwrap();
            assert_eq!(total as usize, streamed.len());
        }
        assert_eq!(streamed, expected, "streamed VXT archive must be 100% bit-exact");
        let (kv_dec, ent_dec) = read_vxt_model(&streamed).unwrap();
        assert_eq!(kv_dec.len(), 2);
        assert_eq!(ent_dec.len(), 3);
    }
}
