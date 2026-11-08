//! Low-level binary serialization cursor and slice helpers for `.vxt` (`VXT1`) archives.

use crate::{Error, Result};

#[inline]
pub(crate) fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

#[inline]
pub(crate) fn pad64(buf: &mut Vec<u8>) {
    while buf.len() % 64 != 0 {
        buf.push(0);
    }
}

pub(crate) fn push_u16_slice(buf: &mut Vec<u8>, s: &[u16]) {
    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
    buf.reserve(s.len() * 2);
    for &v in s {
        buf.extend_from_slice(&v.to_le_bytes());
    }
}

pub(crate) fn push_u32_slice(buf: &mut Vec<u8>, s: &[u32]) {
    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
    buf.reserve(s.len() * 4);
    for &v in s {
        buf.extend_from_slice(&v.to_le_bytes());
    }
}

pub(crate) fn push_f32_slice(buf: &mut Vec<u8>, s: &[f32]) {
    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
    buf.reserve(s.len() * 4);
    for &v in s {
        buf.extend_from_slice(&v.to_le_bytes());
    }
}

pub(crate) fn push_u8_slice(buf: &mut Vec<u8>, s: &[u8]) {
    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
    buf.extend_from_slice(s);
}

pub(crate) fn push_hdr(
    out: &mut Vec<u8>,
    kind: u8,
    sub: u8,
    rows: usize,
    cols: usize,
    pcols: usize,
) {
    out.extend_from_slice(&[kind, sub, 0, 0, 0, 0, 0, 0]);
    out.extend_from_slice(&(rows as u64).to_le_bytes());
    out.extend_from_slice(&(cols as u64).to_le_bytes());
    out.extend_from_slice(&(pcols as u64).to_le_bytes());
}

pub(crate) struct Cur<'a> {
    pub b: &'a [u8],
    pub pos: usize,
}

impl<'a> Cur<'a> {
    pub fn align64(&mut self) -> Result<()> {
        let rem = self.pos % 64;
        if rem != 0 {
            let _ = self.take(64 - rem)?;
        }
        Ok(())
    }

    pub fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| Error::Format("VXT cursor overflow".into()))?;
        if end > self.b.len() {
            return Err(Error::Format("truncated VXT archive".into()));
        }
        let s = &self.b[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    pub fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    pub fn u32(&mut self) -> Result<u32> {
        let s = self.take(4)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }

    pub fn u64(&mut self) -> Result<u64> {
        let s = self.take(8)?;
        Ok(u64::from_le_bytes(s.try_into().unwrap()))
    }

    pub fn usize64(&mut self) -> Result<usize> {
        usize::try_from(self.u64()?).map_err(|_| Error::Format("VXT length overflow".into()))
    }

    pub fn f32_16(&mut self) -> Result<[f32; 16]> {
        let mut out = [0.0f32; 16];
        for slot in &mut out {
            let s = self.take(4)?;
            *slot = f32::from_le_bytes([s[0], s[1], s[2], s[3]]);
        }
        Ok(out)
    }

    pub fn u8_vec(&mut self) -> Result<Vec<u8>> {
        let n = self.usize64()?;
        Ok(self.take(n)?.to_vec())
    }

    pub fn u16_vec(&mut self) -> Result<Vec<u16>> {
        let n = self.usize64()?;
        let bytes = self.take(
            n.checked_mul(2)
                .ok_or_else(|| Error::Format("u16 overflow".into()))?,
        )?;
        #[cfg(target_endian = "little")]
        {
            let mut out = vec![0u16; n];
            if !bytes.is_empty() {
                unsafe {
                    std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr() as *mut u8, bytes.len());
                }
            }
            Ok(out)
        }
        #[cfg(not(target_endian = "little"))]
        {
            Ok(bytes
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect())
        }
    }

    pub fn u32_vec(&mut self) -> Result<Vec<u32>> {
        let n = self.usize64()?;
        let bytes = self.take(
            n.checked_mul(4)
                .ok_or_else(|| Error::Format("u32 overflow".into()))?,
        )?;
        #[cfg(target_endian = "little")]
        {
            let mut out = vec![0u32; n];
            if !bytes.is_empty() {
                unsafe {
                    std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr() as *mut u8, bytes.len());
                }
            }
            Ok(out)
        }
        #[cfg(not(target_endian = "little"))]
        {
            Ok(bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        }
    }

    pub fn f32_vec(&mut self) -> Result<Vec<f32>> {
        let n = self.usize64()?;
        let bytes = self.take(
            n.checked_mul(4)
                .ok_or_else(|| Error::Format("f32 overflow".into()))?,
        )?;
        #[cfg(target_endian = "little")]
        {
            let mut out = vec![0.0f32; n];
            if !bytes.is_empty() {
                unsafe {
                    std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr() as *mut u8, bytes.len());
                }
            }
            Ok(out)
        }
        #[cfg(not(target_endian = "little"))]
        {
            Ok(bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        }
    }
}

pub(crate) struct SectionCrcJob<'a> {
    pub name: String,
    pub slice: &'a [u8],
    pub expected: u64,
}

pub(crate) fn verify_section_crcs(jobs: &[SectionCrcJob<'_>]) -> Result<()> {
    if jobs.len() < 8 {
        for j in jobs {
            if fnv1a64(j.slice) != j.expected {
                return Err(Error::Checksum(format!(
                    "VXT integrity mismatch on tensor '{}'",
                    j.name
                )));
            }
        }
        return Ok(());
    }
    let n_workers = std::thread::available_parallelism()
        .map(|n| n.get().clamp(1, 8))
        .unwrap_or(4);
    let next_idx = std::sync::atomic::AtomicUsize::new(0);
    let bad_idx = std::sync::atomic::AtomicUsize::new(usize::MAX);
    std::thread::scope(|s| {
        for _ in 0..n_workers {
            s.spawn(|| loop {
                let idx = next_idx.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if idx >= jobs.len() || idx >= bad_idx.load(std::sync::atomic::Ordering::Relaxed) {
                    break;
                }
                if fnv1a64(jobs[idx].slice) != jobs[idx].expected {
                    bad_idx.fetch_min(idx, std::sync::atomic::Ordering::Relaxed);
                    break;
                }
            });
        }
    });
    let fail = bad_idx.into_inner();
    if fail < jobs.len() {
        return Err(Error::Checksum(format!(
            "VXT integrity mismatch on tensor '{}'",
            jobs[fail].name
        )));
    }
    Ok(())
}
