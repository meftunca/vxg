//! Fast zero-dependency LZ77 + Canonical Huffman compressor (`VXLZ`) for `.vxt` metadata blobs
//! (shrinks BPE vocabulary & merge tables by `~4x` in `<15 ms`).

use crate::format_io::{push_u8_slice, Cur};
use crate::huff::{huff_decode, huff_encode};
use crate::{Error, Result};

const VXLZ_MAGIC: &[u8; 4] = b"VXLZ";

fn store_raw_meta_blob(raw: &[u8]) -> Vec<u8> {
    if raw.starts_with(VXLZ_MAGIC) {
        // The legacy compressor only emits frames for inputs >= 256 bytes,
        // so a zero decoded length reserves an unambiguous raw escape.
        let mut out = Vec::with_capacity(8 + raw.len());
        out.extend_from_slice(VXLZ_MAGIC);
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(raw);
        out
    } else {
        raw.to_vec()
    }
}

pub(crate) fn compress_meta_blob(raw: &[u8]) -> Vec<u8> {
    if raw.len() < 256 {
        return store_raw_meta_blob(raw);
    }
    let mut head = vec![u32::MAX; 32768];
    let mut tags = Vec::with_capacity(raw.len() / 4);
    let mut lits = Vec::with_capacity(raw.len() / 2);
    let mut off_lo = Vec::new();
    let mut off_hi = Vec::new();
    let mut ext = Vec::new();

    let (mut anchor, mut ip) = (0usize, 0usize);
    let limit = raw.len().saturating_sub(4);
    while ip < limit {
        let w4 = u32::from_le_bytes([raw[ip], raw[ip + 1], raw[ip + 2], raw[ip + 3]]);
        let h = ((w4.wrapping_mul(0x9e37_79b1)) >> 17) as usize;
        let cand = head[h] as usize;
        head[h] = ip as u32;
        if cand < ip && ip - cand <= 65535 && raw[cand..cand + 4] == raw[ip..ip + 4] {
            let mut mlen = 4usize;
            let max_m = (raw.len() - ip).min(270);
            while mlen < max_m && raw[cand + mlen] == raw[ip + mlen] {
                mlen += 1;
            }
            let lrun = ip - anchor;
            let lt = lrun.min(15);
            let mt = (mlen - 4).min(15);
            tags.push((lt as u8) | ((mt as u8) << 4));
            if lt == 15 {
                let mut rem = lrun - 15;
                while rem >= 255 {
                    ext.push(255);
                    rem -= 255;
                }
                ext.push(rem as u8);
            }
            lits.extend_from_slice(&raw[anchor..ip]);
            let off = (ip - cand) as u16;
            off_lo.push((off & 0xff) as u8);
            off_hi.push((off >> 8) as u8);
            if mt == 15 {
                let mut rem = mlen - 19;
                while rem >= 255 {
                    ext.push(255);
                    rem -= 255;
                }
                ext.push(rem as u8);
            }
            ip += mlen;
            anchor = ip;
        } else {
            ip += 1;
        }
    }
    if anchor < raw.len() {
        let lrun = raw.len() - anchor;
        let lt = lrun.min(15);
        tags.push(lt as u8);
        if lt == 15 {
            let mut rem = lrun - 15;
            while rem >= 255 {
                ext.push(255);
                rem -= 255;
            }
            ext.push(rem as u8);
        }
        lits.extend_from_slice(&raw[anchor..]);
    }

    let mut out = Vec::with_capacity(raw.len() / 2);
    out.extend_from_slice(VXLZ_MAGIC);
    out.extend_from_slice(&(raw.len() as u32).to_le_bytes());
    out.extend_from_slice(&(tags.len() as u32).to_le_bytes());
    out.extend_from_slice(&(lits.len() as u32).to_le_bytes());
    out.extend_from_slice(&(off_lo.len() as u32).to_le_bytes());
    huff_encode(&tags, &mut out);
    huff_encode(&lits, &mut out);
    huff_encode(&off_hi, &mut out);
    push_u8_slice(&mut out, &off_lo);
    push_u8_slice(&mut out, &ext);
    if out.len() < raw.len() {
        out
    } else {
        store_raw_meta_blob(raw)
    }
}

pub(crate) fn decompress_meta_blob(blob: &[u8]) -> Result<Vec<u8>> {
    if !blob.starts_with(VXLZ_MAGIC) {
        return Ok(blob.to_vec());
    }
    let mut r = Cur { b: blob, pos: 4 };
    let raw_len = r.u32()? as usize;
    if raw_len == 0 {
        return Ok(blob[8..].to_vec());
    }
    let n_tags = r.u32()? as usize;
    let n_lits = r.u32()? as usize;
    let n_offs = r.u32()? as usize;
    let tags = huff_decode(&mut r, n_tags)?;
    let lits = huff_decode(&mut r, n_lits)?;
    let off_hi = huff_decode(&mut r, n_offs)?;
    let off_lo = r.u8_vec()?;
    let ext = r.u8_vec()?;
    if off_lo.len() != n_offs {
        return Err(Error::Format("corrupt VXLZ offset stream".into()));
    }

    let mut out = Vec::with_capacity(raw_len);
    let (mut lpos, mut opos, mut epos) = (0usize, 0usize, 0usize);
    for &tag in &tags {
        let mut lrun = (tag & 0x0f) as usize;
        if lrun == 15 {
            loop {
                let b = *ext
                    .get(epos)
                    .ok_or_else(|| Error::Format("VXLZ ext EOF".into()))?;
                epos += 1;
                lrun += b as usize;
                if b != 255 {
                    break;
                }
            }
        }
        let lend = lpos
            .checked_add(lrun)
            .filter(|&e| e <= lits.len())
            .ok_or_else(|| Error::Format("VXLZ lit overflow".into()))?;
        out.extend_from_slice(&lits[lpos..lend]);
        lpos = lend;

        if opos < n_offs {
            let off = ((off_lo[opos] as usize) | ((off_hi[opos] as usize) << 8)).max(1);
            opos += 1;
            if off > out.len() {
                return Err(Error::Format("VXLZ invalid match offset".into()));
            }
            let mut mlen = ((tag >> 4) as usize) + 4;
            if (tag >> 4) == 15 {
                loop {
                    let b = *ext
                        .get(epos)
                        .ok_or_else(|| Error::Format("VXLZ ext EOF".into()))?;
                    epos += 1;
                    mlen += b as usize;
                    if b != 255 {
                        break;
                    }
                }
            }
            let start = out.len() - off;
            for k in 0..mlen {
                let b = out[start + k];
                out.push(b);
            }
        }
    }
    if out.len() != raw_len {
        return Err(Error::Format("VXLZ output length mismatch".into()));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_magic_collisions_roundtrip() {
        for suffix in [
            vec![],
            vec![0; 4],
            b"arbitrary metadata".to_vec(),
            (0..1024)
                .map(|i| ((i * 73 + i * i * 19) >> 3) as u8)
                .collect(),
        ] {
            let mut raw = VXLZ_MAGIC.to_vec();
            raw.extend_from_slice(&suffix);
            assert_eq!(
                decompress_meta_blob(&compress_meta_blob(&raw)).unwrap(),
                raw
            );
        }
        let escaped = store_raw_meta_blob(b"VXLZ\0\0\0\0VXLZ");
        assert_eq!(
            decompress_meta_blob(&compress_meta_blob(&escaped)).unwrap(),
            escaped
        );

        // Exercise the incompressible fallback as well as the short path.
        let mut raw = VXLZ_MAGIC.to_vec();
        let mut state = 0x1234_5678u32;
        for _ in 0..1024 {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            raw.push(state as u8);
        }
        let packed = compress_meta_blob(&raw);
        assert_eq!(&packed[4..8], &[0; 4]);
        assert_eq!(decompress_meta_blob(&packed).unwrap(), raw);
    }

    #[test]
    fn legacy_raw_and_compressed_layouts_still_decode() {
        assert_eq!(
            decompress_meta_blob(b"ordinary raw metadata").unwrap(),
            b"ordinary raw metadata"
        );
        assert!(compress_meta_blob(b"").is_empty());
        let raw = vec![b'a'; 4096];
        let packed = compress_meta_blob(&raw);
        assert!(packed.starts_with(VXLZ_MAGIC));
        assert_eq!(&packed[4..8], &4096u32.to_le_bytes());
        assert_eq!(decompress_meta_blob(&packed).unwrap(), raw);
    }
}
