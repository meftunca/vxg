use crate::{Error, Result};

use super::{
    entry_size, max_compressed_size, write_entry, WorkerOutput, HEADER_SIZE, MAGIC,
    VERSION_RELEASE,
};

pub(crate) struct Part {
    pub bytes: Vec<u8>,
    pub kind: u32,
    pub checksum: u32,
    pub raw_size: usize,
}

pub(crate) fn assemble_workers(
    workers: Vec<WorkerOutput>,
    n: usize,
    block_size: usize,
    checksummed: bool,
    version: u16,
) -> Result<Vec<u8>> {
    let bound = max_compressed_size(n, block_size)?;
    let count = workers.iter().map(|w| w.parts.len()).sum::<usize>();
    let mut refs = Vec::with_capacity(count);
    let mut total_payload_len = 0usize;
    for (worker, output) in workers.iter().enumerate() {
        for part in &output.parts {
            let end = part
                .offset
                .checked_add(part.len)
                .ok_or(Error::SizeOverflow)?;
            if output.arena.get(part.offset..end).is_none() || part.len > part.raw_size {
                return Err(Error::InvalidFrame("encoder arena part"));
            }
            total_payload_len = total_payload_len
                .checked_add(part.len)
                .ok_or(Error::SizeOverflow)?;
            refs.push((worker, *part));
        }
    }
    if workers.len() > 1 {
        refs.sort_unstable_by_key(|(_, part)| part.index);
    }
    if refs
        .iter()
        .enumerate()
        .any(|(index, (_, part))| part.index != index)
    {
        return Err(Error::InvalidFrame("encoder block order"));
    }
    let esize = entry_size(version);
    let table_end = HEADER_SIZE
        .checked_add(esize.checked_mul(count).ok_or(Error::SizeOverflow)?)
        .ok_or(Error::SizeOverflow)?;
    let exact_len = table_end
        .checked_add(total_payload_len)
        .ok_or(Error::SizeOverflow)?;
    let mut out = Vec::new();
    out.try_reserve_exact(exact_len)
        .map_err(|_| Error::SizeOverflow)?;
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&version.to_le_bytes());
    out.extend_from_slice(&(u16::from(checksummed)).to_le_bytes());
    out.extend_from_slice(&(n as u64).to_le_bytes());
    out.extend_from_slice(&(block_size as u32).to_le_bytes());
    out.extend_from_slice(&(count as u32).to_le_bytes());
    out.extend_from_slice(&0u64.to_le_bytes());
    out.resize(table_end, 0);
    let mut raw_offset = 0usize;
    let mut payload_offset = table_end;
    for (i, (_, part)) in refs.iter().enumerate() {
        write_entry(
            &mut out,
            HEADER_SIZE + i * esize,
            version,
            raw_offset,
            payload_offset,
            part.raw_size,
            part.len,
            part.checksum,
            part.kind,
        );
        raw_offset = raw_offset
            .checked_add(part.raw_size)
            .ok_or(Error::SizeOverflow)?;
        payload_offset = payload_offset
            .checked_add(part.len)
            .ok_or(Error::SizeOverflow)?;
    }
    if raw_offset != n {
        return Err(Error::InvalidFrame("encoder total"));
    }
    for (worker, part) in refs {
        let end = part
            .offset
            .checked_add(part.len)
            .ok_or(Error::SizeOverflow)?;
        out.extend_from_slice(&workers[worker].arena[part.offset..end]);
    }
    debug_assert!(out.len() <= bound);
    Ok(out)
}

pub(crate) fn assemble_workers_into(
    workers: Vec<WorkerOutput>,
    out: &mut [u8],
    n: usize,
    block_size: usize,
    checksummed: bool,
    version: u16,
) -> Result<usize> {
    let bound = max_compressed_size(n, block_size)?;
    let count = workers.iter().map(|w| w.parts.len()).sum::<usize>();
    let mut refs = Vec::with_capacity(count);
    let mut total_payload_len = 0usize;
    for (worker, output) in workers.iter().enumerate() {
        for part in &output.parts {
            let end = part
                .offset
                .checked_add(part.len)
                .ok_or(Error::SizeOverflow)?;
            if output.arena.get(part.offset..end).is_none() || part.len > part.raw_size {
                return Err(Error::InvalidFrame("encoder arena part"));
            }
            total_payload_len = total_payload_len
                .checked_add(part.len)
                .ok_or(Error::SizeOverflow)?;
            refs.push((worker, *part));
        }
    }
    if workers.len() > 1 {
        refs.sort_unstable_by_key(|(_, part)| part.index);
    }
    if refs
        .iter()
        .enumerate()
        .any(|(index, (_, part))| part.index != index)
    {
        return Err(Error::InvalidFrame("encoder block order"));
    }
    let esize = entry_size(version);
    let table_end = HEADER_SIZE
        .checked_add(esize.checked_mul(count).ok_or(Error::SizeOverflow)?)
        .ok_or(Error::SizeOverflow)?;
    let exact_len = table_end
        .checked_add(total_payload_len)
        .ok_or(Error::SizeOverflow)?;
    if out.len() < exact_len {
        return Err(Error::OutputSize {
            expected: exact_len,
            actual: out.len(),
        });
    }
    out[0..4].copy_from_slice(MAGIC);
    out[4..6].copy_from_slice(&version.to_le_bytes());
    out[6..8].copy_from_slice(&(u16::from(checksummed)).to_le_bytes());
    out[8..16].copy_from_slice(&(n as u64).to_le_bytes());
    out[16..20].copy_from_slice(&(block_size as u32).to_le_bytes());
    out[20..24].copy_from_slice(&(count as u32).to_le_bytes());
    out[24..32].copy_from_slice(&0u64.to_le_bytes());
    let mut raw_offset = 0usize;
    let mut payload_offset = table_end;
    for (i, (_, part)) in refs.iter().enumerate() {
        write_entry(
            out,
            HEADER_SIZE + i * esize,
            version,
            raw_offset,
            payload_offset,
            part.raw_size,
            part.len,
            part.checksum,
            part.kind,
        );
        raw_offset = raw_offset
            .checked_add(part.raw_size)
            .ok_or(Error::SizeOverflow)?;
        payload_offset = payload_offset
            .checked_add(part.len)
            .ok_or(Error::SizeOverflow)?;
    }
    if raw_offset != n {
        return Err(Error::InvalidFrame("encoder total"));
    }
    let mut write_pos = table_end;
    for (worker, part) in refs {
        let end = part
            .offset
            .checked_add(part.len)
            .ok_or(Error::SizeOverflow)?;
        out[write_pos..write_pos + part.len]
            .copy_from_slice(&workers[worker].arena[part.offset..end]);
        write_pos += part.len;
    }
    debug_assert!(write_pos == exact_len && exact_len <= bound);
    Ok(exact_len)
}

pub(crate) fn assemble(
    parts: Vec<Part>,
    n: usize,
    block_size: usize,
    checksummed: bool,
) -> Result<Vec<u8>> {
    let bound = max_compressed_size(n, block_size)?;
    let version = VERSION_RELEASE;
    let esize = entry_size(version);
    let count = parts.len();
    let table_end = HEADER_SIZE
        .checked_add(esize.checked_mul(count).ok_or(Error::SizeOverflow)?)
        .ok_or(Error::SizeOverflow)?;
    let total_payload: usize = parts.iter().map(|p| p.bytes.len()).sum();
    let exact_len = table_end
        .checked_add(total_payload)
        .ok_or(Error::SizeOverflow)?;
    let mut out = Vec::new();
    out.try_reserve_exact(exact_len)
        .map_err(|_| Error::SizeOverflow)?;
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&version.to_le_bytes());
    out.extend_from_slice(&(u16::from(checksummed)).to_le_bytes());
    out.extend_from_slice(&(n as u64).to_le_bytes());
    out.extend_from_slice(&(block_size as u32).to_le_bytes());
    out.extend_from_slice(&(count as u32).to_le_bytes());
    out.extend_from_slice(&0u64.to_le_bytes());
    out.resize(table_end, 0);
    let mut raw_offset = 0usize;
    let mut payload_offset = table_end;
    for (i, p) in parts.iter().enumerate() {
        if p.bytes.len() > p.raw_size {
            return Err(Error::InvalidFrame("encoder expansion"));
        }
        write_entry(
            &mut out,
            HEADER_SIZE + i * esize,
            version,
            raw_offset,
            payload_offset,
            p.raw_size,
            p.bytes.len(),
            p.checksum,
            p.kind,
        );
        raw_offset += p.raw_size;
        payload_offset += p.bytes.len();
    }
    if raw_offset != n {
        return Err(Error::InvalidFrame("encoder total"));
    }
    for p in parts {
        out.extend_from_slice(&p.bytes);
    }
    debug_assert!(out.len() <= bound);
    Ok(out)
}
