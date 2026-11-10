use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};
use std::thread::Thread;

use crate::{block, block::WorkStats, pool, Error, Result};

use super::{
    entry_size, max_compressed_size, v5, write_entry, ArenaPart, WorkerSlice, HEADER_SIZE, MAGIC,
    VERSION_RELEASE, VERSION_V4, VERSION_V5,
};

pub(crate) struct Part {
    pub bytes: Vec<u8>,
    pub kind: u32,
    pub checksum: u32,
    pub raw_size: usize,
}

pub(crate) enum AssembleDest<'a> {
    Alloc(&'a mut Vec<u8>),
    Slice(&'a mut [u8]),
}

#[derive(Clone, Copy)]
struct WorkerMeta {
    arena_len: usize,
    parts_ptr: *const ArenaPart,
    parts_len: usize,
    stats: WorkStats,
}

struct SyncCell<T>(UnsafeCell<T>);
impl<T> SyncCell<T> {
    #[inline]
    fn get(&self) -> *mut T {
        self.0.get()
    }
}
// SAFETY:
// - `slots[w]` and `parkers[w]` are written exclusively by worker `w` BEFORE
//   `done_encode.fetch_add(1, AcqRel)`.
// - The single thread observing `done_encode == workers` has `Acquire` visibility
//   of all `slots`/`parkers` (while all other workers remain blocked inside `finish`
//   keeping their borrowed `parts_ptr` slices alive), exclusively accesses `dest_cell`,
//   writes `plan_cell` and `plan_err_cell`, then publishes `plan_ready.store(state, Release)`.
// - After `plan_ready.load(Acquire) != 0`, `plan_cell` is read immutably and each worker `w`
//   copies exclusively its own disjoint block indices into `out_base`.
unsafe impl<T: Send> Sync for SyncCell<T> {}
unsafe impl Sync for WorkerMeta {}
unsafe impl Send for WorkerMeta {}

fn wake_all(parkers: &[SyncCell<Option<Thread>>]) {
    for slot in parkers {
        if let Some(ref t) = unsafe { &*slot.0.get() } {
            t.unpark();
        }
    }
}

struct DoneGuard<'a> {
    done_encode: &'a AtomicUsize,
    aborted: &'a AtomicBool,
    plan_ready: &'a AtomicU8,
    parkers: &'a [SyncCell<Option<Thread>>],
    workers: usize,
    completed: bool,
}

impl Drop for DoneGuard<'_> {
    fn drop(&mut self) {
        if !self.completed {
            self.aborted.store(true, Ordering::Release);
            if self.done_encode.fetch_add(1, Ordering::AcqRel) + 1 == self.workers {
                self.plan_ready.store(2, Ordering::Release);
                wake_all(self.parkers);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn build_assembly_plan(
    slots: &[SyncCell<Option<WorkerMeta>>],
    n: usize,
    count: usize,
    block_size: usize,
    checksummed: bool,
    version: u16,
    has_var_raw: bool,
    want_row_index: bool,
    opt_dict: Option<(&[u8], usize)>,
    bound: usize,
    dest: &mut AssembleDest<'_>,
) -> Result<(usize, usize, Vec<usize>, WorkStats)> {
    let mut ordered: Vec<Option<ArenaPart>> = vec![None; count];
    let mut total_payload_len = 0usize;
    let mut total_stats = WorkStats::default();
    let mut seen = 0usize;
    let mut any_dict_lz = false;
    for slot in slots {
        let Some(w_meta) = (unsafe { *slot.0.get() }) else {
            return Err(Error::WorkerPanic);
        };
        let arena_len = w_meta.arena_len;
        total_stats += w_meta.stats;
        let parts = unsafe { std::slice::from_raw_parts(w_meta.parts_ptr, w_meta.parts_len) };
        for part in parts {
            let end = part
                .offset
                .checked_add(part.len)
                .ok_or(Error::SizeOverflow)?;
            if end > arena_len || part.len > part.raw_size {
                return Err(Error::InvalidFrame("encoder arena part"));
            }
            if part.index >= count || ordered[part.index].is_some() {
                return Err(Error::InvalidFrame("encoder block order"));
            }
            if part.kind == block::DICT_LZ {
                any_dict_lz = true;
            }
            total_payload_len = total_payload_len
                .checked_add(part.len)
                .ok_or(Error::SizeOverflow)?;
            ordered[part.index] = Some(*part);
            seen += 1;
        }
    }
    if seen != count {
        return Err(Error::InvalidFrame("encoder block order"));
    }
    let active_dict = if any_dict_lz { opt_dict } else { None };
    let esize = entry_size(if version < VERSION_V4 { version } else { VERSION_V4 });
    let base_dir_end = HEADER_SIZE
        .checked_add(esize.checked_mul(count).ok_or(Error::SizeOverflow)?)
        .ok_or(Error::SizeOverflow)?;
    let var_raw_bytes = if has_var_raw {
        count.checked_mul(4).ok_or(Error::SizeOverflow)?
    } else {
        0
    };
    let dict_wire_bytes = active_dict.map_or(0, |(w, _)| w.len());
    let base_table_end = base_dir_end
        .checked_add(var_raw_bytes)
        .and_then(|x| x.checked_add(dict_wire_bytes))
        .ok_or(Error::SizeOverflow)?;
    let base_exact_len = base_table_end
        .checked_add(total_payload_len)
        .ok_or(Error::SizeOverflow)?;
    if active_dict.is_some() && base_exact_len > bound {
        return Err(Error::InvalidFrame("dict_expansion"));
    }
    let cand_row_index =
        version >= VERSION_V4 && count > 0 && (want_row_index || has_var_raw || active_dict.is_some());
    let ri_bytes_cand = if cand_row_index {
        count.checked_mul(4).ok_or(Error::SizeOverflow)?
    } else {
        0
    };
    let has_row_index = if cand_row_index {
        match base_exact_len.checked_add(ri_bytes_cand) {
            Some(with_ri) if with_ri <= bound => true,
            _ if want_row_index && active_dict.is_some() => {
                return Err(Error::InvalidFrame("dict_expansion"));
            }
            _ => false,
        }
    } else {
        false
    };
    let row_index_bytes = if has_row_index { ri_bytes_cand } else { 0 };
    let actual_version = if version < VERSION_V4 {
        version
    } else if active_dict.is_some() || has_var_raw || has_row_index {
        VERSION_V5
    } else {
        VERSION_V4
    };
    let table_end = base_table_end + row_index_bytes;
    let exact_len = base_exact_len + row_index_bytes;
    let header_slice: &mut [u8] = match dest {
        AssembleDest::Alloc(vec) => {
            vec.clear();
            vec.try_reserve_exact(exact_len)
                .map_err(|_| Error::SizeOverflow)?;
            vec.resize(table_end, 0);
            &mut vec[..table_end]
        }
        AssembleDest::Slice(slice) => {
            if slice.len() < exact_len {
                return Err(Error::OutputSize {
                    expected: exact_len,
                    actual: slice.len(),
                });
            }
            &mut slice[..table_end]
        }
    };
    v5::write_header_bytes(
        header_slice,
        actual_version,
        checksummed,
        has_var_raw,
        has_row_index,
        active_dict,
        n,
        block_size,
        count,
    );
    let ri_base = base_dir_end + var_raw_bytes + dict_wire_bytes;
    let mut dst_offsets = Vec::with_capacity(count);
    let mut raw_offset = 0usize;
    let mut payload_offset = table_end;
    for (i, slot) in ordered.iter().enumerate() {
        let part = slot.ok_or(Error::InvalidFrame("encoder block order"))?;
        write_entry(
            header_slice,
            HEADER_SIZE + i * esize,
            actual_version,
            raw_offset,
            payload_offset,
            part.raw_size,
            part.len,
            part.checksum,
            part.kind,
        );
        if has_var_raw {
            let vr_pos = base_dir_end + i * 4;
            header_slice[vr_pos..vr_pos + 4].copy_from_slice(&(part.raw_size as u32).to_le_bytes());
        }
        if has_row_index {
            let ri_pos = ri_base + i * 4;
            header_slice[ri_pos..ri_pos + 4].copy_from_slice(&part.row_meta.to_le_bytes());
        }
        dst_offsets.push(payload_offset);
        raw_offset = raw_offset
            .checked_add(part.raw_size)
            .ok_or(Error::SizeOverflow)?;
        payload_offset = payload_offset
            .checked_add(part.len)
            .ok_or(Error::SizeOverflow)?;
    }
    if let Some((wire, _)) = active_dict {
        let d_pos = base_dir_end + var_raw_bytes;
        header_slice[d_pos..d_pos + wire.len()].copy_from_slice(wire);
    }
    if raw_offset != n {
        return Err(Error::InvalidFrame("encoder total"));
    }
    let out_base_addr = header_slice.as_mut_ptr() as usize;
    Ok((out_base_addr, exact_len, dst_offsets, total_stats))
}

#[inline]
unsafe fn copy_worker_parts(
    arena: &[u8],
    parts: &[ArenaPart],
    out_base_addr: usize,
    dst_offsets: &[usize],
) {
    let src_base = arena.as_ptr();
    let dst_base = out_base_addr as *mut u8;
    for part in parts {
        if part.len > 0 {
            let dst_off = *dst_offsets.get_unchecked(part.index);
            std::ptr::copy_nonoverlapping(
                src_base.add(part.offset),
                dst_base.add(dst_off),
                part.len,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn parallel_encode_assemble<F>(
    workers: usize,
    n: usize,
    count: usize,
    block_size: usize,
    checksummed: bool,
    version: u16,
    has_var_raw: bool,
    want_row_index: bool,
    opt_dict: Option<(&[u8], usize)>,
    dest: AssembleDest<'_>,
    encode_fn: F,
) -> Result<(usize, WorkStats)>
where
    F: Fn(usize, &mut dyn FnMut(WorkerSlice<'_>) -> Result<()>) -> Result<()> + Sync,
{
    let bound = max_compressed_size(n, block_size)?;
    let done_encode = AtomicUsize::new(0);
    let aborted = AtomicBool::new(false);
    let plan_ready = AtomicU8::new(0); // 0 = pending, 1 = ok, 2 = err
    let slots: Vec<SyncCell<Option<WorkerMeta>>> =
        (0..workers).map(|_| SyncCell(UnsafeCell::new(None))).collect();
    let parkers: Vec<SyncCell<Option<Thread>>> =
        (0..workers).map(|_| SyncCell(UnsafeCell::new(None))).collect();
    let dest_cell = SyncCell(UnsafeCell::new(dest));
    let plan_cell: SyncCell<(usize, usize, Vec<usize>, WorkStats)> =
        SyncCell(UnsafeCell::new((0, 0, Vec::new(), WorkStats::default())));
    let plan_err_cell: SyncCell<Option<Error>> = SyncCell(UnsafeCell::new(None));

    pool::parallel_try_run(workers, |w| {
        unsafe {
            *parkers[w].0.get() = Some(std::thread::current());
        }
        let mut guard = DoneGuard {
            done_encode: &done_encode,
            aborted: &aborted,
            plan_ready: &plan_ready,
            parkers: &parkers,
            workers,
            completed: false,
        };
        encode_fn(w, &mut |slice: WorkerSlice<'_>| {
            unsafe {
                *slots[w].0.get() = Some(WorkerMeta {
                    arena_len: slice.arena.len(),
                    parts_ptr: slice.parts.as_ptr(),
                    parts_len: slice.parts.len(),
                    stats: slice.stats,
                });
            }
            guard.completed = true;
            if done_encode.fetch_add(1, Ordering::AcqRel) + 1 == workers {
                let state = if aborted.load(Ordering::Acquire) {
                    2u8
                } else {
                    let dest_mut = unsafe { &mut *dest_cell.get() };
                    match build_assembly_plan(
                        &slots,
                        n,
                        count,
                        block_size,
                        checksummed,
                        version,
                        has_var_raw,
                        want_row_index,
                        opt_dict,
                        bound,
                        dest_mut,
                    ) {
                        Ok(plan_tuple) => {
                            unsafe {
                                *plan_cell.get() = plan_tuple;
                            }
                            1u8
                        }
                        Err(e) => {
                            aborted.store(true, Ordering::Release);
                            unsafe {
                                *plan_err_cell.get() = Some(e);
                            }
                            2u8
                        }
                    }
                };
                plan_ready.store(state, Ordering::Release);
                wake_all(&parkers);
            } else {
                for _ in 0..256 {
                    if plan_ready.load(Ordering::Acquire) != 0 {
                        break;
                    }
                    std::hint::spin_loop();
                }
                while plan_ready.load(Ordering::Acquire) == 0 {
                    std::thread::park();
                }
            }
            if plan_ready.load(Ordering::Acquire) == 1 {
                let (out_base, _, ref dst_offsets, _) = unsafe { &*plan_cell.get() };
                unsafe {
                    copy_worker_parts(slice.arena, slice.parts, *out_base, dst_offsets);
                }
            }
            Ok(())
        })
    })?;

    if let Some(e) = plan_err_cell.0.into_inner() {
        return Err(e);
    }
    if aborted.load(Ordering::Acquire) || plan_ready.load(Ordering::Acquire) != 1 {
        return Err(Error::WorkerPanic);
    }
    let (_, exact_len, _, total_stats) = plan_cell.0.into_inner();
    let dest = dest_cell.0.into_inner();
    if let AssembleDest::Alloc(vec) = dest {
        // SAFETY: `build_assembly_plan` reserved `>= exact_len` bytes and initialized
        // `0..table_end`, and all `workers` copied their disjoint payload slices without error
        // or panic before `pool::parallel_try_run` returned, initializing `table_end..exact_len`.
        unsafe {
            vec.set_len(exact_len);
        }
        debug_assert!(vec.len() <= bound);
    } else {
        debug_assert!(exact_len <= bound);
    }
    Ok((exact_len, total_stats))
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
