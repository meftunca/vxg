use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};
use std::thread::Thread;

use crate::{block::WorkStats, Error, Result};

use super::{
    entry_size, max_compressed_size, write_entry, ArenaPart, WorkerOutput, HEADER_SIZE, MAGIC,
    VERSION_RELEASE,
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
//   of all `slots`/`parkers`, exclusively accesses `dest_cell` and writes `plan_cell`
//   and `plan_err`, then publishes `plan_ready.store(state, Release)`.
// - After `plan_ready.load(Acquire) != 0`, `plan_cell` is read immutably and each
//   `slots[k]` is claimed at most once via `next_copy.fetch_add(1, Relaxed)`.
unsafe impl<T: Send> Sync for SyncCell<T> {}

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

fn build_assembly_plan(
    slots: &[SyncCell<Option<WorkerOutput>>],
    n: usize,
    count: usize,
    block_size: usize,
    checksummed: bool,
    version: u16,
    dest: &mut AssembleDest<'_>,
) -> Result<(usize, usize, Vec<usize>, WorkStats)> {
    let mut ordered: Vec<Option<ArenaPart>> = vec![None; count];
    let mut total_payload_len = 0usize;
    let mut total_stats = WorkStats::default();
    let mut seen = 0usize;
    for slot in slots {
        let Some(ref w_out) = (unsafe { &*slot.0.get() }) else {
            return Err(Error::WorkerPanic);
        };
        let arena_len = w_out.arena.len();
        total_stats += w_out.stats;
        for part in &w_out.parts {
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
    let esize = entry_size(version);
    let table_end = HEADER_SIZE
        .checked_add(esize.checked_mul(count).ok_or(Error::SizeOverflow)?)
        .ok_or(Error::SizeOverflow)?;
    let exact_len = table_end
        .checked_add(total_payload_len)
        .ok_or(Error::SizeOverflow)?;
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
    header_slice[0..4].copy_from_slice(MAGIC);
    header_slice[4..6].copy_from_slice(&version.to_le_bytes());
    header_slice[6..8].copy_from_slice(&(u16::from(checksummed)).to_le_bytes());
    header_slice[8..16].copy_from_slice(&(n as u64).to_le_bytes());
    header_slice[16..20].copy_from_slice(&(block_size as u32).to_le_bytes());
    header_slice[20..24].copy_from_slice(&(count as u32).to_le_bytes());
    header_slice[24..32].copy_from_slice(&0u64.to_le_bytes());
    let mut dst_offsets = Vec::with_capacity(count);
    let mut raw_offset = 0usize;
    let mut payload_offset = table_end;
    for (i, slot) in ordered.iter().enumerate() {
        let part = slot.ok_or(Error::InvalidFrame("encoder block order"))?;
        write_entry(
            header_slice,
            HEADER_SIZE + i * esize,
            version,
            raw_offset,
            payload_offset,
            part.raw_size,
            part.len,
            part.checksum,
            part.kind,
        );
        dst_offsets.push(payload_offset);
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
    dest: AssembleDest<'_>,
    encode_fn: F,
) -> Result<(usize, WorkStats)>
where
    F: Fn(usize) -> Result<WorkerOutput> + Sync,
{
    let bound = max_compressed_size(n, block_size)?;
    let done_encode = AtomicUsize::new(0);
    let next_copy = AtomicUsize::new(0);
    let aborted = AtomicBool::new(false);
    let plan_ready = AtomicU8::new(0); // 0 = pending, 1 = ok, 2 = err
    let slots: Vec<SyncCell<Option<WorkerOutput>>> =
        (0..workers).map(|_| SyncCell(UnsafeCell::new(None))).collect();
    let parkers: Vec<SyncCell<Option<Thread>>> =
        (0..workers).map(|_| SyncCell(UnsafeCell::new(None))).collect();
    let dest_cell = SyncCell(UnsafeCell::new(dest));
    let plan_cell: SyncCell<(usize, usize, Vec<usize>, WorkStats)> =
        SyncCell(UnsafeCell::new((0, 0, Vec::new(), WorkStats::default())));
    let plan_err_cell: SyncCell<Option<Error>> = SyncCell(UnsafeCell::new(None));

    let first_err = std::thread::scope(|scope| -> Option<Error> {
        let run_worker = |w: usize| -> Result<()> {
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
            let out = encode_fn(w)?;
            unsafe {
                *slots[w].0.get() = Some(out);
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
                for _ in 0..64 {
                    if plan_ready.load(Ordering::Acquire) != 0 {
                        break;
                    }
                    std::hint::spin_loop();
                }
                while plan_ready.load(Ordering::Acquire) == 0 {
                    std::thread::park();
                }
            }
            let state = plan_ready.load(Ordering::Acquire);
            loop {
                let k = next_copy.fetch_add(1, Ordering::Relaxed);
                if k >= workers {
                    break;
                }
                if let Some(w_out) = unsafe { (*slots[k].get()).take() } {
                    if state == 1 {
                        let (out_base, _, ref dst_offsets, _) = unsafe { &*plan_cell.get() };
                        unsafe {
                            copy_worker_parts(&w_out.arena, &w_out.parts, *out_base, dst_offsets);
                        }
                    }
                    drop(w_out);
                }
            }
            Ok(())
        };

        let mut handles_uninit = [const {
            core::mem::MaybeUninit::<std::thread::ScopedJoinHandle<'_, Result<()>>>::uninit()
        }; 256];
        let spawned = workers - 1;
        for w in 1..workers {
            handles_uninit[w - 1].write(scope.spawn(move || run_worker(w)));
        }
        let mut first_err = run_worker(0).err();
        for slot in handles_uninit.iter_mut().take(spawned) {
            match unsafe { slot.assume_init_read() }.join() {
                Ok(Ok(())) => {}
                Ok(Err(e)) if first_err.is_none() => first_err = Some(e),
                Err(_) => {
                    aborted.store(true, Ordering::Release);
                    if first_err.is_none() {
                        first_err = Some(Error::WorkerPanic);
                    }
                }
                _ => {}
            }
        }
        first_err
    });

    if let Some(e) = first_err {
        return Err(e);
    }
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
        // `0..table_end`, and all `workers` slots were claimed and copied without error
        // or panic before `std::thread::scope` returned, initializing `table_end..exact_len`.
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
