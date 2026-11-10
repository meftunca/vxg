//! Persistent low-overhead worker pool keeping thread-local arenas warm across calls.
//!
//! Uncontended calls with `<= 64` workers dispatch lock-free via 128-byte cache-aligned slots
//! with a `SeqCst` Dekker sleep/wake handshake (zero kernel syscalls while hot, 0% CPU when idle).
//! Concurrent or nested callers transparently fall back to `std::thread::scope`, making deadlocks
//! impossible while preserving deterministic multi-threaded test execution.

use std::cell::UnsafeCell;
#[cfg(not(target_arch = "wasm32"))]
use std::panic::{catch_unwind, AssertUnwindSafe};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::OnceLock;
#[cfg(not(target_arch = "wasm32"))]
use std::thread::Thread;

use crate::{Error, Result};

#[cfg(not(target_arch = "wasm32"))]
const MAX_FAST_SPAWNED: usize = 63;
#[cfg(not(target_arch = "wasm32"))]
const HOT_SPIN_ITERS: u32 = 4096;

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy)]
struct Job {
    call: unsafe fn(*const (), usize),
    data: *const (),
}

#[cfg(not(target_arch = "wasm32"))]
unsafe fn noop_call(_: *const (), _: usize) {}

#[cfg(not(target_arch = "wasm32"))]
#[repr(align(128))]
struct WorkerSlot {
    target_epoch: AtomicU64,
    sleeping: AtomicBool,
    thread: UnsafeCell<Option<Thread>>,
}

#[cfg(not(target_arch = "wasm32"))]
impl WorkerSlot {
    const fn new() -> Self {
        Self {
            target_epoch: AtomicU64::new(0),
            sleeping: AtomicBool::new(false),
            thread: UnsafeCell::new(None),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
struct FastPool {
    spawned_count: UnsafeCell<usize>,
    next_epoch: AtomicU64,
    remaining: AtomicUsize,
    worker_panicked: AtomicBool,
    job: UnsafeCell<Job>,
    caller_thread: UnsafeCell<Option<Thread>>,
    slots: [WorkerSlot; MAX_FAST_SPAWNED],
}

// SAFETY: Access to `UnsafeCell` fields of `FastPool` is serialized by `POOL_BUSY` (`Acquire`/`Release`)
// for the coordinator, and synchronized with workers via `slot.target_epoch` (`SeqCst`/`Acquire`) before
// job execution and `remaining.fetch_sub(1, AcqRel)` before coordinator teardown.
#[cfg(not(target_arch = "wasm32"))]
unsafe impl Send for FastPool {}
#[cfg(not(target_arch = "wasm32"))]
unsafe impl Sync for FastPool {}

#[cfg(not(target_arch = "wasm32"))]
static POOL_BUSY: AtomicBool = AtomicBool::new(false);
#[cfg(not(target_arch = "wasm32"))]
static FAST_POOL: OnceLock<FastPool> = OnceLock::new();

#[cfg(not(target_arch = "wasm32"))]
fn pool() -> &'static FastPool {
    FAST_POOL.get_or_init(|| FastPool {
        spawned_count: UnsafeCell::new(0),
        next_epoch: AtomicU64::new(0),
        remaining: AtomicUsize::new(0),
        worker_panicked: AtomicBool::new(false),
        job: UnsafeCell::new(Job {
            call: noop_call,
            data: std::ptr::null(),
        }),
        caller_thread: UnsafeCell::new(None),
        slots: [const { WorkerSlot::new() }; MAX_FAST_SPAWNED],
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn worker_loop(slot_idx: usize) {
    let p = pool();
    let slot = &p.slots[slot_idx];
    let wid = slot_idx + 1;
    let mut last_epoch = 0u64;

    loop {
        let mut cur = slot.target_epoch.load(Ordering::Acquire);
        if cur == last_epoch {
            for _ in 0..HOT_SPIN_ITERS {
                std::hint::spin_loop();
                cur = slot.target_epoch.load(Ordering::Acquire);
                if cur != last_epoch {
                    break;
                }
            }
            if cur == last_epoch {
                slot.sleeping.store(true, Ordering::SeqCst);
                cur = slot.target_epoch.load(Ordering::SeqCst);
                if cur == last_epoch {
                    loop {
                        std::thread::park();
                        cur = slot.target_epoch.load(Ordering::Acquire);
                        if cur != last_epoch {
                            break;
                        }
                    }
                }
                slot.sleeping.store(false, Ordering::Relaxed);
            }
        }
        last_epoch = cur;

        let job = unsafe { *p.job.get() };
        let res = catch_unwind(AssertUnwindSafe(|| unsafe { (job.call)(job.data, wid) }));
        if res.is_err() {
            p.worker_panicked.store(true, Ordering::Release);
        }
        if p.remaining.fetch_sub(1, Ordering::AcqRel) == 1 {
            if let Some(ref caller) = unsafe { &*p.caller_thread.get() } {
                caller.unpark();
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
unsafe fn call_trampoline<F: Fn(usize) + Sync>(data: *const (), wid: usize) {
    let f = unsafe { &*(data as *const F) };
    f(wid);
}

#[cfg(not(target_arch = "wasm32"))]
struct BusyJoinGuard<'a> {
    pool: &'a FastPool,
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for BusyJoinGuard<'_> {
    fn drop(&mut self) {
        for _ in 0..512 {
            if self.pool.remaining.load(Ordering::Acquire) == 0 {
                break;
            }
            std::hint::spin_loop();
        }
        while self.pool.remaining.load(Ordering::Acquire) != 0 {
            std::thread::park();
        }
        unsafe {
            *self.pool.job.get() = Job {
                call: noop_call,
                data: std::ptr::null(),
            };
            *self.pool.caller_thread.get() = None;
        }
        POOL_BUSY.store(false, Ordering::Release);
    }
}

/// Executes `f(0..workers)` across `workers` threads (`wid = 0` runs on the calling thread).
/// Returns `true` if any worker thread panicked.
#[cfg(target_arch = "wasm32")]
pub(crate) fn parallel_run_checked<F>(workers: usize, f: F) -> bool
where
    F: Fn(usize) + Sync,
{
    for wid in 0..workers.max(1) {
        f(wid);
    }
    false
}

/// Executes `f(0..workers)` across `workers` threads (`wid = 0` runs on the calling thread).
/// Returns `true` if any worker thread panicked.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn parallel_run_checked<F>(workers: usize, f: F) -> bool
where
    F: Fn(usize) + Sync,
{
    if workers <= 1 {
        f(0);
        return false;
    }
    let spawned = workers - 1;
    if spawned <= MAX_FAST_SPAWNED
        && POOL_BUSY
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
    {
        let p = pool();
        unsafe {
            let cur_spawned = *p.spawned_count.get();
            if cur_spawned < spawned {
                for idx in cur_spawned..spawned {
                    let handle = std::thread::Builder::new()
                        .name(format!("vxg-w{}", idx + 1))
                        .spawn(move || worker_loop(idx))
                        .expect("failed to spawn VXG pool worker");
                    *p.slots[idx].thread.get() = Some(handle.thread().clone());
                    *p.spawned_count.get() = idx + 1;
                }
            }
            *p.job.get() = Job {
                call: call_trampoline::<F>,
                data: (&f as *const F).cast(),
            };
            *p.caller_thread.get() = Some(std::thread::current());
        }
        p.worker_panicked.store(false, Ordering::Relaxed);
        p.remaining.store(spawned, Ordering::Relaxed);
        let epoch = p.next_epoch.fetch_add(1, Ordering::Relaxed) + 1;

        let guard = BusyJoinGuard { pool: p };
        for slot in &p.slots[..spawned] {
            slot.target_epoch.store(epoch, Ordering::SeqCst);
            if slot.sleeping.load(Ordering::SeqCst) {
                if let Some(ref th) = unsafe { &*slot.thread.get() } {
                    th.unpark();
                }
            }
        }

        let caller_res = catch_unwind(AssertUnwindSafe(|| f(0)));
        // Wait for all workers before reading `worker_panicked` or resuming unwind.
        for _ in 0..512 {
            if p.remaining.load(Ordering::Acquire) == 0 {
                break;
            }
            std::hint::spin_loop();
        }
        while p.remaining.load(Ordering::Acquire) != 0 {
            std::thread::park();
        }
        let panicked = caller_res.is_err() || p.worker_panicked.load(Ordering::Acquire);
        drop(guard);
        return panicked;
    }

    // Fallback path when `POOL_BUSY` is held by another concurrent caller or `workers > 64`.
    let any_panic = AtomicBool::new(false);
    std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(spawned);
        for wid in 1..workers {
            let f_ref = &f;
            handles.push(s.spawn(move || f_ref(wid)));
        }
        if catch_unwind(AssertUnwindSafe(|| f(0))).is_err() {
            any_panic.store(true, Ordering::Release);
        }
        for h in handles {
            if h.join().is_err() {
                any_panic.store(true, Ordering::Release);
            }
        }
    });
    any_panic.load(Ordering::Acquire)
}

#[inline]
pub(crate) fn ensure_warm(workers: usize) {
    parallel_run(workers.min(16), |_| {});
}

#[inline]
pub(crate) fn parallel_run<F>(workers: usize, f: F)
where
    F: Fn(usize) + Sync,
{
    if parallel_run_checked(workers, f) {
        panic!("VXG worker thread panicked");
    }
}

#[inline]
pub(crate) fn parallel_try_run<F>(workers: usize, f: F) -> Result<()>
where
    F: Fn(usize) -> Result<()> + Sync,
{
    if workers <= 1 {
        return f(0);
    }
    struct ErrSlot(UnsafeCell<Option<Error>>);
    unsafe impl Sync for ErrSlot {}
    let errs: Vec<ErrSlot> = (0..workers).map(|_| ErrSlot(UnsafeCell::new(None))).collect();
    let panicked = parallel_run_checked(workers, |wid| {
        if let Err(e) = f(wid) {
            unsafe {
                *errs[wid].0.get() = Some(e);
            }
        }
    });
    for slot in errs {
        if let Some(e) = slot.0.into_inner() {
            return Err(e);
        }
    }
    if panicked {
        return Err(Error::WorkerPanic);
    }
    Ok(())
}
