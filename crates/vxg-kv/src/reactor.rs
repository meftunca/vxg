//! Zero-dependency single-threaded native `kqueue` (macOS) / `epoll` (Linux) TCP reactor.
//!
//! Owns [`KvEngine`] directly with **zero `Mutex`/`RwLock` locking**, **zero per-client thread
//! stacks**, and **zero-allocation RESP2 span dispatch** across thousands of concurrent sockets.

#![cfg(any(target_os = "macos", target_os = "linux"))]

use std::collections::HashMap;
use std::io::{self, Write};
use std::net::{TcpListener, TcpStream};
use std::os::fd::{AsRawFd, RawFd};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::blocking::reactor_queue::WaiterQueue;
use crate::dispatch::{dispatch_command_unlocked, ConnState};
use crate::engine::KvEngine;
use crate::fast_path::dispatch_span_batch_unlocked;
use crate::pubsub::{PubSubBroker, PushMessage};
use crate::resp::{RespBuffer, RespVal};
use crate::server::{encode_push, ServerConfig};

struct Client {
    stream: TcpStream,
    buf: RespBuffer,
    out: Vec<u8>,
    out_pos: usize,
    want_write: bool,
    conn: ConnState,
    rx: Receiver<PushMessage>,
}

impl Client {
    fn flush(&mut self) -> io::Result<bool> {
        while self.out_pos < self.out.len() {
            match self.stream.write(&self.out[self.out_pos..]) {
                Ok(0) => return Err(io::Error::new(io::ErrorKind::WriteZero, "eof")),
                Ok(n) => self.out_pos += n,
                Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => return Ok(false),
                Err(e) => return Err(e),
            }
        }
        self.out.clear();
        self.out_pos = 0;
        Ok(true)
    }
}

#[derive(Clone, Copy)]
struct ReadyEvent {
    fd: RawFd,
    readable: bool,
    writable: bool,
    eof: bool,
}

#[cfg(target_os = "macos")]
mod sys {
    use super::ReadyEvent;
    use std::ffi::c_void;
    use std::io;
    use std::os::fd::RawFd;

    const EVFILT_READ: i16 = -1;
    const EVFILT_WRITE: i16 = -2;
    const EV_ADD: u16 = 0x0001;
    const EV_DELETE: u16 = 0x0002;
    const EV_ENABLE: u16 = 0x0004;
    const EV_EOF: u16 = 0x8000;

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct KEvent {
        ident: usize,
        filter: i16,
        flags: u16,
        fflags: u32,
        data: isize,
        udata: *mut c_void,
    }

    #[repr(C)]
    struct Timespec {
        tv_sec: isize,
        tv_nsec: isize,
    }

    extern "C" {
        fn kqueue() -> i32;
        fn kevent(
            kq: i32,
            changelist: *const KEvent,
            nchanges: i32,
            eventlist: *mut KEvent,
            nevents: i32,
            timeout: *const Timespec,
        ) -> i32;
        fn close(fd: i32) -> i32;
    }

    pub struct Poller {
        kq: i32,
        raw: [KEvent; 256],
    }

    impl Poller {
        pub fn new() -> io::Result<Self> {
            let kq = unsafe { kqueue() };
            if kq < 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(Self {
                kq,
                raw: [KEvent::default(); 256],
            })
        }

        pub fn add_read(&self, fd: RawFd) {
            let ev = KEvent {
                ident: fd as usize,
                filter: EVFILT_READ,
                flags: EV_ADD | EV_ENABLE,
                ..KEvent::default()
            };
            unsafe {
                kevent(self.kq, &ev, 1, std::ptr::null_mut(), 0, std::ptr::null());
            }
        }

        pub fn set_write(&self, fd: RawFd, enable: bool) {
            let ev = KEvent {
                ident: fd as usize,
                filter: EVFILT_WRITE,
                flags: if enable { EV_ADD | EV_ENABLE } else { EV_DELETE },
                ..KEvent::default()
            };
            unsafe {
                kevent(self.kq, &ev, 1, std::ptr::null_mut(), 0, std::ptr::null());
            }
        }

        pub fn remove(&self, fd: RawFd) {
            let evs = [
                KEvent {
                    ident: fd as usize,
                    filter: EVFILT_READ,
                    flags: EV_DELETE,
                    ..KEvent::default()
                },
                KEvent {
                    ident: fd as usize,
                    filter: EVFILT_WRITE,
                    flags: EV_DELETE,
                    ..KEvent::default()
                },
            ];
            unsafe {
                kevent(
                    self.kq,
                    evs.as_ptr(),
                    2,
                    std::ptr::null_mut(),
                    0,
                    std::ptr::null(),
                );
            }
        }

        pub fn wait(&mut self, timeout_ms: i32, out: &mut Vec<ReadyEvent>) -> io::Result<()> {
            out.clear();
            let ts = Timespec {
                tv_sec: (timeout_ms / 1000) as isize,
                tv_nsec: ((timeout_ms % 1000) as isize) * 1_000_000,
            };
            let n = unsafe {
                kevent(
                    self.kq,
                    std::ptr::null(),
                    0,
                    self.raw.as_mut_ptr(),
                    self.raw.len() as i32,
                    &ts,
                )
            };
            if n < 0 {
                let err = io::Error::last_os_error();
                if err.kind() == io::ErrorKind::Interrupted {
                    return Ok(());
                }
                return Err(err);
            }
            for ev in &self.raw[..n as usize] {
                out.push(ReadyEvent {
                    fd: ev.ident as RawFd,
                    readable: ev.filter == EVFILT_READ,
                    writable: ev.filter == EVFILT_WRITE,
                    eof: (ev.flags & EV_EOF) != 0,
                });
            }
            Ok(())
        }
    }

    impl Drop for Poller {
        fn drop(&mut self) {
            unsafe {
                close(self.kq);
            }
        }
    }
}

#[cfg(target_os = "linux")]
mod sys {
    use super::ReadyEvent;
    use std::io;
    use std::os::fd::RawFd;

    const EPOLL_CTL_ADD: i32 = 1;
    const EPOLL_CTL_DEL: i32 = 2;
    const EPOLL_CTL_MOD: i32 = 3;
    const EPOLLIN: u32 = 0x001;
    const EPOLLOUT: u32 = 0x004;
    const EPOLLERR: u32 = 0x008;
    const EPOLLHUP: u32 = 0x010;

    #[cfg(target_arch = "x86_64")]
    #[repr(C, packed)]
    #[derive(Clone, Copy, Default)]
    struct EpollEvent {
        events: u32,
        data: u64,
    }

    #[cfg(not(target_arch = "x86_64"))]
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct EpollEvent {
        events: u32,
        data: u64,
    }

    extern "C" {
        fn epoll_create1(flags: i32) -> i32;
        fn epoll_ctl(epfd: i32, op: i32, fd: i32, event: *mut EpollEvent) -> i32;
        fn epoll_wait(epfd: i32, events: *mut EpollEvent, maxevents: i32, timeout: i32) -> i32;
        fn close(fd: i32) -> i32;
    }

    pub struct Poller {
        ep: i32,
        raw: [EpollEvent; 256],
    }

    impl Poller {
        pub fn new() -> io::Result<Self> {
            let ep = unsafe { epoll_create1(0) };
            if ep < 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(Self {
                ep,
                raw: [EpollEvent::default(); 256],
            })
        }

        pub fn add_read(&self, fd: RawFd) {
            let mut ev = EpollEvent {
                events: EPOLLIN,
                data: fd as u64,
            };
            unsafe {
                epoll_ctl(self.ep, EPOLL_CTL_ADD, fd, &mut ev);
            }
        }

        pub fn set_write(&self, fd: RawFd, enable: bool) {
            let mut ev = EpollEvent {
                events: if enable { EPOLLIN | EPOLLOUT } else { EPOLLIN },
                data: fd as u64,
            };
            unsafe {
                epoll_ctl(self.ep, EPOLL_CTL_MOD, fd, &mut ev);
            }
        }

        pub fn remove(&self, fd: RawFd) {
            unsafe {
                epoll_ctl(self.ep, EPOLL_CTL_DEL, fd, std::ptr::null_mut());
            }
        }

        pub fn wait(&mut self, timeout_ms: i32, out: &mut Vec<ReadyEvent>) -> io::Result<()> {
            out.clear();
            let n = unsafe {
                epoll_wait(
                    self.ep,
                    self.raw.as_mut_ptr(),
                    self.raw.len() as i32,
                    timeout_ms,
                )
            };
            if n < 0 {
                let err = io::Error::last_os_error();
                if err.kind() == io::ErrorKind::Interrupted {
                    return Ok(());
                }
                return Err(err);
            }
            for ev in &self.raw[..n as usize] {
                let flags = ev.events;
                let fd = ev.data as RawFd;
                out.push(ReadyEvent {
                    fd,
                    readable: (flags & EPOLLIN) != 0,
                    writable: (flags & EPOLLOUT) != 0,
                    eof: (flags & (EPOLLHUP | EPOLLERR)) != 0,
                });
            }
            Ok(())
        }
    }

    impl Drop for Poller {
        fn drop(&mut self) {
            unsafe {
                close(self.ep);
            }
        }
    }
}

/// Run the single-threaded zero-lock `kqueue`/`epoll` event loop until `shutdown` is set.
pub fn run_reactor(
    listener: TcpListener,
    mut engine: KvEngine,
    broker: Arc<PubSubBroker>,
    shutdown: Arc<AtomicBool>,
    cfg: ServerConfig,
) {
    let _ = listener.set_nonblocking(true);
    let Ok(mut poller) = sys::Poller::new() else {
        return;
    };
    let listen_fd = listener.as_raw_fd();
    poller.add_read(listen_fd);

    let mut clients: HashMap<RawFd, Client> = HashMap::with_capacity(64);
    let mut events: Vec<ReadyEvent> = Vec::with_capacity(256);
    let mut arg_spans: Vec<(usize, usize)> = Vec::with_capacity(256);
    let mut cmd_ranges: Vec<(usize, usize)> = Vec::with_capacity(128);
    let mut to_close: Vec<RawFd> = Vec::with_capacity(16);
    let mut waiters = WaiterQueue::default();

    let mut last_flush = Instant::now();
    let mut last_sync = Instant::now();
    let has_bg = cfg.flush_ms > 0 || cfg.sync_ms > 0;
    let poll_ms: i32 = if has_bg {
        cfg.flush_ms.min(cfg.sync_ms.max(1)).clamp(5, 25) as i32
    } else {
        25
    };

    while !shutdown.load(Ordering::Relaxed) {
        let wait_ms = if waiters.is_empty() { poll_ms } else { poll_ms.min(10) };
        if poller.wait(wait_ms, &mut events).is_err() {
            break;
        }
        let mut any_publish = false;
        to_close.clear();

        for ev in &events {
            if ev.fd == listen_fd {
                loop {
                    match listener.accept() {
                        Ok((stream, _)) => {
                            let _ = stream.set_nonblocking(true);
                            let _ = stream.set_nodelay(true);
                            let fd = stream.as_raw_fd();
                            let (tx, rx) = mpsc::channel();
                            let sub_id = broker.register(tx);
                            poller.add_read(fd);
                            clients.insert(
                                fd,
                                Client {
                                    stream,
                                    buf: RespBuffer::new(),
                                    out: Vec::with_capacity(16_384),
                                    out_pos: 0,
                                    want_write: false,
                                    conn: ConnState::new_with_auth(
                                        sub_id,
                                        engine.cfg.password.is_none(),
                                    ),
                                    rx,
                                },
                            );
                        }
                        Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => break,
                        Err(_) => break,
                    }
                }
                continue;
            }

            let Some(cl) = clients.get_mut(&ev.fd) else {
                continue;
            };
            let mut closed = false;

            if ev.readable {
                loop {
                    match cl.buf.read_more(&mut cl.stream) {
                        Ok(0) => {
                            closed = true;
                            break;
                        }
                        Ok(_) => {}
                        Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => break,
                        Err(_) => {
                            closed = true;
                            break;
                        }
                    }
                }
                if !cl.buf.is_empty() {
                    match cl.buf.parse_batch_spans(&mut arg_spans, &mut cmd_ranges) {
                        Ok(true) => {
                            if !cmd_ranges.is_empty() {
                                let raw = cl.buf.raw_buf();
                                let mut has_blocking = false;
                                for &(a, b) in &cmd_ranges {
                                    if a < b {
                                        let (s, e) = arg_spans[a];
                                        let c = &raw[s..e];
                                        if c.len() == 7 && c.eq_ignore_ascii_case(b"PUBLISH") {
                                            any_publish = true;
                                        } else if c.len() == 5
                                            && (c.eq_ignore_ascii_case(b"BLPOP")
                                                || c.eq_ignore_ascii_case(b"BRPOP"))
                                        {
                                            has_blocking = true;
                                        }
                                    }
                                }
                                if has_blocking {
                                    waiters.dispatch_blocking_batch(
                                        ev.fd, &mut engine, &broker, &mut cl.conn,
                                        raw, &arg_spans, &cmd_ranges, &mut cl.out,
                                    );
                                } else {
                                    dispatch_span_batch_unlocked(
                                        &mut engine, &broker, &mut cl.conn,
                                        raw, &arg_spans, &cmd_ranges, &mut cl.out,
                                    );
                                }
                            }
                        }
                        Ok(false) => {
                            if !cmd_ranges.is_empty() {
                                dispatch_span_batch_unlocked(
                                    &mut engine,
                                    &broker,
                                    &mut cl.conn,
                                    cl.buf.raw_buf(),
                                    &arg_spans,
                                    &cmd_ranges,
                                    &mut cl.out,
                                );
                            }
                            loop {
                                match cl.buf.next_command() {
                                    Ok(Some(args)) => {
                                        if args.first().is_some_and(|c| c.eq_ignore_ascii_case(b"PUBLISH")) {
                                            any_publish = true;
                                        }
                                        if !waiters.try_handle_blocking(
                                            ev.fd, &mut engine, &cl.conn, &args, &mut cl.out,
                                        ) {
                                            for reply in
                                                dispatch_command_unlocked(&mut engine, &broker, &mut cl.conn, args)
                                            {
                                                reply.encode(&mut cl.out);
                                            }
                                        }
                                        if cl.conn.quit {
                                            break;
                                        }
                                    }
                                    Ok(None) => break,
                                    Err(err) => {
                                        RespVal::err(err).encode(&mut cl.out);
                                        break;
                                    }
                                }
                            }
                        }
                        Err(err) => {
                            RespVal::err(err).encode(&mut cl.out);
                        }
                    }
                }
            }

            if !cl.out.is_empty() && (ev.readable || ev.writable) {
                match cl.flush() {
                    Ok(done) => {
                        if !done && !cl.want_write {
                            poller.set_write(ev.fd, true);
                            cl.want_write = true;
                        } else if done && cl.want_write {
                            poller.set_write(ev.fd, false);
                            cl.want_write = false;
                        }
                    }
                    Err(_) => closed = true,
                }
            }

            if closed || cl.conn.quit || (ev.eof && cl.buf.is_empty() && cl.out.is_empty()) {
                to_close.push(ev.fd);
            }
        }

        // Periodic background CAS flush & cross-pod pull (zero lock contention!)
        let mut pulled = false;
        if cfg.flush_ms > 0
            && engine.has_dirty()
            && last_flush.elapsed() >= Duration::from_millis(cfg.flush_ms)
        {
            let _ = engine.commit();
            if engine.cfg.auto_compact_wals > 0
                && engine.active_wal_count() >= engine.cfg.auto_compact_wals
            {
                let _ = engine.compact();
            }
            last_flush = Instant::now();
        }
        if cfg.sync_ms > 0 && last_sync.elapsed() >= Duration::from_millis(cfg.sync_ms) {
            if matches!(engine.pull(), Ok(n) if n > 0) {
                pulled = true;
            }
            last_sync = Instant::now();
        }

        // Deliver any pending Pub/Sub messages to subscribed clients immediately
        if any_publish || pulled || events.is_empty() {
            for (&fd, cl) in &mut clients {
                let mut pushed = false;
                while let Ok(pm) = cl.rx.try_recv() {
                    encode_push(pm).encode(&mut cl.out);
                    pushed = true;
                }
                if pushed {
                    match cl.flush() {
                        Ok(false) if !cl.want_write => {
                            poller.set_write(fd, true);
                            cl.want_write = true;
                        }
                        Err(_) => to_close.push(fd),
                        _ => {}
                    }
                }
            }
        }

        // Wake any blocked BLPOP/BRPOP clients whose list now has data or whose timeout expired
        if !waiters.is_empty() {
            for (wfd, reply) in waiters.poll_ready(&mut engine) {
                if let Some(cl) = clients.get_mut(&wfd) {
                    reply.encode(&mut cl.out);
                    match cl.flush() {
                        Ok(false) if !cl.want_write => {
                            poller.set_write(wfd, true);
                            cl.want_write = true;
                        }
                        Err(_) => to_close.push(wfd),
                        _ => {}
                    }
                }
            }
        }

        for fd in &to_close {
            waiters.remove_fd(*fd);
            poller.remove(*fd);
            if let Some(cl) = clients.remove(fd) {
                broker.unregister(cl.conn.sub_id);
            }
        }
    }

    for (_, cl) in clients {
        broker.unregister(cl.conn.sub_id);
    }
}
