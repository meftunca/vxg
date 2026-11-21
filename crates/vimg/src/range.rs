//! `RangeSource` abstraction (`Slice`, `File`, and pure-std `HTTP/1.1 Range` client) +
//! gap-coalescing byte-range scheduler inspired by VXG's `HybridArena`.

use crate::{Error, Result};

/// Exact I/O accounting returned alongside every decode or zero-decode crop.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RangeStats {
    pub requests: usize,
    pub bytes_fetched: u64,
    pub total_file_bytes: u64,
}

impl RangeStats {
    pub fn fetched_pct(&self) -> f64 {
        if self.total_file_bytes == 0 {
            0.0
        } else {
            (self.bytes_fetched as f64) * 100.0 / (self.total_file_bytes as f64)
        }
    }
}

/// Any local or remote byte-addressable source supporting half-open `[start, end)` range reads.
pub trait RangeSource {
    fn len(&self) -> u64;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Optimal initial prefix size for 1-RTT header+directory fetch.
    fn preferred_prefix_bytes(&self) -> u64 {
        4096
    }
    fn read_ranges(&self, ranges: &[(u64, u64)]) -> Result<Vec<Vec<u8>>>;
}

/// Coalesces an arbitrary list of `[start, end)` byte ranges into sorted, non-overlapping spans
/// merging gaps `<= max_gap` bytes. Returns `(coalesced_spans, mapping)` where `mapping[i]` is
/// `(coalesced_idx, relative_offset_within_coalesced_span)` for input `ranges[i]`.
#[allow(clippy::type_complexity)]
pub fn coalesce_ranges(ranges: &[(u64, u64)], max_gap: u64) -> (Vec<(u64, u64)>, Vec<(usize, usize)>) {
    if ranges.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let mut order: Vec<usize> = (0..ranges.len()).collect();
    order.sort_unstable_by_key(|&i| (ranges[i].0, ranges[i].1));

    let mut coalesced: Vec<(u64, u64)> = Vec::new();
    let mut mapping = vec![(0usize, 0usize); ranges.len()];

    for idx in order {
        let (s, e) = ranges[idx];
        let last_idx = coalesced.len().checked_sub(1);
        if let (Some(ci), Some(last)) = (last_idx, coalesced.last_mut()) {
            if s <= last.1.saturating_add(max_gap) {
                let rel = (s - last.0) as usize;
                if e > last.1 {
                    last.1 = e;
                }
                mapping[idx] = (ci, rel);
                continue;
            }
        }
        let ci = coalesced.len();
        coalesced.push((s, e));
        mapping[idx] = (ci, 0);
    }

    (coalesced, mapping)
}

/// In-memory `&[u8]` range source (zero-copy friendly, works on `wasm32`).
pub struct SliceRangeSource<'a>(pub &'a [u8]);

impl<'a> RangeSource for SliceRangeSource<'a> {
    fn len(&self) -> u64 {
        self.0.len() as u64
    }

    fn read_ranges(&self, ranges: &[(u64, u64)]) -> Result<Vec<Vec<u8>>> {
        let mut out = Vec::with_capacity(ranges.len());
        for &(s, e) in ranges {
            let (s, e) = (s as usize, e as usize);
            if s > e || e > self.0.len() {
                return Err(Error::Truncated);
            }
            out.push(self.0[s..e].to_vec());
        }
        Ok(out)
    }
}

/// Sparse in-memory byte-range source backed by pre-fetched `(offset, bytes)` spans.
/// Ideal for browser `fetch(..., { headers: { Range } })` + `vimg.wasm` partial decoding.
pub struct SparseRangeSource<'a> {
    pub total_len: u64,
    pub spans: Vec<(u64, &'a [u8])>,
}

impl<'a> RangeSource for SparseRangeSource<'a> {
    fn len(&self) -> u64 {
        self.total_len
    }

    fn read_ranges(&self, ranges: &[(u64, u64)]) -> Result<Vec<Vec<u8>>> {
        let mut out = Vec::with_capacity(ranges.len());
        for &(s, e) in ranges {
            if s > e || e > self.total_len {
                return Err(Error::Truncated);
            }
            let found = self.spans.iter().find_map(|&(off, data)| {
                let span_end = off + data.len() as u64;
                if s >= off && e <= span_end {
                    let lo = (s - off) as usize;
                    let hi = (e - off) as usize;
                    Some(data[lo..hi].to_vec())
                } else {
                    None
                }
            });
            out.push(found.ok_or(Error::Truncated)?);
        }
        Ok(out)
    }
}

/// Local file `pread` / `seek+read` range source (non-wasm targets).
#[cfg(not(target_arch = "wasm32"))]
pub struct FileRangeSource {
    file: std::fs::File,
    len: u64,
}

#[cfg(not(target_arch = "wasm32"))]
impl FileRangeSource {
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let file = std::fs::File::open(path).map_err(|e| Error::Io(e.to_string()))?;
        let len = file.metadata().map_err(|e| Error::Io(e.to_string()))?.len();
        Ok(Self { file, len })
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl RangeSource for FileRangeSource {
    fn len(&self) -> u64 {
        self.len
    }

    fn read_ranges(&self, ranges: &[(u64, u64)]) -> Result<Vec<Vec<u8>>> {
        #[cfg(unix)]
        {
            use std::os::unix::fs::FileExt;
            let mut out = Vec::with_capacity(ranges.len());
            for &(s, e) in ranges {
                if s > e || e > self.len {
                    return Err(Error::Truncated);
                }
                let mut buf = vec![0u8; (e - s) as usize];
                self.file.read_exact_at(&mut buf, s).map_err(|err| Error::Io(err.to_string()))?;
                out.push(buf);
            }
            Ok(out)
        }
        #[cfg(not(unix))]
        {
            use std::io::{Read, Seek, SeekFrom};
            let mut f = self.file.try_clone().map_err(|err| Error::Io(err.to_string()))?;
            let mut out = Vec::with_capacity(ranges.len());
            for &(s, e) in ranges {
                if s > e || e > self.len {
                    return Err(Error::Truncated);
                }
                let mut buf = vec![0u8; (e - s) as usize];
                f.seek(SeekFrom::Start(s)).map_err(|err| Error::Io(err.to_string()))?;
                f.read_exact(&mut buf).map_err(|err| Error::Io(err.to_string()))?;
                out.push(buf);
            }
            Ok(out)
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
enum HttpStream {
    Plain(std::net::TcpStream),
    #[cfg(feature = "tls")]
    Tls(Box<rustls::StreamOwned<rustls::ClientConnection, std::net::TcpStream>>),
}

#[cfg(not(target_arch = "wasm32"))]
impl std::io::Read for HttpStream {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self {
            Self::Plain(s) => s.read(buf),
            #[cfg(feature = "tls")]
            Self::Tls(s) => s.read(buf),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl std::io::Write for HttpStream {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match self {
            Self::Plain(s) => s.write(buf),
            #[cfg(feature = "tls")]
            Self::Tls(s) => s.write(buf),
        }
    }
    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            Self::Plain(s) => s.flush(),
            #[cfg(feature = "tls")]
            Self::Tls(s) => s.flush(),
        }
    }
}

/// Pure-Rust production-hardened HTTP/HTTPS `Range: bytes=start-end` client:
/// - Zero external dependencies for `http://`; optional pure-Rust `rustls` (`tls` feature) for `https://`
/// - Connect/read/write timeouts (never hangs on dead origins)
/// - Automatic `301/302/307/308` redirect following including `http:// -> https://` (up to 5 hops)
/// - `Connection: keep-alive` persistent socket reuse with transparent reconnect
/// - Parallel multi-range fetching (up to 4 concurrent streams) for multi-tile queries
/// - Strict response size & header count caps against malicious origins.
#[cfg(not(target_arch = "wasm32"))]
pub struct HttpRangeSource {
    use_tls: bool,
    host_name: String,
    host_port: String,
    host_hdr: String,
    path: String,
    len: u64,
    timeout: std::time::Duration,
    pool: std::sync::Mutex<Option<std::io::BufReader<HttpStream>>>,
}

#[cfg(not(target_arch = "wasm32"))]
const MAX_HTTP_BODY_BYTES: usize = 128 * 1024 * 1024;
#[cfg(not(target_arch = "wasm32"))]
const MAX_HTTP_HEADERS: usize = 128;

#[cfg(all(not(target_arch = "wasm32"), feature = "tls"))]
fn tls_client_config() -> std::sync::Arc<rustls::ClientConfig> {
    static CFG: std::sync::OnceLock<std::sync::Arc<rustls::ClientConfig>> = std::sync::OnceLock::new();
    std::sync::Arc::clone(CFG.get_or_init(|| {
        let mut root_store = rustls::RootCertStore::empty();
        root_store.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
        let cfg = rustls::ClientConfig::builder()
            .with_root_certificates(root_store)
            .with_no_client_auth();
        std::sync::Arc::new(cfg)
    }))
}

#[cfg(not(target_arch = "wasm32"))]
impl HttpRangeSource {
    #[allow(clippy::type_complexity)]
    fn parse_http_url(url: &str) -> Result<(bool, String, String, String, String)> {
        let (use_tls, rest, default_port) = if let Some(r) = url.strip_prefix("https://") {
            #[cfg(not(feature = "tls"))]
            {
                let _ = r;
                return Err(Error::Unsupported("https:// requires the `tls` cargo feature"));
            }
            #[cfg(feature = "tls")]
            (true, r, 443u16)
        } else if let Some(r) = url.strip_prefix("http://") {
            (false, r, 80u16)
        } else {
            return Err(Error::Unsupported("HttpRangeSource supports http:// and https:// URLs"));
        };
        let (authority, path) = match rest.find('/') {
            Some(i) => (&rest[..i], &rest[i..]),
            None => (rest, "/"),
        };
        if authority.is_empty() {
            return Err(Error::InvalidInput("empty HTTP authority"));
        }
        let (host_name, host_port) = if let Some(bracket_end) = authority.rfind(']') {
            let h = authority[1..bracket_end].to_owned();
            let hp = if authority[bracket_end + 1..].starts_with(':') {
                authority.to_owned()
            } else {
                format!("{authority}:{default_port}")
            };
            (h, hp)
        } else if let Some((h, _)) = authority.rsplit_once(':') {
            (h.to_owned(), authority.to_owned())
        } else {
            (authority.to_owned(), format!("{authority}:{default_port}"))
        };
        Ok((use_tls, host_name, host_port, authority.to_owned(), path.to_owned()))
    }

    pub fn connect(url: &str) -> Result<Self> {
        Self::connect_with_timeout(url, std::time::Duration::from_secs(15))
    }

    pub fn connect_with_timeout(url: &str, timeout: std::time::Duration) -> Result<Self> {
        let mut cur_url = url.to_owned();
        for _ in 0..=5 {
            let (use_tls, host_name, host_port, host_hdr, path) = Self::parse_http_url(&cur_url)?;
            let mut conn: Option<std::io::BufReader<HttpStream>> = None;
            let (status, hdrs, _) = Self::exchange_on(
                &mut conn,
                use_tls,
                &host_name,
                &host_port,
                &host_hdr,
                "HEAD",
                &path,
                None,
                timeout,
            )?;
            if matches!(status, 301 | 302 | 307 | 308) {
                let loc = hdrs
                    .iter()
                    .find(|(k, _)| k.eq_ignore_ascii_case("location"))
                    .map(|(_, v)| v.trim().to_owned())
                    .ok_or_else(|| Error::Io(format!("HTTP {status} redirect missing Location")))?;
                let scheme = if use_tls { "https" } else { "http" };
                cur_url = if loc.starts_with("http://") || loc.starts_with("https://") {
                    loc
                } else if loc.starts_with('/') {
                    format!("{scheme}://{host_hdr}{loc}")
                } else {
                    return Err(Error::Unsupported("unsupported redirect target scheme"));
                };
                continue;
            }
            if status != 200 && status != 206 {
                return Err(Error::Io(format!("HTTP HEAD returned status {status}")));
            }
            let len = hdrs
                .iter()
                .find(|(k, _)| k.eq_ignore_ascii_case("content-length"))
                .and_then(|(_, v)| v.trim().parse::<u64>().ok())
                .ok_or_else(|| Error::Io("HTTP HEAD missing Content-Length".into()))?;
            return Ok(Self {
                use_tls,
                host_name,
                host_port,
                host_hdr,
                path,
                len,
                timeout,
                pool: std::sync::Mutex::new(conn),
            });
        }
        Err(Error::Io("too many HTTP redirects".into()))
    }

    fn open_stream(
        use_tls: bool,
        host_name: &str,
        host_port: &str,
        timeout: std::time::Duration,
    ) -> Result<std::io::BufReader<HttpStream>> {
        use std::net::{TcpStream, ToSocketAddrs};
        let mut addrs: Vec<_> = host_port
            .to_socket_addrs()
            .map_err(|e| Error::Io(e.to_string()))?
            .collect();
        if addrs.is_empty() {
            return Err(Error::Io("DNS resolved to 0 addresses".into()));
        }
        // Prefer IPv4 first to avoid broken IPv6 blackholes, falling back across every resolved address.
        addrs.sort_by_key(|a| !a.is_ipv4());
        let per_addr_timeout = timeout.min(std::time::Duration::from_secs(5));
        let mut last_err = None;
        let mut connected = None;
        for addr in &addrs {
            match TcpStream::connect_timeout(addr, per_addr_timeout) {
                Ok(s) => {
                    connected = Some(s);
                    break;
                }
                Err(e) => last_err = Some(e),
            }
        }
        let stream = connected.ok_or_else(|| {
            Error::Io(last_err.map_or_else(|| "connect failed".into(), |e| e.to_string()))
        })?;
        let _ = stream.set_nodelay(true);
        let _ = stream.set_read_timeout(Some(timeout));
        let _ = stream.set_write_timeout(Some(timeout));
        if use_tls {
            #[cfg(feature = "tls")]
            {
                let server_name = rustls::pki_types::ServerName::try_from(host_name.to_owned())
                    .map_err(|_| Error::InvalidInput("invalid TLS server name"))?;
                let conn = rustls::ClientConnection::new(tls_client_config(), server_name)
                    .map_err(|e| Error::Io(format!("TLS init failed: {e}")))?;
                return Ok(std::io::BufReader::new(HttpStream::Tls(Box::new(
                    rustls::StreamOwned::new(conn, stream),
                ))));
            }
            #[cfg(not(feature = "tls"))]
            {
                let _ = host_name;
                return Err(Error::Unsupported("https:// requires the `tls` cargo feature"));
            }
        }
        Ok(std::io::BufReader::new(HttpStream::Plain(stream)))
    }

    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn exchange_on(
        slot: &mut Option<std::io::BufReader<HttpStream>>,
        use_tls: bool,
        host_name: &str,
        host_port: &str,
        host_hdr: &str,
        method: &str,
        path: &str,
        range: Option<(u64, u64)>,
        timeout: std::time::Duration,
    ) -> Result<(u16, Vec<(String, String)>, Vec<u8>)> {
        use std::io::{BufRead, Read, Write};

        let mut req = format!("{method} {path} HTTP/1.1\r\nHost: {host_hdr}\r\nUser-Agent: vimg/0.2\r\nConnection: keep-alive\r\n");
        if let Some((s, e)) = range {
            if e > s {
                req.push_str(&format!("Range: bytes={s}-{}\r\n", e - 1));
            }
        }
        req.push_str("\r\n");

        for attempt in 0..2 {
            let reused = slot.is_some();
            if slot.is_none() {
                *slot = Some(Self::open_stream(use_tls, host_name, host_port, timeout)?);
            }
            let reader = slot.as_mut().unwrap();
            if reader.get_mut().write_all(req.as_bytes()).is_err()
                || reader.get_mut().flush().is_err()
            {
                *slot = None;
                if reused && attempt == 0 {
                    continue;
                }
                return Err(Error::Io("HTTP write failed".into()));
            }

            let mut status_line = String::new();
            let n = reader.read_line(&mut status_line).unwrap_or(0);
            if n == 0 {
                *slot = None;
                if reused && attempt == 0 {
                    continue;
                }
                return Err(Error::Io("HTTP connection closed before status line".into()));
            }
            let status = status_line
                .split_whitespace()
                .nth(1)
                .and_then(|s| s.parse::<u16>().ok())
                .ok_or_else(|| Error::Io(format!("invalid HTTP status line: {status_line}")))?;

            let mut hdrs = Vec::new();
            let mut content_len: Option<usize> = None;
            let mut server_close = false;
            for _ in 0..MAX_HTTP_HEADERS {
                let mut line = String::new();
                reader.read_line(&mut line).map_err(|e| Error::Io(e.to_string()))?;
                let trimmed = line.trim_end_matches(['\r', '\n']);
                if trimmed.is_empty() {
                    break;
                }
                if let Some((k, v)) = trimmed.split_once(':') {
                    let vt = v.trim();
                    if k.eq_ignore_ascii_case("content-length") {
                        content_len = vt.parse().ok();
                    } else if k.eq_ignore_ascii_case("connection") && vt.eq_ignore_ascii_case("close") {
                        server_close = true;
                    }
                    hdrs.push((k.trim().to_owned(), vt.to_owned()));
                }
            }

            let mut body = Vec::new();
            if method != "HEAD" {
                if let Some(cl) = content_len {
                    if cl > MAX_HTTP_BODY_BYTES {
                        *slot = None;
                        return Err(Error::InvalidFormat("HTTP response exceeds safety cap"));
                    }
                    body.resize(cl, 0);
                    reader.read_exact(&mut body).map_err(|e| Error::Io(e.to_string()))?;
                } else {
                    server_close = true;
                    let mut limited = reader.by_ref().take((MAX_HTTP_BODY_BYTES + 1) as u64);
                    limited.read_to_end(&mut body).map_err(|e| Error::Io(e.to_string()))?;
                    if body.len() > MAX_HTTP_BODY_BYTES {
                        *slot = None;
                        return Err(Error::InvalidFormat("HTTP response exceeds safety cap"));
                    }
                }
            }
            if server_close {
                *slot = None;
            }
            return Ok((status, hdrs, body));
        }
        Err(Error::Io("HTTP exchange failed after retry".into()))
    }

    #[allow(clippy::too_many_arguments)]
    fn fetch_one_range(
        slot: &mut Option<std::io::BufReader<HttpStream>>,
        use_tls: bool,
        host_name: &str,
        host_port: &str,
        host_hdr: &str,
        path: &str,
        timeout: std::time::Duration,
        s: u64,
        e: u64,
    ) -> Result<Vec<u8>> {
        if s == e {
            return Ok(Vec::new());
        }
        let (status, _, body) = Self::exchange_on(
            slot,
            use_tls,
            host_name,
            host_port,
            host_hdr,
            "GET",
            path,
            Some((s, e)),
            timeout,
        )?;
        if status == 206 {
            if body.len() != (e - s) as usize {
                return Err(Error::Io(format!(
                    "HTTP 206 length mismatch: got {}, expected {}",
                    body.len(),
                    e - s
                )));
            }
            Ok(body)
        } else if status == 200 {
            let (su, eu) = (s as usize, e as usize);
            if eu > body.len() {
                return Err(Error::Truncated);
            }
            Ok(body[su..eu].to_vec())
        } else {
            Err(Error::Io(format!("HTTP Range GET failed with status {status}")))
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl RangeSource for HttpRangeSource {
    fn len(&self) -> u64 {
        self.len
    }

    fn preferred_prefix_bytes(&self) -> u64 {
        // Fetch 16 KiB in 1 RTT for >=256 KiB remote images (covers header + 3K/4K dir + ICC!).
        if self.len >= 256 * 1024 {
            16_384
        } else {
            4096
        }
    }

    fn read_ranges(&self, ranges: &[(u64, u64)]) -> Result<Vec<Vec<u8>>> {
        if ranges.len() <= 2 {
            let mut slot = self.pool.lock().unwrap();
            let mut out = Vec::with_capacity(ranges.len());
            for &(s, e) in ranges {
                out.push(Self::fetch_one_range(
                    &mut slot,
                    self.use_tls,
                    &self.host_name,
                    &self.host_port,
                    &self.host_hdr,
                    &self.path,
                    self.timeout,
                    s,
                    e,
                )?);
            }
            return Ok(out);
        }

        // Parallel pipelined Range fetching (up to 4 concurrent persistent connections).
        let n_workers = ranges.len().min(4);
        let chunk = ranges.len().div_ceil(n_workers);
        let mut parts: Vec<Result<Vec<Vec<u8>>>> = Vec::with_capacity(n_workers);
        std::thread::scope(|sc| {
            let handles: Vec<_> = ranges
                .chunks(chunk)
                .map(|sub| {
                    sc.spawn(move || {
                        let mut slot = None;
                        let mut local = Vec::with_capacity(sub.len());
                        for &(s, e) in sub {
                            local.push(Self::fetch_one_range(
                                &mut slot,
                                self.use_tls,
                                &self.host_name,
                                &self.host_port,
                                &self.host_hdr,
                                &self.path,
                                self.timeout,
                                s,
                                e,
                            )?);
                        }
                        Ok(local)
                    })
                })
                .collect();
            for h in handles {
                parts.push(h.join().expect("http range worker panicked"));
            }
        });
        let mut out = Vec::with_capacity(ranges.len());
        for p in parts {
            out.extend(p?);
        }
        Ok(out)
    }
}
