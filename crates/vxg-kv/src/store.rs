//! Pluggable `ObjectStore` abstraction with zero-dependency `LocalFsStore` and
//! `HttpObjectStore` (HTTP/1.1 `Range` + `ETag` Conditional PUT CAS).

use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use vxg::{Error as VxgError, RangeReader};

pub type StoreResult<T> = Result<T, String>;
type HttpHdrs = Vec<(String, String)>;
type HttpResp = (u16, HttpHdrs, Vec<u8>);
type HttpReadResp = (u16, HttpHdrs, Vec<u8>, bool);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObjectMeta {
    pub len: u64,
    pub etag: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PutOutcome {
    Written(String),
    PreconditionFailed,
}

/// Minimal S3/R2/HTTP-compatible Object Store trait supporting Range reads and CAS puts.
pub trait ObjectStore: Send + Sync {
    fn head(&self, key: &str) -> StoreResult<Option<ObjectMeta>>;
    fn get(&self, key: &str) -> StoreResult<Option<(Vec<u8>, String)>>;
    fn get_range(&self, key: &str, start: u64, end: u64) -> StoreResult<Vec<u8>>;
    fn put(&self, key: &str, data: &[u8]) -> StoreResult<String>;
    fn put_if_none_match(&self, key: &str, data: &[u8]) -> StoreResult<PutOutcome>;
    fn put_if_match(&self, key: &str, expected_etag: &str, data: &[u8]) -> StoreResult<PutOutcome>;
}

/// Compute a fast, deterministic 64-bit hex ETag (`fnv1a64` + length).
pub fn compute_etag(bytes: &[u8]) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("\"{h:016x}-{}\"", bytes.len())
}

/// Bridge any `Arc<dyn ObjectStore>` + object key directly into `vxg::RangeReader` / `vxg::RemoteFrame`.
pub struct StoreRangeReader {
    store: Arc<dyn ObjectStore>,
    key: String,
    total_len: u64,
}

impl StoreRangeReader {
    pub fn open(store: Arc<dyn ObjectStore>, key: impl Into<String>, total_len: u64) -> Self {
        Self {
            store,
            key: key.into(),
            total_len,
        }
    }
}

impl RangeReader for StoreRangeReader {
    fn len(&self) -> u64 {
        self.total_len
    }

    fn read_range(&mut self, start: u64, end: u64) -> vxg::Result<Vec<u8>> {
        self.store
            .get_range(&self.key, start, end)
            .map_err(VxgError::Backend)
    }
}

/// Local filesystem `ObjectStore` with atomic rename + CAS lock (ideal for local dev/shared NFS/tests).
pub struct LocalFsStore {
    root: PathBuf,
    cas_lock: Mutex<()>,
}

impl LocalFsStore {
    pub fn new(root: impl AsRef<Path>) -> StoreResult<Self> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(&root).map_err(|e| format!("create_dir {}: {e}", root.display()))?;
        Ok(Self {
            root,
            cas_lock: Mutex::new(()),
        })
    }

    fn resolve(&self, key: &str) -> StoreResult<PathBuf> {
        let rel = key.trim_start_matches('/');
        if rel.is_empty() || rel.contains("..") {
            return Err(format!("invalid object key: {key}"));
        }
        Ok(self.root.join(rel))
    }

    fn write_atomic(&self, path: &Path, data: &[u8]) -> StoreResult<String> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| format!("mkdir {}: {e}", parent.display()))?;
        }
        let tmp = path.with_extension(format!(
            "tmp.{}.{}",
            std::process::id(),
            crate::types::now_ms()
        ));
        fs::write(&tmp, data).map_err(|e| format!("write tmp: {e}"))?;
        fs::rename(&tmp, path).map_err(|e| {
            let _ = fs::remove_file(&tmp);
            format!("rename {}: {e}", path.display())
        })?;
        Ok(compute_etag(data))
    }
}

impl ObjectStore for LocalFsStore {
    fn head(&self, key: &str) -> StoreResult<Option<ObjectMeta>> {
        let p = self.resolve(key)?;
        if !p.is_file() {
            return Ok(None);
        }
        let data = fs::read(&p).map_err(|e| format!("read {}: {e}", p.display()))?;
        Ok(Some(ObjectMeta {
            len: data.len() as u64,
            etag: compute_etag(&data),
        }))
    }

    fn get(&self, key: &str) -> StoreResult<Option<(Vec<u8>, String)>> {
        let p = self.resolve(key)?;
        match fs::read(&p) {
            Ok(data) => {
                let etag = compute_etag(&data);
                Ok(Some((data, etag)))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(format!("read {}: {e}", p.display())),
        }
    }

    fn get_range(&self, key: &str, start: u64, end: u64) -> StoreResult<Vec<u8>> {
        if start >= end {
            return Ok(Vec::new());
        }
        let p = self.resolve(key)?;
        let mut f = fs::File::open(&p).map_err(|e| format!("open {}: {e}", p.display()))?;
        f.seek(SeekFrom::Start(start))
            .map_err(|e| format!("seek: {e}"))?;
        let mut buf = vec![0u8; (end - start) as usize];
        f.read_exact(&mut buf)
            .map_err(|e| format!("read_range {start}..{end}: {e}"))?;
        Ok(buf)
    }

    fn put(&self, key: &str, data: &[u8]) -> StoreResult<String> {
        let _g = self.cas_lock.lock().unwrap();
        let p = self.resolve(key)?;
        self.write_atomic(&p, data)
    }

    fn put_if_none_match(&self, key: &str, data: &[u8]) -> StoreResult<PutOutcome> {
        let _g = self.cas_lock.lock().unwrap();
        let p = self.resolve(key)?;
        if p.exists() {
            return Ok(PutOutcome::PreconditionFailed);
        }
        Ok(PutOutcome::Written(self.write_atomic(&p, data)?))
    }

    fn put_if_match(&self, key: &str, expected_etag: &str, data: &[u8]) -> StoreResult<PutOutcome> {
        let _g = self.cas_lock.lock().unwrap();
        let p = self.resolve(key)?;
        let Ok(cur) = fs::read(&p) else {
            return Ok(PutOutcome::PreconditionFailed);
        };
        if compute_etag(&cur) != expected_etag {
            return Ok(PutOutcome::PreconditionFailed);
        }
        Ok(PutOutcome::Written(self.write_atomic(&p, data)?))
    }
}

/// Zero-dependency HTTP/1.1 `ObjectStore` speaking `HEAD`, `GET Range`, and `PUT` with
/// `If-Match` / `If-None-Match` CAS (compatible with `vxg-kv origin`, MinIO, R2/S3 proxies).
pub struct HttpObjectStore {
    host: String,
    port: u16,
    base_path: String,
    pool: Mutex<Option<TcpStream>>,
}

impl HttpObjectStore {
    pub fn connect(base_url: &str) -> StoreResult<Self> {
        let rest = base_url
            .trim_end_matches('/')
            .strip_prefix("http://")
            .ok_or_else(|| "HttpObjectStore URL must start with http://".to_owned())?;
        let (host_port, base_path) = match rest.find('/') {
            Some(i) => (&rest[..i], rest[i..].to_owned()),
            None => (rest, String::new()),
        };
        let (host, port) = match host_port.rsplit_once(':') {
            Some((h, p)) if !h.contains(']') => (
                h.to_owned(),
                p.parse::<u16>().map_err(|_| "invalid HTTP port".to_owned())?,
            ),
            _ => (host_port.to_owned(), 80u16),
        };
        Ok(Self {
            host,
            port,
            base_path,
            pool: Mutex::new(None),
        })
    }

    fn full_path(&self, key: &str) -> String {
        let k = key.trim_start_matches('/');
        if self.base_path.is_empty() {
            format!("/{k}")
        } else {
            format!("{}/{k}", self.base_path.trim_end_matches('/'))
        }
    }

    fn open_tcp(&self) -> StoreResult<TcpStream> {
        let addrs: Vec<_> = (self.host.as_str(), self.port)
            .to_socket_addrs()
            .map_err(|e| format!("DNS {}: {e}", self.host))?
            .collect();
        let mut ordered = Vec::with_capacity(addrs.len());
        ordered.extend(addrs.iter().copied().filter(|a| a.is_ipv4()));
        ordered.extend(addrs.iter().copied().filter(|a| !a.is_ipv4()));
        for addr in ordered {
            if let Ok(s) = TcpStream::connect_timeout(&addr, Duration::from_secs(10)) {
                let _ = s.set_nodelay(true);
                let _ = s.set_read_timeout(Some(Duration::from_secs(15)));
                let _ = s.set_write_timeout(Some(Duration::from_secs(15)));
                return Ok(s);
            }
        }
        Err(format!("connect {}:{} failed", self.host, self.port))
    }

    fn request(
        &self,
        method: &str,
        key: &str,
        extra_hdrs: &[(&str, &str)],
        body: &[u8],
    ) -> StoreResult<HttpResp> {
        let path = self.full_path(key);
        let mut guard = self.pool.lock().unwrap();
        for attempt in 0..2 {
            if guard.is_none() {
                *guard = Some(self.open_tcp()?);
            }
            let mut req = format!(
                "{method} {path} HTTP/1.1\r\nHost: {}:{}\r\nConnection: keep-alive\r\n",
                self.host, self.port
            );
            for (k, v) in extra_hdrs {
                req.push_str(&format!("{k}: {v}\r\n"));
            }
            if !body.is_empty() || method == "PUT" {
                req.push_str(&format!("Content-Length: {}\r\n", body.len()));
            }
            req.push_str("\r\n");

            let s = guard.as_mut().unwrap();
            let write_ok = s.write_all(req.as_bytes()).is_ok()
                && (body.is_empty() || s.write_all(body).is_ok());
            if !write_ok {
                *guard = None;
                if attempt == 0 {
                    continue;
                }
                return Err("HTTP write failed".to_owned());
            }
            match read_http_resp(s, method == "HEAD") {
                Ok((st, hdrs, resp_body, keep)) => {
                    if !keep {
                        *guard = None;
                    }
                    return Ok((st, hdrs, resp_body));
                }
                Err(_) if attempt == 0 => {
                    *guard = None;
                    continue;
                }
                Err(e) => {
                    *guard = None;
                    return Err(e);
                }
            }
        }
        Err("HTTP request retry exhausted".to_owned())
    }
}

fn find_hdr<'a>(hdrs: &'a [(String, String)], name: &str) -> Option<&'a str> {
    hdrs.iter()
        .find(|(k, _)| k.eq_ignore_ascii_case(name))
        .map(|(_, v)| v.as_str())
}

fn read_http_resp(stream: &mut TcpStream, head_only: bool) -> StoreResult<HttpReadResp> {
    let mut raw = Vec::with_capacity(4096);
    let mut tmp = [0u8; 4096];
    let hdr_end = loop {
        let n = stream.read(&mut tmp).map_err(|e| format!("HTTP read: {e}"))?;
        if n == 0 {
            return Err("HTTP EOF".to_owned());
        }
        raw.extend_from_slice(&tmp[..n]);
        if let Some(pos) = raw.windows(4).position(|w| w == b"\r\n\r\n") {
            break pos + 4;
        }
        if raw.len() > 65_536 {
            return Err("HTTP header too large".to_owned());
        }
    };
    let hs = std::str::from_utf8(&raw[..hdr_end]).map_err(|_| "non-UTF8 HTTP hdr".to_owned())?;
    let mut lines = hs.lines();
    let status: u16 = lines
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| "bad HTTP status".to_owned())?;
    let mut hdrs = Vec::new();
    let mut cl: Option<usize> = None;
    let mut close = false;
    for l in lines {
        if let Some((k, v)) = l.split_once(':') {
            let kt = k.trim();
            let vt = v.trim();
            if kt.eq_ignore_ascii_case("content-length") {
                cl = vt.parse().ok();
            } else if kt.eq_ignore_ascii_case("connection") && vt.eq_ignore_ascii_case("close") {
                close = true;
            }
            hdrs.push((kt.to_owned(), vt.to_owned()));
        }
    }
    if head_only {
        return Ok((status, hdrs, Vec::new(), !close));
    }
    let mut body = raw[hdr_end..].to_vec();
    if let Some(len) = cl {
        if body.len() < len {
            let start = body.len();
            body.resize(len, 0);
            stream
                .read_exact(&mut body[start..])
                .map_err(|e| format!("HTTP body read: {e}"))?;
        } else {
            body.truncate(len);
        }
    } else {
        stream
            .read_to_end(&mut body)
            .map_err(|e| format!("HTTP body eof: {e}"))?;
        close = true;
    }
    Ok((status, hdrs, body, !close))
}

impl ObjectStore for HttpObjectStore {
    fn head(&self, key: &str) -> StoreResult<Option<ObjectMeta>> {
        let (st, hdrs, _) = self.request("HEAD", key, &[], &[])?;
        if st == 404 {
            return Ok(None);
        }
        if !(200..300).contains(&st) {
            return Err(format!("HEAD {key} failed: HTTP {st}"));
        }
        let len = find_hdr(&hdrs, "content-length")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let etag = find_hdr(&hdrs, "etag").unwrap_or("").to_owned();
        Ok(Some(ObjectMeta { len, etag }))
    }

    fn get(&self, key: &str) -> StoreResult<Option<(Vec<u8>, String)>> {
        let (st, hdrs, body) = self.request("GET", key, &[], &[])?;
        if st == 404 {
            return Ok(None);
        }
        if !(200..300).contains(&st) {
            return Err(format!("GET {key} failed: HTTP {st}"));
        }
        let etag = find_hdr(&hdrs, "etag")
            .map(str::to_owned)
            .unwrap_or_else(|| compute_etag(&body));
        Ok(Some((body, etag)))
    }

    fn get_range(&self, key: &str, start: u64, end: u64) -> StoreResult<Vec<u8>> {
        if start >= end {
            return Ok(Vec::new());
        }
        let rh = format!("bytes={start}-{}", end - 1);
        let (st, _, body) = self.request("GET", key, &[("Range", &rh)], &[])?;
        if st == 206 && body.len() as u64 == end - start {
            Ok(body)
        } else if st == 200 && body.len() as u64 >= end {
            Ok(body[start as usize..end as usize].to_vec())
        } else {
            Err(format!(
                "GET Range {key} {rh} failed: HTTP {st} ({} bytes)",
                body.len()
            ))
        }
    }

    fn put(&self, key: &str, data: &[u8]) -> StoreResult<String> {
        let (st, hdrs, _) = self.request("PUT", key, &[], data)?;
        if !(200..300).contains(&st) {
            return Err(format!("PUT {key} failed: HTTP {st}"));
        }
        Ok(find_hdr(&hdrs, "etag")
            .map(str::to_owned)
            .unwrap_or_else(|| compute_etag(data)))
    }

    fn put_if_none_match(&self, key: &str, data: &[u8]) -> StoreResult<PutOutcome> {
        let (st, hdrs, _) = self.request("PUT", key, &[("If-None-Match", "*")], data)?;
        if st == 412 {
            return Ok(PutOutcome::PreconditionFailed);
        }
        if !(200..300).contains(&st) {
            return Err(format!("PUT If-None-Match {key} failed: HTTP {st}"));
        }
        let etag = find_hdr(&hdrs, "etag")
            .map(str::to_owned)
            .unwrap_or_else(|| compute_etag(data));
        Ok(PutOutcome::Written(etag))
    }

    fn put_if_match(&self, key: &str, expected_etag: &str, data: &[u8]) -> StoreResult<PutOutcome> {
        let (st, hdrs, _) = self.request("PUT", key, &[("If-Match", expected_etag)], data)?;
        if st == 412 {
            return Ok(PutOutcome::PreconditionFailed);
        }
        if !(200..300).contains(&st) {
            return Err(format!("PUT If-Match {key} failed: HTTP {st}"));
        }
        let etag = find_hdr(&hdrs, "etag")
            .map(str::to_owned)
            .unwrap_or_else(|| compute_etag(data));
        Ok(PutOutcome::Written(etag))
    }
}
