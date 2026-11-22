//! Zero-dependency HTTP/1.1 `Range` client (`HttpRangeReader`) and local origin server
//! (`serve_http_range`, `serve_http_range_listener`) with live `?sql=...&fmt=...` queries.

use std::io::{self, BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::net::{TcpListener, TcpStream, ToSocketAddrs};
use std::path::Path;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::time::Duration;

use crate::remote::RangeReader;
use crate::sql::{execute_sql, SqlFormat, SqlQuery};
use crate::{DecodeOptions, Error, Result};

type HttpHeaders = Vec<(String, String)>;
type HttpResponse = (u16, HttpHeaders, Vec<u8>);
type HttpReadResponse = (u16, HttpHeaders, Vec<u8>, bool);

/// Zero-dependency HTTP/1.1 `Range` reader using persistent `Connection: keep-alive` TCP sockets.
pub struct HttpRangeReader {
    host: String,
    port: u16,
    path: String,
    total_len: u64,
    conn: Option<TcpStream>,
}

impl HttpRangeReader {
    /// Connect to an `http://host[:port]/path` URL and probe `Content-Length` via `HEAD`.
    pub fn connect(url: &str) -> Result<Self> {
        let rest = url
            .strip_prefix("http://")
            .ok_or(Error::InvalidOption("remote URL must begin with http://"))?;
        let (host_port, path) = match rest.find('/') {
            Some(i) => (&rest[..i], rest[i..].to_owned()),
            None => (rest, "/".to_owned()),
        };
        let (host, port) = match host_port.rsplit_once(':') {
            Some((h, p)) if !h.contains(']') => (
                h.to_owned(),
                p.parse::<u16>()
                    .map_err(|_| Error::InvalidOption("invalid port in http:// URL"))?,
            ),
            _ => (host_port.to_owned(), 80u16),
        };
        let mut r = Self {
            host,
            port,
            path,
            total_len: 0,
            conn: None,
        };
        r.total_len = r.probe_length()?;
        Ok(r)
    }

    fn open_tcp(&self) -> Result<TcpStream> {
        let addrs: Vec<_> = (self.host.as_str(), self.port)
            .to_socket_addrs()
            .map_err(|e| Error::Backend(format!("DNS {}: {e}", self.host)))?
            .collect();
        let mut ordered = Vec::with_capacity(addrs.len());
        ordered.extend(addrs.iter().copied().filter(|a| a.is_ipv4()));
        ordered.extend(addrs.iter().copied().filter(|a| !a.is_ipv4()));
        let mut last_err = None;
        for addr in ordered {
            match TcpStream::connect_timeout(&addr, Duration::from_secs(10)) {
                Ok(s) => {
                    let _ = s.set_nodelay(true);
                    let _ = s.set_read_timeout(Some(Duration::from_secs(15)));
                    let _ = s.set_write_timeout(Some(Duration::from_secs(15)));
                    return Ok(s);
                }
                Err(e) => last_err = Some(e),
            }
        }
        Err(Error::Backend(format!(
            "connect {}:{}: {}",
            self.host,
            self.port,
            last_err.map_or_else(|| "no addresses".into(), |e| e.to_string())
        )))
    }

    fn send_http(&mut self, method: &str, range_hdr: Option<&str>) -> Result<HttpResponse> {
        for attempt in 0..2 {
            if self.conn.is_none() {
                self.conn = Some(self.open_tcp()?);
            }
            let mut req = format!(
                "{method} {} HTTP/1.1\r\nHost: {}:{}\r\nConnection: keep-alive\r\n",
                self.path, self.host, self.port
            );
            if let Some(rh) = range_hdr {
                req.push_str(&format!("Range: {rh}\r\n"));
            }
            req.push_str("\r\n");

            let stream = self.conn.as_mut().unwrap();
            if stream.write_all(req.as_bytes()).is_err() {
                self.conn = None;
                if attempt == 0 {
                    continue;
                }
                return Err(Error::Backend("HTTP write failed".into()));
            }
            match read_http_response(stream, method == "HEAD") {
                Ok((status, hdrs, body, keep_alive)) => {
                    if !keep_alive {
                        self.conn = None;
                    }
                    return Ok((status, hdrs, body));
                }
                Err(_) if attempt == 0 => {
                    self.conn = None;
                    continue;
                }
                Err(e) => {
                    self.conn = None;
                    return Err(e);
                }
            }
        }
        Err(Error::Backend("HTTP request failed".into()))
    }

    fn probe_length(&mut self) -> Result<u64> {
        let (st, hdrs, _) = self.send_http("HEAD", None)?;
        if (200..300).contains(&st) {
            if let Some((_, v)) = hdrs
                .iter()
                .find(|(k, _)| k.eq_ignore_ascii_case("content-length"))
            {
                if let Ok(n) = v.trim().parse::<u64>() {
                    return Ok(n);
                }
            }
        }
        let (st2, hdrs2, _) = self.send_http("GET", Some("bytes=0-0"))?;
        if st2 == 206 {
            if let Some((_, v)) = hdrs2
                .iter()
                .find(|(k, _)| k.eq_ignore_ascii_case("content-range"))
            {
                if let Some((_, tot)) = v.rsplit_once('/') {
                    if let Ok(n) = tot.trim().parse::<u64>() {
                        return Ok(n);
                    }
                }
            }
        }
        Err(Error::Backend(format!(
            "HTTP origin did not report Content-Length (HEAD={st}, GET={st2})"
        )))
    }
}

fn read_http_response(stream: &mut TcpStream, head_only: bool) -> Result<HttpReadResponse> {
    let mut raw = Vec::with_capacity(4096);
    let mut tmp = [0u8; 4096];
    let hdr_end = loop {
        let n = stream
            .read(&mut tmp)
            .map_err(|e| Error::Backend(format!("HTTP read: {e}")))?;
        if n == 0 {
            return Err(Error::Truncated);
        }
        raw.extend_from_slice(&tmp[..n]);
        if let Some(pos) = raw.windows(4).position(|w| w == b"\r\n\r\n") {
            break pos + 4;
        }
        if raw.len() > 65_536 {
            return Err(Error::InvalidFrame("HTTP header too large"));
        }
    };
    let hdr_str = std::str::from_utf8(&raw[..hdr_end])
        .map_err(|_| Error::InvalidFrame("non-UTF8 HTTP response header"))?;
    let mut lines = hdr_str.lines();
    let status_line = lines.next().ok_or(Error::InvalidFrame("empty HTTP status"))?;
    let status: u16 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .ok_or(Error::InvalidFrame("bad HTTP status code"))?;
    let mut hdrs = Vec::new();
    let mut content_len: Option<usize> = None;
    let mut close = false;
    for l in lines {
        if let Some((k, v)) = l.split_once(':') {
            let kt = k.trim();
            let vt = v.trim();
            if kt.eq_ignore_ascii_case("content-length") {
                content_len = vt.parse().ok();
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
    if let Some(cl) = content_len {
        if body.len() < cl {
            let start = body.len();
            body.resize(cl, 0);
            stream
                .read_exact(&mut body[start..])
                .map_err(|e| Error::Backend(format!("HTTP body read: {e}")))?;
        } else {
            body.truncate(cl);
        }
    } else {
        stream
            .read_to_end(&mut body)
            .map_err(|e| Error::Backend(format!("HTTP body eof: {e}")))?;
        close = true;
    }
    Ok((status, hdrs, body, !close))
}

impl RangeReader for HttpRangeReader {
    fn len(&self) -> u64 {
        self.total_len
    }

    fn read_range(&mut self, start: u64, end: u64) -> Result<Vec<u8>> {
        if start >= end {
            return Ok(Vec::new());
        }
        let rh = format!("bytes={start}-{}", end - 1);
        let (st, _, body) = self.send_http("GET", Some(&rh))?;
        if st == 206 && body.len() as u64 == end - start {
            Ok(body)
        } else if st == 200 && (body.len() as u64) >= end {
            Ok(body[start as usize..end as usize].to_vec())
        } else {
            Err(Error::Backend(format!(
                "HTTP Range {rh} failed (status={st}, got {} bytes)",
                body.len()
            )))
        }
    }
}

/// Serve a local directory of `.vxg` archives on `127.0.0.1:<port>` with HTTP/1.1 `Range`
/// support and live `GET /file.vxg?sql=...&fmt=json|csv|tsv` compressed-domain SQL queries.
pub fn serve_http_range(dir: &Path, port: u16) -> Result<()> {
    let listener = TcpListener::bind(("127.0.0.1", port))
        .map_err(|e| Error::Backend(format!("bind 127.0.0.1:{port}: {e}")))?;
    println!(
        "vxg HTTP/1.1 Range + Compressed-SQL Server listening on http://127.0.0.1:{port} (dir={})",
        dir.display()
    );
    serve_http_range_listener(listener, dir)
}

/// Accept loop over an already-bound [`TcpListener`] (useful for ephemeral port `0` in tests).
pub fn serve_http_range_listener(listener: TcpListener, dir: &Path) -> Result<()> {
    let root = Arc::new(dir.to_path_buf());
    let active = Arc::new(AtomicUsize::new(0));
    for mut stream in listener.incoming().flatten() {
        if active.load(Ordering::Relaxed) >= 64 {
            let _ = stream.write_all(
                b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            );
            continue;
        }
        active.fetch_add(1, Ordering::AcqRel);
        let root_c = Arc::clone(&root);
        let act_c = Arc::clone(&active);
        std::thread::spawn(move || {
            handle_http_conn(stream, &root_c);
            act_c.fetch_sub(1, Ordering::AcqRel);
        });
    }
    Ok(())
}

fn handle_http_conn(mut stream: TcpStream, root: &Path) {
    let _ = stream.set_nodelay(true);
    let _ = stream.set_read_timeout(Some(Duration::from_secs(15)));
    let Ok(clone) = stream.try_clone() else {
        return;
    };
    let mut reader = BufReader::new(clone);
    loop {
        let mut req_line = String::new();
        match reader.read_line(&mut req_line) {
            Ok(0) | Err(_) => break,
            Ok(_) if req_line.len() > 4096 => break,
            Ok(_) => {}
        }
        let parts: Vec<&str> = req_line.split_whitespace().collect();
        if parts.len() < 2 {
            break;
        }
        let (method, target) = (parts[0], parts[1]);
        let mut range_hdr: Option<String> = None;
        let mut keep_alive = true;
        for _ in 0..64 {
            let mut line = String::new();
            if reader.read_line(&mut line).is_err() || line.trim().is_empty() {
                break;
            }
            if let Some((k, v)) = line.split_once(':') {
                if k.eq_ignore_ascii_case("range") {
                    range_hdr = Some(v.trim().to_owned());
                } else if k.eq_ignore_ascii_case("connection")
                    && v.trim().eq_ignore_ascii_case("close")
                {
                    keep_alive = false;
                }
            }
        }
        let (raw_path, qs) = target.split_once('?').unwrap_or((target, ""));
        let rel = raw_path.trim_start_matches('/');
        if rel.is_empty() || rel.contains("..") {
            let _ = stream.write_all(
                b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            );
            break;
        }
        let fp = root.join(rel);
        if !fp.is_file() {
            let _ = stream.write_all(
                b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            );
            break;
        }
        if !serve_file_request(&mut stream, &fp, method, qs, range_hdr.as_deref(), keep_alive) {
            break;
        }
        if !keep_alive {
            break;
        }
    }
}

fn serve_file_request(
    stream: &mut TcpStream,
    fp: &Path,
    method: &str,
    qs: &str,
    range_hdr: Option<&str>,
    keep_alive: bool,
) -> bool {
    let conn = if keep_alive { "keep-alive" } else { "close" };
    if !qs.is_empty() && fp.extension().and_then(|e| e.to_str()) == Some("vxg") {
        let mut sql_raw = None;
        let mut fmt = SqlFormat::Tsv;
        for pair in qs.split('&') {
            let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
            if k == "sql" {
                sql_raw = Some(pct_decode(v));
            } else if k == "fmt" {
                fmt = match v {
                    "csv" => SqlFormat::Csv,
                    "json" => SqlFormat::Json,
                    _ => SqlFormat::Tsv,
                };
            }
        }
        if let Some(sql_str) = sql_raw {
            let t0 = std::time::Instant::now();
            let res = SqlQuery::parse(&sql_str).and_then(|q| {
                let inp = crate::InputBytes::open_path_mode(fp, true)?;
                execute_sql(inp.as_slice(), &q, fmt, false, &DecodeOptions::default())
            });
            match res {
                Ok(body) => {
                    let ms = t0.elapsed().as_secs_f64() * 1000.0;
                    let mime = match fmt {
                        SqlFormat::Json => "application/x-ndjson",
                        SqlFormat::Csv => "text/csv",
                        SqlFormat::Tsv => "text/tab-separated-values",
                    };
                    let hdr = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: {mime}\r\nContent-Length: {}\r\nX-VXG-Query-Ms: {ms:.2}\r\nConnection: {conn}\r\n\r\n",
                        body.len()
                    );
                    return stream.write_all(hdr.as_bytes()).is_ok()
                        && stream.write_all(&body).is_ok();
                }
                Err(e) => {
                    let msg = e.to_string();
                    let hdr = format!(
                        "HTTP/1.1 400 Bad Request\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: {conn}\r\n\r\n",
                        msg.len()
                    );
                    return stream.write_all(hdr.as_bytes()).is_ok()
                        && stream.write_all(msg.as_bytes()).is_ok();
                }
            }
        }
    }
    let Ok(mut f) = std::fs::File::open(fp) else {
        let _ = stream
            .write_all(b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
        return false;
    };
    let total = f.metadata().map(|m| m.len()).unwrap_or(0);
    if method == "HEAD" {
        let hdr = format!(
            "HTTP/1.1 200 OK\r\nAccept-Ranges: bytes\r\nContent-Length: {total}\r\nConnection: {conn}\r\n\r\n"
        );
        return stream.write_all(hdr.as_bytes()).is_ok();
    }
    if let Some(r) = range_hdr.and_then(|s| s.strip_prefix("bytes=")) {
        if let Some((a, b)) = r.split_once('-') {
            if let (Ok(s), Ok(e)) = (a.parse::<u64>(), b.parse::<u64>()) {
                if s <= e && e < total && f.seek(SeekFrom::Start(s)).is_ok() {
                    let len = (e - s + 1) as usize;
                    let mut buf = vec![0u8; len];
                    if f.read_exact(&mut buf).is_ok() {
                        let hdr = format!(
                            "HTTP/1.1 206 Partial Content\r\nAccept-Ranges: bytes\r\nContent-Range: bytes {s}-{e}/{total}\r\nContent-Length: {len}\r\nConnection: {conn}\r\n\r\n"
                        );
                        return stream.write_all(hdr.as_bytes()).is_ok()
                            && stream.write_all(&buf).is_ok();
                    }
                }
            }
        }
    }
    let hdr = format!(
        "HTTP/1.1 200 OK\r\nAccept-Ranges: bytes\r\nContent-Length: {total}\r\nConnection: {conn}\r\n\r\n"
    );
    stream.write_all(hdr.as_bytes()).is_ok() && io::copy(&mut f, stream).is_ok()
}

fn pct_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'+' {
            out.push(b' ');
            i += 1;
        } else if b[i] == b'%' && i + 2 < b.len() {
            if let Ok(v) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
                out.push(v);
                i += 3;
                continue;
            }
            out.push(b[i]);
            i += 1;
        } else {
            out.push(b[i]);
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}
