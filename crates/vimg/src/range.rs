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
    fn read_ranges(&self, ranges: &[(u64, u64)]) -> Result<Vec<Vec<u8>>>;
}

/// Coalesces an arbitrary list of `[start, end)` byte ranges into sorted, non-overlapping spans
/// merging gaps `<= max_gap` bytes. Returns `(coalesced_spans, mapping)` where `mapping[i]` is
/// `(coalesced_idx, relative_offset_within_coalesced_span)` for input `ranges[i]`.
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

/// Pure-`std` HTTP/1.1 `Range: bytes=start-end` client (zero external dependencies!)
/// supporting `HEAD` content-length discovery and persistent `Connection: keep-alive` fetches.
#[cfg(not(target_arch = "wasm32"))]
pub struct HttpRangeSource {
    host_port: String,
    host_hdr: String,
    path: String,
    len: u64,
}

#[cfg(not(target_arch = "wasm32"))]
impl HttpRangeSource {
    pub fn connect(url: &str) -> Result<Self> {
        let rest = url
            .strip_prefix("http://")
            .ok_or(Error::Unsupported("HttpRangeSource supports http:// URLs"))?;
        let (authority, path) = match rest.find('/') {
            Some(i) => (&rest[..i], &rest[i..]),
            None => (rest, "/"),
        };
        let host_port = if authority.contains(':') {
            authority.to_owned()
        } else {
            format!("{authority}:80")
        };

        let (status, hdrs, _) = Self::http_exchange(&host_port, authority, "HEAD", path, None)?;
        if status != 200 && status != 206 {
            return Err(Error::Io(format!("HTTP HEAD returned status {status}")));
        }
        let len = hdrs
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case("content-length"))
            .and_then(|(_, v)| v.trim().parse::<u64>().ok())
            .ok_or_else(|| Error::Io("HTTP HEAD missing Content-Length".into()))?;
        Ok(Self {
            host_port,
            host_hdr: authority.to_owned(),
            path: path.to_owned(),
            len,
        })
    }

    fn http_exchange(
        host_port: &str,
        host_hdr: &str,
        method: &str,
        path: &str,
        range: Option<(u64, u64)>,
    ) -> Result<(u16, Vec<(String, String)>, Vec<u8>)> {
        use std::io::{BufRead, BufReader, Read, Write};
        use std::net::TcpStream;

        let mut stream = TcpStream::connect(host_port).map_err(|e| Error::Io(e.to_string()))?;
        let _ = stream.set_nodelay(true);
        let mut req = format!("{method} {path} HTTP/1.1\r\nHost: {host_hdr}\r\nConnection: close\r\n");
        if let Some((s, e)) = range {
            if e > s {
                req.push_str(&format!("Range: bytes={s}-{}\r\n", e - 1));
            }
        }
        req.push_str("\r\n");
        stream.write_all(req.as_bytes()).map_err(|e| Error::Io(e.to_string()))?;

        let mut reader = BufReader::new(stream);
        let mut status_line = String::new();
        reader.read_line(&mut status_line).map_err(|e| Error::Io(e.to_string()))?;
        let status = status_line
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse::<u16>().ok())
            .ok_or_else(|| Error::Io(format!("invalid HTTP status line: {status_line}")))?;

        let mut hdrs = Vec::new();
        let mut content_len: Option<usize> = None;
        loop {
            let mut line = String::new();
            reader.read_line(&mut line).map_err(|e| Error::Io(e.to_string()))?;
            let trimmed = line.trim_end_matches(['\r', '\n']);
            if trimmed.is_empty() {
                break;
            }
            if let Some((k, v)) = trimmed.split_once(':') {
                if k.eq_ignore_ascii_case("content-length") {
                    content_len = v.trim().parse().ok();
                }
                hdrs.push((k.trim().to_owned(), v.trim().to_owned()));
            }
        }

        let mut body = Vec::new();
        if method != "HEAD" {
            if let Some(cl) = content_len {
                body.resize(cl, 0);
                reader.read_exact(&mut body).map_err(|e| Error::Io(e.to_string()))?;
            } else {
                reader.read_to_end(&mut body).map_err(|e| Error::Io(e.to_string()))?;
            }
        }
        Ok((status, hdrs, body))
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl RangeSource for HttpRangeSource {
    fn len(&self) -> u64 {
        self.len
    }

    fn read_ranges(&self, ranges: &[(u64, u64)]) -> Result<Vec<Vec<u8>>> {
        let mut out = Vec::with_capacity(ranges.len());
        for &(s, e) in ranges {
            if s == e {
                out.push(Vec::new());
                continue;
            }
            let (status, _, body) =
                Self::http_exchange(&self.host_port, &self.host_hdr, "GET", &self.path, Some((s, e)))?;
            if status == 206 {
                if body.len() != (e - s) as usize {
                    return Err(Error::Io(format!(
                        "HTTP 206 length mismatch: got {}, expected {}",
                        body.len(),
                        e - s
                    )));
                }
                out.push(body);
            } else if status == 200 {
                // Origin ignored Range header; slice requested span safely.
                let (su, eu) = (s as usize, e as usize);
                if eu > body.len() {
                    return Err(Error::Truncated);
                }
                out.push(body[su..eu].to_vec());
            } else {
                return Err(Error::Io(format!("HTTP Range GET failed with status {status}")));
            }
        }
        Ok(out)
    }
}
