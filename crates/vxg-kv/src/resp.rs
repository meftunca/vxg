//! Zero-dependency Redis Serialization Protocol (`RESP2` + inline commands) parser & encoder.
//!
//! Supports full pipelining (multiple commands buffered in one read), binary-safe bulk strings,
//! and standard `redis-cli` / `ioredis` / `redis-py` / `go-redis` wire compatibility.

use std::io::{self, Read, Write};

/// A RESP2/RESP3 value sent over the wire.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RespVal {
    Simple(String),
    Error(String),
    Integer(i64),
    Bulk(Option<Vec<u8>>),
    Array(Option<Vec<RespVal>>),
    Map(Vec<(RespVal, RespVal)>),
}

impl RespVal {
    pub fn ok() -> Self {
        Self::Simple("OK".to_owned())
    }
    pub fn pong() -> Self {
        Self::Simple("PONG".to_owned())
    }
    pub fn nil() -> Self {
        Self::Bulk(None)
    }
    pub fn nil_array() -> Self {
        Self::Array(None)
    }
    pub fn err(msg: impl Into<String>) -> Self {
        Self::Error(msg.into())
    }
    pub fn bulk(b: impl Into<Vec<u8>>) -> Self {
        Self::Bulk(Some(b.into()))
    }
    pub fn int(n: i64) -> Self {
        Self::Integer(n)
    }
    pub fn array(items: Vec<RespVal>) -> Self {
        Self::Array(Some(items))
    }
    pub fn map(pairs: Vec<(RespVal, RespVal)>) -> Self {
        Self::Map(pairs)
    }

    pub fn encode(&self, out: &mut Vec<u8>) {
        match self {
            Self::Simple(s) => {
                out.push(b'+');
                out.extend_from_slice(s.as_bytes());
                out.extend_from_slice(b"\r\n");
            }
            Self::Error(s) => {
                out.push(b'-');
                out.extend_from_slice(s.as_bytes());
                out.extend_from_slice(b"\r\n");
            }
            Self::Integer(n) => {
                out.push(b':');
                out.extend_from_slice(n.to_string().as_bytes());
                out.extend_from_slice(b"\r\n");
            }
            Self::Bulk(None) => out.extend_from_slice(b"$-1\r\n"),
            Self::Bulk(Some(b)) => {
                out.push(b'$');
                out.extend_from_slice(b.len().to_string().as_bytes());
                out.extend_from_slice(b"\r\n");
                out.extend_from_slice(b);
                out.extend_from_slice(b"\r\n");
            }
            Self::Array(None) => out.extend_from_slice(b"*-1\r\n"),
            Self::Array(Some(items)) => {
                out.push(b'*');
                out.extend_from_slice(items.len().to_string().as_bytes());
                out.extend_from_slice(b"\r\n");
                for it in items {
                    it.encode(out);
                }
            }
            Self::Map(pairs) => {
                out.push(b'%');
                out.extend_from_slice(pairs.len().to_string().as_bytes());
                out.extend_from_slice(b"\r\n");
                for (k, v) in pairs {
                    k.encode(out);
                    v.encode(out);
                }
            }
        }
    }

    pub fn to_wire(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.encode(&mut out);
        out
    }

    pub fn write_to<W: Write>(&self, w: &mut W) -> io::Result<()> {
        let buf = self.to_wire();
        w.write_all(&buf)
    }
}

/// Encode a client command (`&[&[u8]]`) as a RESP2 array of bulk strings.
pub fn encode_command(args: &[&[u8]]) -> Vec<u8> {
    let items = args.iter().map(|a| RespVal::bulk(*a)).collect();
    RespVal::array(items).to_wire()
}

/// Incremental RESP2 + inline command buffer supporting pipelined TCP reads.
pub struct RespBuffer {
    buf: Vec<u8>,
    pos: usize,
}

impl Default for RespBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl RespBuffer {
    pub fn new() -> Self {
        Self {
            buf: Vec::with_capacity(32_768),
            pos: 0,
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.pos >= self.buf.len()
    }

    #[inline]
    pub fn raw_buf(&self) -> &[u8] {
        &self.buf
    }

    /// Read more bytes directly into internal buffer (zero intermediate copy). Returns `Ok(0)` on EOF.
    pub fn read_more<R: Read>(&mut self, stream: &mut R) -> io::Result<usize> {
        if self.pos > 0 {
            if self.pos >= self.buf.len() {
                self.buf.clear();
                self.pos = 0;
            } else if self.pos > 4096 {
                self.buf.drain(..self.pos);
                self.pos = 0;
            }
        }
        let old_len = self.buf.len();
        self.buf.resize(old_len + 32_768, 0);
        match stream.read(&mut self.buf[old_len..]) {
            Ok(n) => {
                self.buf.truncate(old_len + n);
                Ok(n)
            }
            Err(e) => {
                self.buf.truncate(old_len);
                Err(e)
            }
        }
    }

    /// Zero-allocation RESP2 + simple inline command scanner.
    /// Appends `(start, end)` byte offsets into `self.buf` for every argument of every complete
    /// command currently buffered, and appends `(arg_start, arg_end)` ranges into `cmd_ranges`.
    /// Returns `Ok(true)` if all buffered commands were parsed with zero allocation, or `Ok(false)`
    /// if a quoted inline command was encountered (caller can fall back to `next_command()`).
    pub fn parse_batch_spans(
        &mut self,
        arg_spans: &mut Vec<(usize, usize)>,
        cmd_ranges: &mut Vec<(usize, usize)>,
    ) -> Result<bool, String> {
        arg_spans.clear();
        cmd_ranges.clear();
        let bytes = &self.buf;
        let len = bytes.len();
        while self.pos < len {
            let b = bytes[self.pos];
            if b == b'\r' || b == b'\n' {
                self.pos += 1;
                continue;
            }
            if b == b'*' {
                let mut cur = self.pos + 1;
                let Some((argc, next)) = parse_usize_crlf(bytes, cur) else {
                    break;
                };
                cur = next;
                if argc == 0 {
                    self.pos = cur;
                    continue;
                }
                let saved_args = arg_spans.len();
                let mut incomplete = false;
                for _ in 0..argc {
                    if cur >= len {
                        incomplete = true;
                        break;
                    }
                    if bytes[cur] != b'$' {
                        arg_spans.truncate(saved_args);
                        return Ok(false);
                    }
                    let Some((blen, body_start)) = parse_usize_crlf(bytes, cur + 1) else {
                        incomplete = true;
                        break;
                    };
                    if blen > 512 * 1024 * 1024 {
                        arg_spans.truncate(saved_args);
                        return Err("ERR bulk string exceeds 512 MiB".to_owned());
                    }
                    let body_end = body_start + blen;
                    if body_end + 2 > len {
                        incomplete = true;
                        break;
                    }
                    arg_spans.push((body_start, body_end));
                    cur = body_end + 2;
                }
                if incomplete {
                    arg_spans.truncate(saved_args);
                    break;
                }
                cmd_ranges.push((saved_args, arg_spans.len()));
                self.pos = cur;
            } else {
                // Simple unquoted inline command (e.g. `PING\r\n`, `GET foo\r\n`)
                let Some(rel_nl) = bytes[self.pos..].iter().position(|&c| c == b'\n') else {
                    if len - self.pos > 1_048_576 {
                        return Err("ERR inline command too large".to_owned());
                    }
                    break;
                };
                let line_end = self.pos + rel_nl;
                let trim_end = if line_end > self.pos && bytes[line_end - 1] == b'\r' {
                    line_end - 1
                } else {
                    line_end
                };
                if bytes[self.pos..trim_end]
                    .iter()
                    .any(|&c| c == b'"' || c == b'\'' || c == b'\\')
                {
                    return Ok(false);
                }
                let saved_args = arg_spans.len();
                let mut i = self.pos;
                while i < trim_end {
                    while i < trim_end && matches!(bytes[i], b' ' | b'\t') {
                        i += 1;
                    }
                    let st = i;
                    while i < trim_end && !matches!(bytes[i], b' ' | b'\t') {
                        i += 1;
                    }
                    if st < i {
                        arg_spans.push((st, i));
                    }
                }
                if arg_spans.len() > saved_args {
                    cmd_ranges.push((saved_args, arg_spans.len()));
                }
                self.pos = line_end + 1;
            }
        }
        Ok(true)
    }

    /// Try to parse one complete command (either RESP2 `*N` array or inline `PING\r\n`).
    /// Returns `Ok(None)` if more bytes are needed.
    pub fn next_command(&mut self) -> Result<Option<Vec<Vec<u8>>>, String> {
        loop {
            let rem = &self.buf[self.pos..];
            if rem.is_empty() {
                return Ok(None);
            }
            if rem[0] == b'\r' || rem[0] == b'\n' {
                self.pos += 1;
                continue;
            }
            if rem[0] == b'*' {
                let Some((val, used)) = parse_resp_frame(rem)? else {
                    return Ok(None);
                };
                self.pos += used;
                return match val {
                    RespVal::Array(Some(items)) => {
                        let mut cmd = Vec::with_capacity(items.len());
                        for it in items {
                            match it {
                                RespVal::Bulk(Some(b)) => cmd.push(b),
                                RespVal::Simple(s) => cmd.push(s.into_bytes()),
                                RespVal::Integer(n) => cmd.push(n.to_string().into_bytes()),
                                _ => return Err("ERR invalid command argument type".to_owned()),
                            }
                        }
                        if cmd.is_empty() {
                            continue;
                        }
                        Ok(Some(cmd))
                    }
                    _ => Err("ERR expected RESP array for command".to_owned()),
                };
            }
            // Inline command (e.g. `PING\r\n` or `SET foo "hello world"\r\n`)
            let Some(nl) = rem.iter().position(|&b| b == b'\n') else {
                if rem.len() > 1_048_576 {
                    return Err("ERR inline command too large".to_owned());
                }
                return Ok(None);
            };
            let line = rem[..nl].strip_suffix(b"\r").unwrap_or(&rem[..nl]);
            let parsed = split_inline(line)?;
            self.pos += nl + 1;
            if parsed.is_empty() {
                continue;
            }
            return Ok(Some(parsed));
        }
    }

    /// Try to parse one arbitrary RESP2 reply (used by clients/tests).
    pub fn next_value(&mut self) -> Result<Option<RespVal>, String> {
        let rem = &self.buf[self.pos..];
        if rem.is_empty() {
            return Ok(None);
        }
        let Some((val, used)) = parse_resp_frame(rem)? else {
            return Ok(None);
        };
        self.pos += used;
        Ok(Some(val))
    }
}

#[inline]
fn parse_usize_crlf(b: &[u8], start: usize) -> Option<(usize, usize)> {
    let len = b.len();
    let mut i = start;
    let mut acc = 0usize;
    let mut any = false;
    while i < len {
        let c = b[i];
        if c.is_ascii_digit() {
            acc = acc.checked_mul(10)?.checked_add((c - b'0') as usize)?;
            any = true;
            i += 1;
        } else if c == b'\r' && i + 1 < len && b[i + 1] == b'\n' && any {
            return Some((acc, i + 2));
        } else {
            return None;
        }
    }
    None
}

fn find_crlf(b: &[u8]) -> Option<usize> {
    b.windows(2).position(|w| w == b"\r\n")
}

pub fn parse_resp_frame(src: &[u8]) -> Result<Option<(RespVal, usize)>, String> {
    if src.is_empty() {
        return Ok(None);
    }
    let Some(eol) = find_crlf(src) else {
        return Ok(None);
    };
    let head = &src[1..eol];
    let after = eol + 2;
    match src[0] {
        b'+' => Ok(Some((
            RespVal::Simple(String::from_utf8_lossy(head).into_owned()),
            after,
        ))),
        b'-' => Ok(Some((
            RespVal::Error(String::from_utf8_lossy(head).into_owned()),
            after,
        ))),
        b':' => {
            let n: i64 = std::str::from_utf8(head)
                .map_err(|_| "ERR bad integer")?
                .parse()
                .map_err(|_| "ERR bad integer")?;
            Ok(Some((RespVal::Integer(n), after)))
        }
        b'$' => {
            let len: isize = std::str::from_utf8(head)
                .map_err(|_| "ERR bad bulk length")?
                .parse()
                .map_err(|_| "ERR bad bulk length")?;
            if len < 0 {
                return Ok(Some((RespVal::Bulk(None), after)));
            }
            let len = len as usize;
            if len > 512 * 1024 * 1024 {
                return Err("ERR bulk string exceeds 512 MiB".to_owned());
            }
            if src.len() < after + len + 2 {
                return Ok(None);
            }
            let data = src[after..after + len].to_vec();
            Ok(Some((RespVal::Bulk(Some(data)), after + len + 2)))
        }
        b'*' => {
            let cnt: isize = std::str::from_utf8(head)
                .map_err(|_| "ERR bad array length")?
                .parse()
                .map_err(|_| "ERR bad array length")?;
            if cnt < 0 {
                return Ok(Some((RespVal::Array(None), after)));
            }
            let cnt = cnt as usize;
            let mut cur = after;
            let mut items = Vec::with_capacity(cnt.min(1024));
            for _ in 0..cnt {
                let Some((v, used)) = parse_resp_frame(&src[cur..])? else {
                    return Ok(None);
                };
                items.push(v);
                cur += used;
            }
            Ok(Some((RespVal::Array(Some(items)), cur)))
        }
        b'%' => {
            let cnt: usize = std::str::from_utf8(head)
                .map_err(|_| "ERR bad map length")?
                .parse()
                .map_err(|_| "ERR bad map length")?;
            let mut cur = after;
            let mut pairs = Vec::with_capacity(cnt.min(512));
            for _ in 0..cnt {
                let Some((k, uk)) = parse_resp_frame(&src[cur..])? else {
                    return Ok(None);
                };
                cur += uk;
                let Some((v, uv)) = parse_resp_frame(&src[cur..])? else {
                    return Ok(None);
                };
                cur += uv;
                pairs.push((k, v));
            }
            Ok(Some((RespVal::Map(pairs), cur)))
        }
        other => Err(format!("ERR protocol error: unexpected byte 0x{other:02x}")),
    }
}

fn split_inline(line: &[u8]) -> Result<Vec<Vec<u8>>, String> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < line.len() {
        while i < line.len() && matches!(line[i], b' ' | b'\t') {
            i += 1;
        }
        if i >= line.len() {
            break;
        }
        if line[i] == b'"' || line[i] == b'\'' {
            let q = line[i];
            i += 1;
            let mut cur = Vec::new();
            let mut closed = false;
            while i < line.len() {
                if line[i] == b'\\' && i + 1 < line.len() {
                    cur.push(match line[i + 1] {
                        b'n' => b'\n',
                        b'r' => b'\r',
                        b't' => b'\t',
                        c => c,
                    });
                    i += 2;
                } else if line[i] == q {
                    i += 1;
                    closed = true;
                    break;
                } else {
                    cur.push(line[i]);
                    i += 1;
                }
            }
            if !closed {
                return Err("ERR unbalanced quotes in inline command".to_owned());
            }
            out.push(cur);
        } else {
            let start = i;
            while i < line.len() && !matches!(line[i], b' ' | b'\t') {
                i += 1;
            }
            out.push(line[start..i].to_vec());
        }
    }
    Ok(out)
}
