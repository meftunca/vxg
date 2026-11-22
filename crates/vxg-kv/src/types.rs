//! Core KV data types, canonical NDJSON record wire format (VXG SQL-queryable & binary-safe),
//! wall-clock helpers, and Redis glob pattern matching (`*`, `?`, `[abc]`).

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::hash::{BuildHasher, Hasher};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Default, Clone, Copy)]
pub struct FxBuildHasher;
pub struct FxHasher(u64);
impl BuildHasher for FxBuildHasher {
    type Hasher = FxHasher;
    #[inline]
    fn build_hasher(&self) -> FxHasher { FxHasher(0) }
}
impl Hasher for FxHasher {
    #[inline]
    fn finish(&self) -> u64 { self.0 }
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        const K: u64 = 0x517cc1b727220a95;
        let (chunks, rem) = bytes.as_chunks::<8>();
        let mut h = self.0;
        for c in chunks { h = (h.rotate_left(5) ^ u64::from_le_bytes(*c)).wrapping_mul(K); }
        for &b in rem { h = (h.rotate_left(5) ^ (b as u64)).wrapping_mul(K); }
        self.0 = h;
    }
    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.0 = (self.0.rotate_left(5) ^ (i as u64)).wrapping_mul(0x517cc1b727220a95);
    }
}
pub type FastMap<K, V> = HashMap<K, V, FxBuildHasher>;

/// Current UTC Unix timestamp in milliseconds.
pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

/// Monotonic Redis Stream entry ID (`ms-seq`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StreamId {
    pub ms: u64,
    pub seq: u64,
}

impl StreamId {
    pub fn format(self) -> String {
        format!("{}-{}", self.ms, self.seq)
    }
}

/// One Redis Stream entry (`id` + ordered field/value pairs).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamEntry {
    pub id: StreamId,
    pub fields: Vec<(Vec<u8>, Vec<u8>)>,
}

/// All supported Redis non-deprecated data types.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Str(Vec<u8>),
    Hash(BTreeMap<Vec<u8>, Vec<u8>>),
    List(VecDeque<Vec<u8>>),
    Set(BTreeSet<Vec<u8>>),
    ZSet(BTreeMap<Vec<u8>, f64>),
    Stream(Vec<StreamEntry>),
}

impl Value {
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Str(_) => "string",
            Self::Hash(_) => "hash",
            Self::List(_) => "list",
            Self::Set(_) => "set",
            Self::ZSet(_) => "zset",
            Self::Stream(_) => "stream",
        }
    }
}

/// Versioned KV record with optional expiry (`expires_at_ms == 0` means no TTL) and tombstone (`value = None`).
#[derive(Clone, Debug, PartialEq)]
pub struct Record {
    pub value: Option<Value>,
    pub expires_at_ms: u64,
    pub version: u64,
}

impl Record {
    #[inline]
    pub fn live(value: Value, expires_at_ms: Option<u64>, version: u64) -> Self {
        Self {
            value: Some(value),
            expires_at_ms: expires_at_ms.unwrap_or(0),
            version,
        }
    }

    #[inline]
    pub fn tombstone(version: u64) -> Self {
        Self {
            value: None,
            expires_at_ms: 0,
            version,
        }
    }

    #[inline]
    pub fn expires_opt(&self) -> Option<u64> {
        (self.expires_at_ms > 0).then_some(self.expires_at_ms)
    }

    #[inline]
    pub fn is_expired(&self, now: u64) -> bool {
        self.expires_at_ms > 0 && now >= self.expires_at_ms
    }

    #[inline]
    pub fn is_live(&self, now: u64) -> bool {
        self.value.is_some() && !self.is_expired(now)
    }

    /// Zero-syscall liveness check: only reads wall clock if the record actually has a TTL.
    #[inline]
    pub fn is_live_lazy(&self) -> bool {
        if self.value.is_none() {
            return false;
        }
        self.expires_at_ms == 0 || now_ms() < self.expires_at_ms
    }
}

/// Redis-style glob matching supporting `*`, `?`, `\x`, and `[a-z]` / `[^a-z]`.
pub fn glob_match(pat: &[u8], s: &[u8]) -> bool {
    let (mut pi, mut si) = (0usize, 0usize);
    let (mut star_p, mut star_s) = (None, 0usize);
    while si < s.len() {
        if pi < pat.len() && pat[pi] == b'*' {
            star_p = Some(pi);
            pi += 1;
            star_s = si;
        } else if pi < pat.len() && pat[pi] == b'?' {
            pi += 1;
            si += 1;
        } else if pi < pat.len() && pat[pi] == b'\\' && pi + 1 < pat.len() {
            if pat[pi + 1] == s[si] {
                pi += 2;
                si += 1;
            } else if let Some(sp) = star_p {
                pi = sp + 1;
                star_s += 1;
                si = star_s;
            } else {
                return false;
            }
        } else if pi < pat.len() && pat[pi] == b'[' {
            if let Some((matched, next_p)) = match_bracket(&pat[pi + 1..], s[si]) {
                if matched {
                    pi += 1 + next_p;
                    si += 1;
                    continue;
                }
            }
            if let Some(sp) = star_p {
                pi = sp + 1;
                star_s += 1;
                si = star_s;
            } else {
                return false;
            }
        } else if pi < pat.len() && pat[pi] == s[si] {
            pi += 1;
            si += 1;
        } else if let Some(sp) = star_p {
            pi = sp + 1;
            star_s += 1;
            si = star_s;
        } else {
            return false;
        }
    }
    while pi < pat.len() && pat[pi] == b'*' {
        pi += 1;
    }
    pi == pat.len()
}

fn match_bracket(p: &[u8], ch: u8) -> Option<(bool, usize)> {
    let mut i = 0;
    let negate = if i < p.len() && p[i] == b'^' {
        i += 1;
        true
    } else {
        false
    };
    let mut matched = false;
    while i < p.len() {
        if p[i] == b']' {
            return Some((if negate { !matched } else { matched }, i + 1));
        }
        if i + 2 < p.len() && p[i + 1] == b'-' && p[i + 2] != b']' {
            if ch >= p[i] && ch <= p[i + 2] {
                matched = true;
            }
            i += 3;
        } else {
            if ch == p[i] {
                matched = true;
            }
            i += 1;
        }
    }
    None
}

pub fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0xf) as usize] as char);
    }
    out
}

pub fn hex_decode(s: &[u8]) -> Option<Vec<u8>> {
    if !s.len().is_multiple_of(2) {
        return None;
    }
    let mut out = Vec::with_capacity(s.len() / 2);
    let nib = |c: u8| match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    };
    for pair in s.as_chunks::<2>().0 {
        out.push((nib(pair[0])? << 4) | nib(pair[1])?);
    }
    Some(out)
}

/// JSON-escape a UTF-8 string into `out` (including surrounding `"`).
pub fn push_json_str(out: &mut Vec<u8>, s: &str) {
    out.push(b'"');
    for b in s.bytes() {
        match b {
            b'"' => out.extend_from_slice(b"\\\""),
            b'\\' => out.extend_from_slice(b"\\\\"),
            b'\n' => out.extend_from_slice(b"\\n"),
            b'\r' => out.extend_from_slice(b"\\r"),
            b'\t' => out.extend_from_slice(b"\\t"),
            0x00..=0x1f => out.extend_from_slice(format!("\\u{b:04x}").as_bytes()),
            _ => out.push(b),
        }
    }
    out.push(b'"');
}

/// Unescape a JSON string literal (without outer quotes) back into bytes.
pub fn unescape_json(raw: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(raw.len());
    let mut i = 0;
    while i < raw.len() {
        if raw[i] == b'\\' && i + 1 < raw.len() {
            match raw[i + 1] {
                b'"' => out.push(b'"'),
                b'\\' => out.push(b'\\'),
                b'/' => out.push(b'/'),
                b'n' => out.push(b'\n'),
                b'r' => out.push(b'\r'),
                b't' => out.push(b'\t'),
                b'b' => out.push(0x08),
                b'f' => out.push(0x0c),
                b'u' if i + 5 < raw.len() => {
                    let s = std::str::from_utf8(&raw[i + 2..i + 6]).ok()?;
                    let cp = u32::from_str_radix(s, 16).ok()?;
                    let ch = char::from_u32(cp)?;
                    let mut buf = [0u8; 4];
                    out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                    i += 6;
                    continue;
                }
                other => out.push(other),
            }
            i += 2;
        } else {
            out.push(raw[i]);
            i += 1;
        }
    }
    Some(out)
}

/// Detect whether `v` is a compact single-line JSON object `{...}` safe to embed verbatim for `VXG.SQL`.
fn is_embeddable_json_obj(v: &[u8]) -> bool {
    if v.len() < 2 || v[0] != b'{' || v[v.len() - 1] != b'}' {
        return false;
    }
    if v.iter().any(|&b| b == b'\n' || b == b'\r' || b == 0) {
        return false;
    }
    let Ok(s) = std::str::from_utf8(v) else {
        return false;
    };
    let mut depth = 0i32;
    let mut in_str = false;
    let mut esc = false;
    for (idx, b) in s.bytes().enumerate() {
        if in_str {
            if esc {
                esc = false;
            } else if b == b'\\' {
                esc = true;
            } else if b == b'"' {
                in_str = false;
            }
        } else {
            match b {
                b'"' => in_str = true,
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 && idx + 1 != s.len() {
                        return false;
                    }
                    if depth < 0 {
                        return false;
                    }
                }
                _ => {}
            }
        }
    }
    !in_str && depth == 0
}

fn encode_complex_payload(val: &Value) -> Vec<u8> {
    let mut buf = Vec::new();
    let push_bytes = |dst: &mut Vec<u8>, b: &[u8]| {
        dst.extend_from_slice(&(b.len() as u32).to_le_bytes());
        dst.extend_from_slice(b);
    };
    match val {
        Value::Hash(m) => {
            buf.extend_from_slice(&(m.len() as u32).to_le_bytes());
            for (k, v) in m {
                push_bytes(&mut buf, k);
                push_bytes(&mut buf, v);
            }
        }
        Value::List(l) => {
            buf.extend_from_slice(&(l.len() as u32).to_le_bytes());
            for item in l {
                push_bytes(&mut buf, item);
            }
        }
        Value::Set(s) => {
            buf.extend_from_slice(&(s.len() as u32).to_le_bytes());
            for item in s {
                push_bytes(&mut buf, item);
            }
        }
        Value::ZSet(z) => {
            buf.extend_from_slice(&(z.len() as u32).to_le_bytes());
            for (m, sc) in z {
                push_bytes(&mut buf, m);
                buf.extend_from_slice(&sc.to_bits().to_le_bytes());
            }
        }
        Value::Stream(entries) => {
            buf.extend_from_slice(&(entries.len() as u32).to_le_bytes());
            for e in entries {
                buf.extend_from_slice(&e.id.ms.to_le_bytes());
                buf.extend_from_slice(&e.id.seq.to_le_bytes());
                buf.extend_from_slice(&(e.fields.len() as u32).to_le_bytes());
                for (k, v) in &e.fields {
                    push_bytes(&mut buf, k);
                    push_bytes(&mut buf, v);
                }
            }
        }
        Value::Str(b) => buf.extend_from_slice(b),
    }
    buf
}

fn decode_complex_payload(kind: &str, bytes: &[u8]) -> Option<Value> {
    let mut pos = 0usize;
    let read_u32 = |p: &mut usize| -> Option<u32> {
        let s = bytes.get(*p..*p + 4)?;
        *p += 4;
        Some(u32::from_le_bytes(s.try_into().ok()?))
    };
    let read_slice = |p: &mut usize| -> Option<Vec<u8>> {
        let len = read_u32(p)? as usize;
        let s = bytes.get(*p..*p + len)?;
        *p += len;
        Some(s.to_vec())
    };
    match kind {
        "hash" => {
            let n = read_u32(&mut pos)? as usize;
            let mut m = BTreeMap::new();
            for _ in 0..n {
                let k = read_slice(&mut pos)?;
                let v = read_slice(&mut pos)?;
                m.insert(k, v);
            }
            Some(Value::Hash(m))
        }
        "list" => {
            let n = read_u32(&mut pos)? as usize;
            let mut l = VecDeque::with_capacity(n);
            for _ in 0..n {
                l.push_back(read_slice(&mut pos)?);
            }
            Some(Value::List(l))
        }
        "set" => {
            let n = read_u32(&mut pos)? as usize;
            let mut s = BTreeSet::new();
            for _ in 0..n {
                s.insert(read_slice(&mut pos)?);
            }
            Some(Value::Set(s))
        }
        "zset" => {
            let n = read_u32(&mut pos)? as usize;
            let mut z = BTreeMap::new();
            for _ in 0..n {
                let m = read_slice(&mut pos)?;
                let sb = bytes.get(pos..pos + 8)?;
                pos += 8;
                let sc = f64::from_bits(u64::from_le_bytes(sb.try_into().ok()?));
                z.insert(m, sc);
            }
            Some(Value::ZSet(z))
        }
        "stream" => {
            let n = read_u32(&mut pos)? as usize;
            let mut entries = Vec::with_capacity(n);
            for _ in 0..n {
                let ms = u64::from_le_bytes(bytes.get(pos..pos + 8)?.try_into().ok()?);
                pos += 8;
                let seq = u64::from_le_bytes(bytes.get(pos..pos + 8)?.try_into().ok()?);
                pos += 8;
                let nf = read_u32(&mut pos)? as usize;
                let mut fields = Vec::with_capacity(nf);
                for _ in 0..nf {
                    let k = read_slice(&mut pos)?;
                    let v = read_slice(&mut pos)?;
                    fields.push((k, v));
                }
                entries.push(StreamEntry {
                    id: StreamId { ms, seq },
                    fields,
                });
            }
            Some(Value::Stream(entries))
        }
        _ => None,
    }
}

/// Encode `(key, record)` into a single canonical NDJSON line (`\n`-terminated).
///
/// UTF-8 JSON object string values are embedded verbatim under `"data":{...}` so VXG's
/// `BLOCK_COLUMN` + `VXG.SQL` can query nested JSON fields directly on compressed blocks!
pub fn encode_record_line(out: &mut Vec<u8>, key: &[u8], rec: &Record) {
    if let Ok(ks) = std::str::from_utf8(key) {
        out.extend_from_slice(b"{\"_k\":");
        push_json_str(out, ks);
    } else {
        out.extend_from_slice(b"{\"_kh\":\"");
        out.extend_from_slice(hex_encode(key).as_bytes());
        out.push(b'"');
    }
    let exp = rec.expires_at_ms;
    out.extend_from_slice(format!(",\"_e\":{exp},\"_v\":{},\"_u\":0", rec.version).as_bytes());
    match &rec.value {
        None => {
            out.extend_from_slice(b",\"_t\":\"del\"}\n");
        }
        Some(Value::Str(b)) if is_embeddable_json_obj(b) => {
            out.extend_from_slice(b",\"_t\":\"json\",\"data\":");
            out.extend_from_slice(b);
            out.extend_from_slice(b"}\n");
        }
        Some(Value::Str(b)) => {
            if let Ok(vs) = std::str::from_utf8(b) {
                out.extend_from_slice(b",\"_t\":\"str\",\"val\":");
                push_json_str(out, vs);
                out.extend_from_slice(b"}\n");
            } else {
                out.extend_from_slice(b",\"_t\":\"bin\",\"hex\":\"");
                out.extend_from_slice(hex_encode(b).as_bytes());
                out.extend_from_slice(b"\"}\n");
            }
        }
        Some(complex) => {
            let tag = complex.type_name();
            let raw = encode_complex_payload(complex);
            out.extend_from_slice(format!(",\"_t\":\"{tag}\",\"hex\":\"").as_bytes());
            out.extend_from_slice(hex_encode(&raw).as_bytes());
            out.extend_from_slice(b"\"}\n");
        }
    }
}

/// Decode a single canonical NDJSON line back into `(key, Record)`.
pub fn decode_record_line(line: &[u8]) -> Option<(Vec<u8>, Record)> {
    let trimmed = line
        .strip_suffix(b"\n")
        .unwrap_or(line)
        .strip_suffix(b"\r")
        .unwrap_or_else(|| line.strip_suffix(b"\n").unwrap_or(line));
    if !trimmed.ends_with(b"}") {
        return None;
    }
    let (key, k_end) = if trimmed.starts_with(b"{\"_k\":\"") {
        let e = find_json_str_end(trimmed, 7)?;
        (unescape_json(&trimmed[7..e])?, e)
    } else if trimmed.starts_with(b"{\"_kh\":\"") {
        let e = find_json_str_end(trimmed, 8)?;
        (hex_decode(&trimmed[8..e])?, e)
    } else {
        return None;
    };
    let rest = &trimmed[k_end + 1..trimmed.len() - 1];
    let exp = parse_u64_field(rest, b"\"_e\":")?;
    let ver = parse_u64_field(rest, b"\"_v\":")?;
    let tag = parse_str_field(rest, b"\"_t\":\"")?;
    let value = match tag.as_slice() {
        b"del" => None,
        b"str" => Some(Value::Str(parse_str_field(rest, b"\"val\":\"")?)),
        b"json" => {
            let p = find_subslice(rest, b"\"data\":")? + 7;
            Some(Value::Str(rest[p..].to_vec()))
        }
        b"bin" => Some(Value::Str(hex_decode(&parse_str_field(rest, b"\"hex\":\"")?)?)),
        b"hash" | b"list" | b"set" | b"zset" | b"stream" => {
            let h = hex_decode(&parse_str_field(rest, b"\"hex\":\"")?)?;
            Some(decode_complex_payload(std::str::from_utf8(&tag).ok()?, &h)?)
        }
        _ => return None,
    };
    Some((
        key,
        Record {
            value,
            expires_at_ms: exp,
            version: ver,
        },
    ))
}

fn find_json_str_end(b: &[u8], start: usize) -> Option<usize> {
    let mut i = start;
    while i < b.len() {
        if b[i] == b'\\' {
            i += 2;
            continue;
        }
        if b[i] == b'"' {
            return Some(i);
        }
        i += 1;
    }
    None
}

fn find_subslice(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

fn parse_u64_field(buf: &[u8], marker: &[u8]) -> Option<u64> {
    let p = find_subslice(buf, marker)? + marker.len();
    let mut e = p;
    while e < buf.len() && buf[e].is_ascii_digit() {
        e += 1;
    }
    std::str::from_utf8(&buf[p..e]).ok()?.parse().ok()
}

fn parse_str_field(buf: &[u8], marker: &[u8]) -> Option<Vec<u8>> {
    let p = find_subslice(buf, marker)? + marker.len();
    let e = find_json_str_end(buf, p)?;
    unescape_json(&buf[p..e])
}
