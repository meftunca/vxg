//! Zero-dependency RedisJSON (`JSON.SET`, `JSON.GET`, `JSON.DEL`/`JSON.FORGET`, `JSON.NUMINCRBY`)
//! stored natively as compact UTF-8 JSON inside `Value::Str` — 100% queryable by `VXG.SQL`!

use crate::dispatch_ds::{arg_str, parse_f64, WRONGTYPE};
use crate::engine::KvEngine;
use crate::resp::RespVal;
use crate::types::{push_json_str, unescape_json, Value};

/// Dispatch `JSON.*` commands. Returns `None` if command belongs elsewhere.
pub fn dispatch_json_cmd(eng: &mut KvEngine, cmd: &str, args: &[Vec<u8>]) -> Option<RespVal> {
    Some(match cmd {
        "JSON.SET" => dispatch_json_set(eng, args),
        "JSON.GET" => dispatch_json_get(eng, args),
        "JSON.DEL" | "JSON.FORGET" => dispatch_json_del(eng, args),
        "JSON.NUMINCRBY" => dispatch_json_numincrby(eng, args),
        _ => return None,
    })
}

#[derive(Clone, Debug)]
enum JVal {
    Null,
    Bool(bool),
    Num(f64, String),
    Str(String),
    Arr(Vec<JVal>),
    Obj(Vec<(String, JVal)>),
}

impl JVal {
    fn encode(&self, out: &mut Vec<u8>) {
        match self {
            Self::Null => out.extend_from_slice(b"null"),
            Self::Bool(b) => out.extend_from_slice(if *b { b"true" } else { b"false" }),
            Self::Num(_, raw) => out.extend_from_slice(raw.as_bytes()),
            Self::Str(s) => push_json_str(out, s),
            Self::Arr(items) => {
                out.push(b'[');
                for (i, it) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(b',');
                    }
                    it.encode(out);
                }
                out.push(b']');
            }
            Self::Obj(fields) => {
                out.push(b'{');
                for (i, (k, v)) in fields.iter().enumerate() {
                    if i > 0 {
                        out.push(b',');
                    }
                    push_json_str(out, k);
                    out.push(b':');
                    v.encode(out);
                }
                out.push(b'}');
            }
        }
    }

    fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(64);
        self.encode(&mut out);
        out
    }
}

struct JParser<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> JParser<'a> {
    fn skip_ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\r' | b'\n') {
            self.i += 1;
        }
    }

    fn parse(b: &'a [u8]) -> Result<JVal, String> {
        let mut p = Self { b, i: 0 };
        let v = p.val()?;
        p.skip_ws();
        if p.i != b.len() {
            return Err("ERR trailing JSON bytes".to_owned());
        }
        Ok(v)
    }

    fn val(&mut self) -> Result<JVal, String> {
        self.skip_ws();
        let &ch = self.b.get(self.i).ok_or("ERR unexpected EOF in JSON")?;
        match ch {
            b'{' => self.obj(),
            b'[' => self.arr(),
            b'"' => Ok(JVal::Str(self.string()?)),
            b't' if self.b[self.i..].starts_with(b"true") => {
                self.i += 4;
                Ok(JVal::Bool(true))
            }
            b'f' if self.b[self.i..].starts_with(b"false") => {
                self.i += 5;
                Ok(JVal::Bool(false))
            }
            b'n' if self.b[self.i..].starts_with(b"null") => {
                self.i += 4;
                Ok(JVal::Null)
            }
            b'-' | b'0'..=b'9' => self.num(),
            _ => Err("ERR invalid JSON value".to_owned()),
        }
    }

    fn string(&mut self) -> Result<String, String> {
        self.i += 1;
        let start = self.i;
        while self.i < self.b.len() {
            if self.b[self.i] == b'\\' {
                self.i += 2;
                continue;
            }
            if self.b[self.i] == b'"' {
                let raw = &self.b[start..self.i];
                self.i += 1;
                let unesc = unescape_json(raw).ok_or("ERR invalid JSON escape")?;
                return String::from_utf8(unesc).map_err(|_| "ERR non-UTF8 JSON string".to_owned());
            }
            self.i += 1;
        }
        Err("ERR unterminated JSON string".to_owned())
    }

    fn num(&mut self) -> Result<JVal, String> {
        let start = self.i;
        while self.i < self.b.len()
            && matches!(self.b[self.i], b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E')
        {
            self.i += 1;
        }
        let s = std::str::from_utf8(&self.b[start..self.i]).map_err(|_| "ERR bad JSON number")?;
        let f: f64 = s.parse().map_err(|_| "ERR bad JSON number")?;
        Ok(JVal::Num(f, s.to_owned()))
    }

    fn arr(&mut self) -> Result<JVal, String> {
        self.i += 1;
        let mut items = Vec::new();
        self.skip_ws();
        if self.b.get(self.i) == Some(&b']') {
            self.i += 1;
            return Ok(JVal::Arr(items));
        }
        loop {
            items.push(self.val()?);
            self.skip_ws();
            match self.b.get(self.i) {
                Some(&b',') => self.i += 1,
                Some(&b']') => {
                    self.i += 1;
                    return Ok(JVal::Arr(items));
                }
                _ => return Err("ERR malformed JSON array".to_owned()),
            }
        }
    }

    fn obj(&mut self) -> Result<JVal, String> {
        self.i += 1;
        let mut fields = Vec::new();
        self.skip_ws();
        if self.b.get(self.i) == Some(&b'}') {
            self.i += 1;
            return Ok(JVal::Obj(fields));
        }
        loop {
            self.skip_ws();
            if self.b.get(self.i) != Some(&b'"') {
                return Err("ERR expected JSON object key".to_owned());
            }
            let k = self.string()?;
            self.skip_ws();
            if self.b.get(self.i) != Some(&b':') {
                return Err("ERR expected ':' in JSON object".to_owned());
            }
            self.i += 1;
            let v = self.val()?;
            fields.push((k, v));
            self.skip_ws();
            match self.b.get(self.i) {
                Some(&b',') => self.i += 1,
                Some(&b'}') => {
                    self.i += 1;
                    return Ok(JVal::Obj(fields));
                }
                _ => return Err("ERR malformed JSON object".to_owned()),
            }
        }
    }
}

fn parse_jpath(path: &str) -> Vec<&str> {
    let p = path.trim();
    if p.is_empty() || p == "$" || p == "." {
        return Vec::new();
    }
    let stripped = p
        .strip_prefix("$.")
        .or_else(|| p.strip_prefix('$'))
        .or_else(|| p.strip_prefix('.'))
        .unwrap_or(p);
    stripped.split('.').filter(|s| !s.is_empty()).collect()
}

fn jval_get<'a>(root: &'a JVal, segs: &[&str]) -> Option<&'a JVal> {
    let mut cur = root;
    for &seg in segs {
        match cur {
            JVal::Obj(fields) => {
                cur = &fields.iter().find(|(k, _)| k == seg)?.1;
            }
            JVal::Arr(items) => {
                let idx: usize = seg.trim_matches(|c| c == '[' || c == ']').parse().ok()?;
                cur = items.get(idx)?;
            }
            _ => return None,
        }
    }
    Some(cur)
}

fn jval_set(root: &mut JVal, segs: &[&str], new_v: JVal) -> Result<(), String> {
    if segs.is_empty() {
        *root = new_v;
        return Ok(());
    }
    let mut cur = root;
    for &seg in &segs[..segs.len() - 1] {
        match cur {
            JVal::Obj(fields) => {
                if let Some(pos) = fields.iter().position(|(k, _)| k == seg) {
                    cur = &mut fields[pos].1;
                } else {
                    fields.push((seg.to_owned(), JVal::Obj(Vec::new())));
                    cur = &mut fields.last_mut().unwrap().1;
                }
            }
            _ => return Err("ERR path parent is not a JSON object".to_owned()),
        }
    }
    let leaf = segs[segs.len() - 1];
    match cur {
        JVal::Obj(fields) => {
            if let Some((_, slot)) = fields.iter_mut().find(|(k, _)| k == leaf) {
                *slot = new_v;
            } else {
                fields.push((leaf.to_owned(), new_v));
            }
            Ok(())
        }
        _ => Err("ERR path parent is not a JSON object".to_owned()),
    }
}

fn jval_del(root: &mut JVal, segs: &[&str]) -> bool {
    if segs.is_empty() {
        return false;
    }
    let mut cur = root;
    for &seg in &segs[..segs.len() - 1] {
        match cur {
            JVal::Obj(fields) => {
                let Some(pos) = fields.iter().position(|(k, _)| k == seg) else {
                    return false;
                };
                cur = &mut fields[pos].1;
            }
            _ => return false,
        }
    }
    let leaf = segs[segs.len() - 1];
    if let JVal::Obj(fields) = cur {
        if let Some(pos) = fields.iter().position(|(k, _)| k == leaf) {
            fields.remove(pos);
            return true;
        }
    }
    false
}

fn dispatch_json_set(eng: &mut KvEngine, args: &[Vec<u8>]) -> RespVal {
    if args.len() < 4 {
        return RespVal::err("ERR wrong number of arguments for 'json.set'");
    }
    let path = arg_str(args, 2).unwrap_or("$");
    let segs = parse_jpath(path);
    let new_v = match JParser::parse(&args[3]) {
        Ok(v) => v,
        Err(e) => return RespVal::err(e),
    };
    let nx = arg_str(args, 4).is_some_and(|s| s.eq_ignore_ascii_case("NX"));
    let xx = arg_str(args, 4).is_some_and(|s| s.eq_ignore_ascii_case("XX"));
    if segs.is_empty() {
        if nx || xx {
            let exists = matches!(eng.get_record_or_load(&args[1]), Ok(Some(_)));
            if (nx && exists) || (xx && !exists) {
                return RespVal::nil();
            }
        }
        eng.set_str_in_place(&args[1], &new_v.to_bytes());
        return RespVal::ok();
    }
    let Ok(rec) = eng.ensure_entry_mut(&args[1], || Value::Str(b"{}".to_vec())) else {
        return RespVal::err("ERR load failed");
    };
    let Some(Value::Str(buf)) = rec.value.as_mut() else {
        return RespVal::err(WRONGTYPE);
    };
    let mut root = match JParser::parse(buf) {
        Ok(v) => v,
        Err(e) => return RespVal::err(e),
    };
    let exists = jval_get(&root, &segs).is_some();
    if (nx && exists) || (xx && !exists) {
        return RespVal::nil();
    }
    if let Err(e) = jval_set(&mut root, &segs, new_v) {
        return RespVal::err(e);
    }
    buf.clear();
    root.encode(buf);
    RespVal::ok()
}

fn dispatch_json_get(eng: &mut KvEngine, args: &[Vec<u8>]) -> RespVal {
    if args.len() < 2 {
        return RespVal::err("ERR wrong number of arguments for 'json.get'");
    }
    let Some(rec) = (match eng.get_record_or_load(&args[1]) {
        Ok(v) => v,
        Err(e) => return RespVal::err(e),
    }) else {
        return RespVal::nil();
    };
    let Some(Value::Str(buf)) = rec.value.as_ref() else {
        return RespVal::err(WRONGTYPE);
    };
    let path = arg_str(args, 2).unwrap_or("$");
    let segs = parse_jpath(path);
    if segs.is_empty() {
        return RespVal::bulk(buf.clone());
    }
    let root = match JParser::parse(buf) {
        Ok(v) => v,
        Err(e) => return RespVal::err(e),
    };
    match jval_get(&root, &segs) {
        Some(v) => RespVal::bulk(v.to_bytes()),
        None => RespVal::nil(),
    }
}

fn dispatch_json_del(eng: &mut KvEngine, args: &[Vec<u8>]) -> RespVal {
    if args.len() < 2 {
        return RespVal::err("ERR wrong number of arguments for 'json.del'");
    }
    let path = arg_str(args, 2).unwrap_or("$");
    let segs = parse_jpath(path);
    if segs.is_empty() {
        return RespVal::int(i64::from(matches!(eng.del_key(&args[1]), Ok(true))));
    }
    let Some(rec) = (match eng.get_or_promote_mut(&args[1]) {
        Ok(v) => v,
        Err(e) => return RespVal::err(e),
    }) else {
        return RespVal::int(0);
    };
    let Some(Value::Str(buf)) = rec.value.as_mut() else {
        return RespVal::err(WRONGTYPE);
    };
    let mut root = match JParser::parse(buf) {
        Ok(v) => v,
        Err(e) => return RespVal::err(e),
    };
    if jval_del(&mut root, &segs) {
        buf.clear();
        root.encode(buf);
        RespVal::int(1)
    } else {
        RespVal::int(0)
    }
}

fn dispatch_json_numincrby(eng: &mut KvEngine, args: &[Vec<u8>]) -> RespVal {
    if args.len() != 4 {
        return RespVal::err("ERR wrong number of arguments for 'json.numincrby'");
    }
    let segs = parse_jpath(arg_str(args, 2).unwrap_or("$"));
    let delta = match parse_f64(&args[3]) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let Some(rec) = (match eng.get_or_promote_mut(&args[1]) {
        Ok(v) => v,
        Err(e) => return RespVal::err(e),
    }) else {
        return RespVal::err("ERR key does not exist");
    };
    let Some(Value::Str(buf)) = rec.value.as_mut() else {
        return RespVal::err(WRONGTYPE);
    };
    let mut root = match JParser::parse(buf) {
        Ok(v) => v,
        Err(e) => return RespVal::err(e),
    };
    let cur_num = match jval_get(&root, &segs) {
        Some(JVal::Num(n, _)) => *n,
        _ => return RespVal::err("ERR target path is not a JSON number"),
    };
    let next = cur_num + delta;
    let formatted = if next.fract() == 0.0 && next.abs() < 1e15 {
        format!("{}", next as i64)
    } else {
        format!("{next}")
    };
    let _ = jval_set(&mut root, &segs, JVal::Num(next, formatted.clone()));
    buf.clear();
    root.encode(buf);
    RespVal::bulk(formatted.into_bytes())
}
