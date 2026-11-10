// Thread-local memoization for repeated `GRAMMAR_PACKET` block headers and single-line JSON
// template `key -> slot` resolution across multi-thousand-block compressed-domain queries.
//
// Safety & Equivalence Invariant:
// 1. `GrammarTplWireCache` matches ONLY when `payload[p_after_raw_len..]` starts with the exact
//    byte-for-byte validated template header wire slice (`template_count .. packed_slices`).
//    All per-block fields (`raw_len`, `row_count`, `selector_slice`, `raw_rows`, `col_slices`)
//    remain uncached and fully validated on every block.
// 2. `JsonTplSlotCache` verifies exact byte equality of every template literal segment `segs[0..=slots]`
//    before consulting its pre-unescaped key table, preserving 100% equivalence on duplicate keys,
//    escaped keys, dotted keys, nested objects, and malformed templates.

struct GrammarTplWireCache {
    wire: Vec<u8>,
    template_count: usize,
    slot_counts: [usize; 16],
    tpl0_delim: Option<u8>,
    tpl0_rel: [(u16, u16); MAX_SLOTS + 1],
    extra_rel: Vec<Vec<(u16, u16)>>,
}

impl GrammarTplWireCache {
    const fn new() -> Self {
        Self {
            wire: Vec::new(),
            template_count: 0,
            slot_counts: [0usize; 16],
            tpl0_delim: None,
            tpl0_rel: [(0u16, 0u16); MAX_SLOTS + 1],
            extra_rel: Vec::new(),
        }
    }
}

struct ParsedJsonTpl {
    obj_paths: Vec<Vec<u8>>,
    top_keys: Vec<(Vec<u8>, usize)>,
    nested_keys: Vec<(Vec<u8>, usize)>,
    leaf_paths: Vec<Vec<u8>>,
}

impl ParsedJsonTpl {
    #[inline]
    fn lookup(&self, target_key: &[u8]) -> Option<Option<usize>> {
        for op in &self.obj_paths {
            if op.as_slice() == target_key {
                return None;
            }
        }
        for (tk, slot) in &self.top_keys {
            if tk.as_slice() == target_key {
                return Some(Some(*slot));
            }
        }
        if target_key.contains(&b'.') {
            for lp in &self.leaf_paths {
                let l = lp.len();
                if l < target_key.len() && target_key[l] == b'.' && target_key.starts_with(lp) {
                    return None;
                }
            }
            for (nk, slot) in &self.nested_keys {
                if nk.as_slice() == target_key {
                    return Some(Some(*slot));
                }
            }
        }
        Some(None)
    }
}

struct JsonTplCacheEntry {
    seg_count: u8,
    concat: Vec<u8>,
    ends: [u16; MAX_SLOTS + 2],
    parsed: Option<ParsedJsonTpl>,
}

impl JsonTplCacheEntry {
    const fn empty() -> Self {
        Self {
            seg_count: 0,
            concat: Vec::new(),
            ends: [0u16; MAX_SLOTS + 2],
            parsed: None,
        }
    }

    #[inline]
    fn matches_segs(&self, segs: &[&[u8]], total_len: usize) -> bool {
        if self.seg_count as usize != segs.len() || self.concat.len() != total_len {
            return false;
        }
        for (i, &seg) in segs.iter().enumerate() {
            let s = self.ends[i] as usize;
            let e = self.ends[i + 1] as usize;
            if &self.concat[s..e] != seg {
                return false;
            }
        }
        true
    }
}

struct JsonTplSlotCache {
    entries: [JsonTplCacheEntry; 4],
    next_evict: usize,
}

impl JsonTplSlotCache {
    const fn new() -> Self {
        Self {
            entries: [
                JsonTplCacheEntry::empty(),
                JsonTplCacheEntry::empty(),
                JsonTplCacheEntry::empty(),
                JsonTplCacheEntry::empty(),
            ],
            next_evict: 0,
        }
    }
}

thread_local! {
    static GRAMMAR_TPL_WIRE_CACHE: std::cell::RefCell<GrammarTplWireCache> =
        const { std::cell::RefCell::new(GrammarTplWireCache::new()) };
    static JSON_TPL_SLOT_CACHE: std::cell::RefCell<JsonTplSlotCache> =
        const { std::cell::RefCell::new(JsonTplSlotCache::new()) };
}

type CachedGrammarMatch<'a> = (
    usize,
    [usize; 16],
    Option<u8>,
    [&'a [u8]; MAX_SLOTS + 1],
    Vec<Vec<&'a [u8]>>,
);

#[inline]
fn try_match_cached_single_tpl(rest: &[u8]) -> Option<(usize, Option<u8>, usize)> {
    GRAMMAR_TPL_WIRE_CACHE.with(|cell| {
        let c = cell.borrow();
        let wlen = c.wire.len();
        if c.template_count != 1
            || wlen == 0
            || rest.len() < wlen
            || &rest[..wlen] != c.wire.as_slice()
        {
            return None;
        }
        Some((c.slot_counts[0], c.tpl0_delim, wlen))
    })
}

#[inline]
fn try_match_cached_grammar_wire<'a>(
    rest: &'a [u8],
    cursor: &mut Cursor<'a>,
) -> Option<CachedGrammarMatch<'a>> {
    GRAMMAR_TPL_WIRE_CACHE.with(|cell| {
        let c = cell.borrow();
        let wlen = c.wire.len();
        if wlen == 0 || rest.len() < wlen || &rest[..wlen] != c.wire.as_slice() {
            return None;
        }
        let mut tpl0_segs = [&[][..]; MAX_SLOTS + 1];
        let s0 = c.slot_counts[0];
        for (dst, &(sr, er)) in tpl0_segs.iter_mut().zip(c.tpl0_rel.iter()).take(s0 + 1) {
            *dst = &rest[sr as usize..er as usize];
        }
        let mut extra_tpl_segs = Vec::with_capacity(c.extra_rel.len());
        for rel_vec in &c.extra_rel {
            let mut v = Vec::with_capacity(rel_vec.len());
            for &(sr, er) in rel_vec {
                v.push(&rest[sr as usize..er as usize]);
            }
            extra_tpl_segs.push(v);
        }
        cursor.pos += wlen;
        Some((
            c.template_count,
            c.slot_counts,
            c.tpl0_delim,
            tpl0_segs,
            extra_tpl_segs,
        ))
    })
}

fn store_cached_grammar_wire(
    rest_wire: &[u8],
    template_count: usize,
    slot_counts: &[usize; 16],
    tpl0_delim: Option<u8>,
    tpl0_segs: &[&[u8]; MAX_SLOTS + 1],
    extra_tpl_segs: &[Vec<&[u8]>],
) {
    if rest_wire.is_empty() || rest_wire.len() > u16::MAX as usize {
        return;
    }
    let base = rest_wire.as_ptr() as usize;
    let end_limit = base + rest_wire.len();
    let mut tpl0_rel = [(0u16, 0u16); MAX_SLOTS + 1];
    for s in 0..=slot_counts[0] {
        let seg = tpl0_segs[s];
        let sp = seg.as_ptr() as usize;
        if sp < base || sp + seg.len() > end_limit {
            return;
        }
        let sr = sp - base;
        tpl0_rel[s] = (sr as u16, (sr + seg.len()) as u16);
    }
    let mut extra_rel = Vec::with_capacity(extra_tpl_segs.len());
    for segs in extra_tpl_segs {
        let mut v = Vec::with_capacity(segs.len());
        for &seg in segs {
            let sp = seg.as_ptr() as usize;
            if sp < base || sp + seg.len() > end_limit {
                return;
            }
            let sr = sp - base;
            v.push((sr as u16, (sr + seg.len()) as u16));
        }
        extra_rel.push(v);
    }
    GRAMMAR_TPL_WIRE_CACHE.with(|cell| {
        let mut c = cell.borrow_mut();
        c.wire.clear();
        c.wire.extend_from_slice(rest_wire);
        c.template_count = template_count;
        c.slot_counts = *slot_counts;
        c.tpl0_delim = tpl0_delim;
        c.tpl0_rel = tpl0_rel;
        c.extra_rel = extra_rel;
    });
}

fn build_json_tpl_entry(segs: &[&[u8]]) -> Option<ParsedJsonTpl> {
    let slots = segs.len().saturating_sub(1);
    if slots == 0 {
        return None;
    }
    let mut path = [0u8; 512];
    let mut path_len = 0usize;
    let mut depth_lens = [0usize; 16];
    let mut depth = 0usize;
    let mut obj_paths: Vec<Vec<u8>> = Vec::new();
    let mut top_keys: Vec<(Vec<u8>, usize)> = Vec::with_capacity(slots);
    let mut nested_keys: Vec<(Vec<u8>, usize)> = Vec::new();
    let mut leaf_paths: Vec<Vec<u8>> = Vec::with_capacity(slots);

    for (s, &seg) in segs.iter().enumerate().take(slots) {
        if seg.contains(&b'\n') {
            return None;
        }
        let mut p = 0usize;
        skip_space(seg, &mut p);
        if s == 0 {
            if seg.get(p) != Some(&b'{') {
                return None;
            }
            p += 1;
            depth = 1;
            depth_lens[0] = 0;
        } else {
            while seg.get(p) == Some(&b'}') {
                if depth <= 1 {
                    return None;
                }
                depth -= 1;
                path_len = depth_lens[depth - 1];
                p += 1;
                skip_space(seg, &mut p);
            }
            if seg.get(p) != Some(&b',') {
                return None;
            }
            p += 1;
        }
        loop {
            skip_space(seg, &mut p);
            let (k_start, k_end) = (p, scan_string(seg, p)?);
            p = k_end;
            let k_dec = crate::ops::unescape_json_str(&seg[k_start..k_end])?;
            let kb = k_dec.as_ref();
            if depth > 1 && kb.contains(&b'.') {
                return None;
            }
            skip_space(seg, &mut p);
            if seg.get(p) != Some(&b':') {
                return None;
            }
            p += 1;
            skip_space(seg, &mut p);
            if seg.get(p) == Some(&b'{') {
                let mut full_obj = Vec::with_capacity(path_len + kb.len());
                full_obj.extend_from_slice(&path[..path_len]);
                full_obj.extend_from_slice(kb);
                obj_paths.push(full_obj);
                if depth >= 16 || path_len + kb.len() + 1 > path.len() {
                    return None;
                }
                path[path_len..path_len + kb.len()].copy_from_slice(kb);
                path[path_len + kb.len()] = b'.';
                path_len += kb.len() + 1;
                depth_lens[depth] = path_len;
                depth += 1;
                p += 1;
                continue;
            }
            if p != seg.len() {
                return None;
            }
            let mut full_leaf = Vec::with_capacity(path_len + kb.len());
            full_leaf.extend_from_slice(&path[..path_len]);
            full_leaf.extend_from_slice(kb);
            if depth == 1 {
                if !top_keys.iter().any(|(k, _)| k.as_slice() == kb) {
                    top_keys.push((kb.to_vec(), s));
                }
            } else if !nested_keys.iter().any(|(k, _)| k.as_slice() == full_leaf.as_slice()) {
                nested_keys.push((full_leaf.clone(), s));
            }
            leaf_paths.push(full_leaf);
            break;
        }
    }
    let last_body = segs[slots]
        .strip_suffix(b"\r\n")
        .or_else(|| segs[slots].strip_suffix(b"\n"))
        .unwrap_or(segs[slots]);
    if last_body.contains(&b'\n') {
        return None;
    }
    let mut lp = 0usize;
    for _ in 0..depth {
        skip_space(last_body, &mut lp);
        if last_body.get(lp) != Some(&b'}') {
            return None;
        }
        lp += 1;
    }
    skip_space(last_body, &mut lp);
    if last_body.get(lp) == Some(&b',') {
        lp += 1;
        skip_space(last_body, &mut lp);
    }
    if lp != last_body.len() {
        return None;
    }
    Some(ParsedJsonTpl {
        obj_paths,
        top_keys,
        nested_keys,
        leaf_paths,
    })
}

#[inline]
fn json_single_line_template_slot_cached(
    segs: &[&[u8]],
    target_key: &[u8],
) -> Option<Option<usize>> {
    if segs.len() < 2 || segs.len() > MAX_SLOTS + 1 {
        return None;
    }
    let mut total_len = 0usize;
    for &s in segs {
        total_len += s.len();
    }
    if total_len > 16_384 {
        return json_single_line_template_slot_uncached(segs, target_key);
    }
    let res = JSON_TPL_SLOT_CACHE.with(|cell| {
        let mut cache = cell.borrow_mut();
        for entry in &cache.entries {
            if entry.matches_segs(segs, total_len) {
                return entry.parsed.as_ref().and_then(|p| p.lookup(target_key));
            }
        }
        let parsed = build_json_tpl_entry(segs);
        let ans = parsed.as_ref().and_then(|p| p.lookup(target_key));
        let slot_idx = cache.next_evict & 3;
        cache.next_evict = cache.next_evict.wrapping_add(1);
        let dst = &mut cache.entries[slot_idx];
        dst.seg_count = segs.len() as u8;
        dst.concat.clear();
        dst.concat.reserve(total_len);
        dst.ends[0] = 0;
        for (i, &seg) in segs.iter().enumerate() {
            dst.concat.extend_from_slice(seg);
            dst.ends[i + 1] = dst.concat.len() as u16;
        }
        dst.parsed = parsed;
        ans
    });
    debug_assert_eq!(res, json_single_line_template_slot_uncached(segs, target_key));
    res
}

pub(crate) fn split_grammar_packet_streams(
    payload: &[u8],
) -> Option<(usize, usize, usize, usize, Option<u16>, Vec<(usize, usize)>)> {
    let mut hdr = parse_grammar_for_ops(payload).ok()??;
    let s0_hdr_end = 1 + hdr.cursor.pos;
    let mut rc_tmp = hdr.row_count;
    let mut rc_uleb_len = 1usize;
    while rc_tmp >= 0x80 {
        rc_tmp >>= 7;
        rc_uleb_len += 1;
    }
    let s0_tpl_end = s0_hdr_end.checked_sub(rc_uleb_len + 1 + hdr.selector_slice.len())?;
    let end_raws = if hdr.row_count <= 1 {
        usize::from(hdr.first_is_raw)
    } else {
        usize::from(hdr.first_is_raw) + usize::from(hdr.last_is_raw)
    };
    let all_ones_rc = (hdr.template_count == 1
        && hdr.row_count > 0
        && hdr.row_count <= 0x3FFF
        && hdr.raw_count == end_raws
        && !(hdr.row_count == 1 && (hdr.first_is_raw || hdr.last_is_raw)))
        .then(|| {
            (hdr.row_count as u16)
                | (u16::from(hdr.first_is_raw) << 14)
                | (u16::from(hdr.last_is_raw) << 15)
        });
    if hdr.raw_count > 0 {
        if hdr.cursor.bytes.get(hdr.cursor.pos) == Some(&0) {
            hdr.cursor.pos += 1;
            let lz_len = hdr.cursor.uleb().ok()?;
            let _ = hdr.cursor.take(lz_len).ok()?;
        } else {
            for _ in 0..hdr.raw_count {
                let len = hdr.cursor.uleb().ok()?;
                let _ = hdr.cursor.take(len).ok()?;
            }
        }
    }
    let s0_raw_end = 1 + hdr.cursor.pos;
    let total_slots: usize = hdr.slot_counts[..hdr.template_count].iter().sum();
    if total_slots == 0 || total_slots > 64 {
        return None;
    }
    let mut spans = Vec::with_capacity(total_slots);
    for _ in 0..total_slots {
        let start = 1 + hdr.cursor.pos;
        let len = hdr.cursor.uleb().ok()?;
        let _ = hdr.cursor.take(len).ok()?;
        let end = 1 + hdr.cursor.pos;
        spans.push((start, end));
    }
    if !hdr.cursor.done() {
        return None;
    }
    Some((
        hdr.p_after_raw_len,
        s0_tpl_end,
        s0_hdr_end,
        s0_raw_end,
        all_ones_rc,
        spans,
    ))
}

pub(crate) fn expand_v6_compact_s0(
    canon_s0: &[u8],
    rc_packed: usize,
    out: &mut Vec<u8>,
) -> Option<(usize, usize)> {
    let rc = rc_packed & 0x3FFF;
    let first_raw = (rc_packed & 0x4000) != 0;
    let last_raw = (rc_packed & 0x8000) != 0;
    if rc == 0 || (rc == 1 && (first_raw || last_raw)) {
        return None;
    }
    let hdr = parse_grammar_for_ops(canon_s0).ok()??;
    if hdr.template_count != 1 || rc > hdr.raw_len.saturating_add(1) {
        return None;
    }
    let s0_hdr_end = 1 + hdr.cursor.pos;
    let mut rc_tmp = hdr.row_count;
    let mut rc_uleb_len = 1usize;
    while rc_tmp >= 0x80 {
        rc_tmp >>= 7;
        rc_uleb_len += 1;
    }
    let tpl_end = s0_hdr_end.checked_sub(rc_uleb_len + 1 + hdr.selector_slice.len())?;
    let start = out.len();
    out.extend_from_slice(&canon_s0[..tpl_end]);
    let mut v = rc;
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
    out.push(1u8);
    let sel_start = out.len();
    out.resize(sel_start + (rc >> 3), 0xFF);
    if (rc & 7) != 0 {
        out.push((1u8 << (rc & 7)) - 1);
    }
    if first_raw {
        out[sel_start] &= !1u8;
    }
    if last_raw {
        out[sel_start + ((rc - 1) >> 3)] &= !(1u8 << ((rc - 1) & 7));
    }
    Some((start, out.len()))
}

pub(crate) fn expand_v6_tpl_s0(
    canon_s0: &[u8],
    sel_tail: &[u8],
    out: &mut Vec<u8>,
) -> Option<(usize, usize)> {
    if sel_tail.len() < 3 {
        return None;
    }
    let hdr = parse_grammar_for_ops(canon_s0).ok()??;
    let s0_hdr_end = 1 + hdr.cursor.pos;
    let mut rc_tmp = hdr.row_count;
    let mut rc_uleb_len = 1usize;
    while rc_tmp >= 0x80 {
        rc_tmp >>= 7;
        rc_uleb_len += 1;
    }
    let tpl_end = s0_hdr_end.checked_sub(rc_uleb_len + 1 + hdr.selector_slice.len())?;
    let start = out.len();
    out.extend_from_slice(&canon_s0[..tpl_end]);
    out.extend_from_slice(sel_tail);
    match parse_grammar_for_ops(&out[start..]) {
        Ok(Some(h2)) if h2.cursor.done() => Some((start, out.len())),
        _ => {
            out.truncate(start);
            None
        }
    }
}

pub(crate) fn resolve_stream0_required_slots(
    stream0: &[u8],
    targets: &[crate::ops::ColTarget<'_>],
) -> Option<u64> {
    let hdr = parse_grammar_for_ops(stream0).ok()??;
    let total_slots: usize = hdr.slot_counts[..hdr.template_count].iter().sum();
    if total_slots == 0 || total_slots > 64 {
        return None;
    }
    let mut mask = 0u64;
    let mut slot_base = 0usize;
    for t in 0..hdr.template_count {
        let segs = hdr.segs(t);
        let slots = hdr.slot_counts[t];
        if !segs.last().is_some_and(|s| s.ends_with(b"\n")) {
            return None;
        }
        let tpl_delim = if t == 0 {
            hdr.tpl0_delim
        } else {
            delimited_template_delim(segs)
        };
        for &col in targets {
            match col {
                crate::ops::ColTarget::JsonKey(key) => {
                    let found = json_single_line_template_slot(segs, key.as_bytes())?;
                    if let Some(s) = found {
                        mask |= 1u64 << (slot_base + s);
                    }
                }
                crate::ops::ColTarget::Delimited { delim, index } => {
                    if tpl_delim != Some(delim) {
                        return None;
                    }
                    if index < slots {
                        mask |= 1u64 << (slot_base + index);
                    }
                }
            }
        }
        slot_base += slots;
    }
    Some(mask)
}
