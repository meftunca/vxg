/// Maximum leaf key-value pairs flattened from a single top-level nested JSON sub-object tree.
const MAX_SUBOBJ_SLOTS: usize = 24;
/// Maximum JSON object nesting depth flattened (`1` = root, `4` = `a.b.c.d` leaf values).
const MAX_NEST_DEPTH: usize = 4;
/// Total slot budget within which sub-objects are flattened (stays `< 32` for stack inline unpack).
const FLATTEN_SLOT_BUDGET: usize = 30;

/// Attempt to recursively flatten nested JSON object `data[sub_start..sub_end]` up to `depth_rem`
/// levels into leaf member value spans `(val_start, val_end)` (relative to `data`).
///
/// Appends flattened spans onto `inner_out[*out_len..]` and returns `true` when `sub` is a
/// non-empty single-line JSON object fitting within `cap` total slots; otherwise restores
/// `*out_len` and returns `false` so the caller keeps `{...}` as a single opaque value slot.
fn try_flatten_json_subobject_rec(
    data: &[u8],
    sub_start: usize,
    sub_end: usize,
    cap: usize,
    depth_rem: usize,
    inner_out: &mut [(u32, u32); MAX_SUBOBJ_SLOTS],
    out_len: &mut usize,
) -> bool {
    let base_len = *out_len;
    let sub_len = sub_end.saturating_sub(sub_start);
    if !(6..=4096).contains(&sub_len) || depth_rem == 0 || base_len >= cap {
        return false;
    }
    let sub = &data[sub_start..sub_end];
    if sub.first() != Some(&b'{') || sub.last() != Some(&b'}') || find_byte(b'\n', sub).is_some() {
        return false;
    }
    let eff_cap = cap.min(MAX_SUBOBJ_SLOTS);
    let mut pos = sub_start + 1;
    let mut fields_seen = 0usize;
    loop {
        skip_space(data, &mut pos);
        if pos >= sub_end {
            *out_len = base_len;
            return false;
        }
        if data[pos] == b'}' {
            pos += 1;
            break;
        }
        if data[pos] != b'"' {
            *out_len = base_len;
            return false;
        }
        let Some(k_end) = scan_string(data, pos) else {
            *out_len = base_len;
            return false;
        };
        pos = k_end;
        if pos >= sub_end {
            *out_len = base_len;
            return false;
        }
        skip_space(data, &mut pos);
        if pos >= sub_end || data[pos] != b':' {
            *out_len = base_len;
            return false;
        }
        pos += 1;
        skip_space(data, &mut pos);
        if pos >= sub_end {
            *out_len = base_len;
            return false;
        }
        let v_start = pos;
        let Some(v_end) = scan_value(data, pos) else {
            *out_len = base_len;
            return false;
        };
        if v_end > sub_end {
            *out_len = base_len;
            return false;
        }
        let mut child_flattened = false;
        if depth_rem > 1 && data[v_start] == b'{' && *out_len < eff_cap {
            let mark = *out_len;
            if try_flatten_json_subobject_rec(
                data,
                v_start,
                v_end,
                eff_cap,
                depth_rem - 1,
                inner_out,
                out_len,
            ) {
                child_flattened = true;
            } else {
                *out_len = mark;
            }
        }
        if !child_flattened {
            if *out_len >= eff_cap {
                *out_len = base_len;
                return false;
            }
            inner_out[*out_len] = (v_start as u32, v_end as u32);
            *out_len += 1;
        }
        fields_seen += 1;
        pos = v_end;
        skip_space(data, &mut pos);
        if pos >= sub_end {
            *out_len = base_len;
            return false;
        }
        match data[pos] {
            b',' => pos += 1,
            b'}' => {
                pos += 1;
                break;
            }
            _ => {
                *out_len = base_len;
                return false;
            }
        }
    }
    if pos == sub_end && fields_seen > 0 && *out_len > base_len {
        true
    } else {
        *out_len = base_len;
        false
    }
}

#[inline]
fn try_flatten_json_subobject(
    data: &[u8],
    sub_start: usize,
    sub_end: usize,
    max_inner: usize,
    inner_out: &mut [(u32, u32); MAX_SUBOBJ_SLOTS],
) -> Option<usize> {
    let mut out_len = 0usize;
    try_flatten_json_subobject_rec(
        data,
        sub_start,
        sub_end,
        max_inner.min(MAX_SUBOBJ_SLOTS),
        MAX_NEST_DEPTH - 1,
        inner_out,
        &mut out_len,
    )
    .then_some(out_len)
}

#[inline]
fn parse_json_record_spans(
    data: &[u8],
    mut pos: usize,
    spans: &mut [(u32, u32); MAX_SLOTS],
    overflow: &mut Vec<(u32, u32)>,
) -> Option<usize> {
    pos += 1;
    let end_body = data.len().saturating_sub(1);
    let single_line = pos <= end_body && find_byte(b'\n', &data[pos..end_body]).is_none();
    let mut count = 0usize;
    let mut inner_buf = [(0u32, 0u32); MAX_SUBOBJ_SLOTS];
    overflow.clear();
    loop {
        skip_space(data, &mut pos);
        if data.get(pos) == Some(&b'}') {
            pos += 1;
            break;
        }
        pos = scan_string(data, pos)?;
        skip_space(data, &mut pos);
        if data.get(pos) != Some(&b':') {
            return None;
        }
        pos += 1;
        skip_space(data, &mut pos);
        let start = pos;
        let end = scan_value(data, pos)?;
        let flattened = if single_line
            && data.get(start) == Some(&b'{')
            && count < FLATTEN_SLOT_BUDGET
        {
            try_flatten_json_subobject(
                data,
                start,
                end,
                FLATTEN_SLOT_BUDGET - count,
                &mut inner_buf,
            )
        } else {
            None
        };
        if let Some(inner_n) = flattened {
            for &(is, ie) in &inner_buf[..inner_n] {
                spans[count] = (is, ie);
                count += 1;
            }
        } else {
            if count < MAX_SLOTS {
                spans[count] = (start as u32, end as u32);
            } else {
                overflow.push((start as u32, end as u32));
            }
            count += 1;
        }
        pos = end;
        skip_space(data, &mut pos);
        match data.get(pos) {
            Some(b',') => pos += 1,
            Some(b'}') => {
                pos += 1;
                break;
            }
            _ => return None,
        }
    }
    skip_space(data, &mut pos);
    if data.get(pos) == Some(&b',') {
        pos += 1;
        skip_space(data, &mut pos);
    }
    if pos != data.len() {
        return None;
    }
    (count > 0).then_some(count)
}
