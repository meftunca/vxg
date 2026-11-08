/// Maximum inner key-value pairs flattened from a single nested JSON sub-object.
const MAX_SUBOBJ_SLOTS: usize = 12;
/// Total slot budget within which sub-objects are flattened (stays `< 32` for stack inline unpack).
const FLATTEN_SLOT_BUDGET: usize = 30;

/// Attempt to flatten one immediate nested JSON object `data[sub_start..sub_end]` into its
/// member value spans `(val_start, val_end)` (relative to `data`).
///
/// Returns `Some(inner_count)` (`1..=MAX_SUBOBJ_SLOTS`) when `sub` is a non-empty single-line
/// JSON object within `max_inner` slots; otherwise returns `None` so the caller keeps the whole
/// `{...}` span as a single opaque value slot.
#[inline]
fn try_flatten_json_subobject(
    data: &[u8],
    sub_start: usize,
    sub_end: usize,
    max_inner: usize,
    inner_out: &mut [(u32, u32); MAX_SUBOBJ_SLOTS],
) -> Option<usize> {
    let sub_len = sub_end.saturating_sub(sub_start);
    if !(6..=4096).contains(&sub_len) || max_inner == 0 {
        return None;
    }
    let sub = &data[sub_start..sub_end];
    if sub.first() != Some(&b'{') || sub.last() != Some(&b'}') || find_byte(b'\n', sub).is_some() {
        return None;
    }
    let cap = max_inner.min(MAX_SUBOBJ_SLOTS);
    let mut pos = sub_start + 1;
    let mut count = 0usize;
    loop {
        skip_space(data, &mut pos);
        if pos >= sub_end {
            return None;
        }
        if data[pos] == b'}' {
            pos += 1;
            break;
        }
        if data[pos] != b'"' {
            return None;
        }
        pos = scan_string(data, pos)?;
        if pos >= sub_end {
            return None;
        }
        skip_space(data, &mut pos);
        if pos >= sub_end || data[pos] != b':' {
            return None;
        }
        pos += 1;
        skip_space(data, &mut pos);
        if pos >= sub_end {
            return None;
        }
        let v_start = pos;
        let v_end = scan_value(data, pos)?;
        if v_end > sub_end || count >= cap {
            return None;
        }
        inner_out[count] = (v_start as u32, v_end as u32);
        count += 1;
        pos = v_end;
        skip_space(data, &mut pos);
        if pos >= sub_end {
            return None;
        }
        match data[pos] {
            b',' => pos += 1,
            b'}' => {
                pos += 1;
                break;
            }
            _ => return None,
        }
    }
    (pos == sub_end && count > 0).then_some(count)
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
