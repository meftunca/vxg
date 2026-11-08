/// Search a validated `{...}` JSON object slice `obj` (starting at `{`) for either an exact key
/// `key` or, if `key` contains `.`, a nested sub-object path `prefix.rest`.
/// When `require_full_line` is `true`, trailing bytes after `}` (other than optional `,` + ws) reject.
pub(crate) fn find_json_obj_path_span(
    obj: &[u8],
    key: &[u8],
    require_full_line: bool,
) -> Option<(usize, usize)> {
    let mut pos = 0usize;
    skip_json_ws(obj, &mut pos);
    if obj.get(pos) != Some(&b'{') {
        return None;
    }
    pos += 1;
    let has_dot = key.contains(&b'.');
    let mut exact_found: Option<(usize, usize)> = None;
    let mut nested_found: Option<(usize, usize)> = None;
    loop {
        skip_json_ws(obj, &mut pos);
        if obj.get(pos) == Some(&b'}') {
            pos += 1;
            break;
        }
        let k_start = pos;
        let k_end = scan_json_str_end(obj, pos)?;
        pos = k_end;
        skip_json_ws(obj, &mut pos);
        if obj.get(pos) != Some(&b':') {
            return None;
        }
        pos += 1;
        skip_json_ws(obj, &mut pos);
        let v_start = pos;
        let v_end = scan_json_val_end(obj, pos)?;
        if exact_found.is_none() {
            if let Some(k_decoded) = unescape_json_str(&obj[k_start..k_end]) {
                let kb = k_decoded.as_ref();
                if kb == key {
                    exact_found = Some((v_start, v_end));
                } else if nested_found.is_none()
                    && has_dot
                    && obj.get(v_start) == Some(&b'{')
                    && key.len() > kb.len()
                    && key.starts_with(kb)
                    && key[kb.len()] == b'.'
                {
                    let sub_slice = &obj[v_start..v_end];
                    let rest = &key[kb.len() + 1..];
                    if let Some((ss, se)) = find_json_obj_path_span(sub_slice, rest, false) {
                        nested_found = Some((v_start + ss, v_start + se));
                    }
                }
            }
        }
        pos = v_end;
        skip_json_ws(obj, &mut pos);
        match obj.get(pos) {
            Some(b',') => pos += 1,
            Some(b'}') => {
                pos += 1;
                break;
            }
            _ => return None,
        }
    }
    if require_full_line {
        skip_json_ws(obj, &mut pos);
        if obj.get(pos) == Some(&b',') {
            pos += 1;
            skip_json_ws(obj, &mut pos);
        }
        if pos != obj.len() {
            return None;
        }
    }
    exact_found.or(nested_found)
}

/// Locate the byte span `(start, end)` of the first top-level `"key":` value (or dotted nested
/// path `"parent.child":` value when no literal `"parent.child"` top-level key exists) in a valid
/// top-level JSON object record `line`.
#[inline]
pub fn find_json_key_span(line: &[u8], key: &[u8]) -> Option<(usize, usize)> {
    find_json_obj_path_span(line, key, true)
}
