//! Single-record projection reference (`project_line`) and column-list validation.

use crate::{Error, Result};

use super::line::{
    encode_json_str, find_json_obj_path_span, scan_json_str_end, scan_json_val_end,
    unescape_json_str, ColTarget,
};

pub(crate) fn validate_projection_cols(cols: &[ColTarget<'_>]) -> Result<()> {
    if cols.is_empty() || cols.len() > 128 {
        return Err(Error::InvalidOption("projection requires 1..=128 columns"));
    }
    match cols[0] {
        ColTarget::JsonKey(_) => {
            if !cols.iter().all(|c| matches!(c, ColTarget::JsonKey(_))) {
                return Err(Error::InvalidOption(
                    "projection cannot mix JSON keys and delimited columns",
                ));
            }
        }
        ColTarget::Delimited { delim: d0, .. } => {
            if matches!(d0, b'\n' | b'\r') {
                return Err(Error::InvalidOption("invalid projection delimiter"));
            }
            for &c in &cols[1..] {
                let ColTarget::Delimited { delim, .. } = c else {
                    return Err(Error::InvalidOption(
                        "projection cannot mix JSON keys and delimited columns",
                    ));
                };
                if delim != d0 {
                    return Err(Error::InvalidOption(
                        "projection columns must share the same delimiter",
                    ));
                }
            }
        }
    }
    Ok(())
}

#[inline]
fn skip_ws(data: &[u8], pos: &mut usize) {
    while *pos < data.len() && matches!(data[*pos], b' ' | b'\t' | b'\r' | b'\n') {
        *pos += 1;
    }
}

/// Exact single-record projection reference: appends the projected record to `dst` and returns
/// `true` iff `line` is a valid structured record matching the mode of `cols`.
pub fn project_line(line: &[u8], cols: &[ColTarget<'_>], dst: &mut Vec<u8>) -> bool {
    if cols.is_empty() || cols.len() > 128 {
        return false;
    }
    let (ending, body_len): (&[u8], usize) = if line.ends_with(b"\r\n") {
        (b"\r\n", line.len() - 2)
    } else if line.ends_with(b"\n") {
        (b"\n", line.len() - 1)
    } else {
        (b"", line.len())
    };
    let mut spans: [Option<(usize, usize)>; 128] = [None; 128];
    let mut exact_matched: [bool; 128] = [false; 128];
    match cols[0] {
        ColTarget::JsonKey(_) => {
            let mut pos = 0usize;
            skip_ws(line, &mut pos);
            if line.get(pos) != Some(&b'{') {
                return false;
            }
            pos += 1;
            loop {
                skip_ws(line, &mut pos);
                if line.get(pos) == Some(&b'}') {
                    pos += 1;
                    break;
                }
                let k_start = pos;
                let Some(k_end) = scan_json_str_end(line, pos) else {
                    return false;
                };
                pos = k_end;
                skip_ws(line, &mut pos);
                if line.get(pos) != Some(&b':') {
                    return false;
                }
                pos += 1;
                skip_ws(line, &mut pos);
                let v_start = pos;
                let Some(v_end) = scan_json_val_end(line, pos) else {
                    return false;
                };
                if let Some(k_dec) = unescape_json_str(&line[k_start..k_end]) {
                    let kb = k_dec.as_ref();
                    for (j, &col) in cols.iter().enumerate() {
                        let ColTarget::JsonKey(tk) = col else {
                            return false;
                        };
                        let tkb = tk.as_bytes();
                        if kb == tkb {
                            if !exact_matched[j] {
                                spans[j] = Some((v_start, v_end));
                                exact_matched[j] = true;
                            }
                        } else if spans[j].is_none()
                            && line.get(v_start) == Some(&b'{')
                            && tkb.len() > kb.len()
                            && tkb.starts_with(kb)
                            && tkb[kb.len()] == b'.'
                        {
                            if let Some((ss, se)) = find_json_obj_path_span(
                                &line[v_start..v_end],
                                &tkb[kb.len() + 1..],
                                false,
                            ) {
                                spans[j] = Some((v_start + ss, v_start + se));
                            }
                        }
                    }
                }
                pos = v_end;
                skip_ws(line, &mut pos);
                match line.get(pos) {
                    Some(b',') => pos += 1,
                    Some(b'}') => {
                        pos += 1;
                        break;
                    }
                    _ => return false,
                }
            }
            skip_ws(line, &mut pos);
            if line.get(pos) == Some(&b',') {
                pos += 1;
                skip_ws(line, &mut pos);
            }
            if pos != line.len() {
                return false;
            }
            dst.push(b'{');
            let mut first_out = true;
            for (j, &col) in cols.iter().enumerate() {
                if let Some((vs, ve)) = spans[j] {
                    let ColTarget::JsonKey(tk) = col else {
                        unreachable!();
                    };
                    if !first_out {
                        dst.push(b',');
                    }
                    first_out = false;
                    encode_json_str(tk.as_bytes(), dst);
                    dst.push(b':');
                    dst.extend_from_slice(&line[vs..ve]);
                }
            }
            dst.push(b'}');
            dst.extend_from_slice(ending);
            true
        }
        ColTarget::Delimited { delim, .. } => {
            if matches!(delim, b'\n' | b'\r') || body_len == 0 {
                return false;
            }
            let body = &line[..body_len];
            let relaxed = delim == b',' || delim == b'\t';
            let mut pos = 0usize;
            let mut count = 0usize;
            loop {
                if pos >= body_len {
                    return false;
                }
                let first = body[pos];
                let end = if first == b'"' {
                    let Some(e) = scan_json_str_end(body, pos) else {
                        return false;
                    };
                    e
                } else {
                    if first == delim || first <= b' ' || first >= 0x7f {
                        return false;
                    }
                    let mut e = pos + 1;
                    while e < body_len {
                        let b = body[e];
                        if b == delim {
                            break;
                        }
                        if b == b'"' || (b < b' ' && b != b'\t') || b == 0x7f {
                            return false;
                        }
                        if !relaxed
                            && (b <= b' '
                                || b >= 0x7f
                                || matches!(b, b'(' | b')' | b';' | b'{' | b'}' | b'[' | b']'))
                        {
                            return false;
                        }
                        e += 1;
                    }
                    e
                };
                for (j, &col) in cols.iter().enumerate() {
                    let ColTarget::Delimited { delim: d, index } = col else {
                        return false;
                    };
                    if d != delim {
                        return false;
                    }
                    if index == count {
                        spans[j] = Some((pos, end));
                    }
                }
                count += 1;
                pos = end;
                if pos == body_len {
                    break;
                }
                if body[pos] != delim {
                    return false;
                }
                pos += 1;
            }
            if count < 2 {
                return false;
            }
            for (j, slot) in spans.iter().enumerate().take(cols.len()) {
                if j > 0 {
                    dst.push(delim);
                }
                if let Some((fs, fe)) = *slot {
                    dst.extend_from_slice(&body[fs..fe]);
                }
            }
            dst.extend_from_slice(ending);
            true
        }
    }
}
