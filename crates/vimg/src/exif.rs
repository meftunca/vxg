//! Minimal, zero-dependency JPEG EXIF Orientation (`0x0112`, values `1..=8`) parser
//! so smartphone portrait/rotated JPEGs are automatically oriented upright on encode.

/// Parses a JPEG byte slice and returns its EXIF Orientation (`1..=8`), defaulting to `1` (normal).
pub fn parse_jpeg_exif_orientation(data: &[u8]) -> u8 {
    if data.len() < 4 || data[0] != 0xFF || data[1] != 0xD8 {
        return 1;
    }
    let mut pos = 2usize;
    while pos + 4 <= data.len() {
        if data[pos] != 0xFF {
            break;
        }
        while pos < data.len() && data[pos] == 0xFF {
            pos += 1;
        }
        if pos >= data.len() {
            break;
        }
        let marker = data[pos];
        pos += 1;
        if marker == 0xD9 || marker == 0xDA {
            break; // EOI or SOS
        }
        if (0xD0..=0xD8).contains(&marker) || marker == 0x01 {
            continue;
        }
        if pos + 2 > data.len() {
            break;
        }
        let seg_len = u16::from_be_bytes([data[pos], data[pos + 1]]) as usize;
        if seg_len < 2 || pos + seg_len > data.len() {
            break;
        }
        if marker == 0xE1 && seg_len >= 16 {
            let seg = &data[pos + 2..pos + seg_len];
            if seg.starts_with(b"Exif\0\0") {
                if let Some(ori) = parse_tiff_orientation(&seg[6..]) {
                    return ori;
                }
            }
        }
        pos += seg_len;
    }
    1
}

fn parse_tiff_orientation(tiff: &[u8]) -> Option<u8> {
    if tiff.len() < 8 {
        return None;
    }
    let little = match &tiff[0..2] {
        b"II" => true,
        b"MM" => false,
        _ => return None,
    };
    let r16 = |o: usize| -> Option<u16> {
        let s = tiff.get(o..o + 2)?;
        Some(if little { u16::from_le_bytes([s[0], s[1]]) } else { u16::from_be_bytes([s[0], s[1]]) })
    };
    let r32 = |o: usize| -> Option<u32> {
        let s = tiff.get(o..o + 4)?;
        Some(if little {
            u32::from_le_bytes([s[0], s[1], s[2], s[3]])
        } else {
            u32::from_be_bytes([s[0], s[1], s[2], s[3]])
        })
    };
    if r16(2)? != 42 {
        return None;
    }
    let ifd0 = r32(4)? as usize;
    let count = r16(ifd0)? as usize;
    let mut entry = ifd0 + 2;
    for _ in 0..count.min(256) {
        if entry + 12 > tiff.len() {
            break;
        }
        let tag = r16(entry)?;
        if tag == 0x0112 {
            let val = r16(entry + 8)?;
            if (1..=8).contains(&val) {
                return Some(val as u8);
            }
        }
        entry += 12;
    }
    None
}

/// Applies EXIF orientation `1..=8` to packed RGB/RGBA pixels, returning `(pixels, new_w, new_h)`.
pub fn apply_exif_orientation(
    pixels: Vec<u8>,
    w: u32,
    h: u32,
    ch: u8,
    orientation: u8,
) -> (Vec<u8>, u32, u32) {
    if orientation <= 1 || orientation > 8 {
        return (pixels, w, h);
    }
    let (wu, hu, cu) = (w as usize, h as usize, ch as usize);
    let swap = matches!(orientation, 5..=8);
    let (nw, nh) = if swap { (hu, wu) } else { (wu, hu) };
    let mut out = vec![0u8; nw * nh * cu];
    for y in 0..hu {
        for x in 0..wu {
            let (dx, dy) = match orientation {
                2 => (wu - 1 - x, y),
                3 => (wu - 1 - x, hu - 1 - y),
                4 => (x, hu - 1 - y),
                5 => (y, x),
                6 => (hu - 1 - y, x),
                7 => (hu - 1 - y, wu - 1 - x),
                8 => (y, wu - 1 - x),
                _ => (x, y),
            };
            let s = (y * wu + x) * cu;
            let d = (dy * nw + dx) * cu;
            out[d..d + cu].copy_from_slice(&pixels[s..s + cu]);
        }
    }
    (out, nw as u32, nh as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_little_and_big_endian_exif_orientation() {
        // Minimal JPEG SOI + APP1 Exif (II, tag 0x0112 = 6 = 90° CW)
        let mut jpg = vec![0xFF, 0xD8, 0xFF, 0xE1, 0x00, 30];
        jpg.extend_from_slice(b"Exif\0\0");
        jpg.extend_from_slice(b"II");
        jpg.extend_from_slice(&42u16.to_le_bytes());
        jpg.extend_from_slice(&8u32.to_le_bytes());
        jpg.extend_from_slice(&1u16.to_le_bytes()); // 1 IFD entry
        jpg.extend_from_slice(&0x0112u16.to_le_bytes());
        jpg.extend_from_slice(&3u16.to_le_bytes());
        jpg.extend_from_slice(&1u32.to_le_bytes());
        jpg.extend_from_slice(&6u16.to_le_bytes());
        jpg.extend_from_slice(&0u16.to_le_bytes());
        assert_eq!(parse_jpeg_exif_orientation(&jpg), 6);

        let (rot, nw, nh) = apply_exif_orientation(vec![1, 2, 3, 4, 5, 6], 2, 1, 3, 6);
        assert_eq!((nw, nh), (1, 2));
        assert_eq!(rot, vec![1, 2, 3, 4, 5, 6]);
    }
}
