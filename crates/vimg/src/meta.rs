//! Extensible trailing TLV metadata block for `VIMG1` (`FLAG_HAS_META`):
//! - `TAG_CICP (1)`: ITU-T H.273 / ISO 23091-2 HDR color signaling (`BT.2020`, `PQ`, `HLG`, `Display P3`, `sRGB`)
//! - `TAG_EXIF (2)`: Raw EXIF TIFF/APP1 camera metadata
//! - `TAG_XMP  (3)`: Raw UTF-8 XMP XML packet
//!
//! Stored at `[chunks_end..total_bytes]` with trailing hardware `CRC32C` so partial HTTP-Range
//! crop/resize queries never fetch metadata bytes unless requested.

use core::fmt;

use crate::checksum::crc32c;
use crate::{Error, Result};

pub const MAX_META_BYTES: u32 = 16 * 1024 * 1024; // 16 MiB metadata cap

const TAG_CICP: u8 = 1;
const TAG_EXIF: u8 = 2;
const TAG_XMP: u8 = 3;

/// ITU-T H.273 / ISO 23091-2 Coding-Independent Code Points (CICP) for SDR & HDR color spaces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cicp {
    pub color_primaries: u8,
    pub transfer_characteristics: u8,
    pub matrix_coefficients: u8,
    pub full_range: bool,
}

impl Cicp {
    /// Standard 8-bit sRGB (BT.709 primaries, sRGB transfer, Identity RGB, full range).
    pub const SRGB: Self = Self {
        color_primaries: 1,
        transfer_characteristics: 13,
        matrix_coefficients: 0,
        full_range: true,
    };
    /// Apple Display P3 (DCI-P3 D65 primaries, sRGB transfer, Identity RGB, full range).
    pub const DISPLAY_P3: Self = Self {
        color_primaries: 12,
        transfer_characteristics: 13,
        matrix_coefficients: 0,
        full_range: true,
    };
    /// HDR10 / Rec.2100 PQ (BT.2020 primaries, SMPTE ST.2084 PQ transfer, Identity RGB, full range).
    pub const BT2020_PQ: Self = Self {
        color_primaries: 9,
        transfer_characteristics: 16,
        matrix_coefficients: 0,
        full_range: true,
    };
    /// Rec.2100 HLG (BT.2020 primaries, ARIB STD-B67 HLG transfer, Identity RGB, full range).
    pub const BT2020_HLG: Self = Self {
        color_primaries: 9,
        transfer_characteristics: 18,
        matrix_coefficients: 0,
        full_range: true,
    };

    pub fn from_preset(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "srgb" => Some(Self::SRGB),
            "p3" | "display-p3" | "display_p3" => Some(Self::DISPLAY_P3),
            "pq" | "hdr10" | "bt2020-pq" => Some(Self::BT2020_PQ),
            "hlg" | "bt2020-hlg" => Some(Self::BT2020_HLG),
            _ => None,
        }
    }
}

impl fmt::Display for Cicp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if *self == Self::SRGB {
            write!(f, "sRGB")
        } else if *self == Self::DISPLAY_P3 {
            write!(f, "Display-P3")
        } else if *self == Self::BT2020_PQ {
            write!(f, "BT.2020/PQ")
        } else if *self == Self::BT2020_HLG {
            write!(f, "BT.2020/HLG")
        } else {
            write!(
                f,
                "CICP({}/{}/{}/{})",
                self.color_primaries,
                self.transfer_characteristics,
                self.matrix_coefficients,
                u8::from(self.full_range)
            )
        }
    }
}

/// Optional rich metadata stored in a `VIMG1` archive's trailing TLV section.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VimgMetadata {
    pub cicp: Option<Cicp>,
    pub exif: Option<Vec<u8>>,
    pub xmp: Option<Vec<u8>>,
}

impl VimgMetadata {
    pub fn is_empty(&self) -> bool {
        self.cicp.is_none() && self.exif.is_none() && self.xmp.is_none()
    }

    pub fn encode_tlv(&self, with_crc: bool) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        if let Some(c) = self.cicp {
            out.push(TAG_CICP);
            out.extend_from_slice(&4u32.to_le_bytes());
            out.push(c.color_primaries);
            out.push(c.transfer_characteristics);
            out.push(c.matrix_coefficients);
            out.push(u8::from(c.full_range));
        }
        if let Some(ref exif) = self.exif {
            if !exif.is_empty() {
                if exif.len() > MAX_META_BYTES as usize {
                    return Err(Error::InvalidInput("EXIF exceeds MAX_META_BYTES"));
                }
                out.push(TAG_EXIF);
                out.extend_from_slice(&(exif.len() as u32).to_le_bytes());
                out.extend_from_slice(exif);
            }
        }
        if let Some(ref xmp) = self.xmp {
            if !xmp.is_empty() {
                if xmp.len() > MAX_META_BYTES as usize {
                    return Err(Error::InvalidInput("XMP exceeds MAX_META_BYTES"));
                }
                out.push(TAG_XMP);
                out.extend_from_slice(&(xmp.len() as u32).to_le_bytes());
                out.extend_from_slice(xmp);
            }
        }
        if out.len() > MAX_META_BYTES as usize {
            return Err(Error::InvalidInput("metadata block exceeds MAX_META_BYTES"));
        }
        if !out.is_empty() && with_crc {
            let crc = crc32c(&out);
            out.extend_from_slice(&crc.to_le_bytes());
        }
        Ok(out)
    }

    pub fn decode_tlv(buf: &[u8], has_crc: bool, verify_crc: bool) -> Result<Self> {
        let body = if has_crc {
            if buf.len() < 4 {
                return Err(Error::Truncated);
            }
            let body = &buf[..buf.len() - 4];
            if verify_crc {
                let expected = u32::from_le_bytes(buf[buf.len() - 4..].try_into().unwrap());
                let actual = crc32c(body);
                if actual != expected {
                    return Err(Error::ChecksumMismatch { expected, actual });
                }
            }
            body
        } else {
            buf
        };

        let mut meta = Self::default();
        let mut pos = 0usize;
        while pos < body.len() {
            if pos + 5 > body.len() {
                return Err(Error::Truncated);
            }
            let tag = body[pos];
            let len = u32::from_le_bytes(body[pos + 1..pos + 5].try_into().unwrap()) as usize;
            pos += 5;
            let end = pos.checked_add(len).ok_or(Error::InvalidFormat("metadata TLV overflow"))?;
            if end > body.len() {
                return Err(Error::Truncated);
            }
            let payload = &body[pos..end];
            match tag {
                TAG_CICP => {
                    if payload.len() != 4 {
                        return Err(Error::InvalidFormat("invalid CICP tag length"));
                    }
                    meta.cicp = Some(Cicp {
                        color_primaries: payload[0],
                        transfer_characteristics: payload[1],
                        matrix_coefficients: payload[2],
                        full_range: payload[3] != 0,
                    });
                }
                TAG_EXIF => meta.exif = Some(payload.to_vec()),
                TAG_XMP => meta.xmp = Some(payload.to_vec()),
                _ => {} // Forward-compatible skip of unknown tags
            }
            pos = end;
        }
        Ok(meta)
    }
}
