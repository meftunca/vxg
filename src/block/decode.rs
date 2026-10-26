use crate::{simd::Engine, Error, Result};

use super::{COLUMN, LZ, RAW};

#[cfg_attr(not(feature = "parser-no-inline"), inline(always))]
fn u32_at(src: &[u8], pos: &mut usize) -> Result<u32> {
    let end = pos.checked_add(4).ok_or(Error::SizeOverflow)?;
    let bytes = src.get(*pos..end).ok_or(Error::Truncated)?;
    *pos = end;
    Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
}
#[cfg_attr(not(feature = "parser-no-inline"), inline(always))]
fn u16_at(src: &[u8], pos: &mut usize) -> Result<u16> {
    let end = pos.checked_add(2).ok_or(Error::SizeOverflow)?;
    let bytes = src.get(*pos..end).ok_or(Error::Truncated)?;
    *pos = end;
    Ok(u16::from_le_bytes(bytes.try_into().unwrap()))
}

/// Copy a proven-disjoint hot opcode-7 match without routing the overwhelmingly
/// 8..=63-byte JSON run buckets through a variable-size libc call. The caller
/// establishes the range, initialized-prefix, and non-overlap proof before
/// entering this function; this helper never services an overlapping LZ copy.
///
/// Each load is unaligned but in-range. Source and destination are disjoint, so
/// the small number of overlapping *destination* stores in the tail case repeat
/// the same source bytes and preserve the exact requested range.
#[inline(always)]
unsafe fn copy_nonoverlapping_short(src: *const u8, dst: *mut u8, len: usize) {
    debug_assert!((8..=63).contains(&len));
    #[inline(always)]
    unsafe fn copy_word(src: *const u8, dst: *mut u8, offset: usize) {
        // SAFETY: the enclosing caller proves [offset, offset + 8) lies in
        // both disjoint requested ranges. Unaligned accesses are intentional.
        let word = unsafe { std::ptr::read_unaligned(src.add(offset).cast::<u64>()) };
        unsafe { std::ptr::write_unaligned(dst.add(offset).cast::<u64>(), word) };
    }

    // SAFETY: `len >= 8`, so the first word is fully in range.
    unsafe { copy_word(src, dst, 0) };
    if len >= 16 {
        // SAFETY: this branch proves the second word is fully in range.
        unsafe { copy_word(src, dst, 8) };
    }
    if len >= 24 {
        // SAFETY: this branch proves the word is fully in range.
        unsafe { copy_word(src, dst, 16) };
    }
    if len >= 32 {
        // SAFETY: this branch proves the word is fully in range.
        unsafe { copy_word(src, dst, 24) };
    }
    if len >= 40 {
        // SAFETY: this branch proves the word is fully in range.
        unsafe { copy_word(src, dst, 32) };
    }
    if len >= 48 {
        // SAFETY: this branch proves the word is fully in range.
        unsafe { copy_word(src, dst, 40) };
    }
    if len >= 56 {
        // SAFETY: this branch proves the word is fully in range.
        unsafe { copy_word(src, dst, 48) };
    }
    // SAFETY: `len >= 8`, so the exact tail word is fully in range. It may
    // overlap a previous destination store, but both stores contain the same
    // source bytes at their corresponding offsets.
    unsafe { copy_word(src, dst, len - 8) };
}
#[derive(Debug, Clone, Copy)]
pub(crate) struct Command {
    pub op: u8,
    pub lit: usize,
    pub run: usize,
    pub distance: usize,
    pub seed: [u8; 8],
    pub literal_start: usize,
}
pub(crate) struct Parser<'a> {
    control: &'a [u8],
    attrs: &'a [u8],
    pub literals: &'a [u8],
    ci: usize,
    ai: usize,
    li: usize,
    pos: usize,
    raw_len: usize,
    remaining: usize,
    recent: usize,
    short_distance: bool,
}
impl<'a> Parser<'a> {
    pub fn new(src: &'a [u8], raw_len: usize, short_distance: bool) -> Result<Self> {
        let mut p = 0;
        let count = u32_at(src, &mut p)? as usize;
        let cs = u32_at(src, &mut p)? as usize;
        let ds = u32_at(src, &mut p)? as usize;
        let ls = u32_at(src, &mut p)? as usize;
        let de = p.checked_add(cs).ok_or(Error::SizeOverflow)?;
        let le = de.checked_add(ds).ok_or(Error::SizeOverflow)?;
        let end = le.checked_add(ls).ok_or(Error::SizeOverflow)?;
        if end != src.len() {
            return Err(Error::InvalidFrame("split stream lengths"));
        }
        if count == 0 || count > raw_len || count > cs / 2 {
            return Err(Error::InvalidFrame("command count"));
        }
        Ok(Self {
            control: &src[p..de],
            attrs: &src[de..le],
            literals: &src[le..end],
            ci: 0,
            ai: 0,
            li: 0,
            pos: 0,
            raw_len,
            remaining: count,
            recent: 0,
            short_distance,
        })
    }
    #[cfg_attr(not(feature = "parser-no-inline"), inline(always))]
    pub fn next(&mut self) -> Result<Option<Command>> {
        if self.remaining == 0 {
            if self.ci != self.control.len()
                || self.ai != self.attrs.len()
                || self.li != self.literals.len()
                || self.pos != self.raw_len
            {
                return Err(Error::InvalidFrame("stream not consumed exactly"));
            }
            return Ok(None);
        }
        let end = self.ci.checked_add(2).ok_or(Error::SizeOverflow)?;
        let raw = self.control.get(self.ci..end).ok_or(Error::Truncated)?;
        let token = u16::from_le_bytes(raw.try_into().unwrap());
        self.ci = end;
        let op = (token & 7) as u8;
        if op == 7 && !self.short_distance {
            return Err(Error::InvalidFrame("reserved opcode"));
        }
        let lc = ((token >> 3) & 31) as usize;
        let rc = (token >> 8) as usize;
        if self.short_distance
            && op == 7
            && lc < 31
            && rc < 255
            && !cfg!(feature = "parser-op7-generic")
        {
            return self.next_short_distance(lc, rc).map(Some);
        }
        let lit = if lc == 31 {
            u32_at(self.control, &mut self.ci)? as usize
        } else {
            lc
        };
        let run = if rc == 255 {
            u32_at(self.control, &mut self.ci)? as usize
        } else {
            rc
        };
        if (lc == 31 && lit < 31) || (rc == 255 && run < 255) {
            return Err(Error::InvalidFrame("non-canonical length"));
        }
        if (op == 0 && (run != 0 || lit == 0)) || (op != 0 && run == 0) {
            return Err(Error::InvalidFrame("zero progress or invalid literal op"));
        }
        if matches!(op, 1 | 2 | 7) && run < 8 {
            return Err(Error::InvalidFrame("match below minmatch"));
        }
        let literal_start = self.li;
        let new_li = self.li.checked_add(lit).ok_or(Error::SizeOverflow)?;
        if new_li > self.literals.len() {
            return Err(Error::Truncated);
        }
        let match_pos = self.pos.checked_add(lit).ok_or(Error::SizeOverflow)?;
        let next_pos = match_pos.checked_add(run).ok_or(Error::SizeOverflow)?;
        if next_pos > self.raw_len {
            return Err(Error::InvalidFrame("command exceeds output"));
        }
        let mut distance = 0;
        let mut seed = [0u8; 8];
        match op {
            1 => {
                distance = u32_at(self.attrs, &mut self.ai)? as usize;
                self.recent = distance;
            }
            2 => {
                distance = self.recent;
            }
            7 => {
                distance = u16_at(self.attrs, &mut self.ai)? as usize;
                self.recent = distance;
            }
            3..=6 => {
                seed[..4].copy_from_slice(&u32_at(self.attrs, &mut self.ai)?.to_le_bytes());
                if op == 6 {
                    seed[4..].copy_from_slice(&u32_at(self.attrs, &mut self.ai)?.to_le_bytes());
                }
                let period = 1usize << (op - 3);
                if period < 4 && seed[period..4].iter().any(|&b| b != 0) {
                    return Err(Error::InvalidFrame("non-canonical seed padding"));
                }
            }
            _ => {}
        }
        if matches!(op, 1 | 2 | 7) && (distance == 0 || distance > match_pos) {
            return Err(Error::InvalidFrame(
                "backreference before initialized prefix",
            ));
        }
        self.remaining -= 1;
        self.pos = next_pos;
        self.li = new_li;
        Ok(Some(Command {
            op,
            lit,
            run,
            distance,
            seed,
            literal_start,
        }))
    }

    /// Common v2 JSON path: a fixed-size short-distance match. This retains
    /// every semantic/bounds check from `next` but avoids the extended-length,
    /// seed, and multi-opcode work on the overwhelmingly frequent opcode 7.
    #[inline(always)]
    fn next_short_distance(&mut self, lit: usize, run: usize) -> Result<Command> {
        if run < 8 {
            return Err(Error::InvalidFrame("match below minmatch"));
        }
        let literal_start = self.li;
        let new_li = self.li.checked_add(lit).ok_or(Error::SizeOverflow)?;
        if new_li > self.literals.len() {
            return Err(Error::Truncated);
        }
        let match_pos = self.pos.checked_add(lit).ok_or(Error::SizeOverflow)?;
        let next_pos = match_pos.checked_add(run).ok_or(Error::SizeOverflow)?;
        if next_pos > self.raw_len {
            return Err(Error::InvalidFrame("command exceeds output"));
        }
        let distance = u16_at(self.attrs, &mut self.ai)? as usize;
        if distance == 0 || distance > match_pos {
            return Err(Error::InvalidFrame(
                "backreference before initialized prefix",
            ));
        }
        self.recent = distance;
        self.remaining -= 1;
        self.pos = next_pos;
        self.li = new_li;
        Ok(Command {
            op: 7,
            lit,
            run,
            distance,
            seed: [0; 8],
            literal_start,
        })
    }

    /// Decode the overwhelmingly common fixed-size v2 command in place.
    ///
    /// This fuses parser state updates, literal copy, and match execution so
    /// the decoder does not materialize a `Command` only to immediately
    /// destructure it.  It deliberately handles only canonical opcode-7
    /// tokens; extensions and every other opcode stay on `next`'s fully
    /// generic path.  Returning `false` never consumes input.
    #[inline(always)]
    fn decode_short_distance_if_present(&mut self, out: &mut [u8], engine: Engine) -> Result<bool> {
        if cfg!(feature = "decode-op7-direct-generic") || self.remaining == 0 {
            return Ok(false);
        }
        let end = self.ci.checked_add(2).ok_or(Error::SizeOverflow)?;
        let raw = match self.control.get(self.ci..end) {
            Some(raw) => raw,
            None => return Err(Error::Truncated),
        };
        let token = u16::from_le_bytes(raw.try_into().unwrap());
        let op = (token & 7) as u8;
        let lit = ((token >> 3) & 31) as usize;
        let run = (token >> 8) as usize;
        if !self.short_distance || op != 7 || lit == 31 || run == 255 {
            return Ok(false);
        }
        self.ci = end;
        if run < 8 {
            return Err(Error::InvalidFrame("match below minmatch"));
        }
        let literal_start = self.li;
        let new_li = self.li.checked_add(lit).ok_or(Error::SizeOverflow)?;
        if new_li > self.literals.len() {
            return Err(Error::Truncated);
        }
        let match_pos = self.pos.checked_add(lit).ok_or(Error::SizeOverflow)?;
        let next_pos = match_pos.checked_add(run).ok_or(Error::SizeOverflow)?;
        if next_pos > self.raw_len {
            return Err(Error::InvalidFrame("command exceeds output"));
        }
        let distance = u16_at(self.attrs, &mut self.ai)? as usize;
        if distance == 0 || distance > match_pos {
            return Err(Error::InvalidFrame(
                "backreference before initialized prefix",
            ));
        }
        if lit != 0 {
            out[self.pos..match_pos].copy_from_slice(&self.literals[literal_start..new_li]);
        }
        // The checked arithmetic above proves `match_pos + run <= out.len()`;
        // the positive-distance check proves the source starts in the initialized
        // prefix.  When the run is no longer than the distance, source and
        // destination are disjoint. Keep overlap cases on the periodic engine.
        if run <= distance && !cfg!(feature = "decode-op7-inline-copy-generic") {
            // SAFETY: the validated range checks above keep both ranges inside
            // `out`, and `run <= distance` makes them non-overlapping.
            unsafe {
                let src = out.as_ptr().add(match_pos - distance);
                let dst = out.as_mut_ptr().add(match_pos);
                if run <= 63 && !cfg!(feature = "decode-op7-inline-short-copy-generic") {
                    copy_nonoverlapping_short(src, dst, run);
                } else {
                    std::ptr::copy_nonoverlapping(src, dst, run);
                }
            }
        } else {
            engine.repeat(out, match_pos, run, distance);
        }
        self.recent = distance;
        self.remaining -= 1;
        self.pos = next_pos;
        self.li = new_li;
        Ok(true)
    }

    /// Execute a fixed-size v2 opcode-7 command after `validate` has accepted
    /// this exact immutable payload. Returning `false` does not consume input.
    ///
    /// # Safety
    /// `validate` must have completed successfully for this parser's payload
    /// and raw length. That proof guarantees each stream access, literal range,
    /// output range, and backreference below. The caller keeps the payload
    /// immutably borrowed through `ValidatedFrame`.
    #[inline(always)]
    unsafe fn decode_short_distance_validated_if_present(
        &mut self,
        out: &mut [u8],
        engine: Engine,
    ) -> bool {
        if self.remaining == 0 {
            return false;
        }
        let token = u16::from_le(std::ptr::read_unaligned(
            self.control.as_ptr().add(self.ci).cast::<u16>(),
        ));
        let op = (token & 7) as u8;
        let lit = ((token >> 3) & 31) as usize;
        let run = (token >> 8) as usize;
        if !self.short_distance || op != 7 || lit == 31 || run == 255 {
            return false;
        }
        let literal_start = self.li;
        let match_pos = self.pos + lit;
        let distance = u16::from_le(std::ptr::read_unaligned(
            self.attrs.as_ptr().add(self.ai).cast::<u16>(),
        )) as usize;
        if lit != 0 {
            std::ptr::copy_nonoverlapping(
                self.literals.as_ptr().add(literal_start),
                out.as_mut_ptr().add(self.pos),
                lit,
            );
        }
        if run <= distance {
            let src = out.as_ptr().add(match_pos - distance);
            let dst = out.as_mut_ptr().add(match_pos);
            if run <= 63 && !cfg!(feature = "decode-op7-inline-short-copy-generic") {
                // SAFETY: the immutable validation proof establishes the same
                // initialized, bounded, non-overlapping ranges as the checked
                // path above; this is only its bounded code-generation variant.
                copy_nonoverlapping_short(src, dst, run);
            } else {
                std::ptr::copy_nonoverlapping(src, dst, run);
            }
        } else {
            engine.repeat(out, match_pos, run, distance);
        }
        self.ci += 2;
        self.ai += 2;
        self.li += lit;
        self.pos = match_pos + run;
        self.recent = distance;
        self.remaining -= 1;
        true
    }
}
pub(crate) fn validate(src: &[u8], kind: u32, raw_len: usize, short_distance: bool) -> Result<()> {
    if kind == RAW {
        if src.len() != raw_len {
            return Err(Error::InvalidFrame("RAW size"));
        }
        return Ok(());
    }
    if kind == COLUMN {
        // The structured decoder checks every bound, count and ordering rule,
        // so a trial decode into a scratch buffer is an exact validation.
        let mut tmp = vec![0u8; raw_len];
        return crate::columns::decode_block_into(src, &mut tmp);
    }
    if kind != LZ {
        return Err(Error::InvalidFrame("block kind"));
    }
    let mut parser = Parser::new(src, raw_len, short_distance)?;
    while parser.next()?.is_some() {}
    Ok(())
}
pub(crate) fn decode(
    src: &[u8],
    kind: u32,
    out: &mut [u8],
    engine: Engine,
    short_distance: bool,
) -> Result<()> {
    if kind == RAW {
        if src.len() != out.len() {
            return Err(Error::InvalidFrame("RAW size"));
        }
        engine.copy(out, src);
        return Ok(());
    }
    if kind == COLUMN {
        crate::columns::decode_block_into(src, out)?;
        return Ok(());
    }
    if kind != LZ {
        return Err(Error::InvalidFrame("block kind"));
    }
    let mut parser = Parser::new(src, out.len(), short_distance)?;
    let mut pos = 0;
    while parser.remaining != 0 {
        if parser.decode_short_distance_if_present(out, engine)? {
            pos = parser.pos;
            continue;
        }
        let c = parser.next()?.expect("nonzero parser command count");
        if c.lit != 0 || cfg!(feature = "decode-unconditional-literal-copy") {
            engine.copy(
                &mut out[pos..pos + c.lit],
                &parser.literals[c.literal_start..c.literal_start + c.lit],
            );
        }
        pos += c.lit;
        match c.op {
            1 | 2 | 7 => engine.repeat(out, pos, c.run, c.distance),
            3..=6 => engine.pattern(&mut out[pos..pos + c.run], &c.seed[..1usize << (c.op - 3)]),
            _ => {}
        }
        pos += c.run;
    }
    // Retain the exact stream-consumption proof after the fused commands.
    debug_assert_eq!(pos, parser.pos);
    let _ = parser.next()?;
    Ok(())
}

/// Decode a payload that has first passed `validate` without repeating
/// per-command checks on its fixed-size v2 opcode-7 fast path.
///
/// # Safety
/// The exact `src`, `kind`, output length, and `short_distance` mode must have
/// passed `validate` successfully and the payload must remain immutable since
/// that check. `out` must have the validated raw length. Violating this
/// contract can make the opcode-7 path perform unchecked reads or writes.
pub(crate) unsafe fn decode_validated(
    src: &[u8],
    kind: u32,
    out: &mut [u8],
    engine: Engine,
    short_distance: bool,
) -> Result<()> {
    if kind == COLUMN {
        // The column decoder is always fully checked; no unchecked fast path.
        return crate::columns::decode_block_into(src, out);
    }
    if kind == RAW {
        engine.copy(out, src);
        return Ok(());
    }
    let mut parser = Parser::new(src, out.len(), short_distance)?;
    let mut pos = 0;
    while parser.remaining != 0 {
        // SAFETY: `ValidatedFrame` invokes this only after `validate` has
        // proven the exact same payload and raw length. Generic commands
        // retain the checked parser for their uncommon shapes.
        if unsafe { parser.decode_short_distance_validated_if_present(out, engine) } {
            pos = parser.pos;
            continue;
        }
        let c = parser.next()?.expect("nonzero parser command count");
        if c.lit != 0 {
            engine.copy(
                &mut out[pos..pos + c.lit],
                &parser.literals[c.literal_start..c.literal_start + c.lit],
            );
        }
        pos += c.lit;
        match c.op {
            1 | 2 | 7 => engine.repeat(out, pos, c.run, c.distance),
            3..=6 => engine.pattern(&mut out[pos..pos + c.run], &c.seed[..1usize << (c.op - 3)]),
            _ => {}
        }
        pos += c.run;
    }
    debug_assert_eq!(pos, out.len());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::copy_nonoverlapping_short;

    #[test]
    fn bounded_nonoverlap_copy_preserves_every_length_and_alignment() {
        for len in 8..=63 {
            for source_alignment in 0..8 {
                for destination_alignment in 0..8 {
                    let source = 16 + source_alignment;
                    let destination = 128 + destination_alignment;
                    let mut bytes = (0..256).map(|index| index as u8 ^ 0xa5).collect::<Vec<_>>();
                    let expected = bytes[source..source + len].to_vec();
                    // SAFETY: the test ranges are fully in `bytes` and disjoint.
                    unsafe {
                        copy_nonoverlapping_short(
                            bytes.as_ptr().add(source),
                            bytes.as_mut_ptr().add(destination),
                            len,
                        );
                    }
                    assert_eq!(&bytes[destination..destination + len], &expected);
                }
            }
        }
    }
}
