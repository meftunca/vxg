//! The same wire commands are supported by CPU, Metal, Vulkan/SPIR-V and CUDA.
use crate::{simd::Engine, Error, Result};

pub const RAW: u32 = 0;
pub const LZ: u32 = 1;
#[cfg(all(feature = "hash13", feature = "hash15"))]
compile_error!("hash13 and hash15 benchmark features are mutually exclusive");
#[cfg(feature = "hash13")]
pub const HASH_BITS: usize = 13;
#[cfg(all(not(feature = "hash13"), feature = "hash15"))]
pub const HASH_BITS: usize = 15;
#[cfg(not(any(feature = "hash13", feature = "hash15")))]
pub const HASH_BITS: usize = 14;
pub const HASH_SIZE: usize = 1 << HASH_BITS;
pub const BLOCK_PREFIX: usize = 16;

#[derive(Default, Debug, Clone, Copy)]
pub struct WorkStats {
    /// Blocks emitted RAW after bounded incompressibility sampling, before the
    /// full hash matcher allocates or scans them.
    pub raw_fast_path_blocks: u64,
    pub raw_sample_words: u64,
    pub probes: u64,
    pub occupied_probes: u64,
    pub candidate_checks: u64,
    pub hash_collisions: u64,
    pub accepted_matches: u64,
    pub matched_bytes: u64,
    pub commands: u64,
    pub literal_bytes: u64,
    pub control_bytes: u64,
    pub attribute_bytes: u64,
    /// Canonical command counts for opcodes 0 through 7.
    pub op_counts: [u64; 8],
    /// Benchmark-only shape telemetry. These fields and their increments are
    /// absent from normal library builds.
    #[cfg(feature = "bench")]
    pub op7_distance_buckets: [u64; 6],
    #[cfg(feature = "bench")]
    pub match_run_buckets: [u64; 6],
    #[cfg(feature = "bench")]
    pub zero_literal_commands: u64,
    /// Number of logical bytes inspected by the LCP matcher (including a
    /// terminating mismatch byte when one exists). This is a work counter,
    /// not a hardware instruction count.
    pub lcp_calls: u64,
    pub lcp_bytes: u64,
}
impl std::ops::AddAssign for WorkStats {
    fn add_assign(&mut self, r: Self) {
        self.raw_fast_path_blocks += r.raw_fast_path_blocks;
        self.raw_sample_words += r.raw_sample_words;
        self.probes += r.probes;
        self.occupied_probes += r.occupied_probes;
        self.candidate_checks += r.candidate_checks;
        self.hash_collisions += r.hash_collisions;
        self.accepted_matches += r.accepted_matches;
        self.matched_bytes += r.matched_bytes;
        self.commands += r.commands;
        self.literal_bytes += r.literal_bytes;
        self.control_bytes += r.control_bytes;
        self.attribute_bytes += r.attribute_bytes;
        for (left, right) in self.op_counts.iter_mut().zip(r.op_counts) {
            *left += right;
        }
        #[cfg(feature = "bench")]
        {
            for (left, right) in self.op7_distance_buckets.iter_mut().zip(r.op7_distance_buckets) { *left += right; }
            for (left, right) in self.match_run_buckets.iter_mut().zip(r.match_run_buckets) { *left += right; }
            self.zero_literal_commands += r.zero_literal_commands;
        }
        self.lcp_calls += r.lcp_calls;
        self.lcp_bytes += r.lcp_bytes;
    }
}
pub(crate) struct Scratch {
    table: Vec<u32>,
    control: Vec<u8>,
    attrs: Vec<u8>,
    literals: Vec<u8>,
    payloads: Vec<u8>,
}
impl Scratch {
    pub fn new() -> Self {
        Self {
            table: vec![0; HASH_SIZE],
            control: Vec::new(),
            attrs: Vec::new(),
            literals: Vec::new(),
            payloads: Vec::new(),
        }
    }
    pub fn reserve_for(&mut self, n: usize) {
        // These bounded hints cover the common command/literal mix without
        // eagerly reserving the full block for every temporary stream.
        let control = n / 8;
        let attrs = n / 8;
        let literals = n / 16;
        if self.control.capacity() < control {
            self.control.reserve(control - self.control.capacity());
        }
        if self.attrs.capacity() < attrs {
            self.attrs.reserve(attrs - self.attrs.capacity());
        }
        if self.literals.capacity() < literals {
            self.literals.reserve(literals - self.literals.capacity());
        }
    }
    pub fn reserve_payloads_for(&mut self, n: usize) {
        if self.payloads.capacity() < n {
            self.payloads.reserve(n - self.payloads.capacity());
        }
    }
    pub fn finish(self) -> Vec<u8> {
        self.payloads
    }
}
#[derive(Debug)]
pub(crate) struct Encoded {
    pub kind: u32,
    pub offset: usize,
    pub len: usize,
    pub stats: WorkStats,
}
fn append_payload(payloads: &mut Vec<u8>, data: &[u8]) -> (usize, usize) {
    let offset = payloads.len();
    payloads.extend_from_slice(data);
    (offset, data.len())
}
#[inline(always)]
fn load_u64_le(data: &[u8], pos: usize) -> u64 {
    u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap())
}
#[inline(always)]
fn hash_word(x: u64) -> usize {
    (x.wrapping_mul(0x9e3779b185ebca87) >> (64 - HASH_BITS)) as usize
}
const RAW_SAMPLE_MIN: usize = 4096;
const RAW_SAMPLE_WORDS: usize = 48;
const RAW_SAMPLE_SLOTS: usize = 256;
/// A bounded, allocation-free negative filter for blocks that have no sampled
/// repeated eight-byte word. A false negative merely uses the regular matcher;
/// a false positive can only select the already-valid RAW representation.
#[inline]
fn sampled_incompressible(data: &[u8]) -> Option<usize> {
    if cfg!(feature = "raw-fastpath-off") {
        return None;
    }
    if data.len() < RAW_SAMPLE_MIN || data.len() < 8 {
        return None;
    }
    let mut words = [0u64; RAW_SAMPLE_SLOTS];
    let mut occupied = [false; RAW_SAMPLE_SLOTS];
    let last = data.len() - 8;
    let stride = (last / (RAW_SAMPLE_WORDS - 1)).max(1);
    for i in 0..RAW_SAMPLE_WORDS {
        let word = load_u64_le(data, (i * stride).min(last));
        let slot = hash_word(word) & (RAW_SAMPLE_SLOTS - 1);
        if occupied[slot] && words[slot] == word {
            return None;
        }
        words[slot] = word;
        occupied[slot] = true;
    }
    Some(RAW_SAMPLE_WORDS)
}
pub(crate) fn inline_token(op: u8, lit: u8, run: u8) -> u16 {
    (op as u16) | ((lit as u16) << 3) | ((run as u16) << 8)
}
fn control(out: &mut Vec<u8>, op: u8, lit: usize, run: usize) {
    let lc = lit.min(31) as u16;
    let rc = run.min(255) as u16;
    out.extend_from_slice(&inline_token(op, lc as u8, rc as u8).to_le_bytes());
    if lit >= 31 {
        out.extend_from_slice(&(lit as u32).to_le_bytes());
    }
    if run >= 255 {
        out.extend_from_slice(&(run as u32).to_le_bytes());
    }
}
fn finish_stats<const STATS: bool>(stats: &mut WorkStats, count: usize, scratch: &Scratch) {
    if STATS {
        stats.commands = count as u64;
        stats.literal_bytes = scratch.literals.len() as u64;
        stats.control_bytes = scratch.control.len() as u64;
        stats.attribute_bytes = scratch.attrs.len() as u64;
    }
}
pub(crate) fn encode<const STATS: bool>(
    data: &[u8],
    scratch: &mut Scratch,
    engine: Engine,
    acceleration: usize,
) -> Encoded {
    let n = data.len();
    let s = scratch;
    if let Some(samples) = sampled_incompressible(data) {
        let mut stats = WorkStats::default();
        if STATS {
            stats.raw_fast_path_blocks = 1;
            stats.raw_sample_words = samples as u64;
        }
        let (offset, len) = append_payload(&mut s.payloads, data);
        return Encoded {
            kind: RAW,
            offset,
            len,
            stats,
        };
    }
    s.reserve_for(n);
    s.table.fill(0);
    s.control.clear();
    s.attrs.clear();
    s.literals.clear();
    let mut stats = WorkStats::default();
    let mut pos = 0usize;
    let mut anchor = 0usize;
    let mut recent = 0usize;
    let mut misses = 0usize;
    let mut count = 0usize;
    while pos + 8 <= n {
        let word = load_u64_le(data, pos);
        let h = hash_word(word);
        let old = s.table[h];
        s.table[h] = (pos + 1) as u32;
        if STATS {
            stats.probes += 1;
            if old != 0 {
                stats.occupied_probes += 1;
            }
        }
        let candidate = old.wrapping_sub(1) as usize;
        if old == 0 || candidate >= pos {
            misses = misses.saturating_add(1);
            let step = acceleration.saturating_add(misses >> 6).min(64);
            pos = pos.saturating_add(step);
            continue;
        }
        if STATS {
            stats.candidate_checks += 1;
        }
        if load_u64_le(data, candidate) != word {
            if STATS {
                stats.hash_collisions += 1;
            }
            misses = misses.saturating_add(1);
            let step = acceleration.saturating_add(misses >> 6).min(64);
            pos = pos.saturating_add(step);
            continue;
        }
        let left = &data[pos + 8..];
        let right = &data[candidate + 8..];
        let common = engine.lcp(left, right);
        let compared = common.saturating_add(1).min(left.len().min(right.len()));
        if STATS {
            stats.lcp_calls += 1;
            stats.lcp_bytes += compared as u64;
        }
        let run = 8 + common;
        let distance = pos - candidate;
        let lit = pos - anchor;
        let op = match distance {
            1 => 3,
            2 => 4,
            4 => 5,
            8 => 6,
            d if d == recent => 2,
            d if d <= u16::MAX as usize => 7,
            _ => 1,
        };
        control(&mut s.control, op, lit, run);
        if STATS {
            stats.op_counts[op as usize] += 1;
            #[cfg(feature = "bench")]
            {
                if lit == 0 { stats.zero_literal_commands += 1; }
                let run_bucket = match run { 0..=15 => 0, 16..=31 => 1, 32..=63 => 2, 64..=127 => 3, 128..=255 => 4, _ => 5 };
                stats.match_run_buckets[run_bucket] += 1;
                if op == 7 {
                    let distance_bucket = match distance { 0..=64 => 0, 65..=256 => 1, 257..=1024 => 2, 1025..=4096 => 3, 4097..=16384 => 4, _ => 5 };
                    stats.op7_distance_buckets[distance_bucket] += 1;
                }
            }
        }
        match op {
            1 => {
                s.attrs.extend_from_slice(&(distance as u32).to_le_bytes());
                recent = distance;
            }
            7 => {
                s.attrs.extend_from_slice(&(distance as u16).to_le_bytes());
                recent = distance;
            }
            3..=6 => {
                let period = 1usize << (op - 3);
                let mut seed = [0u8; 8];
                seed[..period].copy_from_slice(&data[pos..pos + period]);
                s.attrs
                    .extend_from_slice(&seed[..if period == 8 { 8 } else { 4 }]);
            }
            _ => {}
        }
        s.literals.extend_from_slice(&data[anchor..pos]);
        if STATS {
            stats.accepted_matches += 1;
            stats.matched_bytes += run as u64;
        }
        count += 1;
        pos += run;
        anchor = pos;
        misses = 0;
        // Once committed bytes reach n, no later instruction can make this block smaller.
        if BLOCK_PREFIX + s.control.len() + s.attrs.len() + s.literals.len() >= n {
            finish_stats::<STATS>(&mut stats, count, s);
            let (offset, len) = append_payload(&mut s.payloads, data);
            return Encoded {
                kind: RAW,
                offset,
                len,
                stats,
            };
        }
    }
    if anchor < n {
        control(&mut s.control, 0, n - anchor, 0);
        if STATS {
            stats.op_counts[0] += 1;
        }
        s.literals.extend_from_slice(&data[anchor..]);
        count += 1;
    }
    finish_stats::<STATS>(&mut stats, count, s);
    let size = BLOCK_PREFIX + s.control.len() + s.attrs.len() + s.literals.len();
    if size >= n {
        let (offset, len) = append_payload(&mut s.payloads, data);
        return Encoded {
            kind: RAW,
            offset,
            len,
            stats,
        };
    }
    let offset = s.payloads.len();
    for v in [count, s.control.len(), s.attrs.len(), s.literals.len()] {
        s.payloads.extend_from_slice(&(v as u32).to_le_bytes());
    }
    s.payloads.extend_from_slice(&s.control);
    s.payloads.extend_from_slice(&s.attrs);
    s.payloads.extend_from_slice(&s.literals);
    let len = s.payloads.len() - offset;
    Encoded {
        kind: LZ,
        offset,
        len,
        stats,
    }
}

#[cfg_attr(not(feature="parser-no-inline"), inline(always))]
fn u32_at(src: &[u8], pos: &mut usize) -> Result<u32> {
    let end = pos.checked_add(4).ok_or(Error::SizeOverflow)?;
    let bytes = src.get(*pos..end).ok_or(Error::Truncated)?;
    *pos = end;
    Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
}
#[cfg_attr(not(feature="parser-no-inline"), inline(always))]
fn u16_at(src: &[u8], pos: &mut usize) -> Result<u16> {
    let end = pos.checked_add(2).ok_or(Error::SizeOverflow)?;
    let bytes = src.get(*pos..end).ok_or(Error::Truncated)?;
    *pos = end;
    Ok(u16::from_le_bytes(bytes.try_into().unwrap()))
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
    #[cfg_attr(not(feature="parser-no-inline"), inline(always))]
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
        if self.short_distance && op == 7 && lc < 31 && rc < 255 && !cfg!(feature="parser-op7-generic") {
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
        if run < 8 { return Err(Error::InvalidFrame("match below minmatch")); }
        let literal_start = self.li;
        let new_li = self.li.checked_add(lit).ok_or(Error::SizeOverflow)?;
        if new_li > self.literals.len() { return Err(Error::Truncated); }
        let match_pos = self.pos.checked_add(lit).ok_or(Error::SizeOverflow)?;
        let next_pos = match_pos.checked_add(run).ok_or(Error::SizeOverflow)?;
        if next_pos > self.raw_len { return Err(Error::InvalidFrame("command exceeds output")); }
        let distance = u16_at(self.attrs, &mut self.ai)? as usize;
        if distance == 0 || distance > match_pos { return Err(Error::InvalidFrame("backreference before initialized prefix")); }
        self.recent = distance;
        self.remaining -= 1;
        self.pos = next_pos;
        self.li = new_li;
        Ok(Command { op: 7, lit, run, distance, seed: [0; 8], literal_start })
    }

    /// Decode the overwhelmingly common fixed-size v2 command in place.
    ///
    /// This fuses parser state updates, literal copy, and match execution so
    /// the decoder does not materialize a `Command` only to immediately
    /// destructure it.  It deliberately handles only canonical opcode-7
    /// tokens; extensions and every other opcode stay on `next`'s fully
    /// generic path.  Returning `false` never consumes input.
    #[inline(always)]
    fn decode_short_distance_if_present(
        &mut self,
        out: &mut [u8],
        engine: Engine,
    ) -> Result<bool> {
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
            out[self.pos..match_pos]
                .copy_from_slice(&self.literals[literal_start..new_li]);
        }
        engine.repeat(out, match_pos, run, distance);
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
            std::ptr::copy_nonoverlapping(
                out.as_ptr().add(match_pos - distance),
                out.as_mut_ptr().add(match_pos),
                run,
            );
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
        if c.lit != 0 || cfg!(feature="decode-unconditional-literal-copy") {
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
pub(crate) fn decode_validated(
    src: &[u8],
    kind: u32,
    out: &mut [u8],
    engine: Engine,
    short_distance: bool,
) -> Result<()> {
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
