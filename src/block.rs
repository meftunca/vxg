//! The same wire commands are supported by CPU, Metal, Vulkan/SPIR-V and CUDA.
use crate::{simd::Engine, Error, Result};

pub const RAW: u32 = 0;
pub const LZ: u32 = 1;
pub const HASH_BITS: usize = 14;
pub const HASH_SIZE: usize = 1 << HASH_BITS;
pub const BLOCK_PREFIX: usize = 16;

#[derive(Default, Debug, Clone, Copy)]
pub struct WorkStats {
    pub probes: u64,
    pub accepted_matches: u64,
    pub matched_bytes: u64,
    pub commands: u64,
    pub literal_bytes: u64,
}
impl std::ops::AddAssign for WorkStats {
    fn add_assign(&mut self, r: Self) {
        self.probes += r.probes; self.accepted_matches += r.accepted_matches;
        self.matched_bytes += r.matched_bytes; self.commands += r.commands;
        self.literal_bytes += r.literal_bytes;
    }
}
pub(crate) struct Scratch {
    table: Vec<u32>, control: Vec<u8>, attrs: Vec<u8>, literals: Vec<u8>,
}
impl Scratch {
    pub fn new() -> Self { Self { table: vec![0; HASH_SIZE], control: Vec::new(), attrs: Vec::new(), literals: Vec::new() } }
}
#[derive(Debug)]
pub(crate) struct Encoded { pub kind: u32, pub bytes: Vec<u8>, pub stats: WorkStats }
fn hash(data: &[u8], pos: usize) -> usize {
    let x = u64::from_le_bytes(data[pos..pos+8].try_into().unwrap());
    (x.wrapping_mul(0x9e3779b185ebca87) >> (64-HASH_BITS)) as usize
}
pub(crate) fn inline_token(op: u8, lit: u8, run: u8) -> u16 {
    (op as u16) | ((lit as u16) << 3) | ((run as u16) << 8)
}
fn control(out: &mut Vec<u8>, op: u8, lit: usize, run: usize) {
    let lc = lit.min(31) as u16;
    let rc = run.min(255) as u16;
    out.extend_from_slice(&inline_token(op, lc as u8, rc as u8).to_le_bytes());
    if lit >= 31 { out.extend_from_slice(&(lit as u32).to_le_bytes()); }
    if run >= 255 { out.extend_from_slice(&(run as u32).to_le_bytes()); }
}
pub(crate) fn encode(data: &[u8], scratch: &mut Scratch, engine: Engine, acceleration: usize) -> Encoded {
    let n = data.len();
    let s = scratch;
    s.table.fill(0); s.control.clear(); s.attrs.clear(); s.literals.clear();
    let mut stats = WorkStats::default();
    let mut pos = 0usize;
    let mut anchor = 0usize;
    let mut recent = 0usize;
    let mut misses = 0usize;
    let mut count = 0usize;
    while pos+8 <= n {
        let h = hash(data, pos);
        let old = s.table[h];
        s.table[h] = (pos+1) as u32;
        stats.probes += 1;
        let candidate = old.wrapping_sub(1) as usize;
        if old == 0 || candidate >= pos || data[candidate..candidate+8] != data[pos..pos+8] {
            misses = misses.saturating_add(1);
            let step = acceleration.saturating_add(misses >> 6).min(64);
            pos = pos.saturating_add(step);
            continue;
        }
        let run = 8 + engine.lcp(&data[pos+8..], &data[candidate+8..]);
        let distance = pos - candidate;
        let lit = pos-anchor;
        let op = match distance { 1 => 3, 2 => 4, 4 => 5, 8 => 6,
            d if d == recent => 2, _ => 1 };
        control(&mut s.control, op, lit, run);
        match op {
            1 => { s.attrs.extend_from_slice(&(distance as u32).to_le_bytes()); recent = distance; }
            3..=6 => {
                let period = 1usize << (op-3);
                let mut seed = [0u8;8]; seed[..period].copy_from_slice(&data[pos..pos+period]);
                s.attrs.extend_from_slice(&seed[..if period == 8 { 8 } else { 4 }]);
            }
            _ => {}
        }
        s.literals.extend_from_slice(&data[anchor..pos]);
        stats.accepted_matches += 1; stats.matched_bytes += run as u64;
        count += 1;
        pos += run; anchor = pos; misses = 0;
        // Once committed bytes reach n, no later instruction can make this block smaller.
        if BLOCK_PREFIX + s.control.len() + s.attrs.len() + s.literals.len() >= n {
            stats.commands = count as u64; stats.literal_bytes = s.literals.len() as u64;
            return Encoded { kind: RAW, bytes: data.to_vec(), stats };
        }
    }
    if anchor < n { control(&mut s.control, 0, n-anchor, 0); s.literals.extend_from_slice(&data[anchor..]); count += 1; }
    stats.commands = count as u64; stats.literal_bytes = s.literals.len() as u64;
    let size = BLOCK_PREFIX + s.control.len() + s.attrs.len() + s.literals.len();
    if size >= n { return Encoded { kind: RAW, bytes: data.to_vec(), stats }; }
    let mut bytes = Vec::with_capacity(size);
    for v in [count, s.control.len(), s.attrs.len(), s.literals.len()] { bytes.extend_from_slice(&(v as u32).to_le_bytes()); }
    bytes.extend_from_slice(&s.control); bytes.extend_from_slice(&s.attrs); bytes.extend_from_slice(&s.literals);
    Encoded { kind: LZ, bytes, stats }
}

fn u32_at(src: &[u8], pos: &mut usize) -> Result<u32> {
    let end = pos.checked_add(4).ok_or(Error::SizeOverflow)?;
    let bytes = src.get(*pos..end).ok_or(Error::Truncated)?;
    *pos = end;
    Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
}
#[derive(Debug, Clone, Copy)]
pub(crate) struct Command {
    pub op: u8, pub lit: usize, pub run: usize, pub distance: usize,
    pub seed: [u8;8], pub literal_start: usize,
}
pub(crate) struct Parser<'a> {
    control: &'a [u8], attrs: &'a [u8], pub literals: &'a [u8],
    ci: usize, ai: usize, li: usize, pos: usize, raw_len: usize,
    remaining: usize, recent: usize,
}
impl<'a> Parser<'a> {
    pub fn new(src: &'a [u8], raw_len: usize) -> Result<Self> {
        let mut p = 0;
        let count = u32_at(src, &mut p)? as usize;
        let cs = u32_at(src, &mut p)? as usize;
        let ds = u32_at(src, &mut p)? as usize;
        let ls = u32_at(src, &mut p)? as usize;
        let de = p.checked_add(cs).ok_or(Error::SizeOverflow)?;
        let le = de.checked_add(ds).ok_or(Error::SizeOverflow)?;
        let end = le.checked_add(ls).ok_or(Error::SizeOverflow)?;
        if end != src.len() { return Err(Error::InvalidFrame("split stream lengths")); }
        if count == 0 || count > raw_len || count > cs/2 { return Err(Error::InvalidFrame("command count")); }
        Ok(Self { control: &src[p..de], attrs: &src[de..le], literals: &src[le..end],
            ci: 0, ai: 0, li: 0, pos: 0, raw_len, remaining: count, recent: 0 })
    }
    pub fn next(&mut self) -> Result<Option<Command>> {
        if self.remaining == 0 {
            if self.ci != self.control.len() || self.ai != self.attrs.len() || self.li != self.literals.len() || self.pos != self.raw_len {
                return Err(Error::InvalidFrame("stream not consumed exactly"));
            }
            return Ok(None);
        }
        let end = self.ci.checked_add(2).ok_or(Error::SizeOverflow)?;
        let raw = self.control.get(self.ci..end).ok_or(Error::Truncated)?;
        let token = u16::from_le_bytes(raw.try_into().unwrap()); self.ci = end;
        let op = (token & 7) as u8;
        if op == 7 { return Err(Error::InvalidFrame("reserved opcode")); }
        let lc = ((token >> 3) & 31) as usize;
        let rc = (token >> 8) as usize;
        let lit = if lc == 31 { u32_at(self.control, &mut self.ci)? as usize } else { lc };
        let run = if rc == 255 { u32_at(self.control, &mut self.ci)? as usize } else { rc };
        if (lc == 31 && lit < 31) || (rc == 255 && run < 255) { return Err(Error::InvalidFrame("non-canonical length")); }
        if (op == 0 && (run != 0 || lit == 0)) || (op != 0 && run == 0) { return Err(Error::InvalidFrame("zero progress or invalid literal op")); }
        if matches!(op, 1 | 2) && run < 8 { return Err(Error::InvalidFrame("match below minmatch")); }
        let literal_start = self.li;
        let new_li = self.li.checked_add(lit).ok_or(Error::SizeOverflow)?;
        if new_li > self.literals.len() { return Err(Error::Truncated); }
        let match_pos = self.pos.checked_add(lit).ok_or(Error::SizeOverflow)?;
        let next_pos = match_pos.checked_add(run).ok_or(Error::SizeOverflow)?;
        if next_pos > self.raw_len { return Err(Error::InvalidFrame("command exceeds output")); }
        let mut distance = 0;
        let mut seed = [0u8;8];
        match op {
            1 => { distance = u32_at(self.attrs, &mut self.ai)? as usize; self.recent = distance; }
            2 => { distance = self.recent; }
            3..=6 => {
                seed[..4].copy_from_slice(&u32_at(self.attrs, &mut self.ai)?.to_le_bytes());
                if op == 6 { seed[4..].copy_from_slice(&u32_at(self.attrs, &mut self.ai)?.to_le_bytes()); }
                let period = 1usize << (op-3);
                if period < 4 && seed[period..4].iter().any(|&b| b != 0) { return Err(Error::InvalidFrame("non-canonical seed padding")); }
            }
            _ => {}
        }
        if matches!(op, 1 | 2) && (distance == 0 || distance > match_pos) { return Err(Error::InvalidFrame("backreference before initialized prefix")); }
        self.remaining -= 1; self.pos = next_pos; self.li = new_li;
        Ok(Some(Command { op, lit, run, distance, seed, literal_start }))
    }
}
pub(crate) fn validate(src: &[u8], kind: u32, raw_len: usize) -> Result<()> {
    if kind == RAW {
        if src.len() != raw_len { return Err(Error::InvalidFrame("RAW size")); }
        return Ok(());
    }
    if kind != LZ { return Err(Error::InvalidFrame("block kind")); }
    let mut parser = Parser::new(src, raw_len)?;
    while parser.next()?.is_some() {}
    Ok(())
}
pub(crate) fn decode(src: &[u8], kind: u32, out: &mut [u8], engine: Engine) -> Result<()> {
    if kind == RAW {
        if src.len() != out.len() { return Err(Error::InvalidFrame("RAW size")); }
        engine.copy(out, src); return Ok(());
    }
    if kind != LZ { return Err(Error::InvalidFrame("block kind")); }
    let mut parser = Parser::new(src, out.len())?;
    let mut pos = 0;
    while let Some(c) = parser.next()? {
        engine.copy(&mut out[pos..pos+c.lit], &parser.literals[c.literal_start..c.literal_start+c.lit]);
        pos += c.lit;
        match c.op {
            1 | 2 => engine.repeat(out, pos, c.run, c.distance),
            3..=6 => engine.pattern(&mut out[pos..pos+c.run], &c.seed[..1usize << (c.op-3)]),
            _ => {}
        }
        pos += c.run;
    }
    Ok(())
}
