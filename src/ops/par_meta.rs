//! Parallel block metadata inspection and contiguous cross-block boundary assembly.

use std::borrow::Cow;
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::{frame, query, DecodeOptions, Frame, Result, BLOCK_COLUMN};

pub(crate) struct SyncCell<T>(UnsafeCell<T>);

// SAFETY: Each index in a `Vec<SyncCell<T>>` is written at most once by the unique worker
// thread that claims block `b` via atomic `fetch_add`, and read only after the parallel join.
unsafe impl<T: Send> Sync for SyncCell<T> {}

impl<T> SyncCell<T> {
    #[inline]
    pub(crate) const fn new(val: T) -> Self {
        Self(UnsafeCell::new(val))
    }

    #[inline]
    pub(crate) fn get(&self) -> *mut T {
        self.0.get()
    }

    #[inline]
    pub(crate) fn into_inner(self) -> T {
        self.0.into_inner()
    }
}

pub(crate) struct ParInBlock<'a> {
    pub(crate) f_bytes: &'a [u8],
    pub(crate) local_blk: usize,
    pub(crate) version: u16,
    pub(crate) frame_checksummed: bool,
    pub(crate) dict: &'a [u8],
    pub(crate) info: frame::BlockInfo,
    pub(crate) payload: &'a [u8],
    pub(crate) row_meta: Option<(usize, bool)>,
}

impl ParInBlock<'_> {
    #[inline]
    pub(crate) fn decode_into(&self, options: &DecodeOptions, buf: &mut Vec<u8>) -> Result<()> {
        buf.clear();
        buf.resize(self.info.raw_size, 0);
        crate::block::decode_with_dict(
            self.payload,
            self.info.kind,
            buf,
            crate::simd::Engine::new(options.simd)?,
            self.version >= 2,
            self.dict,
        )?;
        if self.frame_checksummed
            && options.verify_checksum
            && crate::checksum::crc32c(buf) != self.info.checksum
        {
            return Err(crate::Error::Checksum {
                block: self.local_blk,
            });
        }
        Ok(())
    }
}

pub(crate) struct ParMeta<'a> {
    pub(crate) decoded_raw: Option<Vec<u8>>,
    pub(crate) row_count: usize,
    pub(crate) had_in_carry: bool,
    pub(crate) cur_global: usize,
    pub(crate) first_frag: Option<Cow<'a, [u8]>>,
    pub(crate) last_unterm_frag: Option<Cow<'a, [u8]>>,
}

struct BoundaryEntry {
    block_index: usize,
    row_index: usize,
    global_row_index: usize,
    start: usize,
    end: usize,
}

pub(crate) struct BoundarySet {
    arena: Vec<u8>,
    entries: Vec<BoundaryEntry>,
}

#[derive(Clone, Copy)]
pub(crate) struct BoundaryRef<'a> {
    pub(crate) block_index: usize,
    pub(crate) row_index: usize,
    pub(crate) global_row_index: usize,
    pub(crate) bytes: &'a [u8],
}

impl BoundarySet {
    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    #[inline]
    pub(crate) fn iter(&self) -> impl Iterator<Item = BoundaryRef<'_>> {
        self.entries.iter().map(|e| BoundaryRef {
            block_index: e.block_index,
            row_index: e.row_index,
            global_row_index: e.global_row_index,
            bytes: &self.arena[e.start..e.end],
        })
    }
}

pub(crate) fn collect_par_blocks<'a>(
    frame_bytes: &'a [u8],
    frames: &'a [Frame<'a>],
) -> Vec<ParInBlock<'a>> {
    let total_blocks: usize = frames.iter().map(|f| f.blocks.len()).sum();
    let mut in_blocks = Vec::with_capacity(total_blocks);
    for (f_bytes, parsed) in query::frame_slices(frame_bytes, frames) {
        let csum = parsed.checksummed();
        let ver = parsed.version();
        let dict = parsed.dict();
        for (local_blk, info) in parsed.blocks().enumerate() {
            in_blocks.push(ParInBlock {
                f_bytes,
                local_blk,
                version: ver,
                frame_checksummed: csum,
                dict,
                info,
                payload: parsed.blocks[local_blk].payload,
                row_meta: parsed.block_row_meta(local_blk),
            });
        }
    }
    in_blocks
}

enum IndepBlock<'a> {
    Grammar(crate::columns::IndepGrammarBoundary<'a>),
    Decoded {
        raw_vec: Vec<u8>,
        rc: usize,
        ends_nl: bool,
    },
}

#[inline]
fn finish_decoded_meta<'a>(
    raw_vec: Vec<u8>,
    rc: usize,
    ends_nl: bool,
    had_in: bool,
    cg: usize,
) -> ParMeta<'a> {
    let first_frag = if had_in && !raw_vec.is_empty() {
        let end = raw_vec
            .iter()
            .position(|&c| c == b'\n')
            .map_or(raw_vec.len(), |i| i + 1);
        Some(Cow::Owned(raw_vec[..end].to_vec()))
    } else {
        None
    };
    let last_unterm_frag = if !raw_vec.is_empty() && !ends_nl {
        let st = raw_vec
            .iter()
            .rposition(|&c| c == b'\n')
            .map_or(0, |i| i + 1);
        Some(Cow::Owned(raw_vec[st..].to_vec()))
    } else {
        None
    };
    ParMeta {
        decoded_raw: Some(raw_vec),
        row_count: rc,
        had_in_carry: had_in,
        cur_global: cg,
        first_frag,
        last_unterm_frag,
    }
}

#[inline]
fn count_raw_rows(raw: &[u8]) -> (usize, bool) {
    if raw.is_empty() {
        return (0, false);
    }
    let ends_nl = raw[raw.len() - 1] == b'\n';
    let nl = raw.iter().filter(|&&c| c == b'\n').count();
    (nl + usize::from(!ends_nl), ends_nl)
}

pub(crate) fn inspect_par_meta<'a>(
    in_blocks: &[ParInBlock<'a>],
    options: &DecodeOptions,
) -> Result<(Vec<ParMeta<'a>>, Vec<usize>, bool)> {
    let (meta, non_empty, any_unterm, _) = inspect_par_meta_from(in_blocks, false, 0, options)?;
    Ok((meta, non_empty, any_unterm))
}

pub(crate) fn inspect_par_meta_from<'a>(
    in_blocks: &[ParInBlock<'a>],
    initial_incoming: bool,
    base_global: usize,
    options: &DecodeOptions,
) -> Result<(Vec<ParMeta<'a>>, Vec<usize>, bool, usize)> {
    let nb = in_blocks.len();
    let mut meta: Vec<ParMeta<'a>> = Vec::with_capacity(nb);
    let mut non_empty = Vec::with_capacity(nb);
    let mut incoming = initial_incoming;
    let mut any_unterm = initial_incoming;
    let mut cur_global = base_global;

    // Zero-barrier O(nb) header fast path when every non-empty block carries `FLAG_ROW_INDEX`
    // and terminates on `\n` (such as `--record-aligned` frames). Avoids touching any block
    // payload or spawning a pre-scan thread-pool pass across multi-GB archives.
    if !initial_incoming
        && nb > 0
        && in_blocks
            .iter()
            .all(|ib| ib.info.raw_size == 0 || ib.row_meta.is_some_and(|(_, ends_nl)| ends_nl))
    {
        for (b, ib) in in_blocks.iter().enumerate() {
            let rc = if ib.info.raw_size == 0 {
                0
            } else {
                ib.row_meta.map_or(0, |(r, _)| r)
            };
            let cg = cur_global;
            if rc > 0 {
                cur_global += rc;
                non_empty.push(b);
            }
            meta.push(ParMeta {
                decoded_raw: None,
                row_count: rc,
                had_in_carry: false,
                cur_global: cg,
                first_frag: None,
                last_unterm_frag: None,
            });
        }
        return Ok((meta, non_empty, false, cur_global));
    }

    if options.threads > 1 && nb >= 8 {
        let workers = options.threads.min(nb / 2).max(2);
        let slots: Vec<SyncCell<Option<Result<IndepBlock<'a>>>>> =
            (0..nb).map(|_| SyncCell::new(None)).collect();
        let next = AtomicUsize::new(0);
        let failed = AtomicBool::new(false);
        let step = if nb >= workers * 16 { 8 } else { 2 };

        crate::pool::parallel_try_run(workers, |_wid| {
            loop {
                if failed.load(Ordering::Relaxed) {
                    break;
                }
                let base = next.fetch_add(step, Ordering::Relaxed);
                if base >= nb {
                    break;
                }
                let end = (base + step).min(nb);
                for b in base..end {
                    let ib = &in_blocks[b];
                    let res: Result<IndepBlock<'a>> = (|| {
                        if ib.info.kind == BLOCK_COLUMN {
                            if let Some(gb) =
                                crate::columns::peek_grammar_boundary_indep(ib.payload)?
                            {
                                return Ok(IndepBlock::Grammar(gb));
                            }
                        }
                        let mut raw_vec = Vec::new();
                        ib.decode_into(options, &mut raw_vec)?;
                        let (rc, ends_nl) = count_raw_rows(&raw_vec);
                        Ok(IndepBlock::Decoded {
                            raw_vec,
                            rc,
                            ends_nl,
                        })
                    })();
                    let is_err = res.is_err();
                    unsafe {
                        *slots[b].get() = Some(res);
                    }
                    if is_err {
                        failed.store(true, Ordering::Relaxed);
                        return Ok(());
                    }
                }
            }
            Ok(())
        })?;

        let mut unpacked: Vec<IndepBlock<'a>> = Vec::with_capacity(nb);
        let mut first_err: Option<crate::Error> = None;
        for slot in slots {
            match slot.into_inner() {
                Some(Ok(v)) if first_err.is_none() => unpacked.push(v),
                Some(Ok(_)) => {}
                Some(Err(e)) => {
                    if first_err.is_none() {
                        first_err = Some(e);
                    }
                }
                None => {}
            }
        }
        if let Some(e) = first_err {
            return Err(e);
        }

        for (b, indep) in unpacked.into_iter().enumerate() {
            let had_in = incoming;
            let cg = cur_global;
            match indep {
                IndepBlock::Grammar(gb) if !(had_in && !gb.first_is_raw && gb.row_count > 0) => {
                    if gb.row_count > 0 {
                        cur_global += gb.row_count - usize::from(had_in);
                        incoming = gb.last_unterminated_raw.is_some();
                        any_unterm |= incoming;
                        non_empty.push(b);
                    }
                    meta.push(ParMeta {
                        decoded_raw: None,
                        row_count: gb.row_count,
                        had_in_carry: had_in,
                        cur_global: cg,
                        first_frag: if had_in { gb.first_raw } else { None },
                        last_unterm_frag: gb.last_unterminated_raw,
                    });
                }
                IndepBlock::Grammar(_) => {
                    let ib = &in_blocks[b];
                    let mut raw_vec = Vec::new();
                    ib.decode_into(options, &mut raw_vec)?;
                    let (rc, ends_nl) = count_raw_rows(&raw_vec);
                    if rc > 0 {
                        cur_global += rc - usize::from(had_in);
                        incoming = !ends_nl;
                        any_unterm |= incoming;
                        non_empty.push(b);
                    }
                    meta.push(finish_decoded_meta(raw_vec, rc, ends_nl, had_in, cg));
                }
                IndepBlock::Decoded {
                    raw_vec,
                    rc,
                    ends_nl,
                } => {
                    if rc > 0 {
                        cur_global += rc - usize::from(had_in);
                        incoming = !ends_nl;
                        any_unterm |= incoming;
                        non_empty.push(b);
                    }
                    meta.push(finish_decoded_meta(raw_vec, rc, ends_nl, had_in, cg));
                }
            }
        }
        return Ok((meta, non_empty, any_unterm, cur_global));
    }

    for (b, ib) in in_blocks.iter().enumerate() {
        let had_in = incoming;
        let cg = cur_global;
        if ib.info.kind == BLOCK_COLUMN {
            if let Some(gb) = crate::columns::peek_grammar_boundary_fast(ib.payload, had_in)? {
                if gb.row_count > 0 {
                    cur_global += gb.row_count - usize::from(had_in);
                    incoming = gb.last_unterminated_raw.is_some();
                    any_unterm |= incoming;
                    non_empty.push(b);
                }
                meta.push(ParMeta {
                    decoded_raw: None,
                    row_count: gb.row_count,
                    had_in_carry: had_in,
                    cur_global: cg,
                    first_frag: gb.first_raw,
                    last_unterm_frag: gb.last_unterminated_raw,
                });
                continue;
            }
        }
        let mut raw_vec = Vec::new();
        ib.decode_into(options, &mut raw_vec)?;
        let (rc, ends_nl) = count_raw_rows(&raw_vec);
        if rc > 0 {
            cur_global += rc - usize::from(had_in);
            incoming = !ends_nl;
            any_unterm |= incoming;
            non_empty.push(b);
        }
        meta.push(finish_decoded_meta(raw_vec, rc, ends_nl, had_in, cg));
    }
    Ok((meta, non_empty, any_unterm, cur_global))
}

#[allow(clippy::type_complexity)]
pub(crate) fn build_window_meta_and_boundary<'a>(
    in_blocks: &[ParInBlock<'a>],
    carry_in: Option<Vec<u8>>,
    base_global: usize,
    is_last_window: bool,
    options: &DecodeOptions,
) -> Result<(
    Vec<ParMeta<'a>>,
    Vec<usize>,
    Option<(usize, Vec<u8>)>,
    BoundarySet,
    Option<Vec<u8>>,
    usize,
)> {
    let (meta, non_empty, any_unterm, next_global) =
        inspect_par_meta_from(in_blocks, carry_in.is_some(), base_global, options)?;
    if !any_unterm && carry_in.is_none() {
        return Ok((
            meta,
            non_empty,
            None,
            BoundarySet { arena: Vec::new(), entries: Vec::new() },
            None,
            next_global,
        ));
    }

    let mut carry_in_row = None;
    let mut carry_out = None;
    let mut k_idx = 0usize;

    if let Some(mut prefix) = carry_in {
        let g_idx = base_global.saturating_sub(1);
        if non_empty.is_empty() {
            if is_last_window {
                carry_in_row = Some((g_idx, prefix));
            } else {
                carry_out = Some(prefix);
            }
            return Ok((
                meta,
                non_empty,
                carry_in_row,
                BoundarySet { arena: Vec::new(), entries: Vec::new() },
                carry_out,
                next_global,
            ));
        }
        let mut m = 0usize;
        while m < non_empty.len() {
            let bm = non_empty[m];
            let Some(ref fm) = meta[bm].first_frag else {
                break;
            };
            prefix.extend_from_slice(fm);
            if meta[bm].row_count == 1 && meta[bm].last_unterm_frag.is_some() {
                m += 1;
            } else {
                break;
            }
        }
        k_idx = m;
        if !prefix.ends_with(b"\n") && !is_last_window {
            carry_out = Some(prefix);
            return Ok((
                meta,
                non_empty,
                None,
                BoundarySet { arena: Vec::new(), entries: Vec::new() },
                carry_out,
                next_global,
            ));
        }
        carry_in_row = Some((g_idx, prefix));
    }

    let mut arena = Vec::with_capacity(non_empty.len() * 128);
    let mut entries = Vec::with_capacity(non_empty.len());
    while k_idx < non_empty.len() {
        let b0 = non_empty[k_idx];
        let Some(ref f0) = meta[b0].last_unterm_frag else {
            k_idx += 1;
            continue;
        };
        let base_b0 = meta[b0].cur_global - usize::from(meta[b0].had_in_carry);
        let start_row = meta[b0].row_count - 1;
        let start_glob = base_b0 + start_row;
        let start = arena.len();
        arena.extend_from_slice(f0);
        let mut m = k_idx + 1;
        while m < non_empty.len() {
            let bm = non_empty[m];
            let Some(ref fm) = meta[bm].first_frag else {
                break;
            };
            arena.extend_from_slice(fm);
            if meta[bm].row_count == 1 && meta[bm].last_unterm_frag.is_some() {
                m += 1;
            } else {
                break;
            }
        }
        let end = arena.len();
        if !arena[start..end].ends_with(b"\n") && !is_last_window {
            carry_out = Some(arena[start..end].to_vec());
            arena.truncate(start);
        } else {
            entries.push(BoundaryEntry {
                block_index: b0,
                row_index: start_row,
                global_row_index: start_glob,
                start,
                end,
            });
        }
        k_idx = m.max(k_idx + 1);
    }

    Ok((meta, non_empty, carry_in_row, BoundarySet { arena, entries }, carry_out, next_global))
}

pub(crate) fn build_boundary_set(
    meta: &[ParMeta<'_>],
    non_empty: &[usize],
    any_unterm: bool,
) -> BoundarySet {
    if !any_unterm {
        return BoundarySet {
            arena: Vec::new(),
            entries: Vec::new(),
        };
    }

    let mut arena = Vec::with_capacity(non_empty.len() * 128);
    let mut entries = Vec::with_capacity(non_empty.len());
    let mut k_idx = 0usize;
    while k_idx < non_empty.len() {
        let b0 = non_empty[k_idx];
        let Some(ref f0) = meta[b0].last_unterm_frag else {
            k_idx += 1;
            continue;
        };
        let base_b0 = meta[b0].cur_global - usize::from(meta[b0].had_in_carry);
        let start_row = meta[b0].row_count - 1;
        let start_glob = base_b0 + start_row;
        let start = arena.len();
        arena.extend_from_slice(f0);
        let mut m = k_idx + 1;
        while m < non_empty.len() {
            let bm = non_empty[m];
            let Some(ref fm) = meta[bm].first_frag else {
                break;
            };
            arena.extend_from_slice(fm);
            if meta[bm].row_count == 1 && meta[bm].last_unterm_frag.is_some() {
                m += 1;
            } else {
                break;
            }
        }
        let end = arena.len();
        entries.push(BoundaryEntry {
            block_index: b0,
            row_index: start_row,
            global_row_index: start_glob,
            start,
            end,
        });
        k_idx = m.max(k_idx + 1);
    }
    BoundarySet { arena, entries }
}

pub(crate) fn build_par_meta_and_boundary<'a>(
    in_blocks: &[ParInBlock<'a>],
    options: &DecodeOptions,
) -> Result<(Vec<ParMeta<'a>>, BoundarySet)> {
    let (meta, non_empty, any_unterm) = inspect_par_meta(in_blocks, options)?;
    let bset = build_boundary_set(&meta, &non_empty, any_unterm);
    Ok((meta, bset))
}
