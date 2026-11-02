use std::borrow::Cow;
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::{frame, DecodeOptions, Frame, Result, BLOCK_COLUMN};

use super::numeric::{self, NumFilter};
use super::{contains_bytes, decode_block_into, frame_slices, RowMatch};

pub(super) enum Pred<'a> {
    Substr(&'a [u8]),
    Num(&'a NumFilter),
}

impl Pred<'_> {
    #[inline]
    pub(super) fn matches(&self, line: &[u8]) -> bool {
        match self {
            Pred::Substr(n) => contains_bytes(line, n),
            Pred::Num(f) => numeric::line_numeric_match(line, f),
        }
    }

    /// Whole-block rejection test usable only for substring predicates.
    #[inline]
    pub(super) fn block_misses(&self, raw: &[u8]) -> bool {
        matches!(self, Pred::Substr(n) if !contains_bytes(raw, n))
    }
}

struct SyncCell<T>(UnsafeCell<T>);
// SAFETY: Each slot `b` is written at most once by the worker claiming `b` via atomic `fetch_add`
// and read only after `std::thread::scope` joins.
unsafe impl<T: Send> Sync for SyncCell<T> {}

impl<T> SyncCell<T> {
    #[inline]
    const fn new(v: T) -> Self {
        Self(UnsafeCell::new(v))
    }
    #[inline]
    fn get(&self) -> *mut T {
        self.0.get()
    }
    #[inline]
    fn into_inner(self) -> T {
        self.0.into_inner()
    }
}

struct QueryInBlock<'a> {
    f_bytes: &'a [u8],
    local_blk: usize,
    info: frame::BlockInfo,
    payload: &'a [u8],
}

struct QueryMeta<'a> {
    decoded_raw: Option<Vec<u8>>,
    row_count: usize,
    had_in_carry: bool,
    cur_global: usize,
    first_frag: Option<Cow<'a, [u8]>>,
    last_unterm_frag: Option<Cow<'a, [u8]>>,
}

fn filter_parallel(
    in_blocks: &[QueryInBlock<'_>],
    pred: &Pred<'_>,
    limit: usize,
    options: &DecodeOptions,
) -> Result<Vec<RowMatch>> {
    let nb = in_blocks.len();
    let mut meta: Vec<QueryMeta<'_>> = Vec::with_capacity(nb);
    let mut non_empty = Vec::with_capacity(nb);
    let mut incoming = false;
    let mut any_unterm = false;
    let mut cur_global = 0usize;

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
                meta.push(QueryMeta {
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
        decode_block_into(
            ib.f_bytes,
            ib.local_blk,
            ib.info.raw_size,
            options,
            &mut raw_vec,
        )?;
        let rc = raw_vec.split_inclusive(|&c| c == b'\n').count();
        let ends_nl = raw_vec.last() == Some(&b'\n');
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
        if rc > 0 {
            cur_global += rc - usize::from(had_in);
            incoming = !ends_nl;
            any_unterm |= incoming;
            non_empty.push(b);
        }
        meta.push(QueryMeta {
            decoded_raw: Some(raw_vec),
            row_count: rc,
            had_in_carry: had_in,
            cur_global: cg,
            first_frag,
            last_unterm_frag,
        });
    }

    let mut boundary_matches = Vec::new();
    if any_unterm {
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
            let mut full = f0.as_ref().to_vec();
            let mut m = k_idx + 1;
            while m < non_empty.len() {
                let bm = non_empty[m];
                let Some(ref fm) = meta[bm].first_frag else {
                    break;
                };
                full.extend_from_slice(fm);
                if meta[bm].row_count == 1 && meta[bm].last_unterm_frag.is_some() {
                    m += 1;
                } else {
                    break;
                }
            }
            if pred.matches(&full) {
                boundary_matches.push(RowMatch {
                    block_index: b0,
                    row_index: start_row,
                    global_row_index: start_glob,
                    bytes: full,
                });
            }
            k_idx = m.max(k_idx + 1);
        }
    }

    let workers = options.threads.min(nb.max(1));
    let next = AtomicUsize::new(0);
    let cutoff = AtomicUsize::new(nb);
    let failed = AtomicBool::new(false);
    let slots: Vec<SyncCell<Option<Result<Vec<RowMatch>>>>> =
        (0..nb).map(|_| SyncCell::new(None)).collect();
    let step = if limit == usize::MAX && nb >= workers * 8 {
        4
    } else {
        1
    };
    let needle = match pred {
        Pred::Substr(n) => *n,
        Pred::Num(_) => &[],
    };
    let num_opt = match pred {
        Pred::Num(f) => Some(*f),
        Pred::Substr(_) => None,
    };

    let run_worker = || {
        let mut dec_buf = Vec::new();
        loop {
            if failed.load(Ordering::Relaxed) {
                break;
            }
            let base = next.fetch_add(step, Ordering::Relaxed);
            if base >= nb || base > cutoff.load(Ordering::Relaxed) {
                break;
            }
            let end = (base + step).min(nb);
            for b in base..end {
                if b > cutoff.load(Ordering::Relaxed) {
                    break;
                }
                let mb = &meta[b];
                if mb.row_count == 0 {
                    continue;
                }
                let ib = &in_blocks[b];
                let blk_lim = limit.saturating_add(usize::from(mb.had_in_carry));
                let res: Result<Vec<RowMatch>> = (|| {
                    let mut local_out = Vec::new();
                    if mb.decoded_raw.is_none() {
                        let mut dummy = if mb.had_in_carry {
                            Some((usize::MAX, 0, 0, Vec::new()))
                        } else {
                            None
                        };
                        if crate::columns::search_grammar_block(
                            ib.payload,
                            b,
                            mb.cur_global,
                            &mut dummy,
                            needle,
                            blk_lim,
                            &mut local_out,
                            num_opt,
                        )?
                        .is_some()
                        {
                            local_out.retain(|rm| rm.block_index != usize::MAX);
                            return Ok(local_out);
                        }
                        local_out.clear();
                    }
                    let raw: &[u8] = if let Some(ref pre) = mb.decoded_raw {
                        pre
                    } else {
                        decode_block_into(
                            ib.f_bytes,
                            ib.local_blk,
                            ib.info.raw_size,
                            options,
                            &mut dec_buf,
                        )?;
                        &dec_buf
                    };
                    if raw.is_empty() || pred.block_misses(raw) {
                        return Ok(local_out);
                    }
                    let base_row = mb.cur_global - usize::from(mb.had_in_carry);
                    let lines: Vec<&[u8]> = raw.split_inclusive(|&c| c == b'\n').collect();
                    let start_i = usize::from(mb.had_in_carry);
                    let end_i = lines.len()
                        - usize::from(lines.last().is_some_and(|l| !l.ends_with(b"\n")));
                    for (row_idx, &lb) in lines.iter().enumerate().take(end_i).skip(start_i) {
                        if pred.matches(lb) {
                            local_out.push(RowMatch {
                                block_index: b,
                                row_index: row_idx,
                                global_row_index: base_row + row_idx,
                                bytes: lb.to_vec(),
                            });
                            if local_out.len() >= limit {
                                break;
                            }
                        }
                    }
                    Ok(local_out)
                })();
                match res {
                    Ok(v) => {
                        if limit < usize::MAX && v.len() >= limit {
                            cutoff.fetch_min(b, Ordering::Relaxed);
                        }
                        if !v.is_empty() {
                            unsafe {
                                *slots[b].get() = Some(Ok(v));
                            }
                        }
                    }
                    Err(e) => {
                        unsafe {
                            *slots[b].get() = Some(Err(e));
                        }
                        failed.store(true, Ordering::Relaxed);
                        return;
                    }
                }
            }
        }
    };

    std::thread::scope(|s| {
        for _ in 1..workers {
            s.spawn(run_worker);
        }
        run_worker();
    });

    let cut = cutoff.load(Ordering::Relaxed);
    let mut out = Vec::new();
    for (b, slot) in slots.into_iter().enumerate() {
        if let Some(res) = slot.into_inner() {
            let v = res?;
            if b <= cut {
                out.extend(v);
            }
        }
    }
    if !boundary_matches.is_empty() {
        out.extend(boundary_matches);
        out.sort_unstable_by_key(|rm| rm.global_row_index);
    }
    if out.len() > limit {
        out.truncate(limit);
    }
    Ok(out)
}

pub(super) fn filter_impl(
    frame_bytes: &[u8],
    pred: &Pred<'_>,
    limit: usize,
    options: &DecodeOptions,
) -> Result<Vec<RowMatch>> {
    options.validate()?;
    if limit == 0 {
        return Ok(Vec::new());
    }
    let frames = Frame::parse_all(frame_bytes, options.max_output)?;
    if options.threads > 1 {
        let mut in_blocks = Vec::new();
        for (f_bytes, parsed) in frame_slices(frame_bytes, &frames) {
            for (local_blk, info) in parsed.blocks().enumerate() {
                in_blocks.push(QueryInBlock {
                    f_bytes,
                    local_blk,
                    info,
                    payload: &f_bytes[info.payload_offset..info.payload_offset + info.payload_size],
                });
            }
        }
        if in_blocks.len() > 1 {
            return filter_parallel(&in_blocks, pred, limit, options);
        }
    }

    let mut out = Vec::new();
    let mut cur_global = 0usize;
    let mut carry: Option<(usize, usize, usize, Vec<u8>)> = None;
    let mut buf: Vec<u8> = Vec::new();
    let mut global_blk = 0usize;

    for (f_bytes, parsed) in frame_slices(frame_bytes, &frames) {
        for (local_blk, info) in parsed.blocks().enumerate() {
            let blk_idx = global_blk;
            global_blk += 1;
            if out.len() >= limit && carry.is_none() {
                return Ok(out);
            }
            if info.kind == BLOCK_COLUMN {
                let payload = &f_bytes[info.payload_offset..info.payload_offset + info.payload_size];
                if let Some(new_rows) = crate::columns::search_grammar_block(
                    payload,
                    blk_idx,
                    cur_global,
                    &mut carry,
                    match pred {
                        Pred::Substr(n) => n,
                        Pred::Num(_) => &[],
                    },
                    limit,
                    &mut out,
                    match pred {
                        Pred::Num(f) => Some(*f),
                        Pred::Substr(_) => None,
                    },
                )? {
                    cur_global += new_rows;
                    if out.len() >= limit && carry.is_none() {
                        return Ok(out);
                    }
                    continue;
                }
            }
            decode_block_into(f_bytes, local_blk, info.raw_size, options, &mut buf)?;
            let raw: &[u8] = &buf;
            if raw.is_empty() {
                continue;
            }
            if carry.is_none() && pred.block_misses(raw) {
                let nl = raw.iter().filter(|&&b| b == b'\n').count();
                if raw[raw.len() - 1] == b'\n' {
                    cur_global += nl;
                } else {
                    let tail_start = raw.iter().rposition(|&b| b == b'\n').map_or(0, |p| p + 1);
                    carry = Some((blk_idx, nl, cur_global + nl, raw[tail_start..].to_vec()));
                    cur_global += nl + 1;
                }
                continue;
            }
            let had_carry = carry.is_some();
            let base_row = if had_carry { cur_global - 1 } else { cur_global };
            let lines: Vec<&[u8]> = raw.split_inclusive(|&b| b == b'\n').collect();
            let blk_rows = lines.len();
            for (row_idx, &line) in lines.iter().enumerate() {
                if row_idx == 0 {
                    if let Some((c_blk, c_row, c_glob, mut c_bytes)) = carry.take() {
                        c_bytes.extend_from_slice(line);
                        if !c_bytes.ends_with(b"\n") && blk_rows == 1 {
                            carry = Some((c_blk, c_row, c_glob, c_bytes));
                            break;
                        }
                        if pred.matches(&c_bytes) && out.len() < limit {
                            out.push(RowMatch {
                                block_index: c_blk,
                                row_index: c_row,
                                global_row_index: c_glob,
                                bytes: c_bytes,
                            });
                        }
                        continue;
                    }
                }
                if row_idx == blk_rows - 1 && !line.ends_with(b"\n") {
                    carry = Some((blk_idx, row_idx, base_row + row_idx, line.to_vec()));
                    break;
                }
                if pred.matches(line) && out.len() < limit {
                    out.push(RowMatch {
                        block_index: blk_idx,
                        row_index: row_idx,
                        global_row_index: base_row + row_idx,
                        bytes: line.to_vec(),
                    });
                }
            }
            cur_global = base_row + blk_rows;
        }
    }
    if let Some((c_blk, c_row, c_glob, c_bytes)) = carry.take() {
        if pred.matches(&c_bytes) && out.len() < limit {
            out.push(RowMatch {
                block_index: c_blk,
                row_index: c_row,
                global_row_index: c_glob,
                bytes: c_bytes,
            });
        }
    }
    Ok(out)
}
