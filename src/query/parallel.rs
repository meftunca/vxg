use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::ops::par_meta::{build_par_meta_and_boundary, collect_par_blocks, ParInBlock, SyncCell};
use crate::{DecodeOptions, Frame, Result, BLOCK_COLUMN};

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

fn filter_parallel(
    in_blocks: &[ParInBlock<'_>],
    pred: &Pred<'_>,
    limit: usize,
    options: &DecodeOptions,
) -> Result<Vec<RowMatch>> {
    let nb = in_blocks.len();
    let (meta, boundary_lines) = build_par_meta_and_boundary(in_blocks, options)?;
    let mut boundary_matches = Vec::new();
    for bl in boundary_lines.iter() {
        if pred.matches(bl.bytes) {
            boundary_matches.push(RowMatch {
                block_index: bl.block_index,
                row_index: bl.row_index,
                global_row_index: bl.global_row_index,
                bytes: bl.bytes.to_vec(),
            });
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

    let run_worker = |_wid: usize| {
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

    crate::pool::parallel_run(workers, run_worker);

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
        let in_blocks = collect_par_blocks(frame_bytes, &frames);
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
                let payload = parsed.blocks[local_blk].payload;
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
