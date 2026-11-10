//! Single-pass contiguous-chunk parallel reducer for order-independent operations
//! (`count_where`, `aggregate_column`, `group_by`).
//!
//! Splits `in_blocks` into `workers` contiguous chunks, peeks only the `<= workers - 1`
//! inter-chunk seam blocks up front (`O(workers)` instead of `O(num_blocks)`), runs a single
//! thread-pool pass with sequential intra-chunk `carry`, and stitches seam lines on join.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::{DecodeOptions, Result, BLOCK_COLUMN};

use super::{ParInBlock, SyncCell};

struct ChunkOutcome<Acc> {
    acc: Acc,
    head_frag: Vec<u8>,
    head_open: bool,
    tail_carry: Option<Vec<u8>>,
}

fn block_ends_unterminated(ib: &ParInBlock<'_>, options: &DecodeOptions) -> Result<bool> {
    if ib.info.raw_size == 0 {
        return Ok(false);
    }
    if let Some((rows, ends_nl)) = ib.row_meta {
        return Ok(rows > 0 && !ends_nl);
    }
    if ib.info.kind == BLOCK_COLUMN {
        if let Some(gb) = crate::columns::peek_grammar_boundary_indep(ib.payload)? {
            if gb.row_count == 0 {
                return Ok(false);
            }
            return Ok(gb.last_unterminated_raw.is_some());
        }
    }
    let mut raw = Vec::new();
    ib.decode_into(options, &mut raw)?;
    Ok(raw.last().is_some_and(|&b| b != b'\n'))
}

#[inline]
fn feed_decoded_lines<Acc, FL>(
    bytes: &[u8],
    carry: &mut Option<Vec<u8>>,
    acc: &mut Acc,
    eval_line: &FL,
) where
    FL: Fn(&[u8], &mut Acc) + Sync,
{
    if bytes.is_empty() {
        return;
    }
    let mut pos = 0usize;
    if let Some(mut c_bytes) = carry.take() {
        if let Some(nl_rel) = bytes.iter().position(|&b| b == b'\n') {
            c_bytes.extend_from_slice(&bytes[..=nl_rel]);
            eval_line(&c_bytes, acc);
            pos = nl_rel + 1;
        } else {
            c_bytes.extend_from_slice(bytes);
            *carry = Some(c_bytes);
            return;
        }
    }
    while pos < bytes.len() {
        match bytes[pos..].iter().position(|&b| b == b'\n') {
            Some(nl_rel) => {
                let end = pos + nl_rel + 1;
                eval_line(&bytes[pos..end], acc);
                pos = end;
            }
            None => {
                *carry = Some(bytes[pos..].to_vec());
                break;
            }
        }
    }
}

/// Execute a single-barrier parallel reduction over `in_blocks` (`nb >= 2`).
///
/// - `eval_grammar(payload, carry, eval_in_carry, acc)` evaluates one `BLOCK_COLUMN` block:
///   when `eval_in_carry` is `false` and `carry` is `Some`, row 0 (incoming carry completion) is
///   skipped while outgoing unterminated last-row carry is still written into `*carry`.
///   Returns `Ok(true)` if handled in compressed domain, `Ok(false)` to fall back to `decode_block_into`.
pub(crate) fn parallel_reduce_blocks<Acc, FG, FL, FM>(
    in_blocks: &[ParInBlock<'_>],
    options: &DecodeOptions,
    eval_grammar: FG,
    eval_line: FL,
    merge_acc: FM,
) -> Result<Acc>
where
    Acc: Default + Send,
    FG: Fn(&[u8], &mut Option<Vec<u8>>, bool, &mut Acc) -> Result<bool> + Sync,
    FL: Fn(&[u8], &mut Acc) + Sync,
    FM: Fn(&mut Acc, Acc),
{
    let nb = in_blocks.len();
    let workers = options.threads.min(nb).max(1);
    let target_chunks = if nb >= workers.saturating_mul(16) {
        workers.saturating_mul(8).min(256).min(nb)
    } else {
        workers
    };
    let chunk_len = nb.div_ceil(target_chunks);
    let num_chunks = nb.div_ceil(chunk_len);
    let actual_workers = workers.min(num_chunks);

    let mut had_in = vec![false; num_chunks];
    for (cid, slot) in had_in.iter_mut().enumerate().skip(1) {
        let start = cid * chunk_len;
        if let Some(prev_idx) = (0..start).rev().find(|&p| in_blocks[p].info.raw_size > 0) {
            *slot = block_ends_unterminated(&in_blocks[prev_idx], options)?;
        }
    }

    let next_chunk = AtomicUsize::new(0);
    let failed = AtomicBool::new(false);
    let outcomes: Vec<SyncCell<Option<Result<ChunkOutcome<Acc>>>>> =
        (0..num_chunks).map(|_| SyncCell::new(None)).collect();

    crate::pool::parallel_run(actual_workers, |_wid| {
        let mut dec_buf = Vec::new();
        loop {
            if failed.load(Ordering::Relaxed) {
                return;
            }
            let cid = next_chunk.fetch_add(1, Ordering::Relaxed);
            if cid >= num_chunks {
                return;
            }
            let start = cid * chunk_len;
            let end = (start + chunk_len).min(nb);
            let mut acc = Acc::default();
            let mut head_open = had_in[cid];
            let mut head_frag = Vec::new();
            let mut carry: Option<Vec<u8>> = None;

            for ib in &in_blocks[start..end] {
                if failed.load(Ordering::Relaxed) {
                    return;
                }
                if ib.info.raw_size == 0 {
                    continue;
                }
                let step_res: Result<()> = (|| {
                    if head_open {
                        if ib.info.kind == BLOCK_COLUMN {
                            if let Some(fb) =
                                crate::columns::peek_grammar_boundary_fast(ib.payload, true)?
                            {
                                if fb.row_count == 0 {
                                    return Ok(());
                                }
                                if fb.row_count == 1 && fb.last_unterminated_raw.is_some() {
                                    head_frag
                                        .extend_from_slice(&fb.last_unterminated_raw.unwrap());
                                    return Ok(());
                                }
                                if fb.first_raw.as_deref().is_some_and(|s| s.ends_with(b"\n")) {
                                    let mut out_carry = Some(Vec::new());
                                    if eval_grammar(ib.payload, &mut out_carry, false, &mut acc)? {
                                        if let Some(first) = fb.first_raw {
                                            head_frag.extend_from_slice(&first);
                                        }
                                        head_open = false;
                                        carry = out_carry;
                                        return Ok(());
                                    }
                                }
                            }
                        }
                        ib.decode_into(options, &mut dec_buf)?;
                        if dec_buf.is_empty() {
                            return Ok(());
                        }
                        if let Some(nl_pos) = dec_buf.iter().position(|&b| b == b'\n') {
                            head_frag.extend_from_slice(&dec_buf[..=nl_pos]);
                            head_open = false;
                            feed_decoded_lines(
                                &dec_buf[nl_pos + 1..],
                                &mut carry,
                                &mut acc,
                                &eval_line,
                            );
                        } else {
                            head_frag.extend_from_slice(&dec_buf);
                        }
                        return Ok(());
                    }

                    if ib.info.kind == BLOCK_COLUMN
                        && eval_grammar(ib.payload, &mut carry, true, &mut acc)?
                    {
                        return Ok(());
                    }
                    ib.decode_into(options, &mut dec_buf)?;
                    feed_decoded_lines(&dec_buf, &mut carry, &mut acc, &eval_line);
                    Ok(())
                })();

                if let Err(e) = step_res {
                    unsafe {
                        *outcomes[cid].get() = Some(Err(e));
                    }
                    failed.store(true, Ordering::Relaxed);
                    return;
                }
            }

            unsafe {
                *outcomes[cid].get() = Some(Ok(ChunkOutcome {
                    acc,
                    head_frag,
                    head_open,
                    tail_carry: carry,
                }));
            }
        }
    });

    let mut unpacked = Vec::with_capacity(num_chunks);
    let mut first_err = None;
    for slot in outcomes {
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

    let mut total = Acc::default();
    let mut seam_carry: Option<Vec<u8>> = None;
    for (wid, out) in unpacked.into_iter().enumerate() {
        if had_in[wid] {
            let mut line = seam_carry.take().unwrap_or_default();
            line.extend_from_slice(&out.head_frag);
            if out.head_open {
                seam_carry = Some(line);
                continue;
            }
            if !line.is_empty() {
                eval_line(&line, &mut total);
            }
        } else if let Some(prev_line) = seam_carry.take() {
            eval_line(&prev_line, &mut total);
        }
        merge_acc(&mut total, out.acc);
        seam_carry = out.tail_carry;
    }
    if let Some(last_line) = seam_carry.take() {
        eval_line(&last_line, &mut total);
    }
    Ok(total)
}
