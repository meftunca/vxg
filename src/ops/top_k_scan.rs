// Single-thread and work-stealing parallel frame scanner for `Top-K` queries.

#[allow(clippy::too_many_arguments)]
pub(crate) fn scan_frame_into_heap(
    frame_bytes: &[u8],
    global_block_offset: usize,
    global_row_offset: usize,
    where_expr: Option<&PredExpr<'_>>,
    project_cols: Option<&[ColTarget<'_>]>,
    order_col: ColTarget<'_>,
    options: &DecodeOptions,
    heap: &mut TopKHeap,
) -> Result<(usize, usize)> {
    let frames = Frame::parse_all(frame_bytes, options.max_output)?;
    if heap.capacity_k() == 0 {
        return Ok((0, 0));
    }
    let in_blocks = collect_par_blocks(frame_bytes, &frames);
    let nb = in_blocks.len();
    if nb == 0 {
        return Ok((0, 0));
    }

    if options.threads > 1 && nb > 1 {
        let (meta, boundary_lines) = build_par_meta_and_boundary(&in_blocks, options)?;
        let total_rows = meta
            .iter()
            .filter(|m| m.row_count > 0)
            .map(|m| m.row_count - usize::from(m.had_in_carry))
            .sum::<usize>();
        for bl in boundary_lines.iter() {
            heap.push_line_if_match(
                bl.bytes,
                global_block_offset + bl.block_index,
                bl.row_index,
                global_row_offset + bl.global_row_index,
                where_expr,
                project_cols,
                order_col,
                None,
            );
        }

        let init_shared = match heap.dir() {
            SortDir::Asc => heap.worst_num_bound().map_or(u64::MAX, |(b, _)| b),
            SortDir::Desc => heap.worst_num_bound().map_or(0, |(b, _)| b),
        };
        let shared_bound = AtomicU64::new(init_shared);
        let seed_floor = heap.worst_floor();
        let workers = options.threads.min(nb);
        let next = AtomicUsize::new(0);
        let failed = AtomicBool::new(false);
        let worker_heaps: Vec<SyncCell<Option<Result<TopKHeap>>>> =
            (0..workers).map(|_| SyncCell::new(None)).collect();
        let step = if nb >= workers * 8 { 4 } else { 1 };
        let k = heap.capacity_k();
        let dir = heap.dir();
        let skip_row_0 = heap.skip_row_0();

        let run_worker = |wid: usize| {
            let mut dec_buf = Vec::new();
            let mut local_heap = TopKHeap::with_floor(k, dir, skip_row_0, seed_floor.clone());
            loop {
                if failed.load(AtomicOrdering::Relaxed) {
                    break;
                }
                let base = next.fetch_add(step, AtomicOrdering::Relaxed);
                if base >= nb {
                    break;
                }
                let end = (base + step).min(nb);
                for b in base..end {
                    let mb = &meta[b];
                    if mb.row_count == 0 {
                        continue;
                    }
                    let ib = &in_blocks[b];
                    let g_blk = global_block_offset + b;
                    let g_cur = global_row_offset + mb.cur_global;
                    let res: Result<()> = (|| {
                        if mb.decoded_raw.is_none() {
                            let mut dummy = mb.had_in_carry.then(|| (usize::MAX, 0, 0, Vec::new()));
                            if crate::columns::topk_grammar_block(
                                ib.payload,
                                g_blk,
                                g_cur,
                                &mut dummy,
                                false,
                                where_expr,
                                project_cols,
                                order_col,
                                Some(&shared_bound),
                                &mut local_heap,
                            )?
                            .is_some()
                            {
                                return Ok(());
                            }
                        }
                        let raw: &[u8] = if let Some(ref pre) = mb.decoded_raw {
                            pre
                        } else {
                            query::decode_block_into(
                                ib.f_bytes,
                                ib.local_blk,
                                ib.info.raw_size,
                                options,
                                &mut dec_buf,
                            )?;
                            &dec_buf
                        };
                        if raw.is_empty() {
                            return Ok(());
                        }
                        let base_row = g_cur - usize::from(mb.had_in_carry);
                        let lines: Vec<&[u8]> = raw.split_inclusive(|&c| c == b'\n').collect();
                        let start_i = usize::from(mb.had_in_carry);
                        let end_i = lines.len()
                            - usize::from(lines.last().is_some_and(|l| !l.ends_with(b"\n")));
                        for (row_idx, &lb) in lines.iter().enumerate().take(end_i).skip(start_i) {
                            local_heap.push_line_if_match(
                                lb,
                                g_blk,
                                row_idx,
                                base_row + row_idx,
                                where_expr,
                                project_cols,
                                order_col,
                                Some(&shared_bound),
                            );
                        }
                        Ok(())
                    })();
                    if let Err(e) = res {
                        unsafe {
                            *worker_heaps[wid].get() = Some(Err(e));
                        }
                        failed.store(true, AtomicOrdering::Relaxed);
                        return;
                    }
                }
            }
            unsafe {
                *worker_heaps[wid].get() = Some(Ok(local_heap));
            }
        };

        crate::pool::parallel_run(workers, run_worker);

        for slot in worker_heaps {
            if let Some(res) = slot.into_inner() {
                heap.merge_from(res?);
            }
        }
        return Ok((nb, total_rows));
    }

    let mut cur_global = global_row_offset;
    let mut carry: Option<(usize, usize, usize, Vec<u8>)> = None;
    let mut buf: Vec<u8> = Vec::new();
    for (local_idx, ib) in in_blocks.iter().enumerate() {
        let blk_idx = global_block_offset + local_idx;
        if ib.info.kind == BLOCK_COLUMN {
            if let Some(new_rows) = crate::columns::topk_grammar_block(
                ib.payload,
                blk_idx,
                cur_global,
                &mut carry,
                true,
                where_expr,
                project_cols,
                order_col,
                None,
                heap,
            )? {
                cur_global += new_rows;
                continue;
            }
        }
        query::decode_block_into(
            ib.f_bytes,
            ib.local_blk,
            ib.info.raw_size,
            options,
            &mut buf,
        )?;
        if buf.is_empty() {
            continue;
        }
        let had_carry = carry.is_some();
        let base_row = cur_global - usize::from(had_carry);
        let lines: Vec<&[u8]> = buf.split_inclusive(|&b| b == b'\n').collect();
        let blk_rows = lines.len();
        for (row_idx, &line_bytes) in lines.iter().enumerate() {
            if row_idx == 0 {
                if let Some((c_blk, c_row, c_glob, mut c_bytes)) = carry.take() {
                    c_bytes.extend_from_slice(line_bytes);
                    if !c_bytes.ends_with(b"\n") && blk_rows == 1 {
                        carry = Some((c_blk, c_row, c_glob, c_bytes));
                        break;
                    }
                    heap.push_line_if_match(
                        &c_bytes,
                        c_blk,
                        c_row,
                        c_glob,
                        where_expr,
                        project_cols,
                        order_col,
                        None,
                    );
                    continue;
                }
            }
            if row_idx + 1 == blk_rows && !line_bytes.ends_with(b"\n") {
                carry = Some((blk_idx, row_idx, base_row + row_idx, line_bytes.to_vec()));
                break;
            }
            heap.push_line_if_match(
                line_bytes,
                blk_idx,
                row_idx,
                base_row + row_idx,
                where_expr,
                project_cols,
                order_col,
                None,
            );
        }
        cur_global = base_row + blk_rows;
    }
    if let Some((c_blk, c_row, c_glob, c_bytes)) = carry.take() {
        heap.push_line_if_match(
            &c_bytes,
            c_blk,
            c_row,
            c_glob,
            where_expr,
            project_cols,
            order_col,
            None,
        );
    }
    Ok((nb, cur_global - global_row_offset))
}

pub(super) fn top_k_by_impl(
    frame_bytes: &[u8],
    where_expr: Option<&PredExpr<'_>>,
    order_col: ColTarget<'_>,
    dir: SortDir,
    limit: usize,
    skip_row_0: bool,
    options: &DecodeOptions,
) -> Result<Vec<RowMatch>> {
    options.validate()?;
    if limit == 0 {
        let _ = Frame::parse_all(frame_bytes, options.max_output)?;
        return Ok(Vec::new());
    }
    let mut heap = TopKHeap::new(limit, dir);
    heap.set_skip_row_0(skip_row_0);
    scan_frame_into_heap(
        frame_bytes,
        0,
        0,
        where_expr,
        None,
        order_col,
        options,
        &mut heap,
    )?;
    Ok(heap.into_sorted_matches())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn select_top_k_by_impl(
    frame_bytes: &[u8],
    where_expr: Option<&PredExpr<'_>>,
    cols: &[ColTarget<'_>],
    order_col: ColTarget<'_>,
    dir: SortDir,
    limit: usize,
    skip_row_0: bool,
    options: &DecodeOptions,
) -> Result<Vec<u8>> {
    options.validate()?;
    validate_projection_cols(cols)?;
    if limit == 0 {
        let _ = Frame::parse_all(frame_bytes, options.max_output)?;
        return Ok(Vec::new());
    }
    let mut heap = TopKHeap::new(limit, dir);
    heap.set_skip_row_0(skip_row_0);
    scan_frame_into_heap(
        frame_bytes,
        0,
        0,
        where_expr,
        Some(cols),
        order_col,
        options,
        &mut heap,
    )?;
    Ok(heap.into_concat_bytes())
}
