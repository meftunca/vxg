//! Public filtered/projected Arrow export entry points (`export_arrow_batch_where`,
//! `export_arrow_batches_where`, `export_arrow_ipc_where`) and windowed `ArrowWindowStream`.

use std::collections::VecDeque;

use crate::{
    arrow::{ArrowBatch, ArrowCol, ArrowData, ArrowType},
    arrow_builder::ColBuilder,
    arrow_merge::{
        coalesce_arrow_batches, concat_arrow_batches, empty_arrow_batch, finalize_chunk_to_batch,
        infer_chunk_col_types,
    },
    arrow_scan::{
        infer_fallback_types, raw_record_batches, resolve_plan, scan_window_par, scan_window_seq,
        target_to_owned_and_name, ArrowWhere, ResolvedPlan, DEFAULT_ARROW_BATCH_ROWS,
        STREAM_WINDOW_BLOCKS,
    },
    frame::BlockInfo,
    ops::{
        csv_header::{with_opt_mapped_expr, DelimHeader, MappedExpr},
        par_meta::{collect_par_blocks, ParInBlock},
        ColTarget, PredExpr,
    },
    query,
    sql::{exec::with_opt_sql_where, OwnedCol, SelectExpr, SelectProjection, SqlQuery},
    DecodeOptions, Frame, Result,
};

fn execute_full_scan(
    frame_bytes: &[u8],
    explicit: Option<(Vec<String>, Vec<OwnedCol>)>,
    where_expr: Option<&PredExpr<'_>>,
    batch_size: usize,
    limit: Option<usize>,
    dec_options: &DecodeOptions,
) -> Result<(Vec<ArrowBatch>, Vec<String>, Vec<ArrowType>)> {
    dec_options.validate()?;
    let lim = limit.unwrap_or(usize::MAX);
    match resolve_plan(frame_bytes, explicit, where_expr, dec_options)? {
        ResolvedPlan::RawRecord => {
            raw_record_batches(frame_bytes, where_expr, batch_size, lim, dec_options)
        }
        ResolvedPlan::Columns { names, cols, hdr_opt, skip_header_row } => {
            let frames = Frame::parse_all(frame_bytes, dec_options.max_output)?;
            let in_blocks = collect_par_blocks(frame_bytes, &frames);
            let targets: Vec<ColTarget<'_>> = cols.iter().map(OwnedCol::as_target).collect();
            let mapped = match (where_expr, hdr_opt.as_ref()) {
                (Some(w), Some(hdr)) => Some(MappedExpr::from_expr(w, hdr)),
                _ => None,
            };
            let chunks: Vec<Vec<ColBuilder>> = if lim == 0 || in_blocks.is_empty() {
                Vec::new()
            } else {
                with_opt_mapped_expr(mapped.as_ref(), |mw| {
                    let eff_where = if mapped.is_some() { mw } else { where_expr };
                    let mut carry = None;
                    let mut rem = lim;
                    if dec_options.threads <= 1
                        || in_blocks.len() == 1
                        || (eff_where.is_none() && lim <= 4096)
                    {
                        let mut skip_r0 = skip_header_row;
                        scan_window_seq(
                            &in_blocks,
                            &names,
                            &targets,
                            eff_where,
                            &mut skip_r0,
                            true,
                            &mut carry,
                            &mut rem,
                            dec_options,
                        )
                    } else {
                        let mut cur_global = 0usize;
                        scan_window_par(
                            &in_blocks,
                            &names,
                            &targets,
                            eff_where,
                            skip_header_row,
                            true,
                            &mut carry,
                            &mut cur_global,
                            &mut rem,
                            dec_options,
                        )
                    }
                })?
            };

            let (inferred, _) = infer_chunk_col_types(&chunks, names.len());
            let fallback = if inferred.iter().any(Option::is_none) {
                Some(infer_fallback_types(
                    frame_bytes,
                    &names,
                    &cols,
                    skip_header_row,
                    dec_options,
                ))
            } else {
                None
            };
            let col_types: Vec<ArrowType> = inferred
                .into_iter()
                .enumerate()
                .map(|(j, opt)| opt.unwrap_or_else(|| fallback.as_ref().unwrap()[j]))
                .collect();

            let raw_batches: Vec<ArrowBatch> = chunks
                .into_iter()
                .map(|c| finalize_chunk_to_batch(c, &col_types))
                .collect();
            let coalesced = coalesce_arrow_batches(raw_batches, batch_size, &names, &col_types);
            Ok((coalesced, names, col_types))
        }
    }
}

/// Export a compressed VXG frame directly into a typed [`ArrowBatch`] with compressed-domain
/// column projection (`cols`) and `WHERE` pushdown (`where_expr`).
///
/// When `cols` is `None`, CSV/TSV headers or NDJSON top-level keys are discovered automatically.
/// Even when `0` rows match `where_expr`, column types are preserved from block-0 schema probing.
pub fn export_arrow_batch_where(
    frame_bytes: &[u8],
    cols: Option<&[ColTarget<'_>]>,
    where_expr: Option<&PredExpr<'_>>,
    limit: Option<usize>,
    dec_options: &DecodeOptions,
) -> Result<ArrowBatch> {
    let explicit = cols.map(|cs| {
        let mut names = Vec::with_capacity(cs.len());
        let mut owned = Vec::with_capacity(cs.len());
        for &c in cs {
            let (n, o) = target_to_owned_and_name(c);
            names.push(n);
            owned.push(o);
        }
        (names, owned)
    });
    let (batches, names, col_types) = execute_full_scan(
        frame_bytes,
        explicit,
        where_expr,
        usize::MAX,
        limit,
        dec_options,
    )?;
    Ok(concat_arrow_batches(batches, &names, &col_types))
}

/// Export a compressed VXG frame into coalesced [`ArrowBatch`]es of ~`batch_size` rows (default
/// [`DEFAULT_ARROW_BATCH_ROWS`] = `65_536` when `batch_size == 0`) with uniform column types.
pub fn export_arrow_batches_where(
    frame_bytes: &[u8],
    cols: Option<&[ColTarget<'_>]>,
    where_expr: Option<&PredExpr<'_>>,
    batch_size: usize,
    limit: Option<usize>,
    dec_options: &DecodeOptions,
) -> Result<Vec<ArrowBatch>> {
    let explicit = cols.map(|cs| {
        let mut names = Vec::with_capacity(cs.len());
        let mut owned = Vec::with_capacity(cs.len());
        for &c in cs {
            let (n, o) = target_to_owned_and_name(c);
            names.push(n);
            owned.push(o);
        }
        (names, owned)
    });
    let eff_bs = if batch_size == 0 { DEFAULT_ARROW_BATCH_ROWS } else { batch_size };
    let (batches, _, _) =
        execute_full_scan(frame_bytes, explicit, where_expr, eff_bs, limit, dec_options)?;
    Ok(batches)
}

/// Serialize a filtered & projected compressed-domain scan directly into an Apache Arrow IPC
/// Stream (`application/vnd.apache.arrow.stream`, MetadataVersion `V5`).
pub fn export_arrow_ipc_where(
    frame_bytes: &[u8],
    cols: Option<&[ColTarget<'_>]>,
    where_expr: Option<&PredExpr<'_>>,
    limit: Option<usize>,
    dec_options: &DecodeOptions,
) -> Result<Vec<u8>> {
    let batch = export_arrow_batch_where(frame_bytes, cols, where_expr, limit, dec_options)?;
    Ok(batch.to_ipc_stream())
}

pub(crate) fn try_scan_sql_query_arrow(
    frame_bytes: &[u8],
    query: &SqlQuery,
    dec_options: &DecodeOptions,
) -> Option<Result<ArrowBatch>> {
    if !query.group_by.is_empty() || !query.order_by.is_empty() {
        return None;
    }
    let explicit = match &query.projection {
        SelectProjection::Star => None,
        SelectProjection::Items(items) => {
            if items.is_empty() || items.iter().any(|it| it.expr.is_agg()) {
                return None;
            }
            let mut names = Vec::with_capacity(items.len());
            let mut owned = Vec::with_capacity(items.len());
            for it in items {
                let SelectExpr::Col(ref c) = it.expr else {
                    return None;
                };
                names.push(it.header_name());
                owned.push(c.clone());
            }
            Some((names, owned))
        }
    };
    Some(with_opt_sql_where(query.where_clause.as_ref(), |w| {
        let (batches, names, col_types) = execute_full_scan(
            frame_bytes,
            explicit,
            w,
            usize::MAX,
            query.limit,
            dec_options,
        )?;
        Ok(concat_arrow_batches(batches, &names, &col_types))
    }))
}

#[derive(Clone, Copy)]
struct SavedBlockLoc {
    f_offset: usize,
    f_len: usize,
    local_blk: usize,
    version: u16,
    checksummed: bool,
    dict_offset: usize,
    dict_len: usize,
    info: BlockInfo,
}

/// Windowed compressed-domain `ArrowArrayStream` cursor scanning `128` blocks (~8 MiB) per step.
pub(crate) struct ArrowWindowStream {
    pub(crate) schema_template: ArrowBatch,
    pending: VecDeque<ArrowBatch>,
    frame_ptr: *const u8,
    frame_len: usize,
    locs: Vec<SavedBlockLoc>,
    names: Vec<String>,
    col_types: Vec<ArrowType>,
    cols: Vec<OwnedCol>,
    where_clause: Option<ArrowWhere>,
    hdr_opt: Option<DelimHeader>,
    dec_options: DecodeOptions,
    batch_size: usize,
    next_block: usize,
    carry: Option<Vec<u8>>,
    skip_seq_r0: bool,
    cur_global: usize,
    rem_limit: usize,
}

unsafe impl Send for ArrowWindowStream {}

impl ArrowWindowStream {
    pub(crate) fn from_single_batch(batch: ArrowBatch) -> Self {
        let schema_cols = batch
            .columns
            .iter()
            .map(|c| ArrowCol {
                name: c.name.clone(),
                validity: None,
                null_count: 0,
                data: match &c.data {
                    ArrowData::Boolean(_) => ArrowData::Boolean(Vec::new()),
                    ArrowData::Int64(_) => ArrowData::Int64(Vec::new()),
                    ArrowData::Float64(_) => ArrowData::Float64(Vec::new()),
                    ArrowData::Utf8 { .. } => ArrowData::Utf8 { offsets: vec![0], values: Vec::new() },
                    ArrowData::LargeUtf8 { .. } => {
                        ArrowData::LargeUtf8 { offsets: vec![0], values: Vec::new() }
                    }
                },
            })
            .collect();
        let mut pending = VecDeque::new();
        if batch.num_rows > 0 {
            pending.push_back(batch);
        }
        Self {
            schema_template: ArrowBatch { num_rows: 0, columns: schema_cols },
            pending,
            frame_ptr: core::ptr::null(),
            frame_len: 0,
            locs: Vec::new(),
            names: Vec::new(),
            col_types: Vec::new(),
            cols: Vec::new(),
            where_clause: None,
            hdr_opt: None,
            dec_options: DecodeOptions::default(),
            batch_size: DEFAULT_ARROW_BATCH_ROWS,
            next_block: 0,
            carry: None,
            skip_seq_r0: false,
            cur_global: 0,
            rem_limit: 0,
        }
    }

    pub(crate) fn open(
        frame_bytes: &[u8],
        explicit: Option<(Vec<String>, Vec<OwnedCol>)>,
        where_clause: Option<ArrowWhere>,
        limit: Option<usize>,
        batch_size: usize,
        dec_options: DecodeOptions,
    ) -> Result<Self> {
        dec_options.validate()?;
        let eff_bs = if batch_size == 0 { DEFAULT_ARROW_BATCH_ROWS } else { batch_size };
        let lim = limit.unwrap_or(usize::MAX);

        let plan = match &where_clause {
            Some(aw) => aw.with_pred_expr(|w| resolve_plan(frame_bytes, explicit, Some(w), &dec_options))?,
            None => resolve_plan(frame_bytes, explicit, None, &dec_options)?,
        };

        let ResolvedPlan::Columns { names, cols, hdr_opt, skip_header_row } = plan else {
            let (batches, r_names, r_types) = match &where_clause {
                Some(aw) => aw.with_pred_expr(|w| {
                    raw_record_batches(frame_bytes, Some(w), eff_bs, lim, &dec_options)
                })?,
                None => raw_record_batches(frame_bytes, None, eff_bs, lim, &dec_options)?,
            };
            let mut st = Self::from_single_batch(empty_arrow_batch(&r_names, &r_types));
            st.pending = batches.into_iter().filter(|b| b.num_rows > 0).collect();
            return Ok(st);
        };

        let frames = Frame::parse_all(frame_bytes, dec_options.max_output)?;
        let base_addr = frame_bytes.as_ptr() as usize;
        let mut locs = Vec::new();
        for (f_bytes, parsed) in query::frame_slices(frame_bytes, &frames) {
            let f_offset = (f_bytes.as_ptr() as usize) - base_addr;
            let dict = parsed.dict();
            let dict_offset = if dict.is_empty() {
                0
            } else {
                (dict.as_ptr() as usize) - (f_bytes.as_ptr() as usize)
            };
            for (local_blk, info) in parsed.blocks().enumerate() {
                locs.push(SavedBlockLoc {
                    f_offset,
                    f_len: f_bytes.len(),
                    local_blk,
                    version: parsed.version(),
                    checksummed: parsed.checksummed(),
                    dict_offset,
                    dict_len: dict.len(),
                    info,
                });
            }
        }

        let mut cursor = Self {
            schema_template: ArrowBatch { num_rows: 0, columns: Vec::new() },
            pending: VecDeque::new(),
            frame_ptr: frame_bytes.as_ptr(),
            frame_len: frame_bytes.len(),
            locs,
            names,
            col_types: Vec::new(),
            cols,
            where_clause,
            hdr_opt,
            dec_options,
            batch_size: eff_bs,
            next_block: 0,
            carry: None,
            skip_seq_r0: skip_header_row,
            cur_global: 0,
            rem_limit: lim,
        };

        let w0_chunks = cursor.scan_next_window_chunks(skip_header_row)?;
        let (inferred, _) = infer_chunk_col_types(&w0_chunks, cursor.names.len());
        let fallback = if inferred.iter().any(Option::is_none) {
            Some(infer_fallback_types(
                frame_bytes,
                &cursor.names,
                &cursor.cols,
                skip_header_row,
                &cursor.dec_options,
            ))
        } else {
            None
        };
        cursor.col_types = inferred
            .into_iter()
            .enumerate()
            .map(|(j, opt)| opt.unwrap_or_else(|| fallback.as_ref().unwrap()[j]))
            .collect();
        cursor.schema_template = empty_arrow_batch(&cursor.names, &cursor.col_types);
        cursor.enqueue_chunks(w0_chunks);
        Ok(cursor)
    }

    fn enqueue_chunks(&mut self, chunks: Vec<Vec<ColBuilder>>) {
        if chunks.is_empty() {
            return;
        }
        let raw_batches: Vec<ArrowBatch> = chunks
            .into_iter()
            .map(|c| finalize_chunk_to_batch(c, &self.col_types))
            .collect();
        for b in coalesce_arrow_batches(raw_batches, self.batch_size, &self.names, &self.col_types) {
            if b.num_rows > 0 {
                self.pending.push_back(b);
            }
        }
    }

    fn scan_next_window_chunks(&mut self, skip_global_row0: bool) -> Result<Vec<Vec<ColBuilder>>> {
        let nb = self.locs.len();
        if self.next_block >= nb || self.rem_limit == 0 {
            self.next_block = nb;
            return Ok(Vec::new());
        }
        let w_start = self.next_block;
        let w_end = (w_start + STREAM_WINDOW_BLOCKS).min(nb);
        let is_last = w_end == nb;
        self.next_block = w_end;

        let frame_bytes = unsafe { core::slice::from_raw_parts(self.frame_ptr, self.frame_len) };
        let mut window = Vec::with_capacity(w_end - w_start);
        for loc in &self.locs[w_start..w_end] {
            let f_bytes = &frame_bytes[loc.f_offset..loc.f_offset + loc.f_len];
            let dict = if loc.dict_len == 0 {
                &[][..]
            } else {
                &f_bytes[loc.dict_offset..loc.dict_offset + loc.dict_len]
            };
            window.push(ParInBlock {
                f_bytes,
                local_blk: loc.local_blk,
                version: loc.version,
                frame_checksummed: loc.checksummed,
                dict,
                info: loc.info,
                payload: &f_bytes
                    [loc.info.payload_offset..loc.info.payload_offset + loc.info.payload_size],
            });
        }

        let targets: Vec<ColTarget<'_>> = self.cols.iter().map(OwnedCol::as_target).collect();
        let use_seq = self.dec_options.threads <= 1 || nb == 1;
        let run = |raw_w: Option<&PredExpr<'_>>,
                   hdr_opt: Option<&DelimHeader>,
                   names: &[String],
                   carry: &mut Option<Vec<u8>>,
                   skip_seq_r0: &mut bool,
                   cur_global: &mut usize,
                   rem_limit: &mut usize,
                   dec_options: &DecodeOptions|
         -> Result<Vec<Vec<ColBuilder>>> {
            let mapped = match (raw_w, hdr_opt) {
                (Some(w), Some(hdr)) => Some(MappedExpr::from_expr(w, hdr)),
                _ => None,
            };
            with_opt_mapped_expr(mapped.as_ref(), |mw| {
                let eff_w = if mapped.is_some() { mw } else { raw_w };
                if use_seq {
                    scan_window_seq(
                        &window,
                        names,
                        &targets,
                        eff_w,
                        skip_seq_r0,
                        is_last,
                        carry,
                        rem_limit,
                        dec_options,
                    )
                } else {
                    scan_window_par(
                        &window,
                        names,
                        &targets,
                        eff_w,
                        skip_global_row0,
                        is_last,
                        carry,
                        cur_global,
                        rem_limit,
                        dec_options,
                    )
                }
            })
        };

        match &self.where_clause {
            Some(aw) => aw.with_pred_expr(|w| {
                run(
                    Some(w),
                    self.hdr_opt.as_ref(),
                    &self.names,
                    &mut self.carry,
                    &mut self.skip_seq_r0,
                    &mut self.cur_global,
                    &mut self.rem_limit,
                    &self.dec_options,
                )
            }),
            None => run(
                None,
                self.hdr_opt.as_ref(),
                &self.names,
                &mut self.carry,
                &mut self.skip_seq_r0,
                &mut self.cur_global,
                &mut self.rem_limit,
                &self.dec_options,
            ),
        }
    }

    pub(crate) fn open_sql(
        frame_bytes: &[u8],
        query: Option<&SqlQuery>,
        dec_options: DecodeOptions,
    ) -> Result<Self> {
        let Some(q) = query else {
            return Self::open(frame_bytes, None, None, None, DEFAULT_ARROW_BATCH_ROWS, dec_options);
        };
        if q.group_by.is_empty() && q.order_by.is_empty() {
            match &q.projection {
                SelectProjection::Star => {
                    return Self::open(
                        frame_bytes,
                        None,
                        q.where_clause.clone().map(ArrowWhere::from_sql_where),
                        q.limit,
                        DEFAULT_ARROW_BATCH_ROWS,
                        dec_options,
                    );
                }
                SelectProjection::Items(items)
                    if !items.is_empty() && !items.iter().any(|it| it.expr.is_agg()) =>
                {
                    let mut names = Vec::with_capacity(items.len());
                    let mut owned = Vec::with_capacity(items.len());
                    let mut all_cols = true;
                    for it in items {
                        if let SelectExpr::Col(ref c) = it.expr {
                            names.push(it.header_name());
                            owned.push(c.clone());
                        } else {
                            all_cols = false;
                            break;
                        }
                    }
                    if all_cols {
                        return Self::open(
                            frame_bytes,
                            Some((names, owned)),
                            q.where_clause.clone().map(ArrowWhere::from_sql_where),
                            q.limit,
                            DEFAULT_ARROW_BATCH_ROWS,
                            dec_options,
                        );
                    }
                }
                _ => {}
            }
        }
        let batch = crate::arrow::export_arrow_batch(frame_bytes, Some(q), &dec_options)?;
        Ok(Self::from_single_batch(batch))
    }

    pub(crate) fn next_batch(&mut self) -> Result<Option<ArrowBatch>> {
        while self.pending.is_empty() && self.next_block < self.locs.len() && self.rem_limit > 0 {
            let chunks = self.scan_next_window_chunks(false)?;
            self.enqueue_chunks(chunks);
        }
        Ok(self.pending.pop_front())
    }
}
