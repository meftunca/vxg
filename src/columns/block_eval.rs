fn unpack_block(bytes: &[u8]) -> Result<Vec<u8>> {
    unpack_block_mode(bytes, true)
}

fn unpack_block_mode(bytes: &[u8], strict: bool) -> Result<Vec<u8>> {
    let mut cursor = Cursor::new(bytes);
    let selector = cursor.byte()?;
    match selector {
        RAW_PACKET => {
            let len = cursor.uleb()?;
            if len > MAX_BLOCK_SIZE {
                return Err(invalid("raw column block length"));
            }
            let out = cursor.take(len)?.to_vec();
            if !cursor.done() {
                return Err(invalid("raw column block trailing bytes"));
            }
            Ok(out)
        }
        GRAMMAR_PACKET => unpack_grammar(&mut cursor, strict),
        STRIDE_PACKET => {
            let mut probe = Cursor::new(cursor.bytes);
            probe.pos = cursor.pos;
            let raw_len = probe.uleb()?;
            if raw_len > MAX_BLOCK_SIZE {
                return Err(invalid("stride raw length"));
            }
            let mut out = vec![0u8; raw_len];
            unpack_stride_into(&mut cursor, &mut out)?;
            Ok(out)
        }
        FAST_LZ_PACKET => {
            let mut probe = Cursor::new(cursor.bytes);
            probe.pos = cursor.pos;
            let raw_len = probe.uleb()?;
            if raw_len > MAX_BLOCK_SIZE {
                return Err(invalid("fast lz raw length"));
            }
            let mut out = vec![0u8; raw_len];
            unpack_fast_lz_into(&mut cursor, &mut out)?;
            Ok(out)
        }
        _ => Err(invalid("column packet selector")),
    }
}

/// Encode one independent block as a canonical structured packet.
///
/// Returns `None` when the grammar packet is not strictly smaller than the raw
/// packet; the caller then keeps its LZ/RAW payload. Every selector, dictionary,
/// table, escape and padding bit is already charged inside the returned packet.
pub fn encode_grammar_block(data: &[u8]) -> Result<Option<Vec<u8>>> {
    if data.is_empty() {
        return Ok(None);
    }
    if data.len() > MAX_BLOCK_SIZE {
        return Err(Error::InvalidOption("column block exceeds maximum"));
    }
    let raw_len = 1 + uleb_len(data.len()) + data.len();
    let mut out = Vec::with_capacity(4096);
    if pack_grammar_into(data, &mut out)? && out.len() < raw_len {
        return Ok(Some(out));
    }
    out.clear();
    if pack_stride_into(data, &mut out).is_some() && out.len() < raw_len {
        if out.len() * 4 > data.len() {
            let mut lz = Vec::with_capacity(out.len());
            if pack_fast_lz_into(data, &mut lz).is_some() && lz.len() < out.len() {
                return Ok(Some(lz));
            }
        }
        return Ok(Some(out));
    }
    out.clear();
    if pack_fast_lz_into(data, &mut out).is_some() && out.len() < raw_len {
        return Ok(Some(out));
    }
    Ok(None)
}

/// Encode one independent block directly into `out` if it compresses smaller than raw.
pub(crate) fn encode_grammar_block_into(
    data: &[u8],
    out: &mut Vec<u8>,
) -> Result<Option<(usize, usize)>> {
    if data.is_empty() {
        return Ok(None);
    }
    if data.len() > MAX_BLOCK_SIZE {
        return Err(Error::InvalidOption("column block exceeds maximum"));
    }
    // A COLUMN block must be strictly smaller than the raw block: the frame directory rejects
    // `cn >= rn` for COLUMN/LZ kinds (those blocks must be stored RAW instead).
    let raw_len = data.len();
    let start = out.len();
    match pack_grammar_into(data, out) {
        Ok(true) => {
            let len = out.len() - start;
            if len < raw_len && (len * 4 < data.len() * 3 || data.len() < 512) {
                return Ok(Some((start, len)));
            }
            out.truncate(start);
        }
        Ok(false) => {
            out.truncate(start);
        }
        Err(err) => {
            out.truncate(start);
            return Err(err);
        }
    }
    if pack_stride_into(data, out).is_some() {
        let s_len = out.len() - start;
        if s_len < raw_len {
            if s_len * 4 > data.len() {
                let mut lz = Vec::with_capacity(s_len);
                if pack_fast_lz_into(data, &mut lz).is_some() && lz.len() < s_len {
                    out.truncate(start);
                    out.extend_from_slice(&lz);
                    return Ok(Some((start, lz.len())));
                }
            }
            return Ok(Some((start, s_len)));
        }
        out.truncate(start);
    }
    if pack_fast_lz_into(data, out).is_some() {
        let lz_len = out.len() - start;
        if lz_len < raw_len {
            return Ok(Some((start, lz_len)));
        }
        out.truncate(start);
    }
    Ok(None)
}

/// Decode and structurally validate one structured packet back to its exact
/// bytes. This is the production path: every structural rule, bound, count and
/// ordering is checked, but representation minimality (full re-encode equality)
/// is not recomputed; frame corruption is covered by the optional CRC32C.
pub fn decode_block(bytes: &[u8]) -> Result<Vec<u8>> {
    unpack_block_mode(bytes, false)
}

/// Decode one structured packet directly into the caller's exact-size buffer.
///
/// This is the production path used by the frame decoder: the row assembly
/// writes straight into `out`, so a structured block costs no intermediate
/// allocation and no second copy. `out` must be exactly the block raw size and
/// the same structural validation as [`decode_block`] applies.
pub fn decode_block_into(bytes: &[u8], out: &mut [u8]) -> Result<()> {
    let mut cursor = Cursor::new(bytes);
    let selector = cursor.byte()?;
    match selector {
        RAW_PACKET => {
            let len = cursor.uleb()?;
            if len > MAX_BLOCK_SIZE || len != out.len() {
                return Err(invalid("raw column block length"));
            }
            let data = cursor.take(len)?;
            out.copy_from_slice(data);
            if !cursor.done() {
                return Err(invalid("raw column block trailing bytes"));
            }
            Ok(())
        }
        GRAMMAR_PACKET => {
            let sink = OutSink::new(out.as_mut_ptr(), out.len());
            let pos = unpack_grammar_inner(&mut cursor, false, sink)?;
            if pos != out.len() {
                return Err(invalid("column output length"));
            }
            Ok(())
        }
        STRIDE_PACKET => unpack_stride_into(&mut cursor, out),
        FAST_LZ_PACKET => unpack_fast_lz_into(&mut cursor, out),
        _ => Err(invalid("column packet selector")),
    }
}

/// Strict oracle decode: additionally re-encodes every model and rejects a
/// payload that is not the exact charged minimum. Used by the research
/// evaluator and the canonicality tests.
pub fn decode_block_strict(bytes: &[u8]) -> Result<Vec<u8>> {
    unpack_block(bytes)
}

fn pack_block(data: &[u8]) -> Result<PackedBlock> {
    if data.len() > MAX_BLOCK_SIZE {
        return Err(Error::InvalidOption("column block exceeds maximum"));
    }
    let raw_len = 1 + uleb_len(data.len()) + data.len();
    let grammar = pack_grammar(data)?;
    if grammar.bytes.len() < raw_len {
        Ok(PackedBlock {
            bytes: grammar.bytes,
            stats: grammar.stats,
        })
    } else {
        Ok(PackedBlock {
            bytes: raw_packet(data),
            stats: BlockStats::default(),
        })
    }
}

fn check_block_size(block_size: usize) -> Result<()> {
    if !(4096..=MAX_BLOCK_SIZE).contains(&block_size) || block_size % 64 != 0 {
        return Err(Error::InvalidOption(
            "column block size must be 4096..1048576 and a multiple of 64",
        ));
    }
    Ok(())
}

/// Build each independent packet once and prove its exact reconstruction before
/// returning its complete directory-plus-payload accounting.
pub fn evaluate(input: &[u8], block_size: usize) -> Result<Evaluation> {
    check_block_size(block_size)?;
    let mut evaluation = Evaluation {
        input_bytes: input.len(),
        block_size,
        blocks: 0,
        grammar_blocks: 0,
        raw_blocks: 0,
        payload_bytes: 0,
        frame_directory_bytes: 32,
        template_count: 0,
        template_rows: 0,
        raw_rows: 0,
        model_counts: [0; 14],
        model_bytes: [0; 14],
        columns: Vec::new(),
        pack_ns: None,
        unpack_ns: None,
    };
    for (block, raw) in input.chunks(block_size).enumerate() {
        let packed = pack_block(raw)?;
        if unpack_block(&packed.bytes)? != raw {
            return Err(invalid("column evaluator roundtrip"));
        }
        evaluation.blocks += 1;
        evaluation.frame_directory_bytes = evaluation
            .frame_directory_bytes
            .checked_add(32)
            .ok_or(Error::SizeOverflow)?;
        evaluation.payload_bytes = evaluation
            .payload_bytes
            .checked_add(packed.bytes.len())
            .ok_or(Error::SizeOverflow)?;
        if packed.stats.grammar {
            evaluation.grammar_blocks += 1;
        } else {
            evaluation.raw_blocks += 1;
        }
        evaluation.template_count += packed.stats.templates;
        evaluation.template_rows += packed.stats.template_rows;
        evaluation.raw_rows += packed.stats.raw_rows;
        for column in packed.stats.columns {
            evaluation.model_counts[column.kind.index()] += 1;
            evaluation.model_bytes[column.kind.index()] = evaluation.model_bytes[column.kind.index()]
                .checked_add(column.payload_bytes)
                .ok_or(Error::SizeOverflow)?;
            evaluation.columns.push(ColumnReport {
                block,
                template: column.template,
                slot: column.slot,
                model: column.kind.name(),
                values: column.values,
                payload_bytes: column.payload_bytes,
            });
        }
    }
    Ok(evaluation)
}

/// Time complete packet construction and exact packet reconstruction separately.
/// Correctness is checked outside the timed loops so comparison scans are not
/// accidentally attributed to either scalar oracle rate.
pub fn time(input: &[u8], block_size: usize, iterations: usize) -> Result<Timing> {
    check_block_size(block_size)?;
    if iterations == 0 {
        return Err(Error::InvalidOption("column timing iterations must be positive"));
    }
    let chunks = input.chunks(block_size).collect::<Vec<_>>();
    let packets = chunks
        .iter()
        .map(|chunk| pack_block(chunk))
        .collect::<Result<Vec<_>>>()?;
    for (packet, raw) in packets.iter().zip(&chunks) {
        if unpack_block(&packet.bytes)? != *raw {
            return Err(invalid("timed column evaluator roundtrip"));
        }
    }
    let pack_start = Instant::now();
    let mut checksum = 0usize;
    for _ in 0..iterations {
        for raw in &chunks {
            checksum ^= pack_block(raw)?.bytes.len();
        }
    }
    black_box(checksum);
    let pack_ns = pack_start.elapsed().as_nanos();
    let unpack_start = Instant::now();
    let mut checksum = 0usize;
    for _ in 0..iterations {
        for packet in &packets {
            checksum ^= unpack_block(&packet.bytes)?.len();
        }
    }
    black_box(checksum);
    Ok(Timing {
        pack_ns,
        unpack_ns: unpack_start.elapsed().as_nanos(),
    })
}

impl Evaluation {
    pub fn apply_timing(&mut self, timing: &Timing) {
        self.pack_ns = Some(timing.pack_ns);
        self.unpack_ns = Some(timing.unpack_ns);
    }

    pub fn to_json(&self, iterations: usize) -> String {
        let total = self
            .frame_directory_bytes
            .checked_add(self.payload_bytes)
            .unwrap();
        let pack_rate = self
            .pack_ns
            .filter(|&ns| ns != 0)
            .map(|ns| self.input_bytes as f64 * 1_000_000_000.0 / ns as f64);
        let unpack_rate = self
            .unpack_ns
            .filter(|&ns| ns != 0)
            .map(|ns| self.input_bytes as f64 * 1_000_000_000.0 / ns as f64);
        let mut out = String::new();
        writeln!(out, "{{").unwrap();
        writeln!(out, "  \"schema\": \"vxg-p5d-structured-columns-r1\",").unwrap();
        writeln!(out, "  \"scope\": \"Complete independent block-local scalar research packet. It serializes templates, row selectors, model metadata and all values, proves exact decode, and charges a 32-byte header plus 32-byte directory entries. It is not a VXG v3 wire format or a codec GB/s benchmark.\",").unwrap();
        writeln!(out, "  \"selection_rule\": \"For each raw block, build RAW and grammar packets and select grammar only when it is strictly shorter. For every grammar column, choose the shortest of raw, canonical byte dictionary, canonical decimal affine, canonical decimal delta, quoted fixed-width lowercase-hex suffix, quoted hex epoch-counter, quoted semicolon segment dictionary, quoted same-year ISO-8601 and the retained periodic/GF(2) models; ties retain raw/earlier model order. No model reads a field name, source path, seed or external state.\",").unwrap();
        writeln!(out, "  \"input_bytes\": {},", self.input_bytes).unwrap();
        writeln!(out, "  \"block_size\": {},", self.block_size).unwrap();
        writeln!(out, "  \"blocks\": {},", self.blocks).unwrap();
        writeln!(out, "  \"grammar_blocks\": {},", self.grammar_blocks).unwrap();
        writeln!(out, "  \"raw_blocks\": {},", self.raw_blocks).unwrap();
        writeln!(out, "  \"payload_bytes\": {},", self.payload_bytes).unwrap();
        writeln!(out, "  \"frame_directory_bytes\": {},", self.frame_directory_bytes).unwrap();
        writeln!(out, "  \"prototype_total_frame_bytes\": {},", total).unwrap();
        writeln!(out, "  \"prototype_total_ratio\": {:.9},", total as f64 / self.input_bytes.max(1) as f64).unwrap();
        writeln!(out, "  \"template_count\": {},", self.template_count).unwrap();
        writeln!(out, "  \"template_rows\": {},", self.template_rows).unwrap();
        writeln!(out, "  \"raw_rows\": {},", self.raw_rows).unwrap();
        writeln!(out, "  \"timing_iterations\": {},", iterations).unwrap();
        writeln!(out, "  \"pack_elapsed_ns\": {},", self.pack_ns.map_or("null".to_owned(), |ns| ns.to_string())).unwrap();
        writeln!(out, "  \"unpack_elapsed_ns\": {},", self.unpack_ns.map_or("null".to_owned(), |ns| ns.to_string())).unwrap();
        writeln!(out, "  \"pack_input_bytes_per_second\": {},", pack_rate.map_or("null".to_owned(), |value| format!("{value:.3}"))).unwrap();
        writeln!(out, "  \"unpack_input_bytes_per_second\": {},", unpack_rate.map_or("null".to_owned(), |value| format!("{value:.3}"))).unwrap();
        writeln!(out, "  \"model_totals\": [").unwrap();
        let model_kinds = [
            ModelKind::Raw,
            ModelKind::Dictionary,
            ModelKind::DecimalAffine,
            ModelKind::HexSuffix,
            ModelKind::Timestamp,
            ModelKind::DecimalPacked,
            ModelKind::DecimalSuffix,
            ModelKind::PeriodicString,
            ModelKind::PeriodicHex,
            ModelKind::Gf2Affine,
            ModelKind::EpochCounter,
            ModelKind::SegmentDict,
            ModelKind::DecimalDelta,
        ];
        for (index, kind) in model_kinds.into_iter().enumerate() {
            writeln!(out, "    {{\"model\":\"{}\",\"columns\":{},\"payload_bytes\":{}}}{}", kind.name(), self.model_counts[kind.index()], self.model_bytes[kind.index()], if index + 1 == model_kinds.len() { "" } else { "," }).unwrap();
        }
        writeln!(out, "  ],").unwrap();
        writeln!(out, "  \"column_decomposition\": [").unwrap();
        for (index, column) in self.columns.iter().enumerate() {
            writeln!(out, "    {{\"block\":{},\"template\":{},\"slot\":{},\"model\":\"{}\",\"values\":{},\"payload_bytes\":{}}}{}", column.block, column.template, column.slot, column.model, column.values, column.payload_bytes, if index + 1 == self.columns.len() { "" } else { "," }).unwrap();
        }
        writeln!(out, "  ]").unwrap();
        writeln!(out, "}}").unwrap();
        out
    }
}

