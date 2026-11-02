fn pack_u64_column_direct(vals: &[u64], sum_digits: usize) -> Option<Vec<u8>> {
    let n = vals.len();
    if n < 2 {
        return None;
    }
    let raw_limit = 1 + n + sum_digits;
    let first = vals[0];
    let second = vals[1];
    let mut min = first.min(second);
    let mut max = first.max(second);
    let step128 = second as i128 - first as i128;
    let mut step64 = 0i64;
    let mut affine_ok = true;
    let mut delta_ok = true;
    let mut max_zigzag = 0u64;
    if let Ok(s64) = i64::try_from(step128) {
        step64 = s64;
        max_zigzag = ((s64 << 1) ^ (s64 >> 63)) as u64;
    } else {
        affine_ok = false;
        delta_ok = false;
    }
    let mut hi_bits = first | second;
    for &v in &vals[2..] {
        min = min.min(v);
        max = max.max(v);
        hi_bits |= v;
    }
    if min == max {
        // Constant column: emit 1-entry `MODEL_DICTIONARY` (`width = 0`) if smaller than raw.
        let (buf, len) = format_inline_decimal(min);
        let tok = &buf[..len as usize];
        let plain_bytes = uleb_len(tok.len()) + tok.len();
        return build_front_dict_packet(1, n, 0, 0, plain_bytes, &[tok], &[], raw_limit, false);
    }
    if (hi_bits >> 63) == 0 {
        let mut prev = second;
        for &v in &vals[2..] {
            let d = (v as i64).wrapping_sub(prev as i64);
            if d != step64 {
                affine_ok = false;
                let zz = ((d << 1) ^ (d >> 63)) as u64;
                max_zigzag = max_zigzag.max(zz);
            }
            prev = v;
        }
    } else {
        let mut prev = second;
        for &v in &vals[2..] {
            let diff128 = v as i128 - prev as i128;
            if diff128 != step128 {
                affine_ok = false;
            }
            if delta_ok {
                if let Ok(d) = i64::try_from(diff128) {
                    let zz = ((d << 1) ^ (d >> 63)) as u64;
                    max_zigzag = max_zigzag.max(zz);
                } else {
                    delta_ok = false;
                }
            }
            prev = v;
        }
    }

    let mut best_bytes: Option<Vec<u8>> = None;
    let mut best_len = raw_limit;
    let mut best_kind = ModelKind::Raw;

    if affine_ok && 17 < best_len {
        let mut out = take_model_buf(17);
        out.push(MODEL_DECIMAL_AFFINE);
        out.extend_from_slice(&first.to_le_bytes());
        out.extend_from_slice(&step64.to_le_bytes());
        best_len = 17;
        best_kind = ModelKind::DecimalAffine;
        best_bytes = Some(out);
    }

    if !(affine_ok && step128 != 0 && n >= 49 && best_kind == ModelKind::DecimalAffine) {
        let packed_width = bit_width(max - min);
        if let Ok(packed_bits) = bit_bytes(n, packed_width) {
            let packed_size = 10 + packed_bits;
            let delta_width = bit_width(max_zigzag);
            let delta_size = if delta_ok {
                bit_bytes(n - 1, delta_width).ok().map(|b| 10 + b)
            } else {
                None
            };
            if delta_size.is_some_and(|ds| ds < packed_size) {
                let ds = delta_size.unwrap();
                let lim = if best_bytes.is_some() {
                    effective_limit(best_len, best_kind, ModelKind::DecimalDelta)
                } else {
                    raw_limit
                };
                if ds < lim {
                    let mut out = take_model_buf(ds);
                    out.push(MODEL_DECIMAL_DELTA);
                    out.extend_from_slice(&first.to_le_bytes());
                    out.push(delta_width);
                    push_bit_values(
                        &mut out,
                        n - 1,
                        delta_width,
                        vals.windows(2).map(|pair| {
                            let d = (pair[1] as i128 - pair[0] as i128) as i64;
                            ((d << 1) ^ (d >> 63)) as u64
                        }),
                    );
                    if let Some(old) = best_bytes.replace(out) {
                        put_model_buf(old);
                    }
                }
            } else {
                let lim = if best_bytes.is_some() {
                    effective_limit(best_len, best_kind, ModelKind::DecimalPacked)
                } else {
                    raw_limit
                };
                if packed_size < lim {
                    let mut out = take_model_buf(packed_size);
                    out.push(MODEL_DECIMAL_PACKED);
                    out.extend_from_slice(&min.to_le_bytes());
                    out.push(packed_width);
                    push_bit_values(&mut out, n, packed_width, vals.iter().map(|&v| v - min));
                    if let Some(old) = best_bytes.replace(out) {
                        put_model_buf(old);
                    }
                }
            }
        }
    }
    best_bytes
}

/// Direct binary `u64` rewrite for canonical unquoted decimal columns (`DecimalSmall`,
/// `DecimalValues`, `DecimalAffine*`, `DecimalPacked`) with zero ASCII formatting or parsing.
fn try_rewrite_numeric_slot<'a>(
    col: &PreparedColumn<'a>,
    ones: usize,
    hits_opt: Option<&[usize]>,
    target_col: crate::ops::ColTarget<'_>,
    op: &crate::ops::ColOp<'_>,
    scratch: &mut DecodeScratch,
) -> Result<Option<SlotColRewrite>> {
    if ones < 2
        || !matches!(col.const_full, ConstValue::None)
        || !matches!(col.const_lead, LeadValue::Empty)
        || !col.const_trail.is_empty()
    {
        return Ok(None);
    }
    let is_pure_decimal = matches!(
        col.stream,
        ColumnStream::DecimalValues { .. }
            | ColumnStream::DecimalSmall { .. }
            | ColumnStream::DecimalAffineSmall { .. }
            | ColumnStream::DecimalAffine { .. }
            | ColumnStream::DecimalPacked { .. }
    );
    if !is_pure_decimal {
        return Ok(None);
    }
    if matches!(target_col, crate::ops::ColTarget::JsonKey(_))
        && matches!(op, crate::ops::ColOp::MapStr(_))
    {
        // Unquoted integer JSON tokens never match `MapStr` (`unescape_json_str` is `None`).
        return Ok(Some(SlotColRewrite::Untouched));
    }
    if let crate::ops::ColTarget::Delimited { delim, .. } = target_col {
        if delim.is_ascii_digit() {
            return Ok(None);
        }
    }

    enum NumOp<'b> {
        Fn(&'b (dyn Fn(u64) -> u64 + Sync)),
        Const(u64),
    }
    let num_op = match op {
        crate::ops::ColOp::MapU64(f) => NumOp::Fn(*f),
        crate::ops::ColOp::SetRaw(raw) => match crate::ops::parse_canonical_u64(raw) {
            Some(c) => NumOp::Const(c),
            None => return Ok(None),
        },
        _ => return Ok(None),
    };

    let u64_start = match col.stream {
        ColumnStream::DecimalValues { u64_offset } => u64_offset,
        ColumnStream::DecimalSmall { u16_offset } => {
            let off = scratch.u64_arena.len();
            scratch.u64_arena.reserve(ones);
            for i in 0..ones {
                scratch.u64_arena.push(u64::from(scratch.u16_arena[u16_offset + i]));
            }
            off
        }
        ColumnStream::DecimalAffineSmall { current, step }
        | ColumnStream::DecimalAffine { current, step } => {
            let off = scratch.u64_arena.len();
            scratch.u64_arena.reserve(ones);
            for i in 0..ones {
                scratch.u64_arena.push(current.wrapping_add(step.wrapping_mul(i as u64)));
            }
            off
        }
        ColumnStream::DecimalPacked { min, mut reader } => {
            let off = scratch.u64_arena.len();
            scratch.u64_arena.reserve(ones);
            for _ in 0..ones {
                scratch.u64_arena.push(min.wrapping_add(reader.next_u64()));
            }
            off
        }
        _ => return Ok(None),
    };

    let vals = &mut scratch.u64_arena[u64_start..u64_start + ones];
    let mut any_changed = false;
    let mut byte_delta = 0isize;
    let mut sum_digits = 0usize;

    if let Some(hits) = hits_opt {
        for &h in hits {
            let old_v = vals[h];
            let new_v = match num_op {
                NumOp::Fn(f) => f(old_v),
                NumOp::Const(c) => c,
            };
            if new_v != old_v {
                vals[h] = new_v;
                any_changed = true;
                byte_delta +=
                    decimal_digits_u64(new_v) as isize - decimal_digits_u64(old_v) as isize;
            }
        }
        if !any_changed {
            return Ok(Some(SlotColRewrite::Untouched));
        }
        for &v in vals.iter() {
            sum_digits += decimal_digits_u64(v);
        }
    } else {
        for slot in vals.iter_mut() {
            let old_v = *slot;
            let new_v = match num_op {
                NumOp::Fn(f) => f(old_v),
                NumOp::Const(c) => c,
            };
            let new_d = decimal_digits_u64(new_v);
            sum_digits += new_d;
            if new_v != old_v {
                *slot = new_v;
                any_changed = true;
                byte_delta += new_d as isize - decimal_digits_u64(old_v) as isize;
            }
        }
        if !any_changed {
            return Ok(Some(SlotColRewrite::Untouched));
        }
    }

    if let Some(model_bytes) = pack_u64_column_direct(vals, sum_digits) {
        return Ok(Some(SlotColRewrite::Rewritten {
            model_bytes,
            byte_delta,
        }));
    }
    Ok(None)
}

/// Direct dictionary rewrite for BOTH `map_column` (`hits_opt == None`) and `update_column`
/// (`hits_opt == Some(hits)`), remapping `u8` index arrays and emitting `MODEL_DICTIONARY` /
/// `MODEL_FRONT_DICT` directly without formatting or hashing `ones` rows.
fn try_rewrite_dict_slot<'a>(
    col: &PreparedColumn<'a>,
    ones: usize,
    hits_opt: Option<&[usize]>,
    target_col: crate::ops::ColTarget<'_>,
    op: &crate::ops::ColOp<'_>,
    scratch: &DecodeScratch,
) -> Result<Option<SlotColRewrite>> {
    let ColumnStream::Dictionary { u8_offset, .. } = col.stream else {
        return Ok(None);
    };
    let Some((slice_off, entries)) = col.dict_slices else {
        return Ok(None);
    };
    if entries == 0 || entries > 256 || ones == 0 {
        return Ok(None);
    }
    let ids = &scratch.u8_arena[u8_offset..u8_offset + ones];

    if let Some(hits) = hits_opt {
        if entries > 128 {
            return Ok(None);
        }
        let mut orig_entries: Vec<&[u8]> = Vec::with_capacity(entries);
        for e in 0..entries {
            let (ptr, len) = scratch.slice_arena[slice_off + e];
            orig_entries.push(unsafe { std::slice::from_raw_parts(ptr, len) });
        }
        // Lazy per-entry transform cache (`0..entries`), creating new temporary IDs `entries + k`:
        let mut cache: Vec<Option<Option<(u8, isize)>>> = vec![None; entries];
        let mut extra_owned: Vec<Vec<u8>> = Vec::new();
        let mut work_ids = ids.to_vec();
        let mut any_changed = false;
        let mut byte_delta = 0isize;

        for &h in hits {
            let orig_id = work_ids[h] as usize;
            if orig_id >= entries {
                continue;
            }
            let cached = match cache[orig_id] {
                Some(res) => res,
                None => {
                    let eb = orig_entries[orig_id];
                    let res = if let Some(nt) = crate::ops::apply_token_op(target_col, eb, op) {
                        if !is_valid_slot_token(target_col, &nt) {
                            return Ok(Some(SlotColRewrite::Fallback));
                        }
                        let new_id_usize = entries + extra_owned.len();
                        if new_id_usize >= 256 {
                            return Ok(None);
                        }
                        let d_len = nt.len() as isize - eb.len() as isize;
                        extra_owned.push(nt);
                        Some((new_id_usize as u8, d_len))
                    } else {
                        None
                    };
                    cache[orig_id] = Some(res);
                    res
                }
            };
            if let Some((new_id, d_len)) = cached {
                work_ids[h] = new_id;
                byte_delta += d_len;
                any_changed = true;
            }
        }
        if !any_changed {
            return Ok(Some(SlotColRewrite::Untouched));
        }
        let total_slots = entries + extra_owned.len();
        let mut all_refs: Vec<&[u8]> = Vec::with_capacity(total_slots);
        all_refs.extend_from_slice(&orig_entries);
        for v in &extra_owned {
            all_refs.push(v.as_slice());
        }
        return pack_remapped_dict(&all_refs, &mut work_ids, byte_delta);
    }

    // `hits_opt == None` (`map_column`): transform each of the `entries` dictionary strings once.
    let mut orig_entries: Vec<&[u8]> = Vec::with_capacity(entries);
    let mut mapped: Vec<Option<Vec<u8>>> = Vec::with_capacity(entries);
    let mut any_changed = false;
    for e in 0..entries {
        let (ptr, len) = scratch.slice_arena[slice_off + e];
        let eb = unsafe { std::slice::from_raw_parts(ptr, len) };
        orig_entries.push(eb);
        let m = crate::ops::apply_token_op(target_col, eb, op);
        if let Some(ref nt) = m {
            if !is_valid_slot_token(target_col, nt) {
                return Ok(Some(SlotColRewrite::Fallback));
            }
            any_changed = true;
        }
        mapped.push(m);
    }
    if !any_changed {
        return Ok(Some(SlotColRewrite::Untouched));
    }
    let mut freq = [0u32; 256];
    for &id in ids {
        freq[id as usize] += 1;
    }
    let mut byte_delta = 0isize;
    let mut final_refs: Vec<&[u8]> = Vec::with_capacity(entries);
    for e in 0..entries {
        if let Some(ref nt) = mapped[e] {
            let d = nt.len() as isize - orig_entries[e].len() as isize;
            byte_delta += d * (freq[e] as isize);
            final_refs.push(nt.as_slice());
        } else {
            final_refs.push(orig_entries[e]);
        }
    }
    if byte_delta == 0 && final_refs == orig_entries {
        return Ok(Some(SlotColRewrite::Untouched));
    }
    let mut work_ids = ids.to_vec();
    pack_remapped_dict(&final_refs, &mut work_ids, byte_delta)
}

fn pack_remapped_dict(
    cand_refs: &[&[u8]],
    work_ids: &mut [u8],
    byte_delta: isize,
) -> Result<Option<SlotColRewrite>> {
    let ones = work_ids.len();
    let mut freq = [0u32; 256];
    for &id in work_ids.iter() {
        freq[id as usize] += 1;
    }
    let mut active_order: Vec<(usize, &[u8])> = Vec::with_capacity(cand_refs.len());
    let mut raw_limit = 1usize;
    let mut all_decimal = true;
    for (i, &s) in cand_refs.iter().enumerate() {
        let f = freq[i] as usize;
        if f > 0 {
            raw_limit += f * (uleb_len(s.len()) + s.len());
            if all_decimal && parse_decimal(s).is_none() {
                all_decimal = false;
            }
            active_order.push((i, s));
        }
    }
    if all_decimal {
        // Let `pack_model_v3` also consider `DecimalPacked` / `DecimalAffine` if entries became decimals.
        let refs: Vec<&[u8]> = work_ids.iter().map(|&id| cand_refs[id as usize]).collect();
        let (model_bytes, _) = pack_model_v3(&refs);
        return Ok(Some(SlotColRewrite::Rewritten {
            model_bytes,
            byte_delta,
        }));
    }
    active_order.sort_unstable_by(|a, b| a.1.cmp(b.1));
    let mut unique: Vec<&[u8]> = Vec::with_capacity(active_order.len());
    let mut remap = [0u8; 256];
    for (orig_id, s) in active_order {
        let rank = if unique.last().copied() == Some(s) {
            unique.len() - 1
        } else {
            unique.push(s);
            unique.len() - 1
        };
        remap[orig_id] = rank as u8;
    }
    let u = unique.len();
    for id in work_ids.iter_mut() {
        *id = remap[*id as usize];
    }
    let width = bit_width(u.saturating_sub(1) as u64);
    let packed_len = bit_bytes(ones, width)?;
    let plain_dict_bytes: usize = unique.iter().map(|e| uleb_len(e.len()) + e.len()).sum();
    let allow_front = u >= 2;
    if let Some(model_bytes) = build_front_dict_packet(
        u,
        ones,
        width,
        packed_len,
        plain_dict_bytes,
        &unique,
        work_ids,
        raw_limit,
        allow_front,
    ) {
        return Ok(Some(SlotColRewrite::Rewritten {
            model_bytes,
            byte_delta,
        }));
    }
    let refs: Vec<&[u8]> = work_ids.iter().map(|&r| unique[r as usize]).collect();
    let (model_bytes, _) = pack_model_v3(&refs);
    Ok(Some(SlotColRewrite::Rewritten {
        model_bytes,
        byte_delta,
    }))
}
