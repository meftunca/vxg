#[derive(Clone, Copy)]
struct UnboundExecColumn<'a> {
    stream: ColumnStream<'a>,
    after_range: (usize, usize),
}

#[derive(Clone, Copy, Default)]
struct UnboundTemplate {
    head_range: (usize, usize),
    col_start: usize,
    col_end: usize,
    row_budget: usize,
}

#[derive(Clone, Copy)]
struct PreparedTemplate {
    head: FastPiece,
    col_ptr: *mut ExecColumn<'static>,
    col_len: usize,
    row_budget: usize,
}

impl PreparedTemplate {
    #[inline(always)]
    const fn empty() -> Self {
        Self {
            head: FastPiece::empty(),
            col_ptr: std::ptr::null_mut(),
            col_len: 0,
            row_budget: 0,
        }
    }
}

impl UnboundTemplate {
    #[inline(never)]
    fn build<'a>(
        segments: &[&'a [u8]],
        cols: &[PreparedColumn<'a>],
        scratch: &mut DecodeScratch,
    ) -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            #[target_feature(enable = "avx2,ssse3,sse4.1,popcnt,bmi1,bmi2")]
            unsafe fn build_x86_v3<'a>(
                segments: &[&'a [u8]],
                cols: &[PreparedColumn<'a>],
                scratch: &mut DecodeScratch,
            ) -> UnboundTemplate {
                UnboundTemplate::build_impl(segments, cols, scratch)
            }
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("ssse3")
                && std::arch::is_x86_feature_detected!("sse4.1")
                && std::arch::is_x86_feature_detected!("popcnt")
                && std::arch::is_x86_feature_detected!("bmi1")
                && std::arch::is_x86_feature_detected!("bmi2")
            {
                return unsafe { build_x86_v3(segments, cols, scratch) };
            }
        }
        Self::build_impl(segments, cols, scratch)
    }

    #[inline(always)]
    fn build_impl<'a>(
        segments: &[&'a [u8]],
        cols: &[PreparedColumn<'a>],
        scratch: &mut DecodeScratch,
    ) -> Self {
        // Step 1: Fold constant (`width == 0` / `step == 0`) columns directly into
        // `scratch.str_arena`, and absorb `const_lead` / `const_trail` for active columns.
        let mut seg_ranges = [(0usize, 0usize); MAX_SLOTS + 1];
        let seg0_start = scratch.str_arena.len();
        scratch.str_arena.extend_from_slice(segments[0]);
        seg_ranges[0] = (seg0_start, scratch.str_arena.len());
        let mut active_len = 0usize;
        let mut active_cols_uninit = core::mem::MaybeUninit::<[PreparedColumn; MAX_SLOTS]>::uninit();
        let active_cols: &mut [PreparedColumn; MAX_SLOTS] =
            unsafe { active_cols_uninit.assume_init_mut() };

        for (slot, &col) in cols.iter().enumerate() {
            let right_seg = segments[slot + 1];
            match col.const_full {
                ConstValue::None => {
                    col.const_lead.push_into(&mut scratch.str_arena);
                    seg_ranges[active_len].1 = scratch.str_arena.len();
                    let next_start = scratch.str_arena.len();
                    scratch.str_arena.extend_from_slice(col.const_trail);
                    scratch.str_arena.extend_from_slice(right_seg);
                    active_cols[active_len] = col;
                    active_len += 1;
                    seg_ranges[active_len] = (next_start, scratch.str_arena.len());
                }
                ConstValue::Slice(s) => {
                    scratch.str_arena.extend_from_slice(s);
                    scratch.str_arena.extend_from_slice(right_seg);
                    seg_ranges[active_len].1 = scratch.str_arena.len();
                }
                ConstValue::Inline { ref buf, len } => {
                    scratch.str_arena.extend_from_slice(&buf[..len as usize]);
                    scratch.str_arena.extend_from_slice(right_seg);
                    seg_ranges[active_len].1 = scratch.str_arena.len();
                }
                ConstValue::QuotedDecimal {
                    prefix,
                    ref buf,
                    len,
                } => {
                    scratch.str_arena.push(b'"');
                    scratch.str_arena.extend_from_slice(prefix);
                    scratch.str_arena.extend_from_slice(&buf[..len as usize]);
                    scratch.str_arena.push(b'"');
                    scratch.str_arena.extend_from_slice(right_seg);
                    seg_ranges[active_len].1 = scratch.str_arena.len();
                }
            }
        }

        let slots = active_len;

        // Step 2: Build `FastPiece`s for `MODEL_DICTIONARY` / `MODEL_FRONT_DICT` entries,
        // absorbing adjacent literal segments when 100% of the segment can be eliminated.
        for slot in 0..slots {
            if let Some((slice_off, slice_len)) = active_cols[slot].dict_slices {
                let max_entry = active_cols[slot].max_emit_len;
                let left_range = seg_ranges[slot];
                let left_len_raw = left_range.1 - left_range.0;
                let absorb_left = left_len_raw > 0 && left_len_raw + max_entry <= 48;
                let left_len = if absorb_left { left_len_raw } else { 0 };

                let right_range = seg_ranges[slot + 1];
                let right_len_raw = right_range.1 - right_range.0;
                let cur_len = left_len + max_entry;
                let right_len = if right_len_raw == 0 || cur_len >= 48 {
                    0
                } else if slot + 1 < slots && active_cols[slot + 1].dict_slices.is_some() {
                    let next_max = active_cols[slot + 1].max_emit_len;
                    if cur_len <= 32
                        && next_max <= 32
                        && cur_len + right_len_raw + next_max <= 64
                    {
                        if cur_len + right_len_raw <= 32 {
                            right_len_raw
                        } else if right_len_raw + next_max <= 32 {
                            0
                        } else {
                            32 - cur_len
                        }
                    } else if cur_len + right_len_raw <= 48 {
                        right_len_raw
                    } else if next_max <= 48 && cur_len + right_len_raw + next_max <= 96 {
                        48 - cur_len
                    } else {
                        0
                    }
                } else if cur_len + right_len_raw <= 48 {
                    right_len_raw
                } else {
                    0
                };

                let (piece_offset, u8_offset) = match active_cols[slot].stream {
                    ColumnStream::Dictionary {
                        piece_offset,
                        u8_offset,
                        ..
                    } => (piece_offset, u8_offset),
                    _ => unreachable!(),
                };

                if left_len > 0 || right_len > 0 {
                    let mut buf = [0u8; 96];
                    if left_len > 0 {
                        let left_bytes = &scratch.str_arena[left_range.0..left_range.1];
                        buf[..left_len].copy_from_slice(left_bytes);
                        seg_ranges[slot] = (left_range.0, left_range.0);
                    }
                    let (r0, r1, r2) = if right_len > 0 {
                        let right_ptr =
                            unsafe { scratch.str_arena.as_ptr().add(right_range.0) };
                        seg_ranges[slot + 1].0 = right_range.0 + right_len;
                        unsafe {
                            (
                                read_u128_partial(right_ptr, right_len),
                                if right_len > 16 {
                                    read_u128_partial(right_ptr.add(16), right_len - 16)
                                } else {
                                    0
                                },
                                if right_len > 32 {
                                    read_u128_partial(right_ptr.add(32), right_len - 32)
                                } else {
                                    0
                                },
                            )
                        }
                    } else {
                        (0, 0, 0)
                    };
                    let max_fused = left_len + max_entry + right_len;
                    let piece_ptr = unsafe { scratch.piece_arena.as_mut_ptr().add(piece_offset) };
                    let slice_ptr = unsafe { scratch.slice_arena.as_ptr().add(slice_off) };
                    let b_ptr = buf.as_mut_ptr();
                    for idx in 0..slice_len {
                        let (ptr, len) = unsafe { *slice_ptr.add(idx) };
                        let total = left_len + len + right_len;
                        unsafe {
                            let dst_m = b_ptr.add(left_len);
                            dst_m
                                .cast::<u128>()
                                .write_unaligned(read_u128_partial(ptr, len));
                            if len > 16 {
                                dst_m.add(16).cast::<u128>().write_unaligned(
                                    read_u128_partial(ptr.add(16), len - 16),
                                );
                                if len > 32 {
                                    dst_m.add(32).cast::<u128>().write_unaligned(
                                        read_u128_partial(ptr.add(32), len - 32),
                                    );
                                }
                            }
                            if right_len > 0 {
                                let dst_r = dst_m.add(len);
                                dst_r.cast::<u128>().write_unaligned(r0);
                                if right_len > 16 {
                                    dst_r.add(16).cast::<u128>().write_unaligned(r1);
                                    if right_len > 32 {
                                        dst_r.add(32).cast::<u128>().write_unaligned(r2);
                                    }
                                }
                            }
                            *piece_ptr.add(idx) = FastPiece {
                                v0: b_ptr.cast::<u128>().read_unaligned(),
                                v1: b_ptr.add(16).cast::<u128>().read_unaligned(),
                                v2: b_ptr.add(32).cast::<u128>().read_unaligned(),
                                ptr: std::ptr::null(),
                                len: total,
                            };
                        }
                    }
                    active_cols[slot].max_emit_len = max_fused;
                    active_cols[slot].stream = ColumnStream::Dictionary {
                        piece_offset,
                        u8_offset,
                        max_len: max_fused,
                    };
                } else {
                    let piece_ptr = unsafe { scratch.piece_arena.as_mut_ptr().add(piece_offset) };
                    let slice_ptr = unsafe { scratch.slice_arena.as_ptr().add(slice_off) };
                    for idx in 0..slice_len {
                        let (ptr, len) = unsafe { *slice_ptr.add(idx) };
                        let s = unsafe { std::slice::from_raw_parts(ptr, len) };
                        unsafe {
                            *piece_ptr.add(idx) = FastPiece::from_slice(s);
                        }
                    }
                }
            }
        }

        let head_range = seg_ranges[0];
        let mut row_budget = (head_range.1 - head_range.0) + 96;
        let col_start = scratch.unbound_col_arena.len();
        scratch.unbound_col_arena.reserve(slots);
        for slot in 0..slots {
            let col = active_cols[slot];
            let after_range = seg_ranges[slot + 1];
            row_budget = row_budget
                .saturating_add(col.max_emit_len)
                .saturating_add(after_range.1 - after_range.0);
            let unbound_col: UnboundExecColumn<'static> = unsafe {
                core::mem::transmute(UnboundExecColumn {
                    stream: col.stream,
                    after_range,
                })
            };
            scratch.unbound_col_arena.push(unbound_col);
        }
        let col_end = scratch.unbound_col_arena.len();

        Self {
            head_range,
            col_start,
            col_end,
            row_budget,
        }
    }
}

