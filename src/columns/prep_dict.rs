impl<'a> PreparedColumn<'a> {
    fn parse_dict_or_front(
        tag: u8,
        mut cursor: Cursor<'a>,
        count: usize,
        scratch: &mut DecodeScratch,
    ) -> Result<Self> {
        let entries_len = cursor.uleb()?;
        if entries_len == 0
            || (tag == MODEL_DICTIONARY && entries_len >= count)
            || entries_len > count
        {
            return Err(invalid("column dictionary cardinality"));
        }
        let slice_offset = scratch.slice_arena.len();
        scratch.slice_arena.reserve(entries_len);
                let mut max_len = 0usize;
                let mut first_entry: &'a [u8] = &[];
                let u8_offset = if tag == MODEL_FRONT_DICT {
                    if entries_len < 2 || entries_len > 256 {
                        return Err(invalid("column front dict cardinality"));
                    }
                    let mode_byte = cursor.byte()?;
                    let dict_mode = mode_byte & 0x03;
                    let idx_mode = mode_byte >> 2;
                    if idx_mode > 3 {
                        return Err(invalid("column front dict mode"));
                    }
                    let width = bit_width((entries_len - 1) as u64);
                    let mut lz_dict_cursor = Cursor::new(&[]);
                    let use_lz_cursor = if dict_mode == 2 {
                        let lz_len = cursor.uleb()?;
                        let lz_bytes = cursor.take(lz_len)?;
                        let mut lzc = Cursor::new(lz_bytes);
                        if lzc.byte()? != FAST_LZ_PACKET {
                            return Err(invalid("column front dict lz selector"));
                        }
                        let mut probe = Cursor::new(lzc.bytes);
                        probe.pos = lzc.pos;
                        let raw_len = probe.uleb()?;
                        if raw_len == 0 || raw_len > 65536 {
                            return Err(invalid("column front dict lz length"));
                        }
                        let arena_start = scratch.raw_arena.len();
                        if arena_start + raw_len > scratch.raw_arena.capacity() {
                            return Err(invalid("column front dict lz arena overflow"));
                        }
                        unsafe {
                            scratch.raw_arena.set_len(arena_start + raw_len);
                        }
                        unpack_fast_lz_into(
                            &mut lzc,
                            &mut scratch.raw_arena[arena_start..arena_start + raw_len],
                        )?;
                        let unpacked_slice: &'a [u8] = unsafe {
                            std::slice::from_raw_parts(
                                scratch.raw_arena.as_ptr().add(arena_start),
                                raw_len,
                            )
                        };
                        lz_dict_cursor = Cursor::new(unpacked_slice);
                        true
                    } else {
                        false
                    };
                    scratch.slice_arena.reserve(entries_len);
                    if dict_mode == 0 {
                        let mut prev_entry: Option<&'a [u8]> = None;
                        for _ in 0..entries_len {
                            let len = cursor.uleb()?;
                            let entry = cursor.take(len)?;
                            if prev_entry.is_some_and(|last| last >= entry) {
                                return Err(invalid("column dictionary ordering"));
                            }
                            prev_entry = Some(entry);
                            if len > max_len {
                                max_len = len;
                            }
                            scratch.slice_arena.push((entry.as_ptr(), len));
                        }
                    } else if dict_mode == 3 {
                        let (pal, pal_len, pal_mixed) = read_front_dict_palette(&mut cursor)?;
                        let len0 = cursor.uleb()?;
                        let raw_first = cursor.take(len0)?;
                        let first_start = scratch.raw_arena.len();
                        if first_start + len0 + 32 > scratch.raw_arena.capacity() {
                            return Err(invalid("column front dictionary arena overflow"));
                        }
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                raw_first.as_ptr(),
                                scratch.raw_arena.as_mut_ptr().add(first_start),
                                len0,
                            );
                            scratch.raw_arena.set_len(first_start + len0);
                            first_entry = std::slice::from_raw_parts(
                                scratch.raw_arena.as_ptr().add(first_start),
                                len0,
                            );
                        }
                        max_len = len0;
                        scratch.slice_arena.push((first_entry.as_ptr(), len0));
                        let mut prev_entry: &'a [u8] = first_entry;
                        #[cfg(target_arch = "aarch64")]
                        let v_pal = unsafe { core::arch::aarch64::vld1q_u8(pal.as_ptr()) };
                        #[cfg(target_arch = "x86_64")]
                        let (x86_v_pal, has_ssse3) = unsafe {
                            (
                                core::arch::x86_64::_mm_loadu_si128(
                                    pal.as_ptr().cast::<core::arch::x86_64::__m128i>(),
                                ),
                                std::arch::is_x86_feature_detected!("ssse3"),
                            )
                        };
                        for _ in 1..entries_len {
                            let (lcp, lcs, raw_hdr) = if cursor.pos + 3 <= cursor.bytes.len() {
                                let p = unsafe { cursor.bytes.as_ptr().add(cursor.pos) };
                                let b0 = unsafe { *p };
                                let b1 = unsafe { *p.add(1) };
                                let b2 = unsafe { *p.add(2) };
                                if (b0 | b1 | b2) < 0x80 {
                                    cursor.pos += 3;
                                    (b0 as usize, b1 as usize, b2 as usize)
                                } else {
                                    (cursor.uleb()?, cursor.uleb()?, cursor.uleb()?)
                                }
                            } else {
                                (cursor.uleb()?, cursor.uleb()?, cursor.uleb()?)
                            };
                            let (mid_len, is_raw) = if pal_mixed {
                                (raw_hdr >> 1, (raw_hdr & 1) != 0)
                            } else {
                                (raw_hdr, false)
                            };
                            if mid_len > 65536 {
                                return Err(invalid("column front dict entry length"));
                            }
                            let packed_mid = if is_raw {
                                cursor.take(mid_len)?
                            } else if pal_len <= 16 {
                                cursor.take((mid_len + 1) >> 1)?
                            } else {
                                cursor.take((mid_len * 5 + 7) >> 3)?
                            };
                            if lcp + lcs > prev_entry.len() {
                                return Err(invalid("column front dictionary bounds"));
                            }
                            let total_len = lcp + mid_len + lcs;
                            let arena_start = scratch.raw_arena.len();
                            if arena_start + total_len + 32 > scratch.raw_arena.capacity() {
                                return Err(invalid("column front dictionary arena overflow"));
                            }
                            #[cfg(target_arch = "aarch64")]
                            let handled_fast = if !is_raw
                                && pal_len <= 16
                                && lcp <= 32
                                && mid_len <= 32
                                && lcs <= 16
                            {
                                unsafe {
                                    use core::arch::aarch64::*;
                                    let dst = scratch.raw_arena.as_mut_ptr().add(arena_start);
                                    let p_ptr = prev_entry.as_ptr();
                                    let lcs_v0 = p_ptr
                                        .add(prev_entry.len() - lcs)
                                        .cast::<u128>()
                                        .read_unaligned();
                                    dst.cast::<u128>()
                                        .write_unaligned(p_ptr.cast::<u128>().read_unaligned());
                                    if lcp > 16 {
                                        dst.add(16).cast::<u128>().write_unaligned(
                                            p_ptr.add(16).cast::<u128>().read_unaligned(),
                                        );
                                    }
                                    if mid_len > 0 {
                                        let packed_len = packed_mid.len();
                                        if (mid_len & 1) != 0
                                            && (*packed_mid.get_unchecked(packed_len - 1) & 0x0f)
                                                != 0
                                        {
                                            return Err(invalid("column front dict nibble padding"));
                                        }
                                        let raw128 =
                                            read_u128_partial(packed_mid.as_ptr(), packed_len);
                                        let raw64 = raw128 as u64;
                                        let v8 = vreinterpret_u8_u64(vdup_n_u64(raw64.to_le()));
                                        let hi = vshr_n_u8(v8, 4);
                                        let lo = vand_u8(v8, vdup_n_u8(0x0f));
                                        let zipped =
                                            vcombine_u8(vzip1_u8(hi, lo), vzip2_u8(hi, lo));
                                        if pal_len < 16 && vmaxvq_u8(zipped) >= pal_len {
                                            return Err(invalid(
                                                "column front dict nibble out of bounds",
                                            ));
                                        }
                                        vst1q_u8(dst.add(lcp), vqtbl1q_u8(v_pal, zipped));
                                        if mid_len > 16 {
                                            let raw64_hi = (raw128 >> 64) as u64;
                                            let v8_hi =
                                                vreinterpret_u8_u64(vdup_n_u64(raw64_hi.to_le()));
                                            let hi2 = vshr_n_u8(v8_hi, 4);
                                            let lo2 = vand_u8(v8_hi, vdup_n_u8(0x0f));
                                            let zipped2 =
                                                vcombine_u8(vzip1_u8(hi2, lo2), vzip2_u8(hi2, lo2));
                                            if pal_len < 16 && vmaxvq_u8(zipped2) >= pal_len {
                                                return Err(invalid(
                                                    "column front dict nibble out of bounds",
                                                ));
                                            }
                                            vst1q_u8(
                                                dst.add(lcp + 16),
                                                vqtbl1q_u8(v_pal, zipped2),
                                            );
                                        }
                                    }
                                    dst.add(lcp + mid_len)
                                        .cast::<u128>()
                                        .write_unaligned(lcs_v0);
                                    scratch.raw_arena.set_len(arena_start + total_len);
                                }
                                true
                            } else {
                                false
                            };
                            #[cfg(target_arch = "x86_64")]
                            let handled_fast = if has_ssse3
                                && !is_raw
                                && pal_len <= 16
                                && lcp <= 32
                                && mid_len <= 32
                                && lcs <= 16
                            {
                                unsafe {
                                    if mid_len > 0
                                        && (mid_len & 1) != 0
                                        && (*packed_mid.get_unchecked(packed_mid.len() - 1) & 0x0f)
                                            != 0
                                    {
                                        return Err(invalid("column front dict nibble padding"));
                                    }
                                    let dst = scratch.raw_arena.as_mut_ptr().add(arena_start);
                                    if !unpack_front_dict_fast_ssse3(
                                        dst,
                                        prev_entry.as_ptr(),
                                        prev_entry.len(),
                                        lcp,
                                        mid_len,
                                        lcs,
                                        packed_mid,
                                        x86_v_pal,
                                        pal_len,
                                    ) {
                                        return Err(invalid(
                                            "column front dict nibble out of bounds",
                                        ));
                                    }
                                    scratch.raw_arena.set_len(arena_start + total_len);
                                }
                                true
                            } else {
                                false
                            };
                            #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
                            let handled_fast = false;
                            if !handled_fast {
                                unsafe {
                                    let dst = scratch.raw_arena.as_mut_ptr().add(arena_start);
                                    if lcp > 0 {
                                        std::ptr::copy_nonoverlapping(
                                            prev_entry.as_ptr(),
                                            dst,
                                            lcp,
                                        );
                                    }
                                    if mid_len > 0 {
                                        if is_raw {
                                            std::ptr::copy_nonoverlapping(
                                                packed_mid.as_ptr(),
                                                dst.add(lcp),
                                                mid_len,
                                            );
                                        } else {
                                            unpack_nibble_bytes_into(
                                                packed_mid,
                                                mid_len,
                                                &pal,
                                                pal_len,
                                                dst.add(lcp),
                                            )?;
                                        }
                                    }
                                    if lcs > 0 {
                                        std::ptr::copy_nonoverlapping(
                                            prev_entry.as_ptr().add(prev_entry.len() - lcs),
                                            dst.add(lcp + mid_len),
                                            lcs,
                                        );
                                    }
                                    scratch.raw_arena.set_len(arena_start + total_len);
                                }
                            }
                            let entry: &'a [u8] = unsafe {
                                std::slice::from_raw_parts(
                                    scratch.raw_arena.as_ptr().add(arena_start),
                                    total_len,
                                )
                            };
                            let ord_ok = if lcp < prev_entry.len() && lcp < total_len {
                                let a = unsafe { *prev_entry.as_ptr().add(lcp) };
                                let b = unsafe { *entry.as_ptr().add(lcp) };
                                if a != b {
                                    a < b
                                } else {
                                    prev_entry[lcp + 1..] < entry[lcp + 1..]
                                }
                            } else {
                                prev_entry.len() < total_len
                            };
                            if !ord_ok {
                                return Err(invalid("column dictionary ordering"));
                            }
                            prev_entry = entry;
                            if total_len > max_len {
                                max_len = total_len;
                            }
                            scratch.slice_arena.push((entry.as_ptr(), total_len));
                        }
                    } else {
                        let dcur = if use_lz_cursor {
                            &mut lz_dict_cursor
                        } else {
                            &mut cursor
                        };
                        let len0 = dcur.uleb()?;
                        let raw_first = dcur.take(len0)?;
                        let first_start = scratch.raw_arena.len();
                        if first_start + len0 + 32 > scratch.raw_arena.capacity() {
                            return Err(invalid("column front dictionary arena overflow"));
                        }
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                raw_first.as_ptr(),
                                scratch.raw_arena.as_mut_ptr().add(first_start),
                                len0,
                            );
                            scratch.raw_arena.set_len(first_start + len0);
                            first_entry = std::slice::from_raw_parts(
                                scratch.raw_arena.as_ptr().add(first_start),
                                len0,
                            );
                        }
                        max_len = len0;
                        scratch.slice_arena.push((first_entry.as_ptr(), len0));
                        let mut prev_entry: &'a [u8] = first_entry;
                        for _ in 1..entries_len {
                            let (lcp, lcs, mid_len) = if dcur.pos + 3 <= dcur.bytes.len() {
                                let p = unsafe { dcur.bytes.as_ptr().add(dcur.pos) };
                                let b0 = unsafe { *p };
                                let b1 = unsafe { *p.add(1) };
                                let b2 = unsafe { *p.add(2) };
                                if (b0 | b1 | b2) < 0x80 {
                                    dcur.pos += 3;
                                    (b0 as usize, b1 as usize, b2 as usize)
                                } else {
                                    (dcur.uleb()?, dcur.uleb()?, dcur.uleb()?)
                                }
                            } else {
                                (dcur.uleb()?, dcur.uleb()?, dcur.uleb()?)
                            };
                            let mid = dcur.take(mid_len)?;
                            if lcp + lcs > prev_entry.len() {
                                return Err(invalid("column front dictionary bounds"));
                            }
                            let total_len = lcp + mid_len + lcs;
                            let arena_start = scratch.raw_arena.len();
                            if arena_start + total_len + 32 > scratch.raw_arena.capacity() {
                                return Err(invalid("column front dictionary arena overflow"));
                            }
                            unsafe {
                                let dst = scratch.raw_arena.as_mut_ptr().add(arena_start);
                                if lcp <= 32 && mid_len <= 32 && lcs <= 16 {
                                    let p_ptr = prev_entry.as_ptr();
                                    let lcs_v0 = p_ptr
                                        .add(prev_entry.len() - lcs)
                                        .cast::<u128>()
                                        .read_unaligned();
                                    dst.cast::<u128>()
                                        .write_unaligned(p_ptr.cast::<u128>().read_unaligned());
                                    if lcp > 16 {
                                        dst.add(16).cast::<u128>().write_unaligned(
                                            p_ptr.add(16).cast::<u128>().read_unaligned(),
                                        );
                                    }
                                    if mid_len > 0 {
                                        let m_ptr = mid.as_ptr();
                                        dst.add(lcp)
                                            .cast::<u128>()
                                            .write_unaligned(read_u128_partial(m_ptr, mid_len));
                                        if mid_len > 16 {
                                            dst.add(lcp + 16).cast::<u128>().write_unaligned(
                                                read_u128_partial(m_ptr.add(16), mid_len - 16),
                                            );
                                        }
                                    }
                                    dst.add(lcp + mid_len)
                                        .cast::<u128>()
                                        .write_unaligned(lcs_v0);
                                } else {
                                    if lcp > 0 {
                                        std::ptr::copy_nonoverlapping(
                                            prev_entry.as_ptr(),
                                            dst,
                                            lcp,
                                        );
                                    }
                                    if mid_len > 0 {
                                        std::ptr::copy_nonoverlapping(
                                            mid.as_ptr(),
                                            dst.add(lcp),
                                            mid_len,
                                        );
                                    }
                                    if lcs > 0 {
                                        std::ptr::copy_nonoverlapping(
                                            prev_entry.as_ptr().add(prev_entry.len() - lcs),
                                            dst.add(lcp + mid_len),
                                            lcs,
                                        );
                                    }
                                }
                                scratch.raw_arena.set_len(arena_start + total_len);
                            }
                            let entry: &'a [u8] = unsafe {
                                std::slice::from_raw_parts(
                                    scratch.raw_arena.as_ptr().add(arena_start),
                                    total_len,
                                )
                            };
                            let ord_ok = if lcp < prev_entry.len() && lcp < total_len {
                                let a = unsafe { *prev_entry.as_ptr().add(lcp) };
                                let b = unsafe { *entry.as_ptr().add(lcp) };
                                if a != b {
                                    a < b
                                } else {
                                    prev_entry[lcp + 1..] < entry[lcp + 1..]
                                }
                            } else {
                                prev_entry.len() < total_len
                            };
                            if !ord_ok {
                                return Err(invalid("column dictionary ordering"));
                            }
                            prev_entry = entry;
                            if total_len > max_len {
                                max_len = total_len;
                            }
                            scratch.slice_arena.push((entry.as_ptr(), total_len));
                        }
                        if use_lz_cursor && !dcur.done() {
                            return Err(invalid("column front dict lz trailing"));
                        }
                    }
                    unpack_front_dict_indices(
                        &mut cursor,
                        count,
                        entries_len,
                        width,
                        idx_mode,
                        &mut scratch.u8_arena,
                    )?
                } else {
                    scratch.slice_arena.reserve(entries_len);
                    let mut prev_entry: Option<&'a [u8]> = None;
                    for i in 0..entries_len {
                        let len = cursor.uleb()?;
                        let entry = cursor.take(len)?;
                        if prev_entry.is_some_and(|last| last >= entry) {
                            return Err(invalid("column dictionary ordering"));
                        }
                        prev_entry = Some(entry);
                        if i == 0 {
                            first_entry = entry;
                        }
                        if len > max_len {
                            max_len = len;
                        }
                        scratch.slice_arena.push((entry.as_ptr(), len));
                    }
                    let width = cursor.byte()?;
                    if width != bit_width((entries_len - 1) as u64) {
                        return Err(invalid("column dictionary width"));
                    }
                    let packed_len = bit_bytes(count, width)?;
                    let packed_bytes = cursor.take(packed_len)?;
                    if !cursor.done() {
                        return Err(invalid("column model trailing bytes"));
                    }
                    if entries_len > 256 {
                        // Wide dictionary (9..=16-bit ids): the u8-indexed exec streams cannot
                        // represent it, so materialize the column as length-prefixed rows and
                        // let the RAW stream emit it. Decode-only; encoder output is unchanged.
                        let mut reader = FastBitReader::new_validated(packed_bytes, count, width)?;
                        let mut total = 0usize;
                        let mut wide_max = 0usize;
                        let mut scan = reader;
                        for _ in 0..count {
                            let idx = scan.next_u64() as usize;
                            if idx >= entries_len {
                                return Err(invalid("column bit index out of bounds"));
                            }
                            let len = scratch.slice_arena[slice_offset + idx].1;
                            total += uleb_len(len) + len;
                            wide_max = wide_max.max(len);
                        }
                        let start = scratch.raw_arena.len();
                        if start + total + 32 > scratch.raw_arena.capacity() {
                            return Err(invalid("column wide dictionary arena overflow"));
                        }
                        unsafe {
                            let mut dst = scratch.raw_arena.as_mut_ptr().add(start);
                            for _ in 0..count {
                                let idx = reader.next_u64() as usize;
                                let (ptr, len) = *scratch.slice_arena.get_unchecked(slice_offset + idx);
                                let mut v = len;
                                loop {
                                    let b = (v & 0x7f) as u8;
                                    v >>= 7;
                                    if v != 0 {
                                        *dst = b | 0x80;
                                        dst = dst.add(1);
                                    } else {
                                        *dst = b;
                                        dst = dst.add(1);
                                        break;
                                    }
                                }
                                std::ptr::copy_nonoverlapping(ptr, dst, len);
                                dst = dst.add(len);
                            }
                            scratch.raw_arena.set_len(start + total);
                        }
                        scratch.slice_arena.truncate(slice_offset);
                        let rows: &'a [u8] = unsafe {
                            std::slice::from_raw_parts(scratch.raw_arena.as_ptr().add(start), total)
                        };
                        return Ok(Self {
                            const_full: ConstValue::None,
                            const_lead: LeadValue::Empty,
                            const_trail: b"",
                            dict_slices: None,
                            max_emit_len: wide_max,
                            stream: ColumnStream::Raw {
                                cursor: Cursor::new(rows),
                            },
                        });
                    }
                    if entries_len == 1 {
                        let _ = FastBitReader::new_validated(packed_bytes, count, width)?;
                        scratch.slice_arena.truncate(slice_offset);
                        return Ok(Self {
                            const_full: ConstValue::Slice(first_entry),
                            const_lead: LeadValue::Empty,
                            const_trail: b"",
                            dict_slices: None,
                            max_emit_len: 0,
                            stream: ColumnStream::Raw {
                                cursor: Cursor::new(&[]),
                            },
                        });
                    }
                    unpack_and_validate_u8(
                        packed_bytes,
                        count,
                        width,
                        0,
                        entries_len,
                        &mut scratch.u8_arena,
                    )?
                };
                let piece_offset = scratch.piece_arena.len();
                scratch.piece_arena.reserve(entries_len);
                unsafe {
                    scratch.piece_arena.set_len(piece_offset + entries_len);
                }
                Ok(Self {
                    const_full: ConstValue::None,
                    const_lead: LeadValue::Empty,
                    const_trail: b"",
                    dict_slices: Some((slice_offset, entries_len)),
                    max_emit_len: max_len,
                    stream: ColumnStream::Dictionary {
                        piece_offset,
                        u8_offset,
                        max_len,
                    },
                })
    }
}

