impl UnboundTemplate {
    /// Bind all arena offsets to raw pointers once all `DecodeScratch` arenas are frozen.
    fn bind(self, scratch: &mut DecodeScratch) -> PreparedTemplate {
        let head = FastPiece::from_slice(&scratch.str_arena[self.head_range.0..self.head_range.1]);
        let u8_base = scratch.u8_arena.as_ptr();
        let u16_base = scratch.u16_arena.as_ptr();
        let u64_base = scratch.u64_arena.as_ptr();
        let piece_base = scratch.piece_arena.as_ptr();
        let exec_start = scratch.exec_col_arena.len();
        let mut i = self.col_start;
        while i < self.col_end {
            let col = scratch.unbound_col_arena[i];
            if col.after_range.0 == col.after_range.1 {
                if let ColumnStream::Dictionary {
                    piece_offset: p0,
                    u8_offset: u0,
                    max_len: m0,
                } = col.stream
                {
                    if i + 3 < self.col_end
                        && scratch.unbound_col_arena[i + 1].after_range.0
                            == scratch.unbound_col_arena[i + 1].after_range.1
                        && scratch.unbound_col_arena[i + 2].after_range.0
                            == scratch.unbound_col_arena[i + 2].after_range.1
                        && scratch.unbound_col_arena[i + 3].after_range.0
                            == scratch.unbound_col_arena[i + 3].after_range.1
                    {
                        if let (
                            ColumnStream::Dictionary {
                                piece_offset: p1,
                                u8_offset: u1,
                                max_len: m1,
                            },
                            ColumnStream::Dictionary {
                                piece_offset: p2,
                                u8_offset: u2,
                                max_len: m2,
                            },
                            ColumnStream::Dictionary {
                                piece_offset: p3,
                                u8_offset: u3,
                                max_len: m3,
                            },
                        ) = (
                            scratch.unbound_col_arena[i + 1].stream,
                            scratch.unbound_col_arena[i + 2].stream,
                            scratch.unbound_col_arena[i + 3].stream,
                        ) {
                            let dp1 = p1.wrapping_sub(p0);
                            let dp2 = p2.wrapping_sub(p0);
                            let dp3 = p3.wrapping_sub(p0);
                            let du1 = u1.wrapping_sub(u0);
                            let du2 = u2.wrapping_sub(u0);
                            let du3 = u3.wrapping_sub(u0);
                            let max4 = m0.max(m1).max(m2).max(m3);
                            if (dp1 | dp2 | dp3 | du1 | du2 | du3) <= 65535 && max4 <= 48 {
                                let p_off = [dp1 as u16, dp2 as u16, dp3 as u16];
                                let u_off = [du1 as u16, du2 as u16, du3 as u16];
                                let stream = unsafe {
                                    if max4 <= 32 {
                                        BoundStream::DictionaryQuad32 {
                                            entries0: piece_base.add(p0),
                                            idx_ptr0: u8_base.add(u0),
                                            p_off,
                                            u_off,
                                        }
                                    } else {
                                        BoundStream::DictionaryQuad48 {
                                            entries0: piece_base.add(p0),
                                            idx_ptr0: u8_base.add(u0),
                                            p_off,
                                            u_off,
                                        }
                                    }
                                };
                                scratch.exec_col_arena.push(ExecColumn {
                                    stream,
                                    after: FastPiece::empty(),
                                });
                                i += 4;
                                continue;
                            }
                        }
                    }
                    if i + 1 < self.col_end
                        && scratch.unbound_col_arena[i + 1].after_range.0
                            == scratch.unbound_col_arena[i + 1].after_range.1
                    {
                        if let ColumnStream::Dictionary {
                            piece_offset: p1,
                            u8_offset: u1,
                            max_len: m1,
                        } = scratch.unbound_col_arena[i + 1].stream
                        {
                            let idx_delta = (u1 as isize).wrapping_sub(u0 as isize);
                            if m0 <= 32 && m1 <= 32 {
                                scratch.exec_col_arena.push(ExecColumn {
                                    stream: unsafe {
                                        BoundStream::DictionaryPair32 {
                                            entries0: piece_base.add(p0),
                                            entries1: piece_base.add(p1),
                                            idx_ptr0: u8_base.add(u0),
                                            idx_delta,
                                        }
                                    },
                                    after: FastPiece::empty(),
                                });
                                i += 2;
                                continue;
                            } else if m0 <= 48 && m1 <= 48 {
                                scratch.exec_col_arena.push(ExecColumn {
                                    stream: unsafe {
                                        BoundStream::DictionaryPair48 {
                                            entries0: piece_base.add(p0),
                                            entries1: piece_base.add(p1),
                                            idx_ptr0: u8_base.add(u0),
                                            idx_delta,
                                        }
                                    },
                                    after: FastPiece::empty(),
                                });
                                i += 2;
                                continue;
                            }
                        }
                    }
                }
            }
            let after = if col.after_range.0 == col.after_range.1 {
                FastPiece::empty()
            } else {
                FastPiece::from_slice(&scratch.str_arena[col.after_range.0..col.after_range.1])
            };
            let bound = unsafe {
                match col.stream {
                    ColumnStream::Raw { cursor } => BoundStream::Raw { cursor },
                    ColumnStream::Dictionary {
                        piece_offset,
                        u8_offset,
                        max_len,
                    } => {
                        if after.len == 0 && max_len <= 16 {
                            BoundStream::Dictionary16 {
                                entries: piece_base.add(piece_offset),
                                idx_ptr: u8_base.add(u8_offset),
                            }
                        } else if after.len == 0 && max_len <= 32 {
                            BoundStream::Dictionary32 {
                                entries: piece_base.add(piece_offset),
                                idx_ptr: u8_base.add(u8_offset),
                            }
                        } else if after.len == 0 && max_len <= 48 {
                            BoundStream::Dictionary48 {
                                entries: piece_base.add(piece_offset),
                                idx_ptr: u8_base.add(u8_offset),
                            }
                        } else {
                            BoundStream::DictionaryLong {
                                entries: piece_base.add(piece_offset),
                                idx_ptr: u8_base.add(u8_offset),
                            }
                        }
                    }
                    ColumnStream::DecimalAffineSmall { current, step } => {
                        if (1..=16).contains(&after.len) {
                            BoundStream::DecimalAffineSmall16 { current, step }
                        } else if (17..=32).contains(&after.len) {
                            BoundStream::DecimalAffineSmall32 { current, step }
                        } else if (33..=48).contains(&after.len) {
                            BoundStream::DecimalAffineSmall48 { current, step }
                        } else {
                            BoundStream::DecimalAffineSmall { current, step }
                        }
                    }
                    ColumnStream::DecimalAffine { current, step } => {
                        BoundStream::DecimalAffine { current, step }
                    }
                    ColumnStream::DecimalSmall { u16_offset } => {
                        let val_ptr = u16_base.add(u16_offset);
                        if (1..=16).contains(&after.len) {
                            BoundStream::DecimalSmall16 { val_ptr }
                        } else if (17..=32).contains(&after.len) {
                            BoundStream::DecimalSmall32 { val_ptr }
                        } else if (33..=48).contains(&after.len) {
                            BoundStream::DecimalSmall48 { val_ptr }
                        } else {
                            BoundStream::DecimalSmall { val_ptr }
                        }
                    }
                    ColumnStream::DecimalPacked { min, reader } => {
                        BoundStream::DecimalPacked { min, reader }
                    }
                    ColumnStream::DecimalValues { u64_offset } => BoundStream::DecimalValues {
                        val_ptr: u64_base.add(u64_offset),
                    },
                    ColumnStream::HexSuffix4 { ptr } => BoundStream::HexSuffix4 { ptr },
                    ColumnStream::HexSuffix8 { ptr } => BoundStream::HexSuffix8 { ptr },
                    ColumnStream::HexSuffix { cursor, width } => {
                        BoundStream::HexSuffix { cursor, width }
                    }
                    ColumnStream::TimestampMinSec {
                        min_u8_offset,
                        sec_u8_offset,
                    } => {
                        let min_ptr = u8_base.add(min_u8_offset);
                        let sec_ptr = u8_base.add(sec_u8_offset);
                        if (1..=16).contains(&after.len) {
                            BoundStream::TimestampMinSec16 { min_ptr, sec_ptr }
                        } else if (17..=32).contains(&after.len) {
                            BoundStream::TimestampMinSec32 { min_ptr, sec_ptr }
                        } else {
                            BoundStream::TimestampMinSec { min_ptr, sec_ptr }
                        }
                    }
                    ColumnStream::TimestampFull { u8_offset, stride } => {
                        BoundStream::TimestampFull {
                            ptr: u8_base.add(u8_offset),
                            stride,
                        }
                    }
                    ColumnStream::Copy16 { u8_offset } => {
                        let ptr = u8_base.add(u8_offset);
                        if (1..=16).contains(&after.len) {
                            BoundStream::Copy16After16 { ptr }
                        } else if (17..=32).contains(&after.len) {
                            BoundStream::Copy16After32 { ptr }
                        } else {
                            BoundStream::Copy16 { ptr }
                        }
                    }
                    ColumnStream::SegmentDict {
                        str_u8_offset,
                        len_u16_offset,
                    } => BoundStream::SegmentDict {
                        str_ptr: u8_base.add(str_u8_offset),
                        len_ptr: u16_base.add(len_u16_offset),
                    },
                    ColumnStream::PeriodicString { cursor, period } => {
                        BoundStream::PeriodicString { cursor, period }
                    }
                    ColumnStream::PeriodicHex { cursor, period } => {
                        BoundStream::PeriodicHex { cursor, period }
                    }
                }
            };
            scratch.exec_col_arena.push(ExecColumn {
                stream: bound,
                after,
            });
            i += 1;
        }
        let exec_end = scratch.exec_col_arena.len();
        PreparedTemplate {
            head,
            col_ptr: unsafe { scratch.exec_col_arena.as_mut_ptr().add(exec_start) },
            col_len: exec_end - exec_start,
            row_budget: self.row_budget,
        }
    }
}

impl PreparedTemplate {
    #[inline(always)]
    unsafe fn emit_row(&mut self, mut dst: *mut u8) -> *mut u8 {
        unsafe {
            if self.head.len > 0 {
                if self.head.len <= 16 {
                    dst = self.head.emit_16(dst);
                } else if self.head.len <= 32 {
                    dst = self.head.emit_32(dst);
                } else {
                    dst = self.head.emit_48(dst);
                }
            }
            let slots = self.col_len;
            let col_ptr = self.col_ptr;
            for slot in 0..slots {
                dst = (*col_ptr.add(slot)).emit(dst);
            }
            dst
        }
    }
}

#[cold]
#[inline(never)]
fn emit_template_row_tail(
    tpl: &mut PreparedTemplate,
    budget: usize,
    base: *mut u8,
    pos: &mut usize,
    capacity: usize,
    tail_buf: &mut core::mem::MaybeUninit<[u8; 4096]>,
) -> Result<()> {
    if budget <= 4096 {
        let written = unsafe {
            let start_ptr = tail_buf.as_mut_ptr().cast::<u8>();
            let end_ptr = tpl.emit_row(start_ptr);
            end_ptr.offset_from(start_ptr) as usize
        };
        if *pos + written > capacity {
            return Err(invalid("column output exceeds length"));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                tail_buf.as_ptr().cast::<u8>(),
                base.add(*pos),
                written,
            );
        }
        *pos += written;
    } else {
        let mut heap_row = vec![0u8; budget];
        let written = unsafe {
            let start_ptr = heap_row.as_mut_ptr();
            let end_ptr = tpl.emit_row(start_ptr);
            end_ptr.offset_from(start_ptr) as usize
        };
        if *pos + written > capacity {
            return Err(invalid("column output exceeds length"));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(heap_row.as_ptr(), base.add(*pos), written);
        }
        *pos += written;
    }
    Ok(())
}

#[inline(always)]
fn emit_template_rows(
    tpl: &mut PreparedTemplate,
    mut remaining: usize,
    base: *mut u8,
    pos: &mut usize,
    capacity: usize,
    tail_buf: &mut core::mem::MaybeUninit<[u8; 4096]>,
) -> Result<()> {
    let budget = tpl.row_budget;
    while remaining > 0 {
        let avail = capacity.saturating_sub(*pos);
        let safe_rows = if budget > 0 {
            (avail / budget).min(remaining)
        } else {
            remaining
        };
        if safe_rows > 0 {
            unsafe {
                let mut dst = base.add(*pos);
                for _ in 0..safe_rows {
                    dst = tpl.emit_row(dst);
                }
                *pos = dst.offset_from(base) as usize;
            }
            remaining -= safe_rows;
            if remaining == 0 {
                break;
            }
            if *pos + budget <= capacity {
                continue;
            }
        }
        emit_template_row_tail(tpl, budget, base, pos, capacity, tail_buf)?;
        remaining -= 1;
    }
    Ok(())
}

