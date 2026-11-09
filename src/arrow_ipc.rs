//! Pure-Rust, zero-dependency Apache Arrow IPC Stream (`MetadataVersion::V5`) serializer and
//! deserializer (`application/vnd.apache.arrow.stream`).

use crate::{
    arrow::{ArrowBatch, ArrowCol, ArrowData, ArrowType},
    Error, Result,
};

const IPC_CONTINUATION: u32 = 0xFFFF_FFFF;
const META_VERSION_V5: i16 = 4;
const MSG_HEADER_SCHEMA: u8 = 1;
const MSG_HEADER_RECORD_BATCH: u8 = 3;

const FB_TYPE_INT: u8 = 2;
const FB_TYPE_FLOAT: u8 = 3;
const FB_TYPE_UTF8: u8 = 5;
const FB_TYPE_BOOL: u8 = 6;
const FB_TYPE_LARGE_UTF8: u8 = 20;
const FB_FLOAT_DOUBLE: i16 = 2;

/// Reverse-growing canonical Google FlatBuffers builder (`soffset > 0`, `uoffset > 0`).
struct FbRev {
    buf: Vec<u8>,
    sp: usize,
}

impl FbRev {
    fn with_capacity(cap: usize) -> Self {
        let c = cap.next_power_of_two().max(1024);
        Self {
            buf: vec![0u8; c],
            sp: c,
        }
    }

    fn ensure(&mut self, extra: usize) {
        if self.sp >= extra + 64 {
            return;
        }
        let used = self.buf.len() - self.sp;
        let new_cap = (self.buf.len() + extra + 1024).next_power_of_two();
        let mut next = vec![0u8; new_cap];
        let new_sp = new_cap - used;
        next[new_sp..].copy_from_slice(&self.buf[self.sp..]);
        self.buf = next;
        self.sp = new_sp;
    }

    #[inline]
    fn pos_from_end(&self) -> usize {
        self.buf.len() - self.sp
    }

    #[inline]
    fn sp_of(&self, off_from_end: usize) -> usize {
        self.buf.len() - off_from_end
    }

    #[inline]
    fn align(&mut self, a: usize) {
        self.ensure(a);
        let used = self.pos_from_end();
        let pad = (a - (used & (a - 1))) & (a - 1);
        self.sp -= pad;
    }

    #[inline]
    fn push_bytes(&mut self, b: &[u8]) {
        self.ensure(b.len());
        self.sp -= b.len();
        self.buf[self.sp..self.sp + b.len()].copy_from_slice(b);
    }

    #[inline]
    fn push_u8(&mut self, v: u8) { self.push_bytes(&[v]); }
    #[inline]
    fn push_u16(&mut self, v: u16) { self.push_bytes(&v.to_le_bytes()); }
    #[inline]
    fn push_i16(&mut self, v: i16) { self.push_bytes(&v.to_le_bytes()); }
    #[inline]
    fn push_u32(&mut self, v: u32) { self.push_bytes(&v.to_le_bytes()); }
    #[inline]
    fn push_i32(&mut self, v: i32) { self.push_bytes(&v.to_le_bytes()); }
    #[inline]
    fn push_i64(&mut self, v: i64) { self.push_bytes(&v.to_le_bytes()); }

    #[inline]
    fn push_uoffset_to(&mut self, target_off_from_end: usize) {
        self.align(4);
        self.ensure(4);
        let slot_sp = self.sp - 4;
        let target_sp = self.sp_of(target_off_from_end);
        debug_assert!(target_sp > slot_sp);
        self.sp = slot_sp;
        let rel = (target_sp - slot_sp) as u32;
        self.buf[slot_sp..slot_sp + 4].copy_from_slice(&rel.to_le_bytes());
    }

    fn create_string(&mut self, s: &str) -> usize {
        self.align(4);
        let pad_plus_nul = 4 - (s.len() & 3);
        self.push_bytes(&[0u8; 4][..pad_plus_nul]);
        self.push_bytes(s.as_bytes());
        self.push_u32(s.len() as u32);
        self.pos_from_end()
    }

    fn create_offset_vec(&mut self, elem_offs_from_end: &[usize]) -> usize {
        self.align(4);
        for &off in elem_offs_from_end.iter().rev() {
            self.push_uoffset_to(off);
        }
        self.push_u32(elem_offs_from_end.len() as u32);
        self.pos_from_end()
    }

    fn create_i64_pair_struct_vec(&mut self, pairs: &[(i64, i64)]) -> usize {
        self.align(8);
        for &(a, b) in pairs.iter().rev() {
            self.push_i64(b);
            self.push_i64(a);
        }
        self.push_u32(pairs.len() as u32);
        self.pos_from_end()
    }

    /// Finish a table whose body started at `body_end_from_end` and whose fields sit at
    /// `field_offs_from_end[id]` (`0` if absent). Emits `soffset_t` + preceding `vtable`.
    fn finish_table(&mut self, body_end_from_end: usize, field_offs_from_end: &[usize]) -> usize {
        self.align(4);
        self.ensure(4 + 4 + field_offs_from_end.len() * 2 + 4);
        let table_sp = self.sp - 4;
        self.sp = table_sp;
        let body_end_sp = self.sp_of(body_end_from_end);
        let table_bytes = (body_end_sp - table_sp) as u16;
        let vtable_bytes = (4 + field_offs_from_end.len() * 2) as u16;

        for &foff in field_offs_from_end.iter().rev() {
            if foff == 0 {
                self.push_u16(0);
            } else {
                let f_sp = self.sp_of(foff);
                self.push_u16((f_sp - table_sp) as u16);
            }
        }
        self.push_u16(table_bytes);
        self.push_u16(vtable_bytes);
        let vtable_sp = self.sp;
        let soff = (table_sp - vtable_sp) as i32;
        self.buf[table_sp..table_sp + 4].copy_from_slice(&soff.to_le_bytes());
        self.buf.len() - table_sp
    }

    fn finish_ipc_message(mut self, root_table_off: usize) -> Vec<u8> {
        // Align so `(sp - 4)` puts total FlatBuffer size on an 8-byte boundary.
        let used = self.pos_from_end() + 4;
        let pad = (8 - (used & 7)) & 7;
        for _ in 0..pad {
            self.push_u8(0);
        }
        self.push_uoffset_to(root_table_off);
        let meta = &self.buf[self.sp..];
        debug_assert_eq!(meta.len() % 8, 0);
        let mut out = Vec::with_capacity(8 + meta.len());
        out.extend_from_slice(&IPC_CONTINUATION.to_le_bytes());
        out.extend_from_slice(&(meta.len() as i32).to_le_bytes());
        out.extend_from_slice(meta);
        out
    }
}

fn encode_type_table(fb: &mut FbRev, dt: ArrowType) -> (u8, usize) {
    match dt {
        ArrowType::Boolean => {
            let end = fb.pos_from_end();
            (FB_TYPE_BOOL, fb.finish_table(end, &[]))
        }
        ArrowType::Utf8 => {
            let end = fb.pos_from_end();
            (FB_TYPE_UTF8, fb.finish_table(end, &[]))
        }
        ArrowType::LargeUtf8 => {
            let end = fb.pos_from_end();
            (FB_TYPE_LARGE_UTF8, fb.finish_table(end, &[]))
        }
        ArrowType::Int64 => {
            let end = fb.pos_from_end();
            fb.push_u8(1); // id 1: is_signed = true
            let f1 = fb.pos_from_end();
            fb.align(4);
            fb.push_i32(64); // id 0: bitWidth = 64
            let f0 = fb.pos_from_end();
            (FB_TYPE_INT, fb.finish_table(end, &[f0, f1]))
        }
        ArrowType::Float64 => {
            let end = fb.pos_from_end();
            fb.align(2);
            fb.push_i16(FB_FLOAT_DOUBLE); // id 0: precision = DOUBLE
            let f0 = fb.pos_from_end();
            (FB_TYPE_FLOAT, fb.finish_table(end, &[f0]))
        }
    }
}

fn build_schema_message(batch: &ArrowBatch) -> Vec<u8> {
    let mut fb = FbRev::with_capacity(512 + batch.columns.len() * 128);
    let mut field_offs = Vec::with_capacity(batch.columns.len());
    for col in &batch.columns {
        let name_off = fb.create_string(&col.name);
        let children_off = fb.create_offset_vec(&[]);
        let (tt, type_off) = encode_type_table(&mut fb, col.data.data_type());
        let end = fb.pos_from_end();
        fb.push_uoffset_to(children_off);
        let f5 = fb.pos_from_end();
        fb.push_uoffset_to(type_off);
        let f3 = fb.pos_from_end();
        fb.push_u8(tt);
        let f2 = fb.pos_from_end();
        fb.push_u8(1); // nullable = true
        let f1 = fb.pos_from_end();
        fb.push_uoffset_to(name_off);
        let f0 = fb.pos_from_end();
        field_offs.push(fb.finish_table(end, &[f0, f1, f2, f3, 0, f5]));
    }
    let fields_vec = fb.create_offset_vec(&field_offs);
    let s_end = fb.pos_from_end();
    fb.push_uoffset_to(fields_vec);
    let sf1 = fb.pos_from_end();
    fb.align(2);
    fb.push_i16(0); // Little-endian
    let sf0 = fb.pos_from_end();
    let schema_off = fb.finish_table(s_end, &[sf0, sf1]);

    let m_end = fb.pos_from_end();
    fb.align(8);
    fb.push_i64(0); // bodyLength = 0
    let mf3 = fb.pos_from_end();
    fb.push_uoffset_to(schema_off);
    let mf2 = fb.pos_from_end();
    fb.push_u8(MSG_HEADER_SCHEMA);
    let mf1 = fb.pos_from_end();
    fb.align(2);
    fb.push_i16(META_VERSION_V5);
    let mf0 = fb.pos_from_end();
    let msg_off = fb.finish_table(m_end, &[mf0, mf1, mf2, mf3]);
    fb.finish_ipc_message(msg_off)
}

#[inline]
fn append_padded8(body: &mut Vec<u8>, bytes: &[u8]) -> (i64, i64) {
    let off = body.len() as i64;
    let len = bytes.len() as i64;
    body.extend_from_slice(bytes);
    let pad = (8 - (body.len() & 7)) & 7;
    body.resize(body.len() + pad, 0);
    (off, len)
}

impl ArrowBatch {
    /// Serialize this [`ArrowBatch`] into a complete Apache Arrow IPC Stream (`V5`).
    pub fn to_ipc_stream(&self) -> Vec<u8> {
        let schema_msg = build_schema_message(self);
        let mut body = Vec::new();
        let mut nodes = Vec::with_capacity(self.columns.len());
        let mut bufs = Vec::with_capacity(self.columns.len() * 3);

        for col in &self.columns {
            nodes.push((self.num_rows as i64, col.null_count as i64));
            if let Some(ref bm) = col.validity {
                bufs.push(append_padded8(&mut body, bm));
            } else {
                bufs.push((body.len() as i64, 0));
            }
            match &col.data {
                ArrowData::Boolean(packed) => {
                    bufs.push(append_padded8(&mut body, packed));
                }
                ArrowData::Int64(vals) => {
                    let mut raw = Vec::with_capacity(vals.len() * 8);
                    for &v in vals {
                        raw.extend_from_slice(&v.to_le_bytes());
                    }
                    bufs.push(append_padded8(&mut body, &raw));
                }
                ArrowData::Float64(vals) => {
                    let mut raw = Vec::with_capacity(vals.len() * 8);
                    for &v in vals {
                        raw.extend_from_slice(&v.to_le_bytes());
                    }
                    bufs.push(append_padded8(&mut body, &raw));
                }
                ArrowData::Utf8 { offsets, values } => {
                    let mut off_b = Vec::with_capacity(offsets.len() * 4);
                    for &o in offsets {
                        off_b.extend_from_slice(&o.to_le_bytes());
                    }
                    bufs.push(append_padded8(&mut body, &off_b));
                    bufs.push(append_padded8(&mut body, values));
                }
                ArrowData::LargeUtf8 { offsets, values } => {
                    let mut off_b = Vec::with_capacity(offsets.len() * 8);
                    for &o in offsets {
                        off_b.extend_from_slice(&o.to_le_bytes());
                    }
                    bufs.push(append_padded8(&mut body, &off_b));
                    bufs.push(append_padded8(&mut body, values));
                }
            }
        }

        let mut fb = FbRev::with_capacity(512 + bufs.len() * 32);
        let bufs_vec = fb.create_i64_pair_struct_vec(&bufs);
        let nodes_vec = fb.create_i64_pair_struct_vec(&nodes);
        let rb_end = fb.pos_from_end();
        fb.push_uoffset_to(bufs_vec);
        let rbf2 = fb.pos_from_end();
        fb.push_uoffset_to(nodes_vec);
        let rbf1 = fb.pos_from_end();
        fb.align(8);
        fb.push_i64(self.num_rows as i64);
        let rbf0 = fb.pos_from_end();
        let rb_off = fb.finish_table(rb_end, &[rbf0, rbf1, rbf2]);

        let m_end = fb.pos_from_end();
        fb.align(8);
        fb.push_i64(body.len() as i64);
        let mf3 = fb.pos_from_end();
        fb.push_uoffset_to(rb_off);
        let mf2 = fb.pos_from_end();
        fb.push_u8(MSG_HEADER_RECORD_BATCH);
        let mf1 = fb.pos_from_end();
        fb.align(2);
        fb.push_i16(META_VERSION_V5);
        let mf0 = fb.pos_from_end();
        let msg_off = fb.finish_table(m_end, &[mf0, mf1, mf2, mf3]);
        let rb_msg = fb.finish_ipc_message(msg_off);

        let mut stream = Vec::with_capacity(schema_msg.len() + rb_msg.len() + body.len() + 8);
        stream.extend_from_slice(&schema_msg);
        stream.extend_from_slice(&rb_msg);
        stream.extend_from_slice(&body);
        stream.extend_from_slice(&IPC_CONTINUATION.to_le_bytes());
        stream.extend_from_slice(&0i32.to_le_bytes()); // EOS
        stream
    }

    /// Parse an [`ArrowBatch`] back from an Apache Arrow IPC Stream (`V5`).
    pub fn from_ipc_stream(src: &[u8]) -> Result<Self> {
        let mut pos = 0usize;
        let next_msg = |p: &mut usize| -> Result<Option<(&[u8], &[u8])>> {
            if *p == src.len() {
                return Ok(None);
            }
            if src.len() - *p < 8 {
                return Err(Error::InvalidFrame("truncated arrow ipc prefix"));
            }
            let first = u32::from_le_bytes(src[*p..*p + 4].try_into().unwrap());
            let meta_len = if first == IPC_CONTINUATION {
                let l = i32::from_le_bytes(src[*p + 4..*p + 8].try_into().unwrap());
                *p += 8;
                l
            } else {
                *p += 4;
                first as i32
            };
            if meta_len == 0 {
                return Ok(None);
            }
            if meta_len < 0 || src.len() - *p < meta_len as usize {
                return Err(Error::InvalidFrame("invalid arrow ipc metadata length"));
            }
            let meta = &src[*p..*p + meta_len as usize];
            *p += meta_len as usize;
            let root = fb_root(meta)?;
            let body_len = fb_i64(meta, root, 3)?.max(0) as usize;
            if src.len() - *p < body_len {
                return Err(Error::InvalidFrame("truncated arrow ipc body"));
            }
            let body = &src[*p..*p + body_len];
            *p += body_len;
            Ok(Some((meta, body)))
        };

        let Some((s_meta, _)) = next_msg(&mut pos)? else {
            return Err(Error::InvalidFrame("empty arrow ipc stream"));
        };
        let s_root = fb_root(s_meta)?;
        if fb_u8(s_meta, s_root, 1)? != MSG_HEADER_SCHEMA {
            return Err(Error::InvalidFrame("expected arrow Schema message first"));
        }
        let schema_tbl = fb_table(s_meta, s_root, 2)?;
        let fields_vec = fb_vec(s_meta, schema_tbl, 1)?;
        let n_fields = fb_u32_at(s_meta, fields_vec)? as usize;
        let mut col_defs = Vec::with_capacity(n_fields);
        for i in 0..n_fields {
            let slot = fields_vec + 4 + i * 4;
            let f_tbl = slot + fb_u32_at(s_meta, slot)? as usize;
            let name = fb_str(s_meta, f_tbl, 0)?;
            let tt = fb_u8(s_meta, f_tbl, 2)?;
            let dt = match tt {
                FB_TYPE_BOOL => ArrowType::Boolean,
                FB_TYPE_INT => ArrowType::Int64,
                FB_TYPE_FLOAT => ArrowType::Float64,
                FB_TYPE_UTF8 => ArrowType::Utf8,
                FB_TYPE_LARGE_UTF8 => ArrowType::LargeUtf8,
                _ => return Err(Error::InvalidFrame("unsupported arrow field type")),
            };
            col_defs.push((name, dt));
        }

        let Some((rb_meta, body)) = next_msg(&mut pos)? else {
            return Ok(Self {
                num_rows: 0,
                columns: Vec::new(),
            });
        };
        let rb_root = fb_root(rb_meta)?;
        if fb_u8(rb_meta, rb_root, 1)? != MSG_HEADER_RECORD_BATCH {
            return Err(Error::InvalidFrame("expected arrow RecordBatch message"));
        }
        let rb_tbl = fb_table(rb_meta, rb_root, 2)?;
        let num_rows = usize::try_from(fb_i64(rb_meta, rb_tbl, 0)?).map_err(|_| Error::SizeOverflow)?;
        let nodes_vec = fb_vec(rb_meta, rb_tbl, 1)?;
        let bufs_vec = fb_vec(rb_meta, rb_tbl, 2)?;
        if fb_u32_at(rb_meta, nodes_vec)? as usize != n_fields {
            return Err(Error::InvalidFrame("arrow FieldNode count mismatch"));
        }
        let n_bufs = fb_u32_at(rb_meta, bufs_vec)? as usize;
        let mut b_idx = 0usize;
        let read_buf = |bi: &mut usize| -> Result<&[u8]> {
            if *bi >= n_bufs {
                return Err(Error::InvalidFrame("arrow Buffer index overflow"));
            }
            let off_pos = bufs_vec + 4 + *bi * 16;
            *bi += 1;
            let b_off = usize::try_from(fb_i64_at(rb_meta, off_pos)?).map_err(|_| Error::SizeOverflow)?;
            let b_len = usize::try_from(fb_i64_at(rb_meta, off_pos + 8)?).map_err(|_| Error::SizeOverflow)?;
            body.get(b_off..b_off.saturating_add(b_len))
                .ok_or(Error::InvalidFrame("arrow body slice out of bounds"))
        };

        let mut columns = Vec::with_capacity(n_fields);
        for (i, (name, dt)) in col_defs.into_iter().enumerate() {
            let n_pos = nodes_vec + 4 + i * 16;
            let null_count = usize::try_from(fb_i64_at(rb_meta, n_pos + 8)?).map_err(|_| Error::SizeOverflow)?;
            let val_slice = read_buf(&mut b_idx)?;
            let validity = if null_count == 0 || val_slice.is_empty() {
                None
            } else {
                Some(val_slice.to_vec())
            };
            let data = match dt {
                ArrowType::Boolean => ArrowData::Boolean(read_buf(&mut b_idx)?.to_vec()),
                ArrowType::Int64 => {
                    let b = read_buf(&mut b_idx)?;
                    if b.len() != num_rows * 8 {
                        return Err(Error::InvalidFrame("arrow Int64 buffer size"));
                    }
                    let vals = b
                        .chunks_exact(8)
                        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                        .collect();
                    ArrowData::Int64(vals)
                }
                ArrowType::Float64 => {
                    let b = read_buf(&mut b_idx)?;
                    if b.len() != num_rows * 8 {
                        return Err(Error::InvalidFrame("arrow Float64 buffer size"));
                    }
                    let vals = b
                        .chunks_exact(8)
                        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                        .collect();
                    ArrowData::Float64(vals)
                }
                ArrowType::Utf8 => {
                    let ob = read_buf(&mut b_idx)?;
                    let vb = read_buf(&mut b_idx)?;
                    if ob.len() != (num_rows + 1) * 4 {
                        return Err(Error::InvalidFrame("arrow Utf8 offsets size"));
                    }
                    let offsets = ob
                        .chunks_exact(4)
                        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                        .collect();
                    ArrowData::Utf8 {
                        offsets,
                        values: vb.to_vec(),
                    }
                }
                ArrowType::LargeUtf8 => {
                    let ob = read_buf(&mut b_idx)?;
                    let vb = read_buf(&mut b_idx)?;
                    if ob.len() != (num_rows + 1) * 8 {
                        return Err(Error::InvalidFrame("arrow LargeUtf8 offsets size"));
                    }
                    let offsets = ob
                        .chunks_exact(8)
                        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                        .collect();
                    ArrowData::LargeUtf8 {
                        offsets,
                        values: vb.to_vec(),
                    }
                }
            };
            columns.push(ArrowCol {
                name,
                validity,
                null_count,
                data,
            });
        }
        Ok(Self { num_rows, columns })
    }
}

fn fb_u16_at(b: &[u8], p: usize) -> Result<u16> {
    let s = b.get(p..p + 2).ok_or(Error::InvalidFrame("fb u16 oob"))?;
    Ok(u16::from_le_bytes(s.try_into().unwrap()))
}
fn fb_u32_at(b: &[u8], p: usize) -> Result<u32> {
    let s = b.get(p..p + 4).ok_or(Error::InvalidFrame("fb u32 oob"))?;
    Ok(u32::from_le_bytes(s.try_into().unwrap()))
}
fn fb_i64_at(b: &[u8], p: usize) -> Result<i64> {
    let s = b.get(p..p + 8).ok_or(Error::InvalidFrame("fb i64 oob"))?;
    Ok(i64::from_le_bytes(s.try_into().unwrap()))
}
fn fb_root(b: &[u8]) -> Result<usize> {
    let r = fb_u32_at(b, 0)? as usize;
    if r >= b.len() {
        return Err(Error::InvalidFrame("fb root oob"));
    }
    Ok(r)
}
fn fb_slot(b: &[u8], tbl: usize, id: usize) -> Result<Option<usize>> {
    let soff = i32::from_le_bytes(
        b.get(tbl..tbl + 4)
            .ok_or(Error::InvalidFrame("fb tbl oob"))?
            .try_into()
            .unwrap(),
    );
    let vt = (tbl as isize)
        .checked_sub(soff as isize)
        .and_then(|x| usize::try_from(x).ok())
        .ok_or(Error::InvalidFrame("fb vtable oob"))?;
    let vlen = fb_u16_at(b, vt)? as usize;
    let entry = 4 + id * 2;
    if entry + 2 > vlen {
        return Ok(None);
    }
    let rel = fb_u16_at(b, vt + entry)? as usize;
    Ok((rel != 0).then_some(tbl + rel))
}
fn fb_u8(b: &[u8], tbl: usize, id: usize) -> Result<u8> {
    match fb_slot(b, tbl, id)? {
        Some(p) => b.get(p).copied().ok_or(Error::InvalidFrame("fb u8 oob")),
        None => Ok(0),
    }
}
fn fb_i64(b: &[u8], tbl: usize, id: usize) -> Result<i64> {
    match fb_slot(b, tbl, id)? {
        Some(p) => fb_i64_at(b, p),
        None => Ok(0),
    }
}
fn fb_table(b: &[u8], tbl: usize, id: usize) -> Result<usize> {
    let p = fb_slot(b, tbl, id)?.ok_or(Error::InvalidFrame("fb missing table"))?;
    Ok(p + fb_u32_at(b, p)? as usize)
}
fn fb_vec(b: &[u8], tbl: usize, id: usize) -> Result<usize> {
    fb_table(b, tbl, id)
}
fn fb_str(b: &[u8], tbl: usize, id: usize) -> Result<String> {
    let sp = fb_table(b, tbl, id)?;
    let len = fb_u32_at(b, sp)? as usize;
    let s = b
        .get(sp + 4..sp + 4 + len)
        .ok_or(Error::InvalidFrame("fb str oob"))?;
    Ok(String::from_utf8_lossy(s).into_owned())
}
