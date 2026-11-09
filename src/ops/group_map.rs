/// Logical key produced by [`group_by`](super::group_by), keeping quoted JSON strings distinct from raw JSON scalars.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GroupKey {
    /// Unescaped string value (quoted JSON `"..."` string or CSV/TSV/PSV field).
    Str(Vec<u8>),
    /// Non-string JSON value token (e.g. `200`, `true`, `false`, `null`).
    RawJson(Vec<u8>),
}

impl GroupKey {
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            Self::Str(b) | Self::RawJson(b) => b,
        }
    }

    #[inline]
    pub fn as_str(&self) -> Option<&str> {
        std::str::from_utf8(self.as_bytes()).ok()
    }
}

/// Deterministic `GROUP BY` output row sorted by [`GroupKey`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroupRow {
    pub key: GroupKey,
    pub stats: AggStats,
}

#[derive(Default)]
pub(crate) struct FxHasher(u64);

impl Hasher for FxHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        const K: u64 = 0x517c_c1b7_2722_0a95;
        let mut h = self.0;
        let mut chunks = bytes.chunks_exact(8);
        for c in &mut chunks {
            let w = u64::from_le_bytes(c.try_into().unwrap());
            h = (h.rotate_left(5) ^ w).wrapping_mul(K);
        }
        let rem = chunks.remainder();
        if !rem.is_empty() {
            let mut tail = 0u64;
            for (i, &b) in rem.iter().enumerate() {
                tail |= u64::from(b) << (i * 8);
            }
            h = (h.rotate_left(5) ^ tail).wrapping_mul(K);
        }
        self.0 = h ^ (bytes.len() as u64);
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.0 = (self.0.rotate_left(5) ^ i).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
}

pub(crate) type FxBuildHasher = BuildHasherDefault<FxHasher>;

#[derive(Default)]
pub(crate) struct GroupMap {
    str_groups: HashMap<Vec<u8>, AggStats, FxBuildHasher>,
    raw_json_groups: HashMap<Vec<u8>, AggStats, FxBuildHasher>,
    u64_small_used: u64,
    u64_small: Option<Box<[AggStats; 64]>>,
    u64_overflow: HashMap<u64, AggStats, FxBuildHasher>,
    u64_is_str: bool,
}

#[inline]
fn fmt_u64_ascii(mut val: u64, buf: &mut [u8; 20]) -> &[u8] {
    let mut pos = 20usize;
    loop {
        pos -= 1;
        buf[pos] = b'0' + (val % 10) as u8;
        val /= 10;
        if val == 0 {
            break;
        }
    }
    &buf[pos..]
}

impl GroupMap {
    #[inline]
    pub(crate) fn merge_decoded(&mut self, is_str: bool, key: &[u8], delta: &AggStats) {
        if delta.row_count == 0 {
            return;
        }
        let map = if is_str {
            &mut self.str_groups
        } else {
            &mut self.raw_json_groups
        };
        if let Some(existing) = map.get_mut(key) {
            existing.merge(delta);
        } else {
            map.insert(key.to_vec(), *delta);
        }
    }

    #[inline]
    pub(crate) fn merge_decoded_unsigned_batch(
        &mut self,
        is_str: bool,
        key: &[u8],
        count: u64,
        sum: u128,
        min: u64,
        max: u64,
    ) {
        if count == 0 {
            return;
        }
        let map = if is_str {
            &mut self.str_groups
        } else {
            &mut self.raw_json_groups
        };
        if let Some(ex) = map.get_mut(key) {
            if !ex.has_signed_or_dec {
                ex.row_count += count;
                ex.num_count += count;
                ex.sum += sum;
                ex.min = Some(ex.min.map_or(min, |m| m.min(min)));
                ex.max = Some(ex.max.map_or(max, |m| m.max(max)));
            } else {
                ex.add_scaled_batch(sum as i128, min as i128, max as i128, 0, count);
            }
        } else {
            map.insert(
                key.to_vec(),
                AggStats {
                    row_count: count,
                    num_count: count,
                    sum,
                    min: Some(min),
                    max: Some(max),
                    ..AggStats::default()
                },
            );
        }
    }

    #[inline]
    pub(crate) fn merge_u64_key(&mut self, col: ColTarget<'_>, val: u64, delta: &AggStats) {
        if delta.row_count == 0 {
            return;
        }
        self.u64_is_str = matches!(col, ColTarget::Delimited { .. });
        if val < 64 {
            let arr = self
                .u64_small
                .get_or_insert_with(|| Box::new([AggStats::default(); 64]));
            self.u64_small_used |= 1u64 << val;
            arr[val as usize].merge(delta);
        } else {
            self.u64_overflow.entry(val).or_default().merge(delta);
        }
    }

    #[inline]
    pub(crate) fn merge_u64_unsigned_batch(
        &mut self,
        col: ColTarget<'_>,
        val: u64,
        count: u64,
        sum: u128,
        min: u64,
        max: u64,
    ) {
        if count == 0 {
            return;
        }
        self.u64_is_str = matches!(col, ColTarget::Delimited { .. });
        let dst = if val < 64 {
            let arr = self
                .u64_small
                .get_or_insert_with(|| Box::new([AggStats::default(); 64]));
            self.u64_small_used |= 1u64 << val;
            &mut arr[val as usize]
        } else {
            self.u64_overflow.entry(val).or_default()
        };
        if !dst.has_signed_or_dec {
            dst.row_count += count;
            dst.num_count += count;
            dst.sum += sum;
            dst.min = Some(dst.min.map_or(min, |m| m.min(min)));
            dst.max = Some(dst.max.map_or(max, |m| m.max(max)));
        } else {
            dst.add_scaled_batch(sum as i128, min as i128, max as i128, 0, count);
        }
    }

    #[inline]
    pub(crate) fn merge_u64_rows_only(&mut self, col: ColTarget<'_>, val: u64, count: u64) {
        if count == 0 {
            return;
        }
        self.u64_is_str = matches!(col, ColTarget::Delimited { .. });
        if val < 64 {
            let arr = self
                .u64_small
                .get_or_insert_with(|| Box::new([AggStats::default(); 64]));
            self.u64_small_used |= 1u64 << val;
            arr[val as usize].row_count += count;
        } else {
            self.u64_overflow.entry(val).or_default().row_count += count;
        }
    }

    #[inline]
    pub(crate) fn add_u64_num(&mut self, col: ColTarget<'_>, val: u64, num: u64) {
        self.u64_is_str = matches!(col, ColTarget::Delimited { .. });
        if val < 64 {
            let arr = self
                .u64_small
                .get_or_insert_with(|| Box::new([AggStats::default(); 64]));
            self.u64_small_used |= 1u64 << val;
            arr[val as usize].add_num(num);
        } else {
            self.u64_overflow.entry(val).or_default().add_num(num);
        }
    }

    #[inline]
    pub(crate) fn add_u64_dec(&mut self, col: ColTarget<'_>, val: u64, dec: super::dec_num::DecNum) {
        self.u64_is_str = matches!(col, ColTarget::Delimited { .. });
        if val < 64 {
            let arr = self
                .u64_small
                .get_or_insert_with(|| Box::new([AggStats::default(); 64]));
            self.u64_small_used |= 1u64 << val;
            arr[val as usize].add_dec(dec);
        } else {
            self.u64_overflow.entry(val).or_default().add_dec(dec);
        }
    }

    #[inline]
    pub(crate) fn add_u64_row_only(&mut self, col: ColTarget<'_>, val: u64) {
        self.u64_is_str = matches!(col, ColTarget::Delimited { .. });
        if val < 64 {
            let arr = self
                .u64_small
                .get_or_insert_with(|| Box::new([AggStats::default(); 64]));
            self.u64_small_used |= 1u64 << val;
            arr[val as usize].add_rows_only(1);
        } else {
            self.u64_overflow.entry(val).or_default().add_rows_only(1);
        }
    }

    pub(crate) fn merge_map(&mut self, other: Self) {
        if other.u64_small_used != 0 {
            self.u64_is_str = other.u64_is_str;
            let dst = self
                .u64_small
                .get_or_insert_with(|| Box::new([AggStats::default(); 64]));
            if let Some(src) = other.u64_small {
                let mut m = other.u64_small_used;
                while m != 0 {
                    let i = m.trailing_zeros() as usize;
                    dst[i].merge(&src[i]);
                    m &= m - 1;
                }
                self.u64_small_used |= other.u64_small_used;
            }
        }
        if !other.u64_overflow.is_empty() {
            self.u64_is_str = other.u64_is_str;
            for (k, st) in other.u64_overflow {
                self.u64_overflow.entry(k).or_default().merge(&st);
            }
        }
        for (k, st) in other.str_groups {
            if let Some(ex) = self.str_groups.get_mut(&k) {
                ex.merge(&st);
            } else {
                self.str_groups.insert(k, st);
            }
        }
        for (k, st) in other.raw_json_groups {
            if let Some(ex) = self.raw_json_groups.get_mut(&k) {
                ex.merge(&st);
            } else {
                self.raw_json_groups.insert(k, st);
            }
        }
    }

    fn flush_u64(&mut self) {
        let is_str = self.u64_is_str;
        let mut buf = [0u8; 20];
        if let Some(arr) = self.u64_small.take() {
            let mut m = self.u64_small_used;
            self.u64_small_used = 0;
            while m != 0 {
                let i = m.trailing_zeros() as usize;
                let s = fmt_u64_ascii(i as u64, &mut buf);
                self.merge_decoded(is_str, s, &arr[i]);
                m &= m - 1;
            }
        }
        let ov = std::mem::take(&mut self.u64_overflow);
        for (k, st) in ov {
            let s = fmt_u64_ascii(k, &mut buf);
            self.merge_decoded(is_str, s, &st);
        }
    }

    pub(crate) fn into_sorted_rows(mut self) -> Vec<GroupRow> {
        self.flush_u64();
        let mut out = Vec::with_capacity(self.str_groups.len() + self.raw_json_groups.len());
        for (k, stats) in self.str_groups {
            out.push(GroupRow {
                key: GroupKey::Str(k),
                stats,
            });
        }
        for (k, stats) in self.raw_json_groups {
            out.push(GroupRow {
                key: GroupKey::RawJson(k),
                stats,
            });
        }
        out.sort_unstable_by(|a, b| a.key.cmp(&b.key));
        out
    }
}
