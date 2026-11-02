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
    pub(crate) fn merge_u64_key(&mut self, col: ColTarget<'_>, val: u64, delta: &AggStats) {
        if delta.row_count == 0 {
            return;
        }
        let mut buf = [0u8; 20];
        let mut pos = 20usize;
        let mut n = val;
        loop {
            pos -= 1;
            buf[pos] = b'0' + (n % 10) as u8;
            n /= 10;
            if n == 0 {
                break;
            }
        }
        let is_str = matches!(col, ColTarget::Delimited { .. });
        self.merge_decoded(is_str, &buf[pos..], delta);
    }

    pub(crate) fn merge_map(&mut self, other: Self) {
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

    pub(crate) fn into_sorted_rows(self) -> Vec<GroupRow> {
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
