//! Compressed-domain `ORDER BY` / `Top-K` heap execution (`top_k_by`, `select_top_k_by`).

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering as AtomicOrdering};

use crate::{query, DecodeOptions, Frame, NumCmp, Result, RowMatch, BLOCK_COLUMN};

use super::dec_num::{DecCmp, DecNum, U64DomainCmp};
use super::expr::{line_matches_expr, PredExpr};
use super::line::{
    decode_delim_str, find_line_col_span, parse_canonical_u64, scan_json_val_end,
    unescape_json_str, ColTarget,
};
use super::project_line::{project_line, validate_projection_cols};
use super::{build_par_meta_and_boundary, collect_par_blocks, SyncCell};

/// Sort direction for compressed-domain `ORDER BY` / `Top-K` queries.
/// Missing or `null` column values always sort last (`NULLS LAST`), and equal keys break ties
/// deterministically by smaller `global_row_index`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SortDir {
    Asc,
    Desc,
}

/// Extracted typed sort key for a structured column value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SortKey {
    /// Canonical unsigned 64-bit integer (`9 < 10 < 100`).
    Num(u64),
    /// Canonical signed integer or fixed-scale decimal (`-10 < -2.5 < 0 < 19.99`).
    Dec(DecNum),
    /// Logical string bytes (JSON-unescaped or CSV-decoded) or non-integer literal.
    Str(Vec<u8>),
    /// Column absent on record, empty delimited field, or JSON `null` (`NULLS LAST`).
    Missing,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SortKeyRef<'a> {
    Num(u64),
    Dec(DecNum),
    Str(&'a [u8]),
    Missing,
}

impl SortKey {
    #[inline]
    pub(crate) fn as_ref(&self) -> SortKeyRef<'_> {
        match self {
            Self::Num(n) => SortKeyRef::Num(*n),
            Self::Dec(d) => SortKeyRef::Dec(*d),
            Self::Str(b) => SortKeyRef::Str(b.as_slice()),
            Self::Missing => SortKeyRef::Missing,
        }
    }
}

impl SortKeyRef<'_> {
    #[inline]
    pub(crate) fn to_owned_key(self) -> SortKey {
        match self {
            Self::Num(n) => SortKey::Num(n),
            Self::Dec(d) => SortKey::Dec(d),
            Self::Str(b) => SortKey::Str(b.to_vec()),
            Self::Missing => SortKey::Missing,
        }
    }
}

/// Rank comparison: `Ordering::Less` means `(a, a_row)` ranks AHEAD of (is better than) `(b, b_row)`.
#[inline]
pub(crate) fn cmp_key_ref(
    a: SortKeyRef<'_>,
    a_row: usize,
    b: SortKeyRef<'_>,
    b_row: usize,
    dir: SortDir,
) -> Ordering {
    match (a, b) {
        (SortKeyRef::Missing, SortKeyRef::Missing) => a_row.cmp(&b_row),
        (SortKeyRef::Missing, _) => Ordering::Greater,
        (_, SortKeyRef::Missing) => Ordering::Less,
        _ => {
            let base = match (a, b) {
                (SortKeyRef::Num(x), SortKeyRef::Num(y)) => x.cmp(&y),
                (SortKeyRef::Dec(x), SortKeyRef::Dec(y)) => x.cmp(&y),
                (SortKeyRef::Num(x), SortKeyRef::Dec(y)) => DecNum::from_u64(x).cmp(&y),
                (SortKeyRef::Dec(x), SortKeyRef::Num(y)) => x.cmp(&DecNum::from_u64(y)),
                (SortKeyRef::Str(x), SortKeyRef::Str(y)) => x.cmp(y),
                (SortKeyRef::Num(_) | SortKeyRef::Dec(_), SortKeyRef::Str(_)) => Ordering::Less,
                (SortKeyRef::Str(_), SortKeyRef::Num(_) | SortKeyRef::Dec(_)) => Ordering::Greater,
                _ => unreachable!(),
            };
            if base == Ordering::Equal {
                a_row.cmp(&b_row)
            } else if dir == SortDir::Asc {
                base
            } else {
                base.reverse()
            }
        }
    }
}

/// Extract a [`SortKey`] from a raw column wire token `raw_tok`.
#[inline]
pub(crate) fn extract_token_sort_key(col: ColTarget<'_>, raw_tok: &[u8]) -> SortKey {
    if raw_tok.is_empty() || raw_tok.contains(&b'\n') || raw_tok.contains(&b'\r') {
        return SortKey::Missing;
    }
    match col {
        ColTarget::JsonKey(_) => {
            if raw_tok == b"null" {
                SortKey::Missing
            } else if let Some(v) = parse_canonical_u64(raw_tok) {
                SortKey::Num(v)
            } else if let Some(d) = DecNum::parse(raw_tok) {
                SortKey::Dec(d)
            } else if raw_tok.first() == Some(&b'"') {
                match unescape_json_str(raw_tok) {
                    Some(s) => SortKey::Str(s.into_owned()),
                    None => SortKey::Missing,
                }
            } else if scan_json_val_end(raw_tok, 0) == Some(raw_tok.len()) {
                SortKey::Str(raw_tok.to_vec())
            } else {
                SortKey::Missing
            }
        }
        ColTarget::Delimited { delim, .. } => {
            if !delim.is_ascii_digit() {
                if let Some(v) = parse_canonical_u64(raw_tok) {
                    return SortKey::Num(v);
                }
                if let Some(d) = DecNum::parse(raw_tok) {
                    return SortKey::Dec(d);
                }
            }
            match decode_delim_str(raw_tok) {
                Some(s) => SortKey::Str(s.into_owned()),
                None => SortKey::Missing,
            }
        }
    }
}

/// Zero-alloc borrow variant of [`extract_token_sort_key`] when no JSON/CSV unescaping is required.
#[inline]
pub(crate) fn extract_token_sort_key_ref<'a>(
    col: ColTarget<'_>,
    raw_tok: &'a [u8],
) -> Option<SortKeyRef<'a>> {
    if raw_tok.is_empty() || raw_tok.contains(&b'\n') || raw_tok.contains(&b'\r') {
        return Some(SortKeyRef::Missing);
    }
    match col {
        ColTarget::JsonKey(_) => {
            if raw_tok == b"null" {
                Some(SortKeyRef::Missing)
            } else if let Some(v) = parse_canonical_u64(raw_tok) {
                Some(SortKeyRef::Num(v))
            } else if let Some(d) = DecNum::parse(raw_tok) {
                Some(SortKeyRef::Dec(d))
            } else if raw_tok.first() == Some(&b'"') {
                if raw_tok.len() >= 2
                    && raw_tok.last() == Some(&b'"')
                    && !raw_tok[1..raw_tok.len() - 1]
                        .iter()
                        .any(|&b| b == b'\\' || b == b'"' || b < 0x20)
                {
                    Some(SortKeyRef::Str(&raw_tok[1..raw_tok.len() - 1]))
                } else {
                    None
                }
            } else if scan_json_val_end(raw_tok, 0) == Some(raw_tok.len()) {
                Some(SortKeyRef::Str(raw_tok))
            } else {
                Some(SortKeyRef::Missing)
            }
        }
        ColTarget::Delimited { delim, .. } => {
            if !delim.is_ascii_digit() {
                if let Some(v) = parse_canonical_u64(raw_tok) {
                    return Some(SortKeyRef::Num(v));
                }
                if let Some(d) = DecNum::parse(raw_tok) {
                    return Some(SortKeyRef::Dec(d));
                }
            }
            if raw_tok.len() >= 2 && raw_tok.first() == Some(&b'"') && raw_tok.last() == Some(&b'"')
            {
                None
            } else {
                Some(SortKeyRef::Str(raw_tok))
            }
        }
    }
}

/// Extract the [`SortKey`] for `col` from a single uncompressed record `line`.
#[inline]
pub fn extract_line_sort_key(line: &[u8], col: ColTarget<'_>) -> SortKey {
    let Some((s, e)) = find_line_col_span(line, col) else {
        return SortKey::Missing;
    };
    extract_token_sort_key(col, &line[s..e])
}

#[derive(Clone, Debug)]
pub(crate) struct HeapEntry {
    pub(crate) dir: SortDir,
    pub(crate) key: SortKey,
    pub(crate) block_index: usize,
    pub(crate) row_index: usize,
    pub(crate) global_row_index: usize,
    pub(crate) bytes: Vec<u8>,
}

impl PartialEq for HeapEntry {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        cmp_key_ref(
            self.key.as_ref(),
            self.global_row_index,
            other.key.as_ref(),
            other.global_row_index,
            self.dir,
        )
    }
}

pub(crate) struct TopKHeap {
    k: usize,
    dir: SortDir,
    skip_row_0: bool,
    floor: Option<(SortKey, usize)>,
    heap: BinaryHeap<HeapEntry>,
}

impl TopKHeap {
    pub(crate) fn new(k: usize, dir: SortDir) -> Self {
        Self {
            k,
            dir,
            skip_row_0: false,
            floor: None,
            heap: BinaryHeap::with_capacity(k.min(1024)),
        }
    }

    pub(crate) fn with_floor(
        k: usize,
        dir: SortDir,
        skip_row_0: bool,
        floor: Option<(SortKey, usize)>,
    ) -> Self {
        Self {
            k,
            dir,
            skip_row_0,
            floor,
            heap: BinaryHeap::with_capacity(k.min(1024)),
        }
    }

    #[inline]
    pub(crate) fn set_skip_row_0(&mut self, skip: bool) {
        self.skip_row_0 = skip;
    }

    #[inline]
    pub(crate) fn skip_row_0(&self) -> bool {
        self.skip_row_0
    }

    #[inline]
    pub(crate) fn capacity_k(&self) -> usize {
        self.k
    }

    #[inline]
    pub(crate) fn dir(&self) -> SortDir {
        self.dir
    }

    #[inline]
    pub(crate) fn worst_floor(&self) -> Option<(SortKey, usize)> {
        if self.k > 0 && self.heap.len() == self.k {
            let w = self.heap.peek().unwrap();
            Some((w.key.clone(), w.global_row_index))
        } else {
            self.floor.clone()
        }
    }

    #[inline]
    pub(crate) fn can_accept_key(&self, key: SortKeyRef<'_>, global_row: usize) -> bool {
        if self.k == 0 || (self.skip_row_0 && global_row == 0) {
            return false;
        }
        self.can_accept_key_at(key, global_row)
    }

    #[inline]
    pub(crate) fn can_accept_key_from(&self, key: SortKeyRef<'_>, min_global_row: usize) -> bool {
        if self.k == 0 {
            return false;
        }
        self.can_accept_key_at(key, min_global_row.max(usize::from(self.skip_row_0)))
    }

    #[inline]
    fn can_accept_key_at(&self, key: SortKeyRef<'_>, global_row: usize) -> bool {
        if let Some((ref fk, f_row)) = self.floor {
            if cmp_key_ref(key, global_row, fk.as_ref(), f_row, self.dir) != Ordering::Less {
                return false;
            }
        }
        if self.heap.len() < self.k {
            return true;
        }
        let worst = self.heap.peek().unwrap();
        cmp_key_ref(
            key,
            global_row,
            worst.key.as_ref(),
            worst.global_row_index,
            self.dir,
        ) == Ordering::Less
    }

    #[inline]
    pub(crate) fn push_entry(&mut self, entry: HeapEntry) {
        if self.k == 0 || (self.skip_row_0 && entry.global_row_index == 0) {
            return;
        }
        if let Some((ref fk, f_row)) = self.floor {
            if cmp_key_ref(
                entry.key.as_ref(),
                entry.global_row_index,
                fk.as_ref(),
                f_row,
                self.dir,
            ) != Ordering::Less
            {
                return;
            }
        }
        if self.heap.len() < self.k {
            self.heap.push(entry);
        } else if let Some(mut worst) = self.heap.peek_mut() {
            if entry < *worst {
                *worst = entry;
            }
        }
    }

    #[inline]
    fn active_worst(&self) -> Option<(&SortKey, usize)> {
        if self.k > 0 && self.heap.len() == self.k {
            let w = self.heap.peek().unwrap();
            Some((&w.key, w.global_row_index))
        } else {
            self.floor.as_ref().map(|(k, r)| (k, *r))
        }
    }

    #[inline]
    pub(crate) fn worst_num_bound(&self) -> Option<(u64, usize)> {
        let (k, r) = self.active_worst()?;
        match *k {
            SortKey::Num(b) => Some((b, r)),
            _ => None,
        }
    }

    #[inline]
    pub(crate) fn update_shared_bound(&self, shared: Option<&AtomicU64>) {
        if let (Some(sh), Some((b, _))) = (shared, self.worst_num_bound()) {
            match self.dir {
                SortDir::Asc => {
                    sh.fetch_min(b, AtomicOrdering::Relaxed);
                }
                SortDir::Desc => {
                    sh.fetch_max(b, AtomicOrdering::Relaxed);
                }
            }
        }
    }

    /// Returns `None` if NO `SortKey::Num(v)` at `global_row >= min_global_row` can enter `self`,
    /// or `Some(NumCmp)` that any candidate `Num(v)` must satisfy.
    #[inline]
    pub(crate) fn num_accept_cmp(
        &self,
        shared: Option<&AtomicU64>,
        min_global_row: usize,
    ) -> Option<NumCmp> {
        if self.k == 0 {
            return None;
        }
        let min_global_row = min_global_row.max(usize::from(self.skip_row_0));
        if self.dir == SortDir::Desc {
            let mut lo = shared.map_or(0, |a| a.load(AtomicOrdering::Relaxed));
            match self.active_worst() {
                Some((SortKey::Str(_), _)) => return None,
                Some((&SortKey::Num(b), w_row)) => {
                    if min_global_row > w_row {
                        if b == u64::MAX {
                            return None;
                        }
                        lo = lo.max(b + 1);
                    } else {
                        lo = lo.max(b);
                    }
                }
                Some((&SortKey::Dec(d), w_row)) => {
                    let dc = if min_global_row > w_row {
                        DecCmp::Gt(d)
                    } else {
                        DecCmp::Ge(d)
                    };
                    match dc.to_u64_domain() {
                        U64DomainCmp::AlwaysFalse => return None,
                        U64DomainCmp::AlwaysTrue => {}
                        U64DomainCmp::Cmp(NumCmp::Ge(b)) => lo = lo.max(b),
                        _ => {}
                    }
                }
                _ => {}
            }
            Some(NumCmp::Ge(lo))
        } else {
            let mut hi = shared.map_or(u64::MAX, |a| a.load(AtomicOrdering::Relaxed));
            match self.active_worst() {
                Some((&SortKey::Num(b), w_row)) => {
                    if min_global_row > w_row {
                        if b == 0 {
                            return None;
                        }
                        hi = hi.min(b - 1);
                    } else {
                        hi = hi.min(b);
                    }
                }
                Some((&SortKey::Dec(d), w_row)) => {
                    let dc = if min_global_row > w_row {
                        DecCmp::Lt(d)
                    } else {
                        DecCmp::Le(d)
                    };
                    match dc.to_u64_domain() {
                        U64DomainCmp::AlwaysFalse => return None,
                        U64DomainCmp::AlwaysTrue => {}
                        U64DomainCmp::Cmp(NumCmp::Le(b)) => hi = hi.min(b),
                        _ => {}
                    }
                }
                _ => {}
            }
            Some(NumCmp::Le(hi))
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn push_line_if_match(
        &mut self,
        line: &[u8],
        block_index: usize,
        row_index: usize,
        global_row_index: usize,
        where_expr: Option<&PredExpr<'_>>,
        project_cols: Option<&[ColTarget<'_>]>,
        order_col: ColTarget<'_>,
        shared: Option<&AtomicU64>,
    ) {
        if let Some(expr) = where_expr {
            if !line_matches_expr(line, expr) {
                return;
            }
        }
        let key = extract_line_sort_key(line, order_col);
        if !self.can_accept_key(key.as_ref(), global_row_index) {
            return;
        }
        let bytes = if let Some(cols) = project_cols {
            let mut proj = Vec::new();
            if !project_line(line, cols, &mut proj) {
                return;
            }
            proj
        } else {
            line.to_vec()
        };
        self.push_entry(HeapEntry {
            dir: self.dir,
            key,
            block_index,
            row_index,
            global_row_index,
            bytes,
        });
        self.update_shared_bound(shared);
    }

    pub(crate) fn merge_from(&mut self, other: TopKHeap) {
        for entry in other.heap {
            self.push_entry(entry);
        }
    }

    pub(crate) fn into_sorted_matches(self) -> Vec<RowMatch> {
        self.heap
            .into_sorted_vec()
            .into_iter()
            .map(|e| RowMatch {
                block_index: e.block_index,
                row_index: e.row_index,
                global_row_index: e.global_row_index,
                bytes: e.bytes,
            })
            .collect()
    }

    pub(crate) fn into_concat_bytes(self) -> Vec<u8> {
        let sorted = self.heap.into_sorted_vec();
        let total_len: usize = sorted.iter().map(|e| e.bytes.len()).sum();
        let mut out = Vec::with_capacity(total_len);
        for e in sorted {
            out.extend_from_slice(&e.bytes);
        }
        out
    }
}

include!("top_k_scan.rs");
