thread_local! {
    static ENCODE_VEC_POOL: std::cell::RefCell<Vec<Vec<&'static [u8]>>> =
        std::cell::RefCell::new(Vec::with_capacity(128));
    static ENCODE_SEG_POOL: std::cell::RefCell<Vec<Vec<&'static [u8]>>> =
        std::cell::RefCell::new(Vec::with_capacity(16));
    static ENCODE_OUTER_POOL: std::cell::RefCell<Vec<Vec<Vec<&'static [u8]>>>> =
        std::cell::RefCell::new(Vec::with_capacity(16));
    static ENCODE_TPL_POOL: std::cell::RefCell<Vec<Vec<Template<'static>>>> =
        std::cell::RefCell::new(Vec::with_capacity(8));
    static COL_BUF_POOL: std::cell::RefCell<Vec<Vec<u8>>> =
        std::cell::RefCell::new(Vec::with_capacity(8));
}

#[inline(always)]
fn take_tpl_vec<'a>() -> Vec<Template<'a>> {
    match ENCODE_TPL_POOL.with(|cell| cell.borrow_mut().pop()) {
        Some(mut v) => {
            v.clear();
            unsafe { core::mem::transmute::<Vec<Template<'static>>, Vec<Template<'a>>>(v) }
        }
        None => Vec::with_capacity(8),
    }
}

#[inline(always)]
fn return_tpl_vec<'a>(mut v: Vec<Template<'a>>) {
    v.clear();
    if (4..=64).contains(&v.capacity()) {
        ENCODE_TPL_POOL.with(|cell| {
            if let Ok(mut pool) = cell.try_borrow_mut() {
                if pool.len() < 8 {
                    let vs = unsafe {
                        core::mem::transmute::<Vec<Template<'a>>, Vec<Template<'static>>>(v)
                    };
                    pool.push(vs);
                }
            }
        });
    }
}

#[inline(always)]
fn take_model_buf(cap: usize) -> Vec<u8> {
    match COL_BUF_POOL.with(|cell| cell.borrow_mut().pop()) {
        Some(mut buf) => {
            buf.clear();
            buf.reserve(cap);
            buf
        }
        None => Vec::with_capacity(cap.max(4096)),
    }
}

#[inline(always)]
fn put_model_buf(mut buf: Vec<u8>) {
    if (64..=131_072).contains(&buf.capacity()) {
        buf.clear();
        COL_BUF_POOL.with(|cell| {
            if let Ok(mut pool) = cell.try_borrow_mut() {
                if pool.len() < 8 {
                    pool.push(buf);
                }
            }
        });
    }
}

/// Smallest capacity across a template's column vectors.
///
/// Columns come from a pool and can have different capacities, so the fast row loops must bound
/// their unchecked writes by the minimum over *all* slots, not by slot 0 alone.
#[inline(always)]
fn min_col_capacity(values: &[Vec<&[u8]>]) -> usize {
    values.iter().map(Vec::capacity).min().unwrap_or(0)
}

#[inline(always)]
fn take_col_vec<'a>(est_rows: usize) -> Vec<&'a [u8]> {
    let v = ENCODE_VEC_POOL.with(|cell| cell.borrow_mut().pop());
    match v {
        Some(mut vec_static) => {
            vec_static.clear();
            vec_static.reserve(est_rows);
            unsafe { core::mem::transmute::<Vec<&'static [u8]>, Vec<&'a [u8]>>(vec_static) }
        }
        None => Vec::with_capacity(est_rows),
    }
}

#[inline(always)]
fn take_seg_vec<'a>(cap: usize) -> Vec<&'a [u8]> {
    match ENCODE_SEG_POOL.with(|cell| cell.borrow_mut().pop()) {
        Some(mut v) => {
            v.clear();
            v.reserve(cap);
            unsafe { core::mem::transmute::<Vec<&'static [u8]>, Vec<&'a [u8]>>(v) }
        }
        None => Vec::with_capacity(cap.max(32)),
    }
}

#[inline(always)]
fn take_outer_col_vec<'a>(slots: usize, est_rows: usize) -> Vec<Vec<&'a [u8]>> {
    let mut outer = match ENCODE_OUTER_POOL.with(|cell| cell.borrow_mut().pop()) {
        Some(mut v) => {
            v.clear();
            v.reserve(slots);
            unsafe { core::mem::transmute::<Vec<Vec<&'static [u8]>>, Vec<Vec<&'a [u8]>>>(v) }
        }
        None => Vec::with_capacity(slots.max(32)),
    };
    for _ in 0..slots {
        outer.push(take_col_vec(est_rows));
    }
    outer
}
