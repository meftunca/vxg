//! Kani harnesses are deliberately narrow: two encoding arithmetic obligations.
//! They are NOT a machine-checked proof of the entire codec or native GPU drivers.
#[kani::proof]
fn inline_token_is_injective() {
    let op: u8 = kani::any();
    let lit: u8 = kani::any();
    let run: u8 = kani::any();
    kani::assume(op < 8);
    kani::assume(lit < 32);
    let word = crate::block::inline_token(op, lit, run);
    assert_eq!((word & 7) as u8, op);
    assert_eq!(((word >> 3) & 31) as u8, lit);
    assert_eq!((word >> 8) as u8, run);
}
#[kani::proof]
fn frame_bound_for_bounded_input_has_no_overflow() {
    let n: usize = kani::any();
    kani::assume(n <= (1usize << 30));
    let bound = crate::max_compressed_size(n, 65536).unwrap();
    let blocks = n / 65536 + usize::from(n % 65536 != 0);
    assert_eq!(bound, n + 32 + 32 * blocks);
    assert!(bound >= n);
}
