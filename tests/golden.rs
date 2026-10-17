use vxg::{decompress, DecodeOptions, Frame};
#[test]
fn independent_model_golden_vectors() {
    let vectors: &[(&[u8], &[u8])] = &[
        (
            include_bytes!("vectors/empty.vxg"),
            include_bytes!("vectors/empty.raw"),
        ),
        (
            include_bytes!("vectors/fill.vxg"),
            include_bytes!("vectors/fill.raw"),
        ),
        (
            include_bytes!("vectors/all-ops.vxg"),
            include_bytes!("vectors/all-ops.raw"),
        ),
        (
            include_bytes!("vectors/match-overlap.vxg"),
            include_bytes!("vectors/match-overlap.raw"),
        ),
        (
            include_bytes!("vectors/rep.vxg"),
            include_bytes!("vectors/rep.raw"),
        ),
        (
            include_bytes!("vectors/multiblock.vxg"),
            include_bytes!("vectors/multiblock.raw"),
        ),
    ];
    for &(frame, expected) in vectors {
        Frame::parse(frame, expected.len())
            .unwrap()
            .validate_commands()
            .unwrap();
        assert_eq!(
            decompress(frame, &DecodeOptions::default()).unwrap(),
            expected
        );
    }
}
#[test]
fn invalid_backreferences_and_lengths_rejected() {
    for bytes in [
        include_bytes!("vectors/invalid-distance-zero.vxg").as_slice(),
        include_bytes!("vectors/invalid-distance-before-prefix.vxg").as_slice(),
        include_bytes!("vectors/invalid-output-overrun.vxg").as_slice(),
        include_bytes!("vectors/invalid-command-count.vxg").as_slice(),
    ] {
        assert!(decompress(bytes, &DecodeOptions::default()).is_err());
    }
}
