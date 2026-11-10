use std::process::Command;
use vxg::{
    decompress, execute_sql, frame_row_count, recover_frame, verify_frame, DecodeOptions, Frame,
    Simd, SqlFormat, SqlQuery, BLOCK_COLUMN, BLOCK_DICT_LZ, BLOCK_LZ, BLOCK_RAW,
};

struct FrozenVector {
    name: &'static str,
    frame: &'static [u8],
    raw: &'static [u8],
    expected_version: u16,
    expected_kind: Option<u32>,
    expect_dict: bool,
}

const FROZEN_VECTORS: &[FrozenVector] = &[
    // Format v1
    FrozenVector {
        name: "v1-empty",
        frame: include_bytes!("vectors/empty.vxg"),
        raw: include_bytes!("vectors/empty.raw"),
        expected_version: 1,
        expected_kind: None,
        expect_dict: false,
    },
    FrozenVector {
        name: "v1-fill",
        frame: include_bytes!("vectors/fill.vxg"),
        raw: include_bytes!("vectors/fill.raw"),
        expected_version: 1,
        expected_kind: Some(BLOCK_LZ),
        expect_dict: false,
    },
    FrozenVector {
        name: "v1-all-ops",
        frame: include_bytes!("vectors/all-ops.vxg"),
        raw: include_bytes!("vectors/all-ops.raw"),
        expected_version: 1,
        expected_kind: Some(BLOCK_LZ),
        expect_dict: false,
    },
    FrozenVector {
        name: "v1-match-overlap",
        frame: include_bytes!("vectors/match-overlap.vxg"),
        raw: include_bytes!("vectors/match-overlap.raw"),
        expected_version: 1,
        expected_kind: Some(BLOCK_LZ),
        expect_dict: false,
    },
    FrozenVector {
        name: "v1-rep",
        frame: include_bytes!("vectors/rep.vxg"),
        raw: include_bytes!("vectors/rep.raw"),
        expected_version: 1,
        expected_kind: Some(BLOCK_LZ),
        expect_dict: false,
    },
    FrozenVector {
        name: "v1-multiblock",
        frame: include_bytes!("vectors/multiblock.vxg"),
        raw: include_bytes!("vectors/multiblock.raw"),
        expected_version: 1,
        expected_kind: None,
        expect_dict: false,
    },
    // Format v2 (opcode 7 short-distance LZ)
    FrozenVector {
        name: "v2-lz",
        frame: include_bytes!("vectors/v2-lz.vxg"),
        raw: include_bytes!("vectors/v2-lz.raw"),
        expected_version: 2,
        expected_kind: Some(BLOCK_LZ),
        expect_dict: false,
    },
    // Format v3 (32-byte directory + structured BLOCK_COLUMN)
    FrozenVector {
        name: "v3-column",
        frame: include_bytes!("vectors/v3-column.vxg"),
        raw: include_bytes!("vectors/v3-column.raw"),
        expected_version: 3,
        expected_kind: Some(BLOCK_COLUMN),
        expect_dict: false,
    },
    // Format v4 (compact 8-byte directory across RAW, LZ, COLUMN NDJSON/CSV, and BMP Stride)
    FrozenVector {
        name: "v4-raw",
        frame: include_bytes!("vectors/v4-raw.vxg"),
        raw: include_bytes!("vectors/v4-raw.raw"),
        expected_version: 4,
        expected_kind: Some(BLOCK_RAW),
        expect_dict: false,
    },
    FrozenVector {
        name: "v4-lz",
        frame: include_bytes!("vectors/v4-lz.vxg"),
        raw: include_bytes!("vectors/v4-lz.raw"),
        expected_version: 4,
        expected_kind: Some(BLOCK_LZ),
        expect_dict: false,
    },
    FrozenVector {
        name: "v4-column-ndjson",
        frame: include_bytes!("vectors/v4-column-ndjson.vxg"),
        raw: include_bytes!("vectors/v4-column-ndjson.raw"),
        expected_version: 4,
        expected_kind: Some(BLOCK_COLUMN),
        expect_dict: false,
    },
    FrozenVector {
        name: "v4-column-csv",
        frame: include_bytes!("vectors/v4-column-csv.vxg"),
        raw: include_bytes!("vectors/v4-column-csv.raw"),
        expected_version: 4,
        expected_kind: Some(BLOCK_COLUMN),
        expect_dict: false,
    },
    FrozenVector {
        name: "v4-stride-bmp",
        frame: include_bytes!("vectors/v4-stride-bmp.vxg"),
        raw: include_bytes!("vectors/v4-stride-bmp.raw"),
        expected_version: 4,
        expected_kind: Some(BLOCK_COLUMN),
        expect_dict: false,
    },
    // Format v5 (FLAG_FRAME_DICT + BLOCK_DICT_LZ, FLAG_VAR_RAW --record-aligned, and combined)
    FrozenVector {
        name: "v5-dict-lz",
        frame: include_bytes!("vectors/v5-dict-lz.vxg"),
        raw: include_bytes!("vectors/v5-dict-lz.raw"),
        expected_version: 5,
        expected_kind: Some(BLOCK_DICT_LZ),
        expect_dict: true,
    },
    FrozenVector {
        name: "v5-record-aligned",
        frame: include_bytes!("vectors/v5-record-aligned.vxg"),
        raw: include_bytes!("vectors/v5-record-aligned.raw"),
        expected_version: 5,
        expected_kind: Some(BLOCK_COLUMN),
        expect_dict: false,
    },
    FrozenVector {
        name: "v5-dict-and-var-raw",
        frame: include_bytes!("vectors/v5-dict-and-var-raw.vxg"),
        raw: include_bytes!("vectors/v5-dict-and-var-raw.raw"),
        expected_version: 5,
        expected_kind: Some(BLOCK_DICT_LZ),
        expect_dict: true,
    },
    FrozenVector {
        name: "v5-dict-lz-fse",
        frame: include_bytes!("vectors/v5-dict-lz-fse.vxg"),
        raw: include_bytes!("vectors/v5-dict-lz-fse.raw"),
        expected_version: 5,
        expected_kind: Some(BLOCK_DICT_LZ),
        expect_dict: true,
    },
    FrozenVector {
        name: "v5-row-index",
        frame: include_bytes!("vectors/v5-row-index.vxg"),
        raw: include_bytes!("vectors/v5-row-index.raw"),
        expected_version: 5,
        expected_kind: Some(BLOCK_COLUMN),
        expect_dict: false,
    },
];

#[test]
fn frozen_v1_to_v5_compatibility_vectors() {
    for vec_case in FROZEN_VECTORS {
        let parsed = Frame::parse(vec_case.frame, vec_case.raw.len() + 64)
            .unwrap_or_else(|e| panic!("failed to parse {}: {e}", vec_case.name));
        assert_eq!(
            parsed.version(),
            vec_case.expected_version,
            "version mismatch on {}",
            vec_case.name
        );
        assert_eq!(
            !parsed.dict().is_empty(),
            vec_case.expect_dict,
            "frame dictionary presence mismatch on {}",
            vec_case.name
        );
        if let Some(expected_kind) = vec_case.expected_kind {
            assert!(
                parsed.blocks().all(|b| b.kind == expected_kind),
                "unexpected block kind in {}",
                vec_case.name
            );
        }
        if vec_case.name == "v5-dict-lz-fse" {
            assert!(
                parsed
                    .blocks()
                    .any(|b| b.payload_size > 0 && (vec_case.frame[b.payload_offset] & 0x10) != 0),
                "v5-dict-lz-fse vector must exercise FLAG_FSE (0x10)"
            );
        }
        if vec_case.name == "v5-row-index" {
            assert!(parsed.has_row_index(), "v5-row-index must carry FLAG_ROW_INDEX (0x0008)");
            let expected_rows = vec_case.raw.split_inclusive(|&b| b == b'\n').count();
            assert_eq!(parsed.indexed_row_count(), Some(expected_rows));
            assert_eq!(frame_row_count(vec_case.frame).unwrap(), expected_rows);
        }
        parsed
            .validate_commands()
            .unwrap_or_else(|e| panic!("validate_commands failed on {}: {e}", vec_case.name));

        for (threads, simd) in [(1, Simd::Auto), (1, Simd::Scalar), (4, Simd::Auto)] {
            let opts = DecodeOptions {
                threads,
                simd,
                ..Default::default()
            };
            let out = decompress(vec_case.frame, &opts).unwrap_or_else(|e| {
                panic!(
                    "decompress failed on {} (t={threads}, simd={simd:?}): {e}",
                    vec_case.name
                )
            });
            assert_eq!(out, vec_case.raw, "byte mismatch on {}", vec_case.name);
            let verified_len = verify_frame(vec_case.frame, &opts)
                .unwrap_or_else(|e| panic!("verify_frame failed on {}: {e}", vec_case.name));
            assert_eq!(verified_len, vec_case.raw.len() as u64);
        }

        // Verify O(1) per-block random access on every frozen vector.
        let mut by_block = Vec::with_capacity(vec_case.raw.len());
        for idx in 0..parsed.block_count() {
            let blk = parsed
                .decode_block(idx, &DecodeOptions::default())
                .unwrap_or_else(|e| panic!("decode_block({idx}) failed on {}: {e}", vec_case.name));
            by_block.extend_from_slice(&blk);
        }
        assert_eq!(by_block, vec_case.raw, "random-access mismatch on {}", vec_case.name);
    }
}

#[test]
fn frozen_structured_vectors_preserve_sql_semantics() {
    let ndjson_vxg = include_bytes!("vectors/v4-column-ndjson.vxg");
    let q_nd = SqlQuery::parse(
        "SELECT region, COUNT(*) FROM frame WHERE status = 'ok' GROUP BY region ORDER BY region ASC",
    )
    .unwrap();
    let sql_nd = execute_sql(
        ndjson_vxg,
        &q_nd,
        SqlFormat::Csv,
        true,
        &DecodeOptions::default(),
    )
    .unwrap();
    let text_nd = String::from_utf8(sql_nd).unwrap();
    assert_eq!(text_nd.trim(), "eu-west-1,54");

    let aligned_vxg = include_bytes!("vectors/v5-record-aligned.vxg");
    let q_csv = SqlQuery::parse(
        "SELECT COUNT(*), SUM(trips) FROM frame WHERE region = 'eu-west-1'",
    )
    .unwrap();
    let sql_csv = execute_sql(
        aligned_vxg,
        &q_csv,
        SqlFormat::Csv,
        true,
        &DecodeOptions {
            threads: 2,
            ..Default::default()
        },
    )
    .unwrap();
    let text_csv = String::from_utf8(sql_csv).unwrap();
    assert_eq!(text_csv.trim(), "87,28623");
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

#[test]
fn partial_recovery_salvages_intact_blocks_from_corrupt_and_truncated_frames() {
    let orig_frame = include_bytes!("vectors/v5-dict-lz.vxg");
    let orig_raw = include_bytes!("vectors/v5-dict-lz.raw");
    let parsed = Frame::parse(orig_frame, 1 << 20).unwrap();
    assert_eq!(parsed.block_count(), 8);
    let blocks: Vec<_> = parsed.blocks().collect();

    let mut bad_mid = orig_frame.to_vec();
    let b3 = blocks[3];
    for p in b3.payload_offset..b3.payload_offset + b3.payload_size.min(32) {
        bad_mid[p] ^= 0xff;
    }
    assert!(
        decompress(&bad_mid, &DecodeOptions::default()).is_err(),
        "strict decompress must reject corrupted block 3"
    );

    let rep = recover_frame(&bad_mid, &DecodeOptions::default()).unwrap();
    assert_eq!(rep.total_blocks, 8);
    assert_eq!(rep.recovered_blocks.len(), 7);
    assert_eq!(rep.failed_blocks.len(), 1);
    assert_eq!(rep.failed_blocks[0].index, 3);

    let mut expected_salvaged = Vec::new();
    for (i, b) in blocks.iter().enumerate() {
        if i != 3 {
            expected_salvaged.extend_from_slice(&orig_raw[b.raw_offset..b.raw_offset + b.raw_size]);
        }
    }
    assert_eq!(rep.data, expected_salvaged);

    let al_frame = include_bytes!("vectors/v5-record-aligned.vxg");
    let al_raw = include_bytes!("vectors/v5-record-aligned.raw");
    let al_parsed = Frame::parse(al_frame, 1 << 20).unwrap();
    let al_blocks: Vec<_> = al_parsed.blocks().collect();
    assert_eq!(al_blocks.len(), 3);
    let cut_at = al_blocks[2].payload_offset + al_blocks[2].payload_size / 2;
    let truncated = &al_frame[..cut_at];

    assert!(decompress(truncated, &DecodeOptions::default()).is_err());
    let rep_trunc = recover_frame(truncated, &DecodeOptions::default()).unwrap();
    assert_eq!(rep_trunc.total_blocks, 3);
    assert_eq!(rep_trunc.recovered_blocks.len(), 2);
    assert_eq!(rep_trunc.failed_blocks.len(), 1);
    assert_eq!(rep_trunc.failed_blocks[0].index, 2);
    let prefix_len = al_blocks[0].raw_size + al_blocks[1].raw_size;
    assert_eq!(rep_trunc.data, &al_raw[..prefix_len]);

    let tmp = std::env::temp_dir().join(format!("vxg_recover_test_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&tmp);
    let bad_path = tmp.join("bad.vxg");
    let out_path = tmp.join("recovered.raw");
    std::fs::write(&bad_path, &bad_mid).unwrap();

    let vxg_bin = env!("CARGO_BIN_EXE_vxg");
    let strict_status = Command::new(vxg_bin)
        .args(["d", bad_path.to_str().unwrap(), out_path.to_str().unwrap(), "-f", "-q"])
        .status()
        .unwrap();
    assert!(!strict_status.success(), "strict CLI decompress must fail on bad.vxg");

    let rec_status = Command::new(vxg_bin)
        .args([
            "d",
            bad_path.to_str().unwrap(),
            out_path.to_str().unwrap(),
            "--recover",
            "-f",
            "-q",
        ])
        .status()
        .unwrap();
    assert!(rec_status.success(), "CLI --recover must succeed when intact blocks exist");
    assert_eq!(std::fs::read(&out_path).unwrap(), expected_salvaged);
    let _ = std::fs::remove_dir_all(&tmp);
}
