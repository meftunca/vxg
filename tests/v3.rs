//! Unified release (v4) integration tests: adaptive structured-column / stride /
//! fast-LZ blocks with the compact 8-byte directory through the public API.
//! Legacy v2 LZ-only frames stay available through `Options::legacy_v2`.

use vxg::{compress, compress_with_stats, decompress, DecodeOptions, Frame, Options};

fn structured(records: usize) -> Vec<u8> {
    let mut out = Vec::new();
    for row in 0..records {
        let epoch = row as u32 / 128;
        let prefix = 0x3b7f_2c4du32.wrapping_add(epoch.wrapping_mul(0x2545_f491));
        out.extend_from_slice(
            format!(
                "{{\"ts\":\"2026-01-{:02}T{:02}:00:00Z\",\"event\":\"session.open\",\"region\":\"eu-1\",\"request_id\":\"r-{prefix:08x}{:08x}\",\"status\":\"ok\",\"sequence\":{row}}}\n",
                row % 28 + 1,
                row % 24,
                row % 128,
            )
            .as_bytes(),
        );
    }
    out
}

fn encode_options(legacy_v2: bool) -> Options {
    Options {
        block_size: 65536,
        threads: 1,
        checksum: true,
        legacy_v2,
        ..Default::default()
    }
}

#[test]
fn default_options_emit_release_frames() {
    let data = structured(500);
    let frame = compress(&data, &Options::default()).unwrap();
    let parsed = Frame::parse(&frame, 1 << 30).unwrap();
    assert_eq!(parsed.version(), 4);
    assert_eq!(decompress(&frame, &DecodeOptions::default()).unwrap(), data);
}

#[test]
fn release_frames_carry_structured_blocks_and_roundtrip() {
    let data = structured(2000);
    let options = encode_options(false);
    let frame = compress(&data, &options).unwrap();
    let (stats_frame, _) = compress_with_stats(&data, &options).unwrap();
    assert_eq!(frame, stats_frame, "compress must equal compress_with_stats");
    assert_eq!(
        compress(&data, &options).unwrap(),
        frame,
        "encode must be deterministic"
    );
    let parsed = Frame::parse(&frame, 1 << 30).unwrap();
    assert_eq!(parsed.version(), 4);
    assert!(parsed.blocks().any(|b| b.kind == vxg::BLOCK_COLUMN));
    let v2 = compress(&data, &encode_options(true)).unwrap();
    assert_eq!(Frame::parse(&v2, 1 << 30).unwrap().version(), 2);
    assert!(
        frame.len() < v2.len(),
        "release {} should beat legacy v2 {}",
        frame.len(),
        v2.len()
    );
    assert!(
        frame.len() < data.len() / 10,
        "structured frame {} should be far below one tenth of {} input bytes",
        frame.len(),
        data.len()
    );
    assert_eq!(decompress(&frame, &DecodeOptions::default()).unwrap(), data);
    let dec = DecodeOptions::default();
    let mut rebuilt = Vec::new();
    for index in 0..parsed.block_count() {
        rebuilt.extend_from_slice(&parsed.decode_block(index, &dec).unwrap());
    }
    assert_eq!(rebuilt, data);
}

#[test]
fn compact_directory_is_eight_bytes_per_block() {
    let data = structured(4000);
    let frame = compress(&data, &encode_options(false)).unwrap();
    let parsed = Frame::parse(&frame, 1 << 30).unwrap();
    let count = parsed.block_count();
    assert!(count > 1);
    let first = parsed.blocks().next().unwrap();
    assert_eq!(first.payload_offset, 32 + 8 * count);
    // Offsets are implied and contiguous.
    let mut raw = 0;
    let mut pay = first.payload_offset;
    for b in parsed.blocks() {
        assert_eq!(b.raw_offset, raw);
        assert_eq!(b.payload_offset, pay);
        raw += b.raw_size;
        pay += b.payload_size;
    }
    assert_eq!(pay, frame.len());
}

#[test]
fn release_supports_multiple_workers_and_checksummed_decode() {
    let data = structured(1200);
    let options = Options {
        block_size: 16384,
        threads: 2,
        checksum: true,
        ..Default::default()
    };
    let frame = compress(&data, &options).unwrap();
    assert_eq!(frame, compress(&data, &Options { threads: 1, ..options.clone() }).unwrap());
    let dec = DecodeOptions {
        threads: 2,
        ..Default::default()
    };
    assert_eq!(decompress(&frame, &dec).unwrap(), data);
}

#[test]
fn validated_replay_accepts_column_and_lz_frames() {
    let options = encode_options(false);
    for data in [
        structured(200),
        (0..4096u32)
            .map(|i| (i.wrapping_mul(2654435761) >> 24) as u8)
            .collect::<Vec<u8>>(),
    ] {
        let frame = compress(&data, &options).unwrap();
        let validated = Frame::parse(&frame, 1 << 30)
            .unwrap()
            .into_validated()
            .unwrap();
        let mut out = vec![0u8; data.len()];
        validated
            .decode_into(&mut out, &DecodeOptions::default())
            .unwrap();
        assert_eq!(out, data);
    }
}

#[test]
fn release_large_blocks_roundtrip() {
    let data = structured(12000);
    for block_size in [131072usize, 262144, 1 << 20] {
        let options = Options {
            block_size,
            ..Default::default()
        };
        let frame = compress(&data, &options).unwrap();
        assert_eq!(decompress(&frame, &DecodeOptions::default()).unwrap(), data);
    }
}

#[test]
fn release_malformed_frames_are_rejected() {
    let data = structured(400);
    let options = encode_options(false);
    let frame = compress(&data, &options).unwrap();
    let parsed = Frame::parse(&frame, 1 << 30).unwrap();
    let column = parsed
        .blocks()
        .find(|b| b.kind == vxg::BLOCK_COLUMN)
        .unwrap();
    // Older versions must not reinterpret a compact directory.
    for v in [2u16, 3] {
        let mut downgraded = frame.clone();
        downgraded[4..6].copy_from_slice(&v.to_le_bytes());
        assert!(Frame::parse(&downgraded, 1 << 30).is_err());
    }
    // Unknown future versions are rejected.
    let mut future = frame.clone();
    future[4..6].copy_from_slice(&5u16.to_le_bytes());
    assert!(Frame::parse(&future, 1 << 30).is_err());
    // Unknown block kind (3) in the compact directory is rejected.
    let mut bad_kind = frame.clone();
    bad_kind[35] |= 0xc0;
    assert!(Frame::parse(&bad_kind, 1 << 30).is_err());
    // A corrupted structured payload fails decode.
    let mut corrupt = frame.clone();
    corrupt[column.payload_offset] ^= 0x40;
    assert!(decompress(&corrupt, &DecodeOptions::default()).is_err());
    // Truncation is rejected.
    assert!(decompress(&frame[..frame.len() - 1], &DecodeOptions::default()).is_err());
}
