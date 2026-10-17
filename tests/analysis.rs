#![cfg(feature = "bench")]

use vxg::{analysis, compress_with_stats, Frame, Options};

fn fixture(n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n);
    let mut index = 0;
    while out.len() < n {
        out.extend_from_slice(
            format!(
                "{{\"tenant\":{},\"event\":\"task.update\",\"id\":{index}}}\n",
                index % 17
            )
            .as_bytes(),
        );
        index += 1;
    }
    out.truncate(n);
    out
}

#[test]
fn command_analysis_matches_encoder_trace_totals() {
    let data = fixture(32_768);
    let options = Options {
        block_size: 4096,
        checksum: false,
        legacy_v2: true,
        ..Options::default()
    };
    let (encoded, stats) = compress_with_stats(&data, &options).unwrap();
    let frame = Frame::parse(&encoded, data.len()).unwrap();
    let analysis = analysis::analyze_frame(&frame).unwrap();
    assert_eq!(analysis.raw_bytes, data.len() as u64);
    assert_eq!(analysis.commands, stats.commands);
    assert_eq!(analysis.literal_bytes, stats.literal_bytes);
    assert_eq!(analysis.matched_bytes, stats.matched_bytes);
    assert_eq!(analysis.op_counts, stats.op_counts);
    assert_eq!(analysis.commands, analysis.op_counts.iter().sum());
    let literals = analysis::literal_blocks(&frame).unwrap();
    assert_eq!(
        literals.iter().map(Vec::len).sum::<usize>(),
        stats.literal_bytes as usize
    );
    assert!(analysis.to_json().contains("\"opcode_literal_run\""));
}

#[test]
fn command_analysis_keeps_raw_blocks_out_of_metadata() {
    let mut state = 0x4d595df4d0f33173u64;
    let data: Vec<u8> = (0..4096)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state as u8
        })
        .collect();
    let encoded = vxg::compress(
        &data,
        &Options {
            block_size: 4096,
            checksum: false,
            ..Options::default()
        },
    )
    .unwrap();
    let frame = Frame::parse(&encoded, data.len()).unwrap();
    let analysis = analysis::analyze_frame(&frame).unwrap();
    assert_eq!(analysis.raw_blocks, 1);
    assert_eq!(analysis.commands, 0);
    assert_eq!(analysis.match_commands, 0);
    assert_eq!(analysis::literal_blocks(&frame).unwrap(), [Vec::<u8>::new()]);
}
