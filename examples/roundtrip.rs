fn main() -> vxg::Result<()> {
    let input = b"database-page\0\0\0\0".repeat(16_384);
    let options = vxg::Options { threads: 4, ..vxg::Options::default() };
    let (encoded, stats) = vxg::compress_with_stats(&input, &options)?;
    let decoded = vxg::decompress(&encoded, &vxg::DecodeOptions::default())?;
    assert_eq!(decoded, input);
    assert!(encoded.len() <= vxg::max_compressed_size(input.len(), options.block_size)?);
    println!("{} -> {} bytes; {:?}", input.len(), encoded.len(), stats);
    Ok(())
}
