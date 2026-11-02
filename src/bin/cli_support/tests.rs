#[cfg(test)]
mod tests {
    use super::*;

    fn s(items: &[&str]) -> Vec<String> {
        items.iter().map(|x| (*x).to_owned()).collect()
    }

    #[test]
    fn cli_version_help_and_auto_threads_work() {
        assert!(run_args(&s(&["--version"])).is_ok());
        assert!(run_args(&s(&["--help"])).is_ok());
        assert!(parse_threads("auto").unwrap() >= 1);
        assert!(parse_threads("0").unwrap() >= 1);
        assert_eq!(parse_threads("4").unwrap(), 4);
    }

    #[test]
    fn cli_compress_verify_info_block_and_decompress_roundtrip() {
        let dir = std::env::temp_dir().join(format!("vxg_cli_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let raw_path = dir.join("input.csv");
        let vxg_path = dir.join("output.vxg");
        let dec_path = dir.join("decoded.csv");
        let blk_path = dir.join("block0.bin");

        let mut data = Vec::new();
        for i in 0..1024 {
            data.extend_from_slice(
                format!(
                    "\"2026-01-01T00:00:00Z\",{i},\"session.open\",\"eu-central-1\",\"worker-{}\"\n",
                    i % 8
                )
                .as_bytes(),
            );
        }
        fs::write(&raw_path, &data).unwrap();

        let raw_str = raw_path.to_str().unwrap();
        let vxg_str = vxg_path.to_str().unwrap();
        let dec_str = dec_path.to_str().unwrap();
        let blk_str = blk_path.to_str().unwrap();

        run_args(&s(&[
            "compress",
            raw_str,
            vxg_str,
            "--block-size",
            "4096",
            "--threads",
            "auto",
            "-q",
        ]))
        .unwrap();
        run_args(&s(&["verify", vxg_str, "--threads", "auto", "-q"])).unwrap();
        run_args(&s(&["info", vxg_str, "--summary"])).unwrap();
        run_args(&s(&["block", vxg_str, "0", blk_str, "-f"])).unwrap();
        assert_eq!(fs::read(&blk_path).unwrap(), &data[..4096]);

        run_args(&s(&["decompress", vxg_str, dec_str, "-f", "-q"])).unwrap();
        assert_eq!(fs::read(&dec_path).unwrap(), data);

        run_args(&s(&[
            "compress",
            raw_str,
            vxg_str,
            "--block-size",
            "4096",
            "--stream",
            "-f",
            "-q",
        ]))
        .unwrap();
        run_args(&s(&["verify", vxg_str, "-q"])).unwrap();
        run_args(&s(&[
            "decompress",
            vxg_str,
            dec_str,
            "--stream",
            "-f",
            "-q",
        ]))
        .unwrap();
        assert_eq!(fs::read(&dec_path).unwrap(), data);

        // Compressed-domain map -> update -> filter-frame -> verify -> decompress:
        let mapped_path = dir.join("mapped.vxg");
        let updated_path = dir.join("updated.vxg");
        let filtered_path = dir.join("filtered.vxg");
        let mapped_str = mapped_path.to_str().unwrap();
        let updated_str = updated_path.to_str().unwrap();
        let filtered_str = filtered_path.to_str().unwrap();

        run_args(&s(&["map", vxg_str, mapped_str, "--col", "csv:3", "--upper", "-q"])).unwrap();
        run_args(&s(&["verify", mapped_str, "-q"])).unwrap();
        run_args(&s(&[
            "update",
            mapped_str,
            updated_str,
            "--where",
            "csv:1>=1000",
            "--col",
            "csv:2",
            "--set",
            "session.tail",
            "-q",
        ]))
        .unwrap();
        run_args(&s(&["verify", updated_str, "-q"])).unwrap();
        run_args(&s(&[
            "filter-frame",
            updated_str,
            filtered_str,
            "--where",
            "csv:2=session.tail",
            "-q",
        ]))
        .unwrap();
        run_args(&s(&["verify", filtered_str, "-q"])).unwrap();
        assert_eq!(vxg::frame_row_count(&fs::read(&filtered_path).unwrap()).unwrap(), 24);

        let _ = fs::remove_dir_all(&dir);
    }
}
