use std::fs;
use std::io::{self, Read};
use vxg::{
    compress, compress_file, decompress_file, mmap_supported, verify_file, verify_frame,
    verify_stream, DecodeOptions, Error, FrameReader, InputBytes, MappedFile, MappedFileMut,
    Options,
};

struct TinyChunkReader<'a> {
    data: &'a [u8],
    pos: usize,
    step: usize,
}

impl<'a> TinyChunkReader<'a> {
    fn new(data: &'a [u8], step: usize) -> Self {
        Self {
            data,
            pos: 0,
            step: step.max(1),
        }
    }
}

impl Read for TinyChunkReader<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.pos >= self.data.len() {
            return Ok(0);
        }
        let n = buf.len().min(self.step).min(self.data.len() - self.pos);
        buf[..n].copy_from_slice(&self.data[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }
}

#[test]
fn mmap_zero_byte_and_multiblock_roundtrip_and_cleanup() {
    let dir = std::env::temp_dir().join(format!("vxg_stage5_mmap_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();

    // 1. 0-byte file edge case:
    let empty_raw = dir.join("empty.bin");
    let empty_vxg = dir.join("empty.vxg");
    let empty_dec = dir.join("empty.out");
    fs::write(&empty_raw, b"").unwrap();

    if mmap_supported() {
        let m = unsafe { MappedFile::open(&empty_raw).unwrap() };
        assert!(m.is_empty());
        assert_eq!(&*m, b"");
        m.advise_sequential();
        m.advise_random();
        m.advise_willneed(0, 64);

        let mm_path = dir.join("empty_mut.bin");
        let mut mm = unsafe { MappedFileMut::create(&mm_path, 0, true).unwrap() };
        assert!(mm.is_empty());
        assert_eq!(&mut *mm, &mut [][..]);
        mm.flush().unwrap();
    }

    let (in0, out0) = compress_file(&empty_raw, &empty_vxg, &Options::default(), false).unwrap();
    assert_eq!(in0, 0);
    assert!(out0 >= 32);
    assert_eq!(
        verify_file(&empty_vxg, &DecodeOptions::default()).unwrap(),
        0
    );
    assert_eq!(
        decompress_file(&empty_vxg, &empty_dec, &DecodeOptions::default(), false).unwrap(),
        0
    );
    assert_eq!(fs::read(&empty_dec).unwrap(), b"");

    // 2. Multi-block structured NDJSON file (1T and 4T) + InputBytes modes:
    let big_raw = dir.join("logs.ndjson");
    let big_vxg = dir.join("logs.vxg");
    let big_dec = dir.join("logs.out");
    let mut payload = Vec::new();
    for i in 0..3000usize {
        payload.extend_from_slice(
            format!(
                "{{\"id\":{i},\"svc\":\"svc-{}\",\"lat\":{},\"ok\":true}}\n",
                i % 7,
                (i * 13) % 500
            )
            .as_bytes(),
        );
    }
    fs::write(&big_raw, &payload).unwrap();

    let ib_map = InputBytes::open_path(&big_raw).unwrap();
    assert_eq!(ib_map.is_mapped(), mmap_supported());
    assert_eq!(&*ib_map, payload.as_slice());
    let ib_own = InputBytes::open_path_mode(&big_raw, false).unwrap();
    assert!(!ib_own.is_mapped());
    assert_eq!(&*ib_own, payload.as_slice());

    let enc4 = Options {
        block_size: 4096,
        threads: 4,
        ..Options::default()
    };
    let dec4 = DecodeOptions {
        threads: 4,
        ..DecodeOptions::default()
    };
    let (raw_b, enc_b) = compress_file(&big_raw, &big_vxg, &enc4, false).unwrap();
    assert_eq!(raw_b, payload.len() as u64);
    assert!(enc_b < raw_b / 2);
    assert_eq!(verify_file(&big_vxg, &dec4).unwrap(), raw_b);

    // Overwrite protection (`force = false`) preserves existing file:
    let guard_path = dir.join("existing.out");
    fs::write(&guard_path, b"DO_NOT_CLOBBER").unwrap();
    assert!(decompress_file(&big_vxg, &guard_path, &dec4, false).is_err());
    assert_eq!(fs::read(&guard_path).unwrap(), b"DO_NOT_CLOBBER");

    assert_eq!(
        decompress_file(&big_vxg, &big_dec, &dec4, false).unwrap(),
        raw_b
    );
    assert_eq!(fs::read(&big_dec).unwrap(), payload);

    if mmap_supported() {
        let mut_out = dir.join("logs_mut_map.out");
        let in_map = unsafe { MappedFile::open(&big_vxg).unwrap() };
        let mut out_map = unsafe { MappedFileMut::create(&mut_out, payload.len(), true).unwrap() };
        vxg::decompress_into(&in_map, &mut out_map, &dec4).unwrap();
        out_map.flush().unwrap();
        drop(out_map);
        assert_eq!(fs::read(&mut_out).unwrap(), payload);
    }

    // 3. Corrupt frame cleanup removes newly created destination file:
    let mut corrupt = fs::read(&big_vxg).unwrap();
    let last = corrupt.len() - 1;
    corrupt[last] ^= 0xFF;
    let corrupt_vxg = dir.join("corrupt.vxg");
    let corrupt_out = dir.join("should_not_exist.out");
    fs::write(&corrupt_vxg, &corrupt).unwrap();
    assert!(verify_file(&corrupt_vxg, &dec4).is_err());
    assert!(decompress_file(&corrupt_vxg, &corrupt_out, &dec4, false).is_err());
    assert!(!corrupt_out.exists());

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn verify_frame_and_verify_stream_zero_alloc_and_corruption_checks() {
    // 1. High-entropy incompressible data -> BLOCK_RAW zero-copy CRC verification:
    let mut rng = 0x9E37_79B9_7F4A_7C15u64;
    let raw_incomp: Vec<u8> = (0..20_000)
        .map(|_| {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng as u8
        })
        .collect();
    let f_raw = compress(
        &raw_incomp,
        &Options {
            block_size: 4096,
            ..Options::default()
        },
    )
    .unwrap();
    for th in [1, 4] {
        let dec = DecodeOptions {
            threads: th,
            ..DecodeOptions::default()
        };
        assert_eq!(verify_frame(&f_raw, &dec).unwrap(), raw_incomp.len() as u64);
    }
    let mut bad_raw = f_raw.clone();
    let tail = bad_raw.len() - 17;
    bad_raw[tail] ^= 0x01;
    assert!(matches!(
        verify_frame(&bad_raw, &DecodeOptions::default()),
        Err(Error::Checksum { .. })
    ));

    // 2. Legacy v2 LZ + v4 Grammar multi-frame concatenated stream over tiny 73-byte chunks:
    let mut structured = Vec::new();
    for i in 0..800usize {
        structured.extend_from_slice(format!("colA,{i},val-{}\n", i % 5).as_bytes());
    }
    let f_lz = compress(
        &structured,
        &Options {
            block_size: 4096,
            legacy_v2: true,
            ..Options::default()
        },
    )
    .unwrap();
    let f_col = compress(
        &structured,
        &Options {
            block_size: 4096,
            ..Options::default()
        },
    )
    .unwrap();
    let mut concat = Vec::new();
    concat.extend_from_slice(&f_raw);
    concat.extend_from_slice(&f_lz);
    concat.extend_from_slice(&f_col);
    let expected_total = (raw_incomp.len() + structured.len() * 2) as u64;

    let dec_mt = DecodeOptions {
        threads: 4,
        max_output: usize::MAX,
        ..DecodeOptions::default()
    };
    assert_eq!(verify_frame(&concat, &dec_mt).unwrap(), expected_total);
    assert_eq!(
        verify_stream(TinyChunkReader::new(&concat, 73), &dec_mt).unwrap(),
        (expected_total, concat.len() as u64)
    );

    // 3. `max_output` guard on both `verify_frame` and `verify_stream`:
    let dec_lim = DecodeOptions {
        max_output: (expected_total as usize) - 1,
        ..DecodeOptions::default()
    };
    assert!(matches!(
        verify_frame(&concat, &dec_lim),
        Err(Error::OutputLimit { .. })
    ));
    assert!(matches!(
        verify_stream(TinyChunkReader::new(&concat, 73), &dec_lim),
        Err(Error::OutputLimit { .. })
    ));

    // 4. `FrameReader` rejects truncated streams and trailing garbage:
    let mut fr = FrameReader::new(
        TinyChunkReader::new(&concat[..concat.len() - 5], 61),
        usize::MAX,
    );
    assert!(fr.next_frame().unwrap().is_some());
    assert!(fr.next_frame().unwrap().is_some());
    assert_eq!(fr.next_frame(), Err(Error::Truncated));

    let mut garbage = f_col.clone();
    garbage.extend_from_slice(b"BAD!");
    let mut fr_bad = FrameReader::new(TinyChunkReader::new(&garbage, 31), usize::MAX);
    assert!(fr_bad.next_frame().unwrap().is_some());
    assert!(fr_bad.next_frame().is_err());
}

#[cfg(unix)]
fn write_cold_file(path: &std::path::Path, bytes: &[u8]) {
    use std::io::Write;
    use std::os::unix::io::AsRawFd;
    let mut f = fs::OpenOptions::new()
        .create(true)
        .truncate(true)
        .read(true)
        .write(true)
        .open(path)
        .unwrap();
    #[cfg(target_os = "macos")]
    unsafe {
        extern "C" {
            fn fcntl(fd: i32, cmd: i32, ...) -> i32;
        }
        const F_NOCACHE: i32 = 48;
        let _ = fcntl(f.as_raw_fd(), F_NOCACHE, 1);
    }
    f.write_all(bytes).unwrap();
    f.sync_all().unwrap();
    #[cfg(target_os = "linux")]
    unsafe {
        extern "C" {
            fn posix_fadvise(fd: i32, offset: i64, len: i64, advice: i32) -> i32;
        }
        const POSIX_FADV_DONTNEED: i32 = 4;
        let _ = posix_fadvise(f.as_raw_fd(), 0, 0, POSIX_FADV_DONTNEED);
    }
}

#[cfg(unix)]
fn fault_in_range(slice: &[u8], start: usize, end: usize) -> u8 {
    let mut acc = 0u8;
    let mut off = start;
    while off < end.min(slice.len()) {
        acc ^= unsafe { std::ptr::read_volatile(&slice[off]) };
        off += 4096;
    }
    if end > start && end <= slice.len() {
        acc ^= unsafe { std::ptr::read_volatile(&slice[end - 1]) };
    }
    std::hint::black_box(acc)
}

#[cfg(unix)]
#[test]
fn partial_cache_30_70_and_scattered_residency_and_single_mincore_per_query() {
    use vxg::{
        aggregate_column, count_where, execute_sql, legacy_four_page_probe, mincore_call_count,
        pread_bytes_count, pread_call_count, reset_mincore_call_count, reset_pread_counters,
        ChunkResidency, ColPred, ColTarget, NumCmp, PredExpr, SqlFormat, SqlQuery, CHUNK_BYTES,
    };

    let dir = std::env::temp_dir().join(format!("vxg_partial_cache_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();

    // Generate a >= 6.5 MiB compressed archive (7 x 1 MiB chunks) using incompressible + structured blocks
    // so both v5 and v6 frames span multiple 1 MiB chunks on disk.
    let mut raw = Vec::with_capacity(20 * 1024 * 1024);
    let mut state = 0x1234_5678_9abc_def0u64;
    for i in 0..180_000usize {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let p1 = state;
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let p2 = state;
        let line = format!(
            "{{\"id\":{i},\"grp\":{},\"amt\":{},\"blob\":\"{:016x}{:016x}{:016x}{:016x}\"}}\n",
            i % 8,
            (i % 100) + 1,
            p1,
            p2,
            p1 ^ p2,
            p1.rotate_left(19)
        );
        raw.extend_from_slice(line.as_bytes());
    }
    let v5_bytes = compress(
        &raw,
        &Options {
            block_size: 65_536,
            threads: 4,
            record_aligned: true,
            ..Options::default()
        },
    )
    .unwrap();
    let v6_bytes = compress(
        &raw,
        &Options {
            block_size: 65_536,
            threads: 4,
            record_aligned: true,
            col_group: true,
            ..Options::default()
        },
    )
    .unwrap();
    assert!(
        v5_bytes.len() >= 3 * CHUNK_BYTES && v6_bytes.len() >= 3 * CHUNK_BYTES,
        "test archive must span multiple 1 MiB chunks (v5={}, v6={})",
        v5_bytes.len(),
        v6_bytes.len()
    );

    let dec = DecodeOptions {
        threads: 4,
        max_output: usize::MAX,
        ..DecodeOptions::default()
    };
    let q = SqlQuery::parse("SELECT grp, COUNT(*), SUM(amt) FROM frame WHERE amt > 50 GROUP BY grp ORDER BY grp ASC").unwrap();
    let expected_sql = execute_sql(&v5_bytes, &q, SqlFormat::Csv, false, &dec).unwrap();

    for (label, frame_bytes) in [("v5", &v5_bytes), ("v6", &v6_bytes)] {
        let path = dir.join(format!("test_{label}.vxg"));
        let num_chunks = frame_bytes.len().div_ceil(CHUNK_BYTES);

        // 1. 30% hot chunks (fault in first 30% of 1 MiB chunks with MADV_RANDOM so OS readahead doesn't warm cold chunks):
        write_cold_file(&path, frame_bytes);
        let map30 = unsafe { MappedFile::open(&path).unwrap() };
        map30.advise_random();
        let hot30 = (num_chunks * 30 / 100).max(1);
        for c in 0..hot30 {
            fault_in_range(&map30, c * CHUNK_BYTES, (c + 1) * CHUNK_BYTES);
        }
        reset_mincore_call_count();
        let res30 = ChunkResidency::probe(&map30);
        assert_eq!(
            mincore_call_count(),
            1,
            "{label}: ChunkResidency::probe must issue exactly 1 mincore syscall"
        );
        assert!(
            res30.resident_chunks >= hot30 && !res30.all_resident(),
            "{label}: 30% hot file must have partial chunk residency ({}/{})",
            res30.resident_chunks,
            num_chunks
        );

        reset_mincore_call_count();
        reset_pread_counters();
        let out30 = execute_sql(&map30, &q, SqlFormat::Csv, false, &dec).unwrap();
        assert_eq!(out30, expected_sql);
        assert!(
            mincore_call_count() <= 1,
            "{label}: execute_sql must call mincore at most once per query (got {})",
            mincore_call_count()
        );
        if label == "v5" {
            // HybridArena must NOT pread the already-resident 30% chunks:
            let cold_bytes_expected = frame_bytes.len() - hot30 * CHUNK_BYTES;
            assert!(
                pread_bytes_count() <= cold_bytes_expected + 65_536 && pread_bytes_count() > 0,
                "v5 30% hot must pread only non-resident chunks (pread={}, expected<={})",
                pread_bytes_count(),
                cold_bytes_expected
            );
        } else {
            // v6 selective stream pread reads only Stream 0 + grp + amt streams (much less than total file):
            assert!(
                pread_bytes_count() < frame_bytes.len() / 2 && pread_call_count() > 0,
                "v6 30% hot must selectively pread only needed column streams (pread={})",
                pread_bytes_count()
            );
        }
        drop(map30);

        // 2. 70% hot chunks (fault in 70% of 1 MiB chunks with MADV_RANDOM):
        write_cold_file(&path, frame_bytes);
        let map70 = unsafe { MappedFile::open(&path).unwrap() };
        map70.advise_random();
        let hot70 = (num_chunks * 70 / 100).max(hot30 + 1).min(num_chunks - 1);
        for c in 0..hot70 {
            fault_in_range(&map70, c * CHUNK_BYTES, (c + 1) * CHUNK_BYTES);
        }
        reset_mincore_call_count();
        reset_pread_counters();
        let out70 = execute_sql(&map70, &q, SqlFormat::Csv, false, &dec).unwrap();
        assert_eq!(out70, expected_sql);
        assert!(mincore_call_count() <= 1);
        if label == "v5" {
            let cold_bytes_expected = frame_bytes.len() - hot70 * CHUNK_BYTES;
            assert!(
                pread_bytes_count() <= cold_bytes_expected + 65_536 && pread_bytes_count() > 0,
                "v5 70% hot must pread only remaining 30% non-resident chunks (pread={}, expected<={})",
                pread_bytes_count(),
                cold_bytes_expected
            );
        }
        drop(map70);

        // 3. Randomly scattered hot pages (including the 4 legacy probe points len/8, len/4, len/2, 3*len/4):
        //    Old 4-page probe falsely classifies the file as 100% warm (`true`) in 4 syscalls,
        //    whereas single-call `ChunkResidency::probe` detects that chunks have missing pages!
        write_cold_file(&path, frame_bytes);
        let map_scat = unsafe { MappedFile::open(&path).unwrap() };
        map_scat.advise_random();
        let len = map_scat.len();
        for &off in &[len / 8, len / 4, len / 2, (len * 3) / 4] {
            fault_in_range(&map_scat, off, off + 4096);
        }
        // Also fault in 1 page at the start of every 512 KiB window (~3% scattered pages across every 1 MiB chunk):
        let mut p_off = 0usize;
        while p_off < len {
            fault_in_range(&map_scat, p_off, p_off + 4096);
            p_off += 512 * 1024;
        }
        reset_mincore_call_count();
        let legacy_warm = legacy_four_page_probe(&map_scat);
        let legacy_calls = mincore_call_count();
        assert!(
            legacy_warm && legacy_calls == 4,
            "legacy 4-page probe is fooled by scattered hot pages into returning true with 4 syscalls"
        );

        reset_mincore_call_count();
        let scat_res = ChunkResidency::probe(&map_scat);
        assert_eq!(mincore_call_count(), 1);
        assert!(
            !scat_res.all_resident(),
            "ChunkResidency::probe detects that 1 MiB chunks have missing pages under scattered residency"
        );

        reset_mincore_call_count();
        reset_pread_counters();
        let out_scat = execute_sql(&map_scat, &q, SqlFormat::Csv, false, &dec).unwrap();
        assert_eq!(out_scat, expected_sql);
        assert!(mincore_call_count() <= 1);
        assert!(pread_call_count() > 0);

        // 4. Fully warm file: 1 mincore call, 0 pread calls:
        fault_in_range(&map_scat, 0, len);
        reset_mincore_call_count();
        reset_pread_counters();
        let pred = PredExpr::col(ColPred::json_num("amt", NumCmp::Gt(50)));
        let cnt = count_where(&map_scat, &pred, &dec).unwrap();
        assert!(cnt > 0);
        assert!(mincore_call_count() <= 1);
        assert_eq!(pread_call_count(), 0, "{label}: warm query must make 0 pread calls");

        reset_mincore_call_count();
        let _agg = aggregate_column(&map_scat, Some(&pred), ColTarget::JsonKey("amt"), &dec).unwrap();
        assert!(mincore_call_count() <= 1);
    }

    let _ = fs::remove_dir_all(&dir);
}
