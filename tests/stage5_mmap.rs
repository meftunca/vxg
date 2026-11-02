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
