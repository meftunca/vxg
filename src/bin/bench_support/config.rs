use vxg::{gpu::Backend, Simd};

use super::R;

#[derive(Clone)]
pub(super) struct Config {
    pub(super) size: usize,
    pub(super) seconds: f64,
    pub(super) samples: usize,
    pub(super) threads: usize,
    pub(super) block_size: usize,
    pub(super) simd: Simd,
    pub(super) csv: String,
    pub(super) samples_csv: Option<String>,
    pub(super) markdown: Option<String>,
    pub(super) suite: String,
    pub(super) files: Vec<String>,
    pub(super) only: Option<String>,
    pub(super) gpu: Option<Backend>,
    pub(super) microbench: bool,
    pub(super) vxg_only: bool,
    pub(super) fast_baselines: bool,
    pub(super) decode_only: bool,
    pub(super) validated_decode: bool,
    pub(super) legacy_v2: bool,
    pub(super) level: u8,
}

#[derive(Default, Clone, Copy, Debug)]
pub(super) struct CodecPoint {
    pub(super) ratio: f64,
    pub(super) enc: f64,
    pub(super) dec: f64,
    pub(super) enc_j_gb: Option<f64>,
    pub(super) dec_j_gb: Option<f64>,
}

impl CodecPoint {
    pub(super) fn envelope(points: &[Self]) -> Self {
        let mut ratio = f64::INFINITY;
        let mut enc = 0.0f64;
        let mut dec = 0.0f64;
        let mut enc_j: Option<f64> = None;
        let mut dec_j: Option<f64> = None;
        for p in points {
            if p.ratio > 0.0 && p.ratio < ratio {
                ratio = p.ratio;
            }
            if p.enc > enc {
                enc = p.enc;
            }
            if p.dec > dec {
                dec = p.dec;
            }
            if let Some(ej) = p.enc_j_gb {
                if ej > 0.0 {
                    enc_j = Some(enc_j.map_or(ej, |cur| cur.min(ej)));
                }
            }
            if let Some(dj) = p.dec_j_gb {
                if dj > 0.0 {
                    dec_j = Some(dec_j.map_or(dj, |cur| cur.min(dj)));
                }
            }
        }
        Self {
            ratio: if ratio.is_finite() { ratio } else { 0.0 },
            enc,
            dec,
            enc_j_gb: enc_j,
            dec_j_gb: dec_j,
        }
    }
}

#[derive(Default, Clone)]
pub(super) struct DatasetSummary {
    pub(super) name: String,
    pub(super) raw_bytes: usize,
    pub(super) vxg: CodecPoint,
    pub(super) lz4_a1: CodecPoint,
    pub(super) lz4_a4: CodecPoint,
    pub(super) lz4hc_9: CodecPoint,
    pub(super) zstd_m3: CodecPoint,
    pub(super) zstd_1: CodecPoint,
    pub(super) zstd_3: CodecPoint,
    pub(super) zstd_9: CodecPoint,
    pub(super) zstd_dict_3: CodecPoint,
    pub(super) zstd_dict_9: CodecPoint,
    pub(super) zstd_whole_19: CodecPoint,
    pub(super) dict_bytes: usize,
    pub(super) dict_l3_bytes: usize,
    pub(super) dict_l9_bytes: usize,
    pub(super) dict_train_ms: f64,
}

impl DatasetSummary {
    pub(super) fn is_bmp_2d(&self) -> bool {
        self.name.ends_with(".bmp")
    }

    pub(super) fn lz4_envelope(&self) -> CodecPoint {
        CodecPoint::envelope(&[self.lz4_a1, self.lz4_a4, self.lz4hc_9])
    }

    pub(super) fn zstd_block_envelope(&self) -> CodecPoint {
        CodecPoint::envelope(&[
            self.zstd_m3,
            self.zstd_1,
            self.zstd_3,
            self.zstd_9,
            self.zstd_dict_3,
            self.zstd_dict_9,
        ])
    }

    pub(super) fn all_block_envelope(&self) -> CodecPoint {
        CodecPoint::envelope(&[self.lz4_envelope(), self.zstd_block_envelope()])
    }
}

fn next<'a>(args: &'a [String], i: &mut usize) -> R<&'a str> {
    *i += 1;
    Ok(args.get(*i).ok_or("missing value")?.as_str())
}

fn parse_threads(s: &str) -> R<usize> {
    if s.eq_ignore_ascii_case("auto") || s == "0" {
        Ok(std::thread::available_parallelism()
            .map(|x| x.get())
            .unwrap_or(1)
            .clamp(1, 256))
    } else {
        Ok(s.parse()?)
    }
}

impl Config {
    pub(super) fn from_args(args: &[String]) -> R<Option<Self>> {
        let mut c = Self {
            size: 4 << 20,
            seconds: 0.1,
            samples: 5,
            threads: 1,
            block_size: 65536,
            simd: Simd::Auto,
            csv: "reports/local-benchmark.csv".into(),
            samples_csv: None,
            markdown: None,
            suite: "default".into(),
            files: Vec::new(),
            only: None,
            gpu: None,
            microbench: false,
            vxg_only: false,
            fast_baselines: false,
            decode_only: false,
            validated_decode: false,
            legacy_v2: false,
            level: 3,
        };
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--level" | "-l" => c.level = next(args, &mut i)?.parse::<u8>()?.clamp(1, 9),
                "--size-mib" => {
                    c.size = next(args, &mut i)?
                        .parse::<usize>()?
                        .checked_mul(1 << 20)
                        .ok_or("size overflow")?
                }
                "--seconds" => c.seconds = next(args, &mut i)?.parse()?,
                "--samples" => c.samples = next(args, &mut i)?.parse()?,
                "--threads" => c.threads = parse_threads(next(args, &mut i)?)?,
                "--block-size" => c.block_size = next(args, &mut i)?.parse()?,
                "--simd" => c.simd = next(args, &mut i)?.parse()?,
                "--csv" => c.csv = next(args, &mut i)?.into(),
                "--samples-csv" => c.samples_csv = Some(next(args, &mut i)?.into()),
                "--markdown" => c.markdown = Some(next(args, &mut i)?.into()),
                "--suite" => c.suite = next(args, &mut i)?.into(),
                "--gpu" => c.gpu = Some(next(args, &mut i)?.parse()?),
                "--file" => c.files.push(next(args, &mut i)?.into()),
                "--only" => c.only = Some(next(args, &mut i)?.into()),
                "--microbench" => c.microbench = true,
                "--vxg-only" => c.vxg_only = true,
                "--fast-baselines" => c.fast_baselines = true,
                "--decode-only" => c.decode_only = true,
                "--validated-decode" => c.validated_decode = true,
                "--v3" | "--v4" => {} // unified pipeline is the default; kept for script compatibility
                "--legacy" => c.legacy_v2 = true,
                "--help" => {
                    println!("vxg-bench [--suite default|v4|src-unique|all] [--size-mib N] [--seconds 0.1] [--samples 5] [--threads 1|auto] [--block-size 65536] [--simd auto] [--csv FILE] [--samples-csv FILE] [--markdown FILE] [--file CORPUS_FILE]... [--only DATASET] [--gpu metal|vulkan|cuda] [--microbench] [--vxg-only] [--fast-baselines] [--decode-only] [--validated-decode] [--legacy]\nBlock baselines use independent blocks, RAW fallback, and no checksum: LZ4 (a1, a4, HC-L9), Zstd (-3, 1, 3, 9), and Zstd+TrainedDict (L3, L9, sweeping 32/64/112/256 KiB per dataset and charging dictionary size), plus whole-file zstd -19 --long as a non-random-access reference. Pass --fast-baselines to skip heavy presets (L9, dict-L9, whole-L19) during quick iteration.");
                    return Ok(None);
                }
                x => return Err(format!("unknown option {x}").into()),
            }
            i += 1;
        }
        if c.size == 0
            || !c.seconds.is_finite()
            || c.seconds <= 0.0
            || c.seconds > 60.0
            || c.samples < 3
            || c.samples > 101
        {
            return Err("invalid size, seconds (0..60], or samples (3..101)".into());
        }
        Ok(Some(c))
    }
}
