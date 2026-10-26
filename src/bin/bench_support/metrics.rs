use std::{
    env, fs,
    io::Write,
    time::{Duration, Instant},
};

use vxg::Simd;

use super::{
    harness::median,
    Config, DatasetSummary, R,
};

#[derive(Clone, Debug, Default)]
pub(super) struct MeasuredOp {
    pub(super) ns_samples: Vec<f64>,
    pub(super) j_per_gb: Option<f64>,
}

#[cfg(target_os = "macos")]
#[inline]
pub(super) fn process_energy_nj() -> Option<u64> {
    #[repr(C)]
    struct RusageInfoV6 {
        ri_uuid: [u8; 16],
        fields: [u64; 56],
    }
    unsafe extern "C" {
        fn getpid() -> i32;
        fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut RusageInfoV6) -> i32;
    }
    const RUSAGE_INFO_V6: i32 = 6;
    const RI_ENERGY_NJ_INDEX: usize = 40; // offset 336 = 16 + 40 * 8
    let mut ri = RusageInfoV6 {
        ri_uuid: [0u8; 16],
        fields: [0u64; 56],
    };
    let rc = unsafe { proc_pid_rusage(getpid(), RUSAGE_INFO_V6, &mut ri) };
    if rc == 0 && ri.fields[RI_ENERGY_NJ_INDEX] > 0 {
        Some(ri.fields[RI_ENERGY_NJ_INDEX])
    } else {
        None
    }
}

#[cfg(not(target_os = "macos"))]
#[inline]
pub(super) fn process_energy_nj() -> Option<u64> {
    None
}

pub(super) fn measure_op<F>(c: &Config, raw_bytes: usize, mut f: F) -> R<MeasuredOp>
where
    F: FnMut() -> R<()>,
{
    f()?;
    f()?; // Warm-up, never included.
    let mut ns_samples = Vec::with_capacity(c.samples);
    let mut j_samples = Vec::with_capacity(c.samples);
    let duration = Duration::from_secs_f64(c.seconds);
    for _ in 0..c.samples {
        let e0 = process_energy_nj();
        let start = Instant::now();
        let mut n = 0u64;
        loop {
            f()?;
            n += 1;
            if start.elapsed() >= duration {
                break;
            }
        }
        let elapsed_ns = start.elapsed().as_secs_f64() * 1e9;
        let e1 = process_energy_nj();
        ns_samples.push(elapsed_ns / n as f64);
        if let (Some(b), Some(a)) = (e0, e1) {
            let delta_nj = a.saturating_sub(b) as f64;
            let total_bytes = (n as f64) * (raw_bytes as f64);
            if delta_nj > 0.0 && total_bytes > 0.0 && elapsed_ns > 0.0 {
                // 1 nJ / 1 byte == 10^-9 J / 10^-9 GB == 1 J / decimal GB
                j_samples.push(delta_nj / total_bytes);
            }
        }
    }
    ns_samples.sort_by(f64::total_cmp);
    j_samples.sort_by(f64::total_cmp);
    let j_per_gb = if j_samples.is_empty() {
        None
    } else {
        Some(median(&j_samples))
    };
    Ok(MeasuredOp {
        ns_samples,
        j_per_gb,
    })
}

/// VXG Performance Index (VPI): `sqrt(Encode_GBps * Decode_GBps) / Ratio`.
/// Higher is better.
#[inline]
pub(super) fn calc_vpi(ratio: f64, enc_gbps: f64, dec_gbps: f64) -> f64 {
    if ratio <= 0.0 || enc_gbps <= 0.0 || dec_gbps <= 0.0 {
        0.0
    } else {
        (enc_gbps * dec_gbps).sqrt() / ratio
    }
}

/// Balanced 3-Axis Cube Index (BCI): `((Encode_GBps * Decode_GBps) / Ratio)^(1/3)`.
/// Equal 1/3 geometric weight on Compression Factor (`1/Ratio`), Encode GB/s, and Decode GB/s.
#[inline]
pub(super) fn calc_bci(ratio: f64, enc_gbps: f64, dec_gbps: f64) -> f64 {
    if ratio <= 0.0 || enc_gbps <= 0.0 || dec_gbps <= 0.0 {
        0.0
    } else {
        ((enc_gbps * dec_gbps) / ratio).cbrt()
    }
}

/// Green Codec Index (GCI): `VPI / sqrt(Enc_J_per_GB * Dec_J_per_GB)`.
/// Combines Compression Factor (`1/Ratio`), Throughput (`GB/s`), and Energy Efficiency (`GB/J`).
#[inline]
pub(super) fn calc_gci(
    ratio: f64,
    enc_gbps: f64,
    dec_gbps: f64,
    enc_j_gb: Option<f64>,
    dec_j_gb: Option<f64>,
) -> Option<f64> {
    let ej = enc_j_gb?;
    let dj = dec_j_gb?;
    if ej <= 0.0 || dj <= 0.0 {
        return None;
    }
    Some(calc_vpi(ratio, enc_gbps, dec_gbps) / (ej * dj).sqrt())
}

fn geomean(vals: impl Iterator<Item = f64>) -> f64 {
    let mut sum_ln = 0.0;
    let mut count = 0usize;
    for v in vals {
        if v > 0.0 && v.is_finite() {
            sum_ln += v.ln();
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        (sum_ln / count as f64).exp()
    }
}

pub(super) fn write_markdown_report(
    path: &str,
    c: &Config,
    simd: Simd,
    lz: &str,
    zs: &str,
    summaries: &[DatasetSummary],
) -> R<()> {
    if let Some(parent) = std::path::Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let mut f = fs::File::create(path)?;
    writeln!(
        f,
        "## VXG v4 Benchmark, Composite Index & Energy Summary (`{}` / `{}`, SIMD=`{:?}`, threads=`{}`, block=`{}`)",
        env::consts::OS,
        env::consts::ARCH,
        simd,
        c.threads,
        c.block_size
    )?;
    writeln!(
        f,
        "\nBaselines: `liblz4 {lz}` (`lz4-c-a1`), `libzstd {zs}` (`zstd-c-l1`, `zstd-c-l3`). Independent `{}`-byte blocks, RAW fallback, checksums OFF.\n",
        c.block_size
    )?;

    writeln!(f, "### 1. Compression Ratio & Throughput (`GB/s`)\n")?;
    writeln!(
        f,
        "| Dataset | Size | VXG v4 Ratio | LZ4 Ratio | Zstd-1 Ratio | VXG v4 Encode | LZ4 Encode | Zstd-1 Encode | VXG v4 Decode | LZ4 Decode | Zstd-1 Decode |"
    )?;
    writeln!(
        f,
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    )?;
    for s in summaries {
        let size_str = if s.raw_bytes >= (1 << 20) && s.raw_bytes % (1 << 20) == 0 {
            format!("{} MiB", s.raw_bytes >> 20)
        } else {
            format!("{:.1} MB", s.raw_bytes as f64 / 1e6)
        };
        writeln!(
            f,
            "| **`{}`** | {} | **{:.4}** | {:.4} | {:.4} | **{:.3} GB/s** | {:.3} GB/s | {:.3} GB/s | **{:.3} GB/s** | {:.3} GB/s | {:.3} GB/s |",
            s.name,
            size_str,
            s.vxg_ratio,
            s.lz4_ratio,
            s.zstd1_ratio,
            s.vxg_enc,
            s.lz4_enc,
            s.zstd1_enc,
            s.vxg_dec,
            s.lz4_dec,
            s.zstd1_dec
        )?;
    }
    if summaries.len() > 1 {
        writeln!(
            f,
            "| **Geometric Mean** | — | **{:.4}** | {:.4} | {:.4} | **{:.3} GB/s** | {:.3} GB/s | {:.3} GB/s | **{:.3} GB/s** | {:.3} GB/s | {:.3} GB/s |",
            geomean(summaries.iter().map(|s| s.vxg_ratio)),
            geomean(summaries.iter().map(|s| s.lz4_ratio)),
            geomean(summaries.iter().map(|s| s.zstd1_ratio)),
            geomean(summaries.iter().map(|s| s.vxg_enc)),
            geomean(summaries.iter().map(|s| s.lz4_enc)),
            geomean(summaries.iter().map(|s| s.zstd1_enc)),
            geomean(summaries.iter().map(|s| s.vxg_dec)),
            geomean(summaries.iter().map(|s| s.lz4_dec)),
            geomean(summaries.iter().map(|s| s.zstd1_dec)),
        )?;
    }

    writeln!(
        f,
        "\n### 2. Unified Codec Performance Index (`VPI` & `BCI`)\n\n- **VXG Performance Index (`VPI`):** `sqrt(Encode_GBps * Decode_GBps) / Ratio` (higher is better)\n- **Balanced 3-Axis Cube Index (`BCI`):** `((Encode_GBps * Decode_GBps) / Ratio)^(1/3)` (equal 1/3 weight on compression factor, encode speed, and decode speed)\n"
    )?;
    writeln!(
        f,
        "| Dataset | VXG v4 `VPI` | LZ4 `VPI` | Zstd-1 `VPI` | VXG vs LZ4 (`VPI`) | VXG vs Zstd-1 (`VPI`) | VXG v4 `BCI` | LZ4 `BCI` | Zstd-1 `BCI` |"
    )?;
    writeln!(
        f,
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    )?;
    for s in summaries {
        let v_vpi = calc_vpi(s.vxg_ratio, s.vxg_enc, s.vxg_dec);
        let l_vpi = calc_vpi(s.lz4_ratio, s.lz4_enc, s.lz4_dec);
        let z_vpi = calc_vpi(s.zstd1_ratio, s.zstd1_enc, s.zstd1_dec);
        let v_bci = calc_bci(s.vxg_ratio, s.vxg_enc, s.vxg_dec);
        let l_bci = calc_bci(s.lz4_ratio, s.lz4_enc, s.lz4_dec);
        let z_bci = calc_bci(s.zstd1_ratio, s.zstd1_enc, s.zstd1_dec);
        writeln!(
            f,
            "| **`{}`** | **{:.1}** | {:.1} | {:.1} | **{:.2}x** | **{:.2}x** | **{:.2}** | {:.2} | {:.2} |",
            s.name,
            v_vpi,
            l_vpi,
            z_vpi,
            if l_vpi > 0.0 { v_vpi / l_vpi } else { 0.0 },
            if z_vpi > 0.0 { v_vpi / z_vpi } else { 0.0 },
            v_bci,
            l_bci,
            z_bci,
        )?;
    }
    if summaries.len() > 1 {
        let gv = geomean(summaries.iter().map(|s| calc_vpi(s.vxg_ratio, s.vxg_enc, s.vxg_dec)));
        let gl = geomean(summaries.iter().map(|s| calc_vpi(s.lz4_ratio, s.lz4_enc, s.lz4_dec)));
        let gz = geomean(summaries.iter().map(|s| calc_vpi(s.zstd1_ratio, s.zstd1_enc, s.zstd1_dec)));
        let bv = geomean(summaries.iter().map(|s| calc_bci(s.vxg_ratio, s.vxg_enc, s.vxg_dec)));
        let bl = geomean(summaries.iter().map(|s| calc_bci(s.lz4_ratio, s.lz4_enc, s.lz4_dec)));
        let bz = geomean(summaries.iter().map(|s| calc_bci(s.zstd1_ratio, s.zstd1_enc, s.zstd1_dec)));
        writeln!(
            f,
            "| **Geometric Mean** | **{:.1}** | {:.1} | {:.1} | **{:.2}x** | **{:.2}x** | **{:.2}** | {:.2} | {:.2} |",
            gv,
            gl,
            gz,
            if gl > 0.0 { gv / gl } else { 0.0 },
            if gz > 0.0 { gv / gz } else { 0.0 },
            bv,
            bl,
            bz,
        )?;
    }

    if summaries.iter().any(|s| s.vxg_enc_j_gb.is_some()) {
        writeln!(
            f,
            "\n### 3. Per-Process CPU Energy Efficiency (`J/GB`, Lower is Better) & Green Codec Index (`GCI`, Higher is Better)\n\n- **Energy Source:** Kernel per-process hardware energy counter (`proc_pid_rusage` `RUSAGE_INFO_V6` `ri_energy_nj` on Apple Silicon).\n- **Green Codec Index (`GCI`):** `VPI / sqrt(Encode_J_per_GB * Decode_J_per_GB)`\n"
        )?;
        writeln!(
            f,
            "| Dataset | VXG Enc `J/GB` | LZ4 Enc `J/GB` | Zstd-1 Enc `J/GB` | VXG Dec `J/GB` | LZ4 Dec `J/GB` | Zstd-1 Dec `J/GB` | VXG `GCI` | LZ4 `GCI` | Zstd-1 `GCI` |"
        )?;
        writeln!(
            f,
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
        )?;
        for s in summaries {
            let v_gci = calc_gci(s.vxg_ratio, s.vxg_enc, s.vxg_dec, s.vxg_enc_j_gb, s.vxg_dec_j_gb).unwrap_or(0.0);
            let l_gci = calc_gci(s.lz4_ratio, s.lz4_enc, s.lz4_dec, s.lz4_enc_j_gb, s.lz4_dec_j_gb).unwrap_or(0.0);
            let z_gci = calc_gci(s.zstd1_ratio, s.zstd1_enc, s.zstd1_dec, s.zstd1_enc_j_gb, s.zstd1_dec_j_gb).unwrap_or(0.0);
            writeln!(
                f,
                "| **`{}`** | **{:.3}** | {:.3} | {:.3} | **{:.3}** | {:.3} | {:.3} | **{:.1}** | {:.1} | {:.1} |",
                s.name,
                s.vxg_enc_j_gb.unwrap_or(0.0),
                s.lz4_enc_j_gb.unwrap_or(0.0),
                s.zstd1_enc_j_gb.unwrap_or(0.0),
                s.vxg_dec_j_gb.unwrap_or(0.0),
                s.lz4_dec_j_gb.unwrap_or(0.0),
                s.zstd1_dec_j_gb.unwrap_or(0.0),
                v_gci,
                l_gci,
                z_gci,
            )?;
        }
        if summaries.len() > 1 {
            let vej = geomean(summaries.iter().filter_map(|s| s.vxg_enc_j_gb));
            let lej = geomean(summaries.iter().filter_map(|s| s.lz4_enc_j_gb));
            let zej = geomean(summaries.iter().filter_map(|s| s.zstd1_enc_j_gb));
            let vdj = geomean(summaries.iter().filter_map(|s| s.vxg_dec_j_gb));
            let ldj = geomean(summaries.iter().filter_map(|s| s.lz4_dec_j_gb));
            let zdj = geomean(summaries.iter().filter_map(|s| s.zstd1_dec_j_gb));
            let vg = geomean(summaries.iter().filter_map(|s| calc_gci(s.vxg_ratio, s.vxg_enc, s.vxg_dec, s.vxg_enc_j_gb, s.vxg_dec_j_gb)));
            let lg = geomean(summaries.iter().filter_map(|s| calc_gci(s.lz4_ratio, s.lz4_enc, s.lz4_dec, s.lz4_enc_j_gb, s.lz4_dec_j_gb)));
            let zg = geomean(summaries.iter().filter_map(|s| calc_gci(s.zstd1_ratio, s.zstd1_enc, s.zstd1_dec, s.zstd1_enc_j_gb, s.zstd1_dec_j_gb)));
            writeln!(
                f,
                "| **Geometric Mean** | **{:.3}** | {:.3} | {:.3} | **{:.3}** | {:.3} | {:.3} | **{:.1}** | {:.1} | {:.1} |",
                vej, lej, zej, vdj, ldj, zdj, vg, lg, zg
            )?;
        }
    }

    f.flush()?;
    Ok(())
}
