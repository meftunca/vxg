use std::{env, fs, io::Write};

use vxg::Simd;

use super::{
    config::{CodecPoint, Config, DatasetSummary},
    metrics::{calc_bci, calc_gci, calc_vpi, geomean},
    R,
};

fn fmt_size(raw_bytes: usize) -> String {
    if raw_bytes >= (1 << 20) && raw_bytes % (1 << 20) == 0 {
        format!("{} MiB", raw_bytes >> 20)
    } else {
        format!("{:.1} MB", raw_bytes as f64 / 1e6)
    }
}

#[inline]
fn point_vpi(p: CodecPoint) -> f64 {
    calc_vpi(p.ratio, p.enc, p.dec)
}

#[inline]
fn point_bci(p: CodecPoint) -> f64 {
    calc_bci(p.ratio, p.enc, p.dec)
}

#[inline]
fn point_gci(p: CodecPoint) -> Option<f64> {
    calc_gci(p.ratio, p.enc, p.dec, p.enc_j_gb, p.dec_j_gb)
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
    let has_heavy = summaries.iter().any(|s| s.lz4hc_9.ratio > 0.0);
    writeln!(
        f,
        "## VXG v4 Benchmark, Chimera Envelope & Energy Summary (`{}` / `{}`, SIMD=`{:?}`, threads=`{}`, block=`{}`)",
        env::consts::OS,
        env::consts::ARCH,
        simd,
        c.threads,
        c.block_size
    )?;
    writeln!(
        f,
        "\nBaselines: `liblz4 {lz}` (`lz4-c-a1`, `lz4-c-a4`{}) and `libzstd {zs}` (`zstd-c-l-3`, `zstd-c-l1`, `zstd-c-l3`{}). Independent `{}`-byte blocks, RAW fallback, checksums OFF.\n",
        if has_heavy { ", `lz4hc-c-l9`" } else { "" },
        if has_heavy { ", `zstd-c-l9`" } else { "" },
        c.block_size
    )?;

    writeln!(f, "### 1. Individual Codec Compression Ratio & Throughput (`GB/s`)\n")?;
    if has_heavy {
        writeln!(
            f,
            "| Dataset | Size | VXG v4 Ratio | LZ4 (a1) | LZ4-HC (L9) | Zstd-1 | Zstd-9 | VXG Enc | LZ4 (a1) Enc | LZ4-HC Enc | Zstd-1 Enc | VXG Dec | LZ4 (a1) Dec | Zstd-1 Dec |"
        )?;
        writeln!(
            f,
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
        )?;
        for s in summaries {
            writeln!(
                f,
                "| **`{}`** | {} | **{:.4}** | {:.4} | {:.4} | {:.4} | {:.4} | **{:.3}** | {:.3} | {:.3} | {:.3} | **{:.3}** | {:.3} | {:.3} |",
                s.name,
                fmt_size(s.raw_bytes),
                s.vxg.ratio,
                s.lz4_a1.ratio,
                s.lz4hc_9.ratio,
                s.zstd_1.ratio,
                s.zstd_9.ratio,
                s.vxg.enc,
                s.lz4_a1.enc,
                s.lz4hc_9.enc,
                s.zstd_1.enc,
                s.vxg.dec,
                s.lz4_a1.dec,
                s.zstd_1.dec,
            )?;
        }
        if summaries.len() > 1 {
            writeln!(
                f,
                "| **Geometric Mean** | — | **{:.4}** | {:.4} | {:.4} | {:.4} | {:.4} | **{:.3}** | {:.3} | {:.3} | {:.3} | **{:.3}** | {:.3} | {:.3} |",
                geomean(summaries.iter().map(|s| s.vxg.ratio)),
                geomean(summaries.iter().map(|s| s.lz4_a1.ratio)),
                geomean(summaries.iter().map(|s| s.lz4hc_9.ratio)),
                geomean(summaries.iter().map(|s| s.zstd_1.ratio)),
                geomean(summaries.iter().map(|s| s.zstd_9.ratio)),
                geomean(summaries.iter().map(|s| s.vxg.enc)),
                geomean(summaries.iter().map(|s| s.lz4_a1.enc)),
                geomean(summaries.iter().map(|s| s.lz4hc_9.enc)),
                geomean(summaries.iter().map(|s| s.zstd_1.enc)),
                geomean(summaries.iter().map(|s| s.vxg.dec)),
                geomean(summaries.iter().map(|s| s.lz4_a1.dec)),
                geomean(summaries.iter().map(|s| s.zstd_1.dec)),
            )?;
        }
    } else {
        writeln!(
            f,
            "| Dataset | Size | VXG v4 Ratio | LZ4 Ratio | Zstd-1 Ratio | VXG v4 Encode | LZ4 Encode | Zstd-1 Encode | VXG v4 Decode | LZ4 Decode | Zstd-1 Decode |"
        )?;
        writeln!(
            f,
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
        )?;
        for s in summaries {
            writeln!(
                f,
                "| **`{}`** | {} | **{:.4}** | {:.4} | {:.4} | **{:.3} GB/s** | {:.3} GB/s | {:.3} GB/s | **{:.3} GB/s** | {:.3} GB/s | {:.3} GB/s |",
                s.name,
                fmt_size(s.raw_bytes),
                s.vxg.ratio,
                s.lz4_a1.ratio,
                s.zstd_1.ratio,
                s.vxg.enc,
                s.lz4_a1.enc,
                s.zstd_1.enc,
                s.vxg.dec,
                s.lz4_a1.dec,
                s.zstd_1.dec,
            )?;
        }
    }

    writeln!(
        f,
        "\n### 2. \"Chimera Envelope\" Showdown (`VXG v4` vs Impossible Best-of-Competitor Composites)\n\nEach **Chimera** row grants a competitor family an impossible synthetic combination on every single dataset simultaneously:\n- **`LZ4 Chimera`:** `min_ratio(lz4-a1, lz4-a4, lz4hc-l9)` + `max_encode_GBps(...)` + `max_decode_GBps(...)` (`LZ4-HC L9` ratio at `LZ4-fast` speed).\n- **`Zstd Chimera`:** `min_ratio(zstd-l-3, zstd-l1, zstd-l3, zstd-l9)` + `max_encode_GBps(...)` + `max_decode_GBps(...)` (`Zstd L9` ratio at `Zstd -3` ultra-fast speed).\n- **`All-Rivals Chimera`:** `min_ratio(all LZ4 & Zstd)` + `max_encode_GBps(all LZ4 & Zstd)` + `max_decode_GBps(all LZ4 & Zstd)` (`Zstd L9` ratio at `LZ4-fast` speed!).\n"
    )?;
    writeln!(
        f,
        "| Dataset | VXG Ratio | LZ4 Chimera Ratio | Zstd Chimera Ratio | VXG Enc | LZ4 Chimera Enc | Zstd Chimera Enc | VXG Dec | LZ4 Chimera Dec | Zstd Chimera Dec | VXG `VPI` | LZ4 Chimera `VPI` | Zstd Chimera `VPI` | All-Rivals Chimera `VPI` |"
    )?;
    writeln!(
        f,
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    )?;
    for s in summaries {
        let lc = s.lz4_chimera();
        let zc = s.zstd_chimera();
        let ac = s.all_rivals_chimera();
        writeln!(
            f,
            "| **`{}`** | **{:.4}** | {:.4} | {:.4} | **{:.3}** | {:.3} | {:.3} | **{:.3}** | {:.3} | {:.3} | **{:.1}** | {:.1} | {:.1} | {:.1} |",
            s.name,
            s.vxg.ratio,
            lc.ratio,
            zc.ratio,
            s.vxg.enc,
            lc.enc,
            zc.enc,
            s.vxg.dec,
            lc.dec,
            zc.dec,
            point_vpi(s.vxg),
            point_vpi(lc),
            point_vpi(zc),
            point_vpi(ac),
        )?;
    }
    if summaries.len() > 1 {
        writeln!(
            f,
            "| **Geometric Mean** | **{:.4}** | {:.4} | {:.4} | **{:.3}** | {:.3} | {:.3} | **{:.3}** | {:.3} | {:.3} | **{:.1}** | {:.1} | {:.1} | {:.1} |",
            geomean(summaries.iter().map(|s| s.vxg.ratio)),
            geomean(summaries.iter().map(|s| s.lz4_chimera().ratio)),
            geomean(summaries.iter().map(|s| s.zstd_chimera().ratio)),
            geomean(summaries.iter().map(|s| s.vxg.enc)),
            geomean(summaries.iter().map(|s| s.lz4_chimera().enc)),
            geomean(summaries.iter().map(|s| s.zstd_chimera().enc)),
            geomean(summaries.iter().map(|s| s.vxg.dec)),
            geomean(summaries.iter().map(|s| s.lz4_chimera().dec)),
            geomean(summaries.iter().map(|s| s.zstd_chimera().dec)),
            geomean(summaries.iter().map(|s| point_vpi(s.vxg))),
            geomean(summaries.iter().map(|s| point_vpi(s.lz4_chimera()))),
            geomean(summaries.iter().map(|s| point_vpi(s.zstd_chimera()))),
            geomean(summaries.iter().map(|s| point_vpi(s.all_rivals_chimera()))),
        )?;
    }

    writeln!(
        f,
        "\n### 3. Unified Codec Performance Index (`VPI` & `BCI`)\n\n- **VXG Performance Index (`VPI`):** `sqrt(Encode_GBps * Decode_GBps) / Ratio` (higher is better)\n- **Balanced 3-Axis Cube Index (`BCI`):** `((Encode_GBps * Decode_GBps) / Ratio)^(1/3)`\n"
    )?;
    writeln!(
        f,
        "| Dataset | VXG `VPI` | LZ4 (a1) `VPI` | Zstd-1 `VPI` | VXG vs LZ4 (a1) | VXG vs LZ4 Chimera | VXG vs Zstd Chimera | VXG vs All-Rivals Chimera | VXG `BCI` | All-Rivals Chimera `BCI` |"
    )?;
    writeln!(
        f,
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    )?;
    for s in summaries {
        let vv = point_vpi(s.vxg);
        let lv = point_vpi(s.lz4_a1);
        let zv = point_vpi(s.zstd_1);
        let lcv = point_vpi(s.lz4_chimera());
        let zcv = point_vpi(s.zstd_chimera());
        let acv = point_vpi(s.all_rivals_chimera());
        writeln!(
            f,
            "| **`{}`** | **{:.1}** | {:.1} | {:.1} | **{:.2}x** | **{:.2}x** | **{:.2}x** | **{:.2}x** | **{:.2}** | {:.2} |",
            s.name,
            vv,
            lv,
            zv,
            if lv > 0.0 { vv / lv } else { 0.0 },
            if lcv > 0.0 { vv / lcv } else { 0.0 },
            if zcv > 0.0 { vv / zcv } else { 0.0 },
            if acv > 0.0 { vv / acv } else { 0.0 },
            point_bci(s.vxg),
            point_bci(s.all_rivals_chimera()),
        )?;
    }
    if summaries.len() > 1 {
        let vv = geomean(summaries.iter().map(|s| point_vpi(s.vxg)));
        let lv = geomean(summaries.iter().map(|s| point_vpi(s.lz4_a1)));
        let zv = geomean(summaries.iter().map(|s| point_vpi(s.zstd_1)));
        let lcv = geomean(summaries.iter().map(|s| point_vpi(s.lz4_chimera())));
        let zcv = geomean(summaries.iter().map(|s| point_vpi(s.zstd_chimera())));
        let acv = geomean(summaries.iter().map(|s| point_vpi(s.all_rivals_chimera())));
        writeln!(
            f,
            "| **Geometric Mean** | **{:.1}** | {:.1} | {:.1} | **{:.2}x** | **{:.2}x** | **{:.2}x** | **{:.2}x** | **{:.2}** | {:.2} |",
            vv,
            lv,
            zv,
            if lv > 0.0 { vv / lv } else { 0.0 },
            if lcv > 0.0 { vv / lcv } else { 0.0 },
            if zcv > 0.0 { vv / zcv } else { 0.0 },
            if acv > 0.0 { vv / acv } else { 0.0 },
            geomean(summaries.iter().map(|s| point_bci(s.vxg))),
            geomean(summaries.iter().map(|s| point_bci(s.all_rivals_chimera()))),
        )?;
    }

    if summaries.iter().any(|s| s.vxg.enc_j_gb.is_some()) {
        writeln!(
            f,
            "\n### 4. Per-Process CPU Energy Efficiency (`J/GB`, Lower is Better) & Green Codec Index (`GCI`, Higher is Better)\n"
        )?;
        writeln!(
            f,
            "| Dataset | VXG Enc `J/GB` | LZ4 Enc `J/GB` | Zstd-1 Enc `J/GB` | VXG Dec `J/GB` | LZ4 Dec `J/GB` | Zstd-1 Dec `J/GB` | VXG `GCI` | LZ4 Chimera `GCI` | Zstd Chimera `GCI` | All-Rivals Chimera `GCI` |"
        )?;
        writeln!(
            f,
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
        )?;
        for s in summaries {
            writeln!(
                f,
                "| **`{}`** | **{:.3}** | {:.3} | {:.3} | **{:.3}** | {:.3} | {:.3} | **{:.1}** | {:.1} | {:.1} | {:.1} |",
                s.name,
                s.vxg.enc_j_gb.unwrap_or(0.0),
                s.lz4_a1.enc_j_gb.unwrap_or(0.0),
                s.zstd_1.enc_j_gb.unwrap_or(0.0),
                s.vxg.dec_j_gb.unwrap_or(0.0),
                s.lz4_a1.dec_j_gb.unwrap_or(0.0),
                s.zstd_1.dec_j_gb.unwrap_or(0.0),
                point_gci(s.vxg).unwrap_or(0.0),
                point_gci(s.lz4_chimera()).unwrap_or(0.0),
                point_gci(s.zstd_chimera()).unwrap_or(0.0),
                point_gci(s.all_rivals_chimera()).unwrap_or(0.0),
            )?;
        }
        if summaries.len() > 1 {
            writeln!(
                f,
                "| **Geometric Mean** | **{:.3}** | {:.3} | {:.3} | **{:.3}** | {:.3} | {:.3} | **{:.1}** | {:.1} | {:.1} | {:.1} |",
                geomean(summaries.iter().filter_map(|s| s.vxg.enc_j_gb)),
                geomean(summaries.iter().filter_map(|s| s.lz4_a1.enc_j_gb)),
                geomean(summaries.iter().filter_map(|s| s.zstd_1.enc_j_gb)),
                geomean(summaries.iter().filter_map(|s| s.vxg.dec_j_gb)),
                geomean(summaries.iter().filter_map(|s| s.lz4_a1.dec_j_gb)),
                geomean(summaries.iter().filter_map(|s| s.zstd_1.dec_j_gb)),
                geomean(summaries.iter().filter_map(|s| point_gci(s.vxg))),
                geomean(summaries.iter().filter_map(|s| point_gci(s.lz4_chimera()))),
                geomean(summaries.iter().filter_map(|s| point_gci(s.zstd_chimera()))),
                geomean(summaries.iter().filter_map(|s| point_gci(s.all_rivals_chimera()))),
            )?;
        }
    }

    f.flush()?;
    Ok(())
}
