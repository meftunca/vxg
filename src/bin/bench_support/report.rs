use std::{env, fs, io::Write};

use vxg::Simd;

use super::{
    config::{CodecPoint, Config, DatasetSummary},
    metrics::{calc_bci, calc_gci, calc_vpi, geomean},
    R,
};

fn short_name(name: &str) -> &str {
    std::path::Path::new(name)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(name)
}

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

fn fmt_min(val: f64, peers: &[f64], prec: usize) -> String {
    if val <= 0.0 {
        return "—".into();
    }
    let best = peers
        .iter()
        .copied()
        .filter(|&x| x > 0.0)
        .fold(f64::INFINITY, f64::min);
    if (val - best).abs() <= 1e-9 {
        format!("**{val:.prec$}**")
    } else {
        format!("{val:.prec$}")
    }
}

/// Formats energy (`J/GB`), refusing to bold codecs with near-raw storage (`ratio >= 0.90`).
fn fmt_min_compressed(val: f64, ratio: f64, peers: &[(f64, f64)], prec: usize) -> String {
    if val <= 0.0 {
        return "—".into();
    }
    if ratio >= 0.90 {
        return format!("{val:.prec$}");
    }
    let best = peers
        .iter()
        .copied()
        .filter(|&(v, r)| v > 0.0 && r > 0.0 && r < 0.90)
        .map(|(v, _)| v)
        .fold(f64::INFINITY, f64::min);
    if best.is_finite() && (val - best).abs() <= 1e-9 {
        format!("**{val:.prec$}**")
    } else {
        format!("{val:.prec$}")
    }
}

/// Formats throughput or composite score, refusing to bold codecs with near-raw storage
/// (`ratio >= 0.90`, i.e. `< 10%` compression) when comparing speed/trade-off winners.
fn fmt_max_compressed(val: f64, ratio: f64, peers: &[(f64, f64)], prec: usize) -> String {
    if val <= 0.0 {
        return "—".into();
    }
    if ratio >= 0.90 {
        return format!("{val:.prec$}");
    }
    let best = peers
        .iter()
        .copied()
        .filter(|&(_, r)| r > 0.0 && r < 0.90)
        .map(|(v, _)| v)
        .fold(0.0f64, f64::max);
    if best > 0.0 && (val - best).abs() <= 1e-9 {
        format!("**{val:.prec$}**")
    } else {
        format!("{val:.prec$}")
    }
}

fn write_table1_row(f: &mut fs::File, label: &str, size_col: &str, s: &DatasetSummary) -> R<()> {
    let blk_ratios = [
        s.vxg.ratio,
        s.lz4_a1.ratio,
        s.zstd_1.ratio,
        s.zstd_9.ratio,
        s.zstd_dict_3.ratio,
        s.zstd_dict_9.ratio,
    ];
    let encs = [
        (s.vxg.enc, s.vxg.ratio),
        (s.lz4_a1.enc, s.lz4_a1.ratio),
        (s.zstd_1.enc, s.zstd_1.ratio),
        (s.zstd_dict_3.enc, s.zstd_dict_3.ratio),
    ];
    let decs = [
        (s.vxg.dec, s.vxg.ratio),
        (s.lz4_a1.dec, s.lz4_a1.ratio),
        (s.zstd_1.dec, s.zstd_1.ratio),
        (s.zstd_dict_3.dec, s.zstd_dict_3.ratio),
    ];
    let whole_r = if s.zstd_whole_19.ratio > 0.0 {
        format!("{:.4}", s.zstd_whole_19.ratio)
    } else {
        "—".into()
    };
    let train_col = if s.dict_train_ms > 0.0 {
        format!("{:.1}", s.dict_train_ms)
    } else {
        "—".into()
    };
    writeln!(
        f,
        "| {label} | {size_col} | {} | {} | {} | {} | {} | {} | {whole_r} | {} | {} | {} | {} | {} | {} | {} | {} | {train_col} |",
        fmt_min(s.vxg.ratio, &blk_ratios, 4),
        fmt_min(s.lz4_a1.ratio, &blk_ratios, 4),
        fmt_min(s.zstd_1.ratio, &blk_ratios, 4),
        fmt_min(s.zstd_9.ratio, &blk_ratios, 4),
        fmt_min(s.zstd_dict_3.ratio, &blk_ratios, 4),
        fmt_min(s.zstd_dict_9.ratio, &blk_ratios, 4),
        fmt_max_compressed(s.vxg.enc, s.vxg.ratio, &encs, 3),
        fmt_max_compressed(s.lz4_a1.enc, s.lz4_a1.ratio, &encs, 3),
        fmt_max_compressed(s.zstd_1.enc, s.zstd_1.ratio, &encs, 3),
        fmt_max_compressed(s.zstd_dict_3.enc, s.zstd_dict_3.ratio, &encs, 3),
        fmt_max_compressed(s.vxg.dec, s.vxg.ratio, &decs, 3),
        fmt_max_compressed(s.lz4_a1.dec, s.lz4_a1.ratio, &decs, 3),
        fmt_max_compressed(s.zstd_1.dec, s.zstd_1.ratio, &decs, 3),
        fmt_max_compressed(s.zstd_dict_3.dec, s.zstd_dict_3.ratio, &decs, 3),
    )?;
    Ok(())
}

fn synth_geomean_summary(rows: &[&DatasetSummary]) -> DatasetSummary {
    let gm_pt = |proj: fn(&DatasetSummary) -> CodecPoint| CodecPoint {
        ratio: geomean(rows.iter().map(|s| proj(s).ratio)),
        enc: geomean(rows.iter().map(|s| proj(s).enc)),
        dec: geomean(rows.iter().map(|s| proj(s).dec)),
        enc_j_gb: {
            let g = geomean(rows.iter().filter_map(|s| proj(s).enc_j_gb));
            (g > 0.0).then_some(g)
        },
        dec_j_gb: {
            let g = geomean(rows.iter().filter_map(|s| proj(s).dec_j_gb));
            (g > 0.0).then_some(g)
        },
    };
    DatasetSummary {
        name: String::new(),
        raw_bytes: 0,
        vxg: gm_pt(|s| s.vxg),
        lz4_a1: gm_pt(|s| s.lz4_a1),
        lz4_a4: gm_pt(|s| s.lz4_a4),
        lz4hc_9: gm_pt(|s| s.lz4hc_9),
        zstd_m3: gm_pt(|s| s.zstd_m3),
        zstd_1: gm_pt(|s| s.zstd_1),
        zstd_3: gm_pt(|s| s.zstd_3),
        zstd_9: gm_pt(|s| s.zstd_9),
        zstd_dict_3: gm_pt(|s| s.zstd_dict_3),
        zstd_dict_9: gm_pt(|s| s.zstd_dict_9),
        zstd_whole_19: gm_pt(|s| s.zstd_whole_19),
        dict_bytes: 0,
        dict_train_ms: geomean(
            rows.iter()
                .map(|s| s.dict_train_ms)
                .filter(|&x| x > 0.0),
        ),
    }
}

fn summary_groups(summaries: &[DatasetSummary]) -> Vec<(String, Vec<&DatasetSummary>)> {
    let mut out = Vec::new();
    if summaries.len() <= 1 {
        return out;
    }
    let non_bmp: Vec<&DatasetSummary> = summaries.iter().filter(|s| !s.is_bmp_2d()).collect();
    let bmp: Vec<&DatasetSummary> = summaries.iter().filter(|s| s.is_bmp_2d()).collect();
    if !non_bmp.is_empty() && !bmp.is_empty() {
        out.push((
            format!("**Text & Structured Geomean ({})**", non_bmp.len()),
            non_bmp,
        ));
        out.push((format!("**2D BMP Geomean ({})**", bmp.len()), bmp));
    }
    out.push((
        format!("**All-Dataset Geomean ({})**", summaries.len()),
        summaries.iter().collect(),
    ));
    out
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
        "## VXG v4/v5 Comparative Benchmark Report (`{}` / `{}`, SIMD=`{:?}`, threads=`{}`, block=`{}`)",
        env::consts::OS,
        env::consts::ARCH,
        simd,
        c.threads,
        c.block_size
    )?;
    writeln!(
        f,
        "\n- **Block-independent codecs (`{}`-byte blocks, random access = YES):** `VXG v4/v5`, `liblz4 {lz}` (`lz4-c-a1`, `lz4-c-a4`{}), and `libzstd {zs}` (`zstd-c-l-3`, `zstd-c-l1`, `zstd-c-l3`, `zstd-dict-l3`{}).\n- **Dictionary accounting:** `zstd-dict-*` trains up to a `32 KiB` dictionary via `ZDICT_trainFromBuffer` (charging `dict_bytes` in compressed size); steady-state `ZSTD_CDict` / `ZSTD_DDict` throughput is reported in `Enc` / `Dec`, with one-time `ZDICT` training time reported separately in `Dict Train (ms)`. `VXG v5` embeds its self-trained `<= 32 KiB` prefix dictionary inside the `.vxg` frame itself, and `VXG Enc` / `Dec` include full dictionary training and parsing time.\n- **Whole-file reference (`random access = NO`):** `zstd-whole-l19-long` (`zstd -19 --long`, `128 MiB` window, single stream) quantifies the compression-ratio difference of removing `{}`-byte block boundaries.\n- Bold cells mark the best block-codec metric in each row; codecs storing near-raw bytes (`ratio >= 0.90`, `< 10%` size reduction) are excluded from speed/trade-off/energy bolding.\n",
        c.block_size,
        if has_heavy { ", `lz4hc-c-l9`" } else { "" },
        if has_heavy {
            ", `zstd-c-l9`, `zstd-dict-l9`"
        } else {
            ""
        },
        c.block_size
    )?;

    writeln!(
        f,
        "### 1. Individual Codec Ratio & Throughput (`GB/s`)\n"
    )?;
    writeln!(
        f,
        "| Dataset | Size | VXG Ratio | LZ4 (a1) | Zstd-1 | Zstd-9 | Zstd+Dict-3 | Zstd+Dict-9 | Zstd-19 `--long` *(whole)* | VXG Enc | LZ4 Enc | Zstd-1 Enc | Zstd+D3 Enc | VXG Dec | LZ4 Dec | Zstd-1 Dec | Zstd+D3 Dec | Dict Train (`ms`) |"
    )?;
    writeln!(
        f,
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    )?;
    for s in summaries {
        write_table1_row(
            &mut f,
            &format!("**`{}`**", short_name(&s.name)),
            &fmt_size(s.raw_bytes),
            s,
        )?;
    }
    let groups = summary_groups(summaries);
    for (label, grp) in &groups {
        let gm = synth_geomean_summary(grp);
        write_table1_row(&mut f, label, "—", &gm)?;
    }

    writeln!(
        f,
        "\n### 2. Parameter-Sweep Upper-Bound Envelopes (`min_ratio` + `max_speed` across Independent Blocks)\n\nEach sweep envelope combines the best block ratio achieved by any tested level in that family with the fastest encode and decode throughput achieved by any preset in that family (`LZ4 Sweep` = `[a1, a4, HC-L9]`, `Zstd+Dict Sweep` = `[-3, 1, 3, 9, dict-3, dict-9]`, `All-Block Sweep` = union of both). Bold cells mark the actual per-axis winner (`Ratio`, `Enc`, `Dec`).\n"
    )?;
    writeln!(
        f,
        "| Dataset | VXG Ratio | LZ4 Sweep Ratio | Zstd+Dict Sweep Ratio | VXG Enc | LZ4 Sweep Enc | Zstd+Dict Sweep Enc | VXG Dec | LZ4 Sweep Dec | Zstd+Dict Sweep Dec | VXG `VPI` | LZ4 Sweep `VPI` | Zstd+Dict Sweep `VPI` | All-Block Sweep `VPI` |"
    )?;
    writeln!(
        f,
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    )?;
    let write_env_row = |f: &mut fs::File, label: &str, v: CodecPoint, lc: CodecPoint, zc: CodecPoint, ac: CodecPoint| -> R<()> {
        let ratios = [v.ratio, lc.ratio, zc.ratio];
        let encs = [(v.enc, v.ratio), (lc.enc, lc.ratio), (zc.enc, zc.ratio)];
        let decs = [(v.dec, v.ratio), (lc.dec, lc.ratio), (zc.dec, zc.ratio)];
        let vpis = [
            (point_vpi(v), v.ratio),
            (point_vpi(lc), lc.ratio),
            (point_vpi(zc), zc.ratio),
            (point_vpi(ac), ac.ratio),
        ];
        writeln!(
            f,
            "| {label} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            fmt_min(v.ratio, &ratios, 4),
            fmt_min(lc.ratio, &ratios, 4),
            fmt_min(zc.ratio, &ratios, 4),
            fmt_max_compressed(v.enc, v.ratio, &encs, 3),
            fmt_max_compressed(lc.enc, lc.ratio, &encs, 3),
            fmt_max_compressed(zc.enc, zc.ratio, &encs, 3),
            fmt_max_compressed(v.dec, v.ratio, &decs, 3),
            fmt_max_compressed(lc.dec, lc.ratio, &decs, 3),
            fmt_max_compressed(zc.dec, zc.ratio, &decs, 3),
            fmt_max_compressed(point_vpi(v), v.ratio, &vpis, 1),
            fmt_max_compressed(point_vpi(lc), lc.ratio, &vpis, 1),
            fmt_max_compressed(point_vpi(zc), zc.ratio, &vpis, 1),
            fmt_max_compressed(point_vpi(ac), ac.ratio, &vpis, 1),
        )?;
        Ok(())
    };
    for s in summaries {
        write_env_row(
            &mut f,
            &format!("**`{}`**", short_name(&s.name)),
            s.vxg,
            s.lz4_envelope(),
            s.zstd_block_envelope(),
            s.all_block_envelope(),
        )?;
    }
    for (label, grp) in &groups {
        let gm = synth_geomean_summary(grp);
        let lc = CodecPoint {
            ratio: geomean(grp.iter().map(|s| s.lz4_envelope().ratio)),
            enc: geomean(grp.iter().map(|s| s.lz4_envelope().enc)),
            dec: geomean(grp.iter().map(|s| s.lz4_envelope().dec)),
            enc_j_gb: None,
            dec_j_gb: None,
        };
        let zc = CodecPoint {
            ratio: geomean(grp.iter().map(|s| s.zstd_block_envelope().ratio)),
            enc: geomean(grp.iter().map(|s| s.zstd_block_envelope().enc)),
            dec: geomean(grp.iter().map(|s| s.zstd_block_envelope().dec)),
            enc_j_gb: None,
            dec_j_gb: None,
        };
        let ac = CodecPoint {
            ratio: geomean(grp.iter().map(|s| s.all_block_envelope().ratio)),
            enc: geomean(grp.iter().map(|s| s.all_block_envelope().enc)),
            dec: geomean(grp.iter().map(|s| s.all_block_envelope().dec)),
            enc_j_gb: None,
            dec_j_gb: None,
        };
        write_env_row(&mut f, label, gm.vxg, lc, zc, ac)?;
    }

    writeln!(
        f,
        "\n### 3. Composite Trade-Off Indices (`VPI` & `BCI`)\n\n> Note: `VPI = sqrt(Encode_GBps * Decode_GBps) / Ratio` and `BCI = ((Encode_GBps * Decode_GBps) / Ratio)^(1/3)` summarize the joint space-speed trade-off; a higher `VPI` or `BCI` does not imply faster raw throughput than `LZ4` on rows where `LZ4` leads in `Enc` or `Dec` above.\n"
    )?;
    writeln!(
        f,
        "| Dataset | VXG `VPI` | LZ4 (a1) `VPI` | Zstd-1 `VPI` | Zstd+Dict-3 `VPI` | VXG `BCI` | LZ4 (a1) `BCI` | Zstd+Dict-3 `BCI` |"
    )?;
    writeln!(
        f,
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    )?;
    let write_idx_row = |f: &mut fs::File, label: &str, s: &DatasetSummary| -> R<()> {
        let vpis = [
            (point_vpi(s.vxg), s.vxg.ratio),
            (point_vpi(s.lz4_a1), s.lz4_a1.ratio),
            (point_vpi(s.zstd_1), s.zstd_1.ratio),
            (point_vpi(s.zstd_dict_3), s.zstd_dict_3.ratio),
        ];
        let bcis = [
            (point_bci(s.vxg), s.vxg.ratio),
            (point_bci(s.lz4_a1), s.lz4_a1.ratio),
            (point_bci(s.zstd_dict_3), s.zstd_dict_3.ratio),
        ];
        writeln!(
            f,
            "| {label} | {} | {} | {} | {} | {} | {} | {} |",
            fmt_max_compressed(point_vpi(s.vxg), s.vxg.ratio, &vpis, 1),
            fmt_max_compressed(point_vpi(s.lz4_a1), s.lz4_a1.ratio, &vpis, 1),
            fmt_max_compressed(point_vpi(s.zstd_1), s.zstd_1.ratio, &vpis, 1),
            fmt_max_compressed(point_vpi(s.zstd_dict_3), s.zstd_dict_3.ratio, &vpis, 1),
            fmt_max_compressed(point_bci(s.vxg), s.vxg.ratio, &bcis, 2),
            fmt_max_compressed(point_bci(s.lz4_a1), s.lz4_a1.ratio, &bcis, 2),
            fmt_max_compressed(point_bci(s.zstd_dict_3), s.zstd_dict_3.ratio, &bcis, 2),
        )?;
        Ok(())
    };
    for s in summaries {
        write_idx_row(&mut f, &format!("**`{}`**", short_name(&s.name)), s)?;
    }
    for (label, grp) in &groups {
        let gm = synth_geomean_summary(grp);
        write_idx_row(&mut f, label, &gm)?;
    }

    if summaries.iter().any(|s| s.vxg.enc_j_gb.is_some()) {
        writeln!(
            f,
            "\n### 4. Per-Process CPU Energy (`J/GB`, Lower is Better) & Green Codec Index (`GCI`)\n"
        )?;
        writeln!(
            f,
            "| Dataset | VXG Enc `J/GB` | LZ4 Enc `J/GB` | Zstd+D3 Enc `J/GB` | VXG Dec `J/GB` | LZ4 Dec `J/GB` | Zstd+D3 Dec `J/GB` | VXG `GCI` | LZ4 `GCI` | Zstd+D3 `GCI` |"
        )?;
        writeln!(
            f,
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
        )?;
        for s in summaries {
            let ej = [
                (s.vxg.enc_j_gb.unwrap_or(0.0), s.vxg.ratio),
                (s.lz4_a1.enc_j_gb.unwrap_or(0.0), s.lz4_a1.ratio),
                (s.zstd_dict_3.enc_j_gb.unwrap_or(0.0), s.zstd_dict_3.ratio),
            ];
            let dj = [
                (s.vxg.dec_j_gb.unwrap_or(0.0), s.vxg.ratio),
                (s.lz4_a1.dec_j_gb.unwrap_or(0.0), s.lz4_a1.ratio),
                (s.zstd_dict_3.dec_j_gb.unwrap_or(0.0), s.zstd_dict_3.ratio),
            ];
            let gcis = [
                (point_gci(s.vxg).unwrap_or(0.0), s.vxg.ratio),
                (point_gci(s.lz4_a1).unwrap_or(0.0), s.lz4_a1.ratio),
                (point_gci(s.zstd_dict_3).unwrap_or(0.0), s.zstd_dict_3.ratio),
            ];
            writeln!(
                f,
                "| **`{}`** | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
                short_name(&s.name),
                fmt_min_compressed(ej[0].0, ej[0].1, &ej, 3),
                fmt_min_compressed(ej[1].0, ej[1].1, &ej, 3),
                fmt_min_compressed(ej[2].0, ej[2].1, &ej, 3),
                fmt_min_compressed(dj[0].0, dj[0].1, &dj, 3),
                fmt_min_compressed(dj[1].0, dj[1].1, &dj, 3),
                fmt_min_compressed(dj[2].0, dj[2].1, &dj, 3),
                fmt_max_compressed(gcis[0].0, gcis[0].1, &gcis, 1),
                fmt_max_compressed(gcis[1].0, gcis[1].1, &gcis, 1),
                fmt_max_compressed(gcis[2].0, gcis[2].1, &gcis, 1),
            )?;
        }
    }

    f.flush()?;
    Ok(())
}
