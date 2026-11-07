use std::time::{Duration, Instant};

use super::{config::Config, harness::median, R};

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

pub(super) fn geomean(vals: impl Iterator<Item = f64>) -> f64 {
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
