//! Dense & recurrent per-head neural operators (`RMSNorm`, `RoPE`, `Gated-DeltaNet`, `Gated-GQA`),
//! token sampling, and teacher-forced logit KL/cosine evaluation metrics.

use crate::fp16::ErrorMetrics;
use crate::gguf_meta::ModelConfig;

pub type CpuStateVecs = (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>);

pub fn alloc_cpu_state_vecs(cfg: &ModelConfig) -> CpuStateVecs {
    let mut k_caches = Vec::with_capacity(cfg.n_layer);
    let mut v_caches = Vec::with_capacity(cfg.n_layer);
    let mut conv_states = Vec::with_capacity(cfg.n_layer);
    let mut ssm_states = Vec::with_capacity(cfg.n_layer);
    let conv_ch = cfg.ssm_inner_size + 2 * cfg.ssm_group_count * cfg.ssm_state_size;
    let conv_hist = cfg.ssm_conv_kernel.saturating_sub(1) * conv_ch;
    let dv = if cfg.ssm_inner_size > 0 {
        cfg.ssm_inner_size / cfg.ssm_dt_rank.max(1)
    } else {
        0
    };
    let ssm_len = cfg.ssm_dt_rank * dv * cfg.ssm_state_size;
    for il in 0..cfg.n_layer {
        if cfg.is_recurrent_layer(il) {
            k_caches.push(Vec::new());
            v_caches.push(Vec::new());
            conv_states.push(vec![0.0f32; conv_hist]);
            ssm_states.push(vec![0.0f32; ssm_len]);
        } else {
            k_caches.push(Vec::with_capacity(256 * cfg.n_head_kv * cfg.head_dim));
            v_caches.push(Vec::with_capacity(256 * cfg.n_head_kv * cfg.head_dim));
            conv_states.push(Vec::new());
            ssm_states.push(Vec::new());
        }
    }
    (k_caches, v_caches, conv_states, ssm_states)
}

#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn rms_norm(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let mut sq = 0.0f32;
    for &v in x {
        sq = v.mul_add(v, sq);
    }
    let scale = 1.0 / ((sq / (x.len() as f32)) + eps).sqrt();
    for ((o, &xv), &wv) in out.iter_mut().zip(x).zip(w) {
        *o = xv * scale * wv;
    }
}

pub fn rms_norm_inplace(x: &mut [f32], w: &[f32], eps: f32) {
    let mut sq = 0.0f32;
    for &v in x.iter() {
        sq = v.mul_add(v, sq);
    }
    let scale = 1.0 / ((sq / (x.len() as f32)) + eps).sqrt();
    for (xv, &wv) in x.iter_mut().zip(w) {
        *xv = *xv * scale * wv;
    }
}

pub fn l2_norm_scaled(x: &[f32], eps: f32, mul: f32, out: &mut [f32]) {
    let mut sq = 0.0f32;
    for &v in x {
        sq = v.mul_add(v, sq);
    }
    let scale = mul / (sq + eps).sqrt();
    for (o, &xv) in out.iter_mut().zip(x) {
        *o = xv * scale;
    }
}

pub fn apply_rope(vec: &mut [f32], pos: usize, rope_dim: usize, freq_base: f32, neox: bool) {
    let rdim = rope_dim.min(vec.len());
    let half = rdim / 2;
    for i in 0..half {
        let freq = freq_base.powf(-(2.0 * (i as f32)) / (rdim as f32));
        let theta = (pos as f32) * freq;
        let (sin_t, cos_t) = theta.sin_cos();
        if neox {
            let x0 = vec[i];
            let x1 = vec[i + half];
            vec[i] = x0 * cos_t - x1 * sin_t;
            vec[i + half] = x0 * sin_t + x1 * cos_t;
        } else {
            let x0 = vec[2 * i];
            let x1 = vec[2 * i + 1];
            vec[2 * i] = x0 * cos_t - x1 * sin_t;
            vec[2 * i + 1] = x0 * sin_t + x1 * cos_t;
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn step_deltanet_recurrent(
    qkv_mixed: &[f32],
    z: &[f32],
    alpha_raw: &[f32],
    beta_raw: &[f32],
    ssm_conv1d: &[f32],
    ssm_a: &[f32],
    ssm_dt_bias: &[f32],
    ssm_norm: &[f32],
    conv_state: &mut [f32],
    ssm_state: &mut [f32],
    n_kh: usize,
    n_vh: usize,
    dk: usize,
    dv: usize,
    k_sz: usize,
    eps: f32,
    y_concat: &mut [f32],
) {
    let conv_ch = qkv_mixed.len();
    let hist = k_sz - 1;
    let mut conv_silu = vec![0.0f32; conv_ch];
    for c in 0..conv_ch {
        let w = &ssm_conv1d[c * k_sz..(c + 1) * k_sz];
        let mut acc = 0.0f32;
        for t in 0..hist {
            acc = conv_state[t * conv_ch + c].mul_add(w[t], acc);
        }
        acc = qkv_mixed[c].mul_add(w[hist], acc);
        for t in 0..(hist - 1) {
            conv_state[t * conv_ch + c] = conv_state[(t + 1) * conv_ch + c];
        }
        conv_state[(hist - 1) * conv_ch + c] = qkv_mixed[c];
        conv_silu[c] = silu(acc);
    }

    let qk_dim = n_kh * dk;
    let d_inner = n_vh * dv;
    let q_all = &conv_silu[0..qk_dim];
    let k_all = &conv_silu[qk_dim..2 * qk_dim];
    let v_all = &conv_silu[2 * qk_dim..2 * qk_dim + d_inner];

    let q_scale = 1.0 / (dk as f32).sqrt();
    let mut q_h = vec![0.0f32; dk];
    let mut k_h = vec![0.0f32; dk];
    let mut o_h = vec![0.0f32; dv];

    for h in 0..n_vh {
        let hk = h / (n_vh / n_kh);
        l2_norm_scaled(&q_all[hk * dk..(hk + 1) * dk], eps, q_scale, &mut q_h);
        l2_norm_scaled(&k_all[hk * dk..(hk + 1) * dk], eps, 1.0, &mut k_h);
        let v_h = &v_all[h * dv..(h + 1) * dv];

        let beta = sigmoid(beta_raw[h]);
        let x_sp = alpha_raw[h] + ssm_dt_bias[h];
        let sp = if x_sp > 20.0 {
            x_sp
        } else if x_sp < -20.0 {
            x_sp.exp()
        } else {
            (1.0 + x_sp.exp()).ln()
        };
        let decay = (sp * ssm_a[h]).exp();

        let sh = &mut ssm_state[h * dv * dk..(h + 1) * dv * dk];
        for jv in 0..dv {
            let row = &mut sh[jv * dk..(jv + 1) * dk];
            let mut sk = 0.0f32;
            for ik in 0..dk {
                let sd = row[ik] * decay;
                row[ik] = sd;
                sk = sd.mul_add(k_h[ik], sk);
            }
            let d_jv = (v_h[jv] - sk) * beta;
            let mut o_jv = 0.0f32;
            for ik in 0..dk {
                let sn = d_jv.mul_add(k_h[ik], row[ik]);
                row[ik] = sn;
                o_jv = sn.mul_add(q_h[ik], o_jv);
            }
            o_h[jv] = o_jv;
        }

        rms_norm_inplace(&mut o_h, ssm_norm, eps);
        let z_h = &z[h * dv..(h + 1) * dv];
        let dst_h = &mut y_concat[h * dv..(h + 1) * dv];
        for jv in 0..dv {
            dst_h[jv] = o_h[jv] * silu(z_h[jv]);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn step_full_gqa(
    q_raw: &[f32],
    k_cur: &mut [f32],
    v_cur: &[f32],
    q_norm: Option<&[f32]>,
    k_norm: Option<&[f32]>,
    gated_q: bool,
    nh: usize,
    nkv: usize,
    hd: usize,
    pos: usize,
    rope_dim: usize,
    rope_freq_base: f32,
    rope_neox: bool,
    eps: f32,
    k_cache: &mut Vec<f32>,
    v_cache: &mut Vec<f32>,
    att_cat: &mut [f32],
) {
    let mut q_cur = vec![0.0f32; nh * hd];
    let mut q_gate = vec![0.0f32; if gated_q { nh * hd } else { 0 }];
    if gated_q {
        for h in 0..nh {
            q_cur[h * hd..(h + 1) * hd].copy_from_slice(&q_raw[h * 2 * hd..h * 2 * hd + hd]);
            q_gate[h * hd..(h + 1) * hd].copy_from_slice(&q_raw[h * 2 * hd + hd..(h + 1) * 2 * hd]);
        }
    } else {
        q_cur.copy_from_slice(q_raw);
    }

    if let Some(qn) = q_norm {
        for h in 0..nh {
            rms_norm_inplace(&mut q_cur[h * hd..(h + 1) * hd], qn, eps);
        }
    }
    if let Some(kn) = k_norm {
        for h in 0..nkv {
            rms_norm_inplace(&mut k_cur[h * hd..(h + 1) * hd], kn, eps);
        }
    }
    for h in 0..nh {
        apply_rope(
            &mut q_cur[h * hd..(h + 1) * hd],
            pos,
            rope_dim,
            rope_freq_base,
            rope_neox,
        );
    }
    for h in 0..nkv {
        apply_rope(
            &mut k_cur[h * hd..(h + 1) * hd],
            pos,
            rope_dim,
            rope_freq_base,
            rope_neox,
        );
    }

    k_cache.extend_from_slice(k_cur);
    v_cache.extend_from_slice(v_cur);
    let seq_len = pos + 1;
    let kv_stride = nkv * hd;
    let scale = 1.0 / (hd as f32).sqrt();
    let mut scores = vec![0.0f32; seq_len];

    for h in 0..nh {
        let kv_h = h / (nh / nkv);
        let qh = &q_cur[h * hd..(h + 1) * hd];
        let mut max_s = f32::NEG_INFINITY;
        for t in 0..seq_len {
            let kh = &k_cache[t * kv_stride + kv_h * hd..t * kv_stride + (kv_h + 1) * hd];
            let mut dot = 0.0f32;
            for d in 0..hd {
                dot = qh[d].mul_add(kh[d], dot);
            }
            let s = dot * scale;
            scores[t] = s;
            if s > max_s {
                max_s = s;
            }
        }
        let mut sum_exp = 0.0f32;
        for s in &mut scores {
            *s = (*s - max_s).exp();
            sum_exp += *s;
        }
        let inv_sum = 1.0 / sum_exp;
        let out_h = &mut att_cat[h * hd..(h + 1) * hd];
        out_h.fill(0.0);
        for t in 0..seq_len {
            let w = scores[t] * inv_sum;
            let vh = &v_cache[t * kv_stride + kv_h * hd..t * kv_stride + (kv_h + 1) * hd];
            for d in 0..hd {
                out_h[d] = w.mul_add(vh[d], out_h[d]);
            }
        }
        if gated_q {
            let gh = &q_gate[h * hd..(h + 1) * hd];
            for d in 0..hd {
                out_h[d] *= sigmoid(gh[d]);
            }
        }
    }
}

pub fn sample_logits(logits: &[f32], temp: f32, rng_state: &mut u64) -> u32 {
    sample_logits_top_k_p(logits, temp, 40, 1.0, rng_state)
}

/// Zero-allocation O(V) top-k + nucleus (top-p) sampler matching HF/vLLM TopK -> TopP order.
pub fn sample_logits_top_k_p(
    logits: &[f32],
    temp: f32,
    top_k: usize,
    top_p: f32,
    rng_state: &mut u64,
) -> u32 {
    if logits.is_empty() {
        return 0;
    }
    if temp <= 1e-5 || top_k <= 1 {
        let mut best_i = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > best_v {
                best_v = v;
                best_i = i;
            }
        }
        return best_i as u32;
    }

    let k = top_k.min(64).min(logits.len());
    let mut top = [(0u32, f32::NEG_INFINITY); 64];
    let mut filled = 0usize;
    let mut min_pos = 0usize;
    let mut min_val = f32::NEG_INFINITY;

    for (i, &v) in logits.iter().enumerate() {
        if !v.is_finite() {
            continue;
        }
        if filled < k {
            top[filled] = (i as u32, v);
            filled += 1;
            if filled == k {
                min_pos = 0;
                min_val = top[0].1;
                for (j, item) in top[..k].iter().enumerate().skip(1) {
                    if item.1 < min_val {
                        min_val = item.1;
                        min_pos = j;
                    }
                }
            }
        } else if v > min_val {
            top[min_pos] = (i as u32, v);
            min_pos = 0;
            min_val = top[0].1;
            for (j, item) in top[..k].iter().enumerate().skip(1) {
                if item.1 < min_val {
                    min_val = item.1;
                    min_pos = j;
                }
            }
        }
    }

    if filled == 0 {
        return 0;
    }
    let cand = &mut top[..filled];
    cand.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

    let max_v = cand[0].1;
    let inv_t = 1.0 / temp;
    let mut total_k = 0.0f32;
    for item in cand.iter_mut() {
        let p = ((item.1 - max_v) * inv_t).exp();
        item.1 = p;
        total_k += p;
    }

    let mut kept_sum = total_k;
    let mut kept_len = cand.len();
    if top_p > 0.0 && top_p < 0.9999 && total_k > 0.0 {
        let thresh = total_k * top_p;
        let mut cum = 0.0f32;
        kept_len = 0;
        for item in cand.iter() {
            cum += item.1;
            kept_len += 1;
            if cum >= thresh {
                break;
            }
        }
        kept_sum = cum;
    }

    *rng_state = rng_state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let r = ((*rng_state >> 40) as f32 / ((1u32 << 24) as f32)) * kept_sum;
    let mut acc = 0.0f32;
    for &(id, p) in &cand[..kept_len] {
        acc += p;
        if acc >= r {
            return id;
        }
    }
    cand[0].0
}

/// Sparse OpenAI `presence_penalty` + DRY (Don't Repeat Yourself) suffix-clone anti-loop tracker.
pub struct TokenPenaltyTracker {
    counts: Vec<u16>,
    is_word: Vec<u8>, // 0 = unknown, 1 = whitespace/exempt, 2 = symbol/digit, 3 = word
    unique: Vec<u32>,
    history: Vec<u32>,
    exempt: Vec<u32>,
}

impl TokenPenaltyTracker {
    pub fn new(n_vocab: usize, exempt: &[u32]) -> Self {
        Self {
            counts: vec![0u16; n_vocab],
            is_word: vec![0u8; n_vocab],
            unique: Vec::with_capacity(512),
            history: Vec::with_capacity(1024),
            exempt: exempt.to_vec(),
        }
    }

    pub fn clear_counts(&mut self) {
        for &id in &self.unique {
            if let Some(slot) = self.counts.get_mut(id as usize) { *slot = 0; }
        }
        self.unique.clear();
        self.history.clear();
    }

    fn classify(&mut self, tok: u32, tokenizer: &crate::tokenizer::BpeTokenizer) -> u8 {
        let idx = tok as usize;
        if idx >= self.is_word.len() || self.exempt.contains(&tok) { return 1; }
        if self.is_word[idx] != 0 { return self.is_word[idx]; }
        let piece = tokenizer.decode(&[tok]);
        let kind = if piece.is_empty() || piece.chars().all(char::is_whitespace) { 1 }
            else if piece.chars().any(char::is_alphabetic) { 3 }
            else { 2 };
        self.is_word[idx] = kind;
        kind
    }

    pub fn record(&mut self, tok: u32, tokenizer: &crate::tokenizer::BpeTokenizer) {
        self.history.push(tok);
        if self.classify(tok, tokenizer) == 3 {
            let idx = tok as usize;
            if self.counts[idx] == 0 {
                self.unique.push(tok);
            }
            self.counts[idx] = self.counts[idx].saturating_add(1);
        }
    }

    pub fn apply(&self, logits: &mut [f32], presence_penalty: f32, frequency_penalty: f32) {
        if presence_penalty > 0.0 || frequency_penalty > 0.0 {
            for &tid in &self.unique {
                let idx = tid as usize;
                let cnt = self.counts[idx];
                if cnt >= 1 && idx < logits.len() {
                    let freq = ((cnt - 1).min(12)) as f32;
                    logits[idx] -= presence_penalty + frequency_penalty * freq;
                }
            }
        }
        let n = self.history.len();
        if n >= 10 {
            let last = self.history[n - 1];
            if self.history[n - 10..].iter().all(|&t| t == last) && (last as usize) < logits.len() {
                logits[last as usize] -= 14.0;
            }
        }
        if n >= 6 {
            let start = n.saturating_sub(1024);
            let last_tok = self.history[n - 1];
            let mut cands: [(u32, usize); 16] = [(0, 0); 16];
            let mut n_cand = 0usize;
            for i in (start + 4)..(n - 1) {
                if self.history[i] != last_tok {
                    continue;
                }
                let nx = self.history[i + 1];
                if self.exempt.contains(&nx)
                    || ((nx as usize) < self.is_word.len() && self.is_word[nx as usize] == 1)
                {
                    continue;
                }
                let max_b = (i - start + 1).min(48);
                let mut mlen = 1usize;
                while mlen < max_b && self.history[i - mlen] == self.history[n - 1 - mlen] {
                    mlen += 1;
                }
                if mlen >= 5 {
                    if let Some(s) = cands[..n_cand].iter_mut().find(|s| s.0 == nx) {
                        s.1 = s.1.max(mlen);
                    } else if n_cand < cands.len() {
                        cands[n_cand] = (nx, mlen);
                        n_cand += 1;
                    }
                }
            }
            for &(nx, mlen) in &cands[..n_cand] {
                if (nx as usize) < logits.len() {
                    logits[nx as usize] -= 0.85 * 1.8f32.powi((mlen - 5).min(8) as i32);
                }
            }
        }
    }
}

/// Detects 14x identical token stutters or 3x consecutive periodic phrase loops `[B][B][B]` (`p >= 4`).
pub fn has_severe_loop(ids: &[u32]) -> bool {
    let n = ids.len();
    if n >= 14 {
        let last = ids[n - 1];
        if ids[n - 14..].iter().all(|&t| t == last) {
            return true;
        }
    }
    let max_p = (n / 3).min(40);
    for p in 4..=max_p {
        let b0 = &ids[n - p..];
        let b1 = &ids[n - 2 * p..n - p];
        if b0 == b1 && b0 == &ids[n - 3 * p..n - 2 * p] {
            return true;
        }
    }
    false
}

/// Strips `<think>...</think>` blocks from historical assistant turns per Qwen3.5 multi-turn spec.
pub fn strip_think_block(text: &str) -> &str {
    let trimmed = text.trim_start();
    if let Some(rest) = trimmed.strip_prefix("<think>") {
        return match rest.split_once("</think>") {
            Some((_, after)) => after.trim_start_matches(['\r', '\n']),
            None => "",
        };
    }
    text
}

#[derive(Clone, Copy, Debug)]
pub struct LogitEvalMetrics {
    pub kl_div: f64,
    pub cosine_sim: f64,
    pub rmse: f64,
    pub top1_match: bool,
}

pub fn compare_logits(ref_logits: &[f32], test_logits: &[f32]) -> LogitEvalMetrics {
    let em = ErrorMetrics::compute(ref_logits, test_logits);
    let max_r = ref_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let max_t = test_logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum_r: f64 = ref_logits.iter().map(|&v| ((v as f64) - max_r).exp()).sum();
    let sum_t: f64 = test_logits
        .iter()
        .map(|&v| ((v as f64) - max_t).exp())
        .sum();
    let lse_r = max_r + sum_r.ln();
    let lse_t = max_t + sum_t.ln();

    let mut kl = 0.0f64;
    let (mut top_r, mut best_r) = (0usize, f32::NEG_INFINITY);
    let (mut top_t, mut best_t) = (0usize, f32::NEG_INFINITY);
    for (i, (&rv, &tv)) in ref_logits.iter().zip(test_logits.iter()).enumerate() {
        if rv > best_r {
            best_r = rv;
            top_r = i;
        }
        if tv > best_t {
            best_t = tv;
            top_t = i;
        }
        let log_p = (rv as f64) - lse_r;
        let p = log_p.exp();
        if p > 1e-15 {
            let log_q = (tv as f64) - lse_t;
            kl += p * (log_p - log_q);
        }
    }
    LogitEvalMetrics {
        kl_div: kl.max(0.0),
        cosine_sim: em.cosine_sim,
        rmse: em.rmse,
        top1_match: top_r == top_t,
    }
}
