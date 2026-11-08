//! Fixed retained-token replay; model loading and prefill excluded from decode timing.
use std::{error::Error, fs, time::Instant};
use vxg_tensor::llm::LlmModel;
fn main() -> Result<(), Box<dyn Error>> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 3 {
        return Err("usage: bench-replay MODEL TRACE.json".into());
    }
    let v: serde_json::Value = serde_json::from_slice(&fs::read(&a[2])?)?;
    let ids = |key: &str| -> Result<Vec<u32>, Box<dyn Error>> {
        v[key]
            .as_array()
            .ok_or("missing IDs")?
            .iter()
            .map(|x| {
                x.as_u64()
                    .and_then(|n| u32::try_from(n).ok())
                    .ok_or_else(|| "invalid ID".into())
            })
            .collect()
    };
    let prompt = ids("prompt_ids")?;
    let teacher = ids("teacher_ids")?;
    if prompt.is_empty() || teacher.is_empty() {
        return Err("empty IDs".into());
    }
    let load = Instant::now();
    let mut m = LlmModel::load_file(&a[1], true, 2)?;
    println!(
        "backend={} load_s={:.6}",
        m.backend_label(),
        load.elapsed().as_secs_f64()
    );
    if !m.backend_label().contains("1-CmdBuf Fused") {
        return Err("fused Metal required".into());
    }
    if prompt
        .iter()
        .chain(&teacher)
        .any(|&t| t as usize >= m.cfg.n_vocab)
    {
        return Err("ID outside vocabulary".into());
    }
    let mut logits = vec![0.0; m.cfg.n_vocab];
    for sample in 0..4 {
        let mut seconds = 0.0;
        for _ in 0..8 {
            let mut state = m.new_state();
            for &t in &prompt {
                m.forward_step(&mut state, t, None)?;
            }
            let start = Instant::now();
            for &t in &teacher {
                m.forward_step(&mut state, t, Some(&mut logits))?;
            }
            seconds += start.elapsed().as_secs_f64();
            if !logits.iter().all(|x| x.is_finite()) {
                return Err("nonfinite logits".into());
            }
            std::hint::black_box(&logits);
        }
        println!(
            "sample={} warmup={} forwards={} seconds={:.9} tps={:.6}",
            sample,
            sample == 0,
            8 * teacher.len(),
            seconds,
            (8 * teacher.len()) as f64 / seconds
        );
    }
    Ok(())
}
