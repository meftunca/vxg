//! Replay exact external-oracle token IDs; retain every vocabulary logit as LE f32.
use std::{error::Error, fs, io::Write};
use vxg_tensor::llm::LlmModel;
fn main() {
    if let Err(e) = run() {
        eprintln!("FAIL: {e}");
        std::process::exit(1)
    }
}
fn ids(v: &serde_json::Value, key: &str) -> Result<Vec<u32>, Box<dyn Error>> {
    let a = v[key].as_array().ok_or("missing token array")?;
    if a.is_empty() || a.len() > 2048 {
        return Err("token array length outside 1..2048".into());
    }
    a.iter()
        .map(|x| {
            x.as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| "invalid token id".into())
        })
        .collect()
}
fn run() -> Result<(), Box<dyn Error>> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 4 {
        return Err("usage: trace-model MODEL TRACE.json OUTPUT.f32".into());
    }
    let v: serde_json::Value = serde_json::from_slice(&fs::read(&a[2])?)?;
    let prompt = ids(&v, "prompt_ids")?;
    let teachers = ids(&v, "teacher_ids")?;
    let mut m = LlmModel::load_file(&a[1], true, 2)?;
    if !m.backend_label().contains("1-CmdBuf Fused") {
        return Err("fused Metal required".into());
    }
    if prompt
        .iter()
        .chain(&teachers)
        .any(|&t| t as usize >= m.cfg.n_vocab)
    {
        return Err("token outside vocabulary".into());
    }
    if v["shape"] != serde_json::json!([teachers.len(), m.cfg.n_vocab]) {
        return Err("trace shape mismatch".into());
    }
    let mut state = m.new_state();
    let mut logits = vec![0f32; m.cfg.n_vocab];
    for (i, &tok) in prompt.iter().enumerate() {
        m.forward_step(
            &mut state,
            tok,
            (i + 1 == prompt.len()).then_some(logits.as_mut_slice()),
        )?;
    }
    let mut out = fs::File::create_new(&a[3])?;
    for (i, &tok) in teachers.iter().enumerate() {
        if !logits.iter().all(|x| x.is_finite()) {
            return Err("nonfinite logits".into());
        }
        for x in &logits {
            out.write_all(&x.to_le_bytes())?;
        }
        println!("STEP {i} teacher={tok}");
        if i + 1 < teachers.len() {
            m.forward_step(&mut state, tok, Some(&mut logits))?;
        }
    }
    out.sync_all()?;
    println!(
        "TRACE backend={} shape=[{},{}]",
        m.backend_label(),
        teachers.len(),
        m.cfg.n_vocab
    );
    Ok(())
}
#[cfg(test)]
mod tests {
    #[test]
    fn reject_bad_ids() {
        for v in [
            serde_json::json!({"x":[]}),
            serde_json::json!({"x":[-1]}),
            serde_json::json!({"x":[1.5]}),
            serde_json::json!({"x":[4294967296u64]}),
        ] {
            assert!(super::ids(&v, "x").is_err());
        }
        assert_eq!(
            super::ids(&serde_json::json!({"x":[0,248044]}), "x").unwrap(),
            vec![0, 248044]
        );
    }
}
