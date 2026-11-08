//! Bounded A/B/C diagnostic. Every stage is a separate process; no raw archive written.
use std::{error::Error,fs,io::{Write,BufWriter},path::Path};
use vxg_tensor::llm::LlmModel;
fn top(l:&[f32])->u32 { l.iter().enumerate().max_by(|a,b|a.1.total_cmp(b.1)).unwrap().0 as u32 }
fn prefill(m:&mut LlmModel,ids:&[u32])->Result<(vxg_tensor::llm::LlmState,Vec<f32>),Box<dyn Error>> {
 let mut s=m.new_state(); let mut l=vec![0.;m.cfg.n_vocab];
 for (i,&t) in ids.iter().enumerate(){m.forward_step(&mut s,t,(i+1==ids.len()).then_some(l.as_mut_slice()))?;} Ok((s,l))
}
fn main(){if let Err(e)=run(){eprintln!("FAIL: {e}");std::process::exit(1)}}
fn run()->Result<(),Box<dyn Error>> {
 let a:Vec<_>=std::env::args().collect(); if a.len()!=5{return Err("usage: quality-ablation A|B|C MODEL PROMPTS.json NEW_OUTPUT_DIR".into())}
 let stage=&a[1]; if !["A","B","C"].contains(&stage.as_str()){return Err("invalid stage".into())}
 let d=Path::new(&a[4]);fs::create_dir(d)?;
 let (mut m,audit)=if stage=="B" {let (m,v)=LlmModel::load_vxt_decoded_raw_diagnostic(&a[2],true,2)?;(m,Some(v))}else{(LlmModel::load_file(&a[2],true,2)?,None)};
 let backend=m.backend_label();let expected=if stage=="C"{"VXT-Turbo Pipelined"}else{"1-CmdBuf Fused"};
 if !backend.contains(expected){return Err(format!("unexpected backend {backend}; required {expected}").into())}
 println!("BACKEND {stage}: {backend}");
 if let Some(v)=audit{fs::write(d.join("decoded-inventory.json"),serde_json::to_vec_pretty(&v)?)?;}
 let prompts:serde_json::Value=serde_json::from_slice(&fs::read(&a[3])?)?;
 let reference=d.parent().ok_or("no parent")?.join("A");let mut result=serde_json::Map::new();
 for (name,prompt) in prompts.as_object().ok_or("prompts object required")? {
  let ids=m.tokenizer.encode_chat(prompt.as_str().ok_or("prompt string required")?,&m.cfg.arch,false);
  let teachers:Option<Vec<u32>>=if stage=="A"{None}else{let v:serde_json::Value=serde_json::from_slice(&fs::read(reference.join(format!("{name}.json")))?)?;
   if v["prompt_ids"]!=serde_json::json!(ids){return Err("tokenizer prompt mismatch".into())}Some(serde_json::from_value(v["teacher_ids"].clone())?)};
  let (mut s,mut l)=prefill(&mut m,&ids)?;let mut teacher_ids=Vec::new();let mut top_ids=Vec::new();
  let mut out=BufWriter::new(fs::OpenOptions::new().write(true).create_new(true).open(d.join(format!("{name}.f32")))?);
  for i in 0..16 {if !l.iter().all(|x|x.is_finite()){return Err("nonfinite logits".into())}let t=teachers.as_ref().map_or_else(||top(&l),|ts|ts[i]);teacher_ids.push(t);top_ids.push(top(&l));
   for x in &l{out.write_all(&x.to_le_bytes())?;}if i<15{m.forward_step(&mut s,t,Some(&mut l))?;}
  }out.flush()?;
  let trace=serde_json::json!({"prompt":prompt,"prompt_ids":ids,"teacher_ids":teacher_ids,"top_ids":top_ids,"shape":[16,m.cfg.n_vocab],"backend":backend});
  fs::write(d.join(format!("{name}.json")),serde_json::to_vec_pretty(&trace)?)?;
  result.insert(name.clone(),trace);println!("TRACE {name} complete");
 }
 let p=prompts["math"].as_str().ok_or("math prompt missing")?;let ids=m.tokenizer.encode_chat(p,&m.cfg.arch,false);let(mut s,mut l)=prefill(&mut m,&ids)?;
 let stops=[m.cfg.eos_token_id,m.tokenizer.token_id("<|im_end|>").unwrap_or(m.cfg.eos_token_id),m.tokenizer.token_id("<|endoftext|>").unwrap_or(m.cfg.eos_token_id)];
 let mut generated=Vec::new();let mut stop=None;
 for _ in 0..256{if !l.iter().all(|x|x.is_finite()){return Err("nonfinite generation logits".into())}let t=top(&l);if stops.contains(&t){stop=Some(t);break}generated.push(t);if generated.len()<256{m.forward_step(&mut s,t,Some(&mut l))?;}}
 let g=serde_json::json!({"prompt_ids":ids,"token_ids":generated,"text":m.tokenizer.decode(&generated),"stop_id":stop,"cap":256,"state_pos":s.pos,"backend":backend});
 fs::write(d.join("math-generation.json"),serde_json::to_vec_pretty(&g)?)?;println!("GENERATION {}",g);
 fs::write(d.join("summary.json"),serde_json::to_vec_pretty(&serde_json::json!({"stage":stage,"backend":backend,"cases":result}))?)?;Ok(())
}
