//! Frozen single-label multiple-choice diagnostic, one model per process.
use std::{error::Error,fs,path::Path};
use vxg_tensor::llm::LlmModel;
fn scores(l:&[f32], ids:&[u32])->Result<(Vec<f64>,usize),Box<dyn Error>> {
 if !l.iter().all(|x|x.is_finite()) {return Err("nonfinite full-vocabulary logits".into())}
 let max=l.iter().copied().fold(f32::NEG_INFINITY,f32::max) as f64;
 let logz=max+l.iter().map(|&x|((x as f64)-max).exp()).sum::<f64>().ln();
 let s:Vec<_>=ids.iter().map(|&t|l[t as usize] as f64-logz).collect();
 let mut best=0;for i in 1..s.len(){if s[i]>s[best]{best=i}}
 Ok((s,best))
}
fn main(){if let Err(e)=run(){eprintln!("FAIL: {e}");std::process::exit(1)}}
fn run()->Result<(),Box<dyn Error>>{
 let a:Vec<_>=std::env::args().collect();if a.len()!=5{return Err("usage: quality-matrix A|baseline|zero|activation MODEL SUITE NEW_DIR".into())}
 let d=Path::new(&a[4]);fs::create_dir(d)?;
 let mut m=LlmModel::load_file(&a[2],true,2)?;let backend=m.backend_label();
 let expected=if a[1]=="A"{"1-CmdBuf Fused"}else{"VXT-Turbo Pipelined"};if !backend.contains(expected){return Err(format!("backend {backend}, expected {expected}").into())}
 println!("BACKEND {backend}");
 let suite:serde_json::Value=serde_json::from_slice(&fs::read(&a[3])?)?;
 let mut rows=Vec::new();
 for item in suite["items"].as_array().ok_or("items")?{
  let choices=item["choices"].as_array().ok_or("choices")?;
  let mut p=format!("Choose the correct option. Respond with only its letter (A, B, C, or D).\n{}\n",item["question"].as_str().ok_or("question")?);
  for (i,c) in choices.iter().enumerate(){p.push_str(&format!("{}. {}\n",(b'A'+i as u8)as char,c.as_str().ok_or("choice")?));}
  let chat=m.tokenizer.encode_chat(&p,&m.cfg.arch,false);
  let prefix=format!("{}Answer:\n",m.tokenizer.decode(&chat));let ids=m.tokenizer.encode(&prefix);
  if ids[..chat.len()]!=chat{return Err("chat prefix roundtrip mismatch".into())}
  let mut label_ids=Vec::new();
  for ch in ['A','B','C','D']{
   let combined=m.tokenizer.encode(&format!("{prefix}{ch}"));
   if combined.len()!=ids.len()+1 || combined[..ids.len()]!=ids{return Err("label prefix boundary is not exactly one token".into())}
   let id=combined[ids.len()];if m.tokenizer.decode(&[id])!=ch.to_string(){return Err("label decode mismatch".into())}label_ids.push(id);
  }
  let frozen=d.parent().ok_or("parent")?.join("A").join(format!("{}.json",item["id"].as_str().ok_or("id")?));
  if a[1]!="A"{let r:serde_json::Value=serde_json::from_slice(&fs::read(frozen)?)?;if r["prompt_ids"]!=serde_json::json!(ids)||r["label_ids"]!=serde_json::json!(label_ids)||r["config"]!=serde_json::json!({"arch":m.cfg.arch,"vocab":m.cfg.n_vocab,"eos":m.cfg.eos_token_id}){return Err("source token/config mismatch".into())}}
  let mut state=m.new_state();let mut l=vec![0.;m.cfg.n_vocab];
  for (i,&t)in ids.iter().enumerate(){m.forward_step(&mut state,t,(i+1==ids.len()).then_some(l.as_mut_slice()))?;}
  let(logprobs,best)=scores(&l,&label_ids)?;
  let row=serde_json::json!({"id":item["id"],"category":item["category"],"answer":item["answer"],"predicted":best,"correct":best==item["answer"].as_u64().ok_or("answer")? as usize,"prompt_ids":ids,"label_ids":label_ids,"choice_logits":label_ids.iter().map(|&id|l[id as usize]).collect::<Vec<_>>(),"full_vocab_logprobs":logprobs,"config":{"arch":m.cfg.arch,"vocab":m.cfg.n_vocab,"eos":m.cfg.eos_token_id},"backend":backend});
  fs::write(d.join(format!("{}.json",item["id"].as_str().unwrap())),serde_json::to_vec_pretty(&row)?)?;
  println!("{} {}",item["id"],row["correct"]);rows.push(row);
 }
 fs::write(d.join("summary.json"),serde_json::to_vec_pretty(&serde_json::json!({"stage":a[1],"backend":backend,"rows":rows}))?)?;Ok(())
}
#[test]fn synthetic_label_mapping(){let l=[-3.,4.,-8.,2.,1.,7.];let(s,b)=scores(&l,&[4,1,3,0]).unwrap();assert_eq!(b,1);assert!(s[1]>s[2]);let expected=4f64-(l.iter().map(|&x|(x as f64).exp()).sum::<f64>()).ln();assert!((s[1]-expected).abs()<1e-12);assert_eq!(scores(&[1.,1.,1.,1.],&[3,2,1,0]).unwrap().1,0);}
