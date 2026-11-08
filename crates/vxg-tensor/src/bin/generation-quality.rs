//! Frozen known-suite exploratory free-generation diagnostic.
use std::{error::Error,fs,path::Path,time::Instant};
use vxg_tensor::llm::LlmModel;
const CAP:usize=384;
const INSTRUCTION:&str="You may reason briefly; end with Final answer: <A/B/C/D>.";
// Strict suffix, single marker: conflicting or repeated finals never score.
fn final_label(s:&str)->Option<usize>{
 let marker="Final answer:";
 if s.matches(marker).count()!=1{return None}
 let (_,tail)=s.split_once(marker)?;
 let t=tail.trim();
 if t.len()!=1{return None}
 match t.as_bytes()[0]{b'A'..=b'D'=>Some((t.as_bytes()[0]-b'A')as usize),_=>None}
}
fn best(l:&[f32])->Result<u32,Box<dyn Error>>{
 if !l.iter().all(|x|x.is_finite()){return Err("nonfinite logits".into())}
 let mut b=0;for i in 1..l.len(){if l[i]>l[b]{b=i}}Ok(b as u32)
}
fn turbo_stage(stage:&str)->bool{matches!(stage,"baseline"|"zero")}
fn expected_backend(stage:&str)->&'static str{if turbo_stage(stage){"VXT-Turbo Pipelined"}else{"1-CmdBuf Fused"}}
fn main(){if let Err(e)=run(){eprintln!("FAIL: {e}");std::process::exit(1)}}
fn run()->Result<(),Box<dyn Error>>{
 let a:Vec<_>=std::env::args().collect();if a.len()!=5{return Err("usage: generation-quality A|baseline|zero|cb4|cb4-greedy MODEL SUITE ROOT".into())}
 if !["A","baseline","zero","cb4","cb4-greedy"].contains(&a[1].as_str()){return Err("stage".into())}
 let root=Path::new(&a[4]);let d=root.join(&a[1]);fs::create_dir_all(&d)?;
 let mut m=LlmModel::load_file(&a[2],true,2)?;let backend=m.backend_label();
 if a[1]=="cb4-greedy"{m.enable_cb4_greedy()?;}
 let expected=expected_backend(&a[1]);if !backend.contains(expected){return Err(format!("unexpected backend {backend}").into())}
 println!("BACKEND {backend}");
 let suite:serde_json::Value=serde_json::from_slice(&fs::read(&a[3])?)?;
 let config=serde_json::json!({"arch":m.cfg.arch,"vocab":m.cfg.n_vocab,"eos":m.cfg.eos_token_id});
 let stop_ids=vec![m.cfg.eos_token_id,m.tokenizer.token_id("<|im_end|>").unwrap_or(m.cfg.eos_token_id),m.tokenizer.token_id("<|endoftext|>").unwrap_or(m.cfg.eos_token_id)];
 let mut rows=Vec::new();
 for item in suite["items"].as_array().ok_or("items")?{
  let id=item["id"].as_str().ok_or("id")?;let out=d.join(format!("{id}.json"));
  let mut p=format!("{}\n",item["question"].as_str().ok_or("question")?);
  for(i,c)in item["choices"].as_array().ok_or("choices")?.iter().enumerate(){p.push_str(&format!("{}. {}\n",(b'A'+i as u8)as char,c.as_str().ok_or("choice")?));}
  p.push_str(INSTRUCTION);let ids=m.tokenizer.encode_chat(&p,&m.cfg.arch,false);
  if a[1]!="A"{let r:serde_json::Value=serde_json::from_slice(&fs::read(root.join("A").join(format!("{id}.json")))?)?;if r["prompt_ids"]!=serde_json::json!(ids)||r["config"]!=config||r["stop_ids"]!=serde_json::json!(stop_ids){return Err("source token/config mismatch".into())}}
  if out.exists(){let r:serde_json::Value=serde_json::from_slice(&fs::read(&out)?)?;if r["prompt_ids"]!=serde_json::json!(ids)||r["config"]!=config||r["cap"]!=CAP{return Err("resume mismatch".into())}rows.push(r);continue}
  let start=Instant::now();let mut state=m.new_state();let mut logits=vec![0.;m.cfg.n_vocab];m.prefill(&mut state,&ids,&mut logits)?;let prefill_s=start.elapsed().as_secs_f64();
  let decode=Instant::now();let mut callback_ids=Vec::new();
  let emitted=if a[1]=="cb4-greedy"{m.generate_greedy_cb4(&mut state,best(&logits)?,CAP,&stop_ids,|t,_|{callback_ids.push(t);true})?}else if turbo_stage(&a[1]){m.generate_greedy_turbo(&mut state,&ids,best(&logits)?,CAP,true,0.,&stop_ids,|t|{callback_ids.push(t);true})?}else{
   let mut v=Vec::new();for i in 0..CAP{let t=best(&logits)?;if stop_ids.contains(&t){break}v.push(t);if i+1<CAP{m.forward_step(&mut state,t,Some(&mut logits))?;}}v
  };
  if (turbo_stage(&a[1])||a[1]=="cb4-greedy")&&callback_ids!=emitted{return Err("callback mismatch".into())}
  let decode_s=decode.elapsed().as_secs_f64();let text=m.tokenizer.decode(&emitted);let truncated=emitted.len()==CAP;
  let predicted=final_label(&text);let gold=item["answer"].as_u64().ok_or("answer")? as usize;
  let status=if truncated{"truncated"}else{match predicted{Some(x)if x==gold=>"completed_correct",Some(_)=>"completed_wrong",None=>"format_unscorable"}};
  let row=serde_json::json!({"id":id,"category":item["category"],"answer":gold,"predicted":predicted,"status":status,"prompt":p,"prompt_ids":ids,"generated_ids":emitted,"text":text,"stop_reason":if truncated{"cap"}else{"stop_id"},"cap":CAP,"emitted_count":emitted.len(),"processed_position":state.pos,"processed_generation_count":state.pos-ids.len(),"prefill_s":prefill_s,"decode_s":decode_s,"sample_s":start.elapsed().as_secs_f64(),"backend":backend,"decode_path":if a[1]=="cb4-greedy"{"cb4-greedy-full-head"}else if turbo_stage(&a[1]){"turbo-fast-default-sketch"}else{"generic-forward-step"},"config":config,"stop_ids":stop_ids,"temperature":0,"presence_penalty":0});
  let tmp=out.with_extension("json.tmp");fs::write(&tmp,serde_json::to_vec_pretty(&row)?)?;fs::rename(tmp,out)?;println!("{} {status} emitted={} pos={}",id,emitted.len(),state.pos);rows.push(row);
 }
 fs::write(d.join("summary.json"),serde_json::to_vec_pretty(&serde_json::json!({"stage":a[1],"rows":rows}))?)?;Ok(())
}
#[test]fn strict_final_parser(){
 assert_eq!(final_label("Reason 42 A B C.\nFinal answer: D\n"),Some(3));
 for s in ["A","Final answer: 3","Final answer: A or B","Final answer: A.","Final answer: A\nmore","Final answer: A\nFinal answer: B","Final answer: A\nFinal answer: A","Final answer: a","Expected answer A; result 42"]{assert_eq!(final_label(s),None,"{s}");}
}

#[test]fn explicit_decode_routes(){for stage in ["A","cb4","cb4-greedy"]{assert!(!turbo_stage(stage));assert_eq!(expected_backend(stage),"1-CmdBuf Fused");}for stage in ["baseline","zero"]{assert!(turbo_stage(stage));assert_eq!(expected_backend(stage),"VXT-Turbo Pipelined");}}
