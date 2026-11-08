"""One streaming CPU FP32 calibration; observers never modify model computations.
Input coordinates are original official Linear pre-input / exported Rust g.
Only block32 energy, no full Hessian or activation retention.
"""
import argparse,hashlib,inspect,json,resource,time
from pathlib import Path
import numpy as np
import torch,transformers
from safetensors import safe_open
from transformers import Qwen3_5ForCausalLM,Qwen3_5TextConfig
from tokenizers import Tokenizer
TEXTS=[
 'A lighthouse keeper recorded wind direction each morning and repaired the lamp before sunset.',
 'Bir marangoz küçük masayı onardı ve kalan tahtaları kuru bir yerde sakladı.',
 'The museum catalog lists clay bowls, bronze tools, and maps drawn beside the river.',
 'def merge_sorted(left, right):\n    result = []\n    while left and right:\n        result.append(left.pop(0) if left[0] < right[0] else right.pop(0))',
 'A garden has seven rows with eight seedlings in each row; rain falls during the night.',
 'Elektrik devresinde direnç yükselince sabit gerilim altında akım azalır. Ölçümleri tabloya yaz.',
 'Describe how ocean tides change over a day using clear observations from a coastal harbor.',
 'A scheduler stores pending jobs in a queue and marks each completed job with a timestamp.'
]
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('manifest');p.add_argument('output');a=p.parse_args();r=Path(a.root);out=Path(a.output);assert not out.exists()
 manifest=json.loads(Path(a.manifest).read_text());torch.set_num_threads(2);torch.set_num_interop_threads(1)
 cfg=Qwen3_5TextConfig(**json.loads((r/'official-config.json').read_text())['text_config']);cfg._attn_implementation='eager'
 with torch.device('meta'):model=Qwen3_5ForCausalLM(cfg)
 expected=model.state_dict();mapping={}
 with safe_open(r/'official-model.safetensors',framework='pt',device='cpu') as f:
  for k in f.keys():
   if not k.startswith('model.language_model.'):continue
   target='model.'+k[len('model.language_model.'):];assert target in expected and list(f.get_slice(k).get_shape())==list(expected[target].shape);mapping[k]=target
  assert len(mapping)==320 and set(expected)==set(mapping.values())|{'lm_head.weight'}
  sd={t:f.get_tensor(k).float() for k,t in mapping.items()}
 sd['lm_head.weight']=sd['model.embed_tokens.weight'];model.load_state_dict(sd,strict=True,assign=True);del sd,expected
 model.model.rotary_emb=type(model.model.rotary_emb)(cfg);model.eval();assert not any(t.is_meta for t in list(model.parameters())+list(model.buffers()))
 stats={};handles=[];module_map={}
 def observer(key,g):
  cols=g['cols'];div=torch.ones(cols,dtype=torch.float64)
  for c,v in g['scales']:div[c]=v
  sums=torch.zeros((cols+31)//32,dtype=torch.float64);stats[key]={'samples':0,'sum':sums,'calls':0,'cols':cols}
  def hook(module,args):
   x=args[0].detach();assert x.shape[-1]==cols and torch.isfinite(x).all();z=x.reshape(-1,cols).double()/div
   assert cols%32==0
   sums.add_(z.square().reshape(-1,cols//32,32).sum((0,2))/32);stats[key]['samples']+=z.shape[0];stats[key]['calls']+=1
   # returning None is essential: no replacement args, no mutation.
  return hook
 for key,g in manifest['groups'].items():
  family,i=key.split('.');i=int(i);layer=model.model.layers[i]
  if family=='attn':
   name=f'model.layers.{i}.linear_attn.in_proj_qkv' if cfg.layer_types[i]=='linear_attention' else f'model.layers.{i}.self_attn.q_proj'
  elif family=='ffn_gu':name=f'model.layers.{i}.mlp.gate_proj'
  elif family=='ssm_out':
   assert cfg.layer_types[i]=='linear_attention';name=f'model.layers.{i}.linear_attn.out_proj'
  else:raise AssertionError(key)
  module=model.get_submodule(name);assert isinstance(module,torch.nn.Linear) and module.in_features==g['cols'];module_map[key]=name;handles.append(module.register_forward_pre_hook(observer(key,g)))
 tok=Tokenizer.from_file(str(r/'official-tokenizer.json'));ids=[tok.encode(t).ids[:32] for t in TEXTS];assert sum(map(len,ids))<=256
 # Persist preregistered inputs/policy before any model run.
 plan={'texts':TEXTS,'ids':ids,'manifest':manifest,'module_map':module_map,'criterion':'discovery mean KL improves AND top1 >= baseline; then validation same criterion; only then one production diagnostic','families_fallback':['tied embedding/head','ffn_down','full_attention output projection']}
 Path(str(out)+'.plan.json').write_text(json.dumps(plan,indent=2)+'\n')
 start=time.time()
 with torch.inference_mode():
  for i,tokens in enumerate(ids):
   cache=None
   for token in tokens:
    res=model(input_ids=torch.tensor([[token]]),past_key_values=cache,use_cache=True,logits_to_keep=1);cache=res.past_key_values
   print('CALIB',i,'tokens',len(tokens),'elapsed',time.time()-start,flush=True)
 for h in handles:h.remove()
 for key,s in stats.items():
  assert s['samples']==sum(map(len,ids)) and s['calls']==s['samples']
  vals=(s.pop('sum')/s['samples']).tolist();assert np.isfinite(vals).all() and min(vals)>=0 and sum(vals)>0;s['block_second_moment']=vals
 src=Path(inspect.getfile(Qwen3_5ForCausalLM))
 profile={'manifest':manifest,'stats':stats,'provenance':{'source_sha256':sha(r/'official-model.safetensors'),'config_sha256':sha(r/'official-config.json'),'tokenizer_sha256':sha(r/'official-tokenizer.json'),'manifest_sha256':sha(a.manifest),'calibration_plan_sha256':sha(str(out)+'.plan.json'),'observer_sha256':sha(__file__),'modeling_sha256':sha(src),'torch':torch.__version__,'transformers':transformers.__version__,'coordinate_proof':'unchanged source model Linear x; observer x/g from exact Rust smoothing plan; orthogonal sign/FWHT32 preserves block energy','module_map':module_map,'elapsed_s':time.time()-start,'maxrss':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}}
 with out.open('x') as f:json.dump(profile,f,indent=2);f.write('\n')
 print('PROFILE',len(stats),'groups',sum(map(len,ids)),'tokens',flush=True)
if __name__=='__main__':main()
