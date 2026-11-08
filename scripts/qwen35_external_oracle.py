"""Bounded independent HF text-only CPU FP32 trace from original SafeTensors."""
import argparse, hashlib, inspect, json, platform, resource, time
from pathlib import Path
import numpy as np
import torch, transformers
from safetensors import safe_open
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

def main():
    p=argparse.ArgumentParser(); p.add_argument('source'); p.add_argument('config'); p.add_argument('tokenizer'); p.add_argument('output'); a=p.parse_args()
    out=Path(a.output); out.mkdir(exist_ok=True)
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    cfg=Qwen3_5TextConfig(**json.loads(Path(a.config).read_text())['text_config']); cfg._attn_implementation='eager'
    with torch.device('meta'): model=Qwen3_5ForCausalLM(cfg)
    # Official multimodal class owns model.language_model; text causal class owns model.
    expected=model.state_dict(); mapping={}
    with safe_open(a.source,framework='pt',device='cpu') as f:
        prefix='model.language_model.'
        source_keys=[k for k in f.keys() if k.startswith(prefix)]
        assert len(source_keys)==320, len(source_keys)
        for k in source_keys:
            target='model.'+k[len(prefix):]
            assert target in expected and list(f.get_slice(k).get_shape())==list(expected[target].shape), (k,target)
            mapping[k]=target
        assert set(expected)==set(mapping.values())|{'lm_head.weight'}
        sd={target:f.get_tensor(k).float() for k,target in mapping.items()}
    sd['lm_head.weight']=sd['model.embed_tokens.weight']
    model.load_state_dict(sd,strict=True,assign=True)
    # Meta construction leaves nonpersistent RoPE buffers; rebuild using official constructor.
    model.model.rotary_emb=type(model.model.rotary_emb)(cfg)
    assert not any(t.is_meta for t in list(model.parameters())+list(model.buffers()))
    model.eval(); del sd
    from tokenizers import Tokenizer
    ids=Tokenizer.from_file(a.tokenizer).encode('Lossless compression preserves every original weight bit. Memory bandwidth affects inference speed. Türkiye Ankara neural network exact weights.').ids[:16]
    assert len(ids)==16
    cache=None; trace=[]; teachers=[]; start=time.time()
    with torch.inference_mode():
        for prompt_step,token in enumerate(ids):
            result=model(input_ids=torch.tensor([[token]]),past_key_values=cache,use_cache=True)
            cache=result.past_key_values
            print('PROMPT',prompt_step,'elapsed',time.time()-start,flush=True)
        for step in range(8):
            logits=result.logits[0,-1].float(); assert torch.isfinite(logits).all()
            trace.append(logits.numpy().copy()); token=int(logits.argmax()); teachers.append(token)
            print('STEP',step,'top1',token,'elapsed',time.time()-start,flush=True)
            if step<7:
                result=model(input_ids=torch.tensor([[token]]),past_key_values=cache,use_cache=True); cache=result.past_key_values
    np.asarray(trace,dtype='<f4').tofile(out/'hf-logits.f32')
    src=Path(inspect.getfile(Qwen3_5ForCausalLM))
    metadata=dict(prompt_ids=ids,teacher_ids=teachers,shape=[8,cfg.vocab_size],dtype='little-endian-f32',transformers=transformers.__version__,torch=torch.__version__,torch_git=torch.version.git_version,platform=platform.platform(),modeling_source=str(src),modeling_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),mapping=mapping,elapsed_s=time.time()-start,maxrss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (out/'trace.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({k:v for k,v in metadata.items() if k!='mapping'}),flush=True)
if __name__=='__main__': main()
