#!/usr/bin/env python3
"""Compare official Qwen3.5-2B text trunk against BF16 GGUF, without model runtimes.
Pinned conversion rules/provenance are recorded in the qualification report.
No implicit name matching: unsupported names/types/layouts fail closed.
"""
import argparse
import collections
import hashlib
import json
import math
import mmap
import struct
from pathlib import Path


def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def gguf(mm):
    pos = 0
    def take(n):
        nonlocal pos
        if pos+n > len(mm): raise ValueError('truncated GGUF')
        b=mm[pos:pos+n];pos+=n;return b
    def integer(fmt): return struct.unpack('<'+fmt,take(struct.calcsize('<'+fmt)))[0]
    def string(): return take(integer('Q')).decode('utf-8')
    formats={0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
    def value(kind):
        if kind in formats: return integer(formats[kind])
        if kind==8:return string()
        if kind==9:
            subtype=integer('I');count=integer('Q')
            # Discard array values; no tokenizer materialization needed.
            for _ in range(count):value(subtype)
            return None
        raise ValueError(f'unsupported metadata kind {kind}')
    if take(4)!=b'GGUF':raise ValueError("not GGUF")
    version=integer('I')
    if version not in (2,3):raise ValueError('unsupported GGUF version')
    count=integer('Q');kvcount=integer('Q');alignment=32
    for _ in range(kvcount):
        name=string();v=value(integer('I'))
        if name=='general.alignment':alignment=v
    tensors={}
    for _ in range(count):
        name=string();shape=[integer('Q') for _ in range(integer('I'))];kind=integer('I');offset=integer('Q')
        if name in tensors:raise ValueError('duplicate GGUF tensor')
        if kind not in (0,1,30):raise ValueError(f'unsupported GGUF dtype {kind}')
        tensors[name]={'shape':list(reversed(shape)),'kind':kind,'offset':offset,'numel':math.prod(shape)}
    base=(pos+alignment-1)//alignment*alignment
    for t in tensors.values():
        t['start']=base+t['offset'];t['size']=t['numel']*(4 if t['kind']==0 else 2)
        if t['start']+t['size']>len(mm):raise ValueError('out of bounds GGUF')
    return tensors


def target_name(name):
    prefix='model.language_model.'
    if not name.startswith(prefix):raise ValueError(f'not language trunk: {name}')
    n=name[len(prefix):]
    if n=='embed_tokens.weight':return 'token_embd.weight'
    if n=='norm.weight':return 'output_norm.weight'
    parts=n.split('.',2)
    if len(parts)!=3 or parts[0]!='layers' or not parts[1].isdigit():raise ValueError(f'unmapped {name}')
    suffix={
      'input_layernorm.weight':'attn_norm.weight',
      'post_attention_layernorm.weight':'post_attention_norm.weight',
      'mlp.down_proj.weight':'ffn_down.weight','mlp.gate_proj.weight':'ffn_gate.weight','mlp.up_proj.weight':'ffn_up.weight',
      'self_attn.q_proj.weight':'attn_q.weight','self_attn.k_proj.weight':'attn_k.weight','self_attn.v_proj.weight':'attn_v.weight','self_attn.o_proj.weight':'attn_output.weight',
      'self_attn.q_norm.weight':'attn_q_norm.weight','self_attn.k_norm.weight':'attn_k_norm.weight',
      'linear_attn.in_proj_qkv.weight':'attn_qkv.weight','linear_attn.in_proj_z.weight':'attn_gate.weight',
      'linear_attn.in_proj_a.weight':'ssm_alpha.weight','linear_attn.in_proj_b.weight':'ssm_beta.weight',
      'linear_attn.out_proj.weight':'ssm_out.weight','linear_attn.conv1d.weight':'ssm_conv1d.weight',
      'linear_attn.norm.weight':'ssm_norm.weight','linear_attn.A_log':'ssm_a','linear_attn.dt_bias':'ssm_dt.bias',
    }.get(parts[2])
    if suffix is None:raise ValueError(f'unmapped {name}')
    return f'blk.{parts[1]}.{suffix}'


def source_value(raw,dtype,i):
    if dtype=='F32':return struct.unpack_from('<f',raw,i*4)[0]
    if dtype=='BF16':return struct.unpack('<f',struct.pack('<I',struct.unpack_from('<H',raw,i*2)[0]<<16))[0]
    raise ValueError(f'unsupported source dtype {dtype}')


def compare(st,gg,cfg):
    h=struct.unpack_from('<Q',st)[0];header=json.loads(st[8:8+h]);base=8+h
    g=gguf(gg);results=[];excluded=[];mapped=set()
    if not cfg['text_config']['linear_num_key_heads']==cfg['text_config']['linear_num_value_heads']==16:raise ValueError('V head reordering unsupported for unequal head counts')
    if not (cfg['tie_word_embeddings'] is True and cfg['text_config']['tie_word_embeddings'] is True):raise ValueError('untied head unsupported')
    for name,s in header.items():
        if name=='__metadata__':continue
        if name.startswith(('model.visual.','mtp.')):
            excluded.append({'name':name,'dtype':s['dtype'],'shape':s['shape'],'numel':math.prod(s['shape'])});continue
        dest=target_name(name)
        if dest in mapped:raise ValueError('duplicate mapped destination')
        mapped.add(dest);t=g.get(dest)
        if t is None:raise ValueError(f'missing GGUF {dest}')
        shape=s['shape'];expected_shape=[d for d in shape if d!=1] if '.conv1d.' in name else shape
        if expected_shape!=t['shape']:raise ValueError(f'shape mismatch {name}: {shape} -> {t["shape"]}')
        n=math.prod(shape)
        if n!=t['numel']:raise ValueError('element count mismatch')
        a,b=s['data_offsets'];raw=memoryview(st)[base+a:base+b]
        if s['dtype'] not in ('BF16','F32'):raise ValueError('unsupported official dtype')
        if len(raw)!=n*(2 if s['dtype']=='BF16' else 4):raise ValueError('source byte size mismatch')
        out=memoryview(gg)[t['start']:t['start']+t['size']]
        transform='identity'
        if name.endswith('.A_log'):transform='negative_exp_f32_scalar_oracle'
        elif name.endswith('norm.weight') and not name.endswith('linear_attn.norm.weight'):transform='add_one_f32'
        elif '.conv1d.' in name:transform='squeeze_only'
        mismatch=0;examples=[];maxabs=0.
        same_dtype=(s['dtype']=='BF16' and t['kind']==30) or (s['dtype']=='F32' and t['kind']==0)
        if same_dtype and transform in ('identity','squeeze_only'):
            for start in range(0,len(raw),1024*1024):
                if bytes(raw[start:start+1024*1024])!=bytes(out[start:start+1024*1024]):
                    width=2 if s['dtype']=='BF16' else 4
                    for offset in range(start,min(len(raw),start+1024*1024),width):
                        aa=bytes(raw[offset:offset+width]);bb=bytes(out[offset:offset+width])
                        if aa!=bb:
                            mismatch+=1
                            if len(examples)<4:examples.append({'index':offset//width,'source_hex':aa.hex(),'gguf_hex':bb.hex()})
        else:
            if t['kind']!=0:raise ValueError(f'unsupported transformed target dtype {dest}')
            for i in range(n):
                v=source_value(raw,s['dtype'],i)
                if transform=='add_one_f32':v=f32(v+1.)
                elif transform=='negative_exp_f32_scalar_oracle':v=f32(-math.exp(v))
                expected=struct.pack('<f',v);actual=bytes(out[i*4:i*4+4])
                if expected!=actual:
                    mismatch+=1;actual_v=struct.unpack('<f',actual)[0];maxabs=max(maxabs,abs(v-actual_v))
                    if len(examples)<4:examples.append({'index':i,'source_value':source_value(raw,s['dtype'],i),'expected_hex':expected.hex(),'gguf_hex':actual.hex(),'expected':v,'actual':actual_v})
        result={'source':name,'target':dest,'shape':shape,'target_shape':t['shape'],'source_dtype':s['dtype'],'target_kind':t['kind'],'transform':transform,'numel':n,'mismatches':mismatch,'maxabs':maxabs,'examples':examples}
        results.append(result);print(('PASS' if mismatch==0 else 'MISMATCH'),dest,transform,'numel',n,'mismatches',mismatch,flush=True)
        del raw,out
    extras=sorted(set(g)-mapped)
    if extras:raise ValueError(f'unmapped GGUF tensors: {extras}')
    return {'results':results,'excluded':excluded,'excluded_counts':dict(collections.Counter('visual' if x['name'].startswith('model.visual.') else 'mtp' for x in excluded)),'tied_head':'official tie_word_embeddings=true; no separate lm_head/output tensor; token_embd weight shared','all_text_tensors_mapped':True,'text_tensor_count':len(results),'text_numel':sum(x['numel'] for x in results),'mismatch_tensors':sum(x['mismatches']>0 for x in results),'mismatch_elements':sum(x['mismatches'] for x in results)}


def main():
    p=argparse.ArgumentParser();p.add_argument('official');p.add_argument('gguf');p.add_argument('config');p.add_argument('output');a=p.parse_args()
    with open(a.official,'rb') as sf,open(a.gguf,'rb') as gf:
        with mmap.mmap(sf.fileno(),0,access=mmap.ACCESS_READ) as st,mmap.mmap(gf.fileno(),0,access=mmap.ACCESS_READ) as gg:
            report=compare(st,gg,json.loads(Path(a.config).read_text()))
    Path(a.output).write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ('results','excluded')},indent=2))
    if report['mismatch_elements']:raise SystemExit(1)
if __name__=='__main__':main()
