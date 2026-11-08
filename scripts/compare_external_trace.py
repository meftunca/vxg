"""Compare complete vocabulary traces without changing the frozen acceptance gate."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('candidate');a=p.parse_args();d=Path(a.directory)
m=json.loads((d/'trace.json').read_text());shape=tuple(m['shape'])
r=np.fromfile(d/'hf-logits.f32',dtype='<f4').reshape(shape).astype(np.float64)
t=np.fromfile(a.candidate,dtype='<f4').reshape(shape).astype(np.float64)
assert np.isfinite(r).all() and np.isfinite(t).all()
rows=[]
for i,(x,y) in enumerate(zip(r,t)):
    diff=x-y
    lx=x-(x.max()+np.log(np.exp(x-x.max()).sum()));ly=y-(y.max()+np.log(np.exp(y-y.max()).sum()))
    row=dict(step=i,teacher=m['teacher_ids'][i],reference_top1=int(x.argmax()),candidate_top1=int(y.argmax()),max_abs=float(abs(diff).max()),rmse=float(np.sqrt(np.mean(diff*diff))),cosine=float(np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))),kl=float(np.sum(np.exp(lx)*(lx-ly))))
    rows.append(row);print(json.dumps(row))
result=dict(finite=True,shape=shape,steps=rows,max_abs=max(v['max_abs'] for v in rows),max_rmse=max(v['rmse'] for v in rows),min_cosine=min(v['cosine'] for v in rows),max_kl=max(v['kl'] for v in rows),top1_hits=sum(v['reference_top1']==v['candidate_top1'] for v in rows))
result['frozen_gate_pass']=result['max_rmse']<=1e-4 and result['min_cosine']>=.99999 and result['top1_hits']==len(rows)
Path(a.candidate+'.comparison.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='steps'}));raise SystemExit(0 if result['frozen_gate_pass'] else 1)
