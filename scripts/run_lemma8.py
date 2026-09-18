"""HarmBench fixed-state tests of Lemma 8, using Section 4's cached heads.

Construct one displacement on the last input token, then reuse it on nested
supports at prediction step 128. This tests the schedule lemma, not inherited
trajectory effects or an independently rematched longer-prompt construction.
"""
from __future__ import annotations
import argparse
import contextlib
import json
import sys
import numpy as np
from prefix.attention_bounds import construct_shift, diameter
from prefix.schedule_bounds import evaluate_schedules
from prefix.notify import notify_on_exit
from prefix.runner import tee_stdout
from run_section5_rc import ROOT, digest, positions, resume_units, wait_for_file


def conditions():
    grid = {(8,1,g,a,'generated') for g in [0,1,4,16,64,127] for a in [0,.25,.5,1,2]}
    grid |= {(m,1,127,1,'generated') for m in [1,8,32,128]}
    grid |= {(8,k,127,1,'generated') for k in [4,16,64]}
    grid |= {(8,k,127,a,'full') for k in [1,-1] for a in [0,.25,.5,1,2]}
    return [dict(id=f'm{m}_k{k}_g{g}_a{a:g}_{schedule}',m=m,k=k,g=g,alpha=a,schedule=schedule)
            for m,k,g,a,schedule in sorted(grid)]


def supports(n,c):
    k = n if c['k']==-1 else c['k']
    if k>n:
        return None
    short = list(range(n-k,n))
    long = list(range(n+127)) if c['schedule']=='full' else short+list(range(n,n+c['g']))
    return short,long


def analyze(limit,start,stride):
    import torch
    torch.set_num_threads(1)
    torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
    weights_path=ROOT/'checkpoints/section4_long_weights.pt'
    weights=torch.load(weights_path,weights_only=True)
    source=json.loads((ROOT/'results/section4_long_manifest.json').read_text())
    manifest=dict(version=1,source=source,weights_sha256=digest(weights_path),conditions=conditions(),
                  step=128,visible_generated=127,epsilon_thresholds=[.001,.01,.05,.1],
                  query='common query at y127; linear projections before QK norm and RoPE',
                  displacement='last-input-token match at first-step query; same alpha*r on nested supports')
    for index in range(start,limit,stride):
        path=ROOT/f'checkpoints/section4_long_activations_{index:03d}.pt'
        wait_for_file(path)
        meta={**manifest,'index':index,'activation_sha256':digest(path)}
        result=ROOT/f'results/lemma8_{index:03d}.json'
        units=[f'{layer}_{head}' for layer in source['layers'] for head in source['heads']]
        old=json.loads(result.read_text()) if result.exists() else None
        if old and old['manifest']==meta and set(old['units'])==set(units):
            print(f'lemma8 {index}: complete, skipped',flush=True)
            continue
        payload=torch.load(path,weights_only=True)
        if len(payload['generated_ids'])!=128:
            raise ValueError('Expected 128-token cache')
        n=payload['base_length']
        def compute(head_id):
            layer,head=map(int,head_id.split('_'))
            h=payload['layers'][layer].float().numpy().astype(np.float64)
            w={key:weights[head_id][key].numpy().astype(np.float64) for key in ['wk','wv','wq']}
            base,_,target=positions(n,8)
            keys,values=h[base]@w['wk'].T,h[base]@w['wv'].T
            q0,q=w['wq']@h[n-1],w['wq']@h[target]
            b=w['wk'].T@q0
            z=b-w['wv'].T@np.linalg.lstsq(w['wv'].T,b,rcond=None)[0]
            shifts={m:construct_shift(h,w['wk'],w['wv'],q0,[n-1],positions(n,m)[1],z)
                    for m in [1,8,32,128]}
            diameters={}
            grid={c['id']:c for c in conditions()}
            def compute_condition(cid):
                c=grid[cid]
                selected=supports(n,c)
                if selected is None:
                    return {**c,'eligible':False,'reason':'input shorter than requested support'}
                short,long=selected
                r=shifts[c['m']]*c['alpha']
                dk,dv=w['wk']@r,w['wv']@r
                cache_key=(c['m'],c['k'],c['alpha'])
                if cache_key not in diameters:
                    vs=values.copy();vs[short]+=dv
                    diameters[cache_key]=diameter(vs)
                row=evaluate_schedules(keys,values,q,short,long,dk,dv,diameters[cache_key])
                tolerance=1e-8*max(1.,row['diameter_short'],row['value_shift_norm'])
                if (row['error']>row['bound_shares']+tolerance or
                    row['bound_shares']>row['bound_score']+tolerance or
                    row['bound_shares']>row['bound_epsilon']+tolerance or
                    row['identity_residual']>tolerance):
                    raise AssertionError(f'Lemma 8 failed: {index}/{head_id}/{cid}')
                return {**c,**row,'eligible':True,'short_count':len(short),'long_count':len(long),
                        'epsilon_bounds':{str(e):e*(row['value_shift_norm']+row['diameter_short'])
                                          if row['epsilon_observed']<=e else None
                                          for e in manifest['epsilon_thresholds']}}
            checkpoint=ROOT/f'checkpoints/lemma8_{index:03d}_{head_id}.json'
            saved=resume_units(checkpoint,{**meta,'head_id':head_id},list(grid),compute_condition)
            return dict(index=index,behavior_id=payload['behavior_id'],layer=layer,head=head,
                        base_length=n,rows=list(saved['units'].values()))
        resume_units(result,meta,units,compute)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit',type=int,default=400)
    parser.add_argument('--start',type=int,default=0)
    parser.add_argument('--stride',type=int,default=1)
    args=parser.parse_args()
    log=ROOT/f'logs/lemma8_{args.start}.log'
    with tee_stdout(log),contextlib.redirect_stderr(sys.stdout),notify_on_exit('lemma8',log_file=str(log)):
        analyze(args.limit,args.start,args.stride)


if __name__=='__main__':
    main()
