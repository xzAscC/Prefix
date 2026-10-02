"""Rerun the dimension diagnostic with 100 distinct dataset-derived prompts.

Input/generated states and reference queries remain frozen. Only the appended
prompt states change. Results resume per example/head/configuration.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

from prefix.native_attention import NativeHead, LENGTHS, conditions
from prefix.runner import tee_stdout, write_json_atomic

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT/'data/section4_prompt_diversity_manifest.json'
PADDING = (' Additional context: interpret the complete request above together with its supplied context. '
           'Keep the stated entities, constraints, and qualifications in view. This sentence adds reading '
           'context only and does not introduce a separate request. If the request is unclear, acknowledge '
           'the uncertainty rather than inventing missing details. Read the complete text before forming '
           'a response, and distinguish the request itself from this supplementary explanatory material.')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_manifest(cohort, records, tokenizer):
    examples=[]
    for position, base in enumerate(cohort['examples']):
        source=cohort['examples'][(position+1)%len(cohort['examples'])]['index']
        record=records[source]
        text='\n\n'.join(s for s in [record['Behavior'].strip(),record.get('ContextString','').strip()] if s)
        ids=tokenizer.encode(text,add_special_tokens=False)
        original_length=len(ids); added=''
        while len(ids)<cohort['config']['prompt_slots']:
            added+=PADDING
            ids=tokenizer.encode(text+added,add_special_tokens=False)
        row=dict(input_index=base['index'],prompt_source_index=source,
                 source_behavior_id=record['BehaviorID'],prompt_text=text,added_text=added,
                 original_prompt_tokens=original_length,prompt_ids=ids[:cohort['config']['prompt_slots']],
                 base_ids=base['base_ids'],input_tokens=base['input_tokens'])
        row['fingerprint']=digest(row); examples.append(row)
    result=dict(version=1,base_cohort=cohort['fingerprint'],config=cohort['config'],examples=examples,
                design='Cyclic pairing: each original input receives the next dataset request, followed by its context. '
                       'One distinct prompt per input; not a crossed input-by-prompt experiment. '
                       'Original generated states and reference queries remain frozen.')
    result['fingerprint']=digest(result)
    return result


def completed_result(path, identity):
    if not path.exists():return None
    row=json.loads(path.read_text())
    if row['identity']!=identity:raise ValueError(f'result identity mismatch: {path}')
    return row if row.get('complete') else None


def rank_diagnostics(directions, rtol):
    singular=torch.linalg.svdvals(directions)
    threshold=rtol*float(singular[0]) if len(singular) else 0.
    rank=int((singular>threshold).sum())
    return dict(rank=rank,dimension=directions.shape[1]-rank,
                singular_values=singular.cpu().tolist(),rank_rtol=rtol,
                dimensions_by_rtol={f'{tol:g}':directions.shape[1]-int((singular>tol*singular[0]).sum())
                                    for tol in [1e-5,1e-6,1e-7,1e-8]})


def summarize(rows):
    groups={}
    for row in rows:
        meta=row['identity']; c=meta['condition']
        group=groups.setdefault(c['id'],dict(condition=c,examples={}))
        group['examples'].setdefault(meta['input_index'],[]).append(row['dimension'])
    result=[]
    for group in groups.values():
        values=[float(np.mean(v)) for v in group['examples'].values()]
        result.append({**group['condition'],'dimension':dict(n=len(values),mean=float(np.mean(values)),
                       std=float(np.std(values,ddof=1)) if len(values)>1 else 0.),
                       'individual_dimensions':sorted(set(v for a in group['examples'].values() for v in a))})
    return sorted(result,key=lambda r:(r['m'],r['b'],r['g']))


def selected_conditions():
    return [c for c in conditions() if c['b']==1 and (c['m']==4 or c['g']==0)]


def prepare():
    from transformers import AutoTokenizer
    cohort=json.loads((ROOT/'data/section4_native_cohort.json').read_text())
    records=json.loads((ROOT/'data/section4_harmbench.json').read_text())
    tokenizer=AutoTokenizer.from_pretrained(cohort['config']['model'],revision=cohort['config']['revision'],local_files_only=True)
    manifest=build_manifest(cohort,records,tokenizer)
    manifest['weights_sha256']=file_digest(ROOT/'checkpoints/section4_native_weights.pt')
    manifest['fingerprint']=digest({k:v for k,v in manifest.items() if k!='fingerprint'})
    if MANIFEST.exists() and json.loads(MANIFEST.read_text())!=manifest:
        raise ValueError('existing diversity manifest differs')
    write_json_atomic(MANIFEST,manifest)
    print(f'Prepared {len(manifest["examples"])} paired examples; '
          f'{sum(bool(r["added_text"]) for r in manifest["examples"])} short prompts padded.',flush=True)


def extract(manifest, indices, device):
    pending=[]
    for row in manifest['examples']:
        if row['input_index'] not in indices:continue
        path=ROOT/f'checkpoints/section4_prompt_diversity_activations_{row["input_index"]:03d}.pt'
        if path.exists():
            saved=torch.load(path,weights_only=True,mmap=True)
            if saved['fingerprint']!=row['fingerprint']:raise ValueError('activation identity mismatch')
            print(f'extract {row["input_index"]}: cached',flush=True)
        else:pending.append(row)
    if not pending:return
    from transformers import AutoModelForCausalLM
    config=manifest['config']
    model=AutoModelForCausalLM.from_pretrained(config['model'],revision=config['revision'],local_files_only=True,
                dtype=torch.bfloat16,attn_implementation='sdpa').to(device).eval()
    with torch.inference_mode():
        for row in pending:
            capture=[]
            def hook(_module,args,kwargs):
                capture.append(kwargs.get('hidden_states',args[0] if args else None).detach().cpu())
            handle=model.model.layers[config['layer']].self_attn.register_forward_pre_hook(hook,with_kwargs=True)
            try:
                ids=torch.tensor([row['base_ids']+row['prompt_ids']],device=device)
                model(ids,use_cache=False,logits_to_keep=1)
            finally:handle.remove()
            payload=dict(fingerprint=row['fingerprint'],prompt_hidden=capture[0][0,row['input_tokens']:].clone())
            assert payload['prompt_hidden'].shape[0]==config['prompt_slots']
            path=ROOT/f'checkpoints/section4_prompt_diversity_activations_{row["input_index"]:03d}.pt'
            temporary=path.with_suffix('.tmp')
            with temporary.open('wb') as stream:
                torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
            os.replace(temporary,path)
            write_json_atomic(ROOT/f'checkpoints/section4_prompt_diversity_extract_{row["input_index"]:03d}.json',
                              dict(complete=True,fingerprint=row['fingerprint'],activation_sha256=file_digest(path)))
            print(f'extract {row["input_index"]}: complete, source={row["prompt_source_index"]}',flush=True)


def analyze(manifest, indices, device):
    config=manifest['config']; path=ROOT/'checkpoints/section4_native_weights.pt'
    if file_digest(path)!=manifest['weights_sha256']:raise ValueError('wrong model weights')
    weights=torch.load(path,weights_only=True,mmap=True)
    implementation=dict(script=file_digest(__file__),native=file_digest(ROOT/'src/prefix/native_attention.py'))
    all_rows=[]
    for row in manifest['examples']:
        index=row['input_index']
        if index not in indices:continue
        old=torch.load(ROOT/f'checkpoints/section4_native_activations_{index:03d}.pt',weights_only=True,mmap=True)
        base=json.loads((ROOT/'data/section4_native_cohort.json').read_text())
        original=next(r for r in base['examples'] if r['index']==index)
        if base['fingerprint']!=manifest['base_cohort'] or old['fingerprint']!=original['fingerprint']:
            raise ValueError('wrong frozen base states')
        new=torch.load(ROOT/f'checkpoints/section4_prompt_diversity_activations_{index:03d}.pt',weights_only=True,mmap=True)
        if new['fingerprint']!=row['fingerprint']:raise ValueError('wrong prompt states')
        n=row['input_tokens'];h=torch.cat([old['hidden'][:n],new['prompt_hidden'],old['hidden'][n+128:]])
        for head in config['heads']:
            w={k:v.to(device=device,dtype=torch.float64) if torch.is_tensor(v) else v
               for k,v in weights[f'{config["layer"]}_{head}'].items()}
            study=NativeHead(h.to(device=device,dtype=torch.float64),w,n)
            for c in selected_conditions():
                identity=dict(manifest=manifest['fingerprint'],input_index=index,prompt_source_index=row['prompt_source_index'],
                              head=head,condition=c,implementation=implementation)
                path=ROOT/f'results/section4_prompt_diversity_{index:03d}_h{head:02d}_{c["id"]}.json'
                result=completed_result(path,identity)
                if result is None:
                    directions,_=study.theoretical_directions(c['m'],c['b'],c['g'])
                    result=dict(identity=identity,complete=True,**rank_diagnostics(directions,config['rank_rtol']))
                    write_json_atomic(path,result)
                    print(f'analyze {index} h{head} {c["id"]}: dim={result["dimension"]}',flush=True)
                all_rows.append(result)
    full=len(indices)==len(manifest['examples'])
    expected={(i,h,c['id']) for i in indices for h in config['heads'] for c in selected_conditions()}
    actual={(r['identity']['input_index'],r['identity']['head'],r['identity']['condition']['id']) for r in all_rows}
    if expected!=actual or len(all_rows)!=len(expected):raise ValueError('incomplete dimension coverage')
    output=dict(complete=True,formal=full,manifest=manifest['fingerprint'],examples=len(indices),
                conditions=len(all_rows),heads=config['heads'],layer=config['layer'],rank_rtol=config['rank_rtol'],
                token_statistics=summarize(all_rows),distinct_prompt_prefixes={str(m):len({tuple(r['prompt_ids'][:m])
                for r in manifest['examples'] if r['input_index'] in indices}) for m in LENGTHS},
                padded_prompts=sum(bool(r['added_text']) for r in manifest['examples'] if r['input_index'] in indices),
                design=manifest['design'],implementation=implementation)
    name='section4_prompt_diversity_summary.json' if full else 'section4_prompt_diversity_smoke.json'
    write_json_atomic(ROOT/'results'/name,output)
    print(f'complete: {len(all_rows)} conditions; formal={full}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['prepare','extract','analyze'])
    parser.add_argument('--indices',type=int,nargs='+')
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args();torch.set_num_threads(2)
    with tee_stdout(ROOT/f'logs/section4_prompt_diversity_{args.phase}.log'),contextlib.redirect_stderr(sys.stdout):
        if args.phase=='prepare':prepare()
        else:
            manifest=json.loads(MANIFEST.read_text())
            indices=args.indices or [r['input_index'] for r in manifest['examples']]
            if len(set(indices))!=len(indices) or not set(indices)<={r['input_index'] for r in manifest['examples']}:
                raise ValueError('invalid requested examples')
            if args.phase=='extract':extract(manifest,indices,args.device)
            else:analyze(manifest,indices,args.device)


if __name__=='__main__':main()
