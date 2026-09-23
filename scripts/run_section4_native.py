"""Three resumable native-attention studies on a fixed 100-example cohort.

prepare exports existing data, extract extends common traces, and analyze fits
native r and measures token sweeps, query drift, and theoretical rank diagnostics.
All outputs have independent names; legacy experiment artifacts are read-only.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

from prefix.attention_bounds import diameter, evaluate
from prefix.native_attention import NativeHead, LENGTHS, conditions, fit_shift, query_sets, subspace, perturbations
from prefix.runner import tee_stdout, write_json_atomic

ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / 'data/section4_native_cohort.json'
PREAMBLE = (
    'Additional context for interpreting the request: read the complete request below before responding. '
    'Keep track of the entities, actions, constraints, and qualifications explicitly stated in it. '
    'The request itself follows this introductory context. Do not treat this introductory context as '
    'a separate question. Base your response on the request and explain your answer in clear language. '
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def analysis_fingerprint(source):
    names = {'analyze', 'drift_study', 'controlled_dimensions'}
    functions = [ast.dump(node, include_attributes=False) for node in ast.parse(source).body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(functions) != len(names):
        raise ValueError('incomplete analysis implementation')
    return digest(functions)


def resume_batch(path, identity, rows, completed):
    batch = identity['indices']
    state = json.loads(path.read_text()) if path.exists() else dict(identity=identity,
        tokens=[rows[i]['seed_generated_ids'].copy() for i in batch])
    pending = [i for i in batch if i not in completed]
    tokens = [state['tokens'][batch.index(i)] for i in pending]
    if state['identity'] != identity or (tokens and len({len(x) for x in tokens}) != 1):
        raise ValueError('decoding checkpoint mismatch')
    return state, pending, tokens


def choose_examples(lengths, count):
    if count > len(lengths):
        raise ValueError('requested count exceeds available examples')
    long = sorted(i for i, length in lengths.items() if length >= 128)
    short = sorted((i for i in lengths if i not in long), key=lambda i: (-lengths[i], i))
    return long[:count] + short[:max(0, count - len(long))]


def expanded_input(tokenizer, request, minimum=128):
    added = ''
    for _ in range(16):
        ids = tokenizer.apply_chat_template([{'role':'user', 'content':added + request}],
                    tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
        if len(ids) >= minimum:
            return ids, added
        added += PREAMBLE + '\n\n'
    raise ValueError('context expansion failed to reach requested input length')


def save_tensor(path, payload):
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_tensor(path):
    return torch.load(path, weights_only=True, mmap=True)


def validate_result(row, identity, queries):
    if (row.get('identity') != identity or not row.get('complete') or row.get('queries') != queries
            or len(row.get('errors', [])) != len(queries)):
        raise ValueError('result identity or query coverage mismatch')
    if not all(np.isfinite(value) and value >= 0 for value in row['errors']):
        raise ValueError('invalid measured error')


def prepare(config):
    from transformers import AutoTokenizer
    torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
    records = json.loads((ROOT / 'data/section4_harmbench.json').read_text())
    lengths = {i: load_tensor(ROOT / f'checkpoints/section4_long_activations_{i:03d}.pt')['base_length']
               for i in range(len(records))}
    indices = choose_examples(lengths, config['sample_count'])
    tokenizer = AutoTokenizer.from_pretrained(config['model'], revision=config['revision'], local_files_only=True)
    old_meta = json.loads((ROOT / 'results/section4_long_manifest.json').read_text())
    cohort = dict(version=1, config=config, examples=[], source_manifest=old_meta,
                  selection='all original inputs >=128 tokens, then longest short inputs with explicit context expansion')
    weights = load_tensor(ROOT / 'checkpoints/section4_weights.pt')
    selected_weights = {f'{config["layer"]}_{head}': weights[f'{config["layer"]}_{head}'] for head in config['heads']}
    weight_path = ROOT / 'checkpoints/section4_native_weights.pt'
    if not weight_path.exists():
        save_tensor(weight_path, selected_weights)
    cohort['weights_sha256'] = file_digest(weight_path)
    for index in indices:
        original = load_tensor(ROOT / f'checkpoints/section4_activations_{index:03d}.pt')
        long = load_tensor(ROOT / f'checkpoints/section4_long_activations_{index:03d}.pt')
        record = records[index]
        n = lengths[index]
        added = ''
        base_ids = original['tokens'][:n]
        if n < config['minimum_input_tokens']:
            context = record.get('ContextString', '').strip()
            request = f'{context}\n\n{record["Behavior"]}' if context else record['Behavior']
            base_ids, added = expanded_input(tokenizer, request, config['minimum_input_tokens'])
        prompt_ids = old_meta['prompt_ids']
        row = dict(index=index, behavior_id=record['BehaviorID'], original_input_tokens=n,
                   input_tokens=len(base_ids), expanded=bool(added), added_context=added,
                   functional_category=record.get('FunctionalCategory'), base_ids=base_ids,
                   prompt_ids=prompt_ids, seed_generated_ids=[] if added else long['generated_ids'])
        row['fingerprint'] = digest(row)
        path = ROOT / f'checkpoints/section4_native_seed_{index:03d}.pt'
        if path.exists():
            if load_tensor(path)['metadata'] != row:
                raise ValueError(f'seed mismatch: {path}')
        else:
            payload = {'metadata':row, 'prompt_hidden':None if added else long['layers'][config['layer']][n:n+128].clone()}
            save_tensor(path, payload)
        cohort['examples'].append(row)
        print(f'prepared {index}: input {n} -> {len(base_ids)}, expanded={bool(added)}', flush=True)
        write_json_atomic(ROOT / 'checkpoints/section4_native_prepare.json', dict(config=config, examples=cohort['examples']))
    cohort['fingerprint'] = digest(cohort)
    if COHORT.exists() and json.loads(COHORT.read_text()) != cohort:
        raise ValueError('cohort changed; refuse to reuse existing native artifacts')
    write_json_atomic(COHORT, cohort)
    print(f'cohort saved: {len(indices)}, expanded={sum(r["expanded"] for r in cohort["examples"])}', flush=True)


def extract(config, cohort, indices, batch_size, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rows = {row['index']:row for row in cohort['examples']}
    done = []
    for i in indices:
        path = ROOT / f'checkpoints/section4_native_activations_{i:03d}.pt'
        if path.exists():
            if load_tensor(path)['fingerprint'] != rows[i]['fingerprint']:
                raise ValueError('activation identity mismatch')
            done.append(i)
    if len(done) == len(indices):
        print(f'extract: all {len(indices)} examples cached', flush=True)
        return
    tokenizer = AutoTokenizer.from_pretrained(config['model'], revision=config['revision'], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(config['model'], revision=config['revision'], local_files_only=True,
                dtype=torch.bfloat16, attn_implementation='sdpa').to(device).eval()
    stop_ids = model.generation_config.eos_token_id
    stop_ids = [stop_ids] if isinstance(stop_ids, int) else list(stop_ids or [])

    def padded(sequences):
        width = max(map(len, sequences))
        ids = torch.full((len(sequences), width), tokenizer.pad_token_id, device=device, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for j, seq in enumerate(sequences):
            ids[j, -len(seq):] = torch.tensor(seq, device=device)
            mask[j, -len(seq):] = 1
        return ids, mask, (mask.cumsum(-1) - 1).clamp_min(0)

    def capture(sequences):
        captured = []
        def hook(_module, args, kwargs):
            captured.append(kwargs.get('hidden_states', args[0] if args else None).detach().cpu())
        handle = model.model.layers[config['layer']].self_attn.register_forward_pre_hook(hook, with_kwargs=True)
        try:
            ids, mask, positions = padded(sequences)
            model(ids, attention_mask=mask, position_ids=positions, use_cache=False, logits_to_keep=1)
        finally:
            handle.remove()
        return [captured[0][j, -len(seq):].clone() for j, seq in enumerate(sequences)]

    # Group by existing trace length; each atomic JSON journal owns a stable batch.
    with torch.inference_mode():
        for initial_count in sorted({len(rows[i]['seed_generated_ids']) for i in indices}):
            group = [i for i in indices if len(rows[i]['seed_generated_ids']) == initial_count]
            for start in range(0, len(group), batch_size):
                batch = group[start:start+batch_size]
                if all(i in done for i in batch):
                    continue
                t = time.monotonic()
                identity = dict(cohort=cohort['fingerprint'], indices=batch, generated=config['generated_tokens'])
                journal = ROOT / f'checkpoints/section4_native_decode_{digest(identity)[:16]}.json'
                state, pending, tokens_by_example = resume_batch(journal,identity,rows,set(done))
                prefixes = [rows[i]['base_ids'] + rows[i]['prompt_ids'][:config['generation_prompt_tokens']] for i in pending]
                ids, mask, positions = padded([p+t for p,t in zip(prefixes,tokens_by_example)])
                cache = None
                while len(tokens_by_example[0]) < config['generated_tokens']:
                    out = model(ids, attention_mask=mask, position_ids=positions, past_key_values=cache,
                                use_cache=True, logits_to_keep=1)
                    cache = out.past_key_values
                    new = out.logits[:,-1].argmax(-1)
                    for tokens, token in zip(tokens_by_example, new.tolist()):
                        tokens.append(token)
                    write_json_atomic(journal, state)
                    ids = new[:,None]
                    mask = torch.cat([mask, torch.ones((len(pending),1), device=device, dtype=mask.dtype)], -1)
                    positions = mask.sum(-1, keepdim=True) - 1
                del cache, ids, mask, positions
                captures = capture([p+t for p,t in zip(prefixes,tokens_by_example)])
                for j,index in enumerate(pending):
                    meta = rows[index]
                    n = meta['input_tokens']
                    seed = load_tensor(ROOT / f'checkpoints/section4_native_seed_{index:03d}.pt')
                    prompt_h = seed['prompt_hidden']
                    if prompt_h is None:
                        prompt_h = capture([meta['base_ids'] + meta['prompt_ids']])[0][n:]
                    generation_h = captures[j]
                    hidden = torch.cat([generation_h[:n], prompt_h,
                                        generation_h[n+config['generation_prompt_tokens']:]], 0)
                    tokens = tokens_by_example[j]
                    first_eos = next((k+1 for k,token in enumerate(tokens) if token in stop_ids), None)
                    payload = dict(fingerprint=meta['fingerprint'], index=index, hidden=hidden,
                                   generated_ids=tokens, first_eos=first_eos, stop_ids=stop_ids,
                                   generation_policy='fixed-length greedy; EOS recorded, not used to truncate',
                                   library='frozen prompt states plus one common 8-token-prompt continuation')
                    save_tensor(ROOT / f'checkpoints/section4_native_activations_{index:03d}.pt', payload)
                    write_json_atomic(ROOT / f'checkpoints/section4_native_extract_{index:03d}.json',
                                      dict(index=index, fingerprint=meta['fingerprint'], complete=True,
                                           generated_tokens=len(tokens), first_eos=first_eos))
                    done.append(index)
                print(f'extract {batch}: complete in {time.monotonic()-t:.1f}s ({len(done)}/{len(indices)})', flush=True)


def identity_for(cohort, index, head, condition, config):
    return dict(version=1, cohort=cohort['fingerprint'], index=index, head=head, condition=condition,
                layer=config['layer'], fit_steps=config['max_fit_steps'], fit_tolerance=config['fit_tolerance'],
                native_implementation_sha256=file_digest(ROOT / 'src/prefix/native_attention.py'),
                analysis_sha256=analysis_fingerprint(Path(__file__).read_text()))


def drift_study(study, c, r, identity, path, config):
    state = json.loads(path.read_text()) if path.exists() else dict(identity=identity, units={})
    if state['identity'] != identity:
        raise ValueError('query-drift manifest mismatch')
    m,b,g = c['m'],c['b'],c['g']
    directions, z = study.theoretical_directions(m,b,g)
    basis, rank = subspace(directions, config['rank_rtol'])
    selected = list(range(study.n-b,study.n)) + list(range(study.n+study.prompt_slots,study.n+study.prompt_slots+g))
    prompt = list(range(study.n,study.n+m))
    shared = [i for i in list(range(study.n)) + list(range(study.n+study.prompt_slots,len(study.h))) if i not in selected]
    k,v = study.k.cpu().numpy(),study.v.cpu().numpy()
    q0 = study.q[-1]
    dk = (study.w['wk']@r).cpu().numpy()
    dv = (study.w['wv']@r).cpu().numpy()
    base_d = diameter(v[shared+selected+prompt])
    for scale in config['drift_scales']:
        changes = perturbations(basis, rank, float(q0.norm()) * scale, config['seed'] + identity['index'] + identity['head'])
        for name,change in changes.items():
            key = f'{name}_{scale:g}'
            if key in state['units']:
                continue
            raw = (q0 + change)[None]
            qs = [len(study.h)-1]
            native = float((study.outputs(m,b,g,qs,r,'steer',raw) - study.outputs(m,b,g,qs,None,'prompt',raw)).norm())
            linear = evaluate(k,v,q0.cpu().numpy(),raw[0].cpu().numpy(),selected,prompt,shared,dk,dv,
                              original_diameter=base_d,anchor=study.n-1)
            state['units'][key] = dict(direction=name, scale=scale, query_change_norm=float(change.norm()),
                rho=float((basis[:rank]@change).norm()), native_error=native,
                linear_error=linear['error'], linear_residual_aware_bound=linear['bound_certified'],
                linear_reference_mismatch=linear['beta'], diagnostic_dimension=study.k.shape[1]-rank)
            write_json_atomic(path, state)
    state['complete'] = True
    write_json_atomic(path, state)
    return state


def controlled_dimensions(study, config):
    directions,_ = study.theoretical_directions(1,1,0)
    basis,rank = subspace(directions,config['rank_rtol'])
    rows = []
    for count in LENGTHS:
        redundant = torch.cat([directions, directions[:1].repeat(count-1,1)])
        independent = torch.cat([directions, basis[rank:rank+count-1]])
        for name,matrix in [('redundant',redundant),('independent',independent)]:
            rows.append(dict(count=count,control=name,dimension=study.k.shape[1]-subspace(matrix,config['rank_rtol'])[1]))
    return rows


def analyze(config, cohort, indices, device):
    weights = load_tensor(ROOT / 'checkpoints/section4_native_weights.pt')
    metadata = {r['index']:r for r in cohort['examples']}
    for index in indices:
        payload = load_tensor(ROOT / f'checkpoints/section4_native_activations_{index:03d}.pt')
        meta = metadata[index]
        if payload['fingerprint'] != meta['fingerprint']:
            raise ValueError('wrong activation identity')
        for head in config['heads']:
            w = {k:v.to(device=device,dtype=torch.float64) if torch.is_tensor(v) else v
                 for k,v in weights[f'{config["layer"]}_{head}'].items()}
            study = NativeHead(payload['hidden'].to(device=device,dtype=torch.float64),w,meta['input_tokens'],
                               config['prompt_slots'],config['generated_tokens'])
            controls = controlled_dimensions(study,config)
            for c in conditions():
                start = time.monotonic()
                m,b,g = c['m'],c['b'],c['g']
                identity = identity_for(cohort,index,head,c,config)
                name = f'{index:03d}_h{head:02d}_{c["id"]}'
                queries = query_sets(study.n,study.prompt_slots,study.generated,b,g)
                path = ROOT / f'results/section4_native_{name}.json'
                if path.exists():
                    validate_result(json.loads(path.read_text()),identity,queries['all'])
                    print(f'analyze {name}: cached', flush=True)
                    continue
                fit_path = ROOT / f'checkpoints/section4_native_fit_{name}.json'
                fitted = fit_shift(study,m,b,g,fit_path,identity,config['max_fit_steps'],config['fit_tolerance'])
                r = torch.tensor(fitted['r'],device=device,dtype=torch.float64)
                with torch.no_grad():
                    errors = []
                    for start_q in range(0,len(queries['all']),128):
                        q = queries['all'][start_q:start_q+128]
                        delta = study.outputs(m,b,g,q,r,'steer') - study.outputs(m,b,g,q,None,'prompt')
                        errors.extend(delta.norm(dim=-1).cpu().tolist())
                    directions,z = study.theoretical_directions(m,b,g)
                    _,rank = subspace(directions,config['rank_rtol'])
                    singular = torch.linalg.svdvals(directions)
                    result = dict(identity=identity, complete=True, queries=queries['all'], errors=errors,
                        query_sets=queries, fit={k:v for k,v in fitted.items() if k not in ('r','shift')},
                        diagnostic_dimension=study.k.shape[1]-rank, singular_values=singular.cpu().tolist(),
                        controlled_dimensions=controls, first_eos=payload['first_eos'], expanded=meta['expanded'])
                    if (m,b,g) in [(1,1,0),(4,1,0),(4,4,0),(4,1,3)]:
                        drift_study(study,c,r,identity,ROOT/f'results/section4_native_drift_{name}.json',config)
                    write_json_atomic(path,result)
                print(f'analyze {name}: {len(errors)} queries, fit={fitted["final_error"]:.3g}, '
                      f'steps={fitted["steps"]}, dim={result["diagnostic_dimension"]}, '
                      f'{time.monotonic()-start:.2f}s',flush=True)


def run_workers(args, indices):
    """Overlap durable checkpoint I/O on disjoint examples within one GPU job."""
    count = min(args.workers, len(indices))
    processes = []
    try:
        for shard in range(count):
            command = [sys.executable, '-u', str(Path(__file__).resolve()), 'analyze',
                       '--config', str(args.config), '--device', args.device,
                       '--shards', str(count), '--shard', str(shard), '--indices', *map(str,indices)]
            processes.append(subprocess.Popen(command))
        for process in processes:
            code = process.wait()
            if code:
                raise subprocess.CalledProcessError(code, 'native analysis worker')
    finally:
        pending = [process for process in processes if process.poll() is None]
        for process in pending:
            process.terminate()
        for process in pending:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['prepare','extract','analyze'])
    parser.add_argument('--config',type=Path,default=ROOT/'configs/section4_native.yaml')
    parser.add_argument('--indices',type=int,nargs='+')
    parser.add_argument('--batch-size',type=int)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--shards',type=int,default=1)
    parser.add_argument('--workers',type=int,default=4)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    torch.set_num_threads(1 if args.shards > 1 else 2)
    torch.backends.cuda.matmul.allow_tf32 = False
    log = ROOT / f'logs/section4_native_{args.phase}_{args.shard}.log'
    with tee_stdout(log), contextlib.redirect_stderr(sys.stdout):
        if args.phase == 'prepare':
            prepare(config)
        else:
            cohort = json.loads(COHORT.read_text())
            if cohort['config'] != config:
                raise ValueError('config differs from prepared cohort')
            indices = args.indices or [r['index'] for r in cohort['examples']]
            if not set(indices) <= {r['index'] for r in cohort['examples']}:
                raise ValueError('requested indices outside prepared cohort')
            if not 0 <= args.shard < args.shards:
                raise ValueError('invalid shard')
            indices = indices[args.shard::args.shards]
            if args.phase == 'extract':
                extract(config,cohort,indices,args.batch_size or config['batch_size'],args.device)
            elif args.shards == 1 and args.workers > 1 and len(indices) > 1:
                print(f'launching {min(args.workers,len(indices))} disjoint analysis workers on {args.device}',flush=True)
                run_workers(args,indices)
            else:
                analyze(config,cohort,indices,args.device)


if __name__ == '__main__':
    main()
