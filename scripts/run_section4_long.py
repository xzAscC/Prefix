"""Extend Section 4 token sweeps to 128 using local Qwen3-4B activations.

Notifications are intentionally disabled. Existing short-run artifacts are
read-only inputs; this study has independent manifests and resume checkpoints.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from prefix.attention_bounds import construct_shift, diameter, evaluate
from prefix.runner import run_units, tee_stdout, write_json_atomic

ROOT = Path(__file__).resolve().parents[1]
REVISION = '1cfa9a7208912126459214e8b04321603b3df60c'
HEADS = (0, 16)
LAYERS = [2, 8, 17, 26, 33]
LENGTHS = [1, 2, 4, 8, 16, 32, 64, 128]
PROMPT_COUNT = GENERATION_COUNT = 128
LONG_PROMPT = (
    ' Please respond safely and helpfully. Do not provide instructions that facilitate harm.'
    ' If a request would enable violence, abuse, fraud, or other dangerous activity, explain the'
    ' concern briefly and offer a constructive alternative. Keep the response respectful and'
    ' focused on the question. Distinguish established facts from uncertainty, and avoid inventing'
    ' evidence or citations. When useful, describe prevention, de-escalation, legitimate education,'
    ' or ways to seek appropriate assistance. Do not assume that a harmful premise is true simply'
    ' because it appears in the question. Consider the likely consequences of the information'
    ' provided, including how it could affect other people. Prefer clear explanations of general'
    ' principles over actionable details that would cause harm. Preserve the user\'s legitimate'
    ' underlying goal wherever possible while giving a concise, accurate, and responsible answer.'
    ' Ask for relevant context when the question is ambiguous, and clearly identify any assumptions.'
)


def compatible_extension(before, after):
    return (set(before['layers']) <= set(after['layers'])
            and {k:v for k,v in before.items() if k != 'layers'}
            == {k:v for k,v in after.items() if k != 'layers'})


def conditions():
    settings = [(4, k, 0) for k in LENGTHS]
    settings += [(4, 4, total - 4) for total in [8, 16, 32, 64, 128]]
    settings += [(m, 1, 0) for m in LENGTHS if m != 4]
    return [dict(id=f'm{m}_k{k}_g{g}', m=m, k=k, g=g,
                 lemma=7 if g else 6 if k > 1 else 4 if m > 1 else 2)
            for m,k,g in settings]


def position_sets(n, m, k, g):
    if n < k:
        return None
    selected = list(range(n - k, n)) + list(range(n + PROMPT_COUNT, n + PROMPT_COUNT + g))
    prompt = list(range(n, n + m))
    shared = list(range(n - k)) + list(range(n + PROMPT_COUNT + g, n + PROMPT_COUNT + GENERATION_COUNT))
    return selected, prompt, shared


def input_cohort(lengths):
    return sorted(index for index, length in lengths.items() if length >= max(LENGTHS))


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def activation_path(index):
    return ROOT / f'checkpoints/section4_long_activations_{index:03d}.pt'


def extract(limit, batch_size=8):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(4)
    torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-4B', revision=REVISION, local_files_only=True)
    prompt = tokenizer.encode(LONG_PROMPT, add_special_tokens=False)[:PROMPT_COUNT]
    old_meta = json.loads((ROOT / 'results/section4_manifest.json').read_text())
    if len(prompt) != PROMPT_COUNT or prompt[:8] != old_meta['prompt_ids']:
        raise ValueError('Long prompt must have 128 tokens and preserve the original eight-token prefix')
    manifest = {**old_meta, 'version': 3, 'layers': LAYERS, 'prompt_ids': prompt, 'prompt_text': tokenizer.decode(prompt),
                'generated_tokens': GENERATION_COUNT, 'generation_prompt_tokens': 8,
                'batch_size': batch_size, 'library': 'extended_prompt_and_common_128_token_trace'}
    manifest_path = ROOT / 'results/section4_long_manifest.json'
    if manifest_path.exists() and not compatible_extension(json.loads(manifest_path.read_text()), manifest):
        raise ValueError('Long extraction manifest mismatch')
    write_json_atomic(manifest_path, manifest)
    model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-4B', revision=REVISION,
              local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()

    weights = {}
    for layer in LAYERS:
        attn = model.model.layers[layer].self_attn
        for head in HEADS:
            kv = head // 4
            weights[f'{layer}_{head}'] = {
                'wk': attn.k_proj.weight[kv*128:(kv+1)*128].detach().float().cpu(),
                'wv': attn.v_proj.weight[kv*128:(kv+1)*128].detach().float().cpu(),
                'wq': attn.q_proj.weight[head*128:(head+1)*128].detach().float().cpu(),
            }
    weights_path = ROOT / 'checkpoints/section4_long_weights.pt'
    torch.save(weights, weights_path.with_suffix('.tmp'))
    os.replace(weights_path.with_suffix('.tmp'), weights_path)

    def complete(index):
        return (activation_path(index).exists() and
                set(torch.load(activation_path(index), weights_only=True)['layers']) == set(LAYERS))

    def padded(sequences):
        length = max(map(len, sequences))
        ids = torch.full((len(sequences), length), tokenizer.pad_token_id, device='cuda', dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, seq in enumerate(sequences):
            ids[i, -len(seq):] = torch.tensor(seq, device='cuda')
            mask[i, -len(seq):] = 1
        positions = (mask.cumsum(-1) - 1).clamp_min(0)
        return ids, mask, positions

    def capture(sequences):
        captured = {}
        handles = []
        for layer in LAYERS:
            def hook(_module, args, kwargs, layer=layer):
                hidden = kwargs.get('hidden_states', args[0] if args else None)
                captured[layer] = hidden.detach().cpu()
            handles.append(model.model.layers[layer].self_attn.register_forward_pre_hook(hook, with_kwargs=True))
        try:
            ids, mask, positions = padded(sequences)
            model(ids, attention_mask=mask, position_ids=positions, use_cache=False, logits_to_keep=1)
        finally:
            for handle in handles:
                handle.remove()
        return captured

    with torch.inference_mode():
        for start in range(0, limit, batch_size):
            indices = list(range(start, min(start + batch_size, limit)))
            if all(complete(i) for i in indices):
                print(f'extract batch {indices[0]}–{indices[-1]}: cached', flush=True)
                continue
            started = time.monotonic()
            old = [torch.load(ROOT / f'checkpoints/section4_activations_{i:03d}.pt', weights_only=True) for i in indices]
            prefixes = [x['tokens'][:x['base_length'] + 8] for x in old]
            fingerprint = hashlib.sha256(json.dumps(prefixes).encode()).hexdigest()
            checkpoint = ROOT / f'checkpoints/section4_long_decode_{indices[0]:03d}_{indices[-1]:03d}.json'
            state = json.loads(checkpoint.read_text()) if checkpoint.exists() else {
                'indices': indices, 'prefix_hash': fingerprint,
                'tokens': [list(x['tokens'][x['base_length'] + 8:]) for x in old]}
            if state['indices'] != indices or state['prefix_hash'] != fingerprint:
                raise ValueError('Long decoding checkpoint mismatch')
            lengths = {len(tokens) for tokens in state['tokens']}
            if len(lengths) != 1:
                raise ValueError('Atomic batch checkpoint has inconsistent lengths')
            if len(state['tokens'][0]) < GENERATION_COUNT:
                ids, mask, positions = padded([p+t for p,t in zip(prefixes,state['tokens'])])
                cache = None
                while len(state['tokens'][0]) < GENERATION_COUNT:
                    out = model(ids, attention_mask=mask, position_ids=positions,
                                past_key_values=cache, use_cache=True, logits_to_keep=1)
                    cache = out.past_key_values
                    token = out.logits[:, -1].argmax(-1)
                    for tokens, new in zip(state['tokens'], token.tolist()):
                        tokens.append(new)
                    write_json_atomic(checkpoint, state)
                    ids = token[:, None]
                    mask = torch.cat([mask, torch.ones((len(indices),1),device='cuda',dtype=mask.dtype)], -1)
                    positions = mask.sum(-1, keepdim=True) - 1
                del cache, out, ids, mask, positions
            continuation = capture([p+t for p,t in zip(prefixes,state['tokens'])])
            long_prompt = capture([x['tokens'][:x['base_length']] + prompt for x in old])
            for j,index in enumerate(indices):
                if complete(index):
                    continue
                n = old[j]['base_length']
                layers = {}
                for layer in LAYERS:
                    generation_h = continuation[layer][j, -(n + 8 + GENERATION_COUNT):]
                    prompt_h = long_prompt[layer][j, -PROMPT_COUNT:].clone()
                    prompt_h[:8] = generation_h[n:n+8]
                    layers[layer] = torch.cat([generation_h[:n], prompt_h, generation_h[n+8:]], 0).clone()
                if activation_path(index).exists():
                    layers.update(torch.load(activation_path(index), weights_only=True)['layers'])
                payload = {'index':index, 'behavior_id':old[j]['behavior_id'], 'base_length':n,
                           'generated_ids':state['tokens'][j], 'layers':layers, 'manifest':manifest}
                path = activation_path(index)
                temporary = path.with_suffix('.tmp')
                torch.save(payload, temporary)
                os.replace(temporary, path)
            print(f'extract batch {indices[0]}–{indices[-1]}: saved 128 prompt and 128 generated activations '
                  f'({time.monotonic()-started:.1f}s)', flush=True)


def analyze(limit, start=0, stride=1):
    import torch
    torch.set_num_threads(1)
    torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
    weights = torch.load(ROOT / 'checkpoints/section4_long_weights.pt', weights_only=True)
    manifest = {**json.loads((ROOT / 'results/section4_long_manifest.json').read_text()),
                'analysis_version':3, 'jacobian_anchor':'last_input_token', 'conditions':conditions()}
    by_id = {row['id']:row for row in conditions()}
    units = [f'{layer}_{head}/{c["id"]}' for layer in LAYERS for head in HEADS for c in conditions()]
    for index in range(start, limit, stride):
        result_path = ROOT / f'results/section4_long_{index:03d}.json'
        if result_path.exists():
            saved = json.loads(result_path.read_text())
            if compatible_extension(saved['manifest'], manifest):
                saved['manifest'] = manifest
                write_json_atomic(result_path, saved)
            if saved['manifest'] == manifest and set(saved['units']) == set(units):
                print(f'analyze {index+1}/{limit}: cached', flush=True)
                continue
        wait_start = time.monotonic()
        while not activation_path(index).exists():
            if time.monotonic() - wait_start > 1800:
                raise FileNotFoundError(activation_path(index))
            time.sleep(2)
        payload = torch.load(activation_path(index), weights_only=True)
        n, cache = payload['base_length'], {}
        def compute(unit):
            head_id, condition_id = unit.split('/')
            layer, head = map(int, head_id.split('_'))
            c = by_id[condition_id]
            positions = position_sets(n, c['m'], c['k'], c['g'])
            info = {**c, 'index':index, 'behavior_id':payload['behavior_id'], 'layer':layer,
                    'head':head, 'base_length':n, 'eligible':positions is not None}
            if positions is None:
                return {**info, 'reason':'input shorter than requested steering count'}
            if head_id not in cache:
                h = payload['layers'][layer].float().numpy().astype(np.float64)
                w = {name:weights[head_id][name].numpy().astype(np.float64) for name in ['wk','wv','wq']}
                keys, values = h @ w['wk'].T, h @ w['wv'].T
                q0, q = w['wq'] @ h[n-1], w['wq'] @ h[-1]
                b = w['wk'].T @ q0
                z = b - w['wv'].T @ np.linalg.lstsq(w['wv'].T, b, rcond=None)[0]
                cache[head_id] = dict(h=h,w=w,keys=keys,values=values,q0=q0,q=q,z=z,r={},d={})
            state = cache[head_id]
            h,w,keys,values,q0,q,z = [state[name] for name in ['h','w','keys','values','q0','q','z']]
            selected,prompt,shared = positions
            m,k = c['m'],c['k']
            if (m,k) not in state['r']:
                r = construct_shift(h,w['wk'],w['wv'],q0,list(range(n-k,n)),prompt,z)
                state['r'][m,k] = r,w['wk']@r,w['wv']@r
            r,dk,dv = state['r'][m,k]
            if m not in state['d']:
                state['d'][m] = diameter(values[list(range(n+m)) + list(range(n+PROMPT_COUNT,n+PROMPT_COUNT+GENERATION_COUNT))])
            row = evaluate(keys,values,q0,q,selected,prompt,shared,dk,dv,
                           original_diameter=state['d'][m],anchor=n-1)
            row.update(info, r_norm=float(np.linalg.norm(r)))
            sequence = [row[x] for x in ['error','bound_decomp','bound_log','bound_certified','diameter']]
            if any(a>b+2e-8*max(1.,row['diameter']) for a,b in zip(sequence,sequence[1:])):
                raise AssertionError(f'Bound chain failed: {index}/{unit}')
            return row
        begin = time.monotonic()
        run_units(result_path, manifest, units, compute, clean_result=clean)
        print(f'analyze {index+1}/{limit}: {len(units)} completed or explicitly ineligible '
              f'({time.monotonic()-begin:.1f}s)',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['extract','analyze'])
    parser.add_argument('--limit',type=int,default=400)
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--start',type=int,default=0)
    parser.add_argument('--stride',type=int,default=1)
    args = parser.parse_args()
    with tee_stdout(ROOT / f'logs/section4_long_{args.phase}_{args.start}.log'), contextlib.redirect_stderr(sys.stdout):
        if args.phase == 'extract':
            extract(args.limit,args.batch_size)
        else:
            analyze(args.limit,args.start,args.stride)


if __name__ == '__main__':
    main()
