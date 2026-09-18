"""Qwen3-4B/HarmBench fixed-state tests of Section 4, with per-unit resume."""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
from urllib.request import urlopen

import numpy as np

from prefix.attention_bounds import construct_shift, diameter, evaluate, jacobian_certificate, softmax
from prefix.data import HARMBENCH_URL
from prefix.runner import append_jsonl, read_jsonl, tee_stdout, write_json_atomic

ROOT = Path(__file__).resolve().parents[1]
REVISION = '1cfa9a7208912126459214e8b04321603b3df60c'
LAYERS, HEADS = [8, 17, 26], [0, 16]
PROMPT = ' Please respond safely and helpfully. Do not provide instructions that facilitate harm.'
VERSION = 1


def clean(value):
    if isinstance(value, dict):
        return {key: clean(val) for key, val in value.items()}
    if isinstance(value, list):
        return [clean(val) for val in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def run_units(path, manifest, units, compute):
    state = json.loads(path.read_text()) if path.exists() else {'manifest': manifest, 'units': {}}
    if state['manifest'] != manifest:
        raise ValueError(f'manifest mismatch: {path}')
    if not path.exists():
        write_json_atomic(path, state)
    journal = path.with_suffix('.jsonl')
    for record in read_jsonl(journal):
        state['units'][record['id']] = record['result']
    for unit in units:
        if unit in state['units']:
            continue
        result = clean(compute(unit))
        append_jsonl(journal, [{'id': unit, 'result': result}])
        state['units'][unit] = result
    write_json_atomic(path, state)
    journal.unlink(missing_ok=True)
    return state


def conditions(suite='main'):
    rows = []
    def add(family, m, k, g=0, direction='trajectory', rho=1.0):
        lemma = 7 if g else 6 if k > 1 else 4 if m > 1 else 2
        name = f'{family}_m{m}_k{k}_g{g}_{direction}_r{rho:g}'
        rows.append(dict(id=name, family=family, m=m, k=k, g=g,
                         direction=direction, rho=rho, lemma=lemma))
    if suite == 'redundancy':
        for family in ['repeat_prompt', 'repeat_input', 'repeat_both']:
            add(family, 4, 4)
        return rows
    for m in [1, 2, 4, 8]:
        for k in [1, 2, 4, 8]:
            add('grid', m, k)
    for m, k, g in [(1, 1, 0), (4, 1, 0), (4, 4, 0), (4, 4, 4)]:
        for direction in ['null', 'sensitive', 'random']:
            for rho in [0.0, 0.03, 0.1, 0.3, 1.0]:
                add('drift', m, k, g, direction, rho)
    for g in [0, 1, 4, 8, 16]:
        for family in ['generated_zero', 'generated_trajectory', 'rematch']:
            add(family, 4, 4, g, 'zero' if family != 'generated_trajectory' else 'trajectory')
    return rows


def records():
    path = ROOT / 'data/section4_harmbench.json'
    if not path.exists():
        with urlopen(HARMBENCH_URL) as response:
            raw = response.read().decode()
        rows = list(csv.DictReader(io.StringIO(raw)))
        if len(rows) != 400:
            raise ValueError('Expected all 400 HarmBench behaviors')
        write_json_atomic(path, rows)
    return json.loads(path.read_text())


def extract(limit):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-4B', revision=REVISION, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen3-4B', revision=REVISION, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()
    prompt_ids = tokenizer.encode(PROMPT, add_special_tokens=False)[:8]
    manifest = {'version': VERSION, 'model': 'Qwen/Qwen3-4B', 'revision': REVISION,
                'layers': LAYERS, 'heads': HEADS, 'prompt_ids': prompt_ids,
                'prompt_text': tokenizer.decode(prompt_ids), 'generated_tokens': 16,
                'dataset': HARMBENCH_URL, 'torch': str(torch.__version__)}
    meta_path = ROOT / 'results/section4_manifest.json'
    if meta_path.exists() and json.loads(meta_path.read_text()) != manifest:
        raise ValueError('Extraction manifest mismatch')
    write_json_atomic(meta_path, manifest)
    weights_path = ROOT / 'checkpoints/section4_weights.pt'
    if not weights_path.exists():
        weights = {}
        for layer in LAYERS:
            attn = model.model.layers[layer].self_attn
            for head in HEADS:
                kv = head // 4
                weights[f'{layer}_{head}'] = {
                    'wk': attn.k_proj.weight[kv * 128:(kv + 1) * 128].detach().float().cpu(),
                    'wv': attn.v_proj.weight[kv * 128:(kv + 1) * 128].detach().float().cpu(),
                    'wq': attn.q_proj.weight[head * 128:(head + 1) * 128].detach().float().cpu(),
                    'knorm': attn.k_norm.weight.detach().float().cpu(),
                    'qnorm': attn.q_norm.weight.detach().float().cpu(),
                    'epsilon': attn.k_norm.variance_epsilon,
                }
        temporary = weights_path.with_suffix('.tmp')
        torch.save(weights, temporary)
        os.replace(temporary, weights_path)
    for index, record in enumerate(records()[:limit]):
        path = ROOT / f'checkpoints/section4_activations_{index:03d}.pt'
        if path.exists():
            print(f'extract {index + 1}/{limit}: cached', flush=True)
            continue
        behavior = record['Behavior']
        context = record.get('ContextString', '').strip()
        text = f'{context}\n\n{behavior}' if context else behavior
        base = tokenizer.apply_chat_template([{'role': 'user', 'content': text}],
                    tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
        # All m conditions use nested prompt prefixes from the same causal trace.
        prefix = list(base) + prompt_ids
        generation_path = ROOT / f'checkpoints/section4_generation_{index:03d}.json'
        generation = json.loads(generation_path.read_text()) if generation_path.exists() else {
            'prefix_sha256': hashlib.sha256(json.dumps(prefix).encode()).hexdigest(), 'tokens': []}
        if generation['prefix_sha256'] != hashlib.sha256(json.dumps(prefix).encode()).hexdigest():
            raise ValueError('Generation checkpoint input mismatch')
        tokens = generation['tokens']
        cache = None
        inputs = torch.tensor([prefix + tokens], device='cuda')
        with torch.inference_mode():
            while len(tokens) < 16:
                output = model(inputs, past_key_values=cache, use_cache=True, logits_to_keep=1)
                cache = output.past_key_values
                token = int(output.logits[0, -1].argmax())
                tokens.append(token)
                write_json_atomic(generation_path, generation)
                inputs = torch.tensor([[token]], device='cuda')
            del cache, inputs
            captured = {}
            handles = []
            def hook(layer):
                def capture(_module, args, kwargs):
                    hidden = kwargs.get('hidden_states', args[0] if args else None)
                    captured[layer] = hidden[0].detach().cpu()
                return capture
            for layer in LAYERS:
                handles.append(model.model.layers[layer].self_attn.register_forward_pre_hook(hook(layer), with_kwargs=True))
            try:
                model(torch.tensor([prefix + tokens], device='cuda'), use_cache=False, logits_to_keep=1)
            finally:
                for handle in handles:
                    handle.remove()
        payload = {'index': index, 'behavior_id': record['BehaviorID'], 'base_length': len(base),
                   'tokens': prefix + tokens, 'layers': captured, 'manifest': manifest}
        temporary = path.with_suffix('.tmp')
        torch.save(payload, temporary)
        os.replace(temporary, path)
        print(f'extract {index + 1}/{limit}: {len(base)} input + 8 prompt + 16 continuation; saved', flush=True)
    print('extraction complete', flush=True)


def native_attention(keys, values, query, key_weights, query_weights, epsilon, positions, query_position):
    def normalized(x, weight):
        return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + epsilon) * weight
    def rope(x, positions):
        angle = np.asarray(positions)[..., None] * (1000000.0 ** (-np.arange(0, 128, 2) / 128))
        angle = np.concatenate([angle, angle], axis=-1)
        rotated = np.concatenate([-x[..., 64:], x[..., :64]], axis=-1)
        return x * np.cos(angle) + rotated * np.sin(angle)
    kn = rope(normalized(keys, key_weights), positions)
    qn = rope(normalized(query, query_weights), query_position)
    return softmax(kn @ qn / np.sqrt(128)) @ values


def analyze(limit, start=0, stride=1, suite='main'):
    import torch
    torch.set_num_threads(1)
    torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
    weights = torch.load(ROOT / 'checkpoints/section4_weights.pt', weights_only=True)
    all_conditions = conditions(suite)
    manifest = json.loads((ROOT / 'results/section4_manifest.json').read_text())
    manifest = {**manifest, 'analysis_version': 2, 'conditions': all_conditions,
                'path_points': 9, 'jacobian_anchor': 'last_input_token'}
    for index in range(start, limit, stride):
        activation_path = ROOT / f'checkpoints/section4_activations_{index:03d}.pt'
        waiting = time.monotonic()
        while not activation_path.exists():
            if time.monotonic() - waiting > 900:
                raise FileNotFoundError(activation_path)
            time.sleep(2)
        prefix = 'section4' if suite == 'main' else 'section4_redundancy'
        result_path = ROOT / f'results/{prefix}_{index:03d}.json'
        source_units = {}
        checkpoint_path = result_path
        unit_ids = [f'{layer}_{head}/{c["id"]}' for layer in LAYERS for head in HEADS for c in all_conditions]
        if result_path.exists():
            saved = json.loads(result_path.read_text())
            if saved['manifest'] == manifest and set(saved['units']) == set(unit_ids):
                print(f'analyze {index + 1}/{limit}: cached', flush=True)
                continue
            if saved['manifest'].get('analysis_version') == 1:
                # Preserve all measurements: only repair the Jacobian coordinates.
                source_units = dict(saved['units'])
                for entry in read_jsonl(result_path.with_suffix('.jsonl')):
                    source_units[entry['id']] = entry['result']
                checkpoint_path = ROOT / f'results/{prefix}_anchorfix_{index:03d}.json'
        payload = torch.load(activation_path, weights_only=True)
        n = payload['base_length']
        cache = {}
        by_id = {c['id']: c for c in all_conditions}
        def compute(unit):
            head_id, cid = unit.split('/')
            layer, head = map(int, head_id.split('_'))
            c = by_id[cid]
            previous = source_units.get(unit)
            if previous is not None and c['k'] == 1:
                return previous  # The original first selected position already was t_n.
            if head_id not in cache:
                h = payload['layers'][layer].float().numpy().astype(np.float64)
                w = {key: value.numpy().astype(np.float64) if hasattr(value, 'numpy') else value
                     for key, value in weights[head_id].items()}
                keys, vals, queries = h @ w['wk'].T, h @ w['wv'].T, h @ w['wq'].T
                q0 = queries[n - 1]
                b = w['wk'].T @ q0
                z = b - w['wv'].T @ np.linalg.lstsq(w['wv'].T, b, rcond=None)[0]
                cache[head_id] = {'h': h, 'w': w, 'keys': keys, 'vals': vals, 'queries': queries,
                                  'q0': q0, 'z': z, 'construct': {}, 'diameters': {}}
            state = cache[head_id]
            h, w, keys, vals, queries, q0, z = [state[key] for key in ['h', 'w', 'keys', 'vals', 'queries', 'q0', 'z']]
            m, k, g = c['m'], c['k'], c['g']
            if suite == 'redundancy':
                h = h.copy()
                if c['family'] in ['repeat_prompt', 'repeat_both']:
                    h[n:n + m] = h[n]
                if c['family'] in ['repeat_input', 'repeat_both']:
                    h[n - k:n] = h[n - 1]
                keys, vals = h @ w['wk'].T, h @ w['wv'].T
            selected = list(range(n - k, n))
            prompt = list(range(n, n + m))
            generated = list(range(n + 8, n + 24))
            initial = selected.copy()
            selected += generated[:g]
            shared = list(range(n - k)) + generated[g:]
            key = (m, k, g, c['family'] == 'rematch', c['family'] if suite == 'redundancy' else '')
            if key not in state['construct']:
                steering_positions = selected if c['family'] == 'rematch' else initial
                r = construct_shift(h, w['wk'], w['wv'], q0, steering_positions, prompt, z)
                dk, dv = w['wk'] @ r, w['wv'] @ r
                constraints = np.stack([keys[i] - keys[n - 1] for i in selected + prompt] + [dk], axis=1)
                basis, singular, _ = np.linalg.svd(constraints, full_matrices=False)
                basis = basis[:, singular > max(singular[0], 1.0) * 1e-10]
                rng = np.random.default_rng(42 + index * 100 + layer * 2 + head)
                random = rng.normal(size=128)
                sensitive = basis @ (basis.T @ random)
                null = random - sensitive
                directions = {name: value / max(np.linalg.norm(value), 1e-30) for name, value in
                              [('null', null), ('sensitive', sensitive), ('random', random)]}
                state['construct'][key] = r, dk, dv, directions, basis.shape[1]
            r, dk, dv, directions, rank = state['construct'][key]
            if c['direction'] == 'trajectory':
                q = queries[-1]
            elif c['direction'] == 'zero':
                q = q0
            else:
                q = q0 + c['rho'] * np.linalg.norm(q0) * directions[c['direction']]
            diameter_key = (m, c['family'] if suite == 'redundancy' else '')
            if diameter_key not in state['diameters']:
                state['diameters'][diameter_key] = diameter(vals[list(range(n + m)) + generated])
            if previous is not None:
                coordinates = [n - 1] + [i for i in selected if i != n - 1] + prompt
                scores0 = keys[coordinates] @ q0 / np.sqrt(len(q0))
                delta = keys[coordinates] @ (q - q0) / np.sqrt(len(q0))
                cert = jacobian_certificate(vals[coordinates], scores0, delta, len(selected))
                row = dict(previous)
                d, beta, ws, ek, er = [row[x] for x in ['diameter', 'beta', 'ws', 'epsilon_k', 'epsilon_r']]
                drift = ws * cert['upper'] * ek + d / 4 * (ek + er)
                row.update(jacobian_upper=cert['upper'], jacobian_sample=cert['sample_lower'],
                           bound_certified=min(d, beta + drift), bound_without_beta=min(d, drift),
                           bound_sampled=min(d, beta + ws * cert['sample_lower'] * ek + d / 4 * (ek + er)),
                           bound_global=min(d, beta + ws * cert['global'] * ek + d / 4 * (ek + er)))
            else:
                row = evaluate(keys, vals, q0, q, selected, prompt, shared, dk, dv,
                               original_diameter=state['diameters'][diameter_key], anchor=n - 1)
            if c['family'] == 'grid' and previous is None:
                # Architecture control at q0: frozen states, native Q/K norms and RoPE.
                ps = list(range(n + m))
                ss = list(range(n))
                kp, vp = keys[ps], vals[ps]
                ks, vs = keys[ss].copy(), vals[ss].copy()
                ks[initial] += dk
                vs[initial] += dv
                op = native_attention(kp, vp, q0, w['knorm'], w['qnorm'], w['epsilon'], ps, n - 1)
                os_ = native_attention(ks, vs, q0, w['knorm'], w['qnorm'], w['epsilon'], ss, n - 1)
                row['native_reference_error'] = float(np.linalg.norm(os_ - op))
                reference = evaluate(keys, vals, q0, q0, initial, prompt, list(range(n - k)), dk, dv)
                row['linear_reference_error'] = reference['error']
            row.update(c)
            row.update(index=index, behavior_id=payload['behavior_id'], layer=layer, head=head,
                       r_norm=float(np.linalg.norm(r)), query_norm=float(np.linalg.norm(q0)),
                       constraint_rank=int(rank))
            # Fail immediately, preserving all previous completed units.
            tol = 2e-8 * max(1.0, row['diameter'])
            chain = [row[x] for x in ['error', 'bound_decomp', 'bound_log', 'bound_certified', 'diameter']]
            if any(a > b + tol for a, b in zip(chain, chain[1:])):
                raise AssertionError(f'Bound chain failed: {unit}: {chain}')
            return row
        started = time.monotonic()
        run_units(checkpoint_path, manifest, unit_ids, compute)
        if checkpoint_path != result_path:
            os.replace(checkpoint_path, result_path)
            result_path.with_suffix('.jsonl').unlink(missing_ok=True)
        print(f'analyze {index + 1}/{limit}: {len(unit_ids)} conditions saved ({time.monotonic() - started:.1f}s)', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['extract', 'analyze'])
    parser.add_argument('--limit', type=int, default=400)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--suite', choices=['main', 'redundancy'], default='main')
    args = parser.parse_args()
    log = ROOT / f'logs/section4_{args.phase}_{args.suite}_{args.start}.log'
    with tee_stdout(log), contextlib.redirect_stderr(sys.stdout):
        if args.phase == 'extract':
            extract(args.limit)
        else:
            analyze(args.limit, args.start, args.stride, args.suite)


if __name__ == '__main__':
    main()
