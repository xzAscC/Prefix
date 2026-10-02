"""Section 5: fixed-state R/C at the prediction of generated token 128.

Reuses Section 4's linear, pre-QK-normalization/pre-RoPE head replay.
The shared continuation came from an eight-token prompted run, not baseline
free generation. No behavioral interpretation is attached to these cosines.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from prefix.attention_bounds import construct_shift
from prefix.attention_cosines import attention_output, measured_cosines, compare_preservation
from prefix.notify import notify_on_exit
from prefix.runner import tee_stdout, write_json_atomic

ROOT = Path(__file__).resolve().parents[1]
LENGTHS = [1, 8, 32, 128]
ALPHAS = [0, .25, .5, 1, 2]
REFERENCE_M = 8


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def wait_for_file(path, timeout=3600):
    started = time.monotonic()
    while not path.exists():
        if time.monotonic() - started > timeout:
            raise FileNotFoundError(path)
        time.sleep(2)


def positions(n, m):
    """y_127 supplies the query predicting y_128; y_128 is never visible."""
    return list(range(n)) + list(range(n + 128, n + 128 + 127)), list(range(n, n + m)), n + 128 + 126


def conditions():
    rows = [dict(id='unsteered', method='unsteered', m=REFERENCE_M, alpha=0, k=0)]
    for m in LENGTHS:
        rows.append(dict(id=f'prompt_m{m}', method='prompt', m=m, alpha=0, k=0))
        for alpha in ALPHAS:
            for method, k in [('single', 1), ('full', -1)]:
                rows.append(dict(id=f'{method}_m{m}_a{alpha:g}', method=method, m=m, alpha=alpha, k=k))
    for k in [4, 16, 64]:
        for alpha in ALPHAS:
            rows.append(dict(id=f'prefix_k{k}_a{alpha:g}', method='prefix', m=REFERENCE_M, alpha=alpha, k=k))
    return rows


def resume_units(path, manifest, units, compute):
    state = json.loads(path.read_text()) if path.exists() else {'manifest': manifest, 'units': {}}
    if state['manifest'] != manifest:
        raise ValueError(f'manifest mismatch: {path}')
    for unit in units:
        if unit not in state['units']:
            state['units'][unit] = compute(unit)
            write_json_atomic(path, state)
            print(f'{path.stem}: {unit} saved', flush=True)
    return state


def analyze(limit, start, stride, layers, heads):
    import torch
    torch.set_num_threads(1)
    torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
    weight_path = ROOT / 'checkpoints/section4_long_weights.pt'
    weights = torch.load(weight_path, weights_only=True)
    source = json.loads((ROOT / 'results/section4_long_manifest.json').read_text())
    manifest = dict(version=1, source=source, weights_sha256=digest(weight_path),
                    conditions=conditions(), reference_prompt_length=REFERENCE_M,
                    step=128, visible_generated=127, layers=layers, heads=heads,
                    direction='normalized W_V r from m=8, k=1 at first-step query',
                    displacement='construct on last input token; reuse for all schedules',
                    scope='fixed-state linear head replay before QK norm and RoPE')
    for index in range(start, limit, stride):
        activation_path = ROOT / f'checkpoints/section4_long_activations_{index:03d}.pt'
        wait_for_file(activation_path)
        example_manifest = {**manifest, 'activation_sha256': digest(activation_path), 'index': index}
        result_path = ROOT / f'results/section5_fixed_{index:03d}.json'
        units = [f'{layer}_{head}' for layer in layers for head in heads]
        # Load only if at least one head remains unfinished.
        old = json.loads(result_path.read_text()) if result_path.exists() else None
        if old and old['manifest'] == example_manifest and set(old['units']) == set(units):
            print(f'{result_path.stem}: complete, skipped', flush=True)
            continue
        payload = torch.load(activation_path, weights_only=True)
        if len(payload['generated_ids']) != 128:
            raise ValueError('Expected exactly 128 cached generated tokens')
        n = payload['base_length']
        def compute(head_id):
            layer, head = map(int, head_id.split('_'))
            h = payload['layers'][layer].float().numpy().astype(np.float64)
            w = {key: weights[head_id][key].numpy().astype(np.float64) for key in ['wk', 'wv', 'wq']}
            base, _, target = positions(n, REFERENCE_M)
            keys, values = h @ w['wk'].T, h @ w['wv'].T
            q0, q = w['wq'] @ h[n - 1], w['wq'] @ h[target]
            b = w['wk'].T @ q0
            z = b - w['wv'].T @ np.linalg.lstsq(w['wv'].T, b, rcond=None)[0]
            shifts = {m: construct_shift(h, w['wk'], w['wv'], q0, [n - 1], positions(n, m)[1], z)
                      for m in LENGTHS}
            d = w['wv'] @ shifts[REFERENCE_M]
            baseline = attention_output(keys[base], values[base], q)
            def compute_condition(condition_id):
                condition = next(c for c in conditions() if c['id'] == condition_id)
                c = dict(condition)
                if c['method'] == 'prefix' and n < c['k']:
                    return {**c, 'eligible': False, 'reason': 'input shorter than selected block'}
                if c['method'] == 'unsteered':
                    output = baseline
                    count = 0
                elif c['method'] == 'prompt':
                    subset = base + positions(n, c['m'])[1]
                    output = attention_output(keys[subset], values[subset], q)
                    count = 0
                else:
                    selected = list(range(len(base))) if c['method'] == 'full' else list(range(n-c['k'], n))
                    r = shifts[c['m']]
                    output = attention_output(keys[base], values[base], q, selected, w['wk'] @ r, w['wv'] @ r, c['alpha'])
                    count = len(selected)
                metrics = measured_cosines(output, baseline, d)
                residual = (metrics['R'] - (1 - (metrics['delta_C']**2 + metrics['delta_perp']) / 2)
                            if metrics['C'] is not None else None)
                if residual is not None and abs(residual) > 1e-10:
                    raise AssertionError('Section 5 decomposition failed')
                return {**c, **metrics, 'eligible': True, 'steered_count': count,
                        'identity_residual': residual}
            checkpoint = ROOT / f'checkpoints/section5_fixed_{index:03d}_{head_id}.json'
            saved = resume_units(checkpoint, {**example_manifest, 'head_id': head_id},
                                 [c['id'] for c in conditions()], compute_condition)
            rows = list(saved['units'].values())
            by_id = {r['id']: r for r in rows}
            comparisons = []
            for m in LENGTHS:
                for alpha in ALPHAS:
                    if by_id[f'single_m{m}_a{alpha:g}']['C'] is None:
                        comparisons.append(dict(m=m, alpha=alpha, concept_defined=False))
                        continue
                    comparison = compare_preservation(by_id[f'single_m{m}_a{alpha:g}'], by_id[f'full_m{m}_a{alpha:g}'])
                    if comparison['R_gap'] < comparison['lower_bound'] - 1e-10:
                        raise AssertionError('Preservation inequality failed')
                    comparisons.append(dict(m=m, alpha=alpha, **comparison))
            return dict(index=index, behavior_id=payload['behavior_id'], layer=layer, head=head,
                        base_length=n, rows=rows, comparisons=comparisons,
                        direction_norm=float(np.linalg.norm(d)),
                        shift_norms={str(m): float(np.linalg.norm(r)) for m, r in shifts.items()})
        resume_units(result_path, example_manifest, units, compute)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=400)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--layers', type=int, nargs='+', default=[2, 8, 17, 26, 33])
    parser.add_argument('--heads', type=int, nargs='+', default=[0, 16])
    args = parser.parse_args()
    log = ROOT / f'logs/section5_fixed_{args.start}.log'
    with tee_stdout(log), contextlib.redirect_stderr(sys.stdout), notify_on_exit('section5-fixed', log_file=str(log)):
        analyze(args.limit, args.start, args.stride, args.layers, args.heads)


if __name__ == '__main__':
    main()
