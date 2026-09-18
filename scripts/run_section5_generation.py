"""Independent 128-token native-model trajectories for the Section 5 R/C study.

Intervene after the selected layer's input norm and before native attention.
Native QK normalization, RoPE, query changes and inherited effects remain active.
The primary generation experiment fixes layer 17/head 0 before measuring results.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys

import numpy as np

from prefix.attention_bounds import construct_shift
from prefix.attention_cosines import measured_cosines
from prefix.notify import notify_on_exit
from prefix.runner import tee_stdout, write_json_atomic
from run_section5_rc import ROOT, LENGTHS, REFERENCE_M, conditions, digest, positions, resume_units, wait_for_file


def export_inputs(path, load, limit):
    rows = json.loads(path.read_text()) if path.exists() else []
    if any(row['index'] != i for i, row in enumerate(rows)):
        raise ValueError('Input export checkpoint is not contiguous')
    for index in range(len(rows), limit):
        payload = load(index)
        rows.append(dict(index=index, behavior_id=payload['behavior_id'],
                         input_ids=payload['tokens'][:payload['base_length']]))
        write_json_atomic(path, rows)
        print(f'Input export {index + 1}/{limit} saved', flush=True)
    return rows


def first_eos_position(tokens, eos_ids):
    eos_ids = [eos_ids] if isinstance(eos_ids, int) else eos_ids
    return next((i+1 for i, token in enumerate(tokens) if token in eos_ids), None)


def selected_positions(method, k, input_length, absolute_positions):
    if method in ('unsteered', 'prompt'):
        return []
    if method == 'full':
        return list(range(len(absolute_positions)))
    return [j for j, position in enumerate(absolute_positions)
            if input_length - k <= position < input_length]


def decode_resume(path, manifest, predict, target=128):
    state = json.loads(path.read_text()) if path.exists() else {'manifest': manifest, 'tokens': []}
    if state['manifest'] != manifest:
        raise ValueError(f'Decode manifest mismatch: {path}')
    while len(state['tokens']) < target:
        token, output = predict(state['tokens'])
        state['tokens'].append(int(token))
        state['final_output'] = np.asarray(output, dtype=float).tolist()
        write_json_atomic(path, state)
    return state


def generate(limit, start, layer, head, condition_ids=None):
    import torch
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(4)
    torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
    source = json.loads((ROOT / 'results/section4_long_manifest.json').read_text())
    inputs_path = ROOT / 'data/section5_inputs.json'
    inputs = json.loads(inputs_path.read_text())
    weight_path = ROOT / 'checkpoints/section4_long_weights.pt'
    weights = torch.load(weight_path, weights_only=True)
    w = {key: weights[f'{layer}_{head}'][key].numpy().astype(np.float64) for key in ['wk','wv','wq']}
    manifest = dict(version=1, model=source['model'], revision=source['revision'], layer=layer, head=head,
                    inputs_sha256=digest(inputs_path), weights_sha256=digest(weight_path), conditions=conditions(),
                    target=128, direction='fixed m=8 first-step constructed value displacement',
                    decoding='greedy; ignore EOS to reach exactly 128 tokens; first EOS recorded',
                    intervention='normalized attention input, before native QK norm and RoPE',
                    reference_prompt_ids=source['prompt_ids'], transformers=__import__('transformers').__version__)
    model = AutoModelForCausalLM.from_pretrained(source['model'], revision=source['revision'],
                local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()
    attn = model.model.layers[layer].self_attn
    head_dim = model.config.head_dim
    for index in range(start, limit):
        activation_path = ROOT / f'checkpoints/section4_long_activations_{index:03d}.pt'
        wait_for_file(activation_path)
        payload = torch.load(activation_path, weights_only=True)
        example = inputs[index]
        if example['index'] != index or example['behavior_id'] != payload['behavior_id']:
            raise ValueError('Input identity mismatch')
        base_ids, n = example['input_ids'], payload['base_length']
        if len(base_ids) != n:
            raise ValueError('Input length mismatch')
        h = payload['layers'][layer].float().numpy().astype(np.float64)
        q0 = w['wq'] @ h[n-1]
        b = w['wk'].T @ q0
        z = b - w['wv'].T @ np.linalg.lstsq(w['wv'].T, b, rcond=None)[0]
        shifts = {m: construct_shift(h, w['wk'], w['wv'], q0, [n-1], positions(n,m)[1], z) for m in LENGTHS}
        d = w['wv'] @ shifts[REFERENCE_M]
        example_manifest = {**manifest, 'index': index, 'activation_sha256': digest(activation_path)}
        result_path = ROOT / f'results/section5_generation_{index:03d}.json'
        baseline_path = ROOT / f'checkpoints/section5_generation_{index:03d}_unsteered.json'

        def compute(condition_id):
            c = next(c for c in conditions() if c['id'] == condition_id)
            if c['method'] == 'prefix' and c['k'] > n:
                return {**c, 'eligible': False, 'reason': 'input shorter than selected block'}
            path = ROOT / f'checkpoints/section5_generation_{index:03d}_{condition_id}.json'
            decode_manifest = {**example_manifest, 'condition': c}
            # Alpha zero is exactly baseline; share completed work rather than regenerate it.
            if c['method'] not in ('prompt','unsteered') and c['alpha'] == 0:
                state = json.loads(baseline_path.read_text())
                state = {**state, 'manifest': decode_manifest, 'reused_condition': 'unsteered'}
                write_json_atomic(path, state)
            else:
                prefix = base_ids + (source['prompt_ids'][:c['m']] if c['method'] == 'prompt' else [])
                shift = torch.tensor(shifts[c['m']], device='cuda', dtype=model.dtype) * c['alpha']
                cache, cached_length, captured = None, 0, {}
                absolute = []
                def intervene(_module, args, kwargs):
                    hidden = kwargs['hidden_states'] if 'hidden_states' in kwargs else args[0]
                    selected = selected_positions(c['method'], c['k'], n, absolute)
                    if selected:
                        changed = hidden.clone()
                        changed[:, selected, :] += shift
                        if 'hidden_states' in kwargs:
                            kwargs = {**kwargs, 'hidden_states': changed}
                        else:
                            args = (changed, *args[1:])
                    return args, kwargs
                def capture(_module, args):
                    captured['output'] = args[0][0, -1, head*head_dim:(head+1)*head_dim].detach().float().cpu().numpy()
                handles = [attn.register_forward_pre_hook(intervene, with_kwargs=True),
                           attn.o_proj.register_forward_pre_hook(capture)]
                def predict(tokens):
                    nonlocal cache, cached_length, absolute
                    sequence = prefix + tokens
                    remaining = sequence[cached_length:]
                    absolute = list(range(cached_length, len(sequence)))
                    ids = torch.tensor([remaining], device='cuda')
                    with torch.inference_mode():
                        out = model(ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
                    cache = out.past_key_values
                    cached_length = len(sequence)
                    return out.logits[0,-1].argmax().item(), captured['output']
                try:
                    state = decode_resume(path, decode_manifest, predict)
                finally:
                    for handle in handles:
                        handle.remove()
                    del cache
            baseline = state if c['method'] == 'unsteered' else json.loads(baseline_path.read_text())
            metrics = measured_cosines(state['final_output'], baseline['final_output'], d)
            eos = model.generation_config.eos_token_id
            first_eos = first_eos_position(state['tokens'], eos)
            return {**c, **metrics, 'eligible': True, 'index': index, 'behavior_id': payload['behavior_id'],
                    'layer': layer, 'head': head, 'base_length': n, 'generated_count': len(state['tokens']),
                    'eos_token_ids': eos if isinstance(eos, list) else [eos],
                    'first_eos': first_eos, 'continued_after_eos': first_eos is not None and first_eos < 128,
                    'tokens_sha256': __import__('hashlib').sha256(json.dumps(state['tokens']).encode()).hexdigest()}
        requested = condition_ids or [c['id'] for c in conditions()]
        if requested[0] != 'unsteered':
            requested = ['unsteered', *requested]
        resume_units(result_path, example_manifest, requested, compute)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=400)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--layer', type=int, default=17)
    parser.add_argument('--head', type=int, default=0)
    parser.add_argument('--conditions', nargs='+', choices=[c['id'] for c in conditions()])
    parser.add_argument('--prepare-inputs', action='store_true')
    args = parser.parse_args()
    log = ROOT / f'logs/section5_generation_{args.start}.log'
    with tee_stdout(log), contextlib.redirect_stderr(sys.stdout), notify_on_exit('section5-generation', log_file=str(log)):
        if args.prepare_inputs:
            import torch
            torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
            export_inputs(ROOT / 'data/section5_inputs.json',
                          lambda i: torch.load(ROOT / f'checkpoints/section4_activations_{i:03d}.pt', weights_only=True),
                          args.limit)
        else:
            generate(args.limit, args.start, args.layer, args.head, args.conditions)


if __name__ == '__main__':
    main()
