"""Native generated-trajectory comparisons of duration and calibrated strength.

Outputs are read directly from the input of the model's o_proj, retaining native
QK normalization, RoPE, changing queries, and generated token trajectories.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import random

import numpy as np
import torch

from prefix.runner import write_json_atomic


def select_cohort(rows):
    rows = sorted(rows, key=lambda r: (r['SemanticCategory'], r['Behavior']))
    if len({r['BehaviorID'] for r in rows}) != len(rows) or len(rows) < 100:
        raise ValueError('need at least 100 unique behavior IDs')
    order = list(range(len(rows)))
    random.Random(42).shuffle(order)
    result = []
    for i in order[:100]:
        row = rows[i]
        context = row['ContextString'].strip()
        text = (context + '\n\n' if context else '') + row['Behavior']
        result.append(dict(source_index=i, behavior_id=row['BehaviorID'],
                           category=row['SemanticCategory'], text=text))
    return result


def selected_mask(absolute_positions, prompt_length, length):
    return [prompt_length - 1 <= p < prompt_length - 1 + length
            for p in absolute_positions]


def calibrated_strength(weights, prompt_length, length, alpha):
    start = prompt_length - 1
    if start < 0 or length < 1 or start + length > len(weights):
        raise ValueError('calibration support is outside the visible context')
    denominator = float(weights[start])
    if denominator <= 0:
        raise ValueError('selected position has zero attention mass')
    beta = alpha * math.fsum(weights[start:start + length]) / denominator
    if not math.isfinite(beta):
        raise ValueError('nonfinite calibrated strength')
    return beta


def sample_summary(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('summary requires finite observations')
    return dict(n=len(values), mean=float(values.mean()),
                std=float(values.std(ddof=1)) if len(values) > 1 else None)


def decode(model, prompt_ids, direction, arms, layer, head, path, manifest,
           target=128, on_step=None, capture_attention=False):
    """Generate synchronized independent arms, checkpointing every new token.

On resume, teacher-force saved tokens once to rebuild KV; completed tokens are
never regenerated. EOS does not stop the fixed-length diagnostic and is recorded.
An intervention length includes the final prompt position, which predicts y_1.
The final captured output predicts y_target (not the state after y_target).
"""
    path = Path(path)
    identity = dict(manifest=manifest, prompt_ids=prompt_ids, arms=arms,
                    layer=layer, head=head, target=target,
                    capture_attention=capture_attention,
                    direction=direction.detach().cpu().float().tolist())
    state = json.loads(path.read_text()) if path.exists() else dict(
        identity=identity, tokens=[[] for _ in arms], trajectory_l2=[], complete=False)
    if state['identity'] != identity:
        raise ValueError(f'decode identity mismatch: {path}')
    if state['complete']:
        return state
    lengths = {len(tokens) for tokens in state['tokens']}
    if len(lengths) != 1 or len(state['tokens']) != len(arms):
        raise ValueError('inconsistent saved batch')
    attn = model.model.layers[layer].self_attn
    head_dim = attn.head_dim
    if not 0 <= head < model.config.num_attention_heads:
        raise ValueError('head out of range')
    device = next(model.parameters()).device
    direction = direction.to(device=device, dtype=model.dtype)
    shifts = torch.tensor([arm['strength'] for arm in arms], device=device,
                          dtype=model.dtype)[:, None, None] * direction[None, None]
    original_config = attn.config
    # Request genuine native attention weights only for baseline calibration.
    if capture_attention:
        attn.config = copy.copy(attn.config)
        attn.config._attn_implementation = 'eager'
    captured, absolute = {}, []
    def intervene(_module, args, kwargs):
        hidden = kwargs.get('hidden_states', args[0] if args else None)
        mask = torch.tensor([selected_mask(absolute, len(prompt_ids), a['length'])
                             for a in arms], device=device, dtype=hidden.dtype)
        changed = hidden + mask[..., None] * shifts
        if 'hidden_states' in kwargs:
            return args, {**kwargs, 'hidden_states': changed}
        return (changed, *args[1:]), kwargs
    def capture_output(_module, args):
        captured['o'] = args[0][:, -1, head*head_dim:(head+1)*head_dim].detach().float().cpu()
    def capture_weights(_module, args, output):
        if output[1] is not None:
            captured['weights'] = output[1][:, head, -1].detach().float().cpu()
    handles = [attn.register_forward_pre_hook(intervene, with_kwargs=True),
               attn.o_proj.register_forward_pre_hook(capture_output)]
    if capture_attention:
        handles.append(attn.register_forward_hook(capture_weights))
    cache, cached_length = None, 0
    try:
        while len(state['tokens'][0]) < target:
            sequences = [prompt_ids + tokens for tokens in state['tokens']]
            absolute = list(range(cached_length, len(sequences[0])))
            ids = torch.tensor([seq[cached_length:] for seq in sequences], device=device)
            with torch.inference_mode():
                out = model(ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache = out.past_key_values
            cached_length = len(sequences[0])
            next_tokens = out.logits[:, -1].argmax(-1).tolist()
            for tokens, token in zip(state['tokens'], next_tokens):
                tokens.append(token)
            output = captured['o']
            if not torch.isfinite(output).all():
                raise ValueError('nonfinite native head output')
            state['outputs'] = output.tolist()
            # Arms are arranged in (distributed, calibrated-single) pairs.
            paired = 2 * (len(arms) // 2)
            if paired:
                state['trajectory_l2'].append((output[:paired:2]-output[1:paired:2]).norm(dim=-1).tolist())
            if capture_attention:
                state['attention'] = captured['weights'].tolist()
            state['complete'] = len(state['tokens'][0]) == target
            eos = model.generation_config.eos_token_id
            eos = [] if eos is None else ([eos] if isinstance(eos, int) else eos)
            state['first_eos'] = [next((i+1 for i,t in enumerate(ts) if t in eos), None)
                                  for ts in state['tokens']]
            write_json_atomic(path, state)
            if on_step:
                on_step(state)
    finally:
        for handle in handles:
            handle.remove()
        attn.config = original_config
        del cache
    return state
