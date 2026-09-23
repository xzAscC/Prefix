"""Frozen-state Qwen3 head interventions with native QK norm, RoPE, and causality.

Prompt and steering arms use their own compact position IDs. A common hidden
query therefore has the appropriate rotary position in each arm. Fitting is
head-specific; these measurements are not whole-model generation outcomes.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from prefix.runner import write_json_atomic

LENGTHS = (1, 2, 4, 8, 16, 32, 64, 128)


def conditions():
    rows = [('input', 4, b, 0) for b in LENGTHS]
    rows += [('mixed', 4, 1, total - 1) for total in LENGTHS[1:]]
    rows += [('prompt', m, 1, 0) for m in LENGTHS if m != 4]
    return [dict(id=f'm{m}_b{b}_g{g}', family=family, m=m, b=b, g=g)
            for family, m, b, g in rows]


def query_sets(n, prompt_slots, generated, b, g):
    if not (1 <= b <= n and 0 <= g < generated and generated >= 2):
        raise ValueError('invalid steering/query support')
    start = n + prompt_slots
    reference = start + generated - 1
    return {
        'reference': reference,
        'all': list(range(n - b)) + list(range(start + g, reference)),
        'pre_prompt': list(range(n - b)),
        'exposed': list(range(start + g, reference)),
        'common': list(range(start + max(LENGTHS) - 1, reference)),
    }


def rmsnorm(x, weight, epsilon):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + epsilon) * weight


def rope(x, positions, theta):
    d = x.shape[-1]
    frequencies = theta ** (-torch.arange(0, d, 2, dtype=x.dtype, device=x.device) / d)
    angles = positions.to(x.dtype)[:, None] * frequencies
    angles = torch.cat([angles, angles], -1)
    rotated = torch.cat([-x[..., d // 2:], x[..., :d // 2]], -1)
    return x * angles.cos() + rotated * angles.sin()


class NativeHead:
    def __init__(self, hidden, weights, n, prompt_slots=128, generated=256, theta=1e6):
        self.h, self.w = hidden, weights
        self.n, self.prompt_slots, self.generated, self.theta = n, prompt_slots, generated, theta
        if len(hidden) != n + prompt_slots + generated:
            raise ValueError('hidden-state library length mismatch')
        self.q = hidden @ weights['wq'].T
        self.k = hidden @ weights['wk'].T
        self.v = hidden @ weights['wv'].T
        self.projection = torch.cat([weights['wk'], weights['wv']])
        # Only this row space affects K/V. Optimizing its projected coordinates
        # is exactly optimizing a realizable hidden-state displacement r.
        gram = self.projection @ self.projection.T
        self.lift = self.projection.T @ torch.linalg.solve(gram, torch.eye(
            len(gram), dtype=gram.dtype, device=gram.device))
        residual = (self.projection @ self.lift - torch.eye(
            len(gram), dtype=gram.dtype, device=gram.device)).abs().max().item()
        if residual > 1e-6:
            raise ValueError(f'K/V row-space lift is not reliable: {residual}')
        self._cache = {}

    def context(self, m, b, g, arm):
        if not (1 <= m <= self.prompt_slots and 1 <= b <= self.n and 0 <= g < self.generated):
            raise ValueError('invalid token configuration')
        generation = list(range(self.n + self.prompt_slots, len(self.h)))
        selected = list(range(self.n - b, self.n))
        if arm == 'prompt':
            return list(range(self.n + m)) + generation, []
        if arm != 'steer':
            raise ValueError('unknown attention arm')
        return list(range(self.n)) + generation, selected + list(range(self.n, self.n + g))

    def prepared(self, m, b, g, arm):
        key = (m, b, g, arm)
        if key not in self._cache:
            indices, selected = self.context(m, b, g, arm)
            positions = torch.arange(len(indices), device=self.h.device)
            mapping = {value: i for i, value in enumerate(indices)}
            affected = torch.zeros(len(indices), dtype=self.h.dtype, device=self.h.device)
            affected[selected] = 1
            self._cache[key] = (indices, positions, mapping, affected)
        return self._cache[key]

    def projected_outputs(self, m, b, g, queries, shift=None, arm='steer', raw_queries=None, query_shift=None):
        indices, positions, mapping, affected = self.prepared(m, b, g, arm)
        qp = torch.tensor([mapping[i] for i in queries], device=self.h.device)
        keys, values = self.k[indices], self.v[indices]
        raw = self.q[queries] if raw_queries is None else raw_queries
        if shift is not None:
            dk, dv = shift.chunk(2)
            keys = keys + affected[:, None] * dk
            values = values + affected[:, None] * dv
            # Used by module-parity tests; experiment queries always exclude S.
            if query_shift is None:
                query_shift = self.w['wq'] @ (self.lift @ shift)
            raw = raw + affected[qp, None] * query_shift
        q = rope(rmsnorm(raw, self.w['qnorm'], self.w['epsilon']), qp, self.theta)
        k = rope(rmsnorm(keys, self.w['knorm'], self.w['epsilon']), positions, self.theta)
        scores = q @ k.T / math.sqrt(k.shape[-1])
        scores = scores.masked_fill(positions[None] > qp[:, None], -torch.inf)
        return scores.softmax(-1) @ values

    def outputs(self, m, b, g, queries, r=None, arm='steer', raw_queries=None):
        shift = None if r is None else self.projection @ r
        query_shift = None if r is None else self.w['wq'] @ r
        return self.projected_outputs(m, b, g, queries, shift, arm, raw_queries, query_shift)

    def theoretical_directions(self, m, b, g):
        """Original linear-key diagnostic; NOT a native-attention theorem."""
        selected = list(range(self.n - b, self.n))
        selected += list(range(self.n + self.prompt_slots, self.n + self.prompt_slots + g))
        prompt = list(range(self.n, self.n + m))
        anchor = self.n - 1
        block = [i for i in selected + prompt if i != anchor]
        q0 = self.q[-1]
        key_gradient = self.w['wk'].T @ q0
        wv = self.w['wv']
        z = key_gradient - wv.T @ torch.linalg.solve(wv @ wv.T, wv @ key_gradient)
        z = z / z.norm().clamp_min(1e-30)
        directions = torch.cat([self.k[block] - self.k[anchor], (self.w['wk'] @ z)[None]])
        return directions, z


def _encode(value):
    if torch.is_tensor(value):
        return {'tensor': value.detach().cpu().tolist()}
    if isinstance(value, dict):
        return {'dict': [[key, _encode(item)] for key, item in value.items()]}
    if isinstance(value, (list, tuple)):
        return [_encode(item) for item in value]
    return value


def _decode(value, like):
    if isinstance(value, dict) and 'tensor' in value:
        return torch.tensor(value['tensor'], dtype=like.dtype, device=like.device)
    if isinstance(value, dict) and 'dict' in value:
        return {key: _decode(item, like) for key, item in value['dict']}
    if isinstance(value, list):
        return [_decode(item, like) for item in value]
    return value


def fit_shift(study, m, b, g, path, manifest, max_steps=100, tolerance=1e-6, on_checkpoint=None):
    """Fit r to native head output at the final query; persist every LBFGS step."""
    path = Path(path)
    identity = dict(manifest=manifest, m=m, b=b, g=g, max_steps=max_steps, tolerance=tolerance)
    saved = json.loads(path.read_text()) if path.exists() else None
    if saved is not None and saved['identity'] != identity:
        raise ValueError(f'optimizer manifest mismatch: {path}')
    if saved is not None and saved['complete']:
        return saved['result']
    q = [len(study.h) - 1]
    target = study.projected_outputs(m, b, g, q, arm='prompt').detach()
    shift = torch.zeros(study.projection.shape[0], device=study.h.device,
                        dtype=study.h.dtype, requires_grad=True)
    optimizer = torch.optim.LBFGS([shift], lr=1., max_iter=1, max_eval=12,
                                history_size=12, line_search_fn='strong_wolfe',
                                tolerance_grad=1e-14, tolerance_change=1e-16)
    with torch.no_grad():
        initial_error = float((study.projected_outputs(m,b,g,q,shift) - target).norm())
    state = saved or dict(identity=identity, steps=0, initial_error=initial_error, complete=False)
    if saved is not None:
        with torch.no_grad():
            shift.copy_(torch.tensor(saved['shift'], device=shift.device, dtype=shift.dtype))
        optimizer.load_state_dict(_decode(saved['optimizer'], shift))

    def error():
        return study.projected_outputs(m,b,g,q,shift) - target

    def closure():
        optimizer.zero_grad()
        loss = error().square().sum()
        loss.backward()
        return loss

    final_error = float(error().detach().norm())
    for step in range(state['steps'], max_steps):
        if final_error <= tolerance:
            break
        optimizer.step(closure)
        final_error = float(error().detach().norm())
        if not math.isfinite(final_error):
            raise FloatingPointError('nonfinite native attention fitting error')
        state.update(steps=step+1, shift=shift.detach().cpu().tolist(),
                     optimizer=_encode(optimizer.state_dict()), last_error=final_error)
        write_json_atomic(path, state)
        if on_checkpoint is not None:
            on_checkpoint(state)
    r = study.lift @ shift.detach()
    result = dict(r=r.cpu().tolist(), shift=shift.detach().cpu().tolist(),
                  initial_error=state['initial_error'], final_error=final_error,
                  steps=state['steps'], converged=final_error <= tolerance,
                  r_norm=float(r.norm()), target_norm=float(target.norm()))
    state.update(complete=True, result=result)
    state.pop('optimizer', None)
    write_json_atomic(path, state)
    return result


def subspace(directions, rtol=1e-7):
    _, singular, vh = torch.linalg.svd(directions, full_matrices=True)
    threshold = rtol * float(singular[0]) if len(singular) else 0.
    rank = int((singular > threshold).sum())
    return vh, rank


def perturbations(basis, rank, norm, seed):
    generator = torch.Generator(device=basis.device).manual_seed(seed)
    result = {}
    for name, space in [('null', basis[rank:]), ('sensitive', basis[:rank]), ('random', basis)]:
        if len(space) == 0:
            continue
        coefficients = torch.randn(len(space), generator=generator, dtype=basis.dtype, device=basis.device)
        direction = coefficients @ space
        result[name] = direction * (norm / direction.norm().clamp_min(1e-30))
    return result
