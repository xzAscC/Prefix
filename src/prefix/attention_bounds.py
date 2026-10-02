"""Float64, fixed-state attention replay for Section 4 (no model approximations hidden).

The path certificate uses a global Lipschitz envelope between sampled Jacobians.
For a block E, ||dJ_E/dtau||_(infinity->2) <= 2 range(delta_E) diameter(V_E).
The difference of two blocks uses the sum of these constants. Each pointwise
operator norm is exact by sign enumeration for <= 8 columns, otherwise bounded
above by the sum of column norms. Finite sampling alone is never called a bound.
"""

from functools import lru_cache

import numpy as np


def kv_head_index(query_head: int, *, num_attention_heads: int, num_key_value_heads: int) -> int:
    """Map a query head to its key/value head for MHA or grouped-query attention."""
    if num_key_value_heads < 1 or num_attention_heads < 1:
        raise ValueError("attention head counts must be positive")
    if num_attention_heads % num_key_value_heads:
        raise ValueError("query head count must be divisible by key/value head count")
    if not 0 <= query_head < num_attention_heads:
        raise ValueError("query head is outside the attention head range")
    return query_head // (num_attention_heads // num_key_value_heads)


def softmax(x):
    z = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return z / z.sum(axis=-1, keepdims=True)


def logsumexp(x):
    maximum = np.max(x)
    return float(maximum + np.log(np.exp(x - maximum).sum()))


def maximum_distance(left, right):
    """Largest cross distance, using centered Gram products to bound memory."""
    origin = right[0]
    left, right = left - origin, right - origin
    right_norms = np.sum(right * right, axis=1)
    largest = 0.0
    for start in range(0, len(left), 256):
        part = left[start:start + 256]
        squared = np.sum(part * part, axis=1)[:, None] + right_norms[None] - 2 * (part @ right.T)
        largest = max(largest, float(squared.max()))
    return np.sqrt(largest)


def diameter(values):
    return 0.0 if len(values) < 2 else maximum_distance(values, values)


def construct_shift(h, wk, wv, q0, selected, prompt, null_vector=None):
    scores = h @ (wk.T @ q0) / np.sqrt(len(q0))
    block = selected + prompt
    delta = softmax(scores[block]) @ h[block] - softmax(scores[selected]) @ h[selected]
    b = wk.T @ q0
    z = null_vector
    if z is None:
        z = b - wv.T @ np.linalg.lstsq(wv.T, b, rcond=None)[0]
    denominator = float(b @ z)
    if abs(denominator) <= 1e-12 * max(1.0, np.linalg.norm(b) * np.linalg.norm(z)):
        raise ValueError('No numerically usable value-null direction')
    needed = np.sqrt(len(q0)) * (logsumexp(scores[block]) - logsumexp(scores[selected]))
    return delta + (needed - b @ delta) / denominator * z


@lru_cache(maxsize=16)
def signs(p):
    if p <= 8:
        return 2 * ((np.arange(2 ** p)[:, None] >> np.arange(p)) & 1) - 1
    rng = np.random.default_rng(p)
    return rng.choice([-1, 1], size=(32, p))


def jacobian_certificate(values, s0, delta, selected_count, points=9):
    """Anchor is position 0; selected positions precede prompt positions."""
    p = len(values) - 1
    db, ds = diameter(values), diameter(values[:selected_count])
    if p == 1 and selected_count == 1:
        endpoints = [s0[1] - s0[0], s0[1] - s0[0] + delta[1] - delta[0]]
        closest = np.clip(0.0, min(endpoints), max(endpoints))
        share = softmax(np.array([0.0, closest]))[1]
        exact = float(share * (1 - share) * db)
        return {'upper': exact, 'sample_lower': exact, 'global': db, 'lipschitz': 0.0}
    score = s0[None] + np.linspace(0, 1, points)[:, None] * delta[None]
    pb, ps = softmax(score), softmax(score[:, :selected_count])
    meanb, means = pb @ values, ps @ values[:selected_count]
    jb = pb[:, :, None] * (values[None] - meanb[:, None])
    js = ps[:, :, None] * (values[None, :selected_count] - means[:, None])
    jb[:, :selected_count] -= js
    jac = jb[:, 1:, :]
    transformed = np.einsum('sp,tpv->tsv', signs(p), jac, optimize=True)
    lower = np.linalg.norm(transformed, axis=-1).max(axis=-1)
    point_upper = lower if p <= 8 else np.linalg.norm(jac, axis=-1).sum(axis=-1)
    lip = 2 * (np.ptp(delta) * db + np.ptp(delta[:selected_count]) * ds)
    upper = min(db + ds, float(point_upper.max() + lip / (2 * (points - 1))))
    return {'upper': upper, 'sample_lower': float(lower.max()),
            'global': db + ds, 'lipschitz': float(lip)}


def evaluate(keys, values, q0, q, selected, prompt, shared, key_shift, value_shift,
             points=9, original_diameter=None, anchor=None):
    """Evaluate outputs independently, then the proof's intermediate bounds."""
    block = selected + prompt
    scale = np.sqrt(len(q))
    scores, scores0 = keys @ q / scale, keys @ q0 / scale
    shift, shift0 = float(q @ key_shift / scale), float(q0 @ key_shift / scale)
    ap, ass = logsumexp(scores[block]), logsumexp(scores[selected]) + shift
    vp = softmax(scores[block]) @ values[block]
    vs = softmax(scores[selected]) @ values[selected] + value_shift
    if shared:
        z = logsumexp(scores[shared])
        wp, ws = softmax(np.array([z, ap]))[1], softmax(np.array([z, ass]))[1]
        context = softmax(scores[shared]) @ values[shared]
    else:
        wp = ws = 1.0
        context = np.zeros(values.shape[1])
    prompt_values = values[shared + block]
    steer_values = np.concatenate([values[shared], values[selected] + value_shift])
    op = softmax(scores[shared + block]) @ prompt_values
    os = softmax(np.concatenate([scores[shared], scores[selected] + shift])) @ steer_values
    modified = values[selected] + value_shift
    base_d = diameter(prompt_values) if original_diameter is None else original_diameter
    cross_d = maximum_distance(modified, prompt_values)
    d = max(base_d, cross_d, diameter(modified))
    ev, ew = ws * (vs - vp), (ws - wp) * (vp - context)
    ek = float(np.max(np.abs((scores - scores0)[block, None] - (scores - scores0)[selected])))
    er = abs(shift - shift0)
    base_value = (softmax(scores0[block]) @ values[block]
                  - softmax(scores0[selected]) @ values[selected] - value_shift)
    base_log = shift0 - (logsumexp(scores0[block]) - logsumexp(scores0[selected]))
    beta_value, beta_weight = ws * np.linalg.norm(base_value), d / 4 * abs(base_log)
    anchor = selected[0] if anchor is None else anchor
    if anchor not in selected:
        raise ValueError('Jacobian anchor must belong to the steered block')
    coordinates = [anchor] + [i for i in selected if i != anchor] + prompt
    certificate = jacobian_certificate(values[coordinates], scores0[coordinates],
                                       (scores - scores0)[coordinates], len(selected), points)
    drift = ws * certificate['upper'] * ek + d / 4 * (ek + er)
    sample_drift = ws * certificate['sample_lower'] * ek + d / 4 * (ek + er)
    beta = beta_value + beta_weight
    b2 = min(d, (ws * np.linalg.norm(values[prompt[0]] - values[selected[0]]) + d) / 4 * ek + d / 4 * er)
    return {k: float(v) for k, v in {
        'error': np.linalg.norm(os - op), 'diameter': d,
        'bound_decomp': min(d, np.linalg.norm(ev) + np.linalg.norm(ew)),
        'bound_log': min(d, np.linalg.norm(ev) + d / 4 * abs(ass - ap)),
        'bound_certified': min(d, beta + drift),
        'bound_sampled': min(d, beta + sample_drift),
        'bound_global': min(d, beta + ws * certificate['global'] * ek + d / 4 * (ek + er)),
        'bound_without_beta': min(d, drift),
        'bound_lemma2': b2 if len(selected) == len(prompt) == 1 else np.nan,
        'value_error': np.linalg.norm(ev), 'weight_error': np.linalg.norm(ew),
        'component_cosine': ev @ ew / max(1e-300, np.linalg.norm(ev) * np.linalg.norm(ew)),
        'beta': beta, 'beta_value': beta_value, 'beta_weight': beta_weight,
        'epsilon_k': ek, 'epsilon_r': er, 'jacobian_upper': certificate['upper'],
        'jacobian_sample': certificate['sample_lower'], 'ws': ws,
        'identity_residual': np.linalg.norm(os - op - ev - ew),
    }.items()}
