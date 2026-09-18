"""Section 5 head-output cosines and preservation identities in float64."""
import numpy as np

from prefix.attention_bounds import softmax


def attention_output(keys, values, query, selected=(), key_shift=None,
                     value_shift=None, alpha=1.0):
    """Replay linear single-head attention, holding query and original states fixed."""
    keys, values, query = (np.asarray(x, dtype=np.float64) for x in (keys, values, query))
    scores = keys @ query / np.sqrt(query.size)
    shifted = values.copy()
    if len(selected):
        indices = np.asarray(selected, dtype=int)
        scores[indices] += alpha * (query @ key_shift) / np.sqrt(query.size)
        shifted[indices] += alpha * value_shift
    return softmax(scores) @ shifted


def cosine_metrics(output, baseline, direction):
    """Use the same baseline and concept direction for every compared method."""
    vectors = [np.asarray(x, dtype=np.float64) for x in (output, baseline, direction)]
    norms = [np.linalg.norm(x) for x in vectors]
    if any(not np.isfinite(n) or n <= 1e-30 for n in norms):
        raise ValueError('Cosines require finite nonzero output, baseline and direction')
    a, b, d = [x / n for x, n in zip(vectors, norms)]
    c, c0 = float(a @ d), float(b @ d)
    change = a - b
    perpendicular = change - (change @ d) * d
    return {'R': float(a @ b), 'C': c, 'C0': c0, 'delta_C': c - c0,
            'delta_perp': float(perpendicular @ perpendicular), 'output_norm': float(norms[0])}


def compare_preservation(prefix, longer):
    if not np.isclose(prefix['C0'], longer['C0'], atol=1e-12, rtol=0):
        raise ValueError('Comparison requires a common baseline and direction')
    orthogonal = (longer['delta_perp'] - prefix['delta_perp']) / 2
    gap = prefix['C'] - longer['C']
    return {'R_gap': prefix['R'] - longer['R'], 'C_gap': gap,
            'identity_rhs': orthogonal - gap * (longer['C'] + prefix['C'] - 2 * prefix['C0']) / 2,
            'lower_bound': orthogonal - 2 * abs(gap)}


def measured_cosines(output, baseline, direction):
    """Retain preservation when a reference displacement vanishes numerically."""
    if np.linalg.norm(direction) <= 1e-30:
        row = cosine_metrics(output, baseline, baseline)
        row.update(C=None, C0=None, delta_C=None, delta_perp=None,
                   concept_defined=False, concept_reason='reference displacement is numerically zero')
        return row
    return cosine_metrics(output, baseline, direction)
