"""Lemma 8 at common query and fixed pre-addition key/value pairs."""
import numpy as np
from prefix.attention_bounds import diameter, softmax


def evaluate_schedules(keys, values, query, short, long, key_shift, value_shift,
                       short_diameter=None):
    short, long = list(short), list(long)
    if len(set(short)) != len(short) or len(set(long)) != len(long) or not set(short) <= set(long):
        raise ValueError('Schedules must have unique nested supports')
    extra = sorted(set(long)-set(short))
    complement = sorted(set(range(len(keys)))-set(extra))
    scores = np.asarray(keys @ query / np.sqrt(len(query)),dtype=np.float64)
    vs = np.asarray(values,dtype=np.float64).copy()
    shift = float(query @ key_shift / np.sqrt(len(query)))
    scores[short] += shift
    vs[short] += value_shift
    ps = softmax(scores)
    longer_scores = scores.copy()
    longer_scores[extra] += shift
    vl = vs.copy()
    vl[extra] += value_shift
    pl = softmax(longer_scores)
    os,ol = ps @ vs, pl @ vl
    ws,wl = float(ps[extra].sum()),float(pl[extra].sum())
    d = diameter(vs) if short_diameter is None else short_diameter
    dv = float(np.linalg.norm(value_shift))
    value_term = wl * value_shift
    weight_term = np.zeros_like(value_shift)
    if extra and complement:
        vt = softmax(scores[extra]) @ vs[extra]
        vc = softmax(scores[complement]) @ vs[complement]
        weight_term = (wl-ws)*(vt-vc)
    return dict(error=float(np.linalg.norm(ol-os)),
                bound_shares=wl*dv+d*abs(wl-ws),
                bound_score=wl*dv+d*abs(shift)/4,
                bound_epsilon=max(ws,wl)*(dv+d),
                epsilon_observed=max(ws,wl), w_short=ws,w_long=wl,
                diameter_short=float(d), value_shift_norm=dv,score_shift=shift,
                value_term_norm=float(np.linalg.norm(value_term)),
                weight_term_norm=float(np.linalg.norm(weight_term)),
                identity_residual=float(np.linalg.norm(ol-os-value_term-weight_term)),
                extra_count=len(extra))
