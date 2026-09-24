import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('section5_summary', Path(__file__).parents[1] / 'scripts/plot_section5_rc.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_summary_counts_behaviors_not_head_observations():
    rows = [dict(index=i, id='single_m8_a1', method='single', m=8, alpha=1, k=1,
                 R=r, C=.5, delta_C=.1, delta_perp=.2)
            for i,r in [(0,.2),(0,.4),(1,.8),(1,1.)]]
    group = module.summarize(rows)[0]
    assert group['n'] == 2
    assert group['R']['mean'] == pytest.approx(.6)


def test_summary_preserves_different_condition_cohorts():
    rows = [dict(index=i, id=condition, method='single', m=8, alpha=1, k=1,
                 R=.8, C=.5, delta_C=.1, delta_perp=.2)
            for i,condition in [(0,'a'),(1,'a'),(1,'b')]]
    assert {g['id']:g['n'] for g in module.summarize(rows)} == {'a':2,'b':1}


def test_undefined_concept_is_counted_without_dropping_valid_R():
    rows = [dict(index=i, id='a', method='single', m=8, alpha=1, k=1,
                 R=.8, C=c, delta_C=c, delta_perp=c) for i,c in [(0,.5),(1,None)]]
    g = module.summarize(rows)[0]
    assert g['n'] == 2
    assert g['R']['n'] == 2
    assert g['C']['n'] == 1
    assert g['C']['undefined'] == 1
