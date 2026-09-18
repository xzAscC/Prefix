import importlib.util
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location('section4_plot', Path(__file__).parents[1] / 'scripts/plot_section4_bounds.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_summary_excludes_roundoff_only_ratios_and_detects_real_violations():
    rows = [
        {'lemma': 2, 'index': 0, 'error': 1e-14, 'diameter': 1., 'bound_certified': 0.,
         'bound_decomp': 1e-14, 'bound_log': 1e-14, 'identity_residual': 1e-14,
         'value_error': 1e-14, 'weight_error': 0.},
        {'lemma': 2, 'index': 1, 'error': .4, 'diameter': 1., 'bound_certified': .2,
         'bound_decomp': .4, 'bound_log': .4, 'identity_residual': 0.,
         'value_error': .3, 'weight_error': .1},
    ]
    result = module.summarize(rows)
    assert result['violations'] == 1
    assert result['nonzero_error_rows'] == 1
    assert result['median_error_over_bound'] == 2.


def test_mean_and_std_use_prompts_not_individual_heads():
    rows = [{'index': 0, 'error': 1.}, {'index': 0, 'error': 3.},
            {'index': 1, 'error': 5.}, {'index': 1, 'error': 7.}]
    value = module.mean_std(rows, 'error')
    assert value['n'] == 2
    assert value['mean'] == 4.
    assert value['std'] == pytest.approx(8 ** .5)
