import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('section5_validation', Path(__file__).parents[1] / 'scripts/validate_section5_rc.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def row(**changes):
    return dict(eligible=True, method='single', alpha=0, R=1., C=.3, C0=.3,
                delta_C=0., delta_perp=0., output_norm=2., **changes)


def test_validator_checks_metric_identity_and_zero_strength():
    module.validate_row(row())
    altered = row(); altered['R'] = .5
    with pytest.raises(ValueError):
        module.validate_row(altered)
    altered = row(); altered['delta_C'] = .1
    with pytest.raises(ValueError):
        module.validate_row(altered)


def test_validator_accepts_explicit_undefined_C_but_never_nan_R():
    value = row(); value.update(C=None, C0=None, delta_C=None, delta_perp=None, concept_defined=False)
    module.validate_row(value)
    value['R'] = float('nan')
    with pytest.raises(ValueError):
        module.validate_row(value)
