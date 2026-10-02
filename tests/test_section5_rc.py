import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('section5', Path(__file__).parents[1] / 'scripts/run_section5_rc.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_prediction_128_excludes_future_token_and_unused_prompt_positions():
    base, prompt, query = module.positions(20, 8)
    assert base == list(range(20)) + list(range(148, 275))
    assert prompt == list(range(20, 28))
    assert query == 274
    assert 275 not in base


def test_conditions_cover_baseline_prompt_full_and_length_strength_sweeps():
    rows = module.conditions()
    assert len({r['id'] for r in rows}) == len(rows)
    assert {r['method'] for r in rows} == {'unsteered', 'prompt', 'single', 'prefix', 'full'}
    assert {r['m'] for r in rows if r['method'] == 'prompt'} == {1, 8, 32, 128}
    assert {r['alpha'] for r in rows if r['method'] == 'full'} == {0, .25, .5, 1, 2}


def test_resume_persists_each_unit_and_rejects_changed_configuration(tmp_path):
    path = tmp_path / 'checkpoint.json'
    calls = []
    def compute(unit):
        calls.append(unit)
        if unit == 'b':
            raise RuntimeError('preemption')
        return {'value': 1}
    with pytest.raises(RuntimeError):
        module.resume_units(path, {'version': 1}, ['a', 'b'], compute)
    result = module.resume_units(path, {'version': 1}, ['a', 'b'], lambda unit: {'value': 2})
    assert result['units'] == {'a': {'value': 1}, 'b': {'value': 2}}
    with pytest.raises(ValueError, match='manifest'):
        module.resume_units(path, {'version': 2}, ['a'], compute)
