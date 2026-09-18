import importlib.util
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location('section4', Path(__file__).parents[1] / 'scripts/run_section4_bounds.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_condition_grid_covers_all_lemmas_and_zero_drift_generated_control():
    conditions = module.conditions()
    assert {x['lemma'] for x in conditions} == {2, 4, 6, 7}
    assert len({x['id'] for x in conditions}) == len(conditions)
    for g in [1, 4, 8, 16]:
        assert any(x['g'] == g and x['family'] == 'generated_zero' for x in conditions)
    assert {(x['m'], x['k']) for x in conditions if x['family'] == 'grid'} == {
        (m, k) for m in [1, 2, 4, 8] for k in [1, 2, 4, 8]}


def test_redundancy_suite_uses_real_state_duplication_controls():
    rows = module.conditions('redundancy')
    assert {r['family'] for r in rows} == {'repeat_prompt', 'repeat_input', 'repeat_both'}
    assert all((r['m'], r['k'], r['g']) == (4, 4, 0) for r in rows)


def test_resume_skips_completed_units_and_rejects_changed_manifest(tmp_path):
    path = tmp_path / 'progress.json'
    calls = []
    def compute(unit):
        calls.append(unit)
        if unit == 'b':
            raise RuntimeError('preempted')
        return {'value': 3}
    with pytest.raises(RuntimeError, match='preempted'):
        module.run_units(path, {'version': 1}, ['a', 'b'], compute)
    assert calls == ['a', 'b']
    calls.clear()
    module.run_units(path, {'version': 1}, ['a', 'b'], lambda x: calls.append(x) or {'value': 4})
    assert calls == ['b']
    with pytest.raises(ValueError, match='manifest'):
        module.run_units(path, {'version': 2}, ['a'], compute)


def test_checkpoint_does_not_rewrite_all_previous_rows_per_condition(tmp_path, monkeypatch):
    path = tmp_path / 'progress.json'
    writes = []
    original = module.write_json_atomic
    def write(target, value):
        writes.append(target)
        original(target, value)
    monkeypatch.setattr(module, 'write_json_atomic', write)
    module.run_units(path, {}, list('abcdef'), lambda x: {'value': x})
    assert len(writes) <= 2
    assert len(__import__('json').loads(path.read_text())['units']) == 6


def test_resume_recovers_valid_journal_rows_before_torn_last_write(tmp_path):
    path = tmp_path / 'progress.json'
    path.write_text('{"manifest": {}, "units": {}}')
    path.with_suffix('.jsonl').write_text('{"id":"a","result":{"value":1}}\n{"id":"b"')
    calls = []
    state = module.run_units(path, {}, ['a', 'b'], lambda unit: calls.append(unit) or {'value': 2})
    assert calls == ['b']
    assert state['units'] == {'a': {'value': 1}, 'b': {'value': 2}}


def test_experiment_never_imports_email_notifications():
    import ast
    tree = ast.parse(Path(module.__file__).read_text())
    assert not any(isinstance(node, ast.ImportFrom) and node.module == 'prefix.notify'
                   for node in ast.walk(tree))
