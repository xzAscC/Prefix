import importlib.util
from pathlib import Path

import pytest
from prefix.runner import run_units


SPEC = importlib.util.spec_from_file_location('section4_long', Path(__file__).parents[1] / 'scripts/run_section4_long.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_sweeps_include_exact_long_counts_without_duplicate_conditions():
    conditions = module.conditions()
    assert len({(r['m'], r['k'], r['g']) for r in conditions}) == len(conditions)
    assert {r['k'] for r in conditions if r['m'] == 4 and r['g'] == 0} == set(module.LENGTHS)
    assert {r['m'] for r in conditions if r['k'] == 1 and r['g'] == 0} == set(module.LENGTHS)
    assert {r['k'] + r['g'] for r in conditions if r['g'] > 0} == {8,16,32,64,128}


def test_short_inputs_are_not_silently_clamped_or_negative_indexed():
    assert module.position_sets(30, 4, 32, 0) is None
    selected, prompt, shared = module.position_sets(128, 4, 128, 0)
    assert len(selected) == 128
    assert len(prompt) == 4
    assert not set(selected) & set(shared)
    selected, prompt, shared = module.position_sets(30, 4, 4, 124)
    assert len(selected) == 128
    assert len(shared) == 30


def test_plot_cohort_is_fixed_at_longest_input_length():
    assert module.input_cohort({0:30, 1:128, 2:400, 3:64}) == [1,2]


def test_notification_import_is_absent():
    import ast
    tree = ast.parse(Path(module.__file__).read_text())
    assert not any(isinstance(node, ast.ImportFrom) and node.module == 'prefix.notify'
                   for node in ast.walk(tree))


def test_layer_extension_preserves_existing_work_only_for_compatible_manifests():
    before = {'layers':[8,17,26], 'revision':'same', 'generated_tokens':128}
    after = {**before, 'layers':[2,8,17,26]}
    assert module.compatible_extension(before, after)
    assert not module.compatible_extension(before, {**after, 'generated_tokens':64})
    assert not module.compatible_extension(after, before)


def test_long_sweep_uses_shared_checkpoint_runner():
    assert module.run_units is run_units
    assert not hasattr(module, 'sibling')
