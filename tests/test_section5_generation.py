import importlib.util
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))

import numpy as np

SPEC = importlib.util.spec_from_file_location('section5_generation', Path(__file__).parents[1] / 'scripts/run_section5_generation.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_steering_mask_uses_absolute_positions_across_cache_resume():
    assert module.selected_positions('single', 1, 5, [0, 1, 2, 3, 4]) == [4]
    assert module.selected_positions('single', 1, 5, [5, 6]) == []
    assert module.selected_positions('prefix', 4, 5, [0, 1, 2, 3, 4, 5]) == [1, 2, 3, 4]
    assert module.selected_positions('full', -1, 5, [0, 1, 2, 3, 4, 5]) == list(range(6))
    assert module.selected_positions('prompt', 0, 5, [0, 1, 2, 3, 4, 5]) == []


def test_resuming_decoding_never_generates_completed_tokens_again(tmp_path):
    path = tmp_path / 'decode.json'
    calls = []
    def predict(tokens):
        calls.append(list(tokens))
        return len(tokens) + 10, np.array([len(tokens), 1.])
    first = module.decode_resume(path, {'same': True}, predict, target=3)
    assert first['tokens'] == [10, 11, 12]
    assert first['final_output'] == [2., 1.]
    calls.clear()
    second = module.decode_resume(path, {'same': True}, predict, target=3)
    assert calls == []
    assert first == second


def test_preemption_saves_last_successful_token(tmp_path):
    import pytest
    path = tmp_path / 'decode.json'
    def crash(tokens):
        if len(tokens) == 2:
            raise RuntimeError('preempted')
        return 7, np.ones(2)
    with pytest.raises(RuntimeError):
        module.decode_resume(path, {}, crash, target=4)
    calls = []
    def resume(tokens):
        calls.append(len(tokens))
        return 8, np.ones(2)
    state = module.decode_resume(path, {}, resume, target=4)
    assert calls == [2, 3]
    assert state['tokens'] == [7, 7, 8, 8]


def test_export_input_ids_is_resumable_and_excludes_prompt_and_continuation(tmp_path):
    path = tmp_path / 'inputs.json'
    calls = []
    def load(index):
        calls.append(index)
        return {'behavior_id': str(index), 'tokens': [1,2,3,4], 'base_length': 2}
    module.export_inputs(path, load, 2)
    module.export_inputs(path, load, 2)
    assert calls == [0,1]
    import json
    assert json.loads(path.read_text())[1]['input_ids'] == [1,2]
