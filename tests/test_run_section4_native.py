import importlib.util
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location('native_run', Path(__file__).parents[1] / 'scripts/run_section4_native.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_experiment_does_not_send_email():
    import ast
    tree = ast.parse(Path(module.__file__).read_text())
    assert not any(isinstance(node, ast.ImportFrom) and node.module == 'prefix.notify'
                   for node in ast.walk(tree))


def test_selection_prioritizes_sufficient_inputs_then_nearest_short_inputs():
    rows = module.choose_examples({0:30, 1:128, 2:200, 3:127, 4:110}, 4)
    assert rows == [1, 2, 3, 4]
    with pytest.raises(ValueError, match='available'):
        module.choose_examples({0:10}, 2)


def test_expansion_keeps_request_and_is_explicitly_recorded():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            self.last_content = messages[0]['content']
            return list(range(len(self.last_content.split())))
    tokenizer = Tokenizer()
    ids, added = module.expanded_input(tokenizer, 'Original request.', minimum=128)
    assert len(ids) >= 128
    assert added and tokenizer.last_content.endswith('Original request.')


def test_native_manifest_changes_when_scientific_settings_change():
    config = dict(model='example', layer=26, heads=[0,16], generated_tokens=256)
    assert module.digest(config) == module.digest(dict(reversed(list(config.items()))))
    assert module.digest(config) != module.digest({**config, 'generated_tokens':128})


def test_partial_batch_reuses_per_example_progress_without_completed_members(tmp_path):
    rows = {1:{'seed_generated_ids':[1]},2:{'seed_generated_ids':[2]},3:{'seed_generated_ids':[3]}}
    path = tmp_path / 'decode.json'
    identity = {'cohort':'test','indices':[1,2,3]}
    path.write_text(__import__('json').dumps({'identity':identity,'tokens':[[1,9],[2,8],[3,7]]}))
    state, pending, tokens = module.resume_batch(path,identity,rows,completed={1})
    assert pending == [2,3]
    assert tokens == [[2,8],[3,7]]
    tokens[0].append(6)
    assert state['tokens'][1] == [2,8,6]


def test_analysis_fingerprint_is_independent_of_extraction_only_changes():
    source = Path(module.__file__).read_text()
    assert module.analysis_fingerprint(source) == module.analysis_fingerprint(source.replace('def extract(', 'def renamed_extract('))
    assert module.analysis_fingerprint(source) != module.analysis_fingerprint(source.replace('base_d = diameter(', 'base_d = 2 * diameter('))


def test_result_validation_rejects_missing_and_misassigned_queries():
    row = {'identity': {'index':1}, 'queries':[3,4], 'errors':[.1,.2], 'complete':True}
    module.validate_result(row, {'index':1}, [3,4])
    with pytest.raises(ValueError):
        module.validate_result(row, {'index':1}, [3,4,5])
    with pytest.raises(ValueError):
        module.validate_result(row, {'index':2}, [3,4])


def test_parallel_workers_partition_examples_and_clean_up_on_failure(monkeypatch):
    from types import SimpleNamespace
    import subprocess
    launched = []
    class Process:
        def __init__(self, command):
            self.command,self.returncode,self.terminated = command,None,False
            launched.append(self)
        def wait(self, timeout=None):
            self.returncode = 17 if self is launched[0] else -15 if self.terminated else 0
            return self.returncode
        def poll(self):
            return self.returncode
        def terminate(self):
            self.terminated = True
        def kill(self):
            self.terminated = True
    monkeypatch.setattr(module.subprocess,'Popen',Process)
    args=SimpleNamespace(workers=4,config=Path('config.yaml'),device='cuda')
    with pytest.raises(subprocess.CalledProcessError):
        module.run_workers(args,[1,2,3,4,5])
    assert len(launched)==4
    assert {p.command[p.command.index('--shard')+1] for p in launched} == {'0','1','2','3'}
    assert all(p.command[p.command.index('--shards')+1]=='4' for p in launched)
    assert all(p.terminated and p.returncode is not None for p in launched[1:])
