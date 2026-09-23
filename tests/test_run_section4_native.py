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


def test_result_validation_rejects_missing_and_misassigned_queries():
    row = {'identity': {'index':1}, 'queries':[3,4], 'errors':[.1,.2], 'complete':True}
    module.validate_result(row, {'index':1}, [3,4])
    with pytest.raises(ValueError):
        module.validate_result(row, {'index':1}, [3,4,5])
    with pytest.raises(ValueError):
        module.validate_result(row, {'index':2}, [3,4])
