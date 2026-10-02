import json

import numpy as np
import pytest
import torch

from prefix.duration_strength import calibrated_strength, selected_mask, sample_summary, decode, select_cohort


def test_cohort_keeps_behavior_ids_and_contexts():
    rows = [dict(BehaviorID=str(i), Behavior=f'question {i}', ContextString='context',
                 SemanticCategory='test') for i in range(100)]
    result = select_cohort(rows)
    assert len({x['behavior_id'] for x in result}) == 100
    assert all(x['text'].startswith('context\n\nquestion ') for x in result)
    with pytest.raises(ValueError, match='unique'):
        select_cohort(rows + [rows[0]])


def test_schedule_starts_at_last_prompt_and_full_covers_exactly_128_predictions():
    assert selected_mask(range(10), 6, 4) == [False]*5 + [True]*4 + [False]
    assert sum(selected_mask(range(133), 6, 128)) == 128
    assert selected_mask([132, 133], 6, 128) == [True, False]


def test_calibration_uses_attention_mass_not_number_of_tokens():
    assert calibrated_strength([.1, .2, .1, .6], 3, 2, .1) == pytest.approx(.7)
    with pytest.raises(ValueError):
        calibrated_strength([.1, .9, 0., 0.], 3, 2, .1)


def test_summary_uses_sample_std_across_examples():
    result = sample_summary([1., 2., 3.])
    assert result == {'n': 3, 'mean': 2., 'std': 1.}


def tiny_model():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(2)
    cfg = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                     num_hidden_layers=2, num_attention_heads=2,
                     num_key_value_heads=1, head_dim=8, eos_token_id=31)
    cfg._attn_implementation = 'eager'
    return Qwen3ForCausalLM(cfg).eval()


def test_native_capture_matches_unmodified_forward_and_zero_intervention(tmp_path):
    model = tiny_model()
    raw = []
    handle = model.model.layers[1].self_attn.o_proj.register_forward_pre_hook(
        lambda m, args: raw.append(args[0][0, -1, :8].detach().clone()))
    with torch.inference_mode():
        expected = model(torch.tensor([[2, 3, 4]]))
    handle.remove()
    arms = [{'id': 'a', 'length': 4, 'strength': 0.},
            {'id': 'b', 'length': 1, 'strength': 0.}]
    result = decode(model, [2, 3, 4], torch.ones(16), arms, 1, 0,
                    tmp_path/'state.json', {'test': 1}, target=1, capture_attention=True)
    assert result['tokens'][0] == [expected.logits[0, -1].argmax().item()]
    np.testing.assert_allclose(result['outputs'][0], raw[0], atol=1e-6)
    np.testing.assert_allclose(result['outputs'][0], result['outputs'][1], atol=1e-6)
    assert len(result['attention']) == 2


def test_native_batched_arms_match_individual_runs_and_resume(tmp_path):
    model = tiny_model()
    direction = torch.arange(16).float() / 20
    arms = [{'id': 'a', 'length': 3, 'strength': .2},
            {'id': 'b', 'length': 1, 'strength': .8}]
    def interrupt(state):
        if len(state['tokens'][0]) == 2:
            raise RuntimeError('preempted')
    path = tmp_path/'batch.json'
    with pytest.raises(RuntimeError, match='preempted'):
        decode(model, [2, 3, 4], direction, arms, 1, 0, path, {}, target=4, on_step=interrupt)
    saved = json.loads(path.read_text())
    observed = []
    batch = decode(model, [2, 3, 4], direction, arms, 1, 0, path, {}, target=4,
                   on_step=lambda s: observed.append(len(s['tokens'][0])))
    assert observed == [3, 4]
    assert batch['tokens'][0][:2] == saved['tokens'][0]
    for i, arm in enumerate(arms):
        single = decode(model, [2, 3, 4], direction, [arm], 1, 0,
                        tmp_path/f'{i}.json', {}, target=4)
        assert batch['tokens'][i] == single['tokens'][0]
        np.testing.assert_allclose(batch['outputs'][i], single['outputs'][0], atol=1e-6)
    before = path.read_bytes()
    decode(model, [2, 3, 4], direction, arms, 1, 0, path, {}, target=4)
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match='identity'):
        decode(model, [2, 3, 4], direction, arms, 1, 0, path, {'different': True}, target=4)
