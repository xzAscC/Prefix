"""Native Qwen3 attention, causal evaluation, and resumable fitting checks."""
import json

import pytest
import torch

from prefix.native_attention import (
    NativeHead, conditions, fit_shift, query_sets, subspace, perturbations,
)


def fixture_head(n=6, prompts=4, generated=8, dim=4):
    rng = torch.Generator().manual_seed(71)
    h = torch.randn(n + prompts + generated, 3 * dim, generator=rng, dtype=torch.float64)
    w = {name: torch.randn(dim, 3 * dim, generator=rng, dtype=torch.float64) / 3
         for name in ('wq', 'wk', 'wv')}
    w.update(qnorm=torch.linspace(.8, 1.2, dim, dtype=torch.float64),
             knorm=torch.linspace(1.1, .9, dim, dtype=torch.float64), epsilon=1e-6)
    return NativeHead(h, w, n, prompts, generated, theta=10000.)


def test_sweep_has_one_input_token_in_every_mixed_condition():
    rows = conditions()
    assert len({r['id'] for r in rows}) == len(rows)
    assert {r['b'] + r['g'] for r in rows if r['family'] == 'mixed'} == {2,4,8,16,32,64,128}
    assert all(r['b'] == 1 for r in rows if r['family'] == 'mixed')
    assert {r['m'] for r in rows if r['b'] == 1 and r['g'] == 0} == {1,2,4,8,16,32,64,128}


def test_queries_exclude_prompt_steered_reference_and_keep_common_cohort():
    q = query_sets(n=140, prompt_slots=128, generated=256, b=4, g=3)
    assert not set(q['all']) & set(range(136, 268))
    assert not set(q['all']) & {268,269,270,523}
    assert q['common'] == list(range(395,523))
    assert set(q['common']) <= set(q['all'])
    assert len(q['pre_prompt']) == 136
    assert q['reference'] == 523


def test_native_head_matches_transformers_with_real_masks_and_gqa():
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, Qwen3RotaryEmbedding
    torch.manual_seed(19)
    config = Qwen3Config(hidden_size=12, num_attention_heads=2, num_key_value_heads=1,
                        head_dim=4, rope_theta=10000., attention_bias=False)
    config._attn_implementation = 'eager'
    attn = Qwen3Attention(config, layer_idx=0).eval()
    rotary = Qwen3RotaryEmbedding(config)
    h = torch.randn(1, 12, 12)
    # Capture the actual concatenated head output before o_proj.
    captured = []
    handle = attn.o_proj.register_forward_pre_hook(lambda _, args: captured.append(args[0].detach()))
    w = {'wq':attn.q_proj.weight[4:8], 'wk':attn.k_proj.weight, 'wv':attn.v_proj.weight,
         'qnorm':attn.q_norm.weight, 'knorm':attn.k_norm.weight, 'epsilon':config.rms_norm_eps}
    study = NativeHead(h[0].double(), {k:v.detach().double() if torch.is_tensor(v) else v
                                     for k,v in w.items()}, 4, 2, 6, theta=10000.)
    r = torch.randn(12, dtype=torch.float64) * .15
    for arm in ('prompt', 'steer'):
        indices, selected = study.context(2, 2, 1, arm)
        hidden = h[:, indices].clone()
        if arm == 'steer':
            hidden[:, selected] += r.float()
        positions = torch.arange(len(indices))[None]
        mask = torch.full((len(indices),len(indices)), float('-inf')).triu(1)[None,None]
        with torch.no_grad():
            attn(hidden, rotary(hidden, positions), mask)
        expected = captured[-1][0,:,4:8]
        actual = study.outputs(2, 2, 1, indices, r if arm == 'steer' else None, arm)
        torch.testing.assert_close(actual.float(), expected, atol=2e-6, rtol=2e-5)
    handle.remove()


def test_future_states_cannot_change_earlier_query_and_prompt_is_causally_invisible():
    study = fixture_head()
    zero = torch.zeros(study.h.shape[1], dtype=torch.float64)
    before = study.outputs(2, 1, 1, [0,1,2], zero, 'steer')
    prompt = study.outputs(2, 1, 1, [0,1,2], None, 'prompt')
    torch.testing.assert_close(before, prompt, atol=1e-12, rtol=1e-12)
    study.h[-1] += 100
    other = NativeHead(study.h, study.w, 6, 4, 8, theta=10000.)
    torch.testing.assert_close(before, other.outputs(2, 1, 1, [0,1,2], zero, 'steer'))


def test_optimizer_fits_native_output_and_resumes_without_repeating(tmp_path):
    study = fixture_head()
    path = tmp_path / 'fit.json'
    result = fit_shift(study, 2, 1, 1, path, {'version':1}, max_steps=80, tolerance=1e-7)
    assert result['final_error'] < 1e-6
    assert result['final_error'] < result['initial_error'] / 1000
    saved = path.read_bytes()
    repeated = fit_shift(study, 2, 1, 1, path, {'version':1}, max_steps=80, tolerance=1e-7)
    assert repeated == result
    assert path.read_bytes() == saved
    with pytest.raises(ValueError, match='manifest'):
        fit_shift(study, 2, 1, 1, path, {'version':2}, max_steps=80)


def test_interrupted_fit_restarts_from_checkpoint(tmp_path):
    study = fixture_head()
    path = tmp_path / 'fit.json'
    def stop(state):
        if state['steps'] >= 1:
            raise RuntimeError('preempted')
    with pytest.raises(RuntimeError, match='preempted'):
        fit_shift(study, 2, 1, 1, path, {}, max_steps=80, tolerance=1e-7, on_checkpoint=stop)
    state = json.loads(path.read_text())
    assert state['steps'] > 0 and not state['complete']
    result = fit_shift(study, 2, 1, 1, path, {}, max_steps=80, tolerance=1e-7)
    assert result['final_error'] < 1e-6


def test_subspace_rank_and_equal_norm_query_controls():
    directions = torch.tensor([[1.,0.,0.,0.],[2.,0.,0.,0.]], dtype=torch.float64)
    basis, rank = subspace(directions)
    assert rank == 1
    changes = perturbations(basis, rank, norm=.4, seed=12)
    for value in changes.values():
        assert value.norm().item() == pytest.approx(.4)
    assert (basis[:rank] @ changes['null']).norm().item() < 1e-12
    assert (basis[:rank] @ changes['sensitive']).norm().item() == pytest.approx(.4)
    assert subspace(torch.cat([directions, torch.tensor([[0.,1.,0.,0.]])]))[1] == 2
