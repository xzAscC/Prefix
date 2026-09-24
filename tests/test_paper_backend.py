import torch
import pytest
from transformers import Qwen3Config, Qwen3ForCausalLM, Olmo3Config, Olmo3ForCausalLM

from prefix.paper_backend import Intervention, PaperBackend


def tiny(family="qwen"):
    config_type, model_type = (Qwen3Config, Qwen3ForCausalLM) if family == "qwen" else (Olmo3Config, Olmo3ForCausalLM)
    torch.manual_seed(13)
    cfg = config_type(vocab_size=32, hidden_size=16, intermediate_size=32,
                      num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                      head_dim=8, eos_token_id=None, attention_dropout=0.)
    return model_type(cfg).eval()


@pytest.mark.parametrize("family", ["qwen", "olmo"])
def test_cached_prefix_matches_full_replay_with_same_position_edits(family):
    model = tiny(family)
    backend = PaperBackend(model, None)
    ids = torch.tensor([[1, 4, 3]])
    direction = torch.arange(16, dtype=torch.float32)
    direction /= direction.norm()
    spec = Intervention(layer=0, direction=direction, coefficient=4., policy='prefix', length=2)
    actual = backend.generate_tokens(ids, 5, spec, eos_ids=[])
    sequence = ids.clone()
    for _ in range(5):
        def edit(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden = hidden.clone()
            end = min(sequence.shape[1], ids.shape[1] + 1)
            hidden[:, ids.shape[1]-1:end] += 4. * direction
            return (hidden, *output[1:]) if isinstance(output, tuple) else hidden
        handle = model.model.layers[0].register_forward_hook(edit)
        try:
            with torch.no_grad():
                token = model(sequence, use_cache=False).logits[:, -1].argmax(-1, keepdim=True)
        finally:
            handle.remove()
        sequence = torch.cat([sequence, token], -1)
    assert actual['token_ids'] == sequence[0, ids.shape[1]:].tolist()
    assert actual['coefficients'] == [4., 4., 0., 0., 0.]
    assert not model.model.layers[0]._forward_hooks


@pytest.mark.parametrize("family", ["qwen", "olmo"])
def test_das_probes_do_not_mutate_live_cache_and_act_uses_probe(family):
    model = tiny(family)
    backend = PaperBackend(model, None)
    ids = torch.tensor([[1, 4, 3]])
    direction = torch.ones(16) / 4
    baseline = backend.generate_tokens(ids, 4, eos_ids=[])
    zero = Intervention(layer=0, direction=torch.zeros(16), policy='das')
    result = backend.generate_tokens(ids, 4, zero, eos_ids=[])
    assert result['token_ids'] == baseline['token_ids']
    assert result['coefficients'] == [0.] * 4
    act = Intervention(layer=0, direction=direction, policy='act',
                       probe_weight=torch.zeros(16), probe_bias=0.)
    result = backend.generate_tokens(ids, 3, act, eos_ids=[])
    assert result['coefficients'] == [6.] * 3


@pytest.mark.parametrize('family', ['qwen', 'olmo'])
def test_das_three_forwards_see_same_inherited_cache(family):
    model = tiny(family)
    seen = []
    def inspect(module, args, kwargs):
        cache = kwargs.get('past_key_values')
        seen.append(0 if cache is None else cache.get_seq_length())
    handle = model.register_forward_pre_hook(inspect, with_kwargs=True)
    try:
        spec = Intervention(layer=0, direction=torch.ones(16) / 4, policy='das')
        PaperBackend(model, None).generate_tokens(torch.tensor([[1, 4, 3]]), 3, spec, eos_ids=[])
    finally:
        handle.remove()
    assert seen == [0, 0, 0, 3, 3, 3, 4, 4, 4]
