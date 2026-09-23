import pytest
import torch

from prefix.fixed_attention_strength import FixedNativeAttention


def fixture():
    from transformers import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, Qwen3RotaryEmbedding
    torch.manual_seed(9)
    cfg = Qwen3Config(hidden_size=16, intermediate_size=32, num_attention_heads=2,
                     num_key_value_heads=1, head_dim=8, num_hidden_layers=1)
    cfg._attn_implementation = 'eager'
    attn = Qwen3Attention(cfg, 0).eval()
    h = torch.randn(1, 10, 16)
    rotary = Qwen3RotaryEmbedding(cfg)
    embeddings = rotary(h, torch.arange(10)[None])
    return attn, h, embeddings


def test_frozen_output_is_the_actual_native_module_head_output():
    attn, h, emb = fixture()
    captured = []
    handle = attn.o_proj.register_forward_pre_hook(lambda m, a: captured.append(a[0].detach()))
    mask = torch.full((1, 1, 10, 10), -torch.inf).triu(1)
    with torch.inference_mode():
        _, weights = attn(h, emb, mask)
    handle.remove()
    study = FixedNativeAttention(attn, h[0], emb, prompt_length=4, head=1)
    output, probabilities = study.output(torch.zeros(16), 0, 0.)
    torch.testing.assert_close(output, captured[0][0, -1, 8:16])
    torch.testing.assert_close(probabilities, weights[0, 1, -1])


def test_query_is_unmodified_and_full_support_excludes_readout():
    attn, h, emb = fixture()
    study = FixedNativeAttention(attn, h[0], emb, prompt_length=4, head=0)
    seen = []
    handle = study.attn.q_proj.register_forward_pre_hook(lambda m,a: seen.append(a[0].clone()))
    study.output(torch.ones(16), 6, .4)
    handle.remove()
    torch.testing.assert_close(seen[0][0, -1], h[0,-1])
    torch.testing.assert_close(seen[0][0,:3], h[0,:3])
    torch.testing.assert_close(seen[0][0,3:9], h[0,3:9]+.4)
    with pytest.raises(ValueError, match='query'):
        study.output(torch.ones(16), 7, .4)
    torch.testing.assert_close(study.hidden, h[0])


def test_single_position_matching_strength_has_zero_error():
    attn, h, emb = fixture()
    study = FixedNativeAttention(attn, h[0], emb, prompt_length=4, head=0)
    direction = torch.randn(16)
    left, _ = study.output(direction, 1, .2)
    right, _ = study.output(direction, 1, .2)
    torch.testing.assert_close(left, right, atol=0., rtol=0.)
