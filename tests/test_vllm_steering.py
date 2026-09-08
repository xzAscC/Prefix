# pyright: reportMissingImports=false
import pytest
import torch

from prefix.vllm_steering import (
    CaptureSink,
    SteeringSchedule,
    attach_capture,
    attach_steering,
    decode_rows,
    prefill_last_rows,
)


class FakeMetadata:
    def __init__(self, lengths, computed):
        self.query_start_loc = torch.tensor(
            [0, *torch.cumsum(torch.tensor(lengths), 0).tolist()]
        )
        self.num_computed_tokens = computed


class FakeLayer:
    def __init__(self, dtype=torch.float32):
        self.dtype = dtype

        def forward(hidden):
            return hidden.to(self.dtype)

        self.forward = forward


def test_prefill_rows_skip_chunked_and_single_token_prefills():
    metadata = FakeMetadata([3, 2, 1, 4], torch.tensor([0, 1, 0, 0]))
    assert prefill_last_rows(metadata) == [2, 9]
    assert prefill_last_rows(FakeMetadata([2], None)) == [1]
    assert prefill_last_rows(FakeMetadata([1], torch.tensor([0]))) is None


def test_decode_rows_only_include_contextual_single_tokens():
    metadata = FakeMetadata([1, 1, 2, 1], torch.tensor([3, 0, 1, 2]))
    assert decode_rows(metadata) == [0, 4]
    assert decode_rows(FakeMetadata([2], torch.tensor([0]))) is None


def test_steering_schedules_sign_dtype_and_detach(monkeypatch):
    metadata = FakeMetadata([2, 1, 1], torch.tensor([0, 2, 0]))
    monkeypatch.setattr("prefix.vllm_steering._current_attn_metadata", lambda: metadata)
    direction = torch.tensor([1.0, -2.0])
    layer = FakeLayer(torch.bfloat16)
    original = layer.forward
    detach = attach_steering(
        layer, 0, direction, 0.5, 2.0, SteeringSchedule("full"), sign=-1
    )
    hidden = torch.zeros(4, 2, dtype=torch.bfloat16)
    result = layer.forward(hidden)
    assert result.dtype == torch.bfloat16
    assert torch.allclose(result[1].float(), torch.tensor([-1.0, 2.0]))
    assert torch.allclose(result[2].float(), torch.tensor([-1.0, 2.0]))
    assert torch.allclose(result[3], hidden[3])
    detach()
    assert layer.forward is original


def test_one_token_skips_decode_and_prefix_uses_resolver(monkeypatch):
    metadata = FakeMetadata([2, 1, 1], torch.tensor([0, 2, 3]))
    monkeypatch.setattr("prefix.vllm_steering._current_attn_metadata", lambda: metadata)
    direction = torch.ones(2)
    layer = FakeLayer()
    attach_steering(layer, 0, direction, 1, 1, SteeringSchedule("one_token"))
    assert torch.equal(layer.forward(torch.zeros(4, 2))[2], torch.zeros(2))

    layer = FakeLayer()
    attach_steering(
        layer,
        0,
        direction,
        1,
        1,
        SteeringSchedule.prefix(2),
        decode_index_resolver=lambda _metadata: [1, 3],
    )
    result = layer.forward(torch.zeros(4, 2))
    assert torch.equal(result[2], direction)
    assert torch.equal(result[3], torch.zeros(2))

    with pytest.raises(RuntimeError, match="resolver"):
        attach_steering(layer, 0, direction, 1, 1, SteeringSchedule.prefix(5))


def test_capture_sees_steered_values_and_records_k(monkeypatch):
    metadata = FakeMetadata([2, 1], torch.tensor([0, 2]))
    monkeypatch.setattr("prefix.vllm_steering._current_attn_metadata", lambda: metadata)
    layer = FakeLayer()
    attach_steering(
        layer,
        0,
        torch.ones(2),
        1,
        1,
        SteeringSchedule("full"),
        decode_index_resolver=lambda _metadata: [7],
    )
    sink = CaptureSink()
    detach = attach_capture(layer, 0, sink, decode_index_resolver=lambda _metadata: [7])
    layer.forward(torch.zeros(3, 2))
    assert sink.rows[0]["phase"] == "prefill"
    assert sink.rows[0]["row"] == 1
    assert sink.rows[0]["k"] is None
    assert sink.rows[0]["slot"] == 0
    assert torch.equal(sink.rows[0]["hidden"], torch.ones(2))
    assert sink.rows[1]["phase"] == "decode"
    assert sink.rows[1]["k"] == 7
    assert sink.rows[1]["slot"] == 1
    assert sink.rows[1]["hidden"].dtype == torch.float32
    detach()
