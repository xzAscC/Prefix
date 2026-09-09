# pyright: reportMissingImports=false
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from prefix import vllm_steering
from prefix.vllm_steering import (
    CaptureSink,
    SteeringSchedule,
    attach_capture,
    attach_steering,
    decode_rows,
    make_decode_index_resolver,
    make_llm,
    prefill_last_rows,
)


def metadata(lengths, seq_lens):
    qsl = [0]
    for length in lengths:
        qsl.append(qsl[-1] + length)
    return SimpleNamespace(
        query_start_loc=torch.tensor(qsl),
        seq_lens=torch.tensor(seq_lens),
    )


class FakeLayer:
    def __init__(self, tuple_output=False):
        self.tuple_output = tuple_output

        def forward(hidden):
            if self.tuple_output:
                return hidden.clone(), hidden.clone()
            return hidden.clone()

        self.forward = forward


def tensor_output(value):
    return value[0] if isinstance(value, tuple) else value


class FakeInputBatch:
    def __init__(self, req_ids, prompt_lens):
        self.req_ids = req_ids
        self.prefill_len_np = np.asarray(prompt_lens, dtype=np.int64)


def fake_llm(req_ids=("a", "b"), prompt_lens=(3, 1)):
    layers = [FakeLayer()]
    holder = {"batch": FakeInputBatch(list(req_ids), list(prompt_lens))}
    runner = SimpleNamespace(
        prepare_inputs=lambda scheduler_output=None, h=holder: h["batch"],
        model=SimpleNamespace(model=SimpleNamespace(layers=layers)),
    )
    worker = SimpleNamespace(model_runner=runner)
    llm = SimpleNamespace(
        llm_engine=SimpleNamespace(
            engine_core=SimpleNamespace(
                engine_core=SimpleNamespace(
                    model_executor=SimpleNamespace(
                        driver_worker=SimpleNamespace(worker=worker)
                    )
                )
            )
        )
    )
    llm._batch_holder = holder
    vllm_steering._LATEST_INPUT_BATCH[id(llm)] = holder["batch"]
    return llm


def set_fake_batch(llm, batch):
    llm._batch_holder["batch"] = batch
    vllm_steering._LATEST_INPUT_BATCH[id(llm)] = batch


@pytest.fixture(autouse=True)
def _clear_input_batch_stash():
    vllm_steering._LATEST_INPUT_BATCH.clear()
    vllm_steering._PREPARE_HOOK_KEYS.clear()
    yield
    vllm_steering._LATEST_INPUT_BATCH.clear()
    vllm_steering._PREPARE_HOOK_KEYS.clear()


def test_row_detection_uses_production_metadata_without_computed_tokens():
    value = metadata([3, 1, 1, 2], [3, 4, 1, 6])
    assert prefill_last_rows(value) == [2, 4, 6]
    assert decode_rows(value) == [3]
    assert not hasattr(value, "num_computed_tokens")


def test_single_token_prompt_is_prefill_and_no_context_is_decode():
    value = metadata([1, 1], [1, 2])
    assert prefill_last_rows(value) == [0]
    assert decode_rows(value) == [1]


def test_resolver_is_lazy_validates_ids_and_returns_k_in_decode_order():
    llm = fake_llm(("prompt", "continuation"), (3, 1))
    resolver = make_decode_index_resolver(llm)
    value = metadata([1, 1], [4, 4])
    assert resolver(value) == [1, 3]
    set_fake_batch(llm, FakeInputBatch(["only"], [3]))
    with pytest.raises(RuntimeError, match="req_ids"):
        resolver(value)


def test_missing_batch_sources_fail_closed():
    llm = fake_llm(("prompt",), (5,))
    runner = llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
    vllm_steering._LATEST_INPUT_BATCH.pop(id(llm))
    resolver = make_decode_index_resolver(llm)
    with pytest.raises(RuntimeError, match="input batch"):
        resolver(metadata([1], [4]))
    with pytest.raises(RuntimeError, match="input batch"):
        vllm_steering._request_ids(llm, 1)
    bare = fake_llm(("prompt",), (5,))
    bare_runner = bare.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
    del bare_runner.prepare_inputs
    with pytest.raises(RuntimeError, match="prepare_inputs"):
        make_decode_index_resolver(bare)


def test_prepare_inputs_wrapper_populates_stash():
    llm = fake_llm(("prompt", "continuation"), (3, 1))
    resolver = make_decode_index_resolver(llm)
    runner = llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
    vllm_steering._LATEST_INPUT_BATCH.pop(id(llm))
    assert runner.prepare_inputs(None) is llm._batch_holder["batch"]
    assert resolver(metadata([1, 1], [4, 4])) == [1, 3]


def test_resolver_rejects_invalid_decode_index():
    llm = fake_llm(("prompt",), (5,))
    resolver = make_decode_index_resolver(llm)
    with pytest.raises(RuntimeError, match="decode index"):
        resolver(metadata([1], [4]))


def test_full_and_one_token_schedules_use_production_rows(monkeypatch):
    value = metadata([2, 1, 1], [2, 3, 4])
    monkeypatch.setattr("prefix.vllm_steering._current_attn_metadata", lambda: value)
    direction = torch.ones(2)

    layer = FakeLayer()
    attach_steering(layer, 0, direction, 1, 1, SteeringSchedule.one_token())
    assert torch.equal(
        tensor_output(layer.forward(torch.zeros(4, 2)))[2], torch.zeros(2)
    )

    layer = FakeLayer()
    attach_steering(layer, 0, direction, 1, 1, SteeringSchedule.full())
    result = tensor_output(layer.forward(torch.zeros(4, 2)))
    assert torch.equal(result[1], direction)
    assert torch.equal(result[2], direction)
    assert torch.equal(result[3], direction)


def test_prefix_m5_intervenes_only_k_1_through_4(monkeypatch):
    value = metadata([2, 1, 1, 1, 1, 1], [2, 3, 4, 5, 6, 7])
    monkeypatch.setattr("prefix.vllm_steering._current_attn_metadata", lambda: value)
    layer = FakeLayer()
    attach_steering(
        layer,
        0,
        torch.ones(2),
        1,
        1,
        SteeringSchedule.prefix(5),
        decode_index_resolver=lambda _: [1, 2, 3, 4, 5],
    )
    result = tensor_output(layer.forward(torch.zeros(7, 2)))
    assert torch.equal(result[1], torch.ones(2))
    assert torch.equal(result[2], torch.ones(2))
    assert torch.equal(result[3], torch.ones(2))
    assert torch.equal(result[4], torch.ones(2))
    assert torch.equal(result[5], torch.ones(2))
    assert torch.equal(result[6], torch.zeros(2))


def test_tuple_steering_edits_mlp_once_and_capture_uses_true_residual(monkeypatch):
    value = metadata([2, 1], [2, 3])
    monkeypatch.setattr("prefix.vllm_steering._current_attn_metadata", lambda: value)
    layer = FakeLayer(tuple_output=True)
    vllm_steering._LATEST_INPUT_BATCH[id(layer)] = FakeInputBatch(("p", "d"), (2, 1))
    attach_steering(layer, 0, torch.ones(2), 1, 1, SteeringSchedule.full())
    sink = CaptureSink()
    attach_capture(layer, 0, sink, decode_index_resolver=lambda _: [2])
    out0, residual = layer.forward(torch.zeros(3, 2))
    assert torch.equal(out0[1], torch.ones(2))
    assert torch.equal(residual[1], torch.zeros(2))
    assert torch.equal(out0[1] + residual[1], torch.ones(2))
    assert torch.equal(sink.rows[0]["hidden"], torch.ones(2))


def test_capture_records_request_ids_and_scalar_dots(monkeypatch):
    value = metadata([2, 1], [2, 4])
    monkeypatch.setattr("prefix.vllm_steering._current_attn_metadata", lambda: value)
    sink = CaptureSink()
    llm = fake_llm(("p", "d"), (2, 1))
    attach_capture(
        llm,
        0,
        sink,
        decode_index_resolver=lambda _: [3],
        scalar_directions=[torch.tensor([1.0, 0.0])],
    )
    llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner.model.model.layers[
        0
    ].forward(torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    assert [(row["phase"], row["request_id"]) for row in sink.rows] == [
        ("prefill", "p"),
        ("decode", "d"),
    ]
    assert sink.rows[0]["dots"] == [3.0]
    assert sink.rows[1]["k"] == 3
    assert sink.rows[1]["norm"] == pytest.approx(
        torch.linalg.vector_norm(torch.tensor([5.0, 6.0])).item()
    )


def test_capture_can_attach_to_two_layers(monkeypatch):
    value = metadata([2], [2])
    monkeypatch.setattr("prefix.vllm_steering._current_attn_metadata", lambda: value)
    first, second = FakeLayer(), FakeLayer()
    first_sink, second_sink = CaptureSink(), CaptureSink()
    vllm_steering._LATEST_INPUT_BATCH[id(first)] = FakeInputBatch(("p",), (2,))
    vllm_steering._LATEST_INPUT_BATCH[id(second)] = FakeInputBatch(("q",), (2,))
    attach_capture(first, 0, first_sink)
    attach_capture(second, 0, second_sink)
    first.forward(torch.ones(2, 2))
    second.forward(torch.ones(2, 2))
    assert len(first_sink.rows) == len(second_sink.rows) == 1


def test_steering_rejects_resolver_length_mismatch(monkeypatch):
    value = metadata([2, 1], [2, 3])
    monkeypatch.setattr("prefix.vllm_steering._current_attn_metadata", lambda: value)
    layer = FakeLayer()
    attach_steering(
        layer,
        0,
        torch.ones(2),
        1,
        1,
        SteeringSchedule.prefix(2),
        decode_index_resolver=lambda _: [],
    )
    with pytest.raises(RuntimeError, match="length"):
        layer.forward(torch.zeros(3, 2))


def test_make_llm_forces_safety_options(monkeypatch):
    class FakeLLM:
        seen = None

        def __init__(self, **kwargs):
            FakeLLM.seen = kwargs

    monkeypatch.setattr("vllm.LLM", FakeLLM)
    make_llm("model", enforce_eager=False, enable_chunked_prefill=True, foo=1)
    assert FakeLLM.seen == {
        "model": "model",
        "enforce_eager": True,
        "enable_chunked_prefill": False,
        "enable_prefix_caching": False,
        "foo": 1,
    }
