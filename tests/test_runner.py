from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from prefix import runner
from prefix.steering import SteeringSchedule


def test_config_and_tee_stdout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("answer: 3\n", encoding="utf-8")
    assert runner.load_config(config_path) == {"answer": 3}
    with pytest.raises(ValueError, match="extension"):
        runner.load_config(tmp_path / "config.toml")

    log_path = tmp_path / "run.log"
    with runner.tee_stdout(log_path):
        print("hello")
    assert capsys.readouterr().out == "hello\n"
    assert log_path.read_text(encoding="utf-8") == "hello\n"


def test_jsonl_resume_and_atomic_json(tmp_path: Path) -> None:
    jsonl = tmp_path / "nested" / "results.jsonl"
    runner.append_jsonl(jsonl, [{"id": "a"}, {"id": "b"}])
    runner.append_jsonl(jsonl, [{"id": "c"}])
    with jsonl.open("a", encoding="utf-8") as output:
        output.write('{"id": "incomplete"')
    assert runner.read_jsonl(jsonl) == [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    assert runner.completed_ids(jsonl) == {"a", "b", "c"}
    assert runner.completed_ids(tmp_path / "missing.jsonl") == set()
    atomic = tmp_path / "state.json"
    runner.write_json_atomic(atomic, {"ok": True})
    assert runner.read_json(atomic) == {"ok": True}
    assert runner.read_json(tmp_path / "missing.json", default=[]) == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The answer is C", "C"),
        ("The answer is (d)", "D"),
        ("answer is A\nMore work\nThe answer is (J)\n  ", "J"),
        ("\n (f) \n", "F"),
        ("The answer is boxed", None),
        ("\nA\nB\n", "B"),
    ],
)
def test_parse_answer_letter(text: str, expected: str | None) -> None:
    assert runner.parse_answer_letter(text) == expected


def test_mmlu_prompt_is_deterministic() -> None:
    prompt = runner.mmlu_prompt("What?", ["one", "two"])
    assert prompt == (
        "What?\n\nA. one\nB. two\n\n"
        "Reason step by step, and make the FINAL line exactly `The answer is (X)` "
        "where X is the letter."
    )


def test_chat_prompt_real_tokenizer() -> None:
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    enabled = runner.chat_prompt(tokenizer, "Say hello", True)
    disabled = runner.chat_prompt(tokenizer, "Say hello", False)
    assert enabled != disabled
    assert "Say hello" in enabled and "Say hello" in disabled
    assert "<|im_start|>assistant" in enabled
    assert "<|im_start|>assistant" in disabled


def test_get_engine_is_lazy_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    runner._engine_singleton = None
    made: list[tuple[str, dict[str, object]]] = []

    def factory(model_id: str, **kwargs: object) -> object:
        made.append((model_id, kwargs))
        return object()

    monkeypatch.setattr(runner.vllm_steering, "make_llm", factory)
    first = runner.get_engine("model", tensor_parallel_size=1)
    second = runner.get_engine("model", tensor_parallel_size=1)
    assert first is second
    assert made == [("model", {"tensor_parallel_size": 1})]
    runner._engine_singleton = None


def test_get_engine_rejects_conflicting_model(monkeypatch: pytest.MonkeyPatch) -> None:
    runner._engine_singleton = None
    monkeypatch.setattr(
        runner.vllm_steering, "make_llm", lambda *args, **kwargs: object()
    )
    runner.get_engine("model")
    with pytest.raises(RuntimeError, match="model_id"):
        runner.get_engine("other")
    runner._engine_singleton = None


def test_get_engine_rejects_conflicting_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    runner._engine_singleton = None
    monkeypatch.setattr(
        runner.vllm_steering, "make_llm", lambda *args, **kwargs: object()
    )
    runner.get_engine("model", tensor_parallel_size=1)
    with pytest.raises(RuntimeError, match="llm_kwargs"):
        runner.get_engine("model", tensor_parallel_size=2)
    runner._engine_singleton = None


def test_capture_prompt_hiddens_aligns_rows_by_request_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sinks: dict[int, runner.CaptureSink] = {}

    def attach(llm, layer, sink, **kwargs):
        sinks[layer] = sink
        return lambda: None

    class Sampling:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class Output:
        def __init__(self, request_id: str):
            self.request_id = request_id

    calls: list[list[str]] = []

    def generate(prompts, params):
        calls.append(prompts)
        sinks[1].rows[:] = [
            {
                "phase": "prefill",
                "request_id": "p1",
                "hidden": torch.tensor([1.0, 2.0]),
            },
            {
                "phase": "prefill",
                "request_id": "p0",
                "hidden": torch.tensor([3.0, 4.0]),
            },
        ]
        sinks[2].rows[:] = [
            {
                "phase": "prefill",
                "request_id": "p1",
                "hidden": torch.tensor([5.0, 6.0]),
            },
            {
                "phase": "prefill",
                "request_id": "p0",
                "hidden": torch.tensor([7.0, 8.0]),
            },
        ]
        return [Output("p0"), Output("p1")]

    monkeypatch.setattr(runner, "attach_capture", attach)
    monkeypatch.setattr(runner, "_sampling_params", lambda **kwargs: Sampling(**kwargs))
    llm = SimpleNamespace(generate=generate)
    result = runner.capture_prompt_hiddens(llm, ["zero", "one"], [1, 2])
    assert torch.equal(result[1], torch.tensor([[3, 4], [1, 2]], dtype=torch.float32))
    assert torch.equal(result[2], torch.tensor([[7, 8], [5, 6]], dtype=torch.float32))
    assert calls == [["zero", "one"]]


def test_capture_prompt_hiddens_never_exceeds_batch_and_preserves_alignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sinks: dict[int, runner.CaptureSink] = {}
    calls: list[list[str]] = []

    def attach(llm, layer, sink, **kwargs):
        sinks[layer] = sink
        return lambda: None

    class Output:
        def __init__(self, request_id: str):
            self.request_id = request_id

    def generate(prompts, params):
        calls.append(prompts)
        for prompt in prompts:
            sinks[1].rows.append(
                {"phase": "prefill", "request_id": prompt, "hidden": [len(prompt)]}
            )
        return [Output(prompt) for prompt in prompts]

    monkeypatch.setattr(runner, "attach_capture", attach)
    monkeypatch.setattr(runner, "_sampling_params", lambda **kwargs: kwargs)
    result = runner.capture_prompt_hiddens(
        SimpleNamespace(generate=generate), ["a", "bb", "ccc"], [1], batch_prompts=2
    )
    assert calls == [["a", "bb"], ["ccc"]]
    assert torch.equal(result[1], torch.tensor([[1], [2], [3]], dtype=torch.float32))


def test_capture_prompt_hiddens_detaches_partial_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detached: list[int] = []

    def attach(llm, layer, sink):
        if layer == 2:
            raise RuntimeError("attach failed")
        return lambda: detached.append(layer)

    monkeypatch.setattr(runner, "attach_capture", attach)
    with pytest.raises(RuntimeError, match="attach failed"):
        runner.capture_prompt_hiddens(SimpleNamespace(), ["p"], [1, 2])
    assert detached == [1]


def test_build_and_round_trip_directions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        runner,
        "capture_prompt_hiddens",
        lambda llm, prompts, layers, batch_prompts=64: {
            layer: torch.tensor([[2.0, 0.0], [0.0, 2.0]])
            if prompts[0] == "pos"
            else torch.tensor([[0.0, 1.0], [0.0, 3.0]])
            for layer in layers
        },
    )
    records = runner.build_dim_directions(object(), ["pos"], ["neg"], [4])
    assert records[4].direction.dtype == torch.float32
    assert records[4].mean_norm == pytest.approx(2.0)
    path = tmp_path / "directions.json"
    runner.save_directions(path, records)
    loaded = runner.load_directions(path)
    assert torch.equal(loaded[4].direction, records[4].direction)
    assert loaded[4].mean_norm == records[4].mean_norm


def test_steered_generate_attaches_and_detaches_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []
    detached: list[str] = []

    def steering(*args, **kwargs):
        calls.append(("steering", kwargs))
        return lambda: detached.append("steering")

    def capture(*args, **kwargs):
        calls.append(("capture", kwargs))
        return lambda: detached.append("capture")

    monkeypatch.setattr(runner, "attach_steering", steering)
    monkeypatch.setattr(runner, "attach_capture", capture)
    monkeypatch.setattr(runner, "make_decode_index_resolver", lambda llm: "resolver")
    monkeypatch.setattr(runner, "_sampling_params", lambda **kwargs: kwargs)

    def generate(prompts, params):
        raise RuntimeError("boom")

    spec = runner.SteeringSpec(
        7, torch.ones(2), 0.5, 2.0, SteeringSchedule.prefix(3), -1.0
    )
    with pytest.raises(RuntimeError, match="boom"):
        runner.steered_generate(
            SimpleNamespace(generate=generate),
            ["p"],
            4,
            spec,
            sink=runner.CaptureSink(),
            scalar_directions=[torch.ones(2)],
        )
    assert [name for name, _ in calls] == ["steering", "capture"]
    assert detached == ["capture", "steering"]
    assert calls[0][1]["decode_index_resolver"] == "resolver"
    assert calls[1][1]["scalar_directions"]


def test_steered_generate_baseline_has_no_steering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "_sampling_params", lambda **kwargs: kwargs)
    llm = SimpleNamespace(
        generate=lambda prompts, params: [
            SimpleNamespace(
                request_id="request-1", outputs=[SimpleNamespace(text="ok")]
            )
        ]
    )
    results = runner.steered_generate(llm, ["p"], 2, None)
    assert results == [runner.GenerateResult(text="ok", request_id="request-1")]


def test_steered_generate_capture_layer_and_missing_layer_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[int] = []

    monkeypatch.setattr(runner, "make_decode_index_resolver", lambda llm: "resolver")
    monkeypatch.setattr(runner, "_sampling_params", lambda **kwargs: kwargs)

    def capture(*args, **kwargs):
        captured.append(kwargs["layer"])
        return lambda: None

    monkeypatch.setattr(runner, "attach_capture", capture)
    llm = SimpleNamespace(
        generate=lambda prompts, params: [
            SimpleNamespace(request_id="r", outputs=[SimpleNamespace(text="ok")])
        ]
    )
    runner.steered_generate(
        llm, ["p"], 2, None, sink=runner.CaptureSink(), capture_layer=20
    )
    assert captured == [20]
    with pytest.raises(
        ValueError, match="capture requires capture_layer or a steering spec"
    ):
        runner.steered_generate(llm, ["p"], 2, None, sink=runner.CaptureSink())


def test_manifest_round_trip_and_stale_hint(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    config = {"layer": 20, "alpha": 1.0}
    runner.write_manifest(path, config)
    runner.verify_manifest(path, config)
    with pytest.raises(RuntimeError, match="stale"):
        runner.verify_manifest(path, {"layer": 21, "alpha": 1.0})
    runner.verify_manifest(tmp_path / "missing.json", config)


def test_completeness_helpers(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    assert runner.missing_ids(path, ["b", "a"]) == ["a", "b"]
    runner.append_jsonl(path, [{"id": "a"}])
    assert runner.missing_ids(path, ["a", "b", "c"]) == ["b", "c"]
    with pytest.raises(RuntimeError, match="phase.*b.*c"):
        runner.require_complete(path, ["a", "b", "c"], label="phase")
    runner.append_jsonl(path, [{"id": "b"}, {"id": "c"}])
    runner.require_complete(path, ["a", "b", "c"], label="phase")


def test_condition_id_formats_parts() -> None:
    assert runner.condition_id("exp3", "val", "full", 20, 1.0) == "exp3/val/full/20/1.0"
    assert runner.condition_id("x", 0.25, "y") == "x/0.25/y"
