from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_spec = importlib.util.spec_from_file_location(
    "run_exp3", Path(__file__).parents[1] / "scripts" / "run_exp3.py"
)
assert _spec and _spec.loader
exp3 = importlib.util.module_from_spec(_spec)
sys.modules["run_exp3"] = exp3
_spec.loader.exec_module(exp3)


def config(limit: int | None = None) -> dict[str, object]:
    return {
        "schedules": ["full", "prefix-5", "one-token"],
        "grid": {
            "layers": [1, 5, 8, 12, 16, 19, 20, 27, 30, 34],
            "alphas_by_schedule": {
                "full": [0.01, 0.03, 0.1, 0.3, 1.0],
                "prefix-5": [1.0, 3.0, 10.0, 30.0, 100.0],
                "one-token": [1.0, 3.0, 10.0, 30.0, 100.0],
            },
        },
        "selection": {"capability_cap": 0.9},
        "validation": {"mmlu_source": "validation"},
        "seed": 42,
    }


def test_condition_enumeration_and_smoke_grid() -> None:
    assert len(exp3.validation_conditions(config())) == 150
    small = exp3.validation_conditions(config(), limit=4)
    assert len(small) == 6
    assert {item.layer for item in small} == {19}
    assert {item.alpha for item in small if item.schedule == "full"} == {0.01, 0.03}
    assert all(item.alpha >= 1 for item in small if item.schedule != "full")


def test_selection_cap_and_fallback() -> None:
    points = [(0.4, 0.95), (0.8, 0.91), (0.9, 0.5)]
    assert exp3.choose_selection(points, 1.0, 0.9) == (1, False)
    assert exp3.choose_selection([(0.4, 0.2), (0.8, 0.3)], 1.0, 0.9) == (1, True)


def test_test_generation_requires_selection(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="selection"):
        exp3.require_selection(tmp_path / "missing.json")


def test_chunked_generation_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "rows.jsonl"
    exp3.append_jsonl(path, [{"id": "0", "text": "done"}])
    calls: list[list[str]] = []
    monkeypatch.setattr(
        exp3,
        "steered_generate",
        lambda llm, prompts, **kwargs: (
            calls.append(prompts)
            or [SimpleNamespace(text=p, request_id=p) for p in prompts]
        ),
    )
    exp3.generate_jsonl(object(), ["a", "b", "c"], path, None, 4, 2)
    assert calls == [["b", "c"]]
    assert {row["id"] for row in exp3.read_jsonl(path)} == {"0", "1", "2"}


def test_judge_requires_complete_generation(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="missing"):
        exp3.judge_jsonl(
            tmp_path / "generation.jsonl",
            tmp_path / "judge.jsonl",
            ["0", "1"],
            lambda x: True,
        )


def test_blocked_judge_item_isolated_and_excluded_from_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generation = tmp_path / "generation.jsonl"
    output = tmp_path / "judge.jsonl"
    exp3.append_jsonl(
        generation, [{"id": "0", "text": "blocked"}, {"id": "1", "text": "ok"}]
    )

    def fake_judge(row: dict[str, object]) -> dict[str, object]:
        if row["id"] == "0":
            raise exp3.JudgeBlocked("blocked")
        return {"condition": "baseline", "i": 1, "safe": True}

    exp3.judge_jsonl(
        generation,
        output,
        ["0", "1"],
        fake_judge,
        blocked_result=lambda row: {
            "condition": "baseline",
            "i": int(row["id"]),
            "safe": None,
        },
    )
    rows = exp3.read_jsonl(output)
    assert rows[0] == {
        "id": "0",
        "condition": "baseline",
        "i": 0,
        "safe": None,
        "blocked": True,
    }
    assert rows[1]["safe"] is True and rows[1]["blocked"] is False

    cfg = config()
    cfg["schedules"] = ["full"]
    cfg["grid"] = {"layers": [20], "alphas_by_schedule": {"full": [0.1]}}
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    val_judge = checkpoints / "exp3_val_judge.jsonl"
    exp3.append_jsonl(
        val_judge,
        [
            {"condition": "baseline", "id": "0", "safe": True, "blocked": False},
            {"condition": "full_l20_a0.1", "id": "0", "safe": None, "blocked": True},
        ],
    )
    for condition in ("baseline", "full_l20_a0.1"):
        exp3.append_jsonl(
            checkpoints / f"exp3_val_{condition}_mmlu.jsonl",
            [{"id": "0", "text": "A"}],
        )
    exp3.append_jsonl(
        checkpoints / "exp3_test_judge.jsonl",
        [
            {"condition": "baseline", "safe": None, "blocked": True},
            {"condition": "baseline", "safe": True, "blocked": False},
            {"condition": "full", "safe": None, "blocked": True},
            {"condition": "full", "safe": True, "blocked": False},
        ],
    )
    monkeypatch.setattr(exp3, "make_pdf", lambda *args: None)
    monkeypatch.setattr(exp3, "load_mmlu_pro", lambda split: [{"answer_letter": "A"}])
    exp3.analyze(cfg, tmp_path)
    scores = exp3.read_json(checkpoints / "exp3_val_scores.json")
    results = exp3.read_json(checkpoints / "exp3_results.json")
    assert scores["full_l20_a0.1"] == {"p_safe": 0.0, "a_mmlu": 1.0, "blocked": 1}
    assert results["baseline"] == {"p_safe": 1.0, "a_mmlu": 0.0, "blocked": 1}
    assert results["full"] == {"p_safe": 1.0, "a_mmlu": 0.0, "blocked": 1}


def test_logprob_resume_and_record_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "logprob.jsonl"
    exp3.append_jsonl(
        path,
        [{"condition": "full", "id": "0", "pred_letter": "A", "logprobs": {"A": -1.0}}],
    )

    class Tokenizer:
        def decode(self, token_id: int) -> str:
            return chr(token_id)

    class LLM:
        def generate(self, prompts, params):
            return [
                SimpleNamespace(
                    request_id=str(i),
                    outputs=[
                        SimpleNamespace(logprobs=[{65: SimpleNamespace(logprob=-0.2)}])
                    ],
                )
                for i, _ in enumerate(prompts, 1)
            ]

    attached: list[str] = []
    monkeypatch.setattr(
        exp3,
        "attach_steering",
        lambda *a, **k: (
            attached.append("attach") or (lambda: attached.append("detach"))
        ),
    )
    monkeypatch.setattr(exp3, "SamplingParams", lambda **kwargs: kwargs)
    spec = exp3.SteeringSpec(20, torch.ones(1), 1.0, 1.0, exp3.SteeringSchedule.full())
    exp3.logprob_pass(LLM(), Tokenizer(), ["b", "c"], path, "full", spec, 2)
    rows = exp3.read_jsonl(path)
    assert len(rows) == 2 and rows[1]["logprobs"] == {"A": -0.2}
    assert attached == ["attach", "detach"]


def test_all_orders_selection_before_test_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    monkeypatch.setattr(exp3, "ROOT", tmp_path)
    monkeypatch.setattr(exp3, "load_config", lambda path: config())
    monkeypatch.setattr(exp3, "verify_manifest", lambda *args: None)
    monkeypatch.setattr(exp3, "write_manifest", lambda *args: None)
    monkeypatch.setattr(
        exp3, "direction_phase", lambda *args: events.append("direction")
    )
    monkeypatch.setattr(
        exp3, "generate_validation_phase", lambda *args: events.append("val-generate")
    )
    monkeypatch.setattr(
        exp3, "judge_validation_phase", lambda *args: events.append("val-judge")
    )

    def select(*args: object) -> None:
        events.append("selection")
        (tmp_path / "checkpoints").mkdir(exist_ok=True)
        (tmp_path / "checkpoints" / "exp3_selection.json").write_text("{}")

    monkeypatch.setattr(exp3, "analyze_selection", select)
    monkeypatch.setattr(
        exp3, "generate_test_phase", lambda *args: events.append("test-generate")
    )
    monkeypatch.setattr(
        exp3, "judge_test_phase", lambda *args: events.append("test-judge")
    )
    monkeypatch.setattr(exp3, "analyze_results", lambda *args: events.append("results"))

    exp3._run(["--phase", "all", "--limit", "4"])

    assert events == [
        "direction",
        "val-generate",
        "val-judge",
        "selection",
        "test-generate",
        "test-judge",
        "results",
    ]


def test_generate_without_selection_skips_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(exp3, "generate_validation_phase", lambda *args: None)
    monkeypatch.setattr(
        exp3, "generate_test_phase", lambda *args: pytest.fail("test generation ran")
    )

    exp3.generate_phase(config(), tmp_path, 4, 2)

    assert "selection" in capsys.readouterr().out
