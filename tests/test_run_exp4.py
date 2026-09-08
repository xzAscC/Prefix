from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
from types import SimpleNamespace
import json

import pytest

_spec = importlib.util.spec_from_file_location(
    "run_exp4", Path(__file__).parents[1] / "scripts" / "run_exp4.py"
)
assert _spec and _spec.loader
exp4 = importlib.util.module_from_spec(_spec)
sys.modules["run_exp4"] = exp4
_spec.loader.exec_module(exp4)


def config() -> dict[str, object]:
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
    }


def test_condition_enumeration_and_smoke_grid() -> None:
    assert len(exp4.validation_conditions(config())) == 300
    small = exp4.validation_conditions(config(), limit=4)
    assert len(small) == 12
    assert {item.layer for item in small} == {19}
    assert all(item.alpha >= 1 for item in small if item.schedule != "full")


def test_selection_fallback_and_orientation() -> None:
    assert exp4.choose_selection([(0.2, 1.0), (0.8, 0.91)], 1.0, 0.9) == (1, False)
    assert exp4.choose_selection([(0.2, 0.1), (0.8, 0.2)], 1.0, 0.9) == (1, True)
    assert (
        exp4.steer_success("pos", {"format_boxed": True, "format_answer_is": False})
        is True
    )
    assert (
        exp4.steer_success("neg", {"format_boxed": True, "format_answer_is": False})
        is False
    )


def test_test_generation_requires_selection(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="selection"):
        exp4.require_selection(tmp_path / "none.json")


def test_chunked_generation_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "rows.jsonl"
    exp4.append_jsonl(path, [{"id": "0", "text": "done"}])
    calls: list[list[str]] = []
    monkeypatch.setattr(
        exp4,
        "steered_generate",
        lambda llm, prompts, **kwargs: (
            calls.append(prompts)
            or [SimpleNamespace(text=p, request_id=p) for p in prompts]
        ),
    )
    exp4.generate_jsonl(object(), ["a", "b", "c"], path, None, 4, 2)
    assert calls == [["b", "c"]]


def test_judge_completeness_gate(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="missing"):
        exp4.judge_jsonl(
            tmp_path / "gen.jsonl", tmp_path / "judge.jsonl", ["0"], lambda x: {}
        )


def test_judge_phase_passes_math_answers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generation = tmp_path / "checkpoints" / "exp4_val_baseline.jsonl"
    exp4.append_jsonl(generation, [{"id": "0", "text": "answer"}])
    answers = [{"problem": "p0", "answer": "42"}]
    seen: list[str] = []

    class FakeJudge:
        def __init__(self, **kwargs: object) -> None:
            pass

        def judge_math(self, response: str, expected_answer: str) -> dict[str, bool]:
            seen.append(expected_answer)
            return {
                "format_boxed": True,
                "format_answer_is": True,
                "answer_correct": True,
            }

    monkeypatch.setattr(exp4, "GeminiJudge", FakeJudge)
    monkeypatch.setattr(exp4, "load_math500", lambda: answers)
    monkeypatch.setattr(
        exp4,
        "math500_partition",
        lambda **kwargs: {"direction": [], "val": [0], "test": []},
    )
    monkeypatch.setattr(
        exp4,
        "validation_conditions",
        lambda cfg, limit: [],
    )
    monkeypatch.setattr(
        exp4,
        "judge_batch",
        lambda judge, rows, max_workers: [judge(row) for row in rows],
    )

    exp4.judge_phase({"judge": {"model": "fake", "region": "fake"}}, tmp_path, 4)

    assert seen == ["42"]


def test_blocked_math_item_isolated_and_excluded_from_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generation = tmp_path / "checkpoints" / "exp4_val_baseline.jsonl"
    exp4.append_jsonl(
        generation,
        [{"id": "0", "text": "blocked"}, {"id": "1", "text": "ok"}],
    )
    answers = [{"problem": "p0", "answer": "42"}, {"problem": "p1", "answer": "42"}]
    real_validation_conditions = exp4.validation_conditions

    class FakeJudge:
        def __init__(self, **kwargs: object) -> None:
            pass

        def judge_math(self, response: str, expected_answer: str) -> dict[str, bool]:
            if response == "blocked":
                raise exp4.JudgeBlocked("blocked")
            return {
                "format_boxed": True,
                "format_answer_is": True,
                "answer_correct": True,
            }

    monkeypatch.setattr(exp4, "GeminiJudge", FakeJudge)
    monkeypatch.setattr(exp4, "load_math500", lambda: answers)
    monkeypatch.setattr(
        exp4,
        "math500_partition",
        lambda **kwargs: {"direction": [], "val": [0, 1], "test": []},
    )
    monkeypatch.setattr(exp4, "validation_conditions", lambda cfg, limit: [])
    monkeypatch.setattr(
        exp4,
        "judge_batch",
        lambda judge, rows, max_workers: [judge(row) for row in rows],
    )
    exp4.judge_phase({"judge": {"model": "fake", "region": "fake"}}, tmp_path, 4)

    rows = exp4.read_jsonl(tmp_path / "checkpoints" / "exp4_val_judge.jsonl")
    assert rows[0] == {
        "id": "baseline/0",
        "condition": "baseline",
        "format_boxed": None,
        "format_answer_is": None,
        "answer_correct": None,
        "blocked": True,
    }
    assert rows[1]["answer_correct"] is True and rows[1]["blocked"] is False

    cfg = config()
    cfg["schedules"] = ["full"]
    cfg["grid"] = {"layers": [20], "alphas_by_schedule": {"full": [0.1]}}
    monkeypatch.setattr(exp4, "validation_conditions", real_validation_conditions)
    val_judge = tmp_path / "checkpoints" / "exp4_val_judge.jsonl"
    exp4.append_jsonl(
        val_judge,
        [
            {"condition": "baseline", "answer_correct": True, "blocked": False},
            {
                "condition": "pos_full_l20_a0.1",
                "format_boxed": None,
                "format_answer_is": None,
                "answer_correct": None,
                "blocked": True,
            },
            {
                "condition": "pos_full_l20_a0.1",
                "format_boxed": True,
                "format_answer_is": True,
                "answer_correct": True,
                "blocked": False,
            },
            {
                "condition": "neg_full_l20_a0.1",
                "format_boxed": None,
                "format_answer_is": None,
                "answer_correct": None,
                "blocked": True,
            },
            {
                "condition": "neg_full_l20_a0.1",
                "format_boxed": True,
                "format_answer_is": True,
                "answer_correct": True,
                "blocked": False,
            },
        ],
    )
    exp4.analyze_selection(cfg, tmp_path, None)
    scores = exp4.read_json(tmp_path / "checkpoints" / "exp4_val_scores.json")
    assert scores["pos_full_l20_a0.1"] == {
        "steer_success": 1.0,
        "a_math": 1.0,
        "blocked": 1,
    }
    exp4.append_jsonl(
        tmp_path / "checkpoints" / "exp4_test_judge.jsonl",
        [
            {"condition": "baseline", "answer_correct": True, "blocked": False},
            {
                "condition": "pos_full",
                "format_boxed": None,
                "format_answer_is": None,
                "answer_correct": None,
                "blocked": True,
            },
            {
                "condition": "pos_full",
                "format_boxed": True,
                "format_answer_is": True,
                "answer_correct": True,
                "blocked": False,
            },
            {
                "condition": "neg_full_l20_a0.1",
                "format_boxed": True,
                "format_answer_is": True,
                "answer_correct": True,
                "blocked": False,
            },
        ],
    )
    monkeypatch.setattr(exp4, "make_pdf", lambda *args: None)
    exp4.analyze_results(cfg, tmp_path, None)
    results = exp4.read_json(tmp_path / "checkpoints" / "exp4_results.json")
    assert results["pos/full"] == {"steer_success": 1.0, "a_math": 1.0, "blocked": 1}


def test_all_orders_selection_before_test_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    monkeypatch.setattr(exp4, "ROOT", tmp_path)
    monkeypatch.setattr(exp4, "load_config", lambda path: config())
    monkeypatch.setattr(exp4, "verify_manifest", lambda *args: None)
    monkeypatch.setattr(exp4, "write_manifest", lambda *args: None)
    monkeypatch.setattr(
        exp4, "direction_phase", lambda *args: events.append("direction")
    )
    monkeypatch.setattr(
        exp4, "generate_validation_phase", lambda *args: events.append("val-generate")
    )
    monkeypatch.setattr(
        exp4, "judge_validation_phase", lambda *args: events.append("val-judge")
    )

    def select(*args: object) -> None:
        events.append("selection")
        (tmp_path / "checkpoints").mkdir(exist_ok=True)
        (tmp_path / "checkpoints" / "exp4_selection.json").write_text("{}")

    monkeypatch.setattr(exp4, "analyze_selection", select)
    monkeypatch.setattr(
        exp4, "generate_test_phase", lambda *args: events.append("test-generate")
    )
    monkeypatch.setattr(
        exp4, "judge_test_phase", lambda *args: events.append("test-judge")
    )
    monkeypatch.setattr(exp4, "analyze_results", lambda *args: events.append("results"))

    exp4._run(["--phase", "all", "--limit", "4"])

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
    monkeypatch.setattr(exp4, "generate_validation_phase", lambda *args: None)
    monkeypatch.setattr(
        exp4, "generate_test_phase", lambda *args: pytest.fail("test generation ran")
    )

    exp4.generate_phase(config(), tmp_path, 4, 2)

    assert "selection" in capsys.readouterr().out
