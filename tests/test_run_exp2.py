from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from prefix import runner
from prefix.judge import JudgeBlocked

_SPEC = importlib.util.spec_from_file_location(
    "run_exp2", Path(__file__).parents[1] / "scripts" / "run_exp2.py"
)
assert _SPEC is not None and _SPEC.loader is not None
run_exp2 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_exp2)


def test_limit_shrinks_exp2_alphas_first_three() -> None:
    config = {"steering": {"alphas": [0.0, 0.1, 1.0, 3.0]}}
    assert run_exp2.apply_limit(config, 6)["steering"]["alphas"] == [0.0, 0.1, 1.0]


def test_alpha_zero_passes_none_and_generation_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "hb.jsonl"
    calls: list[object] = []

    def generate(llm, prompts, **kwargs):
        calls.append(kwargs["spec"])
        return [
            SimpleNamespace(request_id=f"r{i}", text="ok") for i in range(len(prompts))
        ]

    monkeypatch.setattr(run_exp2, "steered_generate", generate)
    monkeypatch.setattr(
        run_exp2, "chat_prompt", lambda tokenizer, text, enable_thinking: text
    )
    monkeypatch.setattr(run_exp2, "condition_id", lambda *parts: "cond")
    monkeypatch.setattr(run_exp2, "CaptureSink", lambda: SimpleNamespace(rows=[]))
    monkeypatch.setattr(
        run_exp2,
        "load_directions",
        lambda path: {20: runner.DirectionRecord(torch.tensor([1.0]), 1.0)},
    )
    run_exp2.generate_harmbench(
        object(),
        {"model": {"layer": 20}, "harmbench": {"max_new_tokens": 4}},
        [{"behavior": "a"}, {"behavior": "b"}],
        [0.0],
        runner.DirectionRecord(torch.tensor([1.0]), 1.0),
        tmp_path,
        2,
    )
    assert calls == [None]


def test_phase_gating_does_not_create_engine_in_judge_or_analyze(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    called = False

    def fail(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("engine should not be created")

    monkeypatch.setattr(run_exp2, "get_engine", fail)
    generation_dir = tmp_path / "generations"
    generation_dir.mkdir()
    runner.append_jsonl(
        generation_dir / "exp2_hb_exp2_hb_0.0.jsonl", [{"id": "0", "text": "x"}]
    )
    run_exp2.judge_harmbench(
        generation_dir,
        tmp_path / "j.jsonl",
        [{"behavior": "b"}],
        [0.0],
        SimpleNamespace(judge_safety=lambda request, response: True),
    )
    assert not called


def test_judge_isolates_blocked_item(tmp_path: Path) -> None:
    generation_dir = tmp_path / "generations"
    generation_dir.mkdir()
    runner.append_jsonl(
        generation_dir / "exp2_hb_exp2_hb_0.0.jsonl",
        [{"id": "0", "text": "x"}, {"id": "1", "text": "y"}],
    )

    def judge_safety(request: str, response: str) -> bool:
        if request == "blocked":
            raise JudgeBlocked("blocked")
        return True

    judge_path = tmp_path / "judge.jsonl"
    run_exp2.judge_harmbench(
        generation_dir,
        judge_path,
        [{"behavior": "ok"}, {"behavior": "blocked"}],
        [0.0],
        SimpleNamespace(judge_safety=judge_safety),
    )

    assert runner.read_jsonl(judge_path) == [
        {"id": "exp2/hb/0.0/0", "alpha": 0.0, "i": 0, "safe": True, "blocked": False},
        {"id": "exp2/hb/0.0/1", "alpha": 0.0, "i": 1, "safe": None, "blocked": True},
    ]


def test_analyze_excludes_blocked_item_and_reports_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    judge = tmp_path / "judge.jsonl"
    generation_dir = tmp_path / "generations"
    generation_dir.mkdir()
    runner.append_jsonl(
        judge,
        [
            {
                "id": "exp2_hb_0.0/0",
                "alpha": 0.0,
                "i": 0,
                "safe": True,
                "blocked": False,
            },
            {
                "id": "exp2_hb_0.0/1",
                "alpha": 0.0,
                "i": 1,
                "safe": None,
                "blocked": True,
            },
            {
                "id": "exp2_hb_0.0/2",
                "alpha": 0.0,
                "i": 2,
                "safe": False,
                "blocked": False,
            },
        ],
    )
    runner.append_jsonl(
        generation_dir / "exp2_mmlu_exp2_mmlu_0.0.jsonl",
        [{"id": "0", "pred": "A"}],
    )
    monkeypatch.setattr(run_exp2, "FIGS", tmp_path / "figs")
    result = run_exp2.analyze(
        {"mmlu": {"n": 1}, "seed": 0},
        generation_dir,
        judge,
        [{"answer_letter": "A"}],
        [0.0],
        tmp_path / "results.json",
    )

    assert result["blocked"] == {"0.0": 1}
    assert result["p_safe"]["0.0"] == 0.5
