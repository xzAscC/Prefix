from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from prefix import runner
from prefix.judge import JudgeBlocked
from prefix.metrics import binned_success_rate

_SPEC = importlib.util.spec_from_file_location(
    "run_exp1", Path(__file__).parents[1] / "scripts" / "run_exp1.py"
)
assert _SPEC is not None and _SPEC.loader is not None
run_exp1 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_exp1)


def test_generation_resumes_in_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "generations.jsonl"
    runner.append_jsonl(path, [{"id": "0", "text": "old"}, {"id": "1", "text": "old"}])
    calls: list[list[str]] = []

    def generate(llm, prompts, **kwargs):
        calls.append(prompts)
        return [
            SimpleNamespace(request_id=f"r{prompt}", text=f"answer-{prompt}")
            for prompt in prompts
        ]

    monkeypatch.setattr(run_exp1, "steered_generate", generate)
    monkeypatch.setattr(
        run_exp1, "chat_prompt", lambda tokenizer, text, enable_thinking: text
    )
    monkeypatch.setattr(run_exp1, "CaptureSink", lambda: SimpleNamespace(rows=[]))
    run_exp1.generate_harmbench(
        object(),
        {"model": {"layer": 20}, "generation": {"max_new_tokens": 4}},
        [{"behavior": str(i)} for i in range(5)],
        runner.DirectionRecord(torch.tensor([1.0]), 1.0),
        path,
        tmp_path / "trajectories.jsonl",
        2,
    )
    assert calls == [["2", "3"], ["4"]]
    assert {row["id"] for row in runner.read_jsonl(path)} == {"0", "1", "2", "3", "4"}


def test_trajectory_assembly_groups_and_sorts_rows() -> None:
    sink = SimpleNamespace(
        rows=[
            {"phase": "decode", "request_id": "b", "k": 2, "dots": [6.0], "norm": 3.0},
            {"phase": "decode", "request_id": "a", "k": 2, "dots": [4.0], "norm": 2.0},
            {"phase": "decode", "request_id": "b", "k": 1, "dots": [2.0], "norm": 1.0},
        ]
    )
    assert run_exp1.trajectory_records(sink, {"a": "0", "b": "1"}) == [
        {"id": "0", "c": [2.0]},
        {"id": "1", "c": [2.0, 2.0]},
    ]


def test_judge_requires_complete_generations(tmp_path: Path) -> None:
    path = tmp_path / "generations.jsonl"
    runner.append_jsonl(path, [{"id": "0", "text": "x"}])
    with pytest.raises(RuntimeError, match="missing"):
        run_exp1.judge_harmbench(
            path,
            tmp_path / "judge.jsonl",
            [{"behavior": "x"}, {"behavior": "y"}],
            object(),
        )


def test_judge_isolates_blocked_item(tmp_path: Path) -> None:
    generations = tmp_path / "generations.jsonl"
    judge = tmp_path / "judge.jsonl"
    runner.append_jsonl(
        generations,
        [{"id": "0", "text": "x"}, {"id": "1", "text": "y"}],
    )

    def judge_safety(request: str, response: str) -> bool:
        if request == "blocked":
            raise JudgeBlocked("blocked")
        return True

    run_exp1.judge_harmbench(
        generations,
        judge,
        [{"behavior": "ok"}, {"behavior": "blocked"}],
        SimpleNamespace(judge_safety=judge_safety),
    )

    assert runner.read_jsonl(judge) == [
        {"id": "0", "safe": True, "blocked": False},
        {"id": "1", "safe": None, "blocked": True},
    ]


def test_analyze_writes_json_and_pdf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generations = tmp_path / "generations.jsonl"
    judge = tmp_path / "judge.jsonl"
    traces = tmp_path / "trajectories.jsonl"
    runner.append_jsonl(
        generations, [{"id": "0", "text": "x"}, {"id": "1", "text": "y"}]
    )
    runner.append_jsonl(
        judge,
        [
            {"id": "0", "safe": True, "blocked": False},
            {"id": "1", "safe": False, "blocked": False},
        ],
    )
    runner.append_jsonl(
        traces, [{"id": "0", "c": [0.1, 0.2]}, {"id": "1", "c": [0.3, 0.4]}]
    )
    monkeypatch.setattr(run_exp1, "CHECKPOINTS", tmp_path)
    monkeypatch.setattr(run_exp1, "FIGS", tmp_path / "figs")
    result = run_exp1.analyze(
        {"analysis": {"n_bins": 2, "early_tokens": [1, 2]}},
        judge,
        traces,
        ["0", "1"],
        tmp_path / "results.json",
    )
    assert result["n"] == 2
    assert result["blocked"] == 0
    assert (tmp_path / "results.json").exists()
    assert (tmp_path / "figs" / "exp1_ct_by_label.pdf").exists()
    assert (tmp_path / "figs" / "exp1_gt_early.pdf").exists()


def test_analyze_excludes_blocked_item_and_reports_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    judge = tmp_path / "judge.jsonl"
    traces = tmp_path / "trajectories.jsonl"
    runner.append_jsonl(
        judge,
        [
            {"id": "0", "safe": True, "blocked": False},
            {"id": "1", "safe": None, "blocked": True},
            {"id": "2", "safe": False, "blocked": False},
        ],
    )
    runner.append_jsonl(
        traces,
        [
            {"id": "0", "c": [0.1]},
            {"id": "1", "c": [0.2]},
            {"id": "2", "c": [0.3]},
        ],
    )
    monkeypatch.setattr(run_exp1, "FIGS", tmp_path / "figs")
    result = run_exp1.analyze(
        {"analysis": {"n_bins": 2, "early_tokens": [1]}},
        judge,
        traces,
        ["0", "1", "2"],
        tmp_path / "results.json",
    )

    assert result["blocked"] == 1
    assert result["n"] == 2
    assert result["gt_early"]["1"]


def test_early_binning_matches_metrics() -> None:
    values = np.array([0.1, 0.2, 0.9])
    labels = np.array([True, False, True])
    expected = binned_success_rate(values, labels, 2)
    assert run_exp1.early_bins(values, labels, 2) == [
        {"bin_center": item.bin_center, "rate": item.rate, "count": item.count}
        for item in expected
    ]
