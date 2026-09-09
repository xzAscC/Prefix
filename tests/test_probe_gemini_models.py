from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from prefix.judge import JudgeBlocked


SCRIPT = Path(__file__).parents[1] / "scripts" / "probe_gemini_models.py"


def entrypoint():
    spec = importlib.util.spec_from_file_location("probe_gemini_models", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_model_order_is_fixed() -> None:
    probe = entrypoint()
    assert probe.DEFAULT_MODELS == (
        "gemini-3.8-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
    )


def test_response_root_resolves_harmbench_jsonl(tmp_path: Path) -> None:
    probe = entrypoint()
    path = tmp_path / "model" / "harmbench" / "responses.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text('{"id":"pair-1"}\n', encoding="utf-8")
    assert probe.resolve_response_path(tmp_path) == path


def test_probe_records_labels_blocks_and_resumes_atomically(tmp_path: Path) -> None:
    probe = entrypoint()
    response_path = tmp_path / "responses.jsonl"
    response_path.write_text(
        "\n".join(
            [
                json.dumps({"id": "p1", "behavior": "request 1", "response": "r1"}),
                json.dumps({"id": "p2", "behavior": "request 2", "response": "r2"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    checkpoint = tmp_path / "checkpoint.json"
    calls: list[tuple[str, str, str]] = []

    class FakeJudge:
        def __init__(self, model: str) -> None:
            self.model = model

        def judge_safety(self, behavior: str, response: str) -> bool:
            calls.append((self.model, behavior, response))
            if behavior == "request 2" and self.model == "gemini-3.8-flash":
                raise JudgeBlocked("prompt blocked by provider: SAFETY")
            return self.model == "gemini-3.8-flash"

    first = probe.run_probe(
        rows=probe.load_response_rows(response_path),
        models=("gemini-3.8-flash", "gemini-3.6-flash"),
        output_path=checkpoint,
        judge_factory=FakeJudge,
        sample_limit=2,
    )
    assert len(first) == 4
    assert first[0]["label"] == "SAFE"
    assert first[1]["status"] == "blocked"
    assert first[1]["prompt_block_reason"] == "SAFETY"
    assert first[2]["label"] == "UNSAFE"
    assert all("latency_seconds" in row and row["error"] is None for row in first[:1])
    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert len(saved["records"]) == 4

    probe.run_probe(
        rows=probe.load_response_rows(response_path),
        models=("gemini-3.8-flash", "gemini-3.6-flash"),
        output_path=checkpoint,
        judge_factory=FakeJudge,
        sample_limit=2,
    )
    assert len(calls) == 4


@pytest.mark.parametrize("limit", [-1])
def test_sample_limit_must_not_be_negative(limit: int, tmp_path: Path) -> None:
    probe = entrypoint()
    with pytest.raises(ValueError, match="sample_limit"):
        probe.run_probe(
            rows=[],
            models=("gemini-3.8-flash",),
            output_path=tmp_path / "checkpoint.json",
            judge_factory=lambda model: Any,
            sample_limit=limit,
        )


def test_probe_digest_binds_ordered_candidates_pairs_rubric_and_settings() -> None:
    probe = entrypoint()
    pairs = [
        {"pair_id": "p1", "behavior": "b1", "response": "r1"},
        {"pair_id": "p2", "behavior": "b2", "response": "r2"},
    ]
    first = probe.probe_config_digest(models=("model-a", "model-b"), pairs=pairs)
    assert first == probe.probe_config_digest(
        models=("model-a", "model-b"), pairs=[dict(pair) for pair in pairs]
    )
    assert first != probe.probe_config_digest(
        models=("model-b", "model-a"), pairs=pairs
    )
    changed = [dict(pairs[0], response="changed"), pairs[1]]
    assert first != probe.probe_config_digest(
        models=("model-a", "model-b"), pairs=changed
    )
    assert first != probe.probe_config_digest(
        models=("model-a", "model-b"), pairs=pairs, settings={"temperature": 1.0}
    )


def test_probe_replaces_retryable_rows_without_duplicates(tmp_path: Path) -> None:
    probe = entrypoint()
    pairs = [
        {"pair_id": "p1", "behavior": "b1", "response": "r1"},
        {"pair_id": "p2", "behavior": "b2", "response": "r2"},
    ]
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(
        json.dumps(
            {
                "config": probe.probe_config(models=("model-a",), pairs=pairs),
                "config_sha256": probe.probe_config_digest(
                    models=("model-a",), pairs=pairs
                ),
                "records": [
                    {"model": "model-a", "pair_id": "p1", "status": "error"},
                    {
                        "model": "model-a",
                        "pair_id": "p2",
                        "status": "ok",
                        "label": "SAFE",
                    },
                ],
            }
        )
    )

    class FakeJudge:
        def __init__(self, model: str) -> None:
            self.model = model

        def judge_safety(self, behavior: str, response: str) -> bool:
            return True

    records = probe.run_probe(
        rows=pairs,
        models=("model-a",),
        output_path=checkpoint,
        judge_factory=FakeJudge,
    )

    assert [(row["model"], row["pair_id"]) for row in records] == [
        ("model-a", "p1"),
        ("model-a", "p2"),
    ]
    assert records[0]["status"] == "ok"
    assert len(records) == 2


def test_probe_rejects_resume_with_changed_rubric_or_settings(tmp_path: Path) -> None:
    probe = entrypoint()
    pairs = [{"pair_id": "p1", "behavior": "b1", "response": "r1"}]
    checkpoint = tmp_path / "checkpoint.json"

    class FakeJudge:
        def __init__(self, model: str) -> None:
            self.model = model

        def judge_safety(self, behavior: str, response: str) -> bool:
            return True

    probe.run_probe(
        rows=pairs,
        models=("model-a",),
        output_path=checkpoint,
        judge_factory=FakeJudge,
        rubric="rubric-v1",
        settings={"temperature": 0.0},
    )

    with pytest.raises(ValueError, match="config digest"):
        probe.run_probe(
            rows=pairs,
            models=("model-a",),
            output_path=checkpoint,
            judge_factory=FakeJudge,
            rubric="rubric-v2",
            settings={"temperature": 0.0},
        )
