from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import subprocess
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Iterator, TypedDict, cast

import pytest

from prefix.preflight_marker import PREFLIGHT_MARKER_REVISIONS, parse_preflight_marker


SCRIPT = Path(__file__).parents[1] / "scripts" / "validate_olmo_truncation.py"
MODEL_ID = "allenai/Olmo-3-7B-Think"
MODEL_REVISION = "d97e442d7cc678210054dbcc9b440894d62c89a4"
CONDITIONS = ("current_zero_shot_1024", "current_zero_shot_2048")
TEST_FORMAL_ROOT = Path("/formal-run")


class SourceRow(TypedDict):
    id: str
    category: str
    question: str
    options: list[str]
    gold: str
    generated_token_count: int
    extracted_answer: str | None
    status: str


class GenerationOutput(TypedDict):
    token_ids: list[int]
    finish_reason: str
    text: str


def entrypoint() -> ModuleType:
    if not SCRIPT.exists():
        pytest.fail(f"missing truncation validation harness: {SCRIPT}")
    spec = importlib.util.spec_from_file_location("validate_olmo_truncation", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_rows() -> list[SourceRow]:
    return [
        {
            "id": "science-null-0",
            "category": "science",
            "question": "What is two plus two?",
            "options": ["3", "4", "5"],
            "gold": "B",
            "generated_token_count": 1024,
            "extracted_answer": None,
            "status": "ok",
        },
        {
            "id": "science-extracted-0",
            "category": "science",
            "question": "Which is a mammal?",
            "options": ["cat", "oak", "rock"],
            "gold": "A",
            "generated_token_count": 100,
            "extracted_answer": "A",
            "status": "ok",
        },
        {
            "id": "history-null-0",
            "category": "history",
            "question": "Who wrote Hamlet?",
            "options": ["Shakespeare", "Austen", "Homer"],
            "gold": "A",
            "generated_token_count": 1024,
            "extracted_answer": None,
            "status": "ok",
        },
        {
            "id": "history-extracted-0",
            "category": "history",
            "question": "Which year followed 2025?",
            "options": ["2024", "2026", "2027"],
            "gold": "B",
            "generated_token_count": 100,
            "extracted_answer": "B",
            "status": "ok",
        },
    ]


def write_jsonl(path: Path, rows: list[SourceRow]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_two_condition_contract_and_streaming_bounded_selection(tmp_path: Path) -> None:
    harness = entrypoint()
    assert tuple(harness.CONDITIONS) == CONDITIONS
    source = tmp_path / "responses.jsonl"
    rows = source_rows()
    write_jsonl(source, rows)

    class OnePassRows(Iterator[SourceRow]):
        def __init__(self, values: list[SourceRow]) -> None:
            self.values = iter(values)

        def __iter__(self) -> OnePassRows:
            return self

        def __next__(self) -> SourceRow:
            return next(self.values)

        def __len__(self) -> int:
            raise AssertionError("selection must not materialize the source")

        def __getitem__(self, index: int) -> SourceRow:
            raise AssertionError(f"selection indexed the source at {index}")

    prepared = harness.prepare_source_selection(
        source,
        rows=OnePassRows(rows),
        quotas={
            "science": {"saturated_null": 1, "saturated_extracted": 1},
            "history": {"saturated_null": 1, "saturated_extracted": 1},
        },
        expected_categories=("science", "history"),
        seed=17,
    )
    assert prepared["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert prepared["population_counts"] == {
        "science/saturated_null": 1,
        "science/unsaturated_extracted": 1,
        "history/saturated_null": 1,
        "history/unsaturated_extracted": 1,
    }
    assert all(
        {"id", "selection_stratum", "population_count", "sample_count", "weight"}
        <= row.keys()
        for row in prepared["records"]
    )


def test_no_steering_producer_json_marker_is_accepted_by_olmo_consumer(
    tmp_path: Path,
) -> None:
    smoke_root = tmp_path / "preflight"
    checkpoint_root = tmp_path / "preflight-checkpoints"
    for model_id, slug in (
        ("Qwen/Qwen3-4B", "Qwen--Qwen3-4B"),
        ("Qwen/Qwen3-14B", "Qwen--Qwen3-14B"),
        (MODEL_ID, "allenai--Olmo-3-7B-Think"),
        ("allenai/Olmo-3-32B-Think", "allenai--Olmo-3-32B-Think"),
    ):
        (smoke_root / slug).mkdir(parents=True)
        (checkpoint_root / slug).mkdir(parents=True)
        (smoke_root / slug / "summary.json").write_bytes(b"summary")
        (checkpoint_root / slug / "manifest.json").write_bytes(b"manifest")

    marker_command = [
        sys.executable,
        str(Path(__file__).parents[1] / "scripts" / "validate_no_steering_smoke.py"),
        "--write-preflight-marker",
        str(tmp_path / "preflight.ok"),
        "--run-root",
        str(tmp_path),
        "--preflight-job-id",
        "job-123",
        "--output-root",
        str(smoke_root),
        "--checkpoint-root",
        str(checkpoint_root),
    ]
    for model_id in (
        "Qwen/Qwen3-4B",
        "Qwen/Qwen3-14B",
        MODEL_ID,
        "allenai/Olmo-3-32B-Think",
    ):
        marker_command.extend(("--model-id", model_id))
    produced = subprocess.run(
        marker_command, cwd=Path(__file__).parents[1], capture_output=True, check=False
    )
    assert produced.returncode == 0, produced.stderr.decode()
    marker_bytes = produced.stdout
    marker = tmp_path / "producer-output.ok"
    marker.write_bytes(marker_bytes)

    harness = entrypoint()
    assert (
        harness.validate_preflight_marker(marker)
        == hashlib.sha256(marker_bytes).hexdigest()
    )


def _write_bound_preflight_marker(root: Path) -> Path:
    marker = root / "preflight.ok"
    models: dict[str, dict[str, str]] = {}
    for model_id in PREFLIGHT_MARKER_REVISIONS:
        slug = model_id.replace("/", "--")
        summary = root / "preflight" / slug / "summary.json"
        manifest = root / "preflight-checkpoints" / slug / "manifest.json"
        summary.parent.mkdir(parents=True, exist_ok=True)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        summary.write_bytes(b"summary")
        manifest.write_bytes(b"manifest")
        models[model_id] = {
            "model_id": model_id,
            "model_revision": PREFLIGHT_MARKER_REVISIONS[model_id],
            "summary_path": str(summary),
            "summary_sha256": hashlib.sha256(summary.read_bytes()).hexdigest(),
            "manifest_path": str(manifest),
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        }
    marker.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "preflight_job_id": "job-123",
                "run_root": str(root),
                "models": models,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return marker


def test_preflight_marker_accepts_canonical_bound_artifacts(tmp_path: Path) -> None:
    marker = _write_bound_preflight_marker(tmp_path)

    parsed = parse_preflight_marker(marker, preflight_job_id="job-123")

    assert parsed["run_root"] == str(tmp_path)


def test_preflight_marker_rejects_symlinked_caller_run_root(tmp_path: Path) -> None:
    marker = _write_bound_preflight_marker(tmp_path)
    caller_alias = tmp_path / "run-root-alias"
    caller_alias.symlink_to(tmp_path, target_is_directory=True)

    parse_preflight_marker(
        marker,
        run_root=tmp_path,
        preflight_job_id="job-123",
    )
    with pytest.raises(ValueError, match="symlink"):
        parse_preflight_marker(
            marker,
            run_root=caller_alias,
            preflight_job_id="job-123",
        )


def test_olmo_marker_validation_rejects_marker_from_another_formal_run_root(
    tmp_path: Path,
) -> None:
    harness = entrypoint()
    marker_root = tmp_path / "marker-run"
    caller_root = tmp_path / "selected-run"
    marker = _write_bound_preflight_marker(marker_root)

    assert harness.validate_preflight_marker(marker, run_root=marker_root / ".")
    with pytest.raises(ValueError, match="run root mismatch"):
        harness.validate_preflight_marker(marker, run_root=caller_root)


def test_preflight_marker_rejects_symlinked_marker_parent(tmp_path: Path) -> None:
    marker = _write_bound_preflight_marker(tmp_path)
    real_marker_parent = tmp_path / "real-marker-parent"
    real_marker_parent.mkdir()
    marker.rename(real_marker_parent / marker.name)
    marker_parent = tmp_path / "marker-parent"
    marker_parent.symlink_to(real_marker_parent, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        parse_preflight_marker(marker_parent / marker.name, preflight_job_id="job-123")


@pytest.mark.parametrize("container", ["preflight", "preflight-checkpoints"])
def test_preflight_marker_rejects_symlinked_artifact_container(
    tmp_path: Path, container: str
) -> None:
    marker = _write_bound_preflight_marker(tmp_path)
    real_container = tmp_path / f"real-{container}"
    (tmp_path / container).rename(real_container)
    (tmp_path / container).symlink_to(real_container, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        parse_preflight_marker(marker, preflight_job_id="job-123")


def test_selection_requires_categories_and_required_saturated_null_cells() -> None:
    harness = entrypoint()
    with pytest.raises(ValueError, match="category"):
        harness.select_source_rows(
            iter(source_rows()[:2]),
            quotas={"science": {"saturated_null": 1}},
            expected_categories=("science", "history"),
        )
    missing_required = [row for row in source_rows() if row["category"] != "science"]
    with pytest.raises(ValueError, match="saturated_null"):
        harness.select_source_rows(
            iter(missing_required),
            quotas={"history": {"saturated_extracted": 1}},
            expected_categories=("history",),
        )


def test_manifest_binds_source_selection_runtime_prompts_and_rejects_stale_duplicates(
    tmp_path: Path,
) -> None:
    harness = entrypoint()
    manifest = harness.build_manifest(
        source_sha256="a" * 64,
        selected_ids=["science-null-0", "history-null-0"],
        selected_weights={"science-null-0": 1.0, "history-null-0": 2.0},
        population_counts={
            "science/saturated_null": 3,
            "history/saturated_null": 2,
        },
        selected_strata={
            "science-null-0": "science/saturated_null",
            "history-null-0": "history/saturated_null",
        },
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        dataset="TIGER-Lab/MMLU-Pro",
        dataset_revision="b189ec765aa7ed75c8acfea42df31fdae71f97be",
        runtime={
            "temperature": 0.0,
            "max_model_len": 8192,
            "budgets": [1024, 2048],
        },
        prompt_hashes={"science-null-0": "b" * 64},
        tokenizer_chat_template_sha256="c" * 64,
        output_paths={"responses": str(tmp_path / "responses.jsonl")},
    )
    manifest["source_path"] = str(_formal_source_path(TEST_FORMAL_ROOT))
    manifest["formal_run_root"] = str(TEST_FORMAL_ROOT)
    manifest["population_counts"] = {
        "science/saturated_null": 6016,
        "history/saturated_null": 6016,
    }
    manifest["population_counts_sha256"] = harness._sha256_value(
        manifest["population_counts"]
    )
    manifest["selected_weights"] = {
        "science-null-0": 6016,
        "history-null-0": 6016,
    }
    manifest["selected_weight_reconstruction"] = harness.SOURCE_COUNT
    evidence = {
        identifier: harness._sha256_value(
            {"source_extracted_answer": None, "source_generated_token_ids": [1, 2]}
        )
        for identifier in ("science-null-0", "history-null-0")
    }
    manifest["source_evidence_sha256"] = evidence
    manifest["source_evidence_binding_sha256"] = harness._sha256_value(
        {
            "source_sha256": manifest["source_sha256"],
            "source_evidence_sha256": evidence,
        }
    )
    harness.validate_manifest(manifest, formal_run_root=TEST_FORMAL_ROOT)
    with pytest.raises(ValueError, match="stale"):
        harness.validate_manifest(
            {**manifest, "runtime": {"temperature": 0.1}},
            formal_run_root=TEST_FORMAL_ROOT,
        )
    with pytest.raises(ValueError, match="duplicate"):
        harness.validate_manifest(
            {**manifest, "selected_ids": ["science-null-0"] * 2},
            formal_run_root=TEST_FORMAL_ROOT,
        )
    with pytest.raises(ValueError, match="population"):
        harness.validate_manifest(
            {**manifest, "population_counts": {"science/saturated_null": 99}},
            formal_run_root=TEST_FORMAL_ROOT,
        )
    with pytest.raises(ValueError, match="strat"):
        harness.validate_manifest(
            {
                **manifest,
                "selected_strata": {
                    "science-null-0": "history/saturated_null",
                    "history-null-0": "history/saturated_null",
                },
            },
            formal_run_root=TEST_FORMAL_ROOT,
        )


def test_generation_and_analysis_reject_source_hash_drift(tmp_path: Path) -> None:
    harness = entrypoint()
    source = tmp_path / "source.jsonl"
    source.write_text('{"id":"x"}\n', encoding="utf-8")
    manifest = {"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    harness.validate_source_sha256(source, manifest)
    source.write_text('{"id":"changed"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="source_sha256"):
        harness.validate_source_sha256(source, manifest)


def test_generation_rows_use_frozen_legacy_parser_for_causal_comparison(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = entrypoint()
    import prefix.runner as runner

    monkeypatch.setattr(
        runner,
        "parse_answer_letter",
        lambda text: "J",
    )
    rows = harness.build_generation_rows(
        [{"id": "x", "condition": CONDITIONS[0]}],
        [
            {
                "token_ids": [11],
                "finish_reason": "length",
                "text": r"A boxed decoy: \boxed{A}",
            }
        ],
    )
    assert rows[0]["extracted_answer"] is None


def test_source_rows_join_by_validated_identity_not_position() -> None:
    harness = entrypoint()
    source = source_rows()[:2]
    dataset = [
        {**source[1], "id": source[1]["id"]},
        {**source[0], "id": source[0]["id"]},
    ]
    joined = harness.join_pinned_dataset_rows(source, dataset)
    assert [row["id"] for row in joined] == [row["id"] for row in source]
    bad = [{**dataset[0], "question": "tampered"}, dataset[1]]
    with pytest.raises(ValueError, match="identity"):
        harness.join_pinned_dataset_rows(source, bad)


def test_cli_and_sbatch_contract_share_all_runtime_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = entrypoint()
    captured: list[argparse.Namespace] = []
    monkeypatch.setattr(harness, "prepare", lambda args: captured.append(args))
    argv = [
        "prepare",
        "--source-responses",
        "source.jsonl",
        "--formal-run-root",
        "formal-run",
        "--output-root",
        "results",
        "--cache-root",
        "cache",
        "--batch-size",
        "8",
        "--max-model-len",
        "8192",
        "--gpu-memory-utilization",
        "0.75",
        "--log-file",
        "run.log",
    ]
    harness.main(argv)
    assert captured and captured[0].phase == "prepare"
    command = harness.build_sbatch_command(argv)
    assert all(flag in command for flag in argv[1::2] if flag.startswith("--"))


@pytest.mark.parametrize("phase", ["prepare", "generate", "analyze"])
def test_formal_cli_phases_require_formal_run_root(
    phase: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = entrypoint()
    monkeypatch.setattr(harness, phase, lambda args: None)
    argv = [phase]
    if phase == "prepare":
        argv.extend(("--source-responses", "source.jsonl"))

    with pytest.raises(SystemExit):
        harness.main(argv)


def test_generation_helpers_validate_counts_and_resume_without_duplicates() -> None:
    harness = entrypoint()
    requests = [{"id": "x", "condition": CONDITIONS[0]}]
    with pytest.raises(ValueError, match="output count"):
        harness.build_generation_rows(requests, [])
    output: GenerationOutput = {
        "token_ids": [11, 12],
        "finish_reason": "length",
        "text": "The answer is (B)",
    }
    rows = harness.build_generation_rows(requests, [output])
    assert rows[0]["token_ids"] == [11, 12]
    assert rows[0]["finish_reason"] == "length"
    merged = harness.merge_generation_rows(
        [{"id": "x", "status": "error", "retryable": True}], rows
    )
    assert [row["id"] for row in merged] == ["x"]
    assert merged[0]["status"] == "ok"


def test_analysis_requires_pairs_provenance_prompt_equality_and_token_prefix() -> None:
    harness = entrypoint()
    manifest = {"selected_ids": ["x"], "conditions": list(CONDITIONS)}
    rows = [
        {
            "id": "x",
            "condition": CONDITIONS[0],
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "prompt_hash": "p",
            "selection_stratum": "saturated_null",
            "generated_token_ids": [1, 2],
            "generated_token_count": 2,
        },
        {
            "id": "x",
            "condition": CONDITIONS[1],
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "prompt_hash": "p",
            "selection_stratum": "saturated_null",
            "generated_token_ids": [1, 2, 3],
            "generated_token_count": 3,
        },
    ]
    harness.validate_analysis(
        rows,
        manifest,
        settings={"temperature": 0.0},
        formal_run_root=TEST_FORMAL_ROOT,
    )
    with pytest.raises(ValueError, match="prefix"):
        harness.validate_analysis(
            [rows[0], {**rows[1], "generated_token_ids": [9, 3, 4]}],
            manifest,
            settings={"temperature": 0.0},
            formal_run_root=TEST_FORMAL_ROOT,
        )
    with pytest.raises(ValueError, match="prompt"):
        harness.validate_analysis(
            [rows[0], {**rows[1], "prompt_hash": "different"}],
            manifest,
            settings={"temperature": 0.0},
            formal_run_root=TEST_FORMAL_ROOT,
        )


def test_baseline_reproduction_requires_answer_and_exact_source_token_sequence() -> (
    None
):
    harness = entrypoint()
    source = {
        "id": "x",
        "extracted_answer": "A",
        "generated_token_ids": [10, 11, 12],
    }
    matching = {**source, "extracted_answer": "A", "generated_token_ids": [10, 11, 12]}
    token_mismatch = {**source, "generated_token_ids": [10, 99, 12]}
    answer_mismatch = {**source, "extracted_answer": "B"}
    assert harness.baseline_reproduction_passes(source, matching) is True
    assert harness.baseline_reproduction_passes(source, token_mismatch) is False
    assert harness.baseline_reproduction_passes(source, answer_mismatch) is False
    assert (
        harness.classify_root_cause(
            {
                "baseline_reproduction": False,
                "paired": {
                    "null_to_answer_rate": 1.0,
                    "weighted_accuracy_delta": 0.5,
                },
            },
            thresholds={"null_rescue": 0.75, "accuracy_delta": 0.1},
        )
        != "CONFIRMED_TRUNCATION"
    )


def test_weighted_metrics_use_baseline_null_rescue_and_actual_budget_hits() -> None:
    harness = entrypoint()
    rows = [
        {
            "id": "null",
            "condition": CONDITIONS[0],
            "selection_weight": 2.0,
            "generated_token_count": 1024,
            "extracted_answer": None,
            "gold": "A",
        },
        {
            "id": "null",
            "condition": CONDITIONS[1],
            "selection_weight": 2.0,
            "generated_token_count": 2048,
            "extracted_answer": "A",
            "gold": "A",
        },
        {
            "id": "correct",
            "condition": CONDITIONS[0],
            "selection_weight": 1.0,
            "generated_token_count": 100,
            "extracted_answer": "A",
            "gold": "A",
        },
        {
            "id": "correct",
            "condition": CONDITIONS[1],
            "selection_weight": 1.0,
            "generated_token_count": 2048,
            "extracted_answer": "A",
            "gold": "A",
        },
    ]
    metrics = harness.aggregate_metrics(rows)
    assert metrics[CONDITIONS[1]]["budget_hit_rate"] == pytest.approx(1.0)
    assert metrics["paired"]["null_to_answer_rate"] == pytest.approx(1.0)
    assert metrics["weighted_accuracy_delta"] > 0
    assert (
        harness.classify_root_cause(
            metrics,
            thresholds={"null_rescue": 0.75, "accuracy_delta": 0.1},
        )
        == "CONFIRMED_TRUNCATION"
    )


def test_aggregate_metrics_reuses_one_shot_iterable_for_all_metric_sections() -> None:
    harness = entrypoint()
    rows = [
        {
            "id": "null",
            "condition": CONDITIONS[0],
            "category": "science",
            "selection_stratum": "science/saturated_null",
            "selection_weight": 2.0,
            "generated_token_count": 1024,
            "extracted_answer": None,
            "gold": "A",
        },
        {
            "id": "null",
            "condition": CONDITIONS[1],
            "category": "science",
            "selection_stratum": "science/saturated_null",
            "selection_weight": 2.0,
            "generated_token_count": 2048,
            "extracted_answer": "A",
            "gold": "A",
        },
        {
            "id": "correct",
            "condition": CONDITIONS[0],
            "category": "history",
            "selection_stratum": "history/unsaturated_extracted",
            "selection_weight": 1.0,
            "generated_token_count": 100,
            "extracted_answer": "A",
            "gold": "A",
        },
        {
            "id": "correct",
            "condition": CONDITIONS[1],
            "category": "history",
            "selection_stratum": "history/unsaturated_extracted",
            "selection_weight": 1.0,
            "generated_token_count": 2048,
            "extracted_answer": "A",
            "gold": "A",
        },
    ]

    assert harness.aggregate_metrics(iter(rows)) == harness.aggregate_metrics(rows)


def _complete_manifest(harness: ModuleType) -> dict[str, object]:
    manifest = cast(
        dict[str, object],
        harness.build_manifest(
            source_sha256="a" * 64,
            selected_ids=["x"],
            selected_weights={"x": 2.0},
            population_counts={"science/saturated_null": 1},
            selected_strata={"x": "science/saturated_null"},
            model_id=MODEL_ID,
            model_revision=MODEL_REVISION,
            dataset="TIGER-Lab/MMLU-Pro",
            dataset_revision="b189ec765aa7ed75c8acfea42df31fdae71f97be",
            runtime={
                "temperature": 0.0,
                "max_model_len": 8192,
                "budgets": [1024, 2048],
                "gpu_memory_utilization": 0.9,
                "batch_size": 8,
                "packages": {"transformers": "test", "vllm": "test"},
            },
            prompt_hashes={"x": "p"},
            tokenizer_chat_template_sha256="c" * 64,
            output_paths={"responses": "responses.jsonl"},
        ),
    )
    manifest["source_path"] = str(_formal_source_path(TEST_FORMAL_ROOT))
    manifest["formal_run_root"] = str(TEST_FORMAL_ROOT)
    manifest["population_counts"] = {"science/saturated_null": harness.SOURCE_COUNT}
    manifest["population_counts_sha256"] = harness._sha256_value(
        manifest["population_counts"]
    )
    manifest["selected_weights"] = {"x": harness.SOURCE_COUNT}
    manifest["selected_weight_reconstruction"] = harness.SOURCE_COUNT
    evidence = {"source_extracted_answer": None, "source_generated_token_ids": [1, 2]}
    manifest["source_evidence_sha256"] = {"x": harness._sha256_value(evidence)}
    manifest["source_evidence_binding_sha256"] = harness._sha256_value(
        {
            "source_sha256": manifest["source_sha256"],
            "source_evidence_sha256": manifest["source_evidence_sha256"],
        }
    )
    return manifest


def _artifact_manifest(
    harness: ModuleType, tmp_path: Path
) -> tuple[dict[str, object], Path]:
    source = _formal_source_path(tmp_path)
    source.parent.mkdir(parents=True)
    source.write_text(
        json.dumps(
            {
                "id": "x",
                "category": "science",
                "question": "question",
                "options": ["A", "B"],
                "gold": "A",
                "generated_token_count": 2,
                "extracted_answer": None,
                "status": "ok",
                "raw_response": {"outputs": [{"token_ids": [1, 2]}]},
                "model_id": MODEL_ID,
                "benchmark": "mmlu_pro",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = _complete_manifest(harness)
    manifest["formal_run_root"] = str(tmp_path)
    manifest["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest["source_path"] = str(source)
    manifest["selected_weights"] = {"x": harness.SOURCE_COUNT}
    manifest["population_counts"] = {"science/saturated_null": harness.SOURCE_COUNT}
    manifest["population_counts_sha256"] = harness._sha256_value(
        manifest["population_counts"]
    )
    manifest["selected_weight_reconstruction"] = harness.SOURCE_COUNT
    evidence = {"source_extracted_answer": None, "source_generated_token_ids": [1, 2]}
    manifest["source_evidence_sha256"] = {"x": harness._sha256_value(evidence)}
    manifest["source_evidence_binding_sha256"] = harness._sha256_value(
        {
            "source_sha256": manifest["source_sha256"],
            "source_evidence_sha256": manifest["source_evidence_sha256"],
        }
    )
    manifest["prompt_hashes"] = {"x": harness.prompt_hash("actual prompt")}
    manifest["output_paths"] = {
        condition: str(tmp_path / f"{condition}.jsonl") for condition in CONDITIONS
    }
    return manifest, source


def _formal_source_path(root: Path) -> Path:
    return (
        root
        / "checkpoints"
        / "allenai--Olmo-3-7B-Think"
        / "mmlu_pro"
        / "responses.jsonl"
    )


def test_manifest_rejects_source_from_another_formal_run_root_and_accepts_canonical(
    tmp_path: Path,
) -> None:
    harness = entrypoint()
    selected_root = tmp_path / "selected-run"
    other_root = tmp_path / "other-run"
    manifest, source = _artifact_manifest(harness, tmp_path)
    manifest["formal_run_root"] = str(selected_root)

    other_source = _formal_source_path(other_root)
    other_source.parent.mkdir(parents=True)
    other_source.write_bytes(source.read_bytes())
    manifest["source_path"] = str(other_source)
    with pytest.raises(ValueError, match="source.*formal run root"):
        harness.validate_manifest(manifest, formal_run_root=selected_root)

    selected_source = _formal_source_path(selected_root)
    selected_source.parent.mkdir(parents=True)
    selected_source.write_bytes(source.read_bytes())
    canonical_manifest = {**manifest, "source_path": str(selected_source)}
    harness.validate_manifest(canonical_manifest, formal_run_root=selected_root)


def test_manifest_rejects_symlinked_source_path_alias_before_resolving(
    tmp_path: Path,
) -> None:
    harness = entrypoint()
    selected_root = tmp_path / "selected-run"
    manifest, source = _artifact_manifest(harness, tmp_path)
    manifest["formal_run_root"] = str(selected_root)
    selected_source = _formal_source_path(selected_root)
    selected_source.parent.mkdir(parents=True)
    selected_source.write_bytes(source.read_bytes())
    alias_root = tmp_path / "selected-run-alias"
    alias_root.symlink_to(selected_root, target_is_directory=True)
    manifest["source_path"] = str(_formal_source_path(alias_root))

    with pytest.raises(ValueError, match="symlink"):
        harness.validate_manifest(manifest, formal_run_root=selected_root)


def _artifact_rows(harness: ModuleType) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for condition, token_ids in zip(CONDITIONS, ([1, 3], [1, 3, 4])):
        row: dict[str, object] = {
            "id": "x",
            "condition": condition,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "prompt": "actual prompt",
            "category": "science",
            "prompt_hash": harness.prompt_hash("actual prompt"),
            "selection_weight": harness.SOURCE_COUNT,
            "selection_stratum": "science/saturated_null",
            "gold": "A",
            "budget": int(condition.rsplit("_", 1)[1]),
            "generated_token_ids": list(token_ids),
            "token_ids": list(token_ids),
            "generated_token_count": len(token_ids),
            "extracted_answer": None if condition == CONDITIONS[0] else "A",
            "status": "ok",
            "retryable": False,
        }
        row.update(
            {
                "source_extracted_answer": None,
                "source_generated_token_ids": [1, 2],
            }
        )
        rows.append(row)
    return rows


def _write_analysis_artifacts(
    harness: ModuleType,
    root: Path,
    manifest: dict[str, object],
    rows: list[dict[str, object]],
) -> None:
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "prepared.json").write_text(
        json.dumps({"manifest": manifest, "records": rows}), encoding="utf-8"
    )
    for condition in CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition]
        (root / f"{condition}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in selected), encoding="utf-8"
        )


def _patch_fake_vllm(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    original = importlib.import_module

    class FakeVllm:
        class SamplingParams:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs

    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: FakeVllm() if name == "vllm" else original(name),
    )


def _patch_idle_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    import prefix.runner as runner

    class IdleEngine:
        def generate(self, prompts: list[str], params: object) -> list[object]:
            return []

    monkeypatch.setattr(runner, "get_engine", lambda *args, **kwargs: IdleEngine())


def _analysis_rows() -> list[dict[str, object]]:
    return [
        {
            "id": "x",
            "condition": condition,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "prompt_hash": "p",
            "prompt": None,
            "category": "science",
            "selection_weight": 12032.0,
            "selection_stratum": "science/saturated_null",
            "gold": "A",
            "budget": int(condition.rsplit("_", 1)[1]),
            "generated_token_ids": [1, 2],
            "generated_token_count": 2,
            "status": "ok",
            "retryable": False,
            "extracted_answer": "A",
            "source_extracted_answer": None,
            "source_generated_token_ids": [1, 2],
        }
        for condition in CONDITIONS
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "error"),
        ("generated_token_ids", []),
        ("extracted_answer", "not-a-letter"),
    ],
)
def test_analysis_rejects_unsuccessful_or_invalid_token_rows(
    field: str, value: object
) -> None:
    harness = entrypoint()
    rows = _analysis_rows()
    rows[0][field] = value
    with pytest.raises(ValueError, match="analysis|row|token|status"):
        harness.validate_analysis(
            rows,
            _complete_manifest(harness),
            settings={"temperature": 0.0},
            formal_run_root=TEST_FORMAL_ROOT,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("prompt_hash", "wrong"),
        ("selection_weight", 3.0),
        ("selection_stratum", "science/unsaturated_null"),
        ("gold", "B"),
        ("model_revision", "stale"),
        ("budget", 999),
    ],
)
def test_analysis_validates_every_row_against_manifest(
    field: str, value: object
) -> None:
    harness = entrypoint()
    rows = _analysis_rows()
    rows[0][field] = value
    with pytest.raises(ValueError, match="manifest|provenance|stale|mismatch"):
        harness.validate_analysis(
            rows,
            _complete_manifest(harness),
            settings={"temperature": 0.0},
            formal_run_root=TEST_FORMAL_ROOT,
        )


def test_analysis_validates_manifest_data_revision_and_settings() -> None:
    harness = entrypoint()
    manifest = _complete_manifest(harness)
    with pytest.raises(ValueError, match="stale|pinned|settings"):
        harness.validate_analysis(
            _analysis_rows(),
            {**manifest, "dataset_revision": "stale"},
            settings={"temperature": 0.0},
            formal_run_root=TEST_FORMAL_ROOT,
        )
    with pytest.raises(ValueError, match="settings"):
        harness.validate_analysis(
            _analysis_rows(),
            manifest,
            settings={"temperature": 0.1},
            formal_run_root=TEST_FORMAL_ROOT,
        )


def test_classification_requires_source_answer_and_exact_tokens() -> None:
    harness = entrypoint()
    assert (
        harness.classify_root_cause(
            {
                "baseline_reproduction": False,
                "source_reproduction": {
                    "extraction_rate": 1.0,
                    "token_sequence_rate": 0.0,
                },
                "paired": {"null_to_answer_rate": 1.0},
                "weighted_accuracy_delta": 0.5,
            },
            thresholds={"null_rescue": 0.75, "accuracy_delta": 0.1},
        )
        == "MIXED_OR_INDETERMINATE"
    )


def test_generation_error_cannot_preserve_answer_and_must_signal_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = entrypoint()
    output = tmp_path / "current_zero_shot_1024.jsonl"
    prepared = [
        {
            "id": "x",
            "condition": CONDITIONS[0],
            "prompt": "prompt",
            "prompt_hash": "p",
            "selection_weight": 12032.0,
            "selection_stratum": "science/saturated_null",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "budget": 1024,
            "extracted_answer": "A",
            "source_extracted_answer": "A",
        }
    ]
    root = tmp_path
    (root / "prepared.json").write_text(
        json.dumps({"manifest": _complete_manifest(harness), "records": prepared}),
        encoding="utf-8",
    )

    class BrokenEngine:
        def generate(self, prompts: list[str], params: object) -> None:
            raise RuntimeError("generation failed")

    import prefix.runner as runner

    monkeypatch.setattr(runner, "get_engine", lambda *args, **kwargs: BrokenEngine())
    monkeypatch.setattr(
        harness, "_read_json", lambda path: json.loads(path.read_text())
    )
    monkeypatch.setattr(harness, "_check_existing_rows", lambda *args, **kwargs: None)
    monkeypatch.setattr(harness, "validate_source_sha256", lambda *args: None)
    args = argparse.Namespace(
        output_root=root,
        source_responses=None,
        formal_run_root=TEST_FORMAL_ROOT,
        log_file=tmp_path / "run.log",
        gpu_memory_utilization=0.9,
        batch_size=8,
    )
    with pytest.raises(RuntimeError, match="generation failed"):
        harness.generate(args)
    saved = [json.loads(line) for line in output.read_text().splitlines()]
    assert saved[0]["extracted_answer"] is None


def test_prepared_and_standalone_manifests_must_be_identical(tmp_path: Path) -> None:
    harness = entrypoint()
    manifest = _complete_manifest(harness)
    prepared = {"manifest": manifest, "records": []}
    standalone = {**manifest, "source_path": manifest["source_path"]}
    (tmp_path / "prepared.json").write_text(json.dumps(prepared), encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps(standalone), encoding="utf-8")
    assert json.loads((tmp_path / "prepared.json").read_text())[
        "manifest"
    ] == json.loads((tmp_path / "manifest.json").read_text())


def test_jsonl_resume_accepts_only_truncated_final_tail(tmp_path: Path) -> None:
    harness = entrypoint()
    path = tmp_path / "rows.jsonl"
    path.write_text('{"id":"x","status":"ok"}\n{"id":"y"', encoding="utf-8")
    assert list(harness._jsonl(path)) == [{"id": "x", "status": "ok"}]


@pytest.mark.parametrize("reader_name", ["_jsonl", "_read_json"])
@pytest.mark.parametrize("alias_kind", ["file", "parent"])
def test_internal_json_reads_reject_symlinked_files_and_parents(
    tmp_path: Path, reader_name: str, alias_kind: str
) -> None:
    harness = entrypoint()
    extension = ".jsonl" if reader_name == "_jsonl" else ".json"
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    real_path = real_parent / f"artifact{extension}"
    real_path.write_text(
        '{"id":"x"}\n' if extension == ".jsonl" else '{"id":"x"}',
        encoding="utf-8",
    )
    if alias_kind == "file":
        alias = tmp_path / f"alias{extension}"
        alias.symlink_to(real_path)
    else:
        alias_parent = tmp_path / "alias-parent"
        alias_parent.symlink_to(real_parent, target_is_directory=True)
        alias = alias_parent / real_path.name

    with pytest.raises(ValueError, match="symlink"):
        result = getattr(harness, reader_name)(alias)
        if reader_name == "_jsonl":
            list(result)


def test_source_evidence_binding_rejects_zero_selected_id_matches(
    tmp_path: Path,
) -> None:
    harness = entrypoint()
    source = tmp_path / "responses.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "unselected",
                "extracted_answer": None,
                "raw_response": {"outputs": [{"token_ids": [1, 2]}]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "selected_ids": ["selected"],
        "source_evidence_sha256": {"selected": "a" * 64},
    }

    with pytest.raises(ValueError, match="source evidence"):
        harness._validate_source_evidence_binding(source, manifest)


def test_formal_source_path_requires_an_explicit_run_root(tmp_path: Path) -> None:
    harness = entrypoint()
    with pytest.raises(ValueError, match="formal run root"):
        harness.validate_formal_source_path(tmp_path / "source.jsonl", None)


def test_existing_formal_rows_do_not_exempt_basename_source_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = entrypoint()
    monkeypatch.setattr(harness, "validate_manifest", lambda *args, **kwargs: None)
    manifest = _complete_manifest(harness)
    manifest["source_path"] = "source.jsonl"
    row = {
        "id": "x",
        "condition": CONDITIONS[0],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "prompt_hash": "p",
        "selection_weight": harness.SOURCE_COUNT,
        "selection_stratum": "science/saturated_null",
        "status": "error",
        "retryable": True,
    }
    with pytest.raises(ValueError, match="source evidence"):
        harness._check_existing_rows(
            [row], manifest, CONDITIONS[0], formal_run_root=TEST_FORMAL_ROOT
        )


def test_selected_row_source_evidence_requires_nonempty_tokens() -> None:
    harness = entrypoint()
    manifest = _complete_manifest(harness)
    row = {
        "id": "x",
        "source_extracted_answer": None,
        "source_generated_token_ids": [],
    }
    manifest["source_evidence_sha256"] = {
        "x": harness._sha256_value(
            {"source_extracted_answer": None, "source_generated_token_ids": []}
        )
    }
    with pytest.raises(ValueError, match="source evidence|token"):
        harness._validate_row_source_evidence(row, manifest, required=True)


def test_selected_row_source_logprobs_must_align_with_tokens() -> None:
    harness = entrypoint()
    manifest = _complete_manifest(harness)
    row = {
        "id": "x",
        "source_extracted_answer": None,
        "source_generated_token_ids": [1, 2],
        "source_generated_logprobs": [-0.1],
    }
    with pytest.raises(ValueError, match="logprobs|aligned"):
        harness._validate_row_source_evidence(row, manifest, required=True)


def test_manifest_rejects_empty_selected_source_evidence() -> None:
    harness = entrypoint()
    manifest = _complete_manifest(harness)
    empty_evidence = {
        "x": harness._sha256_value(
            {"source_extracted_answer": None, "source_generated_token_ids": []}
        )
    }
    manifest["source_evidence_sha256"] = empty_evidence
    manifest["source_evidence_binding_sha256"] = harness._sha256_value(
        {
            "source_sha256": manifest["source_sha256"],
            "source_evidence_sha256": empty_evidence,
        }
    )
    with pytest.raises(ValueError, match="source evidence|token"):
        harness.validate_manifest(manifest, formal_run_root=TEST_FORMAL_ROOT)


def test_resume_rejects_duplicate_successful_rows() -> None:
    harness = entrypoint()
    row = {
        "id": "x",
        "condition": CONDITIONS[0],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "prompt_hash": "p",
        "selection_weight": 12032.0,
        "selection_stratum": "science/saturated_null",
        "status": "ok",
        "budget": 1024,
        "source_extracted_answer": None,
        "source_generated_token_ids": [1, 2],
    }
    with pytest.raises(ValueError, match="duplicate"):
        harness._check_existing_rows(
            [row, {**row}],
            _complete_manifest(harness),
            CONDITIONS[0],
            formal_run_root=TEST_FORMAL_ROOT,
        )


def test_retry_replacement_rejects_duplicate_new_rows() -> None:
    harness = entrypoint()
    with pytest.raises(ValueError, match="duplicate"):
        harness.merge_generation_rows(
            [{"id": "x", "status": "error", "retryable": True}],
            [
                {"id": "x", "status": "ok"},
                {"id": "x", "status": "ok"},
            ],
        )


def test_generation_runtime_must_match_manifest() -> None:
    harness = entrypoint()
    manifest = _complete_manifest(harness)
    runtime = cast(dict[str, object], manifest["runtime"])
    with pytest.raises(ValueError, match="runtime|settings"):
        harness.validate_manifest(
            {**manifest, "runtime": {**runtime, "gpu_memory_utilization": 0.8}},
            formal_run_root=TEST_FORMAL_ROOT,
        )


def test_formal_source_errors_are_rejected_before_selection() -> None:
    harness = entrypoint()
    rows = source_rows()
    rows[0]["status"] = "error"
    with pytest.raises(ValueError, match="status|successful|source"):
        harness.select_source_rows(
            rows,
            quotas={
                "science": {"saturated_null": 1},
                "history": {"saturated_null": 1},
            },
            expected_categories=("science", "history"),
        )


def test_transition_rates_use_transition_specific_eligible_denominators() -> None:
    harness = entrypoint()
    rows = [
        {
            "id": "a",
            "condition": CONDITIONS[0],
            "extracted_answer": "A",
            "gold": "B",
            "selection_weight": 1.0,
            "category": "science",
            "selection_stratum": "saturated_null",
        },
        {
            "id": "a",
            "condition": CONDITIONS[1],
            "extracted_answer": "A",
            "gold": "A",
            "selection_weight": 1.0,
            "category": "science",
            "selection_stratum": "saturated_null",
        },
        {
            "id": "b",
            "condition": CONDITIONS[0],
            "extracted_answer": "B",
            "gold": "B",
            "selection_weight": 9.0,
            "category": "history",
            "selection_stratum": "saturated_null",
        },
        {
            "id": "b",
            "condition": CONDITIONS[1],
            "extracted_answer": "A",
            "gold": "B",
            "selection_weight": 9.0,
            "category": "history",
            "selection_stratum": "saturated_null",
        },
    ]
    metrics = harness.aggregate_metrics(rows)
    assert metrics["paired"]["incorrect_to_correct_rate"] == pytest.approx(1.0)
    assert metrics["paired"]["correct_to_incorrect_rate"] == pytest.approx(1.0)


def test_per_stratum_metrics_remain_separated_by_condition() -> None:
    harness = entrypoint()
    rows = [
        {
            "id": "x",
            "condition": CONDITIONS[0],
            "extracted_answer": "A",
            "gold": "A",
            "selection_weight": 1.0,
            "category": "science",
            "selection_stratum": "saturated_null",
        },
        {
            "id": "x",
            "condition": CONDITIONS[1],
            "extracted_answer": "B",
            "gold": "A",
            "selection_weight": 1.0,
            "category": "science",
            "selection_stratum": "saturated_null",
        },
    ]
    per_stratum = harness.aggregate_metrics(rows)["per_stratum"]
    assert set(per_stratum) == {
        f"{CONDITIONS[0]}/science/saturated_null",
        f"{CONDITIONS[1]}/science/saturated_null",
    }


def test_source_selection_checkpoint_resumes_completed_rows_and_validates_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = entrypoint()
    source = tmp_path / "source.jsonl"
    rows = source_rows()
    write_jsonl(source, rows)
    checkpoint = tmp_path / "prepare.checkpoint.json"
    config = {"seed": 17, "expected_categories": ["science", "history"]}

    first = harness.select_source_rows(
        iter(rows),
        quotas={
            "science": {"saturated_null": 1, "unsaturated_extracted": 1},
            "history": {"saturated_null": 1, "unsaturated_extracted": 1},
        },
        expected_categories=("science", "history"),
        checkpoint_path=checkpoint,
        checkpoint_config=config,
        source_path=source,
    )
    assert first["source_count"] == 4
    saved = [
        json.loads(line) for line in checkpoint.read_text(encoding="utf-8").splitlines()
    ]
    assert saved[0]["event"] == "header"
    assert saved[-1]["event"] == "complete"
    assert saved[-1]["processed_rows"] == 4

    original = harness._validate_formal_provenance
    calls = 0

    def fail_if_recomputed(row: object) -> None:
        nonlocal calls
        calls += 1
        original(cast(dict[str, object], row))

    monkeypatch.setattr(harness, "_validate_formal_provenance", fail_if_recomputed)
    resumed = harness.select_source_rows(
        iter(rows),
        quotas={
            "science": {"saturated_null": 1, "unsaturated_extracted": 1},
            "history": {"saturated_null": 1, "unsaturated_extracted": 1},
        },
        expected_categories=("science", "history"),
        checkpoint_path=checkpoint,
        checkpoint_config=config,
        source_path=source,
    )
    assert resumed["records"] == first["records"]
    assert calls == 0

    with pytest.raises(ValueError, match="checkpoint|digest"):
        harness.select_source_rows(
            iter(rows),
            quotas={
                "science": {"saturated_null": 1, "unsaturated_extracted": 1},
                "history": {"saturated_null": 1, "unsaturated_extracted": 1},
            },
            expected_categories=("science", "history"),
            checkpoint_path=checkpoint,
            checkpoint_config={"seed": 99},
            source_path=source,
        )


def test_source_selection_failure_leaves_checkpoint_for_partial_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = entrypoint()
    source = tmp_path / "source.jsonl"
    rows = source_rows()
    write_jsonl(source, rows)
    checkpoint = tmp_path / "prepare.checkpoint.json"
    kwargs = {
        "quotas": {
            "science": {"saturated_null": 1, "unsaturated_extracted": 1},
            "history": {"saturated_null": 1, "unsaturated_extracted": 1},
        },
        "expected_categories": ("science", "history"),
        "checkpoint_path": checkpoint,
        "checkpoint_config": {"seed": 17},
        "source_path": source,
    }
    original_append = harness._append_jsonl
    writes = 0

    def fail_after_first(path: Path, value: object) -> None:
        nonlocal writes
        if path == checkpoint:
            writes += 1
            if writes == 3:
                raise RuntimeError("simulated preemption")
        original_append(path, value)

    monkeypatch.setattr(harness, "_append_jsonl", fail_after_first)
    with pytest.raises(RuntimeError, match="preemption"):
        harness.select_source_rows(iter(rows), **kwargs)
    partial = [
        json.loads(line) for line in checkpoint.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event"] for event in partial] == ["header", "row"]
    with checkpoint.open("a", encoding="utf-8") as stream:
        stream.write('{"event":"row"')

    monkeypatch.setattr(harness, "_append_jsonl", original_append)
    calls = 0
    original_validate = harness._validate_formal_provenance

    def count_remaining(row: object) -> None:
        nonlocal calls
        calls += 1
        original_validate(cast(dict[str, object], row))

    monkeypatch.setattr(harness, "_validate_formal_provenance", count_remaining)
    result = harness.select_source_rows(iter(rows), **kwargs)
    assert result["source_count"] == 4
    assert calls == 3
    complete = [
        json.loads(line) for line in checkpoint.read_text(encoding="utf-8").splitlines()
    ]
    assert complete[-1]["event"] == "complete"
    assert complete[-1]["processed_rows"] == 4

    monkeypatch.setattr(harness, "_validate_formal_provenance", original_validate)
    restarted = harness.select_source_rows(iter(rows), **kwargs)
    assert restarted == result


def test_source_selection_checkpoint_writes_linear_event_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = entrypoint()
    source = tmp_path / "large-source.jsonl"
    rows = cast(
        list[SourceRow],
        [
            {
                "id": f"row-{index}",
                "category": "science",
                "question": "question",
                "options": ["A", "B"],
                "gold": "A",
                "generated_token_count": 1024,
                "extracted_answer": None,
                "status": "ok",
            }
            for index in range(200)
        ],
    )
    write_jsonl(source, rows)
    checkpoint = tmp_path / "large.checkpoint.jsonl"
    original_append = harness._append_jsonl
    serialized_bytes: list[int] = []

    def instrument(path: Path, events: list[dict[str, object]]) -> None:
        serialized_bytes.append(
            sum(len(json.dumps(event, sort_keys=True)) for event in events)
        )
        original_append(path, events)

    monkeypatch.setattr(harness, "_append_jsonl", instrument)
    result = harness.select_source_rows(
        iter(rows),
        quotas={"science": {"saturated_null": 1}},
        expected_categories=("science",),
        expected_count=len(rows),
        checkpoint_path=checkpoint,
        checkpoint_config={"seed": 17},
        source_path=source,
    )
    assert result["source_count"] == len(rows)
    assert len(serialized_bytes) == len(rows) + 2
    assert sum(serialized_bytes) < len(rows) * 2_000


def test_checkpoint_replay_compares_normalized_source_prefix(tmp_path: Path) -> None:
    harness = entrypoint()
    source = tmp_path / "source.jsonl"
    rows = source_rows()
    write_jsonl(source, rows)
    checkpoint = tmp_path / "prepare.checkpoint.jsonl"
    kwargs = {
        "quotas": {
            "science": {"saturated_null": 1, "unsaturated_extracted": 1},
            "history": {"saturated_null": 1, "unsaturated_extracted": 1},
        },
        "expected_categories": ("science", "history"),
        "checkpoint_path": checkpoint,
        "checkpoint_config": {"seed": 17},
        "source_path": source,
    }
    harness.select_source_rows(iter(rows), **kwargs)
    altered = [dict(row) for row in rows]
    altered[0]["question"] = "tampered prefix"
    with pytest.raises(ValueError, match="row|source|checkpoint|digest"):
        harness.select_source_rows(iter(altered), **kwargs)


def test_checkpoint_requires_candidate_key_for_positive_quota(tmp_path: Path) -> None:
    harness = entrypoint()
    source = tmp_path / "source.jsonl"
    rows = source_rows()
    write_jsonl(source, rows)
    checkpoint = tmp_path / "prepare.checkpoint.jsonl"
    kwargs = {
        "quotas": {
            "science": {"saturated_null": 1, "unsaturated_extracted": 1},
            "history": {"saturated_null": 1, "unsaturated_extracted": 1},
        },
        "expected_categories": ("science", "history"),
        "checkpoint_path": checkpoint,
        "checkpoint_config": {"seed": 17},
        "source_path": source,
    }
    harness.select_source_rows(iter(rows), **kwargs)
    events = [json.loads(line) for line in checkpoint.read_text().splitlines()]
    events[1]["candidate_key"] = None
    checkpoint.write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="candidate"):
        harness.select_source_rows(iter(rows), **kwargs)


def test_checkpoint_rejects_events_after_terminal_complete(tmp_path: Path) -> None:
    harness = entrypoint()
    source = tmp_path / "source.jsonl"
    rows = source_rows()
    write_jsonl(source, rows)
    checkpoint = tmp_path / "prepare.checkpoint.jsonl"
    kwargs = {
        "quotas": {
            "science": {"saturated_null": 1, "unsaturated_extracted": 1},
            "history": {"saturated_null": 1, "unsaturated_extracted": 1},
        },
        "expected_categories": ("science", "history"),
        "checkpoint_path": checkpoint,
        "checkpoint_config": {"seed": 17},
        "source_path": source,
    }
    harness.select_source_rows(iter(rows), **kwargs)
    events = [json.loads(line) for line in checkpoint.read_text().splitlines()]
    events.append(dict(events[1], row_index=len(rows)))
    checkpoint.write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="terminal|complete|event"):
        harness.select_source_rows(iter(rows), **kwargs)


def _prepare_test_args(tmp_path: Path) -> argparse.Namespace:
    source = _formal_source_path(tmp_path)
    source.parent.mkdir(parents=True)
    source.write_text("source\n", encoding="utf-8")
    cache = tmp_path / "cache"
    cache.mkdir()
    root = tmp_path / "output"
    return argparse.Namespace(
        source_responses=source,
        formal_run_root=tmp_path,
        output_root=root,
        cache_root=cache,
        seed=17,
        batch_size=8,
        max_model_len=8192,
        gpu_memory_utilization=0.9,
    )


def _prepare_test_selection() -> dict[str, object]:
    row = {
        "id": "mmlu-pro-0",
        "category": "science",
        "question": "What is two plus two?",
        "options": ["3", "4", "5"],
        "gold": "B",
        "generated_token_count": 1024,
        "extracted_answer": None,
        "status": "ok",
        "weight": 2.0,
        "selection_stratum": "science/saturated_null",
        "raw_response": {"outputs": [{"token_ids": [1, 2]}]},
        "metadata": {"source_extracted_answer": None},
    }
    return {
        "records": [row],
        "source_sha256": "a" * 64,
        "population_counts": {"science/saturated_null": 1},
        "shortfalls": {},
        "source_count": 1,
    }


def test_prepare_uses_append_only_journal_and_replays_without_tokenization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = entrypoint()
    args = _prepare_test_args(tmp_path)
    selection = _prepare_test_selection()
    monkeypatch.setattr(harness, "prepare_source_selection", lambda *a, **k: selection)
    import prefix.data as data

    monkeypatch.setattr(
        data,
        "load_mmlu_pro",
        lambda *a, **k: [
            {
                "id": "mmlu-pro-0",
                "question": "What is two plus two?",
                "options": ["3", "4", "5"],
                "category": "science",
                "answer_letter": "B",
            }
        ],
    )

    class FakeTokenizer:
        chat_template = "template-v1"

        def encode(self, prompt: str, *, add_special_tokens: bool) -> list[int]:
            return [1, 2]

    monkeypatch.setattr(harness, "_load_tokenizer", lambda: FakeTokenizer())
    render_calls = 0

    def render(tokenizer: object, row: object) -> str:
        nonlocal render_calls
        render_calls += 1
        return "rendered prompt"

    monkeypatch.setattr(harness, "_current_prompt", render)
    harness._prepare_impl(args)
    journal = args.output_root / "prepared.records.jsonl"
    assert journal.exists()
    assert not (args.output_root / "prepare.records.json").exists()
    events = [json.loads(line) for line in journal.read_text().splitlines()]
    assert [event["event"] for event in events] == ["header", "record", "complete"]
    assert set(events[1]["conditions"]) == set(CONDITIONS)
    assert events[1]["prompt_hash"] == harness.prompt_hash("rendered prompt")
    assert render_calls == 1

    monkeypatch.setattr(
        harness,
        "_current_prompt",
        lambda *a, **k: pytest.fail("completed unit was rerendered"),
    )
    harness._prepare_impl(args)
    assert render_calls == 1


def test_prepare_journal_rejects_runtime_config_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = entrypoint()
    args = _prepare_test_args(tmp_path)
    selection = _prepare_test_selection()
    monkeypatch.setattr(harness, "prepare_source_selection", lambda *a, **k: selection)
    import prefix.data as data

    monkeypatch.setattr(
        data,
        "load_mmlu_pro",
        lambda *a, **k: [
            {
                "id": "mmlu-pro-0",
                "question": "What is two plus two?",
                "options": ["3", "4", "5"],
                "category": "science",
                "answer_letter": "B",
            }
        ],
    )
    monkeypatch.setattr(
        harness,
        "_load_tokenizer",
        lambda: type(
            "Tokenizer",
            (),
            {
                "chat_template": "template-v1",
                "encode": lambda self, prompt, add_special_tokens: [1, 2],
            },
        )(),
    )
    monkeypatch.setattr(harness, "_current_prompt", lambda *a: "rendered prompt")
    harness._prepare_impl(args)
    args.batch_size = 4
    with pytest.raises(ValueError, match="journal|digest|drift|config"):
        harness._prepare_impl(args)


def test_analyze_artifacts_require_exact_baseline_source_tokens_for_classification(
    tmp_path: Path,
) -> None:
    harness = entrypoint()
    manifest, _ = _artifact_manifest(harness, tmp_path)
    rows = _artifact_rows(harness)
    _write_analysis_artifacts(harness, tmp_path, manifest, rows)
    harness.analyze(
        argparse.Namespace(
            output_root=tmp_path,
            source_responses=None,
            formal_run_root=tmp_path,
            null_rescue_threshold=0.75,
            accuracy_delta_threshold=0.1,
        )
    )
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["source_reproduction"]["extraction_rate"] == 1.0
    assert metrics["source_reproduction"]["token_sequence_rate"] == 0.0
    assert metrics["classification"] == "MIXED_OR_INDETERMINATE"


@pytest.mark.parametrize(
    ("field", "value"),
    [("prompt", "tampered prompt"), ("generated_token_count", 999)],
)
def test_analyze_artifacts_validate_prompt_content_and_token_count(
    tmp_path: Path, field: str, value: object
) -> None:
    harness = entrypoint()
    manifest, _ = _artifact_manifest(harness, tmp_path)
    rows = _artifact_rows(harness)
    rows[0][field] = value
    _write_analysis_artifacts(harness, tmp_path, manifest, rows)
    with pytest.raises(ValueError, match="prompt|token|count"):
        harness.analyze(
            argparse.Namespace(
                output_root=tmp_path,
                source_responses=None,
                formal_run_root=tmp_path,
                null_rescue_threshold=0.75,
                accuracy_delta_threshold=0.1,
            )
        )


def test_generate_rejects_embedded_vs_standalone_manifest_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = entrypoint()
    manifest, _ = _artifact_manifest(harness, tmp_path)
    (tmp_path / "prepared.json").write_text(
        json.dumps({"manifest": manifest, "records": []}), encoding="utf-8"
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps({**manifest, "dataset_revision": "stale"}), encoding="utf-8"
    )
    _patch_fake_vllm(monkeypatch)
    _patch_idle_engine(monkeypatch)
    with pytest.raises(ValueError, match="manifest|mismatch|stale"):
        harness.generate(
            argparse.Namespace(
                output_root=tmp_path,
                source_responses=None,
                formal_run_root=tmp_path,
                log_file=tmp_path / "generate.log",
                gpu_memory_utilization=0.9,
                batch_size=8,
            )
        )


@pytest.mark.parametrize(("batch_size", "gpu_memory_utilization"), [(4, 0.9), (8, 0.8)])
def test_generate_cli_runtime_must_match_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    batch_size: int,
    gpu_memory_utilization: float,
) -> None:
    harness = entrypoint()
    manifest, _ = _artifact_manifest(harness, tmp_path)
    (tmp_path / "prepared.json").write_text(
        json.dumps({"manifest": manifest, "records": []}), encoding="utf-8"
    )
    _patch_fake_vllm(monkeypatch)
    _patch_idle_engine(monkeypatch)
    with pytest.raises(ValueError, match="runtime|settings|batch|utilization"):
        harness.generate(
            argparse.Namespace(
                output_root=tmp_path,
                source_responses=None,
                formal_run_root=tmp_path,
                log_file=tmp_path / "generate.log",
                gpu_memory_utilization=gpu_memory_utilization,
                batch_size=batch_size,
            )
        )


@pytest.mark.parametrize(("field", "value"), [("gold", "B"), ("category", "tampered")])
def test_analysis_rejects_gold_and_category_tampering(
    field: str, value: object
) -> None:
    harness = entrypoint()
    rows = _analysis_rows()
    for row in rows:
        row[field] = value
    with pytest.raises(ValueError, match="gold|category|manifest|mismatch"):
        harness.validate_analysis(
            rows,
            _complete_manifest(harness),
            settings={"temperature": 0.0},
            formal_run_root=TEST_FORMAL_ROOT,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "schema_version",
        "source_path",
        "zero_weight",
        "weight_coverage",
        "prompt_coverage",
        "output_coverage",
        "package_coverage",
    ],
)
def test_manifest_requires_exact_keys_and_selection_provenance(mutation: str) -> None:
    harness = entrypoint()
    manifest = _complete_manifest(harness)
    if mutation == "schema_version":
        manifest.pop("schema_version")
    elif mutation == "source_path":
        manifest.pop("source_path")
    elif mutation == "zero_weight":
        manifest["selected_weights"] = {"x": 0.0}
    elif mutation == "weight_coverage":
        manifest["selected_weights"] = {}
    elif mutation == "prompt_coverage":
        manifest["prompt_hashes"] = {}
    elif mutation == "output_coverage":
        manifest["output_paths"] = {}
    else:
        cast(dict[str, object], manifest["runtime"]).pop("packages")
    with pytest.raises(ValueError, match="manifest|provenance|weight|coverage"):
        harness.validate_manifest(manifest, formal_run_root=TEST_FORMAL_ROOT)


def test_unicode_stable_id_matches_no_steering_production_algorithm() -> None:
    harness = entrypoint()
    spec = importlib.util.spec_from_file_location(
        "run_no_steering_for_stable_id", SCRIPT.parent / "run_no_steering.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    record = {"question": "café — π", "options": ["雪", "é"], "category": "科学"}
    assert harness._production_id("mmlu_pro", 7, record) == module._stable_record_id(
        "mmlu_pro", 7, record
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_id", "wrong/model"),
        ("model_revision", "stale"),
        ("benchmark", "wrong_benchmark"),
        ("budget", 2048),
        ("temperature", 0.7),
    ],
)
def test_formal_source_rows_validate_model_revision_benchmark_budget_temperature(
    field: str, value: object
) -> None:
    harness = entrypoint()
    rows = [dict(row) for row in source_rows()]
    for row in rows:
        row["metadata"] = {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "benchmark": "mmlu_pro",
            "budget": 1024,
            "temperature": 0.0,
        }
    cast(dict[str, object], rows[0]["metadata"])[field] = value
    with pytest.raises(ValueError, match="provenance|benchmark|budget|temperature"):
        harness.select_source_rows(
            rows,
            quotas={
                "science": {"saturated_null": 1},
                "history": {"saturated_null": 1},
            },
            expected_categories=("science", "history"),
        )


def test_weighted_selection_denominators_reconstruct_full_population() -> None:
    harness = entrypoint()
    categories = tuple(harness.EXPECTED_CATEGORIES)
    rows = [
        {
            "id": f"row-{index}",
            "category": categories[index % len(categories)],
            "question": "question",
            "options": ["A", "B"],
            "gold": "A",
            "generated_token_count": 1024,
            "extracted_answer": None,
            "status": "ok",
        }
        for index in range(12032)
    ]
    result = harness.select_source_rows(
        rows,
        quotas={category: {"saturated_null": 1} for category in categories},
        expected_categories=categories,
        expected_count=12032,
    )
    assert sum(result["population_counts"].values()) == 12032
    assert sum(row["weight"] for row in result["records"]) == pytest.approx(12032)


def _production_source_row() -> dict[str, object]:
    return {
        "id": "mmlu-pro-0",
        "model_id": MODEL_ID,
        "benchmark": "mmlu_pro",
        "category": "science",
        "question": "What is two plus two?",
        "options": ["3", "4", "5"],
        "gold": "B",
        "generated_token_count": 1024,
        "extracted_answer": None,
        "status": "ok",
        "metadata": {
            "provenance": {
                "revision": MODEL_REVISION,
                "sampling": {"max_tokens": 1024, "temperature": 0.0},
            }
        },
        "raw_response": {
            "outputs": [{"token_ids": [11, 12, 13]}],
        },
    }


def _exact_producer_source_row() -> dict[str, object]:
    token_ids = list(range(1024))
    return {
        "id": "mmlu-pro-exact-0",
        "model_id": MODEL_ID,
        "benchmark": "mmlu_pro",
        "category": "science",
        "question": "What is two plus two?",
        "options": ["3", "4", "5"],
        "gold": "B",
        "generated_token_count": len(token_ids),
        "extracted_answer": None,
        "status": "ok",
        "metadata": {
            "provenance": {
                "model_id": MODEL_ID,
                "revision": MODEL_REVISION,
                "steering": False,
                "quantization": False,
                "sampling": {"max_tokens": 1024, "temperature": 0.0},
            }
        },
        "raw_response": {"outputs": [{"token_ids": token_ids}]},
    }


def test_exact_producer_row_schema_is_selectable() -> None:
    harness = entrypoint()
    result = harness.select_source_rows(
        [_exact_producer_source_row()],
        quotas={"science": {"saturated_null": 1}},
        expected_categories=("science",),
    )
    assert [row["id"] for row in result["records"]] == ["mmlu-pro-exact-0"]


def test_runtime_provenance_booleans_match_no_steering_producer() -> None:
    harness = entrypoint()
    row = _exact_producer_source_row()
    provenance = cast(dict[str, object], row["metadata"])["provenance"]
    assert cast(dict[str, object], provenance)["steering"] is False
    assert cast(dict[str, object], provenance)["quantization"] is False
    result = harness.select_source_rows(
        [row],
        quotas={"science": {"saturated_null": 1}},
        expected_categories=("science",),
    )
    assert result["source_count"] == 1


@pytest.mark.parametrize(
    "mutation",
    [
        ("model_id",),
        ("benchmark",),
        ("metadata", "provenance", "model_id"),
        ("metadata", "provenance", "revision"),
        ("metadata", "provenance", "steering"),
        ("metadata", "provenance", "quantization"),
        ("metadata", "provenance", "sampling", "max_tokens"),
        ("metadata", "provenance", "sampling", "temperature"),
    ],
)
def test_source_rows_without_exact_producer_markers_cannot_bypass_selection(
    mutation: tuple[str, ...],
) -> None:
    harness = entrypoint()
    row = _exact_producer_source_row()
    target: dict[str, object] = row
    for key in mutation[:-1]:
        target = cast(dict[str, object], target[key])
    target.pop(mutation[-1], None)
    with pytest.raises(ValueError, match="provenance|model|benchmark|sampling"):
        harness.select_source_rows(
            [row],
            quotas={"science": {"saturated_null": 1}},
            expected_categories=("science",),
        )


def test_legacy_source_row_without_producer_markers_is_not_selectable() -> None:
    harness = entrypoint()
    row = dict(source_rows()[0])
    with pytest.raises(ValueError, match="provenance|producer|formal"):
        harness.select_source_rows(
            [row],
            quotas={"science": {"saturated_null": 1}},
            expected_categories=("science",),
        )


@pytest.mark.parametrize("field", ["prompt", "category"])
def test_analysis_requires_prompt_and_category_row_fields(field: str) -> None:
    harness = entrypoint()
    row_set = _analysis_rows()
    row_set[0].pop(field, None)
    with pytest.raises(ValueError, match="prompt|category|row"):
        harness.validate_analysis(
            row_set,
            _complete_manifest(harness),
            settings={"temperature": 0.0},
            formal_run_root=TEST_FORMAL_ROOT,
        )


def test_manifest_requires_source_evidence_hash_for_exact_selected_ids() -> None:
    harness = entrypoint()
    manifest = _complete_manifest(harness)
    manifest["source_evidence_sha256"] = {
        "x": harness._sha256_value(
            {"source_extracted_answer": None, "source_generated_token_ids": [1, 2]}
        )
    }
    harness.validate_manifest(manifest, formal_run_root=TEST_FORMAL_ROOT)
    for evidence in ({}, {"other": "a" * 64}, {"x": "b" * 64, "other": "c" * 64}):
        with pytest.raises(ValueError, match="source|evidence|coverage|manifest"):
            harness.validate_manifest(
                {**manifest, "source_evidence_sha256": evidence},
                formal_run_root=TEST_FORMAL_ROOT,
            )


def test_synchronized_source_output_manifest_evidence_tampering_is_rejected(
    tmp_path: Path,
) -> None:
    harness = entrypoint()
    source = tmp_path / "source.jsonl"
    altered_ids = list(range(1024))
    source_row = _exact_producer_source_row()
    source_row["id"] = "x"
    source_row["raw_response"] = {"outputs": [{"token_ids": altered_ids}]}
    source.write_text(json.dumps(source_row) + "\n", encoding="utf-8")
    manifest, _ = _artifact_manifest(harness, tmp_path)
    manifest["source_path"] = str(source)
    manifest["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest["source_evidence_sha256"] = {
        "x": harness._sha256_value(
            {"source_extracted_answer": None, "source_generated_token_ids": altered_ids}
        )
    }
    rows = _artifact_rows(harness)
    rows[0]["source_generated_token_ids"] = altered_ids
    _write_analysis_artifacts(harness, tmp_path, manifest, rows)
    (tmp_path / "prepared.json").write_text(
        json.dumps({"manifest": manifest, "records": rows}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="source|evidence|digest|token"):
        harness.analyze(
            argparse.Namespace(
                output_root=tmp_path,
                source_responses=None,
                formal_run_root=tmp_path,
                null_rescue_threshold=0.75,
                accuracy_delta_threshold=0.1,
            )
        )


@pytest.mark.parametrize(
    "field", ["python", "packages", "harness_sha256", "uv_lock_sha256"]
)
def test_manifest_requires_current_runtime_fingerprint_fields(field: str) -> None:
    harness = entrypoint()
    manifest = _complete_manifest(harness)
    runtime = cast(dict[str, object], manifest["runtime"])
    runtime.pop(field)
    runtime["fingerprint_sha256"] = harness._sha256_value(
        {key: value for key, value in runtime.items() if key != "fingerprint_sha256"}
    )
    with pytest.raises(
        ValueError, match="runtime|fingerprint|package|harness|lock|python"
    ):
        harness.validate_manifest(manifest, formal_run_root=TEST_FORMAL_ROOT)


def test_manifest_runtime_fingerprint_is_compared_to_current_execution() -> None:
    harness = entrypoint()
    manifest = _complete_manifest(harness)
    runtime = cast(dict[str, object], manifest["runtime"])
    runtime["python"] = "0.0.0"
    runtime["fingerprint_sha256"] = harness._sha256_value(
        {key: value for key, value in runtime.items() if key != "fingerprint_sha256"}
    )
    with pytest.raises(ValueError, match="runtime|python|fingerprint|stale"):
        harness.validate_manifest(manifest, formal_run_root=TEST_FORMAL_ROOT)


@pytest.mark.parametrize(
    "field",
    [
        "prompt_hashes",
        "selected_categories",
        "selected_gold",
        "selected_weights",
        "selected_strata",
    ],
)
def test_manifest_selection_maps_require_exact_selected_id_keys(field: str) -> None:
    harness = entrypoint()
    manifest = _complete_manifest(harness)
    values = cast(dict[str, object], manifest[field])
    values["extra"] = next(iter(values.values()))
    with pytest.raises(ValueError, match="coverage|keys|manifest|provenance"):
        harness.validate_manifest(manifest, formal_run_root=TEST_FORMAL_ROOT)


def test_manifest_requires_exactly_two_condition_output_keys() -> None:
    harness = entrypoint()
    manifest = _complete_manifest(harness)
    manifest["output_paths"] = {"unexpected": "x"}
    with pytest.raises(ValueError, match="output|condition|coverage|manifest"):
        harness.validate_manifest(manifest, formal_run_root=TEST_FORMAL_ROOT)


def test_manifest_population_and_weights_must_reconstruct_full_source() -> None:
    harness = entrypoint()
    manifest = _complete_manifest(harness)
    manifest["population_counts"] = {"science/saturated_null": 1}
    manifest["population_counts_sha256"] = harness._sha256_value(
        manifest["population_counts"]
    )
    with pytest.raises(ValueError, match="population|12032|denominator"):
        harness.validate_manifest(manifest, formal_run_root=TEST_FORMAL_ROOT)

    manifest = _complete_manifest(harness)
    manifest["population_counts"] = {"science/saturated_null": 12032}
    manifest["population_counts_sha256"] = harness._sha256_value(
        manifest["population_counts"]
    )
    manifest["selected_weights"] = {"x": 1.0, "y": 12031.0}
    with pytest.raises(ValueError, match="weight|12032|denominator"):
        harness.validate_manifest(manifest, formal_run_root=TEST_FORMAL_ROOT)


def test_generate_requires_both_twin_artifacts_even_without_output_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = entrypoint()
    manifest, _ = _artifact_manifest(harness, tmp_path)
    manifest["output_paths"] = {"unexpected": "x"}
    (tmp_path / "prepared.json").write_text(
        json.dumps({"manifest": manifest, "records": []}), encoding="utf-8"
    )
    _patch_fake_vllm(monkeypatch)
    _patch_idle_engine(monkeypatch)
    with pytest.raises(
        (FileNotFoundError, ValueError), match="manifest|prepared|output"
    ):
        harness.generate(
            argparse.Namespace(
                output_root=tmp_path,
                source_responses=None,
                formal_run_root=tmp_path,
                log_file=tmp_path / "generate.log",
                gpu_memory_utilization=0.9,
                batch_size=8,
            )
        )


@pytest.mark.parametrize(
    "raw_response",
    [None, {"outputs": [{"token_ids": [11]}]}],
)
def test_selection_requires_full_source_token_evidence(
    raw_response: object,
) -> None:
    harness = entrypoint()
    row = _production_source_row()
    row["raw_response"] = raw_response
    with pytest.raises(ValueError, match="token|evidence|coverage"):
        harness.select_source_rows(
            [row],
            quotas={"science": {"saturated_null": 1}},
            expected_categories=("science",),
        )


@pytest.mark.parametrize(
    ("location", "value"),
    [
        (("model_id",), "wrong/model"),
        (("benchmark",), "wrong_benchmark"),
        (("metadata", "provenance", "revision"), "stale"),
        (("metadata", "provenance", "sampling", "max_tokens"), 2048),
        (("metadata", "provenance", "sampling", "temperature"), 0.7),
    ],
)
def test_selection_rejects_each_production_provenance_drift(
    location: tuple[str, ...], value: object
) -> None:
    harness = entrypoint()
    row = _production_source_row()
    target = row
    for key in location[:-1]:
        target = cast(dict[str, object], target[key])
    target[location[-1]] = value
    with pytest.raises(ValueError, match="provenance|model|benchmark|sampling"):
        harness.select_source_rows(
            [row],
            quotas={"science": {"saturated_null": 1}},
            expected_categories=("science",),
        )


@pytest.mark.parametrize("missing", ["manifest.json"])
def test_generate_requires_both_direct_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    harness = entrypoint()
    manifest, _ = _artifact_manifest(harness, tmp_path)
    (tmp_path / "prepared.json").write_text(
        json.dumps({"manifest": manifest, "records": []}), encoding="utf-8"
    )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / missing).unlink()
    _patch_fake_vllm(monkeypatch)
    _patch_idle_engine(monkeypatch)
    with pytest.raises((FileNotFoundError, ValueError), match="manifest|prepared"):
        harness.generate(
            argparse.Namespace(
                output_root=tmp_path,
                source_responses=None,
                formal_run_root=tmp_path,
                log_file=tmp_path / "generate.log",
                gpu_memory_utilization=0.9,
                batch_size=8,
            )
        )


@pytest.mark.parametrize("missing", ["prepared.json"])
def test_analyze_requires_both_direct_artifacts(tmp_path: Path, missing: str) -> None:
    harness = entrypoint()
    manifest, _ = _artifact_manifest(harness, tmp_path)
    rows = _artifact_rows(harness)
    _write_analysis_artifacts(harness, tmp_path, manifest, rows)
    (tmp_path / "prepared.json").write_text(
        json.dumps({"manifest": manifest, "records": rows}), encoding="utf-8"
    )
    (tmp_path / missing).unlink()
    with pytest.raises((FileNotFoundError, ValueError), match="manifest|prepared"):
        harness.analyze(
            argparse.Namespace(
                output_root=tmp_path,
                source_responses=None,
                formal_run_root=tmp_path,
                null_rescue_threshold=0.75,
                accuracy_delta_threshold=0.1,
            )
        )


@pytest.mark.parametrize("mutation", ["output", "category"])
def test_manifest_requires_exact_prompt_output_and_category_coverage(
    mutation: str,
) -> None:
    harness = entrypoint()
    manifest = _complete_manifest(harness)
    if mutation == "output":
        manifest["output_paths"] = {CONDITIONS[0]: "baseline.jsonl"}
    else:
        manifest["selected_categories"] = {"x": "not-a-production-category"}
    with pytest.raises(ValueError, match="coverage|category|manifest|provenance"):
        harness.validate_manifest(manifest, formal_run_root=TEST_FORMAL_ROOT)


def test_manifest_requires_population_denominator_of_12032() -> None:
    harness = entrypoint()
    manifest = _complete_manifest(harness)
    manifest["population_counts"] = {"science/saturated_null": 12031}
    manifest["population_counts_sha256"] = harness._sha256_value(
        manifest["population_counts"]
    )
    with pytest.raises(ValueError, match="population|12032|denominator"):
        harness.validate_manifest(manifest, formal_run_root=TEST_FORMAL_ROOT)


def test_weighted_condition_denominators_are_12032() -> None:
    harness = entrypoint()
    metrics = harness.aggregate_metrics(_analysis_rows())
    for condition in CONDITIONS:
        assert metrics[condition]["weighted_denominator"] == pytest.approx(12032)


def test_analyze_rejects_synchronized_source_evidence_tampering(tmp_path: Path) -> None:
    harness = entrypoint()
    manifest, source = _artifact_manifest(harness, tmp_path)
    rows = _artifact_rows(harness)
    _write_analysis_artifacts(harness, tmp_path, manifest, rows)
    manifest["source_sha256"] = hashlib.sha256(
        b'{"id":"formal-x","extracted_answer":"B","token_ids":[9]}\n'
    ).hexdigest()
    rows[0]["source_extracted_answer"] = "B"
    rows[0]["source_generated_token_ids"] = [9]
    source.write_text(
        '{"id":"formal-x","extracted_answer":"B","token_ids":[9]}\n',
        encoding="utf-8",
    )
    _write_analysis_artifacts(harness, tmp_path, manifest, rows)
    (tmp_path / "prepared.json").write_text(
        json.dumps({"manifest": manifest, "records": rows}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="source|evidence|digest|token"):
        harness.analyze(
            argparse.Namespace(
                output_root=tmp_path,
                source_responses=None,
                formal_run_root=tmp_path,
                null_rescue_threshold=0.75,
                accuracy_delta_threshold=0.1,
            )
        )


@pytest.mark.parametrize("evidence", ["missing", "zero"])
def test_analyze_rejects_zero_or_partial_baseline_source_token_coverage(
    tmp_path: Path, evidence: str
) -> None:
    harness = entrypoint()
    manifest, _ = _artifact_manifest(harness, tmp_path)
    rows = _artifact_rows(harness)
    if evidence == "missing":
        rows[0].pop("source_generated_token_ids")
    else:
        rows[0]["source_generated_token_ids"] = []
    _write_analysis_artifacts(harness, tmp_path, manifest, rows)
    (tmp_path / "prepared.json").write_text(
        json.dumps({"manifest": manifest, "records": rows}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="source|token|coverage|evidence"):
        harness.analyze(
            argparse.Namespace(
                output_root=tmp_path,
                source_responses=None,
                formal_run_root=tmp_path,
                null_rescue_threshold=0.75,
                accuracy_delta_threshold=0.1,
            )
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("python", "0.0.0"),
        ("packages", {"transformers": "drift", "vllm": "drift"}),
        ("harness_sha256", "0" * 64),
        ("uv_lock_sha256", "1" * 64),
    ],
)
def test_manifest_rejects_current_execution_provenance_drift(
    field: str, value: object
) -> None:
    harness = entrypoint()
    manifest = _complete_manifest(harness)
    runtime = cast(dict[str, object], manifest["runtime"])
    runtime[field] = value
    with pytest.raises(ValueError, match="runtime|package|harness|lock|python"):
        harness.validate_manifest(manifest, formal_run_root=TEST_FORMAL_ROOT)


def test_shared_four_model_audit_contract_reuses_one_olmo_derived_subset(
    tmp_path: Path,
) -> None:
    harness = entrypoint()
    from prefix import no_steering

    assert tuple(no_steering.MODEL_MATRIX) == (
        "Qwen/Qwen3-4B",
        "Qwen/Qwen3-14B",
        "allenai/Olmo-3-7B-Think",
        "allenai/Olmo-3-32B-Think",
    )
    assert len(harness.EXPECTED_CATEGORIES) == 14
    assert "other" in harness.EXPECTED_CATEGORIES

    quotas = {
        category: dict(harness.DEFAULT_QUOTAS)
        for category in harness.EXPECTED_CATEGORIES
    }
    source = [
        {
            "id": f"olmo-derived-{category}-{stratum}-{index}",
            "category": category,
            "question": f"Question {category} {stratum} {index}",
            "options": ["A", "B"],
            "gold": "A",
            "generated_token_count": 1024 if stratum.startswith("saturated") else 100,
            "extracted_answer": "A" if stratum.endswith("extracted") else None,
            "status": "ok",
        }
        for category in harness.EXPECTED_CATEGORIES
        for stratum, quota in harness.DEFAULT_QUOTAS.items()
        for index in range(quota)
    ]
    source.extend(
        {
            "id": f"olmo-derived-filler-{index}",
            "category": "biology",
            "question": f"Filler question {index}",
            "options": ["A", "B"],
            "gold": "A",
            "generated_token_count": 1024,
            "extracted_answer": None,
            "status": "ok",
        }
        for index in range(12032 - len(source))
    )
    selected = harness.select_source_rows(
        source,
        quotas=quotas,
        expected_categories=harness.EXPECTED_CATEGORIES,
        expected_count=12032,
    )
    selected_records = cast(list[dict[str, object]], selected["records"])
    selected_ids = [str(row["id"]) for row in selected_records]
    selected_weights = {str(row["id"]): row["weight"] for row in selected_records}
    selected_strata = {
        str(row["id"]): str(row["selection_stratum"]) for row in selected_records
    }
    selected_categories = {
        str(row["id"]): str(row["category"]) for row in selected_records
    }
    assert len(selected_ids) == sum(harness.DEFAULT_QUOTAS.values()) * 14

    model_contracts: list[dict[str, object]] = []
    for model_id in no_steering.MODEL_MATRIX:
        spec = no_steering.model_spec(model_id)
        conditions = list(CONDITIONS)
        rows = [
            {
                "id": identifier,
                "condition": condition,
                "model_id": model_id,
                "model_revision": spec.revision,
                "parser_version": "legacy-v1",
                "prompt": f"prompt:{identifier}",
                "prompt_hash": harness.prompt_hash(f"prompt:{identifier}"),
                "generated_token_ids": [1, 2]
                if condition == CONDITIONS[0]
                else [1, 2, 3],
                "finish_reason": "length",
                "status": "ok",
                "budget": int(condition.rsplit("_", 1)[1]),
                "selection_weight": selected_weights[identifier],
                "selection_stratum": selected_strata[identifier],
                "category": selected_categories[identifier],
            }
            for identifier in selected_ids
            for condition in conditions
        ]
        model_contracts.append(
            {
                "model_id": model_id,
                "model_revision": spec.revision,
                "model_slug": spec.slug,
                "conditions": conditions,
                "output_paths": {
                    condition: str(tmp_path / spec.slug / f"{condition}.jsonl")
                    for condition in conditions
                },
                "checkpoint_keys": [
                    f"{model_id}:{condition}:{identifier}"
                    for condition in conditions
                    for identifier in selected_ids
                ],
                "rows": rows,
                "resume_pending": [f"{model_id}:{CONDITIONS[1]}:{selected_ids[-1]}"],
            }
        )

    contract = {
        "dataset": "TIGER-Lab/MMLU-Pro",
        "dataset_revision": harness.DATASET_REVISION,
        "source_model": MODEL_ID,
        "source_model_revision": MODEL_REVISION,
        "selected_ids": selected_ids,
        "selected_weights": selected_weights,
        "selected_strata": selected_strata,
        "selected_categories": selected_categories,
        "quotas": quotas,
        "models": model_contracts,
        "parser_version": "legacy-v1",
        "escalation": {
            "requested": False,
            "thresholds": {"null_rescue": 0.75, "accuracy_delta": 0.1},
            "budget": None,
        },
    }

    harness.validate_shared_model_audit_contract(contract)


def test_shared_audit_4096_escalation_is_conditional_and_metadata_bound() -> None:
    harness = entrypoint()
    assert hasattr(harness, "validate_shared_model_audit_contract")
    contract = {
        "conditions": list(CONDITIONS),
        "escalation": {
            "requested": True,
            "budget": 4096,
            "thresholds": {"null_rescue": 0.75, "accuracy_delta": 0.1},
            "observed": {"null_rescue": 0.8, "accuracy_delta": 0.2},
        },
        "parser_version": "legacy-v1",
    }
    harness.validate_shared_model_audit_contract(contract)


def test_production_sized_default_selection_rejects_unprovenanced_rows() -> None:
    harness = entrypoint()
    categories = tuple(harness.EXPECTED_CATEGORIES)
    rows = [
        {
            "id": f"production-{index}",
            "category": categories[index % len(categories)],
            "question": "question",
            "options": ["A", "B"],
            "gold": "A",
            "generated_token_count": 1024,
            "extracted_answer": None,
            "status": "ok",
        }
        for index in range(harness.SOURCE_COUNT)
    ]
    quotas = {category: {"saturated_null": 0} for category in categories}
    with pytest.raises(ValueError, match="provenance|producer|formal"):
        harness.select_source_rows(
            rows,
            quotas=quotas,
            expected_categories=categories,
            expected_count=harness.SOURCE_COUNT,
            require_formal_provenance=True,
        )


@pytest.mark.parametrize("field", ["steering", "quantization"])
def test_producer_provenance_requires_explicit_false_booleans(field: str) -> None:
    harness = entrypoint()
    row = _exact_producer_source_row()
    cast(dict[str, object], cast(dict[str, object], row["metadata"])["provenance"])[
        field
    ] = None
    with pytest.raises(ValueError, match="provenance"):
        harness.select_source_rows(
            [row],
            quotas={"science": {"saturated_null": 1}},
            expected_categories=("science",),
        )


def test_manifest_derives_weights_from_population_and_selected_strata() -> None:
    harness = entrypoint()
    manifest = _complete_manifest(harness)
    manifest["source_path"] = str(_formal_source_path(TEST_FORMAL_ROOT))
    manifest["population_counts"] = {"science/saturated_null": harness.SOURCE_COUNT}
    manifest["population_counts_sha256"] = harness._sha256_value(
        manifest["population_counts"]
    )
    manifest["selected_weights"] = {"x": 1.0, "y": 12031.0}
    manifest["selected_weight_reconstruction"] = harness.SOURCE_COUNT
    evidence = {"source_extracted_answer": None, "source_generated_token_ids": [1, 2]}
    manifest["source_evidence_sha256"] = {"x": harness._sha256_value(evidence)}
    manifest["source_evidence_binding_sha256"] = harness._sha256_value(
        {
            "source_sha256": manifest["source_sha256"],
            "source_evidence_sha256": manifest["source_evidence_sha256"],
        }
    )
    with pytest.raises(ValueError, match="weight|stratum|sample|reconstruct"):
        harness.validate_manifest(manifest, formal_run_root=TEST_FORMAL_ROOT)


def test_generation_and_analysis_rows_bind_source_evidence_hashes() -> None:
    harness = entrypoint()
    manifest = _complete_manifest(harness)
    manifest["source_path"] = str(_formal_source_path(TEST_FORMAL_ROOT))
    manifest["population_counts"] = {"science/saturated_null": harness.SOURCE_COUNT}
    manifest["population_counts_sha256"] = harness._sha256_value(
        manifest["population_counts"]
    )
    manifest["selected_weights"] = {"x": harness.SOURCE_COUNT}
    manifest["selected_weight_reconstruction"] = harness.SOURCE_COUNT
    evidence = {"source_extracted_answer": None, "source_generated_token_ids": [1, 2]}
    manifest["source_evidence_sha256"] = {"x": harness._sha256_value(evidence)}
    manifest["source_evidence_binding_sha256"] = harness._sha256_value(
        {
            "source_sha256": manifest["source_sha256"],
            "source_evidence_sha256": manifest["source_evidence_sha256"],
        }
    )
    rows = _analysis_rows()
    for row in rows:
        row["selection_weight"] = harness.SOURCE_COUNT
        row.update(evidence)
    rows[0]["source_generated_token_ids"] = [9, 9]
    with pytest.raises(ValueError, match="source|evidence|digest"):
        harness._check_existing_rows(
            rows[:1], manifest, CONDITIONS[0], formal_run_root=TEST_FORMAL_ROOT
        )
    with pytest.raises(ValueError, match="source|evidence|digest"):
        harness.validate_analysis(
            rows,
            manifest,
            settings={"temperature": 0.0},
            formal_run_root=TEST_FORMAL_ROOT,
        )


def test_preflight_marker_parser_rejects_stale_or_unrelated_content(
    tmp_path: Path,
) -> None:
    harness = entrypoint()
    marker = tmp_path / "preflight.ok"
    marker.write_text(
        "\n".join(
            [
                "Qwen/Qwen3-4B=75c5e2c1e5c9a0f7f3e45f3e1c4d0a3e0f6c3f9d",
                "Qwen/Qwen3-14B=8f8a8d6d3e8c8c5b6b4f2f5d0b2d8d4c0b3a1f9e",
                f"{MODEL_ID}={MODEL_REVISION}",
                "allenai/Olmo-3-32B-Think=stale",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="marker|revision|model"):
        harness.validate_preflight_marker(marker)
    marker.write_text("unrelated\n", encoding="utf-8")
    with pytest.raises(ValueError, match="marker|revision|model"):
        harness.validate_preflight_marker(marker)


def test_prepare_checkpoint_binds_marker_source_model_and_config_digests(
    tmp_path: Path,
) -> None:
    harness = entrypoint()
    source = tmp_path / "source.jsonl"
    write_jsonl(source, source_rows())
    checkpoint = tmp_path / "prepare.checkpoint.jsonl"
    config = {
        "seed": 17,
        "expected_categories": ["science", "history"],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "marker_path": str(tmp_path / "preflight.ok"),
        "marker_sha256": "m" * 64,
    }
    kwargs = {
        "quotas": {
            "science": {"saturated_null": 1, "unsaturated_extracted": 1},
            "history": {"saturated_null": 1, "unsaturated_extracted": 1},
        },
        "expected_categories": ("science", "history"),
        "checkpoint_path": checkpoint,
        "checkpoint_config": config,
        "source_path": source,
    }
    harness.select_source_rows(iter(source_rows()), **kwargs)
    header = json.loads(checkpoint.read_text(encoding="utf-8").splitlines()[0])
    assert header["model_id"] == MODEL_ID
    assert header["model_revision"] == MODEL_REVISION
    assert header["marker_sha256"] == "m" * 64
    assert header["config_digest"] == harness._sha256_value(
        {
            "source_path": str(source),
            "source_sha256": header["source_sha256"],
            "config": config,
        }
    )
    tampered = checkpoint.read_text(encoding="utf-8").replace("m" * 64, "t" * 64)
    checkpoint.write_text(tampered, encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint|digest"):
        harness.select_source_rows(iter(source_rows()), **kwargs)


def test_prepare_checkpoint_rejects_formal_source_or_root_drift(tmp_path: Path) -> None:
    harness = entrypoint()
    source = tmp_path / "source.jsonl"
    write_jsonl(source, source_rows())
    checkpoint = tmp_path / "prepare.checkpoint.jsonl"
    config = {
        "seed": 17,
        "expected_categories": ["science", "history"],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "formal_run_root": str(tmp_path / "formal-a"),
        "source_path": str(source),
    }
    kwargs = {
        "quotas": {
            "science": {"saturated_null": 1, "unsaturated_extracted": 1},
            "history": {"saturated_null": 1, "unsaturated_extracted": 1},
        },
        "expected_categories": ("science", "history"),
        "checkpoint_path": checkpoint,
        "checkpoint_config": config,
        "source_path": source,
    }
    harness.select_source_rows(iter(source_rows()), **kwargs)
    header = json.loads(checkpoint.read_text(encoding="utf-8").splitlines()[0])
    assert header["source_path"] == str(source)
    assert header["config_digest"] == harness._sha256_value(
        {
            "source_path": str(source),
            "source_sha256": header["source_sha256"],
            "config": config,
        }
    )

    with pytest.raises(ValueError, match="checkpoint|digest"):
        harness.select_source_rows(
            iter(source_rows()),
            **{
                **kwargs,
                "checkpoint_config": {
                    **config,
                    "formal_run_root": str(tmp_path / "formal-b"),
                },
            },
        )
    with pytest.raises(ValueError, match="checkpoint|digest"):
        other_source = tmp_path / "other-source.jsonl"
        other_source.write_bytes(source.read_bytes())
        harness.select_source_rows(
            iter(source_rows()),
            **{**kwargs, "source_path": other_source},
        )


def test_retry_replacement_preserves_source_evidence_and_history_without_duplicates() -> (
    None
):
    harness = entrypoint()
    prior = {
        "id": "x",
        "status": "error",
        "retryable": True,
        "source_extracted_answer": "A",
        "source_generated_token_ids": [7, 8],
        "retry_history": [{"attempt": 1, "error": "first"}],
    }
    replacement = {
        "id": "x",
        "status": "ok",
        "retryable": False,
        "source_extracted_answer": "A",
        "source_generated_token_ids": [7, 8],
    }
    merged = harness.merge_generation_rows([prior], [replacement])
    assert len(merged) == 1
    assert merged[0]["status"] == "ok"
    assert merged[0]["source_extracted_answer"] == "A"
    assert merged[0]["source_generated_token_ids"] == [7, 8]
    assert merged[0]["retry_history"] == prior["retry_history"]


def test_retryable_error_preserves_source_evidence_and_tampering_fails() -> None:
    harness = entrypoint()
    manifest = _complete_manifest(harness)
    manifest["source_path"] = str(_formal_source_path(TEST_FORMAL_ROOT))
    manifest["population_counts"] = {"science/saturated_null": harness.SOURCE_COUNT}
    manifest["population_counts_sha256"] = harness._sha256_value(
        manifest["population_counts"]
    )
    manifest["selected_weights"] = {"x": harness.SOURCE_COUNT}
    manifest["selected_weight_reconstruction"] = harness.SOURCE_COUNT
    evidence = {"source_extracted_answer": "A", "source_generated_token_ids": [7, 8]}
    manifest["source_evidence_sha256"] = {"x": harness._sha256_value(evidence)}
    manifest["source_evidence_binding_sha256"] = harness._sha256_value(
        {
            "source_sha256": manifest["source_sha256"],
            "source_evidence_sha256": manifest["source_evidence_sha256"],
        }
    )
    row = {
        "id": "x",
        "condition": CONDITIONS[0],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "prompt_hash": "p",
        "selection_weight": harness.SOURCE_COUNT,
        "selection_stratum": "science/saturated_null",
        "status": "error",
        "retryable": True,
        **evidence,
    }
    harness._check_existing_rows(
        [row], manifest, CONDITIONS[0], formal_run_root=TEST_FORMAL_ROOT
    )
    with pytest.raises(ValueError, match="source|evidence|digest"):
        harness._check_existing_rows(
            [{**row, "source_generated_token_ids": [9]}],
            manifest,
            CONDITIONS[0],
            formal_run_root=TEST_FORMAL_ROOT,
        )


def _two_id_manifest(harness: ModuleType) -> dict[str, object]:
    manifest = cast(
        dict[str, object],
        harness.build_manifest(
            source_sha256="a" * 64,
            selected_ids=["x", "y"],
            selected_weights={"x": 2.0, "y": 2.0},
            population_counts={"science/saturated_null": 2},
            selected_strata={
                "x": "science/saturated_null",
                "y": "science/saturated_null",
            },
            model_id=MODEL_ID,
            model_revision=MODEL_REVISION,
            dataset="TIGER-Lab/MMLU-Pro",
            dataset_revision="b189ec765aa7ed75c8acfea42df31fdae71f97be",
            runtime={
                "temperature": 0.0,
                "max_model_len": 8192,
                "budgets": [1024, 2048],
                "gpu_memory_utilization": 0.9,
                "batch_size": 8,
                "packages": {"transformers": "test", "vllm": "test"},
            },
            prompt_hashes={"x": "p", "y": "p"},
            tokenizer_chat_template_sha256="c" * 64,
            output_paths={"responses": "responses.jsonl"},
        ),
    )
    manifest["source_path"] = str(_formal_source_path(TEST_FORMAL_ROOT))
    manifest["formal_run_root"] = str(TEST_FORMAL_ROOT)
    manifest["population_counts"] = {"science/saturated_null": harness.SOURCE_COUNT}
    manifest["population_counts_sha256"] = harness._sha256_value(
        manifest["population_counts"]
    )
    manifest["selected_weights"] = {
        "x": harness.SOURCE_COUNT / 2,
        "y": harness.SOURCE_COUNT / 2,
    }
    manifest["selected_weight_reconstruction"] = harness.SOURCE_COUNT
    manifest["source_evidence_sha256"] = {
        identifier: harness._sha256_value(
            {
                "source_extracted_answer": None,
                "source_generated_token_ids": [1, 2],
            }
        )
        for identifier in ("x", "y")
    }
    return manifest


def _prepared_generation_rows() -> list[dict[str, object]]:
    return [
        {
            "id": identifier,
            "condition": CONDITIONS[0],
            "prompt": f"prompt-{identifier}",
            "prompt_hash": "p",
            "selection_weight": 6016.0,
            "selection_stratum": "science/saturated_null",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "budget": 1024,
            "source_extracted_answer": None,
            "source_generated_token_ids": [1, 2],
        }
        for identifier in ("x", "y")
    ]


def _retryable_error_row(harness: ModuleType) -> dict[str, object]:
    history: list[object] = [{"error": "first attempt failed"}]
    return {
        "id": "x",
        "condition": CONDITIONS[0],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "prompt_hash": "p",
        "selection_weight": 6016.0,
        "selection_stratum": "science/saturated_null",
        "status": "error",
        "retryable": True,
        "error": "first attempt failed",
        "source_extracted_answer": None,
        "source_generated_token_ids": [1, 2],
        "retry_history": history,
        "retry_history_sha256": harness._sha256_value(history),
    }


def _scripted_output(token_ids: list[int]) -> object:
    completion = SimpleNamespace(
        token_ids=list(token_ids),
        finish_reason="stop",
        text="The answer is (A)",
    )
    return SimpleNamespace(outputs=[completion])


class _ScriptedEngine:
    def __init__(self, plan: list[object]) -> None:
        self.plan = list(plan)

    def generate(self, prompts: list[str], params: object) -> list[object]:
        step = self.plan.pop(0)
        if isinstance(step, Exception):
            raise step
        return [_scripted_output(cast(list[int], step)) for _ in prompts]


def _generation_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        output_root=tmp_path,
        source_responses=None,
        formal_run_root=TEST_FORMAL_ROOT,
        log_file=tmp_path / "run.log",
        gpu_memory_utilization=0.9,
        batch_size=8,
    )


def test_generate_retry_batches_preserve_earlier_replacements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = entrypoint()
    monkeypatch.setattr(harness, "validate_manifest", lambda *args, **kwargs: None)
    manifest = _two_id_manifest(harness)
    cast(dict[str, object], manifest["runtime"])["batch_size"] = 1
    (tmp_path / "prepared.json").write_text(
        json.dumps({"manifest": manifest, "records": _prepared_generation_rows()}),
        encoding="utf-8",
    )
    output_path = tmp_path / f"{CONDITIONS[0]}.jsonl"
    output_path.write_text(
        json.dumps(_retryable_error_row(harness), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    engine = _ScriptedEngine([[5, 6], [7, 8]])
    _patch_fake_vllm(monkeypatch)
    import prefix.runner as runner

    monkeypatch.setattr(runner, "get_engine", lambda *args, **kwargs: engine)
    monkeypatch.setattr(harness, "validate_source_sha256", lambda *args: None)
    args = _generation_args(tmp_path)
    args.batch_size = 1
    harness.generate(args)
    saved = {
        str(row["id"]): row
        for row in (
            json.loads(line)
            for line in output_path.read_text(encoding="utf-8").splitlines()
        )
    }
    assert set(saved) == {"x", "y"}
    assert saved["x"]["status"] == "ok"
    assert saved["x"]["generated_token_ids"] == [5, 6]
    assert saved["x"]["retry_history"] == [{"error": "first attempt failed"}]
    assert saved["y"]["status"] == "ok"
    assert saved["y"]["generated_token_ids"] == [7, 8]


def test_generate_error_after_earlier_batch_keeps_persisted_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = entrypoint()
    monkeypatch.setattr(harness, "validate_manifest", lambda *args, **kwargs: None)
    manifest = _two_id_manifest(harness)
    cast(dict[str, object], manifest["runtime"])["batch_size"] = 1
    (tmp_path / "prepared.json").write_text(
        json.dumps({"manifest": manifest, "records": _prepared_generation_rows()}),
        encoding="utf-8",
    )
    output_path = tmp_path / f"{CONDITIONS[0]}.jsonl"
    engine = _ScriptedEngine([[5, 6], RuntimeError("second batch failed")])
    _patch_fake_vllm(monkeypatch)
    import prefix.runner as runner

    monkeypatch.setattr(runner, "get_engine", lambda *args, **kwargs: engine)
    monkeypatch.setattr(harness, "validate_source_sha256", lambda *args: None)
    args = _generation_args(tmp_path)
    args.batch_size = 1
    with pytest.raises(RuntimeError, match="second batch failed"):
        harness.generate(args)
    saved = {
        str(row["id"]): row
        for row in (
            json.loads(line)
            for line in output_path.read_text(encoding="utf-8").splitlines()
        )
    }
    assert set(saved) == {"x", "y"}
    assert saved["x"]["status"] == "ok"
    assert saved["x"]["generated_token_ids"] == [5, 6]
    assert saved["y"]["status"] == "error"
    assert saved["y"]["retryable"] is True
    assert saved["y"]["retry_history"] == [{"error": "second batch failed"}]


def test_prepare_impl_acquires_root_lock_before_unlocked_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = entrypoint()
    args = _prepare_test_args(tmp_path)
    events: list[str] = []

    @contextlib.contextmanager
    def lock(root: Path) -> Iterator[Path]:
        events.append(f"acquire:{root}")
        yield root
        events.append("release")

    monkeypatch.setattr(harness, "exclusive_mutation_lock", lock)
    monkeypatch.setattr(
        harness,
        "_prepare_impl_unlocked",
        lambda received: events.append(f"work:{received.output_root}"),
    )

    harness._prepare_impl(args)

    assert events == [
        f"acquire:{args.output_root}",
        f"work:{args.output_root}",
        "release",
    ]


def test_generate_acquires_root_lock_before_provider_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = entrypoint()
    args = _generation_args(tmp_path)
    events: list[str] = []

    @contextlib.contextmanager
    def lock(root: Path) -> Iterator[Path]:
        events.append(f"acquire:{root}")
        yield root
        events.append("release")

    monkeypatch.setattr(harness, "exclusive_mutation_lock", lock)
    monkeypatch.setattr(
        harness,
        "_generate_impl",
        lambda received: events.append(f"provider:{received.output_root}"),
    )

    harness.generate(args)

    assert events == [
        f"acquire:{args.output_root}",
        f"provider:{args.output_root}",
        "release",
    ]


def test_analyze_impl_acquires_root_lock_before_artifact_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = entrypoint()
    args = argparse.Namespace(output_root=tmp_path)
    events: list[str] = []

    @contextlib.contextmanager
    def lock(root: Path) -> Iterator[Path]:
        events.append(f"acquire:{root}")
        yield root
        events.append("release")

    monkeypatch.setattr(harness, "exclusive_mutation_lock", lock)
    monkeypatch.setattr(
        harness,
        "_analyze_impl_unlocked",
        lambda received: events.append(f"work:{received.output_root}"),
    )

    harness._analyze_impl(args)

    assert events == [
        f"acquire:{args.output_root}",
        f"work:{args.output_root}",
        "release",
    ]


def test_same_root_attempt_fails_before_second_prepare_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = entrypoint()
    first_args = _prepare_test_args(tmp_path)
    second_args = argparse.Namespace(**vars(first_args))
    started = threading.Event()
    release = threading.Event()
    work: list[Path] = []

    def unlocked(args: argparse.Namespace) -> None:
        work.append(args.output_root)
        started.set()
        assert release.wait(timeout=5)

    monkeypatch.setattr(harness, "_prepare_impl_unlocked", unlocked)
    first_error: list[BaseException] = []

    def run_first() -> None:
        try:
            harness._prepare_impl(first_args)
        except BaseException as error:
            first_error.append(error)

    thread = threading.Thread(target=run_first)
    thread.start()
    assert started.wait(timeout=5)
    with pytest.raises(RuntimeError, match="already locked"):
        harness._prepare_impl(second_args)
    release.set()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert first_error == []
    assert work == [first_args.output_root]
