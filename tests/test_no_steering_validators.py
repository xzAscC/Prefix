from __future__ import annotations

import importlib.util
import hashlib
import json
import math
import shutil
from pathlib import Path
from types import ModuleType
from collections.abc import Mapping

import pytest

from prefix import no_steering


ROOT = Path(__file__).parents[1]
_MISSING = object()


def _module(name: str) -> ModuleType:
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config_digest(config: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            config, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _formal_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    logprobs: list[float] | None = None,
    summary_ppl: dict[str, object] | None = None,
    conditional_ppl: dict[str, object] | None = None,
    aggregate_roots: bool = True,
    schema_version: int = 1,
    finish_reason: object = _MISSING,
) -> tuple[ModuleType, Path, Path]:
    formal = _module("validate_no_steering_formal")
    model_id = "Qwen/Qwen3-4B"
    spec = no_steering.model_spec(model_id)
    monkeypatch.setattr(
        formal, "DATASET_SCOPES", {"mmlu_pro": {"count": 1, "split": "test"}}
    )
    aggregate_checkpoint = tmp_path / "checkpoints"
    aggregate_result = tmp_path / "results"
    checkpoint = aggregate_checkpoint / spec.slug
    result = aggregate_result / spec.slug
    checkpoint.mkdir(parents=True)
    result.mkdir(parents=True)
    config = {
        "schema_version": schema_version,
        "model_id": model_id,
        "model_slug": spec.slug,
        "model_revision": spec.revision,
        "prompt_template_version": "chat-v1",
        "steering": False,
        "quantization": False,
        "sampling": {
            "max_tokens": 1024,
            "temperature": 0.0,
            "logprobs": 1,
            "revision": spec.revision,
            "quantization": None,
            "max_model_len": 8192,
            "gpu_memory_utilization": 0.9,
        },
        "output_roots": {
            "checkpoint": str(aggregate_checkpoint if aggregate_roots else checkpoint),
            "output": str(aggregate_result if aggregate_roots else result),
        },
        "benchmark_manifest": {
            "mmlu_pro": {
                "split": "test",
                "expected_count": 1,
                "loaded_count": 1,
                "complete": True,
                "limited": False,
            }
        },
        "benchmark_ids": {"mmlu_pro": ["m0"]},
        "benchmark_content_sha256": {
            "mmlu_pro": hashlib.sha256(
                json.dumps(
                    [{"id": "m0", "content": {}}],
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
        },
    }
    (checkpoint / "manifest.json").write_text(
        json.dumps({"config_sha256": _config_digest(config), "config": config})
    )
    response = checkpoint / "mmlu_pro" / "responses.jsonl"
    response.parent.mkdir()
    values = logprobs if logprobs is not None else [-0.5, -1.0]
    response_row: dict[str, object] = {
        "id": "m0",
        "benchmark": "mmlu_pro",
        "model_id": model_id,
        "status": "ok",
        "generated_token_count": len(values),
        "selected_generated_token_logprobs": values,
        "metadata": {"provenance": {"model_id": model_id, "revision": spec.revision}},
    }
    if finish_reason is not _MISSING:
        response_row["finish_reason"] = finish_reason
    response.write_text(json.dumps(response_row) + "\n")
    expected_ppl = math.exp(-sum(values) / len(values))
    payload = summary_ppl or {
        "ppl": expected_ppl,
        "selected_token_count": len(values),
        "generated_token_count": len(values),
        "covered_records": 1,
        "total_records": 1,
        "coverage_ratio": 1.0,
    }
    (result / "summary.json").write_text(
        json.dumps(
            {
                "provenance": {"model_id": model_id, "revision": spec.revision},
                "ppl": payload,
            }
        )
    )
    (result / "conditional_ppl.json").write_text(
        json.dumps(conditional_ppl if conditional_ppl is not None else payload)
    )
    return formal, checkpoint, result


@pytest.mark.parametrize("finish_reason", ["length", "stop"])
def test_schema_v2_requires_and_validates_finish_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    finish_reason: str,
) -> None:
    formal, checkpoint, result = _formal_fixture(
        tmp_path,
        monkeypatch,
        schema_version=2,
        finish_reason=finish_reason,
    )

    formal.main(
        [
            "--result-root",
            str(result),
            "--checkpoint-root",
            str(checkpoint),
            "--model-id",
            "Qwen/Qwen3-4B",
        ]
    )


@pytest.mark.parametrize(
    ("finish_reason", "expected_error"),
    [(None, "finish_reason"), ("eos", "finish_reason")],
)
def test_schema_v2_rejects_missing_or_invalid_finish_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    finish_reason: object,
    expected_error: str,
) -> None:
    if finish_reason is None:
        formal, checkpoint, result = _formal_fixture(
            tmp_path, monkeypatch, schema_version=2
        )
    else:
        formal, checkpoint, result = _formal_fixture(
            tmp_path,
            monkeypatch,
            schema_version=2,
            finish_reason=finish_reason,
        )

    with pytest.raises(ValueError, match=expected_error):
        formal.main(
            [
                "--result-root",
                str(result),
                "--checkpoint-root",
                str(checkpoint),
                "--model-id",
                "Qwen/Qwen3-4B",
            ]
        )


def test_schema_v1_preserves_explicit_legacy_unavailable_finish_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    formal, checkpoint, result = _formal_fixture(
        tmp_path,
        monkeypatch,
        schema_version=1,
        finish_reason="unavailable",
    )

    formal.main(
        [
            "--result-root",
            str(result),
            "--checkpoint-root",
            str(checkpoint),
            "--model-id",
            "Qwen/Qwen3-4B",
        ]
    )
    assert (
        json.loads((checkpoint / "mmlu_pro" / "responses.jsonl").read_text())[
            "finish_reason"
        ]
        == "unavailable"
    )


def test_formal_validator_unwraps_runner_manifest_and_uses_checkpoint_responses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    formal = _module("validate_no_steering_formal")
    model_id = "Qwen/Qwen3-4B"
    setattr(formal, "DATASET_SCOPES", {"mmlu_pro": {"count": 1, "split": "test"}})
    slug = no_steering.model_spec(model_id).slug
    checkpoint = tmp_path / "checkpoint" / slug
    result = tmp_path / "result" / slug
    checkpoint.mkdir(parents=True)
    result.mkdir(parents=True)
    config = {
        "schema_version": 1,
        "model_id": model_id,
        "model_slug": no_steering.model_spec(model_id).slug,
        "model_revision": no_steering.model_spec(model_id).revision,
        "prompt_template_version": "chat-v1",
        "steering": False,
        "quantization": False,
        "sampling": {
            "max_tokens": 1024,
            "temperature": 0.0,
            "logprobs": 1,
            "revision": no_steering.model_spec(model_id).revision,
            "quantization": None,
            "max_model_len": 8192,
            "gpu_memory_utilization": 0.9,
        },
        "output_roots": {
            "checkpoint": str(checkpoint),
            "output": str(result),
        },
        "benchmark_manifest": {
            "mmlu_pro": {
                "split": "test",
                "expected_count": 1,
                "complete": True,
                "loaded_count": 1,
                "limited": False,
            }
        },
        "benchmark_ids": {"mmlu_pro": ["m0"]},
    }
    (checkpoint / "manifest.json").write_text(
        json.dumps({"config_sha256": _config_digest(config), "config": config})
    )
    response_path = checkpoint / "mmlu_pro" / "responses.jsonl"
    response_path.parent.mkdir()
    response_path.write_text(
        json.dumps(
            {
                "id": "m0",
                "status": "ok",
                "model_id": model_id,
                "benchmark": "mmlu_pro",
                "generated_token_count": 1,
                "selected_generated_token_logprobs": [-1.0],
                "metadata": {
                    "provenance": {
                        "model_id": model_id,
                        "revision": no_steering.model_spec(model_id).revision,
                    }
                },
            }
        )
        + "\n"
    )
    (result / "summary.json").write_text(
        json.dumps(
            {
                "provenance": {
                    "model_id": model_id,
                    "revision": no_steering.model_spec(model_id).revision,
                },
                "ppl": {
                    "ppl": math.exp(1.0),
                    "selected_token_count": 1,
                    "generated_token_count": 1,
                    "covered_records": 1,
                    "total_records": 1,
                    "coverage_ratio": 1.0,
                },
            }
        )
    )
    (result / "conditional_ppl.json").write_text(
        json.dumps(
            {
                "ppl": math.exp(1.0),
                "selected_token_count": 1,
                "generated_token_count": 1,
                "covered_records": 1,
                "total_records": 1,
                "coverage_ratio": 1.0,
            }
        )
    )
    formal.main(
        [
            "--result-root",
            str(result),
            "--checkpoint-root",
            str(checkpoint),
            "--model-id",
            model_id,
        ]
    )


def test_formal_validator_accepts_model_scoped_roots_with_aggregate_manifest_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    formal, checkpoint, result = _formal_fixture(tmp_path, monkeypatch)
    formal.main(
        [
            "--result-root",
            str(result),
            "--checkpoint-root",
            str(checkpoint),
            "--model-id",
            "Qwen/Qwen3-4B",
        ]
    )


def test_formal_validator_streams_response_jsonl_without_read_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    formal, checkpoint, result = _formal_fixture(tmp_path, monkeypatch)
    original_read_text = Path.read_text

    def reject_response_read_text(
        self: Path,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> str:
        if self.name == "responses.jsonl":
            raise AssertionError("responses.jsonl must be streamed")
        return original_read_text(
            self, encoding=encoding, errors=errors, newline=newline
        )

    monkeypatch.setattr(Path, "read_text", reject_response_read_text)
    formal.main(
        [
            "--result-root",
            str(result),
            "--checkpoint-root",
            str(checkpoint),
            "--model-id",
            "Qwen/Qwen3-4B",
        ]
    )


def test_formal_validator_accepts_identical_tree_copied_to_local_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    formal, checkpoint, result = _formal_fixture(tmp_path / "remote", monkeypatch)
    local = tmp_path / "local-run"
    local_checkpoint = local / "checkpoints" / checkpoint.name
    local_result = local / "results" / result.name
    shutil.copytree(checkpoint, local_checkpoint)
    shutil.copytree(result, local_result)

    formal.main(
        [
            "--result-root",
            str(local_result),
            "--checkpoint-root",
            str(local_checkpoint),
            "--model-id",
            "Qwen/Qwen3-4B",
        ]
    )


@pytest.mark.parametrize("container", ["artifacts", "outputs"])
def test_formal_validator_rejects_relocated_root_with_wrong_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, container: str
) -> None:
    formal, checkpoint, result = _formal_fixture(tmp_path / "remote", monkeypatch)
    local = tmp_path / "local-run"
    local_checkpoint = local / container / checkpoint.name
    local_result = local / "results" / result.name
    shutil.copytree(checkpoint, local_checkpoint)
    shutil.copytree(result, local_result)

    with pytest.raises(ValueError, match="checkpoint root"):
        formal.main(
            [
                "--result-root",
                str(local_result),
                "--checkpoint-root",
                str(local_checkpoint),
                "--model-id",
                "Qwen/Qwen3-4B",
            ]
        )


def test_formal_validator_rejects_relocated_root_with_wrong_slug(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    formal, checkpoint, result = _formal_fixture(tmp_path / "remote", monkeypatch)
    local = tmp_path / "local-run"
    local_checkpoint = local / "checkpoints" / "not-the-model"
    local_result = local / "results" / result.name
    shutil.copytree(checkpoint, local_checkpoint)
    shutil.copytree(result, local_result)

    with pytest.raises(ValueError, match="model-scoped"):
        formal.main(
            [
                "--result-root",
                str(local_result),
                "--checkpoint-root",
                str(local_checkpoint),
                "--model-id",
                "Qwen/Qwen3-4B",
            ]
        )


def test_formal_validator_rejects_ids_changed_in_relocated_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    formal, checkpoint, result = _formal_fixture(tmp_path / "remote", monkeypatch)
    local = tmp_path / "local-run"
    local_checkpoint = local / "checkpoints" / checkpoint.name
    local_result = local / "results" / result.name
    shutil.copytree(checkpoint, local_checkpoint)
    shutil.copytree(result, local_result)
    response_path = local_checkpoint / "mmlu_pro" / "responses.jsonl"
    response = json.loads(response_path.read_text().strip())
    response["id"] = "changed-id"
    response_path.write_text(json.dumps(response) + "\n")

    with pytest.raises(ValueError, match="ids mismatch"):
        formal.main(
            [
                "--result-root",
                str(local_result),
                "--checkpoint-root",
                str(local_checkpoint),
                "--model-id",
                "Qwen/Qwen3-4B",
            ]
        )


def test_formal_validator_rejects_wrong_manifest_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    formal, checkpoint, result = _formal_fixture(tmp_path, monkeypatch)
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    manifest["config_sha256"] = "0" * 64
    (checkpoint / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="config digest"):
        formal.main(
            [
                "--result-root",
                str(result),
                "--checkpoint-root",
                str(checkpoint),
                "--model-id",
                "Qwen/Qwen3-4B",
            ]
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("steering", True),
        ("quantization", "int8"),
        ("model_revision", "0" * 40),
        ("sampling", {"temperature": 0.7}),
    ],
)
def test_formal_validator_rejects_execution_contract_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    formal, checkpoint, result = _formal_fixture(tmp_path, monkeypatch)
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    config = manifest["config"]
    config[field] = value
    manifest["config_sha256"] = _config_digest(config)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="steering|quantization|sampling|revision"):
        formal.main(
            [
                "--result-root",
                str(result),
                "--checkpoint-root",
                str(checkpoint),
                "--model-id",
                "Qwen/Qwen3-4B",
            ]
        )


@pytest.mark.parametrize(
    "summary_ppl",
    [
        {
            "ppl": None,
            "selected_token_count": 2,
            "generated_token_count": 2,
            "covered_records": 1,
            "total_records": 1,
            "coverage_ratio": 1.0,
        },
        {
            "ppl": 999999.0,
            "selected_token_count": 2,
            "generated_token_count": 2,
            "covered_records": 1,
            "total_records": 1,
            "coverage_ratio": 1.0,
        },
        {
            "ppl": math.exp(0.75),
            "selected_token_count": 1,
            "generated_token_count": 2,
            "covered_records": 1,
            "total_records": 1,
            "coverage_ratio": 0.5,
        },
    ],
)
def test_formal_validator_rejects_unrecomputed_or_incomplete_ppl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, summary_ppl: dict[str, object]
) -> None:
    formal, checkpoint, result = _formal_fixture(
        tmp_path, monkeypatch, summary_ppl=summary_ppl
    )
    with pytest.raises(ValueError, match="PPL|coverage|token"):
        formal.main(
            [
                "--result-root",
                str(result),
                "--checkpoint-root",
                str(checkpoint),
                "--model-id",
                "Qwen/Qwen3-4B",
            ]
        )


def test_formal_validator_rejects_conditional_ppl_disagreement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    formal, checkpoint, result = _formal_fixture(
        tmp_path,
        monkeypatch,
        conditional_ppl={
            "ppl": 999999.0,
            "selected_token_count": 2,
            "generated_token_count": 2,
            "covered_records": 1,
            "total_records": 1,
            "coverage_ratio": 1.0,
        },
    )
    with pytest.raises(ValueError, match="PPL|conditional_ppl|disagree"):
        formal.main(
            [
                "--result-root",
                str(result),
                "--checkpoint-root",
                str(checkpoint),
                "--model-id",
                "Qwen/Qwen3-4B",
            ]
        )


def test_finalizer_rejects_changed_bound_response_content(tmp_path: Path) -> None:
    finalizer = _module("finalize_no_steering")
    model_id = "Qwen/Qwen3-4B"
    model_root = tmp_path / finalizer.model_spec(model_id).slug
    response_root = tmp_path / "responses"
    response_paths: dict[str, Path] = {}
    for benchmark in finalizer.DATASET_SCOPES:
        response = (
            response_root
            / finalizer.model_spec(model_id).slug
            / benchmark
            / "responses.jsonl"
        )
        response.parent.mkdir(parents=True, exist_ok=True)
        response.write_text('{"id":"m0"}\n')
        response_paths[benchmark] = response
    for benchmark, scope in finalizer.DATASET_SCOPES.items():
        score = model_root / benchmark / "scores.jsonl"
        score.parent.mkdir(parents=True, exist_ok=True)
        score.write_text("")
    score_files = {
        benchmark: {
            "path": str((model_root / benchmark / "scores.jsonl").resolve()),
            "content_sha256": hashlib.sha256(
                (model_root / benchmark / "scores.jsonl").read_bytes()
            ).hexdigest(),
        }
        for benchmark in finalizer.DATASET_SCOPES
    }
    manifest = {
        "schema_version": 1,
        "model_id": model_id,
        "model_slug": finalizer.model_spec(model_id).slug,
        "judge_model": "gemini-3.5-flash-lite",
        "response_root": str(response_root),
        "rubric_version": "gemini-rubric-v1",
        **finalizer._expected_scoring_contract(),
        "response_files": {
            benchmark: {
                "ids": [],
                "path": str(response_paths[benchmark]),
                "content_sha256": "stale",
            }
            for benchmark in finalizer.DATASET_SCOPES
        },
        "score_files": score_files,
    }
    (model_root / "scoring_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="digest"):
        finalizer.validate_scoring(tmp_path)


def test_smoke_validator_accepts_complete_canonical_ppl_summary(
    tmp_path: Path,
) -> None:
    smoke = _module("validate_no_steering_smoke")
    model_id = "Qwen/Qwen3-4B"
    spec = no_steering.model_spec(model_id)
    checkpoint = tmp_path / "checkpoint"
    result = tmp_path / "result"
    checkpoint.mkdir()
    result.mkdir()
    benchmark_ids = {
        benchmark: [f"{benchmark}-0"] for benchmark in smoke.DATASET_SCOPES
    }
    config = {
        "schema_version": 1,
        "model_id": model_id,
        "model_slug": spec.slug,
        "model_revision": spec.revision,
        "limited": True,
        "benchmark_ids": benchmark_ids,
        "benchmark_manifest": {
            benchmark: {"loaded_count": 1, "complete": False, "limited": True}
            for benchmark in smoke.DATASET_SCOPES
        },
    }
    (checkpoint / "manifest.json").write_text(
        json.dumps({"config": config, "config_sha256": smoke._config_sha256(config)})
    )
    for benchmark, identifiers in benchmark_ids.items():
        response = checkpoint / benchmark / "responses.jsonl"
        response.parent.mkdir()
        response.write_text(
            json.dumps(
                {
                    "id": identifiers[0],
                    "benchmark": benchmark,
                    "model_id": model_id,
                    "status": "ok",
                    "generated_token_count": 1,
                    "selected_generated_token_logprobs": [-0.5],
                    "metadata": {
                        "provenance": {
                            "model_id": model_id,
                            "revision": spec.revision,
                        }
                    },
                }
            )
            + "\n"
        )
    (result / "summary.json").write_text(
        json.dumps(
            {
                "provenance": {"model_id": model_id, "revision": spec.revision},
                "limited": True,
                "ppl": {
                    "ppl": 1.5,
                    "selected_token_count": 3,
                    "generated_token_count": 3,
                    "covered_records": 3,
                    "total_records": 3,
                    "coverage_ratio": 1.0,
                },
            }
        )
    )

    smoke.validate_smoke_roots(
        result_root=result, checkpoint_root=checkpoint, model_id=model_id
    )


def test_smoke_validator_accepts_schema_v2_manifest_from_run_producer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    smoke = _module("validate_no_steering_smoke")
    producer = _module("run_no_steering")
    model_id = "Qwen/Qwen3-4B"
    scopes = {
        "harmbench": {"count": 1, "split": "test"},
        "mmlu_pro": {"count": 1, "split": "test"},
        "math500": {"count": 1, "split": "test"},
    }
    monkeypatch.setattr(smoke, "DATASET_SCOPES", scopes)
    monkeypatch.setattr(producer, "DATASET_SCOPES", scopes)
    benchmarks = {
        "harmbench": [{"id": "h0", "behavior": "test behavior"}],
        "mmlu_pro": [{"id": "m0", "question": "1+1?", "options": ["1", "2"]}],
        "math500": [{"id": "x0", "problem": "1+1", "answer": "2"}],
    }
    checkpoint = tmp_path / "checkpoints"
    result = tmp_path / "results"
    spec = no_steering.model_spec(model_id)
    checkpoint_model = checkpoint / spec.slug
    result_model = result / spec.slug
    checkpoint_model.mkdir(parents=True)
    result_model.mkdir(parents=True)
    config = producer.checkpoint_manifest(
        benchmarks,
        model_id=model_id,
        sampling={
            "max_tokens": 1024,
            "temperature": 0.0,
            "logprobs": 1,
        },
        batch_prompts=1,
        max_model_len=8192,
        output_roots={"checkpoint": str(checkpoint), "output": str(result)},
        limited=True,
    )
    (checkpoint_model / "manifest.json").write_text(
        json.dumps(
            {"config": config, "config_sha256": _config_digest(config)},
            sort_keys=True,
        )
    )
    for benchmark, rows in benchmarks.items():
        row = rows[0]
        metadata: dict[str, object] = {
            key: value for key, value in row.items() if key != "id"
        }
        metadata["provenance"] = {
            "model_id": model_id,
            "revision": spec.revision,
        }
        response = checkpoint_model / benchmark / "responses.jsonl"
        response.parent.mkdir()
        response.write_text(
            json.dumps(
                {
                    "id": row["id"],
                    "benchmark": benchmark,
                    "model_id": model_id,
                    "status": "ok",
                    "finish_reason": "stop",
                    "generated_token_count": 1,
                    "selected_generated_token_logprobs": [-0.5],
                    "metadata": metadata,
                }
            )
            + "\n"
        )
    (result_model / "summary.json").write_text(
        json.dumps(
            {
                "provenance": {"model_id": model_id, "revision": spec.revision},
                "limited": True,
                "ppl": {
                    "ppl": 1.5,
                    "selected_token_count": 3,
                    "generated_token_count": 3,
                    "covered_records": 3,
                    "total_records": 3,
                    "coverage_ratio": 1.0,
                },
            }
        )
    )

    smoke.validate_smoke_roots(
        result_root=result_model,
        checkpoint_root=checkpoint_model,
        model_id=model_id,
    )

    response_path = checkpoint_model / "harmbench" / "responses.jsonl"
    response = json.loads(response_path.read_text())
    response["finish_reason"] = "eos"
    response_path.write_text(json.dumps(response) + "\n")
    with pytest.raises(ValueError, match="finish_reason"):
        smoke.validate_smoke_roots(
            result_root=result_model,
            checkpoint_root=checkpoint_model,
            model_id=model_id,
        )
    response["finish_reason"] = "stop"
    response_path.write_text(json.dumps(response) + "\n")

    manifest_path = checkpoint_model / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["config"]["benchmark_content_sha256"]["harmbench"] = "0" * 64
    manifest["config_sha256"] = _config_digest(manifest["config"])
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    with pytest.raises(ValueError, match="content digest"):
        smoke.validate_smoke_roots(
            result_root=result_model,
            checkpoint_root=checkpoint_model,
            model_id=model_id,
        )


def test_smoke_marker_rejects_empty_stale_wrong_model_and_wrong_job(
    tmp_path: Path,
) -> None:
    smoke = _module("validate_no_steering_smoke")
    marker = tmp_path / "preflight.ok"
    marker.write_text("", encoding="utf-8")
    for model_id, job_id in (
        ("Qwen/Qwen3-4B", "job-1"),
        ("Qwen/Qwen3-14B", "job-1"),
        ("Qwen/Qwen3-4B", "job-2"),
    ):
        marker.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "complete",
                    "preflight_job_id": "job-1",
                    "run_root": str(tmp_path / "run"),
                    "models": {model_id: {"model_id": model_id}},
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="marker|model|job|digest"):
            smoke.validate_preflight_marker(
                marker,
                model_id=model_id,
                run_root=tmp_path / "run",
                preflight_job_id=job_id,
            )


def test_formal_validator_rejects_child_response_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    formal, checkpoint, result = _formal_fixture(tmp_path, monkeypatch)
    real = tmp_path / "real-responses.jsonl"
    response = checkpoint / "mmlu_pro" / "responses.jsonl"
    real.write_bytes(response.read_bytes())
    response.unlink()
    response.symlink_to(real)
    with pytest.raises(ValueError, match="symlink"):
        formal.main(
            [
                "--result-root",
                str(result),
                "--checkpoint-root",
                str(checkpoint),
                "--model-id",
                "Qwen/Qwen3-4B",
            ]
        )


def test_formal_validator_rejects_harmbench_content_digest_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    formal, checkpoint, result = _formal_fixture(
        tmp_path, monkeypatch, schema_version=2, finish_reason="stop"
    )
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["config"]["benchmark_content_sha256"] = {"mmlu_pro": "0" * 64}
    manifest["config_sha256"] = _config_digest(manifest["config"])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="content|digest"):
        formal.main(
            [
                "--result-root",
                str(result),
                "--checkpoint-root",
                str(checkpoint),
                "--model-id",
                "Qwen/Qwen3-4B",
            ]
        )
