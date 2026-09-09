from __future__ import annotations

import importlib.util
import hashlib
import json
import math
from pathlib import Path
from types import ModuleType
from collections.abc import Mapping

import pytest

from prefix import no_steering


ROOT = Path(__file__).parents[1]


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
        "schema_version": 1,
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
    }
    (checkpoint / "manifest.json").write_text(
        json.dumps({"config_sha256": _config_digest(config), "config": config})
    )
    response = checkpoint / "mmlu_pro" / "responses.jsonl"
    response.parent.mkdir()
    values = logprobs if logprobs is not None else [-0.5, -1.0]
    response.write_text(
        json.dumps(
            {
                "id": "m0",
                "benchmark": "mmlu_pro",
                "model_id": model_id,
                "status": "ok",
                "generated_token_count": len(values),
                "selected_generated_token_logprobs": values,
                "metadata": {
                    "provenance": {"model_id": model_id, "revision": spec.revision}
                },
            }
        )
        + "\n"
    )
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
