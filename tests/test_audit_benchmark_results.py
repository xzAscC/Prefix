from __future__ import annotations

import hashlib
import importlib.util
import json
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator

import pytest

from prefix.no_steering import DATASET_SCOPES, MODEL_MATRIX, model_spec
from prefix.runner import parse_answer_letter


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "audit_benchmark_results.py"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("audit_benchmark_results", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def _fake_notification(events: list[tuple[str, object]]) -> Any:
    @contextmanager
    def notification(task: str, *, log_file: str | Path) -> Iterator[None]:
        events.append(("enter", (task, Path(log_file))))
        try:
            yield
        except BaseException as error:
            events.append(("exit", type(error)))
            raise
        else:
            events.append(("exit", None))

    return notification


@pytest.fixture
def formal_run(tmp_path: Path) -> Path:
    root = tmp_path / "formal-run"
    for model_id in MODEL_MATRIX:
        spec = model_spec(model_id)
        for benchmark in DATASET_SCOPES:
            response_rows: list[dict[str, object]] = []
            score_rows: list[dict[str, object]] = []
            for index in range(2):
                identifier = f"{benchmark}-{index}"
                if benchmark == "mmlu_pro":
                    response_rows.append(
                        {
                            "id": identifier,
                            "benchmark": benchmark,
                            "model_id": model_id,
                            "raw_response": {
                                "text": "The answer is (A)"
                                if index == 0
                                else "The answer is (B)"
                            },
                            "extracted_answer": "B" if index == 0 else "B",
                            "gold": "A",
                            "generated_token_count": 1024 if index == 0 else 512,
                            "status": "ok",
                        }
                    )
                    score_rows.append(
                        {
                            "id": identifier,
                            "benchmark": benchmark,
                            "status": "ok",
                            "correct": False,
                        }
                    )
                elif benchmark == "math500":
                    response_rows.append(
                        {
                            "id": identifier,
                            "benchmark": benchmark,
                            "model_id": model_id,
                            "raw_response": {
                                "text": r"\boxed{1/2}"
                                if index == 0
                                else "No final answer"
                            },
                            "extracted_answer": "0.5" if index == 0 else None,
                            "gold": "0.5",
                            "generated_token_count": 1024 if index == 0 else 256,
                            "status": "ok",
                        }
                    )
                    score_rows.append(
                        {
                            "id": identifier,
                            "benchmark": benchmark,
                            "status": "ok" if index == 0 else "unparseable",
                            "answer_correct": index == 0,
                            "judge_status": "ok" if index == 0 else "unparseable",
                        }
                    )
                else:
                    response_rows.append(
                        {
                            "id": identifier,
                            "benchmark": benchmark,
                            "model_id": model_id,
                            "raw_response": {"text": "safe"},
                            "generated_token_count": 128,
                            "status": "ok",
                        }
                    )
                    score_rows.append(
                        {
                            "id": identifier,
                            "benchmark": benchmark,
                            "status": "ok",
                            "gemini_label": "safe",
                        }
                    )
            response_path = (
                root / "checkpoints" / spec.slug / benchmark / "responses.jsonl"
            )
            score_path = root / "results" / spec.slug / benchmark / "scores.jsonl"
            response_sha = _write_jsonl(response_path, response_rows)
            score_sha = _write_jsonl(score_path, score_rows)
            (score_path.parent / "bindings.json").write_text(
                json.dumps(
                    {
                        "model_id": model_id,
                        "revision": spec.revision,
                        "response_sha256": response_sha,
                        "score_sha256": score_sha,
                    }
                ),
                encoding="utf-8",
            )
    return root


def test_cli_requires_run_root_output_json_and_log_file_and_reports_exact_models(
    formal_run: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    output = tmp_path / "audit.json"
    log = tmp_path / "audit.log"
    monkeypatch.setattr(module, "notify_on_exit", _fake_notification([]))

    result = module.main(
        [
            "--formal-run-root",
            str(formal_run),
            "--output-json",
            str(output),
            "--log-file",
            str(log),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert result == payload
    assert payload["models"] == [
        {
            "model_id": model_id,
            "revision": model_spec(model_id).revision,
            "slug": model_spec(model_id).slug,
        }
        for model_id in MODEL_MATRIX
    ]
    assert set(payload["per_model"]) == set(MODEL_MATRIX)
    assert "audit" in log.read_text(encoding="utf-8")


def test_cli_wraps_audit_once_and_notifies_on_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    events: list[tuple[str, object]] = []
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(module, "notify_on_exit", _fake_notification(events))

    def fake_audit(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(module, "audit", fake_audit)
    log = tmp_path / "audit.log"

    assert module.main(
        [
            "--formal-run-root",
            str(tmp_path / "formal-run"),
            "--output-json",
            str(tmp_path / "audit.json"),
            "--log-file",
            str(log),
        ]
    ) == {"status": "ok"}

    assert calls == [
        {
            "formal_run_root": tmp_path / "formal-run",
            "output_json": tmp_path / "audit.json",
            "log_file": log,
        }
    ]
    assert events == [
        ("enter", ("audit-benchmark-results", log)),
        ("exit", None),
    ]


def test_cli_notifies_on_failure_and_preserves_original_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    events: list[tuple[str, object]] = []
    monkeypatch.setattr(module, "notify_on_exit", _fake_notification(events))
    failure = ValueError("audit failed")

    def fail_audit(**kwargs: object) -> dict[str, object]:
        raise failure

    monkeypatch.setattr(module, "audit", fail_audit)
    log = tmp_path / "audit.log"

    with pytest.raises(ValueError) as raised:
        module.main(
            [
                "--formal-run-root",
                str(tmp_path / "formal-run"),
                "--output-json",
                str(tmp_path / "audit.json"),
                "--log-file",
                str(log),
            ]
        )

    assert raised.value is failure
    assert events == [
        ("enter", ("audit-benchmark-results", log)),
        ("exit", ValueError),
    ]


def test_metrics_cover_mmlu_parser_deltas_saturation_and_finish_reason(
    formal_run: Path, tmp_path: Path
) -> None:
    module = _module()
    payload = module.audit(
        formal_run_root=formal_run,
        output_json=tmp_path / "audit.json",
        log_file=tmp_path / "audit.log",
    )

    metrics = payload["per_model"][MODEL_MATRIX[0]]["mmlu_pro"]
    assert metrics["legacy_extracted_accuracy"] == pytest.approx(0.0)
    assert metrics["fixed_extracted_accuracy"] == pytest.approx(0.5)
    assert metrics["parser_transition_delta"] == pytest.approx(0.5)
    assert metrics["parser_correctness_delta"] == pytest.approx(0.5)
    assert metrics["generated_token_count_saturation"] == {
        "saturated": 1,
        "unsaturated": 1,
        "denominator": 2,
        "rate": pytest.approx(0.5),
    }
    assert metrics["finish_reason"] == {
        "available": 0,
        "unavailable": 2,
        "denominator": 2,
    }


def test_mmlu_current_parser_reuses_runner_parser_but_legacy_field_stays_frozen(
    formal_run: Path,
) -> None:
    module = _module()
    model_id = MODEL_MATRIX[0]
    spec = model_spec(model_id)
    response_path = (
        formal_run / "checkpoints" / spec.slug / "mmlu_pro" / "responses.jsonl"
    )
    rows = list(module._iter_jsonl(response_path))
    rows[0]["raw_response"] = {"text": r"Reasoning... \boxed{A}"}
    _write_jsonl(response_path, rows)

    assert parse_answer_letter(r"Reasoning... \boxed{A}") == "A"
    metrics = module._audit_benchmark(
        formal_run_root=formal_run, model_id=model_id, benchmark="mmlu_pro"
    )

    assert metrics["legacy_extracted_accuracy"] == pytest.approx(0.0)
    assert metrics["fixed_extracted_accuracy"] == pytest.approx(0.5)
    assert metrics["parser_transition_delta"] == pytest.approx(0.5)


def test_math_metrics_preserve_indeterminate_denominators_and_gemini_categories(
    formal_run: Path, tmp_path: Path
) -> None:
    module = _module()
    payload = module.audit(
        formal_run_root=formal_run,
        output_json=tmp_path / "audit.json",
        log_file=tmp_path / "audit.log",
    )

    metrics = payload["per_model"][MODEL_MATRIX[0]]["math500"]
    assert metrics["deterministic_extraction_status"] == {
        "answer": 1,
        "null": 1,
        "ambiguous": 0,
        "indeterminate": 0,
        "denominator": 2,
    }
    assert metrics["deterministic_equivalence_status"]["equivalent"] == 1
    assert metrics["deterministic_equivalence_status"]["indeterminate"] == 1
    assert metrics["gemini"]["correct"] == 1
    assert metrics["gemini"]["status"] == {"ok": 1, "unparseable": 1}
    assert metrics["agreement_categories"] == {
        "both_correct": 1,
        "both_incorrect": 0,
        "deterministic_only": 0,
        "gemini_only": 0,
        "both_indeterminate": 1,
        "denominator": 2,
    }
    assert metrics["generated_token_count_saturation"]["denominator"] == 2
    assert metrics["deterministic_accuracy_denominator"] == 1
    assert metrics["gemini_accuracy_denominator"] == 1
    assert metrics["agreement_denominator"] == 1


def test_math_agreement_has_four_determinate_cells_and_separate_indeterminate_denominators(
    formal_run: Path,
) -> None:
    module = _module()
    model_id = MODEL_MATRIX[0]
    spec = model_spec(model_id)
    response_path = (
        formal_run / "checkpoints" / spec.slug / "math500" / "responses.jsonl"
    )
    score_path = formal_run / "results" / spec.slug / "math500" / "scores.jsonl"
    response_rows: list[dict[str, object]] = [
        {
            "id": "math-both-correct",
            "benchmark": "math500",
            "raw_response": {"text": r"\boxed{1}"},
            "gold": "1",
        },
        {
            "id": "math-deterministic-only",
            "benchmark": "math500",
            "raw_response": {"text": r"\boxed{2}"},
            "gold": "2",
        },
        {
            "id": "math-gemini-only",
            "benchmark": "math500",
            "raw_response": {"text": "No final answer"},
            "gold": "3",
        },
        {
            "id": "math-both-incorrect",
            "benchmark": "math500",
            "raw_response": {"text": r"\boxed{4}"},
            "gold": "5",
        },
        {
            "id": "math-indeterminate",
            "benchmark": "math500",
            "raw_response": {"text": r"\boxed{\sqrt{2}}"},
            "gold": "2",
        },
    ]
    score_rows: list[dict[str, object]] = [
        {
            "id": "math-both-correct",
            "benchmark": "math500",
            "status": "ok",
            "answer_correct": True,
        },
        {
            "id": "math-deterministic-only",
            "benchmark": "math500",
            "status": "ok",
            "answer_correct": False,
        },
        {
            "id": "math-gemini-only",
            "benchmark": "math500",
            "status": "ok",
            "answer_correct": True,
        },
        {
            "id": "math-both-incorrect",
            "benchmark": "math500",
            "status": "ok",
            "answer_correct": False,
        },
        {
            "id": "math-indeterminate",
            "benchmark": "math500",
            "status": "unparseable",
            "answer_correct": False,
        },
    ]
    _write_jsonl(response_path, response_rows)
    _write_jsonl(score_path, score_rows)

    metrics = module._audit_benchmark(
        formal_run_root=formal_run, model_id=model_id, benchmark="math500"
    )

    assert metrics["agreement_categories"] == {
        "both_correct": 1,
        "both_incorrect": 1,
        "deterministic_only": 1,
        "gemini_only": 1,
        "both_indeterminate": 1,
        "denominator": 5,
    }
    assert metrics["deterministic_accuracy_denominator"] == 4
    assert metrics["gemini_accuracy_denominator"] == 4
    assert metrics["agreement_denominator"] == 4
    assert metrics["deterministic_equivalence_status"]["indeterminate"] == 1


def test_hashes_stream_both_jsonl_files_and_rejects_stale_bindings(
    formal_run: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    original_read_text = Path.read_text

    def reject_jsonl_materialization(self: Path, *args: Any, **kwargs: Any) -> str:
        if self.suffix == ".jsonl":
            raise AssertionError("JSONL must be streamed")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", reject_jsonl_materialization)
    module.audit(
        formal_run_root=formal_run,
        output_json=tmp_path / "audit.json",
        log_file=tmp_path / "audit.log",
    )

    stale = next(formal_run.glob("results/*/math500/scores.jsonl"))
    stale.write_bytes(stale.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA|digest|stale|binding"):
        module.audit(
            formal_run_root=formal_run,
            output_json=tmp_path / "stale.json",
            log_file=tmp_path / "stale.log",
        )


def _write_retained_run_manifests_without_bindings(formal_run: Path) -> None:
    for model_id in MODEL_MATRIX:
        spec = model_spec(model_id)
        checkpoint_manifest = formal_run / "checkpoints" / spec.slug / "manifest.json"
        checkpoint_manifest.write_text(
            json.dumps(
                {
                    "config": {
                        "model_id": model_id,
                        "model_revision": spec.revision,
                    }
                }
            ),
            encoding="utf-8",
        )
        response_files: dict[str, object] = {}
        score_files: dict[str, object] = {}
        for benchmark in DATASET_SCOPES:
            response_path = (
                formal_run / "checkpoints" / spec.slug / benchmark / "responses.jsonl"
            )
            score_path = formal_run / "results" / spec.slug / benchmark / "scores.jsonl"
            response_files[benchmark] = {
                "path": str(response_path),
                "content_sha256": hashlib.sha256(
                    response_path.read_bytes()
                ).hexdigest(),
                "ids": [row["id"] for row in _module()._iter_jsonl(response_path)],
            }
            score_files[benchmark] = {
                "path": str(score_path),
                "content_sha256": hashlib.sha256(score_path.read_bytes()).hexdigest(),
            }
            (score_path.parent / "bindings.json").unlink()
        scoring_manifest = formal_run / "scoring" / spec.slug / "scoring_manifest.json"
        scoring_manifest.parent.mkdir(parents=True, exist_ok=True)
        scoring_manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "model_id": model_id,
                    "model_slug": spec.slug,
                    "response_files": response_files,
                    "score_files": score_files,
                }
            ),
            encoding="utf-8",
        )


def test_audit_accepts_real_retained_manifests_when_bindings_are_absent(
    formal_run: Path, tmp_path: Path
) -> None:
    module = _module()
    _write_retained_run_manifests_without_bindings(formal_run)

    payload = module.audit(
        formal_run_root=formal_run,
        output_json=tmp_path / "audit.json",
        log_file=tmp_path / "audit.log",
    )

    assert set(payload["per_model"]) == set(MODEL_MATRIX)


def test_retained_fallback_rejects_duplicate_response_ids_before_metrics(
    formal_run: Path, tmp_path: Path
) -> None:
    module = _module()
    _write_retained_run_manifests_without_bindings(formal_run)
    model_id = MODEL_MATRIX[0]
    spec = model_spec(model_id)
    response_path = (
        formal_run / "checkpoints" / spec.slug / "mmlu_pro" / "responses.jsonl"
    )
    score_path = formal_run / "results" / spec.slug / "mmlu_pro" / "scores.jsonl"
    rows = list(module._iter_jsonl(response_path))[:1]
    _write_jsonl(response_path, [rows[0], {**rows[0]}])
    _write_jsonl(score_path, list(module._iter_jsonl(score_path))[:1])

    scoring_manifest = formal_run / "scoring" / spec.slug / "scoring_manifest.json"
    payload = json.loads(scoring_manifest.read_text(encoding="utf-8"))
    payload["response_files"]["mmlu_pro"] = {
        "path": str(response_path),
        "content_sha256": hashlib.sha256(response_path.read_bytes()).hexdigest(),
        "ids": [str(rows[0]["id"]), str(rows[0]["id"])],
    }
    payload["score_files"]["mmlu_pro"]["content_sha256"] = hashlib.sha256(
        score_path.read_bytes()
    ).hexdigest()
    scoring_manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate|cardinality|ids"):
        module.audit(
            formal_run_root=formal_run,
            output_json=tmp_path / "audit.json",
            log_file=tmp_path / "audit.log",
        )


def test_retained_fallback_rejects_mutated_score_after_bindings_are_removed(
    formal_run: Path, tmp_path: Path
) -> None:
    module = _module()
    _write_retained_run_manifests_without_bindings(formal_run)
    model_id = MODEL_MATRIX[0]
    spec = model_spec(model_id)
    score_path = formal_run / "results" / spec.slug / "math500" / "scores.jsonl"
    rows = list(module._iter_jsonl(score_path))
    rows[0]["answer_correct"] = not rows[0]["answer_correct"]
    _write_jsonl(score_path, rows)

    with pytest.raises(ValueError, match="score|digest|binding|SHA"):
        module.audit(
            formal_run_root=formal_run,
            output_json=tmp_path / "audit.json",
            log_file=tmp_path / "audit.log",
        )


def test_retained_fallback_rejects_missing_score_binding(
    formal_run: Path, tmp_path: Path
) -> None:
    module = _module()
    _write_retained_run_manifests_without_bindings(formal_run)
    spec = model_spec(MODEL_MATRIX[0])
    scoring_manifest = formal_run / "scoring" / spec.slug / "scoring_manifest.json"
    payload = json.loads(scoring_manifest.read_text(encoding="utf-8"))
    del payload["score_files"]["math500"]
    scoring_manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="score|digest|binding|manifest"):
        module.audit(
            formal_run_root=formal_run,
            output_json=tmp_path / "audit.json",
            log_file=tmp_path / "audit.log",
        )


@pytest.mark.parametrize("stale_field", ["model_id", "revision", "response_sha256"])
def test_retained_manifest_fallback_rejects_stale_identity_or_digest(
    formal_run: Path, tmp_path: Path, stale_field: str
) -> None:
    module = _module()
    _write_retained_run_manifests_without_bindings(formal_run)
    spec = model_spec(MODEL_MATRIX[0])
    scoring_manifest = formal_run / "scoring" / spec.slug / "scoring_manifest.json"
    payload = json.loads(scoring_manifest.read_text(encoding="utf-8"))
    if stale_field == "model_id":
        payload["model_id"] = "stale/model"
    elif stale_field == "revision":
        checkpoint = formal_run / "checkpoints" / spec.slug / "manifest.json"
        checkpoint_payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        checkpoint_payload["config"]["model_revision"] = "stale-revision"
        checkpoint.write_text(json.dumps(checkpoint_payload), encoding="utf-8")
    elif stale_field == "response_sha256":
        payload["response_files"]["mmlu_pro"]["content_sha256"] = "stale-response"
    scoring_manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="stale|digest|binding|SHA"):
        module.audit(
            formal_run_root=formal_run,
            output_json=tmp_path / "audit.json",
            log_file=tmp_path / "audit.log",
        )


def test_checkpoint_is_atomic_per_model_benchmark_and_revalidates_completed_units(
    formal_run: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    output = tmp_path / "audit.json"
    log = tmp_path / "audit.log"
    module.audit(formal_run_root=formal_run, output_json=output, log_file=log)

    checkpoint = tmp_path / "audit.checkpoint.json"
    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert saved["completed_units"] == [
        {"model_id": model_id, "benchmark": benchmark}
        for model_id in MODEL_MATRIX
        for benchmark in DATASET_SCOPES
    ]
    assert saved["source_sha256"]
    assert all(
        unit["metrics_schema"] == module.METRICS_SCHEMA
        and unit["metrics_sha256"] == module._metrics_sha256(unit["metrics"])
        for unit in saved["units"]
    )
    assert not list(tmp_path.glob(".audit.checkpoint.json.*"))

    calls: list[tuple[str, str]] = []
    original = module._audit_benchmark

    def record_call(*args: Any, **kwargs: Any) -> Any:
        calls.append((str(kwargs["model_id"]), str(kwargs["benchmark"])))
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_audit_benchmark", record_call)
    module.audit(formal_run_root=formal_run, output_json=output, log_file=log)
    assert calls == []

    response = next(formal_run.glob("checkpoints/*/mmlu_pro/responses.jsonl"))
    response.write_bytes(response.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA|digest|stale|source"):
        module.audit(formal_run_root=formal_run, output_json=output, log_file=log)


@pytest.mark.parametrize("mutation", ["count", "generated_token_count_saturation"])
def test_checkpoint_recomputes_tampered_metrics_from_bound_sources(
    formal_run: Path,
    tmp_path: Path,
    mutation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    output = tmp_path / "audit.json"
    log = tmp_path / "audit.log"
    module.audit(formal_run_root=formal_run, output_json=output, log_file=log)

    checkpoint = tmp_path / "audit.checkpoint.json"
    checkpoint_payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    unit = checkpoint_payload["units"][0]
    if mutation == "count":
        unit["metrics"]["count"] = 999
    else:
        unit["metrics"]["generated_token_count_saturation"]["denominator"] = 999
    checkpoint.write_text(json.dumps(checkpoint_payload), encoding="utf-8")

    calls: list[tuple[str, str]] = []
    original = module._audit_benchmark

    def record_call(*args: Any, **kwargs: Any) -> Any:
        calls.append((str(kwargs["model_id"]), str(kwargs["benchmark"])))
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_audit_benchmark", record_call)
    resumed = module.audit(formal_run_root=formal_run, output_json=output, log_file=log)
    metrics = resumed["per_model"][unit["model_id"]][unit["benchmark"]]
    assert metrics["count"] == 2
    assert metrics["generated_token_count_saturation"]["denominator"] == 2
    assert calls == [(str(unit["model_id"]), str(unit["benchmark"]))]


def test_checkpoint_recomputes_legacy_unit_without_metric_metadata(
    formal_run: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    output = tmp_path / "audit.json"
    log = tmp_path / "audit.log"
    module.audit(formal_run_root=formal_run, output_json=output, log_file=log)

    checkpoint = tmp_path / "audit.checkpoint.json"
    checkpoint_payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    unit = checkpoint_payload["units"][0]
    del unit["metrics_sha256"]
    del unit["metrics_schema"]
    checkpoint.write_text(json.dumps(checkpoint_payload), encoding="utf-8")

    calls: list[tuple[str, str]] = []
    original = module._audit_benchmark

    def record_call(*args: Any, **kwargs: Any) -> Any:
        calls.append((str(kwargs["model_id"]), str(kwargs["benchmark"])))
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_audit_benchmark", record_call)
    module.audit(formal_run_root=formal_run, output_json=output, log_file=log)

    assert calls == [(str(unit["model_id"]), str(unit["benchmark"]))]


def test_checkpoint_rejects_invalid_metric_schema_instead_of_emitting_it(
    formal_run: Path, tmp_path: Path
) -> None:
    module = _module()
    output = tmp_path / "audit.json"
    log = tmp_path / "audit.log"
    module.audit(formal_run_root=formal_run, output_json=output, log_file=log)

    checkpoint = tmp_path / "audit.checkpoint.json"
    checkpoint_payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    unit = checkpoint_payload["units"][0]
    del unit["metrics"]["count"]
    checkpoint.write_text(json.dumps(checkpoint_payload), encoding="utf-8")

    resumed = module.audit(formal_run_root=formal_run, output_json=output, log_file=log)
    metrics = resumed["per_model"][unit["model_id"]][unit["benchmark"]]
    assert metrics["count"] == 2


def test_atomic_write_fsyncs_checkpoint_and_output_parent_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    fsync_calls: list[int] = []
    directory_opens: list[Path] = []
    original_open = module.os.open

    def record_directory_open(path: str | Path, flags: int, *args: Any) -> int:
        if flags & getattr(module.os, "O_DIRECTORY", 0):
            directory_opens.append(Path(path))
        return original_open(path, flags, *args)

    monkeypatch.setattr(module.os, "open", record_directory_open)
    monkeypatch.setattr(
        module.os, "fsync", lambda descriptor: fsync_calls.append(descriptor)
    )

    checkpoint = tmp_path / "checkpoint" / "audit.checkpoint.json"
    output = tmp_path / "output" / "audit.json"
    module._atomic_write(checkpoint, {"kind": "checkpoint"})
    module._atomic_write(output, {"kind": "output"})

    assert directory_opens == [checkpoint.parent, output.parent]
    assert len(fsync_calls) == 4


def test_atomic_write_cleans_temporary_file_when_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    destination = tmp_path / "audit.json"

    def fail_replace(*args: Any, **kwargs: Any) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        module._atomic_write(destination, {"status": "failed"})

    assert not list(destination.parent.glob(f".{destination.name}.*"))


def test_audit_rejects_symlinked_formal_root(formal_run: Path, tmp_path: Path) -> None:
    module = _module()
    linked_root = tmp_path / "formal-run-link"
    linked_root.symlink_to(formal_run, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        module.audit(
            formal_run_root=linked_root,
            output_json=tmp_path / "audit.json",
            log_file=tmp_path / "audit.log",
        )


@pytest.mark.parametrize("component", ["checkpoints", "results"])
def test_audit_rejects_symlinked_artifact_directory(
    formal_run: Path, tmp_path: Path, component: str
) -> None:
    module = _module()
    target = formal_run / f"real-{component}"
    (formal_run / component).rename(target)
    (formal_run / component).symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        module.audit(
            formal_run_root=formal_run,
            output_json=tmp_path / "audit.json",
            log_file=tmp_path / "audit.log",
        )


@pytest.mark.parametrize("artifact", ["response", "score"])
def test_audit_rejects_symlinked_jsonl_artifact(
    formal_run: Path, tmp_path: Path, artifact: str
) -> None:
    module = _module()
    spec = model_spec(MODEL_MATRIX[0])
    if artifact == "response":
        path = formal_run / "checkpoints" / spec.slug / "mmlu_pro" / "responses.jsonl"
    else:
        path = formal_run / "results" / spec.slug / "mmlu_pro" / "scores.jsonl"
    target = tmp_path / path.name
    target.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        module.audit(
            formal_run_root=formal_run,
            output_json=tmp_path / "audit.json",
            log_file=tmp_path / "audit.log",
        )


def test_audit_rejects_symlinked_discovered_binding_and_retained_manifests(
    formal_run: Path, tmp_path: Path
) -> None:
    module = _module()
    model_id = MODEL_MATRIX[0]
    spec = model_spec(model_id)
    binding = formal_run / "results" / spec.slug / "mmlu_pro" / "bindings.json"
    binding_target = tmp_path / "binding.json"
    binding_target.write_bytes(binding.read_bytes())
    binding.unlink()
    binding.symlink_to(binding_target)

    with pytest.raises(ValueError, match="symlink"):
        module.audit(
            formal_run_root=formal_run,
            output_json=tmp_path / "binding-audit.json",
            log_file=tmp_path / "binding-audit.log",
        )

    _write_retained_run_manifests_without_bindings(formal_run)
    checkpoint_manifest = formal_run / "checkpoints" / spec.slug / "manifest.json"
    manifest_target = tmp_path / "manifest.json"
    manifest_target.write_bytes(checkpoint_manifest.read_bytes())
    checkpoint_manifest.unlink()
    checkpoint_manifest.symlink_to(manifest_target)

    with pytest.raises(ValueError, match="symlink"):
        module.audit(
            formal_run_root=formal_run,
            output_json=tmp_path / "manifest-audit.json",
            log_file=tmp_path / "manifest-audit.log",
        )


@pytest.mark.parametrize("kind", ["output", "log", "checkpoint"])
def test_audit_rejects_symlinked_persistence_paths(
    formal_run: Path, tmp_path: Path, kind: str
) -> None:
    module = _module()
    output = tmp_path / "audit.json"
    log = tmp_path / "audit.log"
    module.audit(formal_run_root=formal_run, output_json=output, log_file=log)

    if kind == "output":
        target = tmp_path / "output-target.json"
        target.write_bytes(output.read_bytes())
        output.unlink()
        output.symlink_to(target)
    elif kind == "log":
        target = tmp_path / "log-target.log"
        target.write_bytes(log.read_bytes())
        log.unlink()
        log.symlink_to(target)
    else:
        checkpoint = tmp_path / "audit.checkpoint.json"
        target = tmp_path / "checkpoint-target.json"
        target.write_bytes(checkpoint.read_bytes())
        checkpoint.unlink()
        checkpoint.symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        module.audit(formal_run_root=formal_run, output_json=output, log_file=log)
