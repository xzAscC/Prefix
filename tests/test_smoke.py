from __future__ import annotations

import json
import importlib
import math
from pathlib import Path

import pytest

from prefix import no_steering


validate_smoke_result = importlib.import_module("prefix.smoke").validate_smoke_result


MODEL_ID = no_steering.MODEL_MATRIX[0]
REVISION = no_steering.model_spec(MODEL_ID).revision
BENCHMARKS = {"math500": 1}


def _write_result(
    root: Path,
    *,
    model_id: str = MODEL_ID,
    revision: str = REVISION,
    status: str = "ok",
    generated_token_count: int = 2,
    logprobs: list[float] | None = None,
    summary: dict[str, object] | None = None,
) -> None:
    benchmark = root / "math500"
    benchmark.mkdir(parents=True)
    row = {
        "id": "example-1",
        "benchmark": "math500",
        "model_id": model_id,
        "raw_response": "answer",
        "status": status,
        "generated_token_count": generated_token_count,
        "selected_generated_token_logprobs": (
            ["-0.1", "-0.2"] if logprobs is None else logprobs
        ),
        "metadata": {"provenance": {"model_id": model_id, "revision": revision}},
    }
    (benchmark / "responses.jsonl").write_text(json.dumps(row) + "\n")
    payload = {
        "provenance": {"model_id": model_id, "revision": revision},
        "ppl": {
            "ppl": math.exp(0.15),
            "selected_token_count": 2,
            "generated_token_count": 2,
            "covered_records": 1,
            "total_records": 1,
            "coverage_ratio": 1.0,
        },
        "scores": {"math500": {"count": 1}},
    }
    (root / "summary.json").write_text(json.dumps(summary or payload))


def test_validates_successful_smoke_and_exact_token_coverage(tmp_path: Path) -> None:
    _write_result(tmp_path)

    result = validate_smoke_result(
        result_root=tmp_path,
        model_id=MODEL_ID,
        expected_benchmarks=BENCHMARKS,
    )

    assert result.model_id == MODEL_ID
    assert result.records_by_benchmark == BENCHMARKS
    assert result.response_paths == (
        (tmp_path / "math500" / "responses.jsonl").resolve(),
    )


@pytest.mark.parametrize(
    "change, match",
    [
        (lambda row: row.update(status="error"), "error"),
        (lambda row: row.update(model_id="wrong/model"), "model_id"),
        (
            lambda row: row["metadata"]["provenance"].update(revision="0" * 40),
            "revision",
        ),
        (lambda row: row.update(generated_token_count=3), "coverage"),
        (lambda row: row.update(selected_generated_token_logprobs=[]), "coverage"),
    ],
)
def test_rejects_invalid_response_rows(tmp_path: Path, change, match: str) -> None:
    _write_result(tmp_path)
    response = tmp_path / "math500" / "responses.jsonl"
    row = json.loads(response.read_text())
    change(row)
    response.write_text(json.dumps(row) + "\n")

    with pytest.raises(ValueError, match=match):
        validate_smoke_result(
            result_root=tmp_path,
            model_id=MODEL_ID,
            expected_benchmarks=BENCHMARKS,
        )


def test_rejects_missing_benchmark_and_summary(tmp_path: Path) -> None:
    with pytest.raises(
        (FileNotFoundError, ValueError), match="summary|responses|benchmark"
    ):
        validate_smoke_result(
            result_root=tmp_path,
            model_id=MODEL_ID,
            expected_benchmarks=BENCHMARKS,
        )


def test_rejects_summary_count_mismatch(tmp_path: Path) -> None:
    _write_result(
        tmp_path, summary={"provenance": {"model_id": MODEL_ID, "revision": REVISION}}
    )

    with pytest.raises(ValueError, match="summary|count"):
        validate_smoke_result(
            result_root=tmp_path,
            model_id=MODEL_ID,
            expected_benchmarks=BENCHMARKS,
        )


@pytest.mark.parametrize(
    "field",
    [
        "ppl",
        "selected_token_count",
        "generated_token_count",
        "covered_records",
        "total_records",
        "coverage_ratio",
    ],
)
def test_rejects_missing_canonical_ppl_field(tmp_path: Path, field: str) -> None:
    _write_result(tmp_path)
    summary_path = tmp_path / "summary.json"
    summary = json.loads(summary_path.read_text())
    del summary["ppl"][field]
    summary_path.write_text(json.dumps(summary))

    with pytest.raises(ValueError, match="PPL|coverage|count"):
        validate_smoke_result(
            result_root=tmp_path,
            model_id=MODEL_ID,
            expected_benchmarks=BENCHMARKS,
        )


def test_rejects_legacy_ppl_fields(tmp_path: Path) -> None:
    _write_result(tmp_path)
    summary_path = tmp_path / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["ppl"] = {"records": 1, "selected_generated_token_count": 2}
    summary_path.write_text(json.dumps(summary))

    with pytest.raises(ValueError, match="PPL|coverage|count"):
        validate_smoke_result(
            result_root=tmp_path,
            model_id=MODEL_ID,
            expected_benchmarks=BENCHMARKS,
        )


@pytest.mark.parametrize(
    "field, value",
    [
        ("ppl", 0.0),
        ("selected_token_count", 1),
        ("generated_token_count", 3),
        ("covered_records", 0),
        ("total_records", 2),
        ("coverage_ratio", 0.5),
    ],
)
def test_rejects_incorrect_canonical_ppl_value(
    tmp_path: Path, field: str, value: object
) -> None:
    _write_result(tmp_path)
    summary_path = tmp_path / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["ppl"][field] = value
    summary_path.write_text(json.dumps(summary))

    with pytest.raises(ValueError, match="PPL|coverage|count"):
        validate_smoke_result(
            result_root=tmp_path,
            model_id=MODEL_ID,
            expected_benchmarks=BENCHMARKS,
        )
