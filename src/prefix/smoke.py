from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from .no_steering import MODEL_MATRIX, model_spec


@dataclass(frozen=True, slots=True)
class SmokeValidation:
    model_id: str
    summary_path: Path
    response_paths: tuple[Path, ...]
    records_by_benchmark: Mapping[str, int]


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"missing smoke output: {path}") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON smoke output: {path}") from error


def _read_rows(path: Path) -> list[dict[str, object]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"missing smoke responses: {path}") from None
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid response JSON at {path}:{line_number}"
            ) from error
        if not isinstance(row, dict):
            raise ValueError(f"response row must be an object at {path}:{line_number}")
        rows.append(row)
    return rows


def _validate_logprob_coverage(row: Mapping[str, object], path: Path) -> None:
    count = row.get("generated_token_count")
    logprobs = row.get("selected_generated_token_logprobs")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError(f"generated-token coverage is invalid in {path}")
    if not isinstance(logprobs, list) or len(logprobs) != count:
        raise ValueError(f"response-token logprob coverage mismatch in {path}")
    for value in logprobs:
        try:
            number = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"response-token logprob coverage is invalid in {path}"
            ) from error
        if not math.isfinite(number) or number > 0:
            raise ValueError(f"response-token logprob coverage is invalid in {path}")


def validate_smoke_result(
    *,
    result_root: str | Path,
    model_id: str,
    expected_benchmarks: Mapping[str, int],
) -> SmokeValidation:
    """Validate one completed GPU smoke result without loading a model or using a device."""
    spec = model_spec(model_id)
    root = Path(result_root)
    summary_path = root / "summary.json"
    summary = _read_json(summary_path)
    if not isinstance(summary, dict):
        raise ValueError("summary must be a JSON object")
    provenance = summary.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("summary is missing provenance")
    if provenance.get("model_id") != model_id:
        raise ValueError("summary model_id does not match expected model")
    if provenance.get("revision") != spec.revision:
        raise ValueError("summary revision does not match pinned revision")

    rows: list[dict[str, object]] = []
    response_paths: list[Path] = []
    for benchmark, expected_count in expected_benchmarks.items():
        if not isinstance(expected_count, int) or expected_count < 1:
            raise ValueError(f"invalid expected benchmark count for {benchmark}")
        response_path = (root / benchmark / "responses.jsonl").resolve()
        response_paths.append(response_path)
        benchmark_rows = _read_rows(response_path)
        if len(benchmark_rows) != expected_count:
            raise ValueError(f"benchmark {benchmark} has incomplete response coverage")
        rows.extend(benchmark_rows)

    ids: list[str] = []
    for row in rows:
        if row.get("status") != "ok":
            raise ValueError("smoke output contains error-only or failed response rows")
        if row.get("model_id") != model_id:
            raise ValueError("response model_id does not match expected model")
        if row.get("benchmark") not in expected_benchmarks:
            raise ValueError("response contains an unexpected benchmark")
        provenance = row.get("metadata", {})
        if not isinstance(provenance, dict) or not isinstance(
            provenance.get("provenance"), dict
        ):
            raise ValueError("response is missing provenance")
        row_provenance = provenance["provenance"]
        if row_provenance.get("model_id") != model_id:
            raise ValueError("response provenance model_id does not match pinned model")
        if row_provenance.get("revision") != spec.revision:
            raise ValueError(
                "response provenance revision does not match pinned revision"
            )
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in ids:
            raise ValueError("smoke responses contain missing or duplicate ids")
        ids.append(identifier)
        _validate_logprob_coverage(row, summary_path)

    actual_counts = Counter(str(row["benchmark"]) for row in rows)
    if dict(actual_counts) != dict(expected_benchmarks):
        raise ValueError("smoke benchmark coverage is partial")

    ppl = summary.get("ppl")
    if not isinstance(ppl, dict):
        raise ValueError("summary is missing canonical PPL coverage")
    required_ppl_fields = {
        "ppl",
        "selected_token_count",
        "generated_token_count",
        "covered_records",
        "total_records",
        "coverage_ratio",
    }
    if not required_ppl_fields.issubset(ppl):
        raise ValueError("summary is missing canonical PPL coverage fields")
    ppl_value = ppl["ppl"]
    selected_token_count = ppl["selected_token_count"]
    generated_token_count = ppl["generated_token_count"]
    covered_records = ppl["covered_records"]
    total_records = ppl["total_records"]
    coverage_ratio = ppl["coverage_ratio"]
    expected_generated_tokens = sum(
        cast(int, row["generated_token_count"]) for row in rows
    )
    expected_selected_tokens = sum(
        len(cast(list[object], row["selected_generated_token_logprobs"]))
        for row in rows
    )
    if (
        isinstance(ppl_value, bool)
        or not isinstance(ppl_value, (int, float))
        or not math.isfinite(float(ppl_value))
        or float(ppl_value) <= 0
        or isinstance(selected_token_count, bool)
        or not isinstance(selected_token_count, int)
        or selected_token_count != expected_selected_tokens
        or isinstance(generated_token_count, bool)
        or not isinstance(generated_token_count, int)
        or generated_token_count != expected_generated_tokens
        or isinstance(covered_records, bool)
        or not isinstance(covered_records, int)
        or covered_records != len(rows)
        or isinstance(total_records, bool)
        or not isinstance(total_records, int)
        or total_records != len(rows)
        or isinstance(coverage_ratio, bool)
        or not isinstance(coverage_ratio, (int, float))
        or not math.isfinite(float(coverage_ratio))
        or float(coverage_ratio) != covered_records / total_records
    ):
        raise ValueError("summary canonical PPL coverage does not match responses")

    return SmokeValidation(
        model_id, summary_path, tuple(response_paths), dict(actual_counts)
    )


def validate_smoke_matrix(
    *,
    result_roots: Mapping[str, str | Path],
    expected_benchmarks: Mapping[str, int],
) -> tuple[SmokeValidation, ...]:
    """Validate the exact pinned model set, in matrix order."""
    if tuple(result_roots) != MODEL_MATRIX:
        raise ValueError(
            "smoke result matrix does not contain the exact required models"
        )
    return tuple(
        validate_smoke_result(
            result_root=result_roots[model_id],
            model_id=model_id,
            expected_benchmarks=expected_benchmarks,
        )
        for model_id in result_roots
    )


__all__ = ["SmokeValidation", "validate_smoke_matrix", "validate_smoke_result"]
