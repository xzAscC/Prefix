from __future__ import annotations

import argparse
import hashlib
import importlib
from importlib import util
import json
import os
import tempfile
from pathlib import Path
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

from prefix import no_steering
from prefix.data import load_harmbench, load_math500, load_mmlu_pro
from prefix.judge import (
    GeminiJudge,
    JudgeBlocked,
    JudgeParseError,
    JudgeRetryableError,
)
from prefix.notify import notify_on_exit
from prefix.runner import (
    append_jsonl,
    chat_prompt,
    get_engine,
    mmlu_prompt,
    prepare_checkpoint_manifest,
    parse_answer_letter,
    read_jsonl,
    tee_stdout,
    write_json_atomic,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_MATRIX = no_steering.MODEL_MATRIX
DATASET_SCOPES = no_steering.DATASET_SCOPES
PROMPT_TEMPLATE_VERSION = "chat-v1"
DEFAULT_JUDGE_MODEL = "gemini-3.5-flash-lite"
SCORING_SCHEMA_VERSION = 1
SCORING_RUBRIC_VERSION = "gemini-rubric-v1"


def model_spec(model_id: str) -> no_steering.ModelSpec:
    return no_steering.model_spec(model_id)


def _check_model_snapshots() -> Any:
    try:
        module = importlib.import_module("scripts.no_steering_preflight")
    except ModuleNotFoundError as error:
        if error.name != "scripts":
            raise
        module_spec = util.spec_from_file_location(
            "no_steering_preflight",
            Path(__file__).with_name("no_steering_preflight.py"),
        )
        if module_spec is None or module_spec.loader is None:
            raise ImportError("cannot load no_steering_preflight")
        module = util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    return module.check_model_snapshots


def _stable_record_id(
    benchmark: str, position: int, record: Mapping[str, object]
) -> str:
    supplied = record.get("id")
    if supplied is not None:
        identifier = str(supplied)
        if identifier:
            return identifier
    payload = json.dumps(
        {str(key): value for key, value in record.items() if key != "id"},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:16]
    return f"{benchmark}-{position}-{digest}"


def _validate_benchmark_records(
    benchmark: str, records: Sequence[Mapping[str, object]]
) -> list[dict[str, object]]:
    scope = DATASET_SCOPES[benchmark]
    expected = int(cast(int, scope["count"]))
    if len(records) != expected:
        raise ValueError(
            f"{benchmark} loaded {len(records)} records; expected {expected}"
        )

    normalized: list[dict[str, object]] = []
    identifiers: set[str] = set()
    for position, record in enumerate(records):
        normalized_record = dict(record)
        identifier = _stable_record_id(benchmark, position, normalized_record)
        if identifier in identifiers:
            raise ValueError(f"{benchmark} contains duplicate record id: {identifier}")
        identifiers.add(identifier)
        normalized_record["id"] = identifier
        normalized.append(normalized_record)
    return normalized


def _require_unique_ids(rows: Sequence[Mapping[str, object]], label: str) -> None:
    identifiers: set[str] = set()
    for row in rows:
        if "id" not in row:
            raise ValueError(f"{label} row is missing an id")
        identifier = str(row["id"])
        if identifier in identifiers:
            raise ValueError(f"{label} contains duplicate id: {identifier}")
        identifiers.add(identifier)


def benchmark_manifest(
    benchmarks: Mapping[str, Sequence[Mapping[str, object]]], *, limited: bool = False
) -> dict[str, dict[str, object]]:
    """Describe loaded coverage for checkpoint manifests without changing records."""
    return {
        benchmark: {
            "split": scope["split"],
            "expected_count": int(cast(int, scope["count"])),
            "loaded_count": len(benchmarks.get(benchmark, ())),
            "complete": len(benchmarks.get(benchmark, ()))
            == int(cast(int, scope["count"]))
            and not limited,
            "limited": limited,
        }
        for benchmark, scope in DATASET_SCOPES.items()
    }


def checkpoint_manifest(
    benchmarks: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    model_id: str,
    sampling: Mapping[str, object],
    batch_prompts: int,
    max_model_len: int,
    output_roots: Mapping[str, object],
    limited: bool,
) -> dict[str, object]:
    spec = model_spec(model_id)
    return {
        "schema_version": 1,
        "model_id": spec.model_id,
        "model_slug": spec.slug,
        "model_revision": spec.revision,
        "steering": spec.steering,
        "quantization": spec.quantization or False,
        "benchmark_manifest": benchmark_manifest(benchmarks, limited=limited),
        "benchmark_ids": {
            name: [str(row["id"]) for row in rows] for name, rows in benchmarks.items()
        },
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "sampling": dict(sampling),
        "batch_prompts": batch_prompts,
        "max_model_len": max_model_len,
        "output_roots": dict(output_roots),
        "limited": limited,
    }


def load_benchmarks(
    *,
    load: Callable[..., list[dict[str, object]]] | None = None,
) -> dict[str, list[dict[str, object]]]:
    """Load the complete, pinned benchmark scopes in stable benchmark order."""
    if load is not None:
        return {
            name: _validate_benchmark_records(
                name,
                load(
                    name,
                    split=cast(str, scope["split"]),
                    count=int(cast(int, scope["count"])),
                ),
            )
            for name, scope in DATASET_SCOPES.items()
        }
    return {
        "harmbench": _validate_benchmark_records("harmbench", load_harmbench()),
        "mmlu_pro": _validate_benchmark_records("mmlu_pro", load_mmlu_pro("test")),
        "math500": _validate_benchmark_records("math500", load_math500()),
    }


def _json_native(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_native(item) for item in value]
    if not isinstance(value, (str, bytes, bytearray)) and hasattr(value, "tolist"):
        return _json_native(cast(Any, value).tolist())
    if hasattr(value, "__dict__"):
        return _json_native(vars(value))
    return str(value)


def _output_dict(raw_response: Mapping[str, Any]) -> Mapping[str, Any]:
    outputs = raw_response.get("outputs")
    if isinstance(outputs, Sequence) and outputs:
        first = outputs[0]
        if isinstance(first, Mapping):
            return first
    return {}


def _selected_logprobs(raw_response: Mapping[str, Any]) -> list[float]:
    direct = raw_response.get("selected_generated_token_logprobs")
    if isinstance(direct, Sequence) and not isinstance(direct, (str, bytes)):
        return [float(str(cast(Any, value))) for value in direct]
    output = _output_dict(raw_response)
    values = output.get("logprobs", raw_response.get("logprobs", []))
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return []
    token_ids = output.get("token_ids", raw_response.get("token_ids", []))
    if (
        values
        and isinstance(token_ids, Sequence)
        and not isinstance(token_ids, (str, bytes))
    ):
        if len(values) != len(token_ids):
            raise ValueError(
                "logprob length must match generated token_ids length "
                f"({len(values)} != {len(token_ids)})"
            )
    result: list[float] = []
    for index, value in enumerate(values):
        token_id = token_ids[index] if isinstance(token_ids, Sequence) else None
        if isinstance(value, Mapping) and token_id is not None:
            selected = value.get(token_id)
            if selected is None:
                selected = value.get(str(token_id))
            if selected is not None:
                value = selected
        if isinstance(value, Mapping):
            value = value.get("logprob")
        elif hasattr(value, "logprob"):
            value = getattr(cast(Any, value), "logprob")
        if value is not None:
            result.append(float(str(cast(Any, value))))
        else:
            raise ValueError(f"missing selected logprob for generated token {token_id}")
    return result


def _token_count(raw_response: Mapping[str, Any], key: str, fallback: int) -> int:
    value = raw_response.get(key)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return len(value)
    output = _output_dict(raw_response)
    value = output.get(key)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return len(value)
    return fallback


def _has_logprob_payload(raw_response: Mapping[str, Any]) -> bool:
    if (
        "selected_generated_token_logprobs" in raw_response
        or "logprobs" in raw_response
    ):
        return True
    output = _output_dict(raw_response)
    return "logprobs" in output


def response_only_ppl(rows: Sequence[Mapping[str, object]]) -> float:
    selected = [
        {"selected_logprobs": row.get("selected_generated_token_logprobs", [])}
        for row in rows
    ]
    return no_steering.conditional_ppl(selected)


def ppl_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    successful = [row for row in rows if row.get("status", "ok") == "ok"]
    selected_count = sum(
        len(cast(Sequence[object], row.get("selected_generated_token_logprobs", [])))
        for row in successful
    )
    generated_count = sum(
        int(cast(int, row.get("generated_token_count", 0))) for row in successful
    )
    covered_records = sum(
        bool(row.get("selected_generated_token_logprobs", [])) for row in successful
    )
    result: dict[str, object] = {
        "ppl": None,
        "selected_token_count": selected_count,
        "generated_token_count": generated_count,
        "covered_records": covered_records,
        "total_records": len(rows),
        "coverage_ratio": (
            selected_count / generated_count if generated_count else 0.0
        ),
    }
    complete = (
        bool(successful)
        and all(
            len(
                cast(Sequence[object], row.get("selected_generated_token_logprobs", []))
            )
            == int(cast(int, row.get("generated_token_count", 0)))
            and int(cast(int, row.get("generated_token_count", 0))) > 0
            for row in successful
        )
        and len(successful) == len(rows)
    )
    if selected_count and complete:
        result["ppl"] = response_only_ppl(successful)
    return result


def harmbench_label(label: str) -> str:
    """Keep Gemini's label verbatim; this is not canonical HarmBench ASR."""
    return label


def score_records(rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for benchmark in ("mmlu_pro", "math500"):
        selected = [
            row
            for row in rows
            if row.get("benchmark") == benchmark and row.get("status", "ok") == "ok"
        ]
        if benchmark == "mmlu_pro":
            correct = sum(
                row.get("extracted_answer") == row.get("gold") for row in selected
            )
        else:
            correct = sum(
                bool(
                    cast(Mapping[str, object], row.get("judge", {})).get(
                        "answer_correct"
                    )
                )
                if isinstance(row.get("judge", {}), Mapping)
                else False
                for row in selected
            )
        result[benchmark] = {
            "accuracy": correct / len(selected) if selected else None,
            "correct": correct,
            "count": len(selected),
        }
    labels = [
        str(row["gemini_label"])
        for row in rows
        if row.get("benchmark") == "harmbench"
        and row.get("status", "ok") == "ok"
        and row.get("gemini_label") is not None
    ]
    result["harmbench"] = {
        "canonical": False,
        "label": labels[0] if len(set(labels)) == 1 and labels else None,
        "labels": {label: labels.count(label) for label in sorted(set(labels))},
        "count": len(labels),
    }
    return result


def _response_text(row: Mapping[str, object]) -> str:
    raw = row.get("raw_response", {})
    if isinstance(raw, Mapping):
        return str(raw.get("text", ""))
    return ""


def score_responses(
    rows: Sequence[Mapping[str, object]],
    *,
    output_path: str | Path,
    judge: Any,
    judge_model: str | None = None,
) -> list[dict[str, object]]:
    existing = read_jsonl(output_path)
    _require_unique_ids(existing, "existing scores")
    _require_unique_ids(rows, "score input")
    done = {
        str(row["id"])
        for row in existing
        if "id" in row and not _score_row_is_retryable(row)
    }
    retry_ids = {str(row["id"]) for row in existing if _score_row_is_retryable(row)}
    original_order = [str(row["id"]) for row in existing]
    if retry_ids:
        _remove_rows_once(output_path, retry_ids)
    retry_order_ids = retry_ids
    retry_ids = set()
    written: list[dict[str, object]] = []
    for response in rows:
        identifier = str(response["id"])
        if identifier in done:
            continue
        row: dict[str, object] = {
            "id": identifier,
            "benchmark": response.get("benchmark"),
            "prompt": response.get("prompt", ""),
            "gold": response.get("gold"),
            "extracted_answer": response.get("extracted_answer"),
            "metadata": response.get("metadata", {}),
            "status": response.get("status", "ok"),
            "error": response.get("error"),
            "retryable": False,
            "judge_model": judge_model,
        }
        if response.get("status", "ok") != "ok":
            append_jsonl(output_path, [row])
            written.append(row)
            done.add(identifier)
            continue
        benchmark = str(response.get("benchmark", ""))
        try:
            if benchmark == "mmlu_pro":
                row["correct"] = response.get("extracted_answer") == response.get(
                    "gold"
                )
            elif benchmark == "harmbench":
                metadata = response.get("metadata", {})
                request = (
                    str(
                        cast(Mapping[str, object], metadata).get(
                            "behavior", response.get("prompt", "")
                        )
                    )
                    if isinstance(metadata, Mapping)
                    else str(response.get("prompt", ""))
                )
                safe = bool(judge.judge_safety(request, _response_text(response)))
                row.update(
                    {
                        "gemini_label": harmbench_label("safe" if safe else "unsafe"),
                        "canonical": False,
                    }
                )
            elif benchmark == "math500":
                result = judge.judge_math(
                    _response_text(response), str(response.get("gold", ""))
                )
                row.update(result)
            else:
                raise ValueError(f"unsupported benchmark for scoring: {benchmark}")
        except JudgeBlocked as error:
            row.update(
                {
                    "status": "blocked",
                    "blocked": True,
                    "error": str(error),
                    "retryable": False,
                }
            )
        except JudgeParseError as error:
            row.update(
                {
                    "status": "unparseable",
                    "unparseable": True,
                    "error": str(error),
                    "retryable": False,
                }
            )
        except JudgeRetryableError as error:
            row.update({"status": "error", "error": str(error), "retryable": True})
        except RuntimeError as error:
            row.update({"status": "error", "error": str(error), "retryable": True})
        except Exception as error:
            row.update({"status": "error", "error": str(error), "retryable": False})
        append_jsonl(output_path, [row])
        written.append(row)
        done.add(identifier)
    if retry_order_ids:
        current = read_jsonl(output_path)
        by_id = {str(row["id"]): row for row in current}
        original_order_set = set(original_order)
        ordered_ids = original_order + [
            str(row["id"])
            for row in current
            if str(row["id"]) not in original_order_set
        ]
        _rewrite_jsonl(
            output_path,
            [by_id[identifier] for identifier in ordered_ids if identifier in by_id],
        )
    return written


def _rewrite_jsonl(path: str | Path, rows: Sequence[Mapping[str, object]]) -> None:
    target = Path(path)
    _reject_persistence_symlinks(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _reject_persistence_symlinks(path: Path) -> None:
    current = Path(path.anchor) if path.is_absolute() else Path.cwd()
    for component in path.parts[1:] if path.is_absolute() else path.parts:
        current /= component
        if current.is_symlink():
            raise ValueError(
                f"refusing symlinked persistence path component: {current}"
            )


def _persist_generation_batch(
    output_path: str | Path,
    rows: Sequence[dict[str, object]],
    retry_ids: set[str],
) -> None:
    if not rows:
        return
    if not retry_ids.intersection(str(row["id"]) for row in rows):
        append_jsonl(output_path, list(rows))
        return
    current = read_jsonl(output_path)
    replacements = {str(row["id"]): row for row in rows}
    compacted: list[Mapping[str, object]] = []
    seen: set[str] = set()
    for row in current:
        identifier = str(row.get("id"))
        if identifier in seen:
            continue
        seen.add(identifier)
        compacted.append(replacements.get(identifier, row))
    for row in rows:
        if str(row["id"]) not in seen:
            compacted.append(row)
    _rewrite_jsonl(output_path, compacted)


def _remove_rows_once(output_path: str | Path, identifiers: set[str]) -> None:
    if not identifiers:
        return
    current = read_jsonl(output_path)
    _rewrite_jsonl(
        output_path,
        [row for row in current if str(row.get("id")) not in identifiers],
    )


def _score_row_is_retryable(row: Mapping[str, object]) -> bool:
    if row.get("status") != "error":
        return False
    if "retryable" not in row:
        return True
    return bool(row.get("retryable"))


def validate_score_completeness(
    rows: Sequence[Mapping[str, object]], *, limited: bool
) -> None:
    if limited:
        return
    for benchmark, scope in DATASET_SCOPES.items():
        selected = [row for row in rows if row.get("benchmark") == benchmark]
        expected = int(cast(int, scope["count"]))
        if len(selected) != expected:
            raise ValueError(
                f"{benchmark} denominator {len(selected)} does not match expected {expected}"
            )


def score_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for benchmark in ("mmlu_pro", "math500", "harmbench"):
        selected = [row for row in rows if row.get("benchmark") == benchmark]
        status_counts: dict[str, int] = {}
        for row in selected:
            status = str(row.get("status", "ok"))
            status_counts[status] = status_counts.get(status, 0) + 1
        correct = sum(
            bool(row.get("correct", row.get("answer_correct", False)))
            for row in selected
        )
        summary: dict[str, object] = {
            "correct": correct,
            "denominator": len(selected),
            "accuracy": correct / len(selected) if selected else None,
            "status_counts": status_counts,
        }
        if benchmark == "harmbench":
            unsafe = sum(row.get("gemini_label") == "unsafe" for row in selected)
            summary.update(
                {
                    "canonical": False,
                    "unsafe": unsafe,
                    "unsafe_rate": unsafe / len(selected) if selected else None,
                    "labels": {
                        label: sum(row.get("gemini_label") == label for row in selected)
                        for label in sorted(
                            {
                                str(row["gemini_label"])
                                for row in selected
                                if row.get("gemini_label") is not None
                            }
                        )
                    },
                }
            )
        result[benchmark] = summary
    return result


def pending_score_summary() -> dict[str, dict[str, object]]:
    return {
        benchmark: {
            "status": "pending",
            "correct": 0,
            "denominator": 0,
            "accuracy": None,
            "status_counts": {},
        }
        for benchmark in ("mmlu_pro", "math500", "harmbench")
    }


def generation_settings(
    max_tokens: int, max_model_len: int, gpu_memory_utilization: float
) -> dict[str, object]:
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    if max_model_len < max_tokens:
        raise ValueError("max_model_len must be at least max_tokens")
    if not 0.1 <= gpu_memory_utilization <= 0.95:
        raise ValueError("gpu_memory_utilization must be between 0.1 and 0.95")
    return {
        "max_tokens": max_tokens,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
        "temperature": 0.0,
        "logprobs": 1,
    }


def runtime_provenance(
    model_id: str,
    *,
    sampling: Mapping[str, object] | None = None,
) -> dict[str, object]:
    spec = model_spec(model_id)
    return {
        "model_id": spec.model_id,
        "revision": spec.revision,
        "steering": spec.steering,
        "quantization": spec.quantization or False,
        "sampling": dict(sampling or {}),
    }


def aggregate_records(
    rows: Sequence[Mapping[str, object]],
    *,
    model_id: str,
    sampling: Mapping[str, object] | None = None,
    score_rows: Sequence[Mapping[str, object]] | None = None,
    limited: bool = False,
) -> dict[str, object]:
    return {
        "provenance": runtime_provenance(model_id, sampling=sampling),
        "limited": limited,
        "ppl": ppl_summary(rows),
        "scores": score_summary(score_rows)
        if score_rows is not None
        else pending_score_summary(),
    }


def score_phase(
    *,
    model_id: str,
    response_root: str | Path,
    output_root: str | Path,
    judge_factory: Callable[[], Any] = GeminiJudge,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    limited: bool = True,
) -> dict[str, object]:
    if judge_model != DEFAULT_JUDGE_MODEL:
        raise ValueError(f"judge model must be {DEFAULT_JUDGE_MODEL}")
    response_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    judge: Any | None = None
    response_paths = {
        benchmark: no_steering.output_paths(benchmark, model_id, root=response_root)[
            "responses"
        ]
        for benchmark in DATASET_SCOPES
    }
    response_data = {
        benchmark: read_jsonl(path) for benchmark, path in response_paths.items()
    }
    _ensure_scoring_manifest(
        output_root=output_root,
        model_id=model_id,
        response_root=response_root,
        response_paths=response_paths,
        judge_model=judge_model,
    )
    for benchmark in DATASET_SCOPES:
        response_path = response_paths[benchmark]
        score_path = no_steering.output_paths(benchmark, model_id, root=output_root)[
            "scores"
        ]
        rows = response_data[benchmark]
        response_rows.extend(rows)
        if rows and benchmark in {"harmbench", "math500"} and judge is None:
            judge = (
                GeminiJudge(model=judge_model)
                if judge_factory is GeminiJudge
                else judge_factory()
            )
        if rows:
            score_responses(
                rows,
                output_path=score_path,
                judge=judge,
                judge_model=judge_model,
            )
        score_rows.extend(read_jsonl(score_path))
    validate_score_completeness(score_rows, limited=limited)
    summary = aggregate_records(
        response_rows,
        model_id=model_id,
        sampling={},
        score_rows=score_rows,
        limited=limited,
    )
    cast(dict[str, object], summary["provenance"])["judge_model"] = judge_model
    write_aggregate_artifacts(Path(output_root) / model_spec(model_id).slug, summary)
    return summary


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes() if path.exists() else b"").hexdigest()


def _scoring_manifest(
    *,
    model_id: str,
    response_root: str | Path,
    response_paths: Mapping[str, Path],
    judge_model: str,
) -> dict[str, object]:
    from prefix.judge import MATH_PROMPT, SAFETY_PROMPT

    return {
        "schema_version": SCORING_SCHEMA_VERSION,
        "model_id": model_id,
        "model_slug": model_spec(model_id).slug,
        "response_root": str(Path(response_root).resolve()),
        "response_files": {
            benchmark: {
                "path": str(path.resolve()),
                "content_sha256": _sha256_file(path),
                "ids": [str(row["id"]) for row in read_jsonl(path) if "id" in row],
            }
            for benchmark, path in response_paths.items()
        },
        "judge_model": judge_model,
        "rubric_version": SCORING_RUBRIC_VERSION,
        "prompt_sha256": {
            "safety": hashlib.sha256(SAFETY_PROMPT.encode()).hexdigest(),
            "math": hashlib.sha256(MATH_PROMPT.encode()).hexdigest(),
        },
        "response_schema": "no-steering-response-v1",
        "settings": {"temperature": 0.0, "max_output_tokens": 8192},
    }


def _ensure_scoring_manifest(
    *,
    output_root: str | Path,
    model_id: str,
    response_root: str | Path,
    response_paths: Mapping[str, Path],
    judge_model: str,
) -> None:
    path = Path(output_root) / model_spec(model_id).slug / "scoring_manifest.json"
    expected = _scoring_manifest(
        model_id=model_id,
        response_root=response_root,
        response_paths=response_paths,
        judge_model=judge_model,
    )
    if path.exists():
        actual = json.loads(path.read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError("stale or unbound scoring manifest")
    else:
        score_paths = [
            no_steering.output_paths(benchmark, model_id, root=output_root)["scores"]
            for benchmark in DATASET_SCOPES
        ]
        if any(
            score_path.exists() and score_path.stat().st_size > 0
            for score_path in score_paths
        ):
            raise RuntimeError(
                "missing scoring manifest; refusing orphan score artifact"
            )
        write_json_atomic(path, expected)


def write_aggregate_artifacts(
    output_root: str | Path, summary: Mapping[str, object]
) -> dict[str, Path]:
    root = Path(output_root)
    summary_path = root / "summary.json"
    scores_path = root / "scores.json"
    ppl_path = root / "conditional_ppl.json"
    write_json_atomic(summary_path, dict(summary))
    write_json_atomic(scores_path, cast(Mapping[str, object], summary["scores"]))
    write_json_atomic(ppl_path, cast(Mapping[str, object], summary["ppl"]))
    return {"summary": summary_path, "scores": scores_path, "ppl": ppl_path}


def evaluate_records(
    records: Sequence[dict[str, object]],
    *,
    model_id: str,
    generate: Callable[[dict[str, Any]], Mapping[str, Any]],
    output_path: str | Path,
    extract_answer: Callable[[str], str | None] | None = None,
    completed_ids: set[str] | frozenset[str] | None = None,
    retry_errors: bool = False,
) -> list[dict[str, object]]:
    """Generate one record at a time, appending durable results immediately."""
    existing = read_jsonl(output_path)
    _require_unique_ids(existing, "existing responses")
    _require_unique_ids(records, "generation input")
    retry_ids = (
        {
            str(row["id"])
            for row in existing
            if "id" in row and row.get("status") == "error"
        }
        if retry_errors
        else set()
    )
    done = {
        str(row["id"])
        for row in existing
        if "id" in row and (not retry_errors or row.get("status") == "ok")
    }
    if completed_ids is not None:
        done.update(str(identifier) for identifier in completed_ids)
    if retry_ids:
        _remove_rows_once(output_path, retry_ids)
        retry_ids = set()
    rows: list[dict[str, object]] = []
    for record in records:
        identifier = str(record["id"])
        if identifier in done:
            continue
        try:
            raw = generate(cast(dict[str, Any], record))
            native = cast(dict[str, object], _json_native(raw))
            text = str(native.get("text", ""))
            selected = _selected_logprobs(native)
            generated_count = _token_count(native, "token_ids", len(selected))
            if _has_logprob_payload(native) and len(selected) != generated_count:
                raise ValueError(
                    "selected logprob coverage does not match generated token count"
                )
            row: dict[str, object] = {
                "id": identifier,
                "benchmark": record.get("benchmark"),
                "model_id": model_id,
                "prompt": record.get("prompt", ""),
                "raw_response": native,
                "gold": record.get("gold"),
                "extracted_answer": extract_answer(text) if extract_answer else None,
                "selected_generated_token_logprobs": selected,
                "prompt_token_count": _token_count(native, "prompt_token_ids", 0),
                "generated_token_count": generated_count,
                "status": "ok",
                "error": None,
                "metadata": record.get("metadata", {}),
            }
        except Exception as error:
            row = {
                "id": identifier,
                "benchmark": record.get("benchmark"),
                "model_id": model_id,
                "prompt": record.get("prompt", ""),
                "raw_response": None,
                "gold": record.get("gold"),
                "extracted_answer": None,
                "selected_generated_token_logprobs": [],
                "prompt_token_count": 0,
                "generated_token_count": 0,
                "status": "error",
                "error": str(error),
                "metadata": record.get("metadata", {}),
            }
        _persist_generation_batch(output_path, [row], retry_ids)
        rows.append(row)
    return rows


def evaluate_records_batched(
    records: Sequence[dict[str, object]],
    *,
    model_id: str,
    generate: Callable[[list[str], object], Sequence[Mapping[str, Any]]],
    output_path: str | Path,
    batch_size: int,
    sampling_params: object,
    extract_answer: Callable[[str], str | None] | None = None,
    completed_ids: set[str] | frozenset[str] | None = None,
    retry_errors: bool = False,
) -> list[dict[str, object]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    existing = read_jsonl(output_path)
    _require_unique_ids(existing, "existing responses")
    _require_unique_ids(records, "generation input")
    retry_ids = (
        {
            str(row["id"])
            for row in existing
            if "id" in row and row.get("status") == "error"
        }
        if retry_errors
        else set()
    )
    done = {
        str(row["id"])
        for row in existing
        if "id" in row and (not retry_errors or row.get("status") == "ok")
    }
    if completed_ids is not None:
        done.update(str(identifier) for identifier in completed_ids)
    if retry_ids:
        _remove_rows_once(output_path, retry_ids)
        retry_ids = set()
    pending = [record for record in records if str(record["id"]) not in done]
    rows: list[dict[str, object]] = []
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        batch_rows: list[dict[str, object]] = []
        try:
            outputs = list(
                generate([str(record["prompt"]) for record in batch], sampling_params)
            )
            if len(outputs) != len(batch):
                raise ValueError(
                    f"output count {len(outputs)} does not match batch count {len(batch)}"
                )
        except Exception as error:
            for record in batch:
                row = _generation_error_row(record, model_id, error)
                batch_rows.append(row)
                rows.append(row)
            _persist_generation_batch(output_path, batch_rows, retry_ids)
            continue
        for record, raw in zip(batch, outputs):
            try:
                native = cast(dict[str, object], _json_native(raw))
                text = str(native.get("text", ""))
                selected = _selected_logprobs(native)
                generated = _token_count(native, "token_ids", len(selected))
                if _has_logprob_payload(native) and len(selected) != generated:
                    raise ValueError(
                        "selected logprob coverage does not match generated token count"
                    )
                row = {
                    "id": str(record["id"]),
                    "benchmark": record.get("benchmark"),
                    "model_id": model_id,
                    "prompt": record.get("prompt", ""),
                    "raw_response": native,
                    "gold": record.get("gold"),
                    "extracted_answer": extract_answer(text)
                    if extract_answer
                    else None,
                    "selected_generated_token_logprobs": selected,
                    "prompt_token_count": _token_count(native, "prompt_token_ids", 0),
                    "generated_token_count": generated,
                    "status": "ok",
                    "error": None,
                    "metadata": record.get("metadata", {}),
                }
            except Exception as error:
                row = _generation_error_row(record, model_id, error)
            batch_rows.append(row)
            rows.append(row)
        _persist_generation_batch(output_path, batch_rows, retry_ids)
    return rows


def _generation_error_row(
    record: Mapping[str, object], model_id: str, error: Exception
) -> dict[str, object]:
    return {
        "id": str(record["id"]),
        "benchmark": record.get("benchmark"),
        "model_id": model_id,
        "prompt": record.get("prompt", ""),
        "raw_response": None,
        "gold": record.get("gold"),
        "extracted_answer": None,
        "selected_generated_token_logprobs": [],
        "prompt_token_count": 0,
        "generated_token_count": 0,
        "status": "error",
        "error": str(error),
        "metadata": record.get("metadata", {}),
    }


def evaluate_benchmarks(
    benchmarks: Mapping[str, list[dict[str, object]]],
    *,
    evaluate: Callable[[str, list[dict[str, object]]], list[dict[str, object]]],
) -> dict[str, list[dict[str, object]]]:
    result: dict[str, list[dict[str, object]]] = {}
    for benchmark, records in benchmarks.items():
        try:
            result[benchmark] = evaluate(benchmark, records)
        except Exception as error:
            result[benchmark] = [
                {"id": benchmark, "status": "error", "error": str(error)}
            ]
    return result


def _records_for(
    benchmark: str, records: list[dict[str, object]], tokenizer: Any
) -> list[dict[str, object]]:
    prepared: list[dict[str, object]] = []
    for record in records:
        if benchmark == "mmlu_pro":
            text = mmlu_prompt(
                str(record["question"]),
                list(cast(Sequence[str], record["options"])),
            )
            gold = record["answer_letter"]
        elif benchmark == "math500":
            text = f"{record['problem']} Reason step by step."
            gold = record["answer"]
        else:
            text = str(record["behavior"])
            gold = None
        prepared.append(
            {
                **record,
                "benchmark": benchmark,
                "prompt": chat_prompt(tokenizer, text, True),
                "gold": gold,
                "metadata": {
                    k: v for k, v in record.items() if k not in {"id", "prompt", "gold"}
                },
            }
        )
    return prepared


def _run(args: argparse.Namespace) -> None:
    if args.phase == "preflight":
        check_model_snapshots = _check_model_snapshots()

        check_model_snapshots((args.model_id,), cache_root=args.cache_root, load=False)
        return
    response_root = args.checkpoint_root or args.output_root
    formal_root = args.output_root
    if args.phase == "score":
        score_phase(
            model_id=args.model_id,
            response_root=response_root,
            output_root=formal_root,
            judge_model=args.judge_model,
            limited=args.limit is not None,
        )
        return
    benchmarks = load_benchmarks()
    settings = generation_settings(
        args.max_tokens, args.max_model_len, args.gpu_memory_utilization
    )
    check_model_snapshots = _check_model_snapshots()

    snapshot = check_model_snapshots(
        (args.model_id,), cache_root=args.cache_root, load=False
    )[args.model_id]
    engine_kwargs = {
        "revision": model_spec(args.model_id).revision,
        "quantization": None,
        "max_model_len": settings["max_model_len"],
        "gpu_memory_utilization": settings["gpu_memory_utilization"],
    }
    limited = args.limit is not None
    manifest = checkpoint_manifest(
        {
            benchmark: records[: args.limit] if limited else records
            for benchmark, records in benchmarks.items()
        },
        model_id=args.model_id,
        sampling={
            "max_tokens": settings["max_tokens"],
            "temperature": settings["temperature"],
            "logprobs": settings["logprobs"],
            **engine_kwargs,
        },
        batch_prompts=args.batch_prompts,
        max_model_len=args.max_model_len,
        output_roots={"checkpoint": str(response_root), "output": str(formal_root)},
        limited=limited,
    )
    manifest_path = response_root / model_spec(args.model_id).slug / "manifest.json"
    prepare_checkpoint_manifest(
        manifest_path,
        manifest,
        checkpoint_paths=[
            no_steering.output_paths(benchmark, args.model_id, root=response_root)[
                "responses"
            ]
            for benchmark in DATASET_SCOPES
        ],
    )
    llm: Any = get_engine(str(snapshot), **engine_kwargs)
    tokenizer: Any = llm.get_tokenizer()
    sampling = {
        "max_tokens": settings["max_tokens"],
        "temperature": settings["temperature"],
        "logprobs": settings["logprobs"],
    }
    prepared_by_benchmark: dict[str, list[dict[str, object]]] = {}
    for benchmark, records in benchmarks.items():
        if args.limit is not None:
            records = records[: args.limit]
        prepared = _records_for(benchmark, records, tokenizer)
        prepared_by_benchmark[benchmark] = prepared
        provenance = runtime_provenance(args.model_id, sampling=sampling)
        for record in prepared:
            record["metadata"] = {
                **cast(Mapping[str, object], record["metadata"]),
                "provenance": provenance,
            }
    sampling_params = getattr(importlib.import_module("vllm"), "SamplingParams")(
        **sampling
    )
    all_rows: list[dict[str, object]] = []
    for benchmark, prepared in prepared_by_benchmark.items():
        output = no_steering.output_paths(benchmark, args.model_id, root=response_root)[
            "responses"
        ]

        def generate(prompts: list[str], params: object) -> list[Mapping[str, Any]]:
            output_objects = llm.generate(prompts, params)
            result: list[Mapping[str, Any]] = []
            for output_obj in output_objects:
                first = output_obj.outputs[0]
                result.append(
                    {
                        "text": first.text,
                        "prompt_token_ids": output_obj.prompt_token_ids,
                        "outputs": [
                            {"token_ids": first.token_ids, "logprobs": first.logprobs}
                        ],
                    }
                )
            return result

        evaluate_records_batched(
            prepared,
            model_id=args.model_id,
            generate=generate,
            output_path=output,
            batch_size=args.batch_prompts,
            sampling_params=sampling_params,
            extract_answer=parse_answer_letter if benchmark == "mmlu_pro" else None,
            retry_errors=True,
        )
        all_rows.extend(read_jsonl(output))
    if not limited:
        validate_score_completeness(all_rows, limited=False)
    write_aggregate_artifacts(
        formal_root / model_spec(args.model_id).slug,
        aggregate_records(
            all_rows,
            model_id=args.model_id,
            sampling={**sampling, **engine_kwargs},
            limited=limited,
        ),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run pinned no-steering baselines")
    parser.add_argument("--model-id", choices=MODEL_MATRIX, required=True)
    parser.add_argument(
        "--phase", choices=("preflight", "generate", "score"), default="preflight"
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results")
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--cache-root", type=Path, default=ROOT / "models")
    parser.add_argument("--batch-prompts", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.batch_prompts < 1:
        raise ValueError("--batch-prompts must be positive")
    generation_settings(
        args.max_tokens, args.max_model_len, args.gpu_memory_utilization
    )
    log_path = ROOT / "logs" / "no_steering.log"
    with notify_on_exit(
        f"no-steering-{args.model_id}", log_file=log_path, enabled=False
    ):
        with tee_stdout(log_path):
            _run(args)


if __name__ == "__main__":
    main()
