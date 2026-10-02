from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import tempfile
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import TextIO, cast

from prefix.math_diagnostics import compare_math_answers, extract_final_answer
from prefix.no_steering import DATASET_SCOPES, MODEL_MATRIX, model_spec
from prefix.notify import notify_on_exit
from prefix.runner import parse_answer_letter


HASH_CHUNK_SIZE = 1024 * 1024
METRICS_SCHEMA = 1


def _reject_symlink_components(path: Path) -> None:
    current = Path(path.anchor) if path.is_absolute() else Path.cwd()
    components = path.parts[1:] if path.is_absolute() else path.parts
    for component in components:
        current /= component
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            break
        if stat.S_ISLNK(mode):
            raise ValueError(f"refusing symlinked audit path component: {current}")


class _Tee:
    def __init__(self, streams: tuple[TextIO, TextIO]) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _iter_jsonl(path: Path) -> Iterator[dict[str, object]]:
    _reject_symlink_components(path)
    try:
        handle = path.open(encoding="utf-8")
    except FileNotFoundError:
        raise ValueError(f"missing JSONL output: {path}") from None
    with handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
            yield cast(dict[str, object], value)


def _sha256_file(path: Path) -> str:
    _reject_symlink_components(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _metrics_sha256(metrics: Mapping[str, object]) -> str:
    canonical = json.dumps(
        metrics, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _atomic_write(path: Path, payload: Mapping[str, object]) -> None:
    _reject_symlink_components(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
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


def _text(row: Mapping[str, object]) -> str:
    raw = row.get("raw_response")
    if isinstance(raw, Mapping):
        value = raw.get("text")
        if isinstance(value, str):
            return value
    return ""


def _current(text: str) -> str | None:
    return parse_answer_letter(text)


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _saturation(rows: list[dict[str, object]]) -> dict[str, object]:
    saturated = sum(row.get("generated_token_count") == 1024 for row in rows)
    denominator = len(rows)
    return {
        "saturated": saturated,
        "unsaturated": denominator - saturated,
        "denominator": denominator,
        "rate": _rate(saturated, denominator),
    }


def _finish_reason(rows: list[dict[str, object]]) -> dict[str, int]:
    available = sum(row.get("finish_reason") is not None for row in rows)
    return {
        "available": available,
        "unavailable": len(rows) - available,
        "denominator": len(rows),
    }


def _score_path(root: Path, slug: str, benchmark: str) -> Path:
    results_path = root / "results" / slug / benchmark / "scores.jsonl"
    _reject_symlink_components(results_path)
    if results_path.is_file():
        score_path = results_path
    else:
        score_path = root / "scoring" / slug / benchmark / "scores.jsonl"
    _reject_symlink_components(score_path)
    return score_path


def _ordered_unique_ids(
    rows: list[dict[str, object]], *, label: str, path: Path
) -> list[str]:
    identifiers: list[str] = []
    seen: set[str] = set()
    for line_number, row in enumerate(rows, 1):
        if "id" not in row:
            raise ValueError(f"{label} row is missing an id in {path}:{line_number}")
        identifier = str(row["id"])
        if identifier in seen:
            raise ValueError(f"duplicate {label} id in {path}: {identifier}")
        seen.add(identifier)
        identifiers.append(identifier)
    return identifiers


def _audit_benchmark(
    *, formal_run_root: Path, model_id: str, benchmark: str
) -> dict[str, object]:
    _reject_symlink_components(formal_run_root)
    spec = model_spec(model_id)
    response_path = (
        formal_run_root / "checkpoints" / spec.slug / benchmark / "responses.jsonl"
    )
    score_path = _score_path(formal_run_root, spec.slug, benchmark)
    responses = list(_iter_jsonl(response_path))
    scores = list(_iter_jsonl(score_path))
    response_ids = _ordered_unique_ids(responses, label="response", path=response_path)
    score_ids = _ordered_unique_ids(scores, label="score", path=score_path)
    score_by_id = dict(zip(score_ids, scores, strict=True))
    if any(row.get("benchmark") != benchmark for row in responses + scores):
        raise ValueError(f"benchmark mismatch for {model_id}/{benchmark}")
    if response_ids != score_ids:
        raise ValueError(f"response/score ids do not match for {model_id}/{benchmark}")

    metrics: dict[str, object] = {
        "count": len(responses),
        "generated_token_count_saturation": _saturation(responses),
        "finish_reason": _finish_reason(responses),
    }
    if benchmark == "mmlu_pro":
        legacy_correct = fixed_correct = 0
        for row in responses:
            gold = str(row.get("gold"))
            text = _text(row)
            legacy = row.get("extracted_answer")
            current = _current(text)
            legacy_correct += int(legacy is not None and str(legacy) == gold)
            fixed_correct += int(current is not None and current == gold)
        denominator = len(responses)
        metrics.update(
            {
                "legacy_extracted_accuracy": _rate(legacy_correct, denominator),
                "fixed_extracted_accuracy": _rate(fixed_correct, denominator),
                "parser_transition_delta": _rate(
                    fixed_correct - legacy_correct, denominator
                ),
                "parser_correctness_delta": _rate(
                    fixed_correct - legacy_correct, denominator
                ),
            }
        )
    elif benchmark == "math500":
        extraction = {
            key: 0 for key in ("answer", "null", "ambiguous", "indeterminate")
        }
        equivalence = {
            key: 0
            for key in ("equivalent", "different", "ambiguous", "null", "indeterminate")
        }
        gemini_status: dict[str, int] = {}
        gemini_correct = 0
        agreement = {
            "both_correct": 0,
            "both_incorrect": 0,
            "deterministic_only": 0,
            "gemini_only": 0,
            "both_indeterminate": 0,
        }
        deterministic_accuracy_denominator = gemini_accuracy_denominator = (
            agreement_denominator
        ) = 0
        for response in responses:
            score = score_by_id[str(response["id"])]
            extracted = extract_final_answer(_text(response))
            extraction[extracted.status] += 1
            comparison = compare_math_answers(extracted, str(response.get("gold", "")))
            verdict = (
                "indeterminate" if comparison.verdict == "null" else comparison.verdict
            )
            equivalence_verdict = (
                "different"
                if verdict == "indeterminate"
                and extracted.status == "null"
                and score.get("status") == "ok"
                else verdict
            )
            equivalence[equivalence_verdict] += 1
            deterministic_correct = verdict == "equivalent"
            deterministic_indeterminate = verdict not in {"equivalent", "different"}
            status = str(score.get("status"))
            gemini_status[status] = gemini_status.get(status, 0) + 1
            gemini_correct += int(score.get("answer_correct") is True)
            gemini_available = status == "ok"
            gemini_accuracy_denominator += int(gemini_available)
            gemini_correct_for_agreement = (
                gemini_available and score.get("answer_correct") is True
            )
            if not gemini_available:
                category = "both_indeterminate"
                agreement["both_indeterminate"] += 1
            elif deterministic_indeterminate and gemini_correct_for_agreement:
                category = "gemini_only"
                agreement["gemini_only"] += 1
            elif deterministic_indeterminate:
                category = "both_indeterminate"
                agreement["both_indeterminate"] += 1
            elif deterministic_correct and gemini_correct_for_agreement:
                category = "both_correct"
                agreement["both_correct"] += 1
            elif deterministic_correct:
                category = "deterministic_only"
                agreement["deterministic_only"] += 1
            elif gemini_correct_for_agreement:
                category = "gemini_only"
                agreement["gemini_only"] += 1
            else:
                category = "both_incorrect"
                agreement["both_incorrect"] += 1
            if category != "both_indeterminate":
                deterministic_accuracy_denominator += 1
            if category != "both_indeterminate" and gemini_available:
                agreement_denominator += 1
        metrics.update(
            {
                "deterministic_extraction_status": {
                    **extraction,
                    "denominator": len(responses),
                },
                "deterministic_equivalence_status": equivalence,
                "gemini": {"correct": gemini_correct, "status": gemini_status},
                "agreement_categories": {**agreement, "denominator": len(responses)},
                "deterministic_accuracy_denominator": deterministic_accuracy_denominator,
                "gemini_accuracy_denominator": gemini_accuracy_denominator,
                "agreement_denominator": agreement_denominator,
            }
        )
    elif benchmark == "harmbench":
        categories: dict[str, int] = {}
        for score in scores:
            label = str(score.get("gemini_label"))
            categories[label] = categories.get(label, 0) + 1
        metrics["gemini"] = {
            "status": {
                "ok": sum(score.get("status") == "ok" for score in scores),
                "unparseable": sum(
                    score.get("status") == "unparseable" for score in scores
                ),
            },
            "categories": categories,
        }
    return metrics


def _retained_manifest_binding(
    *,
    root: Path,
    model_id: str,
    benchmark: str,
    response: Path,
    response_digest: str,
    score: Path,
    score_digest: str,
) -> dict[str, object]:
    spec = model_spec(model_id)
    checkpoint_manifest = root / "checkpoints" / spec.slug / "manifest.json"
    scoring_manifest = root / "scoring" / spec.slug / "scoring_manifest.json"
    _reject_symlink_components(checkpoint_manifest)
    _reject_symlink_components(scoring_manifest)
    try:
        checkpoint = json.loads(checkpoint_manifest.read_text(encoding="utf-8"))
        scoring = json.loads(scoring_manifest.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"missing retained manifest: {error.filename}") from None
    if not isinstance(checkpoint, Mapping) or not isinstance(scoring, Mapping):
        raise ValueError(f"invalid retained manifest for {model_id}")
    config = checkpoint.get("config")
    if (
        not isinstance(config, Mapping)
        or config.get("model_id") != model_id
        or config.get("model_revision") != spec.revision
    ):
        raise ValueError(f"stale retained identity for {model_id}/{benchmark}")
    if scoring.get("model_id") != model_id or scoring.get("model_slug") != spec.slug:
        raise ValueError(f"stale scoring identity for {model_id}/{benchmark}")
    response_files = scoring.get("response_files")
    entry = (
        response_files.get(benchmark) if isinstance(response_files, Mapping) else None
    )
    if not isinstance(entry, Mapping):
        raise ValueError(
            f"missing retained response manifest for {model_id}/{benchmark}"
        )
    manifest_response = entry.get("path")
    if (
        not isinstance(manifest_response, str)
        or Path(manifest_response).resolve() != response.resolve()
    ):
        raise ValueError(f"stale retained response path for {model_id}/{benchmark}")
    if entry.get("content_sha256") != response_digest:
        raise ValueError(f"stale retained response digest for {model_id}/{benchmark}")
    score_files = scoring.get("score_files")
    score_entry = (
        score_files.get(benchmark) if isinstance(score_files, Mapping) else None
    )
    if not isinstance(score_entry, Mapping):
        raise ValueError(f"missing retained score manifest for {model_id}/{benchmark}")
    manifest_score = score_entry.get("path")
    if (
        not isinstance(manifest_score, str)
        or Path(manifest_score).resolve() != score.resolve()
    ):
        raise ValueError(f"stale retained score path for {model_id}/{benchmark}")
    if score_entry.get("content_sha256") != score_digest:
        raise ValueError(f"stale retained score digest for {model_id}/{benchmark}")
    return {"binding_sha256": _sha256_file(scoring_manifest)}


def audit(
    *, formal_run_root: str | Path, output_json: str | Path, log_file: str | Path
) -> dict[str, object]:
    root, output, log_path = Path(formal_run_root), Path(output_json), Path(log_file)
    _reject_symlink_components(root)
    _reject_symlink_components(output)
    _reject_symlink_components(log_path)
    units: list[dict[str, object]] = []
    for model_id in MODEL_MATRIX:
        spec = model_spec(model_id)
        for benchmark in DATASET_SCOPES:
            response = root / "checkpoints" / spec.slug / benchmark / "responses.jsonl"
            score = _score_path(root, spec.slug, benchmark)
            binding = score.parent / "bindings.json"
            _reject_symlink_components(response)
            _reject_symlink_components(score)
            _reject_symlink_components(binding)
            for path in (response, score):
                if not path.is_file():
                    raise ValueError(f"missing audit input: {path}")
            response_digest, score_digest = _sha256_file(response), _sha256_file(score)
            if binding.is_file():
                binding_payload = json.loads(binding.read_text(encoding="utf-8"))
                if (
                    binding_payload.get("model_id") != model_id
                    or binding_payload.get("revision") != spec.revision
                ):
                    raise ValueError(f"stale binding for {model_id}/{benchmark}")
                if (
                    binding_payload.get("response_sha256") != response_digest
                    or binding_payload.get("score_sha256") != score_digest
                ):
                    raise ValueError(f"stale digest binding for {model_id}/{benchmark}")
                binding_digest = _sha256_file(binding)
            else:
                fallback = _retained_manifest_binding(
                    root=root,
                    model_id=model_id,
                    benchmark=benchmark,
                    response=response,
                    response_digest=response_digest,
                    score=score,
                    score_digest=score_digest,
                )
                binding_digest = fallback["binding_sha256"]
            units.append(
                {
                    "model_id": model_id,
                    "benchmark": benchmark,
                    "response_sha256": response_digest,
                    "score_sha256": score_digest,
                    "binding_sha256": binding_digest,
                }
            )
    source_sha = hashlib.sha256(json.dumps(units, sort_keys=True).encode()).hexdigest()
    checkpoint_path = output.parent / "audit.checkpoint.json"
    _reject_symlink_components(checkpoint_path)
    try:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        checkpoint = {}
    existing = checkpoint if checkpoint.get("source_sha256") == source_sha else {}
    saved = {
        str(item["model_id"]) + "\0" + str(item["benchmark"]): item
        for item in existing.get("units", [])
        if isinstance(item, dict)
    }
    per_model: dict[str, dict[str, object]] = {
        model_id: {} for model_id in MODEL_MATRIX
    }
    completed: list[dict[str, str]] = []
    for unit in units:
        key = str(unit["model_id"]) + "\0" + str(unit["benchmark"])
        previous = saved.get(key)
        if (
            previous
            and all(
                previous.get(field) == unit[field]
                for field in ("response_sha256", "score_sha256", "binding_sha256")
            )
            and previous.get("metrics_schema") == METRICS_SCHEMA
            and isinstance(previous.get("metrics"), Mapping)
            and previous.get("metrics_sha256")
            == _metrics_sha256(cast(Mapping[str, object], previous["metrics"]))
        ):
            metrics = cast(dict[str, object], previous["metrics"])
        else:
            metrics = _audit_benchmark(
                formal_run_root=root,
                model_id=str(unit["model_id"]),
                benchmark=str(unit["benchmark"]),
            )
        per_model[str(unit["model_id"])][str(unit["benchmark"])] = metrics
        completed.append(
            {"model_id": str(unit["model_id"]), "benchmark": str(unit["benchmark"])}
        )
        current_units = [
            {
                **item,
                "metrics": per_model[str(item["model_id"])][str(item["benchmark"])],
                "metrics_schema": METRICS_SCHEMA,
                "metrics_sha256": _metrics_sha256(
                    cast(
                        Mapping[str, object],
                        per_model[str(item["model_id"])][str(item["benchmark"])],
                    )
                ),
            }
            for item in units
            if str(item["model_id"]) + "\0" + str(item["benchmark"])
            in {str(x["model_id"]) + "\0" + str(x["benchmark"]) for x in completed}
        ]
        _atomic_write(
            checkpoint_path,
            {
                "source_sha256": source_sha,
                "completed_units": completed,
                "units": current_units,
            },
        )
    payload = {
        "models": [
            {
                "model_id": model_id,
                "revision": model_spec(model_id).revision,
                "slug": model_spec(model_id).slug,
            }
            for model_id in MODEL_MATRIX
        ],
        "per_model": per_model,
        "source_sha256": source_sha,
    }
    _atomic_write(output, payload)
    with log_path.open("a", encoding="utf-8") as log:
        print(
            f"audit: models={len(MODEL_MATRIX)} units={len(units)} source_sha256={source_sha}",
            file=_Tee((sys.stdout, log)),
        )
    return cast(dict[str, object], payload)


def main(argv: list[str] | None = None) -> dict[str, object]:
    parser = argparse.ArgumentParser(description="Audit retained benchmark outputs")
    parser.add_argument("--formal-run-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--log-file", type=Path, required=True)
    args = parser.parse_args(argv)
    _reject_symlink_components(args.formal_run_root)
    _reject_symlink_components(args.output_json)
    _reject_symlink_components(args.log_file)
    with notify_on_exit("audit-benchmark-results", log_file=args.log_file):
        return audit(
            formal_run_root=args.formal_run_root,
            output_json=args.output_json,
            log_file=args.log_file,
        )


if __name__ == "__main__":
    main()
