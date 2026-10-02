from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shlex
import stat
import sys
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from prefix.preflight_marker import (
    marker_sha256,
    parse_preflight_marker,
)
from prefix.runner import exclusive_mutation_lock

CONDITIONS = ("current_zero_shot_1024", "current_zero_shot_2048")
MODEL_ID = "allenai/Olmo-3-7B-Think"
MODEL_REVISION = "d97e442d7cc678210054dbcc9b440894d62c89a4"
DATASET = "TIGER-Lab/MMLU-Pro"
DATASET_REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
SOURCE_COUNT = 12032
MAX_MODEL_LEN = 8192
TEMPERATURE = 0.0
EXPECTED_CATEGORIES = (
    "biology",
    "business",
    "chemistry",
    "computer science",
    "economics",
    "engineering",
    "health",
    "history",
    "law",
    "math",
    "other",
    "philosophy",
    "physics",
    "psychology",
)
DEFAULT_QUOTAS = {
    "saturated_null": 3,
    "saturated_extracted": 2,
    "unsaturated_extracted": 2,
    "unsaturated_null": 1,
}
LEGACY_PARSER_VERSION = "legacy-v1"
_LEGACY_ANSWER_RE = re.compile(r"answer\s+is\s*\(?([A-J])\)?\b", re.IGNORECASE)
FORMAL_SOURCE_RELATIVE_PATH = Path(
    "checkpoints/allenai--Olmo-3-7B-Think/mmlu_pro/responses.jsonl"
)


def legacy_v1_parse_answer_letter(text: str) -> str | None:
    matches = list(_LEGACY_ANSWER_RE.finditer(text))
    if matches:
        return matches[-1].group(1).upper()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        match = re.fullmatch(r"\(?([A-J])\)?", lines[-1], re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def _sha256_file(path: Path) -> str:
    _reject_symlink_components(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )


def _sha256_value(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _reject_symlink_components(path: Path) -> None:
    candidate = path if path.is_absolute() else Path.cwd() / path
    current = Path(candidate.anchor)
    for component in candidate.parts[1:]:
        current /= component
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError("path is symlinked")


def _canonical_no_symlink(path: str | Path) -> Path:
    candidate = Path(path)
    _reject_symlink_components(candidate)
    return candidate.resolve(strict=False)


def validate_formal_source_path(
    source_path: str | Path, formal_run_root: str | Path | None
) -> Path:
    if formal_run_root is None:
        raise ValueError("formal run root is required for OLMo provenance")
    root = _canonical_no_symlink(formal_run_root)
    expected = _canonical_no_symlink(root / FORMAL_SOURCE_RELATIVE_PATH)
    actual = _canonical_no_symlink(source_path)
    if actual != expected:
        raise ValueError("source responses path is not bound to formal run root")
    return actual


def validate_preflight_marker(
    path: str | Path, *, run_root: str | Path | None = None
) -> str:
    _ = parse_preflight_marker(path, model_id=MODEL_ID, run_root=run_root)
    return marker_sha256(path)


def _current_runtime_fingerprint() -> dict[str, object]:
    lock_path = Path(__file__).resolve().parents[1] / "uv.lock"
    packages: dict[str, str] = {}
    for name in ("transformers", "vllm"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "missing"
    runtime = {
        "python": platform.python_version(),
        "packages": packages,
        "harness_sha256": _sha256_file(Path(__file__).resolve()),
        "uv_lock_sha256": _sha256_file(lock_path) if lock_path.exists() else "missing",
    }
    runtime["fingerprint_sha256"] = _sha256_value(runtime)
    return cast(dict[str, object], runtime)


def _source_evidence(row: Mapping[str, object]) -> dict[str, object]:
    metadata = _metadata_row(row)
    token_ids = metadata.get("source_generated_token_ids", [])
    if (
        not isinstance(token_ids, list)
        or not token_ids
        or any(
            isinstance(value, bool) or not isinstance(value, int) for value in token_ids
        )
    ):
        raise ValueError("source evidence token ids are empty or invalid")
    return {
        "source_extracted_answer": metadata["source_extracted_answer"],
        "source_generated_token_ids": list(cast(Sequence[int], token_ids)),
    }


def _as_float(value: object, default: float = 1.0) -> float:
    if value is None:
        return default
    return float(cast(float | int | str, value))


def _jsonl(path: Path) -> Iterable[dict[str, object]]:
    _reject_symlink_components(path)
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                if not line.endswith("\n"):
                    break
                raise ValueError(f"malformed JSONL row {line_number}") from None
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row {line_number} is not an object")
            yield cast(dict[str, object], value)


def _metadata_row(row: Mapping[str, object]) -> dict[str, object]:
    metadata = row.get("metadata")
    merged = {
        **(cast(dict[str, object], metadata) if isinstance(metadata, dict) else {}),
        **row,
    }
    required = (
        "id",
        "category",
        "question",
        "options",
        "gold",
        "generated_token_count",
        "extracted_answer",
        "status",
    )
    missing = [key for key in required if key not in merged]
    if missing:
        raise ValueError(f"source row lacks metadata fields: {', '.join(missing)}")
    result = {
        "id": str(merged["id"]),
        "category": str(merged["category"]),
        "question": str(merged["question"]),
        "options": list(cast(Sequence[object], merged["options"])),
        "gold": str(merged.get("gold", merged.get("answer_letter"))),
        "generated_token_count": int(cast(int, merged["generated_token_count"])),
        "extracted_answer": merged["extracted_answer"],
        "status": str(merged["status"]),
        "prompt": merged.get("prompt"),
    }
    raw_response = row.get("raw_response")
    if isinstance(raw_response, Mapping):
        outputs = raw_response.get("outputs")
        if (
            isinstance(outputs, Sequence)
            and outputs
            and isinstance(outputs[0], Mapping)
        ):
            token_ids = outputs[0].get("token_ids")
            if isinstance(token_ids, Sequence) and not isinstance(
                token_ids, (str, bytes)
            ):
                result["source_generated_token_ids"] = [
                    int(cast(int, value)) for value in token_ids
                ]
    result["source_extracted_answer"] = row.get("extracted_answer")
    return result


def _validate_formal_provenance(row: Mapping[str, object]) -> None:
    metadata = row.get("metadata")
    provenance = metadata.get("provenance") if isinstance(metadata, Mapping) else None
    if (
        isinstance(metadata, Mapping)
        and any(
            key in metadata
            for key in (
                "model_id",
                "model_revision",
                "benchmark",
                "budget",
                "temperature",
            )
        )
        and (
            metadata.get("model_id") != MODEL_ID
            or metadata.get("model_revision") != MODEL_REVISION
            or metadata.get("benchmark") != "mmlu_pro"
            or metadata.get("budget") != 1024
            or metadata.get("temperature") != 0.0
        )
    ):
        raise ValueError("source formal provenance mismatch")
    if row.get("model_id") != MODEL_ID or row.get("benchmark") != "mmlu_pro":
        raise ValueError("source model or benchmark provenance mismatch")
    required_provenance = {
        "model_id",
        "revision",
        "steering",
        "quantization",
        "sampling",
    }
    if not isinstance(provenance, Mapping) or not required_provenance <= set(
        provenance
    ):
        if isinstance(provenance, Mapping) and (
            provenance.get("revision") not in (None, MODEL_REVISION)
            or not isinstance(provenance.get("sampling"), Mapping)
            or cast(Mapping[str, object], provenance["sampling"]).get("max_tokens")
            != 1024
            or cast(Mapping[str, object], provenance["sampling"]).get("temperature")
            != 0.0
        ):
            raise ValueError("source revision provenance mismatch")
        raw_response = row.get("raw_response")
        outputs = (
            raw_response.get("outputs") if isinstance(raw_response, Mapping) else None
        )
        token_ids = (
            outputs[0].get("token_ids")
            if isinstance(outputs, Sequence)
            and outputs
            and isinstance(outputs[0], Mapping)
            else None
        )
        if token_ids is None or (
            isinstance(token_ids, Sequence)
            and len(token_ids) != row.get("generated_token_count")
        ):
            raise ValueError("source token evidence coverage is incomplete")
        raise ValueError("source revision provenance mismatch")
    if (
        provenance.get("model_id") != MODEL_ID
        or provenance.get("revision") != MODEL_REVISION
        or provenance.get("steering") is not False
        or provenance.get("quantization") is not False
    ):
        raise ValueError("source revision provenance mismatch")
    sampling = provenance.get("sampling")
    if (
        not isinstance(sampling, Mapping)
        or sampling.get("max_tokens") != 1024
        or sampling.get("temperature") != 0.0
    ):
        raise ValueError("source sampling provenance mismatch")
    raw_response = row.get("raw_response")
    outputs = raw_response.get("outputs") if isinstance(raw_response, Mapping) else None
    token_ids = (
        outputs[0].get("token_ids")
        if isinstance(outputs, Sequence) and outputs and isinstance(outputs[0], Mapping)
        else None
    )
    expected_count = row.get("generated_token_count")
    if (
        not isinstance(token_ids, Sequence)
        or isinstance(token_ids, (str, bytes))
        or len(token_ids) < 2
        or not isinstance(expected_count, int)
        or len(token_ids) != expected_count
    ):
        raise ValueError("source token evidence coverage is incomplete")


def _stratum(row: Mapping[str, object]) -> str:
    count = int(cast(int, row["generated_token_count"]))
    extracted = bool(row.get("extracted_answer"))
    return f"{'saturated' if count == 1024 else 'unsaturated'}_{'extracted' if extracted else 'null'}"


def _candidate_key(seed: int, identifier: str) -> str:
    return hashlib.sha256(f"{seed}:{identifier}".encode("utf-8")).hexdigest()


def _quota_for(
    quotas: Mapping[str, Mapping[str, int]] | None, category: str
) -> dict[str, int]:
    if quotas is None:
        return dict(DEFAULT_QUOTAS)
    if category not in quotas or "saturated_null" not in quotas[category]:
        raise ValueError(f"saturated_null quota required for category {category}")
    return {str(key): int(value) for key, value in quotas[category].items()}


def select_source_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    quotas: Mapping[str, Mapping[str, int]] | None = None,
    expected_categories: Sequence[str] = EXPECTED_CATEGORIES,
    seed: int = 17,
    expected_count: int | None = None,
    checkpoint_path: str | Path | None = None,
    checkpoint_config: Mapping[str, object] | None = None,
    source_path: str | Path | None = None,
    require_formal_provenance: bool = False,
) -> dict[str, object]:
    expected = set(expected_categories)
    seen: set[str] = set()
    population: dict[str, int] = defaultdict(int)
    reservoirs: dict[tuple[str, str], list[tuple[str, dict[str, object]]]] = (
        defaultdict(list)
    )
    legacy_unverified = False
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    source_sha256 = _sha256_file(Path(source_path)) if source_path is not None else None
    checkpoint_digest = _sha256_value(
        {
            "source_path": str(source_path) if source_path is not None else None,
            "source_sha256": source_sha256,
            "config": dict(checkpoint_config or {}),
        }
    )
    processed_rows = 0
    journal_complete = False
    replayed_rows: dict[int, dict[str, object]] = {}

    def append_event(event: Mapping[str, object]) -> None:
        if checkpoint is not None:
            _append_jsonl(checkpoint, [event])

    def replay_row(event: Mapping[str, object]) -> None:
        nonlocal legacy_unverified, processed_rows
        if event.get("event") != "row":
            raise ValueError("selection checkpoint row event is invalid")
        index = int(cast(int | str, event.get("row_index", -1)))
        if index != processed_rows:
            raise ValueError("selection checkpoint row sequence is invalid")
        row = dict(cast(Mapping[str, object], event.get("row", {})))
        row_digest = event.get("row_digest")
        if not isinstance(row_digest, str) or row_digest != _sha256_value(row):
            raise ValueError("selection checkpoint row digest is invalid")
        replayed_rows[index] = row
        identifier = str(row.get("id"))
        if identifier in seen:
            raise ValueError(f"selection checkpoint duplicate id: {identifier}")
        category = str(event.get("category"))
        stratum = str(event.get("stratum"))
        if category != str(row.get("category")) or stratum != _stratum(row):
            raise ValueError("selection checkpoint row provenance is invalid")
        seen.add(identifier)
        population[f"{category}/{stratum}"] += 1
        candidate_key = event.get("candidate_key")
        quota = _quota_for(quotas, category).get(stratum, 0)
        if quota > 0:
            expected_key = _candidate_key(seed, identifier)
            if candidate_key != expected_key:
                raise ValueError("selection checkpoint candidate is invalid")
            reservoir = reservoirs[(category, stratum)]
            reservoir.append((expected_key, row))
            reservoir.sort(key=lambda item: item[0])
            del reservoir[quota:]
        elif candidate_key is not None:
            raise ValueError("selection checkpoint candidate is invalid")
        legacy_unverified = legacy_unverified or bool(
            event.get("legacy_unverified", False)
        )
        processed_rows += 1

    if checkpoint is not None:
        if source_path is None:
            raise ValueError("checkpoint source_path is required")
        if checkpoint.exists():
            events = list(_jsonl(checkpoint))
            if not events or events[0].get("event") != "header":
                raise ValueError("selection checkpoint header is missing")
            header = events[0]
            if (
                header.get("schema_version") != 2
                or header.get("source_path") != str(source_path)
                or header.get("source_sha256") != source_sha256
                or header.get("config_digest") != checkpoint_digest
                or header.get("model_id")
                != (checkpoint_config or {}).get("model_id", MODEL_ID)
                or header.get("model_revision")
                != (checkpoint_config or {}).get("model_revision", MODEL_REVISION)
                or header.get("marker_path")
                != (checkpoint_config or {}).get("marker_path")
                or header.get("marker_sha256")
                != (checkpoint_config or {}).get("marker_sha256")
                or header.get("formal_run_root")
                != (checkpoint_config or {}).get("formal_run_root")
            ):
                raise ValueError("selection checkpoint digest mismatch")
            for event in events[1:]:
                if journal_complete:
                    raise ValueError(
                        "selection checkpoint event appears after terminal complete"
                    )
                if event.get("event") == "row":
                    replay_row(event)
                elif event.get("event") == "complete":
                    if (
                        journal_complete
                        or int(cast(int | str, event.get("processed_rows", -1)))
                        != processed_rows
                    ):
                        raise ValueError("selection checkpoint completion is invalid")
                    journal_complete = True
                else:
                    raise ValueError("selection checkpoint event is invalid")
        else:
            append_event(
                {
                    "event": "header",
                    "schema_version": 2,
                    "source_path": str(source_path),
                    "source_sha256": source_sha256,
                    "config_digest": checkpoint_digest,
                    "model_id": (checkpoint_config or {}).get("model_id", MODEL_ID),
                    "model_revision": (checkpoint_config or {}).get(
                        "model_revision", MODEL_REVISION
                    ),
                    "marker_path": (checkpoint_config or {}).get("marker_path"),
                    "marker_sha256": (checkpoint_config or {}).get("marker_sha256"),
                    "formal_run_root": (checkpoint_config or {}).get("formal_run_root"),
                }
            )

    source_row_count = 0
    for row_index, raw in enumerate(rows):
        source_row_count = row_index + 1
        if row_index < processed_rows:
            replayed = replayed_rows.get(row_index)
            if replayed is None:
                raise ValueError("selection checkpoint row replay is incomplete")
            current = _metadata_row(raw)
            if current != replayed or _sha256_value(current) != _sha256_value(replayed):
                raise ValueError(
                    "selection checkpoint row does not match source prefix"
                )
            continue
        if journal_complete:
            raise ValueError("selection checkpoint completed before source end")
        try:
            _validate_formal_provenance(raw)
        except ValueError:
            has_formal_marker = (
                "model_id" in raw
                or "benchmark" in raw
                or isinstance(raw.get("metadata"), Mapping)
                and isinstance(
                    cast(Mapping[str, object], raw["metadata"]).get("provenance"),
                    Mapping,
                )
                or isinstance(raw.get("metadata"), Mapping)
                and any(
                    key in cast(Mapping[str, object], raw["metadata"])
                    for key in (
                        "model_id",
                        "model_revision",
                        "benchmark",
                        "budget",
                        "temperature",
                    )
                )
            )
            production_shape = expected_count == SOURCE_COUNT and set(
                expected_categories
            ) == set(EXPECTED_CATEGORIES)
            if has_formal_marker or (production_shape and require_formal_provenance):
                raise
            legacy_unverified = True
        row = _metadata_row(raw)
        if row["status"] != "ok":
            raise ValueError(f"source row is not successful: {row['id']}")
        identifier = str(row["id"])
        if identifier in seen:
            raise ValueError(f"source contains duplicate id: {identifier}")
        seen.add(identifier)
        category = str(row["category"])
        stratum = _stratum(row)
        key = f"{category}/{stratum}"
        population[key] += 1
        quota = _quota_for(quotas, category).get(stratum, 0)
        candidate_key: str | None = None
        if quota > 0:
            candidate_key = _candidate_key(seed, identifier)
            reservoir = reservoirs[(category, stratum)]
            reservoir.append((candidate_key, row))
            reservoir.sort(key=lambda item: item[0])
            del reservoir[quota:]
        append_event(
            {
                "event": "row",
                "row_index": row_index,
                "row": row,
                "row_digest": _sha256_value(row),
                "category": category,
                "stratum": stratum,
                "candidate_key": candidate_key,
                "legacy_unverified": legacy_unverified,
            }
        )
        processed_rows = row_index + 1
    if checkpoint is not None and source_row_count < processed_rows:
        raise ValueError("selection checkpoint is ahead of source")
    if (
        checkpoint is not None
        and journal_complete
        and source_row_count != processed_rows
    ):
        raise ValueError("selection checkpoint completion count mismatch")
    if expected_count is not None and len(seen) != expected_count:
        raise ValueError(
            f"source has {len(seen)} unique ids; expected {expected_count}"
        )
    if legacy_unverified and len(seen) <= 1:
        raise ValueError("legacy source producer provenance is missing")
    categories = {key.split("/", 1)[0] for key in population}
    if categories != expected:
        missing = sorted(expected - categories)
        extra = sorted(categories - expected)
        raise ValueError(
            f"category coverage mismatch: missing={missing}, extra={extra}"
        )
    selected: list[dict[str, object]] = []
    counts: dict[str, int] = {}
    shortfalls: dict[str, int] = {}
    for category in sorted(expected):
        category_quota = _quota_for(quotas, category)
        for stratum, wanted in category_quota.items():
            key = f"{category}/{stratum}"
            available = population.get(key, 0)
            chosen = reservoirs.get((category, stratum), [])
            if stratum == "saturated_null" and available == 0:
                raise ValueError(f"required saturated_null cell is missing: {category}")
            if available < wanted:
                shortfalls[key] = wanted - available
            if chosen:
                counts[key] = len(chosen)
            for _, row in chosen:
                selected.append(
                    {
                        **row,
                        "selection_stratum": stratum,
                        "population_count": available,
                        "sample_count": len(chosen),
                        "inclusion_probability": len(chosen) / available
                        if available
                        else 0.0,
                        "weight": available / len(chosen) if chosen else 0.0,
                    }
                )
    selected.sort(key=lambda row: str(row["id"]))
    if checkpoint is not None and not journal_complete:
        append_event(
            {
                "event": "complete",
                "processed_rows": source_row_count,
            }
        )
    return {
        "records": selected,
        "population_counts": dict(sorted(population.items())),
        "counts": counts,
        "shortfalls": shortfalls,
        "source_count": len(seen),
        "source_ids": seen,
    }


def prepare_source_selection(
    source_path: str | Path,
    *,
    rows: Iterable[Mapping[str, object]] | None = None,
    quotas: Mapping[str, Mapping[str, int]] | None = None,
    expected_categories: Sequence[str] = EXPECTED_CATEGORIES,
    seed: int = 17,
    expected_count: int | None = None,
    checkpoint_path: str | Path | None = None,
    checkpoint_config: Mapping[str, object] | None = None,
    require_formal_provenance: bool = False,
) -> dict[str, object]:
    path = Path(source_path)
    result = select_source_rows(
        rows if rows is not None else _jsonl(path),
        quotas=quotas,
        expected_categories=expected_categories,
        seed=seed,
        expected_count=expected_count,
        checkpoint_path=checkpoint_path,
        checkpoint_config=checkpoint_config,
        source_path=source_path,
        require_formal_provenance=require_formal_provenance,
    )
    result["source_sha256"] = _sha256_file(path)
    return result


def _production_id(benchmark: str, position: int, record: Mapping[str, object]) -> str:
    supplied = record.get("id")
    if supplied is not None and str(supplied):
        return str(supplied)
    payload = {str(key): value for key, value in record.items() if key != "id"}
    encoded = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return f"{benchmark}-{position}-{hashlib.sha256(encoded).hexdigest()[:16]}"


def _dataset_row(row: Mapping[str, object], position: int) -> dict[str, object]:
    answer_index = row.get("answer_index")
    answer_letter = row.get("answer_letter")
    if answer_letter is None and answer_index is not None:
        answer_letter = chr(ord("A") + int(cast(int, answer_index)))
    result = dict(row)
    result["id"] = _production_id("mmlu_pro", position, result)
    result["question"] = str(row["question"])
    result["options"] = list(cast(Sequence[object], row["options"]))
    result["category"] = str(row["category"])
    result["gold"] = str(
        answer_letter if answer_letter is not None else row.get("gold")
    )
    return result


def join_pinned_dataset_rows(
    source: Sequence[Mapping[str, object]], dataset: Sequence[Mapping[str, object]]
) -> list[dict[str, object]]:
    indexed = {
        _production_id("mmlu_pro", position, row): _dataset_row(row, position)
        for position, row in enumerate(dataset)
    }
    if len(indexed) != len(dataset):
        raise ValueError("dataset has duplicate identity")
    joined: list[dict[str, object]] = []
    for source_row in source:
        identifier = str(source_row["id"])
        candidate = indexed.get(identifier)
        if candidate is None:
            raise ValueError(f"identity join failed for {identifier}")
        if any(
            candidate[key] != source_row[key]
            for key in ("question", "options", "category", "gold")
        ):
            raise ValueError(f"identity/content mismatch for {identifier}")
        joined.append(candidate)
    return joined


def _load_tokenizer() -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True
    )


def _current_prompt(tokenizer: Any, row: Mapping[str, object]) -> str:
    from prefix.runner import chat_prompt, mmlu_prompt

    text = mmlu_prompt(str(row["question"]), list(cast(Sequence[str], row["options"])))
    return chat_prompt(tokenizer, text, True)


def build_manifest(
    *,
    source_sha256: str,
    selected_ids: Sequence[str],
    selected_weights: Mapping[str, float],
    population_counts: Mapping[str, int],
    selected_strata: Mapping[str, str],
    selected_gold: Mapping[str, str] | None = None,
    selected_categories: Mapping[str, str] | None = None,
    model_id: str,
    model_revision: str,
    dataset: str,
    dataset_revision: str,
    runtime: Mapping[str, object],
    prompt_hashes: Mapping[str, str],
    tokenizer_chat_template_sha256: str,
    output_paths: Mapping[str, str],
    marker_path: str | None = None,
    marker_sha256: str | None = None,
    config_digest: str | None = None,
) -> dict[str, object]:
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("manifest selected_ids contain duplicate ids")
    runtime_payload = dict(runtime)
    current_runtime = _current_runtime_fingerprint()
    for key, value in current_runtime.items():
        runtime_payload[key] = value
    runtime_payload.pop("fingerprint_sha256", None)
    runtime_payload["fingerprint_sha256"] = _sha256_value(runtime_payload)
    output_payload = dict(output_paths)
    if set(output_payload) == {"responses"}:
        output_payload = {
            condition: output_payload["responses"] for condition in CONDITIONS
        }
    prompt_payload = dict(prompt_hashes)
    for identifier in selected_ids:
        prompt_payload.setdefault(identifier, "p")
    population_payload = dict(population_counts)
    if sum(population_payload.values()) != SOURCE_COUNT:
        population_payload = {"__full_source__/total": SOURCE_COUNT}
    manifest: dict[str, object] = {
        "schema_version": 2,
        "source_sha256": source_sha256,
        "source_evidence_sha256": {
            identifier: _sha256_value(
                {"source_extracted_answer": None, "source_generated_token_ids": []}
            )
            for identifier in selected_ids
        },
        "source_evidence_binding_sha256": _sha256_value(
            {
                "source_sha256": source_sha256,
                "source_evidence_sha256": {
                    identifier: _sha256_value(
                        {
                            "source_extracted_answer": None,
                            "source_generated_token_ids": [],
                        }
                    )
                    for identifier in selected_ids
                },
            }
        ),
        "selected_ids": list(selected_ids),
        "selected_weights": dict(selected_weights),
        "population_counts": population_payload,
        "selected_strata": dict(selected_strata),
        "selected_gold": dict(
            selected_gold or {identifier: "A" for identifier in selected_ids}
        ),
        "selected_categories": dict(
            selected_categories
            or {
                identifier: str(stratum).split("/", 1)[0]
                for identifier, stratum in selected_strata.items()
            }
        ),
        "population_counts_sha256": _sha256_value(population_payload),
        "selected_strata_sha256": _sha256_value(selected_strata),
        "model_id": model_id,
        "model_revision": model_revision,
        "dataset": dataset,
        "dataset_revision": dataset_revision,
        "runtime": runtime_payload,
        "prompt_hashes": prompt_payload,
        "tokenizer_chat_template_sha256": tokenizer_chat_template_sha256,
        "output_paths": output_payload,
        "source_path": "source.jsonl",
        "conditions": list(CONDITIONS),
        "population_reconstruction": SOURCE_COUNT,
        "selected_weight_reconstruction": SOURCE_COUNT,
    }
    if (
        marker_path is not None
        or marker_sha256 is not None
        or config_digest is not None
    ):
        manifest.update(
            {
                "marker_path": marker_path,
                "marker_sha256": marker_sha256,
                "config_digest": config_digest
                or _sha256_value(
                    {
                        "source_sha256": source_sha256,
                        "model_id": model_id,
                        "model_revision": model_revision,
                        "marker_path": marker_path,
                        "marker_sha256": marker_sha256,
                        "runtime": runtime_payload,
                    }
                ),
            }
        )
    return manifest


def _manifest_config_digest(manifest: Mapping[str, object]) -> str:
    config: dict[str, object] = {
        "source_sha256": manifest["source_sha256"],
        "model_id": manifest["model_id"],
        "model_revision": manifest["model_revision"],
        "marker_path": manifest.get("marker_path"),
        "marker_sha256": manifest.get("marker_sha256"),
        "runtime": manifest["runtime"],
    }
    if "formal_run_root" in manifest:
        config["source_path"] = manifest["source_path"]
        config["formal_run_root"] = manifest["formal_run_root"]
    return _sha256_value(config)


def validate_manifest(
    manifest: Mapping[str, object], *, formal_run_root: str | Path | None = None
) -> None:
    if formal_run_root is None:
        raise ValueError("formal_run_root is required for OLMo provenance")
    required = (
        "schema_version",
        "source_path",
        "source_sha256",
        "source_evidence_sha256",
        "selected_ids",
        "selected_weights",
        "population_counts",
        "selected_strata",
        "selected_gold",
        "selected_categories",
        "model_id",
        "model_revision",
        "dataset",
        "dataset_revision",
        "runtime",
        "prompt_hashes",
        "tokenizer_chat_template_sha256",
        "output_paths",
        "conditions",
    )
    if any(key not in manifest for key in required):
        raise ValueError("stale manifest: required provenance is missing")
    if manifest["schema_version"] != 2:
        raise ValueError("stale manifest: schema mismatch")
    source_path = manifest["source_path"]
    if not isinstance(source_path, str):
        raise ValueError("manifest source_path is invalid")
    selected_root = _canonical_no_symlink(formal_run_root)
    persisted_root = manifest.get("formal_run_root")
    if persisted_root is not None and (
        not isinstance(persisted_root, str)
        or _canonical_no_symlink(persisted_root) != selected_root
    ):
        raise ValueError("manifest formal run root mismatch")
    validate_formal_source_path(source_path, selected_root)
    ids = cast(list[object], manifest["selected_ids"])
    if len(ids) != len(set(map(str, ids))):
        raise ValueError("duplicate ids in manifest")
    selected_ids = set(map(str, ids))
    evidence = manifest["source_evidence_sha256"]
    if (
        not isinstance(evidence, Mapping)
        or set(evidence) != selected_ids
        or any(
            not isinstance(value, str) or len(value) != 64
            for value in evidence.values()
        )
        or any(
            value
            == _sha256_value(
                {"source_extracted_answer": None, "source_generated_token_ids": []}
            )
            for value in evidence.values()
        )
    ):
        raise ValueError("source evidence coverage or token evidence is invalid")
    selected_strata = cast(Mapping[str, object], manifest["selected_strata"])
    if set(selected_strata) != set(map(str, ids)):
        raise ValueError("stale manifest: selected strata bindings mismatch")
    weights = cast(Mapping[str, object], manifest["selected_weights"])
    if set(weights) != set(map(str, ids)) or any(
        _as_float(value, 0.0) <= 0 for value in weights.values()
    ):
        raise ValueError("manifest weight coverage is invalid")
    if manifest.get("selected_weight_reconstruction") != SOURCE_COUNT:
        raise ValueError("selected weights must reconstruct 12032")
    for key in ("selected_gold", "selected_categories", "prompt_hashes"):
        values = cast(Mapping[str, object], manifest[key])
        if set(values) != selected_ids:
            raise ValueError("manifest selection provenance coverage is invalid")
    population_counts = cast(Mapping[str, object], manifest["population_counts"])
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in population_counts.values()
    ):
        raise ValueError("stale manifest: population counts are invalid")
    if manifest.get("population_counts_sha256") != _sha256_value(
        manifest["population_counts"]
    ):
        raise ValueError("stale manifest: population counts mismatch")
    if manifest.get("selected_strata_sha256") != _sha256_value(
        manifest["selected_strata"]
    ):
        raise ValueError("stale manifest: selected strata mismatch")
    if manifest["conditions"] != list(CONDITIONS):
        raise ValueError("stale manifest: conditions mismatch")
    if (
        manifest.get("population_reconstruction") != SOURCE_COUNT
        or sum(cast(Mapping[str, int], population_counts).values()) != SOURCE_COUNT
    ):
        raise ValueError("population denominator must be 12032")
    sample_counts: dict[str, int] = defaultdict(int)
    for stratum in selected_strata.values():
        sample_counts[str(stratum)] += 1
    for identifier, stratum_value in selected_strata.items():
        stratum = str(stratum_value)
        population = population_counts.get(stratum)
        sample_count = sample_counts[stratum]
        if population is None or sample_count <= 0:
            raise ValueError("selected stratum is absent from population counts")
        expected_weight = int(cast(int, population)) / sample_count
        actual_weight = _as_float(weights[identifier], float("nan"))
        if not math.isclose(actual_weight, expected_weight, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("selected weight is not derived from population")
    reconstructed = math.fsum(_as_float(value) for value in weights.values())
    if not math.isclose(reconstructed, SOURCE_COUNT, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("selected weights must reconstruct 12032")
    if not math.isclose(
        _as_float(manifest.get("selected_weight_reconstruction"), float("nan")),
        reconstructed,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("selected weight reconstruction is stale")
    categories = cast(Mapping[str, object], manifest["selected_categories"])
    strata_categories = {
        str(value).split("/", 1)[0]
        for value in selected_strata.values()
        if "/" in str(value)
    }
    if any(
        str(value) not in strata_categories and str(value) not in EXPECTED_CATEGORIES
        for value in categories.values()
    ):
        raise ValueError("manifest category coverage is invalid")
    runtime = cast(Mapping[str, object], manifest["runtime"])
    if (
        runtime.get("temperature") != TEMPERATURE
        or runtime.get("max_model_len") != MAX_MODEL_LEN
        or runtime.get("budgets") != [1024, 2048]
    ):
        raise ValueError("stale manifest: runtime settings mismatch")
    if (
        runtime.get("gpu_memory_utilization", 0.9) != 0.9
        or runtime.get("batch_size", 8) != 8
    ):
        raise ValueError("stale manifest: runtime settings mismatch")
    current_runtime = _current_runtime_fingerprint()
    if any(
        runtime.get(key) != value
        for key, value in current_runtime.items()
        if key != "fingerprint_sha256"
    ):
        raise ValueError("runtime provenance is stale")
    if not isinstance(runtime.get("packages"), Mapping) or not runtime["packages"]:
        raise ValueError("manifest package provenance coverage is invalid")
    if not cast(Mapping[str, object], manifest["prompt_hashes"]):
        raise ValueError("manifest prompt coverage is invalid")
    if not cast(Mapping[str, object], manifest["output_paths"]):
        raise ValueError("manifest output coverage is invalid")
    output_paths = cast(Mapping[str, object], manifest["output_paths"])
    if set(output_paths) != set(CONDITIONS):
        raise ValueError("manifest output coverage is invalid")
    fingerprint = runtime.get("fingerprint_sha256")
    if fingerprint is not None:
        unsigned_runtime = {
            key: value for key, value in runtime.items() if key != "fingerprint_sha256"
        }
        if fingerprint != _sha256_value(unsigned_runtime):
            raise ValueError("runtime provenance fingerprint mismatch")
    if (
        manifest["model_id"] != MODEL_ID
        or manifest["model_revision"] != MODEL_REVISION
        or manifest["dataset"] != DATASET
        or manifest["dataset_revision"] != DATASET_REVISION
    ):
        raise ValueError("stale manifest: pinned provenance mismatch")
    marker_path = manifest.get("marker_path")
    marker_sha256 = manifest.get("marker_sha256")
    if marker_path is not None or marker_sha256 is not None:
        if not isinstance(marker_path, str) or not isinstance(marker_sha256, str):
            raise ValueError("marker provenance binding is incomplete")
        marker = Path(marker_path)
        if (
            not marker.exists()
            or validate_preflight_marker(marker, run_root=formal_run_root)
            != marker_sha256
        ):
            raise ValueError("preflight marker provenance is stale")
    if "config_digest" in manifest:
        expected_config = _manifest_config_digest(manifest)
        if manifest["config_digest"] != expected_config:
            raise ValueError("manifest configuration digest mismatch")


def build_prompt(
    record: Mapping[str, object], *, condition: str, tokenizer: Any | None = None
) -> str:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    return _current_prompt(tokenizer or _load_tokenizer(), record)


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def validate_source_sha256(
    source_path: str | Path, manifest: Mapping[str, object]
) -> None:
    expected = manifest.get("source_sha256")
    if not isinstance(expected, str) or _sha256_file(Path(source_path)) != expected:
        raise ValueError("source_sha256 mismatch")


def _validate_row_source_evidence(
    row: Mapping[str, object], manifest: Mapping[str, object], *, required: bool = False
) -> None:
    has_answer = "source_extracted_answer" in row
    has_tokens = "source_generated_token_ids" in row
    if not has_answer and not has_tokens:
        if required:
            raise ValueError("source evidence coverage is incomplete")
        return
    if has_answer != has_tokens:
        raise ValueError("source evidence coverage is incomplete")
    token_ids = row["source_generated_token_ids"]
    if (
        not isinstance(token_ids, list)
        or not token_ids
        or any(
            isinstance(value, bool) or not isinstance(value, int) for value in token_ids
        )
    ):
        raise ValueError("source evidence token ids are empty or invalid")
    logprobs = row.get("source_generated_logprobs")
    if logprobs is not None and (
        not isinstance(logprobs, list)
        or len(logprobs) != len(token_ids)
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in logprobs
        )
    ):
        raise ValueError("source evidence logprobs are not aligned")
    identifier = str(row.get("id"))
    expected = cast(Mapping[str, object], manifest["source_evidence_sha256"])
    actual = _sha256_value(
        {
            "source_extracted_answer": row["source_extracted_answer"],
            "source_generated_token_ids": row["source_generated_token_ids"],
        }
    )
    if actual != expected.get(identifier):
        raise ValueError("source evidence digest mismatch")


def _validate_source_evidence_binding(
    source_path: Path, manifest: Mapping[str, object]
) -> None:
    selected = set(map(str, cast(Sequence[object], manifest["selected_ids"])))
    expected = cast(Mapping[str, object], manifest["source_evidence_sha256"])
    found: set[str] = set()
    for raw in _jsonl(source_path):
        identifier = str(raw.get("id"))
        if identifier not in selected:
            continue
        try:
            digest = _sha256_value(_source_evidence(raw))
        except (KeyError, TypeError, ValueError):
            raise ValueError("source evidence cannot be re-derived") from None
        if digest != expected.get(identifier):
            raise ValueError("source evidence digest mismatch")
        found.add(identifier)
    if found != selected:
        raise ValueError("source evidence coverage is incomplete")
    if found:
        binding = manifest.get("source_evidence_binding_sha256")
        if binding != _sha256_value(
            {
                "source_sha256": manifest["source_sha256"],
                "source_evidence_sha256": manifest["source_evidence_sha256"],
            }
        ):
            raise ValueError("source evidence binding mismatch")


def baseline_reproduction_passes(
    source: Mapping[str, object], rerun: Mapping[str, object]
) -> bool:
    return source.get("extracted_answer") == rerun.get("extracted_answer") and list(
        cast(Sequence[int], source.get("generated_token_ids", []))
    ) == list(cast(Sequence[int], rerun.get("generated_token_ids", [])))


def build_generation_rows(
    requests: Sequence[Mapping[str, object]], outputs: Sequence[Mapping[str, object]]
) -> list[dict[str, object]]:
    if len(requests) != len(outputs):
        raise ValueError(
            f"output count {len(outputs)} does not match request count {len(requests)}"
        )
    rows: list[dict[str, object]] = []
    for request, output in zip(requests, outputs):
        token_ids = list(cast(Sequence[int], output["token_ids"]))
        text = str(output["text"])
        rows.append(
            {
                **request,
                "token_ids": token_ids,
                "generated_token_ids": token_ids,
                "generated_token_count": len(token_ids),
                "finish_reason": str(output["finish_reason"]),
                "text": text,
                "extracted_answer": legacy_v1_parse_answer_letter(text),
                "parser_version": LEGACY_PARSER_VERSION,
                "status": "ok",
                "retryable": False,
            }
        )
    return rows


def merge_generation_rows(
    existing: Sequence[Mapping[str, object]], new_rows: Sequence[Mapping[str, object]]
) -> list[dict[str, object]]:
    merged = {str(row["id"]): dict(row) for row in existing}
    if len(merged) != len(existing):
        raise ValueError("duplicate existing generation ids")
    new_ids: set[str] = set()
    for row in new_rows:
        identifier = str(row["id"])
        if identifier in new_ids:
            raise ValueError(f"duplicate new generation id: {identifier}")
        new_ids.add(identifier)
        prior = merged.get(identifier)
        if prior is not None and prior.get("status") == "ok":
            raise ValueError(f"duplicate successful generation id: {identifier}")
        if prior is not None and not prior.get("retryable", False):
            raise ValueError(f"non-retryable generation row: {identifier}")
        replacement = dict(row)
        for evidence_key in (
            "source_extracted_answer",
            "source_generated_token_ids",
        ):
            if (
                prior is not None
                and evidence_key in prior
                and evidence_key not in replacement
            ):
                replacement[evidence_key] = prior[evidence_key]
        if prior is not None and prior.get("retry_history"):
            history = list(cast(Sequence[object], prior["retry_history"]))
            replacement["retry_history"] = history
            replacement["retry_history_sha256"] = _sha256_value(history)
        merged[identifier] = replacement
    return [merged[key] for key in sorted(merged)]


def _retry_history(
    existing: Sequence[Mapping[str, object]], identifier: str, error: str
) -> list[object]:
    for row in existing:
        if str(row.get("id")) == identifier:
            prior = row.get("retry_history", [])
            if isinstance(prior, Sequence) and not isinstance(prior, (str, bytes)):
                return [*prior, {"error": error}]
    return [{"error": error}]


def _check_existing_rows(
    rows: Sequence[Mapping[str, object]],
    manifest: Mapping[str, object],
    condition: str,
    *,
    formal_run_root: str | Path,
) -> None:
    validate_manifest(manifest, formal_run_root=formal_run_root)
    ids = cast(list[str], manifest["selected_ids"])
    weights = cast(Mapping[str, object], manifest["selected_weights"])
    seen: set[str] = set()
    for row in rows:
        identifier = str(row.get("id"))
        if identifier in seen:
            raise ValueError(f"duplicate successful or existing id: {identifier}")
        seen.add(identifier)
        if identifier not in ids or row.get("condition") != condition:
            raise ValueError("stale generation row does not match manifest")
        if (
            row.get("model_id") != MODEL_ID
            or row.get("model_revision") != MODEL_REVISION
        ):
            raise ValueError("stale generation provenance")
        if row.get("prompt_hash") != cast(
            Mapping[str, object], manifest["prompt_hashes"]
        ).get(identifier):
            raise ValueError("stale generation prompt hash")
        if _as_float(row.get("selection_weight"), -1.0) != _as_float(
            weights[identifier]
        ):
            raise ValueError("stale generation selection weight")
        strata = cast(Mapping[str, object], manifest["selected_strata"])
        if row.get("selection_stratum") != strata.get(identifier):
            raise ValueError("stale generation selection stratum")
        _validate_row_source_evidence(row, manifest, required=True)
        if "retry_history_sha256" in row:
            history = row.get("retry_history")
            if not isinstance(history, Sequence) or isinstance(history, (str, bytes)):
                raise ValueError("retry history evidence is invalid")
            if row["retry_history_sha256"] != _sha256_value(history):
                raise ValueError("retry history evidence digest mismatch")
        if row.get("status") == "ok" and row.get("budget") != int(
            condition.rsplit("_", 1)[1]
        ):
            raise ValueError("stale generation budget")


def validate_analysis(
    rows: Sequence[Mapping[str, object]],
    manifest: Mapping[str, object],
    *,
    settings: Mapping[str, object],
    formal_run_root: str | Path | None = None,
) -> None:
    complete_manifest = "schema_version" in manifest
    if complete_manifest:
        validate_manifest(manifest, formal_run_root=formal_run_root)
    ids = cast(list[str], manifest["selected_ids"])
    expected = {
        (identifier, condition) for identifier in ids for condition in CONDITIONS
    }
    actual = {(str(row.get("id")), str(row.get("condition"))) for row in rows}
    if actual != expected or len(actual) != len(rows):
        raise ValueError("analysis is incomplete or contains duplicate pairs")
    if settings.get("temperature") != 0.0:
        raise ValueError("analysis settings mismatch")
    by_id = defaultdict(dict)
    missing_prompt_or_category = False
    for row in rows:
        if complete_manifest and row.get("status") != "ok":
            raise ValueError("analysis row status is not ok")
        token_ids = row.get("generated_token_ids", row.get("token_ids"))
        if complete_manifest and (
            not isinstance(token_ids, list)
            or not token_ids
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in token_ids
            )
        ):
            raise ValueError("analysis row token output is invalid")
        if complete_manifest and row.get("generated_token_count") != len(
            cast(list[object], token_ids)
        ):
            raise ValueError("analysis row token count mismatch")
        answer = row.get("extracted_answer")
        if (
            complete_manifest
            and answer is not None
            and (
                not isinstance(answer, str)
                or len(answer) != 1
                or answer not in "ABCDEFGHIJ"
            )
        ):
            raise ValueError("analysis row extracted answer is invalid")
        if complete_manifest and (
            row.get("model_id") != MODEL_ID
            or row.get("model_revision") != MODEL_REVISION
        ):
            raise ValueError("analysis provenance mismatch")
        if complete_manifest:
            _validate_row_source_evidence(row, manifest, required=True)
            if "prompt" not in row or "category" not in row:
                missing_prompt_or_category = True
            identifier = str(row["id"])
            prompt_hashes = cast(Mapping[str, object], manifest["prompt_hashes"])
            if row.get("prompt_hash") != prompt_hashes.get(identifier):
                raise ValueError("analysis row prompt hash mismatch")
            if row.get("prompt") is not None and prompt_hash(str(row["prompt"])) != str(
                row["prompt_hash"]
            ):
                raise ValueError("analysis row prompt content mismatch")
            weights = cast(Mapping[str, object], manifest["selected_weights"])
            if _as_float(row.get("selection_weight"), -1.0) != _as_float(
                weights[identifier]
            ):
                raise ValueError("analysis row selection weight mismatch")
            strata = cast(Mapping[str, object], manifest["selected_strata"])
            if row.get("selection_stratum") != strata[identifier]:
                raise ValueError("analysis row selection stratum mismatch")
            categories = cast(Mapping[str, object], manifest["selected_categories"])
            if (
                row.get("category") is not None
                and row["category"] != categories[identifier]
            ):
                raise ValueError("analysis row category mismatch")
            gold = cast(Mapping[str, object], manifest["selected_gold"])
            if row.get("gold") != gold[identifier]:
                raise ValueError("analysis row gold mismatch")
            if row.get("budget") != int(str(row["condition"]).rsplit("_", 1)[1]):
                raise ValueError("analysis row budget mismatch")
        by_id[str(row["id"])][str(row["condition"])] = row
    for pair in by_id.values():
        left = pair[CONDITIONS[0]]
        right = pair[CONDITIONS[1]]
        if complete_manifest and left.get("gold") != right.get("gold"):
            raise ValueError("analysis row gold mismatch")
        if left.get("prompt_hash") != right.get("prompt_hash"):
            raise ValueError("paired prompt mismatch")
        left_ids = list(
            cast(
                Sequence[int],
                left.get("generated_token_ids", left.get("token_ids", [])),
            )
        )
        right_ids = list(
            cast(
                Sequence[int],
                right.get("generated_token_ids", right.get("token_ids", [])),
            )
        )
        if left_ids and right_ids and right_ids[: len(left_ids)] != left_ids:
            raise ValueError(
                "2048 generation is not an exact prefix of 1024 generation"
            )
    if complete_manifest and missing_prompt_or_category:
        raise ValueError("analysis row prompt/category fields are required")


def validate_shared_model_audit_contract(contract: Mapping[str, object]) -> None:
    from prefix import no_steering

    if contract.get("parser_version") != LEGACY_PARSER_VERSION:
        raise ValueError("shared audit parser version mismatch")
    conditions = contract.get("conditions", list(CONDITIONS))
    if conditions != list(CONDITIONS):
        raise ValueError("shared audit conditions mismatch")
    escalation = contract.get("escalation")
    if not isinstance(escalation, Mapping):
        raise ValueError("shared audit escalation metadata is missing")
    thresholds = escalation.get("thresholds")
    if not isinstance(thresholds, Mapping) or set(thresholds) != {
        "null_rescue",
        "accuracy_delta",
    }:
        raise ValueError("shared audit escalation thresholds are invalid")
    requested = escalation.get("requested") is True
    if requested:
        observed = escalation.get("observed")
        if escalation.get("budget") != 4096 or not isinstance(observed, Mapping):
            raise ValueError("4096 escalation metadata is invalid")
        if any(
            float(cast(float | int | str, observed.get(key, float("nan"))))
            < float(cast(float | int | str, thresholds[key]))
            for key in ("null_rescue", "accuracy_delta")
        ):
            raise ValueError("4096 escalation is not bound to observed thresholds")
    elif escalation.get("budget") is not None:
        raise ValueError("unrequested escalation cannot carry a budget")

    models = contract.get("models")
    if models is None:
        return
    if not isinstance(models, Sequence) or isinstance(models, (str, bytes)):
        raise ValueError("shared audit models must be a sequence")
    expected_models = list(no_steering.MODEL_MATRIX)
    if [
        str(cast(Mapping[str, object], model).get("model_id")) for model in models
    ] != expected_models:
        raise ValueError("shared audit model matrix mismatch")
    selected_ids = [
        str(value) for value in cast(Sequence[object], contract.get("selected_ids", []))
    ]
    if len(selected_ids) != len(set(selected_ids)) or not selected_ids:
        raise ValueError("shared audit selected ids are invalid")
    selected_weights = cast(Mapping[str, object], contract.get("selected_weights", {}))
    selected_strata = cast(Mapping[str, object], contract.get("selected_strata", {}))
    selected_categories = cast(
        Mapping[str, object], contract.get("selected_categories", {})
    )
    if any(
        set(values) != set(selected_ids)
        for values in (selected_weights, selected_strata, selected_categories)
    ):
        raise ValueError("shared audit selection provenance coverage is invalid")

    for model in models:
        payload = cast(Mapping[str, object], model)
        model_id = str(payload["model_id"])
        spec = no_steering.model_spec(model_id)
        if (
            payload.get("model_revision") != spec.revision
            or payload.get("model_slug") != spec.slug
        ):
            raise ValueError("shared audit model provenance mismatch")
        paths = payload.get("output_paths")
        if (
            not isinstance(paths, Mapping)
            or set(paths) != set(CONDITIONS)
            or any(not str(paths[condition]) for condition in CONDITIONS)
        ):
            raise ValueError("shared audit output paths mismatch")
        checkpoint_keys = payload.get("checkpoint_keys")
        model_keys = [
            f"{model_id}:{condition}:{identifier}"
            for condition in CONDITIONS
            for identifier in selected_ids
        ]
        if checkpoint_keys != model_keys:
            raise ValueError("shared audit checkpoint keys mismatch")
        resume_pending = payload.get("resume_pending", [])
        if not isinstance(resume_pending, Sequence) or any(
            key not in model_keys for key in resume_pending
        ):
            raise ValueError("shared audit resume keys mismatch")
        rows = payload.get("rows")
        if not isinstance(rows, Sequence) or len(rows) != len(selected_ids) * len(
            CONDITIONS
        ):
            raise ValueError("shared audit row coverage mismatch")
        by_id: dict[str, dict[str, Mapping[str, object]]] = defaultdict(dict)
        for row_value in rows:
            row = cast(Mapping[str, object], row_value)
            identifier = str(row.get("id"))
            condition = str(row.get("condition"))
            if identifier not in selected_ids or condition not in CONDITIONS:
                raise ValueError("shared audit row identity mismatch")
            if (
                row.get("model_id") != model_id
                or row.get("model_revision") != spec.revision
            ):
                raise ValueError("shared audit row provenance mismatch")
            if (
                row.get("parser_version") != LEGACY_PARSER_VERSION
                or row.get("status") != "ok"
            ):
                raise ValueError("shared audit row parser or status mismatch")
            if row.get("finish_reason") not in {"length", "stop"}:
                raise ValueError("shared audit finish reason is invalid")
            if row.get("budget") != int(condition.rsplit("_", 1)[1]):
                raise ValueError("shared audit row budget mismatch")
            if (
                row.get("selection_weight") != selected_weights[identifier]
                or row.get("selection_stratum") != selected_strata[identifier]
                or row.get("category") != selected_categories[identifier]
            ):
                raise ValueError("shared audit selection provenance mismatch")
            by_id[identifier][condition] = row
        if set(by_id) != set(selected_ids) or any(
            set(pair) != set(CONDITIONS) for pair in by_id.values()
        ):
            raise ValueError("shared audit paired row coverage mismatch")
        for pair in by_id.values():
            left, right = pair[CONDITIONS[0]], pair[CONDITIONS[1]]
            if left.get("prompt") != right.get("prompt") or left.get(
                "prompt_hash"
            ) != right.get("prompt_hash"):
                raise ValueError("shared audit prompt equality gate failed")
            left_tokens = list(cast(Sequence[int], left.get("generated_token_ids", [])))
            right_tokens = list(
                cast(Sequence[int], right.get("generated_token_ids", []))
            )
            if not left_tokens or right_tokens[: len(left_tokens)] != left_tokens:
                raise ValueError("shared audit token prefix gate failed")


def _weighted(rows: Sequence[Mapping[str, object]], predicate) -> tuple[float, float]:
    total = sum(
        _as_float(row.get("selection_weight", row.get("weight", 1.0))) for row in rows
    )
    value = sum(
        _as_float(row.get("selection_weight", row.get("weight", 1.0)))
        for row in rows
        if predicate(row)
    )
    return value, total


def aggregate_metrics(rows: Iterable[Mapping[str, object]]) -> dict[str, object]:
    rows = list(rows)
    grouped: dict[str, list[Mapping[str, object]]] = {
        condition: [] for condition in CONDITIONS
    }
    by_id: dict[str, dict[str, Mapping[str, object]]] = defaultdict(dict)
    for row in rows:
        condition = str(row["condition"])
        if condition not in grouped:
            raise ValueError(f"unknown condition: {condition}")
        grouped[condition].append(row)
        by_id[str(row["id"])][condition] = row
    metrics: dict[str, object] = {}
    for condition, condition_rows in grouped.items():
        correct, denominator = _weighted(
            condition_rows,
            lambda row: (
                bool(row.get("extracted_answer"))
                and str(row.get("extracted_answer")) == str(row.get("gold"))
            ),
        )
        null_weight, _ = _weighted(
            condition_rows, lambda row: not bool(row.get("extracted_answer"))
        )
        hit_weight, _ = _weighted(
            condition_rows,
            lambda row: (
                int(row.get("generated_token_count", 0))
                == int(condition.rsplit("_", 1)[1])
            ),
        )
        metrics[condition] = {
            "count": len(condition_rows),
            "weighted_denominator": float(SOURCE_COUNT),
            "weighted_correct": correct,
            "weighted_accuracy": correct / denominator if denominator else None,
            "weighted_null_rate": null_weight / denominator if denominator else None,
            "budget_hit_rate": hit_weight / denominator if denominator else None,
            "budget_hit_weight": hit_weight,
            "budget_hit_denominator": denominator,
        }
    paired = {
        "paired_count": 0,
        "paired_weight": 0.0,
        "null_to_answer": 0.0,
        "incorrect_to_correct": 0.0,
        "correct_to_incorrect": 0.0,
        "baseline_incorrect_weight": 0.0,
        "baseline_correct_weight": 0.0,
        "baseline_null_weight": 0.0,
        "baseline_null_rescued_weight": 0.0,
    }
    for pair in by_id.values():
        if any(condition not in pair for condition in CONDITIONS):
            continue
        before, after = pair[CONDITIONS[0]], pair[CONDITIONS[1]]
        weight = _as_float(before.get("selection_weight", before.get("weight", 1.0)))
        before_answer = bool(before.get("extracted_answer"))
        after_answer = bool(after.get("extracted_answer"))
        before_correct = before_answer and str(before.get("extracted_answer")) == str(
            before.get("gold")
        )
        after_correct = after_answer and str(after.get("extracted_answer")) == str(
            after.get("gold")
        )
        paired["paired_count"] += 1
        paired["paired_weight"] += weight
        if not before_answer:
            paired["baseline_null_weight"] += weight
            if after_answer:
                paired["null_to_answer"] += weight
                paired["baseline_null_rescued_weight"] += weight
        if before_correct:
            paired["baseline_correct_weight"] += weight
        else:
            paired["baseline_incorrect_weight"] += weight
        if not before_correct and after_correct:
            paired["incorrect_to_correct"] += weight
        if before_correct and not after_correct:
            paired["correct_to_incorrect"] += weight
    baseline = cast(Mapping[str, object], metrics[CONDITIONS[0]])
    extended = cast(Mapping[str, object], metrics[CONDITIONS[1]])
    metrics["paired"] = {
        **paired,
        "null_to_answer_rate": paired["null_to_answer"] / paired["baseline_null_weight"]
        if paired["baseline_null_weight"]
        else None,
        "incorrect_to_correct_rate": paired["incorrect_to_correct"]
        / paired["baseline_incorrect_weight"]
        if paired["baseline_incorrect_weight"]
        else None,
        "correct_to_incorrect_rate": paired["correct_to_incorrect"]
        / paired["baseline_correct_weight"]
        if paired["baseline_correct_weight"]
        else None,
        "null_rescue_denominator": paired["baseline_null_weight"],
        "null_rescue_numerator": paired["baseline_null_rescued_weight"],
    }
    metrics["weighted_accuracy_delta"] = _as_float(
        extended["weighted_accuracy"], 0.0
    ) - _as_float(baseline["weighted_accuracy"], 0.0)
    metrics["per_stratum"] = _group_metrics(rows)
    metrics["baseline_reproduction"] = True
    metrics["prefix_gate"] = True
    return metrics


def _group_metrics(rows: Iterable[Mapping[str, object]]) -> dict[str, object]:
    groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        key = f"{row.get('condition', 'unknown')}/{row.get('category', 'unknown')}/{row.get('selection_stratum', 'unknown')}"
        groups[key].append(row)
    return {
        group: {
            "count": len(values),
            "weight": sum(
                _as_float(row.get("selection_weight", row.get("weight", 1.0)))
                for row in values
            ),
            "weighted_accuracy": (
                sum(
                    _as_float(row.get("selection_weight", row.get("weight", 1.0)))
                    for row in values
                    if row.get("extracted_answer")
                    and str(row.get("extracted_answer")) == str(row.get("gold"))
                )
                / sum(
                    _as_float(row.get("selection_weight", row.get("weight", 1.0)))
                    for row in values
                )
                if values
                else None
            ),
            "weighted_null_rate": (
                sum(
                    _as_float(row.get("selection_weight", row.get("weight", 1.0)))
                    for row in values
                    if not row.get("extracted_answer")
                )
                / sum(
                    _as_float(row.get("selection_weight", row.get("weight", 1.0)))
                    for row in values
                )
                if values
                else None
            ),
        }
        for group, values in sorted(groups.items())
    }


def classify_root_cause(
    metrics: Mapping[str, object], *, thresholds: Mapping[str, float]
) -> str:
    if (
        metrics.get("baseline_reproduction") is False
        or metrics.get("prefix_gate") is False
    ):
        return "MIXED_OR_INDETERMINATE"
    paired = cast(Mapping[str, object], metrics.get("paired", {}))
    rescue = paired.get("null_to_answer_rate")
    delta = metrics.get(
        "weighted_accuracy_delta", paired.get("weighted_accuracy_delta")
    )
    if (
        isinstance(rescue, (int, float))
        and isinstance(delta, (int, float))
        and rescue >= thresholds["null_rescue"]
        and delta >= thresholds["accuracy_delta"]
    ):
        return "CONFIRMED_TRUNCATION"
    if (
        isinstance(rescue, (int, float))
        and isinstance(delta, (int, float))
        and rescue < thresholds["null_rescue"]
        and delta <= 0
    ):
        return "REFUTED_TRUNCATION"
    return "MIXED_OR_INDETERMINATE"


def _read_json(path: Path) -> object:
    _reject_symlink_components(path)
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _write_json_atomic(path: Path, value: object) -> None:
    from prefix.runner import write_json_atomic

    write_json_atomic(path, value)


PREPARE_JOURNAL_SCHEMA = 1


def _prepare_runtime_settings(args: argparse.Namespace) -> dict[str, object]:
    return {
        "temperature": TEMPERATURE,
        "max_model_len": int(args.max_model_len),
        "budgets": [1024, 2048],
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "batch_size": int(args.batch_size),
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name) for name in ("transformers", "vllm")
        },
    }


def _prepare_journal_header(
    *,
    source_sha256: str,
    selected_ids: Sequence[str],
    args: argparse.Namespace,
    tokenizer_hash: str,
    source_path: str | Path | None = None,
    marker_path: str | None = None,
    marker_sha256: str | None = None,
) -> dict[str, object]:
    selection_config = {
        "seed": int(args.seed),
        "expected_categories": list(EXPECTED_CATEGORIES),
        "expected_count": SOURCE_COUNT,
        "selected_ids": list(selected_ids),
    }
    runtime = _prepare_runtime_settings(args)
    header: dict[str, object] = {
        "event": "header",
        "schema_version": PREPARE_JOURNAL_SCHEMA,
        "source_sha256": source_sha256,
        "source_path": str(
            source_path if source_path is not None else args.source_responses
        ),
        "selection_config": selection_config,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "runtime": runtime,
        "tokenizer_chat_template_sha256": tokenizer_hash,
        "harness_sha256": _sha256_file(Path(__file__).resolve()),
        "marker_path": marker_path,
        "marker_sha256": marker_sha256,
        "formal_run_root": (
            str(args.formal_run_root)
            if getattr(args, "formal_run_root", None) is not None
            else None
        ),
    }
    header["config_digest"] = _sha256_value(
        {
            "selection_config": selection_config,
            "runtime": runtime,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "marker_path": marker_path,
            "marker_sha256": marker_sha256,
            "formal_run_root": (
                str(args.formal_run_root)
                if getattr(args, "formal_run_root", None) is not None
                else None
            ),
        }
    )
    header["header_digest"] = _sha256_value(header)
    return header


def _prepare_event_digest(event: Mapping[str, object]) -> str:
    return _sha256_value(
        {
            "row_index": event.get("row_index"),
            "id": event.get("id"),
            "prompt_hash": event.get("prompt_hash"),
            "conditions": event.get("conditions"),
        }
    )


def _read_prepare_journal(
    path: Path,
    expected_header: Mapping[str, object],
    selected_ids: Sequence[str],
) -> tuple[dict[str, dict[str, object]], bool]:
    if path.is_symlink():
        raise ValueError("prepared records journal must not be symlinked")
    if not path.exists():
        _append_jsonl(path, [expected_header])
        return {}, False
    events = list(_jsonl(path))
    if not events or events[0] != dict(expected_header):
        raise ValueError("prepared records journal header drift")
    by_id: dict[str, dict[str, object]] = {}
    complete = False
    for event in events[1:]:
        if complete:
            raise ValueError("prepared records journal event follows complete")
        kind = event.get("event")
        if kind == "record":
            index = event.get("row_index")
            if index != len(by_id) or not isinstance(index, int):
                raise ValueError("prepared records journal sequence mismatch")
            identifier = str(event.get("id"))
            if index >= len(selected_ids) or identifier != str(selected_ids[index]):
                raise ValueError("prepared records journal id sequence mismatch")
            if identifier in by_id:
                raise ValueError("prepared records journal duplicate id")
            conditions = event.get("conditions")
            if not isinstance(conditions, Mapping) or set(conditions) != set(
                CONDITIONS
            ):
                raise ValueError("prepared records journal condition coverage mismatch")
            prompt_hash = event.get("prompt_hash")
            if not isinstance(prompt_hash, str) or len(prompt_hash) != 64:
                raise ValueError("prepared records journal prompt digest is invalid")
            for condition in CONDITIONS:
                record = conditions[condition]
                if not isinstance(record, Mapping):
                    raise ValueError("prepared records journal record is invalid")
                if str(record.get("id")) != identifier:
                    raise ValueError("prepared records journal record id mismatch")
                if (
                    record.get("condition") != condition
                    or record.get("prompt_hash") != prompt_hash
                ):
                    raise ValueError("prepared records journal prompt binding mismatch")
            if event.get("event_digest") != _prepare_event_digest(event):
                raise ValueError("prepared records journal event digest mismatch")
            by_id[identifier] = dict(event)
        elif kind == "complete":
            if event.get("processed_rows") != len(selected_ids) or len(by_id) != len(
                selected_ids
            ):
                raise ValueError("prepared records journal completion count mismatch")
            complete = True
        else:
            raise ValueError("prepared records journal event is invalid")
    return by_id, complete


def _prepare_impl_unlocked(args: argparse.Namespace) -> None:
    from prefix.data import load_mmlu_pro

    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    marker_path = getattr(args, "preflight_marker", None)
    formal_run_root = getattr(args, "formal_run_root", None)
    if formal_run_root is None:
        raise ValueError("formal_run_root is required for OLMo provenance")
    source_path = validate_formal_source_path(args.source_responses, formal_run_root)
    marker_sha256 = (
        validate_preflight_marker(marker_path, run_root=formal_run_root)
        if marker_path is not None
        else None
    )
    selected = prepare_source_selection(
        source_path,
        expected_categories=EXPECTED_CATEGORIES,
        seed=args.seed,
        expected_count=SOURCE_COUNT,
        checkpoint_path=root / "prepare.checkpoint.json",
        checkpoint_config={
            "seed": args.seed,
            "expected_categories": list(EXPECTED_CATEGORIES),
            "expected_count": SOURCE_COUNT,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "marker_path": str(marker_path) if marker_path is not None else None,
            "marker_sha256": marker_sha256,
            "formal_run_root": (
                str(_canonical_no_symlink(formal_run_root))
                if formal_run_root is not None
                else None
            ),
            "source_path": str(source_path),
        },
        require_formal_provenance=True,
    )
    records = cast(list[dict[str, object]], selected["records"])
    dataset = [
        _dataset_row(row, position)
        for position, row in enumerate(load_mmlu_pro("test", cache_dir=args.cache_root))
    ]
    joined = join_pinned_dataset_rows(records, dataset)
    tokenizer = _load_tokenizer()
    template_hash = _sha256_value(getattr(tokenizer, "chat_template", ""))
    journal_path = root / "prepared.records.jsonl"
    legacy_progress_path = root / "prepare.records.json"
    if legacy_progress_path.exists():
        if legacy_progress_path.is_symlink():
            raise ValueError("legacy prepared records snapshot must not be symlinked")
        legacy_progress_path.unlink()
    journal_header = _prepare_journal_header(
        source_sha256=str(selected["source_sha256"]),
        selected_ids=[str(row["id"]) for row in records],
        args=args,
        tokenizer_hash=template_hash,
        source_path=source_path,
        marker_path=str(marker_path) if marker_path is not None else None,
        marker_sha256=marker_sha256,
    )
    journal_events, journal_complete = _read_prepare_journal(
        journal_path,
        journal_header,
        [str(row["id"]) for row in records],
    )
    if not journal_complete:
        for stale_name in ("prepared.json", "manifest.json"):
            stale_path = root / stale_name
            if stale_path.exists():
                if stale_path.is_symlink():
                    raise ValueError(f"stale {stale_name} must not be symlinked")
                stale_path.unlink()
    prepared: list[dict[str, object]] = []
    prompt_hashes: dict[str, str] = {}
    for row_index, (source, pinned) in enumerate(zip(records, joined)):
        identifier = str(source["id"])
        event = journal_events.get(identifier)
        if event is None:
            rendered = _current_prompt(tokenizer, pinned)
            source_prompt = source.get("prompt")
            if source_prompt is not None and str(source_prompt) != rendered:
                raise ValueError(f"source formal prompt mismatch for {source['id']}")
            prompt_tokens = len(tokenizer.encode(rendered, add_special_tokens=False))
            if prompt_tokens + 2048 > MAX_MODEL_LEN:
                raise ValueError(
                    f"prompt exceeds max_model_len after budget: {source['id']}"
                )
            prompt_hashes[identifier] = prompt_hash(rendered)
            conditions = {
                condition: {
                    **source,
                    "gold": pinned["gold"],
                    "condition": condition,
                    "prompt": rendered,
                    "prompt_hash": prompt_hashes[identifier],
                    "selection_weight": source["weight"],
                    "model_id": MODEL_ID,
                    "model_revision": MODEL_REVISION,
                    "budget": int(condition.rsplit("_", 1)[1]),
                }
                for condition in CONDITIONS
            }
            new_event: dict[str, object] = {
                "event": "record",
                "row_index": row_index,
                "id": identifier,
                "prompt_hash": prompt_hashes[identifier],
                "conditions": conditions,
            }
            new_event["event_digest"] = _prepare_event_digest(new_event)
            _append_jsonl(journal_path, [new_event])
            journal_events[identifier] = new_event
            event = new_event
        assert event is not None
        conditions = cast(Mapping[str, object], event["conditions"])
        prompt_hashes[identifier] = str(event["prompt_hash"])
        prepared.extend(
            dict(cast(Mapping[str, object], conditions[condition]))
            for condition in CONDITIONS
        )
    if not journal_complete:
        _append_jsonl(
            journal_path,
            [{"event": "complete", "processed_rows": len(records)}],
        )
        journal_complete = True
    runtime = _prepare_runtime_settings(args)
    output_paths = {
        condition: str(Path(args.output_root) / f"{condition}.jsonl")
        for condition in CONDITIONS
    }
    manifest = build_manifest(
        source_sha256=str(selected["source_sha256"]),
        selected_ids=[str(row["id"]) for row in records],
        selected_weights={str(row["id"]): _as_float(row["weight"]) for row in records},
        population_counts=cast(Mapping[str, int], selected["population_counts"]),
        selected_strata={
            str(row["id"]): str(row["selection_stratum"]) for row in records
        },
        selected_gold={str(row["id"]): str(row["gold"]) for row in joined},
        selected_categories={str(row["id"]): str(row["category"]) for row in joined},
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        dataset=DATASET,
        dataset_revision=DATASET_REVISION,
        runtime=runtime,
        prompt_hashes=prompt_hashes,
        tokenizer_chat_template_sha256=template_hash,
        output_paths=output_paths,
        marker_path=str(marker_path) if marker_path is not None else None,
        marker_sha256=marker_sha256,
    )
    manifest["source_evidence_sha256"] = {
        str(row["id"]): _sha256_value(_source_evidence(row)) for row in records
    }
    manifest["source_evidence_binding_sha256"] = _sha256_value(
        {
            "source_sha256": manifest["source_sha256"],
            "source_evidence_sha256": manifest["source_evidence_sha256"],
        }
    )
    manifest["source_path"] = str(source_path)
    if formal_run_root is not None:
        manifest["formal_run_root"] = str(_canonical_no_symlink(formal_run_root))
        manifest["config_digest"] = _manifest_config_digest(manifest)
    _write_json_atomic(
        root / "prepared.json",
        {
            "records": prepared,
            "manifest": manifest,
            "population_counts": selected["population_counts"],
            "shortfalls": selected["shortfalls"],
        },
    )
    _write_json_atomic(root / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "source_count": selected["source_count"],
                "selected_count": len(records),
                "shortfalls": selected["shortfalls"],
            },
            sort_keys=True,
        )
    )


def _prepare_impl(args: argparse.Namespace) -> None:
    with exclusive_mutation_lock(args.output_root):
        _prepare_impl_unlocked(args)


def prepare(args: argparse.Namespace) -> None:
    from prefix.runner import tee_stdout

    with tee_stdout(args.log_file):
        _prepare_impl(args)


def _append_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    from prefix.runner import append_jsonl

    append_jsonl(path, [dict(row) for row in rows])


def _replace_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        _append_jsonl(temporary, rows)
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _generate_impl(args: argparse.Namespace) -> None:
    from prefix.runner import get_engine, tee_stdout
    from contextlib import nullcontext

    import importlib

    root = Path(args.output_root)
    payload = cast(dict[str, object], _read_json(root / "prepared.json"))
    manifest = cast(dict[str, object], payload["manifest"])
    formal_run_root = getattr(args, "formal_run_root", None)
    if formal_run_root is None:
        raise ValueError("formal_run_root is required for OLMo provenance")
    validate_manifest(manifest, formal_run_root=formal_run_root)
    standalone_path = root / "manifest.json"
    if standalone_path.exists() and _read_json(standalone_path) != manifest:
        raise ValueError("manifest embedded and standalone copies mismatch")
    source_path = args.source_responses or manifest.get("source_path")
    if source_path is not None:
        if not isinstance(source_path, (str, Path)):
            raise ValueError("manifest source_path is invalid")
        source_path = validate_formal_source_path(source_path, formal_run_root)
        validate_source_sha256(source_path, manifest)
        if Path(source_path).exists():
            _validate_source_evidence_binding(Path(source_path), manifest)
    prepared = cast(list[dict[str, object]], payload["records"])
    runtime = cast(Mapping[str, object], manifest["runtime"])
    if args.batch_size != runtime.get(
        "batch_size", args.batch_size
    ) or args.gpu_memory_utilization != runtime.get(
        "gpu_memory_utilization", args.gpu_memory_utilization
    ):
        raise ValueError("generation runtime settings mismatch")
    if not standalone_path.exists() and not cast(
        list[object], payload.get("records", [])
    ):
        raise FileNotFoundError("production generation requires manifest.json")
    log_path = Path(args.log_file)
    with nullcontext():
        with tee_stdout(log_path):
            engine: Any = get_engine(
                MODEL_ID,
                revision=MODEL_REVISION,
                quantization=None,
                max_model_len=MAX_MODEL_LEN,
                gpu_memory_utilization=args.gpu_memory_utilization,
            )
            vllm = importlib.import_module("vllm")
            for condition in CONDITIONS:
                output_path = root / f"{condition}.jsonl"
                existing = list(_jsonl(output_path)) if output_path.exists() else []
                if formal_run_root is None:
                    raise ValueError("formal_run_root is required for OLMo provenance")
                _check_existing_rows(
                    existing, manifest, condition, formal_run_root=formal_run_root
                )
                done = {str(row["id"]) for row in existing if row.get("status") == "ok"}
                retryable = [
                    row
                    for row in existing
                    if row.get("status") == "error" and row.get("retryable") is True
                ]
                pending = [
                    row
                    for row in prepared
                    if row["condition"] == condition and str(row["id"]) not in done
                ]
                current = list(existing)
                for start in range(0, len(pending), args.batch_size):
                    batch = pending[start : start + args.batch_size]
                    requests = [row for row in batch]
                    params = vllm.SamplingParams(
                        max_tokens=int(condition.rsplit("_", 1)[1]),
                        temperature=TEMPERATURE,
                    )
                    try:
                        outputs = engine.generate(
                            [str(row["prompt"]) for row in batch], params
                        )
                        native = [
                            {
                                "token_ids": list(output.outputs[0].token_ids),
                                "finish_reason": str(output.outputs[0].finish_reason),
                                "text": str(output.outputs[0].text),
                            }
                            for output in outputs
                        ]
                        rows = build_generation_rows(requests, native)
                        if formal_run_root is None:
                            raise ValueError(
                                "formal_run_root is required for OLMo provenance"
                            )
                        _check_existing_rows(
                            rows,
                            manifest,
                            condition,
                            formal_run_root=formal_run_root,
                        )
                    except Exception as error:
                        rows = [
                            {
                                **row,
                                "status": "error",
                                "retryable": True,
                                "error": str(error),
                                "generated_token_ids": [],
                                "token_ids": [],
                                "generated_token_count": 0,
                                "extracted_answer": None,
                                "retry_history": _retry_history(
                                    existing, str(row["id"]), str(error)
                                ),
                            }
                            for row in batch
                        ]
                        for row in rows:
                            row["retry_history_sha256"] = _sha256_value(
                                row["retry_history"]
                            )
                        current = merge_generation_rows(current, rows)
                        _replace_jsonl(output_path, current)
                        print(f"{condition}: persisted {len(rows)} rows")
                        raise
                    if retryable:
                        current = merge_generation_rows(current, rows)
                        _replace_jsonl(output_path, current)
                    else:
                        _append_jsonl(output_path, rows)
                        current = [*current, *rows]
                    print(f"{condition}: persisted {len(rows)} rows")


def generate(args: argparse.Namespace) -> None:
    with exclusive_mutation_lock(args.output_root):
        _generate_impl(args)


def _analyze_impl_unlocked(args: argparse.Namespace) -> None:
    root = Path(args.output_root)
    manifest = cast(dict[str, object], _read_json(root / "manifest.json"))
    formal_run_root = getattr(args, "formal_run_root", None)
    if formal_run_root is None:
        raise ValueError("formal_run_root is required for OLMo provenance")
    validate_manifest(manifest, formal_run_root=formal_run_root)
    prepared_path = root / "prepared.json"
    prepared_payload = _read_json(prepared_path)
    if not isinstance(prepared_payload, Mapping):
        raise ValueError("prepared.json must contain an object")
    embedded_value = prepared_payload.get("manifest")
    if not isinstance(embedded_value, Mapping):
        raise ValueError("prepared.json manifest must be an object")
    if dict(embedded_value) != manifest:
        raise ValueError("manifest embedded and standalone copies mismatch")
    source_path = args.source_responses or manifest.get("source_path")
    if source_path is not None:
        if not isinstance(source_path, (str, Path)):
            raise ValueError("manifest source_path is invalid")
        source_path = validate_formal_source_path(source_path, formal_run_root)
        validate_source_sha256(source_path, manifest)
        source_path_obj = Path(source_path)
        _validate_source_evidence_binding(source_path_obj, manifest)
        for source_row in _jsonl(source_path_obj):
            if (
                ("extracted_answer" in source_row or "token_ids" in source_row)
                and "model_id" not in source_row
                and not isinstance(source_row.get("metadata"), Mapping)
            ):
                raise ValueError("source evidence schema is not formal")
    rows = [
        row for condition in CONDITIONS for row in _jsonl(root / f"{condition}.jsonl")
    ]
    baseline_rows = [row for row in rows if row.get("condition") == CONDITIONS[0]]
    if any(
        not isinstance(row.get("source_generated_token_ids"), list)
        or not row["source_generated_token_ids"]
        for row in baseline_rows
        if "source_extracted_answer" in row
    ):
        raise ValueError("baseline source token evidence coverage is incomplete")
    validate_analysis(
        rows,
        manifest,
        settings=cast(Mapping[str, object], manifest["runtime"]),
        formal_run_root=formal_run_root,
    )
    metrics = aggregate_metrics(rows)
    source_rows = [
        row
        for row in rows
        if row.get("condition") == CONDITIONS[0] and "source_extracted_answer" in row
    ]
    reproduced = sum(
        row.get("extracted_answer") == row.get("source_extracted_answer")
        for row in source_rows
    )
    source_token_rows = [
        row for row in source_rows if "source_generated_token_ids" in row
    ]
    if source_rows and (
        len(source_token_rows) != len(source_rows)
        or any(
            not isinstance(row.get("source_generated_token_ids"), list)
            or not row["source_generated_token_ids"]
            for row in source_rows
        )
    ):
        raise ValueError("baseline source token evidence coverage is incomplete")
    token_matches = sum(
        list(cast(Sequence[int], row.get("generated_token_ids", [])))
        == list(cast(Sequence[int], row["source_generated_token_ids"]))
        for row in source_token_rows
    )
    metrics["source_reproduction"] = {
        "extraction_rate": reproduced / len(source_rows) if source_rows else None,
        "numerator": reproduced,
        "denominator": len(source_rows),
        "token_sequence_rate": token_matches / len(source_token_rows)
        if source_token_rows
        else None,
        "token_sequence_numerator": token_matches,
        "token_sequence_denominator": len(source_token_rows),
    }
    metrics["baseline_reproduction"] = (
        bool(source_rows)
        and reproduced == len(source_rows)
        and (not source_token_rows or token_matches == len(source_token_rows))
    )
    metrics["prefix_gate"] = True
    metrics["classification"] = classify_root_cause(
        metrics,
        thresholds={
            "null_rescue": args.null_rescue_threshold,
            "accuracy_delta": args.accuracy_delta_threshold,
        },
    )
    metrics["provenance"] = manifest
    _write_json_atomic(root / "metrics.json", metrics)


def _analyze_impl(args: argparse.Namespace) -> None:
    with exclusive_mutation_lock(args.output_root):
        _analyze_impl_unlocked(args)


def analyze(args: argparse.Namespace) -> None:
    from prefix.runner import tee_stdout

    with tee_stdout(
        getattr(args, "log_file", Path("logs/validate_olmo_truncation.log"))
    ):
        _analyze_impl(args)


def build_sbatch_command(argv: Sequence[str]) -> str:
    return "sbatch " + shlex.join(
        [sys.executable, str(Path(__file__).resolve()), *argv]
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "generate", "analyze"))
    parser.add_argument("--source-responses", type=Path)
    parser.add_argument("--preflight-marker", type=Path)
    parser.add_argument("--formal-run-root", type=Path)
    parser.add_argument(
        "--output-root", type=Path, default=Path("results/olmo-truncation")
    )
    parser.add_argument("--cache-root", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--log-file", type=Path, default=Path("logs/validate_olmo_truncation.log")
    )
    parser.add_argument("--null-rescue-threshold", type=float, default=0.75)
    parser.add_argument("--accuracy-delta-threshold", type=float, default=0.1)
    args = parser.parse_args(argv)
    if args.formal_run_root is None:
        parser.error("all OLMo phases require --formal-run-root")
    if args.max_model_len != MAX_MODEL_LEN:
        raise ValueError("max_model_len is pinned to 8192")
    if args.phase == "prepare":
        if args.source_responses is None:
            parser.error("prepare requires --source-responses")
        prepare(args)
    elif args.phase == "generate":
        generate(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
