from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, cast

from prefix.no_steering import DATASET_SCOPES, MODEL_MATRIX, model_spec


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"missing smoke artifact: {path}") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid smoke JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"smoke JSON must be an object: {path}")
    return cast(dict[str, Any], payload)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"missing smoke responses: {path}") from None
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid response JSON at {path}:{line_number}"
            ) from error
        if not isinstance(row, dict):
            raise ValueError(f"response row must be an object at {path}:{line_number}")
        rows.append(cast(dict[str, Any], row))
    return rows


def _config_sha256(config: dict[str, Any]) -> str:
    canonical = json.dumps(
        config, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_smoke_roots(
    *, result_root: Path, checkpoint_root: Path, model_id: str
) -> None:
    spec = model_spec(model_id)
    summary = _read_json(result_root / "summary.json")
    provenance = summary.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("smoke summary is missing provenance")
    if provenance.get("model_id") != model_id:
        raise ValueError("smoke summary model mismatch")
    if provenance.get("revision") != spec.revision:
        raise ValueError("smoke summary revision mismatch")
    if summary.get("limited") is not True:
        raise ValueError("smoke summary must be limited")
    ppl = summary.get("ppl")
    expected_records = len(DATASET_SCOPES)
    if not isinstance(ppl, dict):
        raise ValueError("smoke summary lacks PPL coverage")
    selected_tokens = ppl.get("selected_token_count")
    generated_tokens = ppl.get("generated_token_count")
    ppl_value = ppl.get("ppl")
    if (
        ppl.get("covered_records") != expected_records
        or ppl.get("total_records") != expected_records
        or ppl.get("coverage_ratio") != 1.0
        or not isinstance(selected_tokens, int)
        or selected_tokens <= 0
        or generated_tokens != selected_tokens
        or not isinstance(ppl_value, (int, float))
        or not math.isfinite(float(ppl_value))
        or float(ppl_value) <= 0
    ):
        raise ValueError("smoke summary record count mismatch")

    wrapper = _read_json(checkpoint_root / "manifest.json")
    config = wrapper.get("config")
    if not isinstance(config, dict):
        raise ValueError("smoke manifest lacks config wrapper")
    config = cast(dict[str, Any], config)
    if wrapper.get("config_sha256") != _config_sha256(config):
        raise ValueError("smoke manifest config digest mismatch")
    if config.get("schema_version") != 1:
        raise ValueError("smoke manifest schema mismatch")
    if config.get("model_id") != model_id:
        raise ValueError("smoke manifest model mismatch")
    if config.get("model_slug") != spec.slug:
        raise ValueError("smoke manifest slug mismatch")
    if config.get("model_revision") != spec.revision:
        raise ValueError("smoke manifest revision mismatch")
    if config.get("limited") is not True:
        raise ValueError("smoke manifest must be limited")

    benchmark_ids = config.get("benchmark_ids")
    benchmark_manifest = config.get("benchmark_manifest")
    if not isinstance(benchmark_ids, dict) or not isinstance(benchmark_manifest, dict):
        raise ValueError("smoke manifest lacks benchmark contract")
    seen_ids: set[str] = set()
    for benchmark in DATASET_SCOPES:
        entry = benchmark_manifest.get(benchmark)
        declared = benchmark_ids.get(benchmark)
        if (
            not isinstance(entry, dict)
            or entry.get("loaded_count") != 1
            or entry.get("complete") is not False
            or entry.get("limited") is not True
            or not isinstance(declared, list)
            or len(declared) != 1
        ):
            raise ValueError(
                f"smoke manifest is not a one-row contract for {benchmark}"
            )
        path = checkpoint_root / benchmark / "responses.jsonl"
        rows = _read_rows(path)
        if len(rows) != 1:
            raise ValueError(f"smoke response count mismatch for {benchmark}")
        row = rows[0]
        identifier = row.get("id")
        if (
            identifier != declared[0]
            or not isinstance(identifier, str)
            or identifier in seen_ids
        ):
            raise ValueError(f"smoke response id mismatch for {benchmark}")
        seen_ids.add(identifier)
        if row.get("benchmark") != benchmark or row.get("model_id") != model_id:
            raise ValueError(f"smoke response identity mismatch for {benchmark}")
        if row.get("status") != "ok":
            raise ValueError(f"smoke response is not successful for {benchmark}")
        metadata = row.get("metadata")
        response_provenance = (
            metadata.get("provenance") if isinstance(metadata, dict) else None
        )
        if (
            not isinstance(response_provenance, dict)
            or response_provenance.get("model_id") != model_id
            or response_provenance.get("revision") != spec.revision
        ):
            raise ValueError(f"smoke response provenance mismatch for {benchmark}")
        token_count = row.get("generated_token_count")
        logprobs = row.get("selected_generated_token_logprobs")
        if (
            not isinstance(token_count, int)
            or token_count <= 0
            or not isinstance(logprobs, list)
        ):
            raise ValueError(f"smoke token coverage is invalid for {benchmark}")
        if len(logprobs) != token_count or any(
            not math.isfinite(float(value)) or float(value) > 0 for value in logprobs
        ):
            raise ValueError(f"smoke token coverage mismatch for {benchmark}")


def _infer_checkpoint_root(result_root: Path) -> Path:
    parts = list(result_root.resolve().parts)
    try:
        index = len(parts) - 1 - parts[::-1].index("results")
    except ValueError:
        raise ValueError(
            "--checkpoint-root is required when result root has no results component"
        ) from None
    parts[index] = "checkpoints"
    return Path(*parts)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate a no-steering smoke result")
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--model-id", choices=MODEL_MATRIX, required=True)
    parser.add_argument("--marker", type=Path, help="deprecated compatibility option")
    args = cast(argparse.Namespace, parser.parse_args(argv))
    result_root = cast(Path, args.result_root)
    model_id = cast(str, args.model_id)
    checkpoint_root = cast(Path | None, args.checkpoint_root)
    if checkpoint_root is None:
        checkpoint_root = _infer_checkpoint_root(result_root)
    _ = args.marker
    validate_smoke_roots(
        result_root=result_root,
        checkpoint_root=checkpoint_root,
        model_id=model_id,
    )
    print(f"smoke validation: OK ({model_id})")


if __name__ == "__main__":
    main()
