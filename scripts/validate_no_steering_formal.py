from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import cast

from prefix.no_steering import DATASET_SCOPES, MODEL_MATRIX, model_spec


PPL_FIELDS = (
    "ppl",
    "selected_token_count",
    "generated_token_count",
    "covered_records",
    "total_records",
    "coverage_ratio",
)


def _read_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"response row must be an object at {path}")
        rows.append(cast(dict[str, object], row))
    return rows


def _config_sha256(config: dict[str, object]) -> str:
    encoded = json.dumps(
        config, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _expected_sampling(revision: str) -> dict[str, object]:
    return {
        "max_tokens": 1024,
        "temperature": 0.0,
        "logprobs": 1,
        "revision": revision,
        "quantization": None,
        "max_model_len": 8192,
        "gpu_memory_utilization": 0.9,
    }


def _root_matches(
    declared: object,
    supplied: Path,
    model_slug: str,
    container_name: str,
) -> bool:
    declared_path = Path(str(declared)).resolve()
    supplied_path = supplied.resolve()
    if declared_path == supplied_path:
        return True
    return (
        declared_path.name == container_name
        and supplied_path.parent.name == container_name
        and supplied_path.name == model_slug
    )


def _validate_response_root(
    root: Path, model_id: str, expected: dict[str, int], revision: str
) -> tuple[dict[str, list[dict[str, object]]], set[str]]:
    by_benchmark: dict[str, list[dict[str, object]]] = {}
    identifiers: set[str] = set()
    for benchmark, count in expected.items():
        path = root / benchmark / "responses.jsonl"
        rows = _read_rows(path)
        if len(rows) != count:
            raise ValueError(f"formal response coverage is incomplete for {benchmark}")
        by_benchmark[benchmark] = rows
        for row in rows:
            if row.get("status") != "ok" or row.get("model_id") != model_id:
                raise ValueError("formal response contains an invalid status or model")
            if row.get("benchmark") != benchmark:
                raise ValueError("formal response benchmark mismatch")
            identifier = row.get("id")
            if (
                not isinstance(identifier, str)
                or not identifier
                or identifier in identifiers
            ):
                raise ValueError("formal responses contain missing or duplicate ids")
            identifiers.add(identifier)
            metadata = row.get("metadata")
            provenance = (
                metadata.get("provenance") if isinstance(metadata, dict) else None
            )
            if (
                not isinstance(provenance, dict)
                or provenance.get("model_id") != model_id
            ):
                raise ValueError("formal response provenance model mismatch")
            if provenance.get("revision") != revision:
                raise ValueError("formal response provenance revision mismatch")
            generated = row.get("generated_token_count")
            logprobs = row.get("selected_generated_token_logprobs")
            if (
                isinstance(generated, bool)
                or not isinstance(generated, int)
                or generated <= 0
                or not isinstance(logprobs, list)
                or len(logprobs) != generated
            ):
                raise ValueError("formal response logprob coverage is invalid")
            for value in logprobs:
                if isinstance(value, bool):
                    raise ValueError("formal response logprob coverage is invalid")
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    raise ValueError(
                        "formal response logprob coverage is invalid"
                    ) from None
                if not math.isfinite(number) or number > 0:
                    raise ValueError("formal response logprob coverage is invalid")
    return by_benchmark, identifiers


def _number_matches(actual: object, expected: float, field: str) -> None:
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        raise ValueError(f"formal PPL field {field} is invalid")
    value = float(actual)
    if not math.isfinite(value) or not math.isclose(
        value, expected, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError(f"formal PPL field {field} mismatch")


def _validate_ppl_payload(payload: object, expected: dict[str, float | int]) -> None:
    if not isinstance(payload, dict):
        raise ValueError("formal PPL payload is missing")
    for field in PPL_FIELDS:
        if field not in payload:
            raise ValueError(f"formal PPL field {field} is missing")
        expected_value = expected[field]
        actual = payload[field]
        if isinstance(expected_value, float):
            _number_matches(actual, expected_value, field)
        elif actual != expected_value or isinstance(actual, bool):
            raise ValueError(f"formal PPL field {field} mismatch")


def _recompute_ppl(
    rows: list[dict[str, object]], total_records: int
) -> dict[str, float | int]:
    selected_token_count = 0
    generated_token_count = 0
    covered_records = 0
    total_nll = 0.0
    for row in rows:
        values = cast(list[object], row["selected_generated_token_logprobs"])
        generated_token_count += cast(int, row["generated_token_count"])
        selected_token_count += len(values)
        if values:
            covered_records += 1
        total_nll -= sum(float(cast(float | int | str, value)) for value in values)
    if selected_token_count == 0:
        raise ValueError("formal PPL has zero selected tokens")
    ppl = math.exp(total_nll / selected_token_count)
    return {
        "ppl": ppl,
        "selected_token_count": selected_token_count,
        "generated_token_count": generated_token_count,
        "covered_records": covered_records,
        "total_records": total_records,
        "coverage_ratio": selected_token_count / generated_token_count,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate formal no-steering outputs")
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--model-id", choices=MODEL_MATRIX, required=True)
    args = cast(argparse.Namespace, parser.parse_args(argv))
    result_root = cast(Path, args.result_root)
    checkpoint_root = cast(Path, args.checkpoint_root)
    model_id = cast(str, args.model_id)
    spec = model_spec(model_id)
    if checkpoint_root.name != spec.slug or result_root.name != spec.slug:
        raise ValueError("formal roots must be model-scoped")

    manifest_path = checkpoint_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("formal checkpoint manifest must be an object")
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise ValueError("formal checkpoint manifest lacks config wrapper")
    if manifest.get("config_sha256") != _config_sha256(cast(dict[str, object], config)):
        raise ValueError("formal checkpoint manifest config digest mismatch")
    config = cast(dict[str, object], config)
    if config.get("schema_version") != 1:
        raise ValueError("formal checkpoint manifest schema mismatch")
    if config.get("model_id") != model_id:
        raise ValueError("formal checkpoint manifest model mismatch")
    if config.get("model_slug") != spec.slug:
        raise ValueError("formal checkpoint manifest slug mismatch")
    if config.get("model_revision") != spec.revision:
        raise ValueError("formal checkpoint manifest revision mismatch")
    if config.get("steering") is not False:
        raise ValueError("formal checkpoint manifest steering mismatch")
    if config.get("quantization") is not False:
        raise ValueError("formal checkpoint manifest quantization mismatch")
    if config.get("prompt_template_version") != "chat-v1":
        raise ValueError("formal checkpoint manifest prompt mismatch")
    if config.get("sampling") != _expected_sampling(spec.revision):
        raise ValueError("formal checkpoint manifest sampling settings mismatch")
    output_roots = config.get("output_roots")
    if not isinstance(output_roots, dict):
        raise ValueError("formal checkpoint manifest lacks output roots")
    if not _root_matches(
        output_roots.get("checkpoint"), checkpoint_root, spec.slug, "checkpoints"
    ):
        raise ValueError("formal checkpoint manifest checkpoint root mismatch")
    if not _root_matches(output_roots.get("output"), result_root, spec.slug, "results"):
        raise ValueError("formal checkpoint manifest result root mismatch")

    expected = {
        name: int(cast(int, scope["count"])) for name, scope in DATASET_SCOPES.items()
    }
    benchmark_manifest = config.get("benchmark_manifest")
    if not isinstance(benchmark_manifest, dict):
        raise ValueError("formal checkpoint manifest lacks benchmark coverage")
    for benchmark, scope in DATASET_SCOPES.items():
        entry = benchmark_manifest.get(benchmark)
        if (
            not isinstance(entry, dict)
            or entry.get("split") != scope["split"]
            or entry.get("expected_count") != expected[benchmark]
            or entry.get("loaded_count") != expected[benchmark]
            or entry.get("complete") is not True
            or entry.get("limited") is not False
        ):
            raise ValueError(
                f"formal checkpoint dataset manifest mismatch for {benchmark}"
            )
    benchmark_ids = config.get("benchmark_ids")
    if not isinstance(benchmark_ids, dict):
        raise ValueError("formal checkpoint manifest lacks benchmark ids")

    rows_by_benchmark, identifiers = _validate_response_root(
        checkpoint_root, model_id, expected, spec.revision
    )
    all_rows = [row for rows in rows_by_benchmark.values() for row in rows]
    for benchmark, rows in rows_by_benchmark.items():
        declared_ids = benchmark_ids.get(benchmark)
        actual_ids = [cast(str, row["id"]) for row in rows]
        if declared_ids != actual_ids:
            raise ValueError(f"formal checkpoint ids mismatch for {benchmark}")
    if len(identifiers) != sum(expected.values()):
        raise ValueError("formal checkpoint dataset coverage mismatch")

    try:
        summary = json.loads((result_root / "summary.json").read_text(encoding="utf-8"))
        conditional = json.loads(
            (result_root / "conditional_ppl.json").read_text(encoding="utf-8")
        )
    except FileNotFoundError:
        raise ValueError(
            "formal result is missing summary or conditional_ppl"
        ) from None
    if not isinstance(summary, dict) or not isinstance(summary.get("provenance"), dict):
        raise ValueError("formal result summary lacks provenance")
    provenance = cast(dict[str, object], summary["provenance"])
    if (
        provenance.get("model_id") != model_id
        or provenance.get("revision") != spec.revision
    ):
        raise ValueError("formal result summary model mismatch")
    expected_ppl = _recompute_ppl(all_rows, sum(expected.values()))
    summary_ppl = summary.get("ppl")
    _validate_ppl_payload(summary_ppl, expected_ppl)
    _validate_ppl_payload(conditional, expected_ppl)
    for field in PPL_FIELDS:
        if isinstance(summary_ppl, dict) and isinstance(conditional, dict):
            if field in summary_ppl and field in conditional:
                if field == "ppl":
                    _number_matches(
                        conditional[field], float(summary_ppl[field]), field
                    )
                elif conditional[field] != summary_ppl[field]:
                    raise ValueError("summary and conditional_ppl disagree")
    print(f"formal validation: OK ({model_id})")


if __name__ == "__main__":
    main()
