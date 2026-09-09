from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from prefix.judge import SAFETY_PROMPT, GeminiJudge, JudgeBlocked
from prefix.runner import read_json, read_jsonl, tee_stdout, write_json_atomic


DEFAULT_MODELS = (
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
)
PROBE_SETTINGS: dict[str, Any] = {
    "temperature": 0.0,
    "max_output_tokens": 8192,
    "safety_settings": "BLOCK_NONE",
}
_RETRYABLE_RECORD_STATUSES = {"error", "pending"}
_BLOCK_REASON_RE = re.compile(r"provider:\s*(\S+)", re.IGNORECASE)


def resolve_response_path(response_path: str | Path) -> Path:
    path = Path(response_path)
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"response JSONL/root does not exist: {path}")
    direct = path / "responses.jsonl"
    if direct.is_file():
        return direct
    candidates = sorted(path.glob("**/harmbench/responses.jsonl"))
    if len(candidates) != 1:
        raise ValueError(
            f"response root must contain exactly one harmbench/responses.jsonl; "
            f"found {len(candidates)}"
        )
    return candidates[0]


def _pair_id(row: Mapping[str, Any], index: int) -> str:
    supplied = row.get("id", row.get("pair_id"))
    if supplied is not None and str(supplied):
        return str(supplied)
    digest = hashlib.sha256(
        json.dumps(dict(row), sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    return f"harmbench-{index}-{digest}"


def _response_text(row: Mapping[str, Any]) -> str:
    for key in ("response", "text"):
        value = row.get(key)
        if value is not None:
            return str(value)
    raw = row.get("raw_response")
    if isinstance(raw, Mapping) and raw.get("text") is not None:
        return str(raw["text"])
    raise ValueError("response row has no precomputed response text")


def load_response_rows(response_path: str | Path) -> list[dict[str, Any]]:
    rows = read_jsonl(resolve_response_path(response_path))
    normalized: list[dict[str, Any]] = []
    ids: set[str] = set()
    for index, row in enumerate(rows):
        behavior = row.get("behavior")
        if behavior is None and isinstance(row.get("metadata"), Mapping):
            behavior = row["metadata"].get("behavior")
        if behavior is None:
            raise ValueError(f"response row {index} has no HarmBench behavior")
        pair_id = _pair_id(row, index)
        if pair_id in ids:
            raise ValueError(f"duplicate HarmBench pair id: {pair_id}")
        ids.add(pair_id)
        normalized.append(
            {
                "pair_id": pair_id,
                "behavior": str(behavior),
                "response": _response_text(row),
            }
        )
    return normalized


def probe_config(
    *,
    models: Sequence[str],
    pairs: Sequence[Mapping[str, Any]],
    rubric: str = SAFETY_PROMPT,
    settings: Mapping[str, Any] = PROBE_SETTINGS,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "models": [str(model) for model in models],
        "pairs": [dict(pair) for pair in pairs],
        "rubric": rubric,
        "settings": dict(settings),
    }


def probe_config_digest(
    *,
    models: Sequence[str],
    pairs: Sequence[Mapping[str, Any]],
    rubric: str = SAFETY_PROMPT,
    settings: Mapping[str, Any] = PROBE_SETTINGS,
) -> str:
    canonical = json.dumps(
        probe_config(models=models, pairs=pairs, rubric=rubric, settings=settings),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _checkpoint_records(path: str | Path) -> list[dict[str, Any]]:
    value = read_json(path, default={})
    if value is None:
        return []
    if not isinstance(value, Mapping):
        raise ValueError("probe checkpoint must contain a records list")
    checkpoint_records = value.get("records", [])
    if not isinstance(checkpoint_records, list):
        raise ValueError("probe checkpoint must contain a records list")
    return [dict(row) for row in checkpoint_records if isinstance(row, Mapping)]


def _blocked_reason(error: BaseException) -> str | None:
    match = _BLOCK_REASON_RE.search(str(error))
    return match.group(1) if match else None


def run_probe(
    *,
    rows: Sequence[Mapping[str, Any]],
    models: Sequence[str],
    output_path: str | Path,
    judge_factory: Callable[[str], Any] = lambda model: GeminiJudge(model=model),
    sample_limit: int | None = None,
    rubric: str = SAFETY_PROMPT,
    settings: Mapping[str, Any] = PROBE_SETTINGS,
) -> list[dict[str, Any]]:
    if sample_limit is not None and sample_limit < 0:
        raise ValueError("sample_limit must not be negative")
    selected = list(rows[:sample_limit]) if sample_limit is not None else list(rows)
    config = probe_config(
        models=models, pairs=selected, rubric=rubric, settings=settings
    )
    config_digest = probe_config_digest(
        models=models, pairs=selected, rubric=rubric, settings=settings
    )
    state = read_json(output_path, default={})
    if state in (None, {}):
        records: list[dict[str, Any]] = []
    elif not isinstance(state, Mapping):
        raise ValueError("probe checkpoint must be an object")
    else:
        if state.get("config") != config or state.get("config_sha256") != config_digest:
            raise ValueError("probe checkpoint config digest mismatch")
        records = _checkpoint_records(output_path)
    expected_keys = {
        (str(model), str(row["pair_id"])) for model in models for row in selected
    }
    records_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        key = (str(record.get("model")), str(record.get("pair_id")))
        if key not in expected_keys or key in records_by_key:
            raise ValueError("probe checkpoint contains duplicate or stale records")
        records_by_key[key] = record
    for model in models:
        judge = judge_factory(model)
        for row in selected:
            pair_id = str(row["pair_id"])
            key = (model, pair_id)
            existing = records_by_key.get(key)
            if (
                existing is not None
                and existing.get("status") not in _RETRYABLE_RECORD_STATUSES
            ):
                continue
            if existing is not None:
                del records_by_key[key]
            started = time.monotonic()
            http_status: int | None = None
            provider_outcome = "not_called"

            def observe_transport(
                url: str, headers: dict[str, Any], body: dict[str, Any]
            ) -> tuple[int, dict[str, Any]]:
                nonlocal http_status, provider_outcome
                status, payload = judge._default_transport(url, headers, body)
                http_status = status
                provider_outcome = "success" if 200 <= status < 300 else "http_error"
                return status, payload

            if hasattr(judge, "_transport"):
                judge._transport = observe_transport
            result: dict[str, Any] = {
                "model": model,
                "pair_id": pair_id,
                "http_status": None,
                "provider_outcome": provider_outcome,
                "prompt_block_reason": None,
                "prompt_status": "not_called",
                "status": "pending",
                "label": None,
                "latency_seconds": None,
                "error": None,
            }
            try:
                safe = bool(
                    judge.judge_safety(str(row["behavior"]), str(row["response"]))
                )
                result.update(
                    {
                        "label": "SAFE" if safe else "UNSAFE",
                        "prompt_status": "accepted",
                        "status": "ok",
                        "provider_outcome": provider_outcome,
                    }
                )
            except JudgeBlocked as error:
                result.update(
                    {
                        "prompt_status": "blocked",
                        "status": "blocked",
                        "provider_outcome": "blocked",
                        "prompt_block_reason": _blocked_reason(error),
                        "error": str(error),
                    }
                )
            except Exception as error:
                result.update(
                    {
                        "prompt_status": "error",
                        "status": "error",
                        "provider_outcome": provider_outcome,
                        "error": str(error),
                    }
                )
            result["http_status"] = http_status
            result["latency_seconds"] = time.monotonic() - started
            records_by_key[key] = result
            records = [
                records_by_key[(candidate_model, str(candidate["pair_id"]))]
                for candidate_model in models
                for candidate in selected
                if (candidate_model, str(candidate["pair_id"])) in records_by_key
            ]
            write_json_atomic(
                output_path,
                {
                    "config": config,
                    "config_sha256": config_digest,
                    "records": records,
                },
            )
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return records


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Probe approved Vertex Gemini safety judges"
    )
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--responses", "--response-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    models = tuple(args.models or DEFAULT_MODELS)
    with tee_stdout(args.log):
        rows = load_response_rows(args.responses)
        print(f"Loaded {len(rows)} precomputed HarmBench response pairs")
        run_probe(
            rows=rows,
            models=models,
            output_path=args.output,
            sample_limit=args.limit,
        )


if __name__ == "__main__":
    main()
