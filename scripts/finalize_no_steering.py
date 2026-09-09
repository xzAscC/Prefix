from __future__ import annotations

"""Validate a complete batch and deliver its single completion notification.

Notification state is deliberately at-most-once across ambiguous crashes:
``missing -> claimed -> sent`` and known send failures become ``failed``.
Only ``failed`` is retried.  A durable pre-existing ``claimed`` state means
SMTP may already have accepted the message, so it suppresses another send.
"""

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, TextIO, cast

from prefix.judge import MATH_PROMPT, SAFETY_PROMPT
from prefix.no_steering import DATASET_SCOPES, MODEL_MATRIX, model_spec
from prefix.notify import send_batch_notification


SCRIPT_DIR = Path(__file__).resolve().parent
HASH_CHUNK_SIZE = 1024 * 1024


def _load_script(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / f"{name}.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _formal_main() -> Any:
    return _load_script("validate_no_steering_formal").main


def _runner() -> Any:
    return _load_script("run_no_steering")


def _expected_scoring_contract() -> dict[str, object]:
    return {
        "judge_model": "gemini-3.5-flash-lite",
        "rubric_version": "gemini-rubric-v1",
        "prompt_sha256": {
            "safety": hashlib.sha256(SAFETY_PROMPT.encode()).hexdigest(),
            "math": hashlib.sha256(MATH_PROMPT.encode()).hexdigest(),
        },
        "response_schema": "no-steering-response-v1",
        "settings": {"temperature": 0.0, "max_output_tokens": 8192},
    }


def _generation_paths(root: Path) -> tuple[Path, Path]:
    results = root / "results"
    checkpoints = root / "checkpoints"
    if not results.is_dir() or not checkpoints.is_dir():
        raise ValueError(
            "generation root must contain results/ and checkpoints/ directories"
        )
    return results, checkpoints


def validate_generation(generation_root: str | Path) -> dict[str, int]:
    results_root, checkpoints_root = _generation_paths(Path(generation_root))
    total = 0
    for model_id in MODEL_MATRIX:
        formal_root = results_root / model_spec(model_id).slug
        checkpoint_root = checkpoints_root / model_spec(model_id).slug
        _formal_main()(
            [
                "--result-root",
                str(formal_root),
                "--checkpoint-root",
                str(checkpoint_root),
                "--model-id",
                model_id,
            ]
        )
        total += sum(
            int(cast(int, scope["count"])) for scope in DATASET_SCOPES.values()
        )
    return {"models": len(MODEL_MATRIX), "records": total}


def _iter_jsonl(path: Path) -> Iterator[dict[str, object]]:
    try:
        handle = path.open(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"missing score output: {path}") from None
    with handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid score JSON at {path}:{line_number}"
                ) from error
            if not isinstance(row, dict):
                raise ValueError(f"score row must be an object at {path}:{line_number}")
            yield cast(dict[str, object], row)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return list(_iter_jsonl(path))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def validate_scoring(
    scoring_root: str | Path,
    *,
    generation_checkpoint_root: str | Path | None = None,
) -> dict[str, object]:
    runner = _runner()
    root = Path(scoring_root)
    counts = {benchmark: 0 for benchmark in DATASET_SCOPES}
    for model_id in MODEL_MATRIX:
        model_root = root / model_spec(model_id).slug
        manifest_path = model_root / "scoring_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise ValueError(f"missing scoring manifest for {model_id}") from None
        if not isinstance(manifest, dict):
            raise ValueError(f"invalid scoring manifest for {model_id}")
        expected_contract = _expected_scoring_contract()
        if manifest.get("judge_model") != expected_contract["judge_model"]:
            raise ValueError(f"scoring manifest judge model mismatch for {model_id}")
        if manifest.get("model_id") != model_id:
            raise ValueError(f"scoring manifest model mismatch for {model_id}")
        if manifest.get("model_slug") != model_spec(model_id).slug:
            raise ValueError(f"scoring manifest slug mismatch for {model_id}")
        if manifest.get("schema_version") != 1:
            raise ValueError(f"scoring manifest schema mismatch for {model_id}")
        if manifest.get("rubric_version") != expected_contract["rubric_version"]:
            raise ValueError(f"scoring manifest rubric mismatch for {model_id}")
        if manifest.get("prompt_sha256") != expected_contract["prompt_sha256"]:
            raise ValueError(f"scoring manifest prompt mismatch for {model_id}")
        if manifest.get("response_schema") != expected_contract["response_schema"]:
            raise ValueError(
                f"scoring manifest response schema mismatch for {model_id}"
            )
        if manifest.get("settings") != expected_contract["settings"]:
            raise ValueError(f"scoring manifest settings mismatch for {model_id}")
        response_files = manifest.get("response_files")
        if not isinstance(response_files, dict):
            raise ValueError(f"scoring manifest lacks response bindings for {model_id}")
        response_root = manifest.get("response_root")
        if not isinstance(response_root, str):
            raise ValueError(f"scoring manifest lacks response root for {model_id}")
        expected_response_root = Path(response_root).resolve()
        if generation_checkpoint_root is not None:
            selected_root = Path(generation_checkpoint_root).resolve()
            if expected_response_root != selected_root:
                raise ValueError(
                    f"scoring manifest response root mismatch for {model_id}: "
                    f"expected {selected_root}, found {expected_response_root}"
                )
            expected_response_root = selected_root
        rows: list[dict[str, object]] = []
        for benchmark in DATASET_SCOPES:
            path = root / model_spec(model_id).slug / benchmark / "scores.jsonl"
            benchmark_rows = _read_jsonl(path)
            identifiers: set[str] = set()
            for row in benchmark_rows:
                if row.get("benchmark") != benchmark:
                    raise ValueError(f"score benchmark mismatch in {path}")
                identifier = row.get("id")
                if not isinstance(identifier, str) or not identifier:
                    raise ValueError(f"score row is missing an id in {path}")
                if identifier in identifiers:
                    raise ValueError(f"duplicate score id in {path}: {identifier}")
                identifiers.add(identifier)
                if row.get("status") != "ok":
                    raise ValueError(
                        f"unexpected scoring status in {path}: {row.get('status')!r}"
                    )
                required = {
                    "mmlu_pro": "correct",
                    "math500": "answer_correct",
                    "harmbench": "gemini_label",
                }[benchmark]
                if not isinstance(
                    row.get(required), bool if required != "gemini_label" else str
                ):
                    raise ValueError(f"score payload is incomplete in {path}")
            bound = response_files.get(benchmark)
            if (
                not isinstance(bound, dict)
                or not isinstance(bound.get("ids"), list)
                or not isinstance(bound.get("path"), str)
                or not isinstance(bound.get("content_sha256"), str)
            ):
                raise ValueError(f"score manifest lacks ids for {benchmark}")
            response_path = Path(bound["path"])
            expected_path = (
                expected_response_root
                / model_spec(model_id).slug
                / benchmark
                / "responses.jsonl"
            )
            if response_path.resolve() != expected_path:
                raise ValueError(f"response path is not canonical for {benchmark}")
            if not response_path.exists():
                raise ValueError(f"missing bound response file for {benchmark}")
            digest = _sha256_file(response_path)
            if digest != bound["content_sha256"]:
                raise ValueError(f"response content digest mismatch for {benchmark}")
            response_ids = {str(row.get("id")) for row in _iter_jsonl(response_path)}
            if response_ids != {str(value) for value in bound["ids"]}:
                raise ValueError(f"response ids do not match manifest for {benchmark}")
            if identifiers != {str(value) for value in bound["ids"]}:
                raise ValueError(f"score ids do not match response manifest in {path}")
            rows.extend(benchmark_rows)
            counts[benchmark] += len(benchmark_rows)
        runner.validate_score_completeness(rows, limited=False)
    expected = {
        name: int(cast(int, scope["count"])) * len(MODEL_MATRIX)
        for name, scope in DATASET_SCOPES.items()
    }
    if counts != expected:
        raise ValueError(f"scoring counts {counts} do not match expected {expected}")
    return {"models": len(MODEL_MATRIX), "counts": counts}


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


def _atomic_write(path: Path, payload: Mapping[str, object]) -> None:
    _reject_persistence_symlinks(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _claim(path: Path, payload: Mapping[str, object]) -> bool:
    _reject_persistence_symlinks(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)
    return True


def _claim_failed(path: Path, payload: Mapping[str, object]) -> bool:
    _reject_persistence_symlinks(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    _reject_persistence_symlinks(lock_path)
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            if _existing_status(path) != "failed":
                return False
            _atomic_write(path, payload)
            return True
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _reject_persistence_symlinks(path: Path) -> None:
    current = Path(path.anchor) if path.is_absolute() else Path.cwd()
    components = path.parts[1:] if path.is_absolute() else path.parts
    for component in components:
        current /= component
        if current.is_symlink():
            raise ValueError(
                f"refusing symlinked persistence path component: {current}"
            )


def _existing_status(path: Path) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return "claimed"
    return (
        str(payload.get("status", "claimed"))
        if isinstance(payload, dict)
        else "claimed"
    )


def finalize(
    *,
    generation_root: str | Path,
    scoring_root: str | Path,
    state_path: str | Path,
    batch_label: str | None = None,
    log_path: str | Path | None = None,
) -> str:
    generation = validate_generation(generation_root)
    scoring = validate_scoring(
        scoring_root,
        generation_checkpoint_root=Path(generation_root) / "checkpoints",
    )
    label = batch_label or "default"
    task = f"no-steering {label}"
    details = (
        f"models={generation['models']}; generation_records={generation['records']}; "
        f"scores="
        + ", ".join(
            f"{name}={count}"
            for name, count in cast(Mapping[str, int], scoring["counts"]).items()
        )
    )
    state = Path(state_path)
    claim = {
        "event": "completed",
        "status": "claimed",
        "notification_attempted": True,
        "task": task,
        "details": details,
    }
    if not _claim(state, claim):
        status = _existing_status(state)
        if status in {"sent", "claimed"}:
            print(f"finalization: already {status}")
            return status
        if status != "failed":
            print(f"finalization: already {status}")
            return status
        if not _claim_failed(state, claim):
            status = _existing_status(state)
            print(f"finalization: already {status}")
            return status
        print("finalization: retrying failed notification")

    print(f"finalization: claimed ({details})")
    try:
        delivered = send_batch_notification(task, "completed", details=details)
    except Exception as error:
        print(f"finalization: notification failed: {error}", file=sys.stderr)
        _atomic_write(state, {**claim, "status": "failed", "error": str(error)})
        return "failed"
    if not delivered:
        print("finalization: notification was not delivered", file=sys.stderr)
        _atomic_write(state, {**claim, "status": "failed"})
        return "failed"
    _atomic_write(state, {**claim, "status": "sent"})
    print("finalization: notification attempted")
    return "sent"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Finalize a complete no-steering batch"
    )
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--scoring-root", type=Path, required=True)
    parser.add_argument("--state-path", type=Path, required=True)
    parser.add_argument("--batch-label")
    parser.add_argument(
        "--log-path", type=Path, default=Path("logs/finalize_no_steering.log")
    )
    args = parser.parse_args(argv)
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    with args.log_path.open("a", encoding="utf-8") as log:
        original = sys.stdout
        original_error = sys.stderr
        sys.stdout = cast(TextIO, cast(object, _Tee((original, log))))
        sys.stderr = cast(TextIO, cast(object, _Tee((original_error, log))))
        try:
            print("finalization: validating artifacts")
            result = finalize(
                generation_root=args.generation_root,
                scoring_root=args.scoring_root,
                state_path=args.state_path,
                batch_label=args.batch_label,
                log_path=args.log_path,
            )
            if result == "failed":
                raise SystemExit(1)
        finally:
            sys.stdout = original
            sys.stderr = original_error


if __name__ == "__main__":
    main()
