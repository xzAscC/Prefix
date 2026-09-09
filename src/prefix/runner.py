from __future__ import annotations

import json
import os
import re
import stat
import sys
import tempfile
from contextlib import contextmanager
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from . import vllm_steering
from .steering import SteeringSchedule, dim_direction, mean_hidden_norm
from .vllm_steering import (
    CaptureSink,
    attach_capture,
    attach_steering,
    make_decode_index_resolver,
)


_engine_singleton: Any | None = None
_engine_config: tuple[str, tuple[tuple[str, Any], ...]] | None = None
_ANSWER_RE = re.compile(r"answer\s+is\s*\(?([A-J])\)?\b", re.IGNORECASE)
JSONL_TAIL_BYTES = 64 * 1024


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    config_path = Path(path)
    if config_path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(f"unknown configuration extension: {config_path.suffix}")
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("configuration must contain a mapping")
    return value


class _Tee:
    def __init__(self, original: Any, log_file: Any) -> None:
        self.original = original
        self.log_file = log_file

    def write(self, value: str) -> int:
        self.original.write(value)
        self.log_file.write(value)
        self.flush()
        return len(value)

    def flush(self) -> None:
        self.original.flush()
        self.log_file.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.original, name)


@contextmanager
def tee_stdout(log_path: str | Path) -> Iterator[None]:
    """Duplicate stdout to an appended, line-buffered log file."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", buffering=1) as log_file:
        original = sys.stdout
        sys.stdout = _Tee(original, log_file)  # type: ignore[assignment]
        try:
            yield
        finally:
            sys.stdout = original
            log_file.flush()


def append_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    """Append records and force them to stable storage."""
    path = Path(path)
    _reject_symlink_components(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_separator = False
    with path.open("a+b") as output:
        output.seek(0, os.SEEK_END)
        file_size = output.tell()
        tail_size = min(file_size, JSONL_TAIL_BYTES)
        if tail_size:
            output.seek(file_size - tail_size)
            data = output.read(tail_size)
            last_newline = data.rfind(b"\n")
            if last_newline < 0 and file_size > JSONL_TAIL_BYTES:
                raise RuntimeError(
                    "cannot repair JSONL final record within bounded tail window"
                )
            trailing = data[last_newline + 1 :]
            if trailing.strip():
                try:
                    json.loads(trailing)
                except json.JSONDecodeError:
                    output.truncate(file_size - tail_size + max(0, last_newline + 1))
                else:
                    needs_separator = True
            elif last_newline >= 0 and last_newline + 1 < len(data):
                output.truncate(file_size - tail_size + last_newline + 1)
        output.seek(0, os.SEEK_END)
        if needs_separator:
            output.write(b"\n")
        for record in records:
            output.write(
                (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
            )
        output.flush()
        os.fsync(output.fileno())
    _fsync_directory(path.parent)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as input_file:
        lines = input_file.readlines()
    records: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break
            raise
        if not isinstance(record, dict):
            raise ValueError("JSONL records must be objects")
        records.append(record)
    return records


def completed_ids(path: str | Path, id_key: str = "id") -> set[str]:
    return {str(record[id_key]) for record in read_jsonl(path) if id_key in record}


def missing_ids(path: str | Path, expected: list[str], id_key: str = "id") -> list[str]:
    completed = completed_ids(path, id_key)
    return sorted(set(expected) - completed)


def require_complete(
    path: str | Path, expected: list[str], id_key: str = "id", label: str = ""
) -> None:
    missing = missing_ids(path, expected, id_key)
    if missing:
        prefix = f"{label}: " if label else ""
        raise RuntimeError(f"{prefix}missing ids: {missing[:10]}")


def write_json_atomic(path: str | Path, obj) -> None:
    path = Path(path)
    _reject_symlink_components(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(obj, output, ensure_ascii=False)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _reject_symlink_components(path: Path) -> None:
    current = Path(path.anchor) if path.is_absolute() else Path.cwd()
    for component in path.parts[1:] if path.is_absolute() else path.parts:
        current /= component
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise ValueError(
                f"refusing symlinked persistence path component: {current}"
            )


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_json(path: str | Path, default=None):
    try:
        with Path(path).open(encoding="utf-8") as input_file:
            return json.load(input_file)
    except FileNotFoundError:
        return default


def _config_sha256(config: dict[str, Any]) -> str:
    import hashlib

    canonical = json.dumps(
        config, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_manifest(path: str | Path, config: dict[str, Any]) -> None:
    write_json_atomic(path, {"config_sha256": _config_sha256(config), "config": config})


def _checkpoint_has_rows(paths: Iterable[str | Path]) -> bool:
    return any(Path(path).exists() and Path(path).stat().st_size > 0 for path in paths)


def verify_manifest(
    path: str | Path,
    config: dict[str, Any],
    *,
    checkpoint_paths: Iterable[str | Path] = (),
) -> None:
    manifest = read_json(path)
    if manifest is None:
        if _checkpoint_has_rows(checkpoint_paths):
            raise RuntimeError(
                "missing checkpoint manifest; refusing to resume nonempty checkpoint"
            )
        return
    if manifest.get("config_sha256") != _config_sha256(config):
        raise RuntimeError(
            "stale checkpoint manifest; move stale checkpoints before continuing"
        )


def prepare_checkpoint_manifest(
    path: str | Path,
    config: dict[str, Any],
    *,
    checkpoint_paths: Iterable[str | Path] = (),
) -> None:
    """Bind a checkpoint's rows to its configuration before generation starts.

    A missing manifest is valid only when every checkpoint row file is empty or
    absent.  This lets a clean run create its manifest atomically while making
    an otherwise ambiguous resume fail closed.
    """
    checkpoint_paths = tuple(checkpoint_paths)
    if read_json(path) is None:
        if _checkpoint_has_rows(checkpoint_paths):
            raise RuntimeError(
                "missing checkpoint manifest; refusing to resume nonempty checkpoint"
            )
        write_manifest(path, config)
        return
    verify_manifest(path, config, checkpoint_paths=checkpoint_paths)


def condition_id(*parts: Any) -> str:
    return "/".join(
        repr(part) if isinstance(part, float) else str(part) for part in parts
    )


def chat_prompt(tokenizer, user_text: str, enable_thinking: bool) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=enable_thinking,
    )


def mmlu_prompt(question: str, options: list[str]) -> str:
    labelled = "\n".join(
        f"{chr(ord('A') + index)}. {option}" for index, option in enumerate(options)
    )
    return (
        f"{question}\n\n{labelled}\n\n"
        "Reason step by step, and make the FINAL line exactly `The answer is (X)` "
        "where X is the letter."
    )


def parse_answer_letter(text: str) -> str | None:
    matches = list(_ANSWER_RE.finditer(text))
    if matches:
        return matches[-1].group(1).upper()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        match = re.fullmatch(r"\(?([A-J])\)?", lines[-1], re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def get_engine(model_id: str, **llm_kwargs):
    global _engine_config, _engine_singleton
    if _engine_singleton is None:
        _engine_singleton = vllm_steering.make_llm(model_id, **llm_kwargs)
        _engine_config = (model_id, tuple(sorted(llm_kwargs.items())))
    else:
        assert _engine_config is not None
        initialized_model_id, initialized_kwargs = _engine_config
        if model_id != initialized_model_id:
            raise RuntimeError(
                f"conflicting model_id: initialized with {initialized_model_id!r}, "
                f"requested {model_id!r}"
            )
        requested_kwargs = tuple(sorted(llm_kwargs.items()))
        if requested_kwargs != initialized_kwargs:
            raise RuntimeError(
                f"conflicting llm_kwargs: initialized with {dict(initialized_kwargs)!r}, "
                f"requested {llm_kwargs!r}"
            )
    return _engine_singleton


def _sampling_params(*, max_tokens: int, temperature: float):
    from vllm import SamplingParams  # type: ignore[import-not-found]

    return SamplingParams(max_tokens=max_tokens, temperature=temperature)


@dataclass(frozen=True)
class DirectionRecord:
    direction: torch.Tensor
    mean_norm: float


def capture_prompt_hiddens(
    llm, prompts: list[str], layers: list[int], batch_prompts: int = 64
) -> dict[int, torch.Tensor]:
    """Capture aligned last-prompt-token residuals in one greedy pass."""
    if batch_prompts < 1:
        raise ValueError("batch_prompts must be at least 1")
    sinks = {layer: CaptureSink() for layer in layers}
    detachers = []
    try:
        for layer, sink in sinks.items():
            detachers.append(attach_capture(llm, layer, sink))
        outputs = []
        params = _sampling_params(max_tokens=1, temperature=0.0)
        for start in range(0, len(prompts), batch_prompts):
            outputs.extend(llm.generate(prompts[start : start + batch_prompts], params))
    finally:
        for detach in reversed(detachers):
            detach()

    request_ids = [str(output.request_id) for output in outputs]
    result: dict[int, torch.Tensor] = {}
    for layer, sink in sinks.items():
        rows = {
            str(row["request_id"]): row["hidden"]
            for row in sink.rows
            if row.get("phase") == "prefill" and "hidden" in row
        }
        missing = [request_id for request_id in request_ids if request_id not in rows]
        if missing:
            raise RuntimeError(
                f"missing prefill capture rows for request ids: {missing}"
            )
        result[layer] = torch.stack(
            [
                torch.as_tensor(rows[request_id]).float().cpu()
                for request_id in request_ids
            ]
        )
    return result


def build_dim_directions(
    llm, pos: list[str], neg: list[str], layers: list[int], batch_prompts: int = 64
) -> dict[int, DirectionRecord]:
    """Build DIM directions, normalizing mean norm over the union of pos and neg rows."""
    hiddens = capture_prompt_hiddens(llm, pos + neg, layers, batch_prompts)
    pos_count = len(pos)
    records: dict[int, DirectionRecord] = {}
    for layer, values in hiddens.items():
        positive, negative = values[:pos_count], values[pos_count:]
        records[layer] = DirectionRecord(
            direction=dim_direction(positive, negative),
            mean_norm=mean_hidden_norm(values),
        )
    return records


def save_directions(path, records: dict[int, DirectionRecord]) -> None:
    obj = {
        str(layer): {
            "direction": record.direction.float().cpu().tolist(),
            "mean_norm": record.mean_norm,
        }
        for layer, record in records.items()
    }
    write_json_atomic(path, obj)


def load_directions(path) -> dict[int, DirectionRecord]:
    obj = read_json(path)
    if not isinstance(obj, dict):
        raise ValueError("direction file must contain a mapping")
    return {
        int(layer): DirectionRecord(
            direction=torch.tensor(value["direction"], dtype=torch.float32),
            mean_norm=float(value["mean_norm"]),
        )
        for layer, value in obj.items()
    }


@dataclass(frozen=True)
class SteeringSpec:
    layer: int
    direction: torch.Tensor
    alpha: float
    mean_norm: float
    schedule: SteeringSchedule
    sign: float = 1.0


@dataclass(frozen=True)
class GenerateResult:
    text: str
    request_id: str


def steered_generate(
    llm,
    prompts: list[str],
    max_tokens: int,
    spec: SteeringSpec | None,
    batch_prompts: int = 256,
    sink: CaptureSink | None = None,
    scalar_directions: list[torch.Tensor] | None = None,
    capture_layer: int | None = None,
) -> list[GenerateResult]:
    if batch_prompts < 1:
        raise ValueError("batch_prompts must be at least 1")
    detachers = []
    resolver = (
        make_decode_index_resolver(llm)
        if spec is not None or sink is not None
        else None
    )
    try:
        if spec is not None:
            detachers.append(
                attach_steering(
                    llm,
                    layer=spec.layer,
                    direction=spec.direction,
                    alpha=spec.alpha,
                    mean_norm=spec.mean_norm,
                    schedule=spec.schedule,
                    sign=spec.sign,
                    decode_index_resolver=resolver,
                )
            )
        if sink is not None:
            if capture_layer is None and spec is None:
                raise ValueError("capture requires capture_layer or a steering spec")
            if capture_layer is None:
                assert spec is not None
                capture_target_layer = spec.layer
            else:
                capture_target_layer = capture_layer
            detachers.append(
                attach_capture(
                    llm,
                    layer=capture_target_layer,
                    sink=sink,
                    decode_index_resolver=resolver,
                    scalar_directions=scalar_directions,
                )
            )
        params = _sampling_params(max_tokens=max_tokens, temperature=0.0)
        results: list[GenerateResult] = []
        for start in range(0, len(prompts), batch_prompts):
            outputs = llm.generate(prompts[start : start + batch_prompts], params)
            results.extend(
                GenerateResult(
                    text=output.outputs[0].text, request_id=str(output.request_id)
                )
                for output in outputs
            )
        return results
    finally:
        for detach in reversed(detachers):
            detach()


__all__ = [
    "CaptureSink",
    "DirectionRecord",
    "GenerateResult",
    "SteeringSpec",
    "JSONL_TAIL_BYTES",
    "append_jsonl",
    "build_dim_directions",
    "capture_prompt_hiddens",
    "chat_prompt",
    "completed_ids",
    "get_engine",
    "load_config",
    "load_directions",
    "missing_ids",
    "mmlu_prompt",
    "parse_answer_letter",
    "prepare_checkpoint_manifest",
    "read_json",
    "read_jsonl",
    "save_directions",
    "steered_generate",
    "tee_stdout",
    "condition_id",
    "require_complete",
    "verify_manifest",
    "write_manifest",
    "write_json_atomic",
]
