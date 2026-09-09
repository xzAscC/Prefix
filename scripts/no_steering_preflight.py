from __future__ import annotations

import argparse
import json
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Any

import torch

from prefix import no_steering
from prefix.data import validate_offline_dataset_caches


ROOT = Path(__file__).resolve().parents[1]
LOCAL_FULL_LOAD_MODEL_IDS = (
    "Qwen/Qwen3-4B",
    "allenai/Olmo-3-7B-Think",
)
PROTOCOL_ONLY_MODEL_IDS = (
    "Qwen/Qwen3-14B",
    "allenai/Olmo-3-32B-Think",
)


def model_spec(model_id: str) -> no_steering.ModelSpec:
    return no_steering.model_spec(model_id)


def _snapshot_path(model_id: str, cache_root: str | Path) -> Path:
    spec = model_spec(model_id)
    return (
        Path(cache_root) / "hub" / f"models--{spec.slug}" / "snapshots" / spec.revision
    )


def _offline_environment() -> None:
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[name] = "1"


def _validate_snapshot(model_id: str, snapshot: Path) -> None:
    spec = model_spec(model_id)
    if snapshot.name != spec.revision or snapshot.is_symlink() or not snapshot.is_dir():
        raise ValueError(f"snapshot is not the exact pinned revision for {model_id}")
    try:
        repo_root = snapshot.parents[1].resolve(strict=True)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"snapshot repository is missing for {model_id}"
        ) from error
    config_path = snapshot / "config.json"
    _validate_nonempty_file(config_path, repo_root, "config")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid config.json for {model_id}") from error
    if not isinstance(config, dict):
        raise ValueError(f"config.json must be an object for {model_id}")
    if config.get("quantization_config") is not None or config.get("quantization"):
        raise ValueError(
            f"quantization is forbidden for no-steering assets: {model_id}"
        )

    for path in snapshot.rglob("*"):
        if path.is_symlink():
            try:
                target = path.resolve(strict=True)
            except FileNotFoundError as error:
                raise FileNotFoundError(f"broken symlink/blob in {path}") from error
            if repo_root not in target.parents:
                raise ValueError(f"snapshot reference escapes cache repo: {path}")

    _validate_weights(model_id, snapshot, repo_root)

    tokenizer_candidates = ("tokenizer.json", "tokenizer.model", "spiece.model")
    if not any(
        _is_valid_nonempty_file(snapshot / name, repo_root)
        for name in tokenizer_candidates
    ):
        raise FileNotFoundError(
            f"snapshot has no nonempty tokenizer asset for {model_id}: {snapshot}"
        )
    _validate_nonempty_file(
        snapshot / "tokenizer_config.json", repo_root, "tokenizer config"
    )


def _validate_nonempty_file(path: Path, repo_root: Path, kind: str) -> Path:
    try:
        target = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"missing {kind}: {path}") from error
    if target != repo_root and repo_root not in target.parents:
        raise ValueError(f"{kind} escapes cache repo: {path}")
    if not target.is_file() or target.stat().st_size == 0:
        raise ValueError(f"{kind} must be a nonempty file: {path}")
    return target


def _is_valid_nonempty_file(path: Path, repo_root: Path) -> bool:
    try:
        _validate_nonempty_file(path, repo_root, "tokenizer")
    except (FileNotFoundError, ValueError):
        return False
    return True


def _validate_weights(model_id: str, snapshot: Path, repo_root: Path) -> None:
    direct_suffixes = (".safetensors", ".bin", ".pt", ".pth")
    direct = [
        path
        for path in snapshot.iterdir()
        if path.is_file() and path.name.endswith(direct_suffixes)
    ]
    indexes = [
        path
        for path in snapshot.iterdir()
        if path.name.endswith((".safetensors.index.json", ".bin.index.json"))
    ]
    shard_like = [
        path
        for path in direct
        if re.search(r"-\d{5}-of-\d{5}\.(?:safetensors|bin|pt|pth)$", path.name)
    ]
    if shard_like and not indexes:
        raise ValueError(
            f"snapshot contains shard-like weights without a weight index for {model_id}"
        )
    for path in direct:
        _validate_nonempty_file(path, repo_root, "weight file")

    for index in indexes:
        _validate_nonempty_file(index, repo_root, "weight index")
        try:
            payload = json.loads(index.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid weight index for {model_id}: {index}") from error
        weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"weight index has no nonempty weight_map: {index}")
        for parameter, shard_name in weight_map.items():
            if not isinstance(parameter, str) or not isinstance(shard_name, str):
                raise ValueError(f"weight index has invalid shard entry: {index}")
            shard = Path(shard_name)
            if shard.is_absolute() or ".." in shard.parts:
                raise ValueError(f"weight shard escapes cache repo: {shard_name}")
            _validate_nonempty_file(snapshot / shard, repo_root, "weight shard")

    if not direct and not indexes:
        raise FileNotFoundError(
            f"snapshot has no direct weights or weight index for {model_id}"
        )


def _write_atomic(path: Path, content: str) -> None:
    _reject_symlink_components(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
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
            raise ValueError(f"refusing symlinked preflight path component: {current}")


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def check_model_snapshots(
    model_ids: tuple[str, ...] | list[str],
    *,
    cache_root: str | Path,
    load: bool = False,
    manifest_path: str | Path | None = None,
    marker_path: str | Path | None = None,
) -> dict[str, Path]:
    """Check every pinned snapshot before doing any device discovery.

    ``load`` is intentionally separate from this check: protocol-only models
    need their assets verified, but must not be loaded by local smoke runs.
    """
    _offline_environment()
    requested = tuple(model_ids)
    if (
        manifest_path is not None or marker_path is not None
    ) and requested != no_steering.MODEL_MATRIX:
        raise ValueError("asset manifest and marker require the exact required models")
    snapshots = {
        model_id: _snapshot_path(model_id, cache_root) for model_id in requested
    }
    missing = [
        model_id
        for model_id, path in snapshots.items()
        if path.is_symlink() or not path.is_dir()
    ]
    if missing:
        first = missing[0]
        raise FileNotFoundError(
            f"missing exact pinned revision for {first}: {snapshots[first]}"
        )
    for model_id, snapshot in snapshots.items():
        _validate_snapshot(model_id, snapshot)
    if manifest_path is not None:
        payload = {
            "models": {
                model_id: {
                    "model_id": model_id,
                    "revision": model_spec(model_id).revision,
                    "snapshot": str(snapshot),
                }
                for model_id, snapshot in snapshots.items()
            }
        }
        _write_atomic(
            Path(manifest_path), json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
    if marker_path is not None:
        _write_atomic(
            Path(marker_path),
            "\n".join(
                f"{model_id}={model_spec(model_id).revision}" for model_id in snapshots
            )
            + "\n",
        )
    if load:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for local no-steering model loading")
    return snapshots


def load_model(
    *,
    model_id: str,
    snapshot: str | Path,
    revision: str,
    steering: bool = False,
    quantization: str | bool | None = None,
) -> Any:
    """Load one pinned, unsteered local model without downloading assets."""
    if steering:
        raise ValueError("no-steering preflight cannot enable steering")
    if quantization not in (None, False):
        raise ValueError("no-steering preflight cannot enable quantization")
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        str(snapshot),
        revision=revision,
        local_files_only=True,
        device_map="auto",
        torch_dtype="auto",
    )


def load_local_models(
    *,
    cache_root: str | Path,
    model_ids: tuple[str, ...] | list[str] = no_steering.MODEL_MATRIX,
    snapshots: dict[str, Path] | None = None,
) -> dict[str, Any]:
    """Load only the two models permitted for local full-load smoke tests."""
    requested = tuple(model_ids)
    no_steering.validate_model_matrix(no_steering.MODEL_SPECS)
    checked = snapshots or check_model_snapshots(requested, cache_root=cache_root)
    return {
        model_id: load_model(
            model_id=model_id,
            snapshot=checked[model_id],
            revision=model_spec(model_id).revision,
            steering=False,
            quantization=None,
        )
        for model_id in LOCAL_FULL_LOAD_MODEL_IDS
        if model_id in requested
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate local no-steering assets")
    parser.add_argument("--model-id", action="append", choices=no_steering.MODEL_MATRIX)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(os.environ.get("HF_HOME", ROOT / "models")),
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--marker", type=Path)
    parser.add_argument("--dataset-cache-root", type=Path)
    parser.add_argument("--phase", choices=("check", "load"), default="check")
    parser.add_argument(
        "--limit", type=int, help="accepted for SDSC smoke-script compatibility"
    )
    args = parser.parse_args(argv)
    model_ids = tuple(args.model_id or no_steering.MODEL_MATRIX)
    if args.dataset_cache_root is not None:
        validate_offline_dataset_caches(args.dataset_cache_root)
    snapshots = check_model_snapshots(
        model_ids,
        cache_root=args.cache_root,
        load=args.phase == "load",
        manifest_path=args.manifest,
        marker_path=args.marker,
    )
    if args.phase == "load":
        load_local_models(
            cache_root=args.cache_root, model_ids=model_ids, snapshots=snapshots
        )
    print({model_id: str(path) for model_id, path in snapshots.items()})


if __name__ == "__main__":
    main()
