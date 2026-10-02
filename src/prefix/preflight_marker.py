from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path
from typing import Any, cast

PREFLIGHT_MARKER_REVISIONS = {
    "Qwen/Qwen3-4B": "1cfa9a7208912126459214e8b04321603b3df60c",
    "Qwen/Qwen3-14B": "40c069824f4251a91eefaf281ebe4c544efd3e18",
    "allenai/Olmo-3-7B-Think": "d97e442d7cc678210054dbcc9b440894d62c89a4",
    "allenai/Olmo-3-32B-Think": "f2edda15216e738ef2bb73771e11890e152b2112",
}


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
            raise ValueError("preflight path is symlinked")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_preflight_marker(
    path: str | Path,
    *,
    model_id: str | None = None,
    run_root: str | Path | None = None,
    preflight_job_id: str | None = None,
) -> dict[str, Any]:
    marker_path = Path(path)
    _reject_symlink_components(marker_path)
    try:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("preflight marker is missing or invalid") from error
    if not isinstance(payload, dict):
        raise ValueError("preflight marker must be a JSON object")
    if payload.get("schema_version") != 1 or payload.get("status") != "complete":
        raise ValueError("preflight marker schema or status mismatch")
    marker_job_id = payload.get("preflight_job_id")
    if not isinstance(marker_job_id, str) or not marker_job_id:
        raise ValueError("preflight marker job binding is invalid")
    if preflight_job_id is not None and marker_job_id != preflight_job_id:
        raise ValueError("preflight marker job mismatch")
    marker_run_root = payload.get("run_root")
    if not isinstance(marker_run_root, str) or not Path(marker_run_root).is_absolute():
        raise ValueError("preflight marker run root is invalid")
    _reject_symlink_components(Path(marker_run_root))
    if run_root is not None:
        caller_run_root = Path(run_root)
        _reject_symlink_components(caller_run_root)
        if Path(marker_run_root).resolve() != caller_run_root.resolve():
            raise ValueError("preflight marker run root mismatch")
    models = payload.get("models")
    if not isinstance(models, dict) or set(models) != set(PREFLIGHT_MARKER_REVISIONS):
        raise ValueError("preflight marker model set mismatch")
    for expected_model, expected_revision in PREFLIGHT_MARKER_REVISIONS.items():
        entry = models.get(expected_model)
        if not isinstance(entry, dict):
            raise ValueError("preflight marker model entry mismatch")
        if set(entry) != {
            "model_id",
            "model_revision",
            "summary_path",
            "summary_sha256",
            "manifest_path",
            "manifest_sha256",
        }:
            raise ValueError("preflight marker model entry schema mismatch")
        if (
            entry["model_id"] != expected_model
            or entry["model_revision"] != expected_revision
        ):
            raise ValueError("preflight marker model revision mismatch")
        for kind in ("summary", "manifest"):
            artifact = entry[f"{kind}_path"]
            digest = entry[f"{kind}_sha256"]
            if not isinstance(artifact, str) or not Path(artifact).is_absolute():
                raise ValueError("preflight marker artifact path is not absolute")
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError("preflight marker artifact digest is invalid")
            artifact_path = Path(artifact)
            container = "preflight" if kind == "summary" else "preflight-checkpoints"
            expected_artifact = (
                Path(marker_run_root)
                / container
                / expected_model.replace("/", "--")
                / f"{kind}.json"
            )
            if artifact_path != expected_artifact:
                raise ValueError("preflight marker artifact is not run-root bound")
            _reject_symlink_components(artifact_path)
            if not artifact_path.is_file() or artifact_path.is_symlink():
                raise ValueError("preflight marker artifact is missing or symlinked")
            if _sha256_file(artifact_path) != digest:
                raise ValueError("preflight marker artifact digest mismatch")
    if model_id is not None and model_id not in PREFLIGHT_MARKER_REVISIONS:
        raise ValueError("preflight marker model is unrelated")
    return cast(dict[str, Any], payload)


def marker_sha256(path: str | Path) -> str:
    return _sha256_file(Path(path))
