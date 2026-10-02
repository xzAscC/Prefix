"""Shared .env loading (gitignored repo-root dotenv; real env wins)."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


_ALLOWED_DOTENV_KEYS = frozenset(
    {
        "PREFIX_GMAIL_USER",
        "PREFIX_GMAIL_APP_PASSWORD",
        "PREFIX_NOTIFY_TO",
        "GOOGLE_CLOUD_PROJECT",
        "PREFIX_DATA_CACHE",
    }
)


def _trusted_dotenv(path: Path) -> bool:
    try:
        if path.is_symlink() or path.resolve(strict=True) != path:
            return False
        stat_result = path.stat()
    except OSError:
        return False
    if stat_result.st_uid != os.getuid() or stat_result.st_mode & 0o022:
        return False

    getfacl = shutil.which("getfacl")
    if getfacl is None:
        return False
    result = subprocess.run(
        [getfacl, "-cp", "--", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return False
    for line in result.stdout.splitlines():
        if line in {"", "# file:", "# owner:", "# group:"}:
            continue
        if ":" in line and line.rsplit(":", 1)[-1].find("w") >= 0:
            if not line.startswith("user::") and not line.startswith("default:user::"):
                return False
    return True


def dotenv_path() -> Path:
    return Path(__file__).resolve().parents[2] / ".env"


def load_dotenv(path: Path | None = None) -> dict[str, str]:
    path = path if path is not None else dotenv_path()
    values: dict[str, str] = {}
    if not path.is_file() or not _trusted_dotenv(path):
        return values
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key in _ALLOWED_DOTENV_KEYS:
            values[key] = value
    return values


def ensure_env() -> None:
    for key, value in load_dotenv().items():
        _ = os.environ.setdefault(key, value)


__all__ = ["dotenv_path", "ensure_env", "load_dotenv"]
