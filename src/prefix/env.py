"""Shared .env loading (gitignored repo-root dotenv; real env wins)."""

from __future__ import annotations

import os
from pathlib import Path


def dotenv_path() -> Path:
    return Path(__file__).resolve().parents[2] / ".env"


def load_dotenv(path: Path | None = None) -> dict[str, str]:
    path = path if path is not None else dotenv_path()
    values: dict[str, str] = {}
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
        if key:
            values[key] = value
    return values


def ensure_env() -> None:
    for key, value in load_dotenv().items():
        os.environ.setdefault(key, value)


__all__ = ["dotenv_path", "ensure_env", "load_dotenv"]
