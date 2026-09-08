"""Gmail notifications for experiment completion and failure.

Wrap experiment entrypoints with :func:`notify_on_exit` to receive an email
when the wrapped block finishes or raises::

    from prefix.notify import notify_on_exit

    with notify_on_exit("exp2-harmbench-sweep", log_file="logs/exp2.log"):
        main()

Credentials are read from the environment (or a gitignored ``.env`` at the
repo root; real environment variables win over ``.env`` values):

    PREFIX_GMAIL_USER=huohuangcw@gmail.com
    PREFIX_GMAIL_APP_PASSWORD=<16-char app password>
    PREFIX_NOTIFY_TO=<comma-separated recipients; default: sender>

Notification failures (missing creds, SMTP down, blocked egress on cluster
compute nodes) never crash the wrapped run: they print a warning to stderr.
"""

from __future__ import annotations

import os
import smtplib
import socket
import ssl
import sys
import time
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from email.message import EmailMessage
from pathlib import Path

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_TIMEOUT = 30


def _dotenv_path() -> Path:
    return Path(__file__).resolve().parents[2] / ".env"


def _load_dotenv(path: Path | None = None) -> dict[str, str]:
    """Parse a ``KEY=VALUE`` file; ignores comments, blanks, quoted values."""
    path = path if path is not None else _dotenv_path()
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


def _ensure_env() -> None:
    for key, value in _load_dotenv().items():
        os.environ.setdefault(key, value)


def configured() -> bool:
    _ensure_env()
    return bool(os.environ.get("PREFIX_GMAIL_USER")) and bool(
        os.environ.get("PREFIX_GMAIL_APP_PASSWORD")
    )


def _recipients() -> list[str]:
    raw = os.environ.get("PREFIX_NOTIFY_TO") or os.environ.get("PREFIX_GMAIL_USER", "")
    return [addr.strip() for addr in raw.split(",") if addr.strip()]


def send_email(subject: str, body: str, *, to: list[str] | None = None) -> None:
    """Send a plain-text email via Gmail SMTP; raises on any failure."""
    _ensure_env()
    user = os.environ.get("PREFIX_GMAIL_USER", "")
    password = os.environ.get("PREFIX_GMAIL_APP_PASSWORD", "").replace(" ", "")
    if not user or not password:
        raise RuntimeError(
            "Gmail notify not configured: set PREFIX_GMAIL_USER and "
            "PREFIX_GMAIL_APP_PASSWORD (see src/prefix/notify.py docstring)"
        )
    recipients = to if to is not None else _recipients()
    if not recipients:
        raise RuntimeError("Gmail notify has no recipients (PREFIX_NOTIFY_TO empty)")

    msg = EmailMessage()
    msg["From"] = user
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject
    msg.set_content(body)

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=SMTP_TIMEOUT) as server:
        server.starttls(context=ssl.create_default_context())
        server.login(user, password)
        server.send_message(msg)


def _try_send(subject: str, body: str) -> None:
    try:
        send_email(subject, body)
        print(f"[notify] sent: {subject}", file=sys.stderr)
    except Exception as exc:  # notification must never kill the run
        print(f"[notify] FAILED to send '{subject}': {exc}", file=sys.stderr)


def _tail(path: str | Path, lines: int) -> str:
    try:
        content = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(content[-lines:]) if lines > 0 else ""


@contextmanager
def notify_on_exit(
    task: str,
    *,
    log_file: str | Path | None = None,
    tail_lines: int = 40,
) -> Iterator[None]:
    """Email ``task`` success/failure when the wrapped block exits.

    On exception the traceback (plus the tail of ``log_file`` when given) is
    emailed and the exception re-raised. On success a short summary email is
    sent. Notification failures are swallowed with a stderr warning.
    """
    start = time.monotonic()
    host = socket.gethostname()
    try:
        yield
    except BaseException:
        body = [
            f"task: {task}",
            f"host: {host}",
            f"status: FAILED after {time.monotonic() - start:,.0f}s",
            "",
            traceback.format_exc(),
        ]
        if log_file is not None:
            body += [
                "",
                f"--- tail of {log_file} (last {tail_lines} lines) ---",
                _tail(log_file, tail_lines),
            ]
        _try_send(f"[Prefix] FAILED: {task}", "\n".join(body))
        raise
    else:
        body = [
            f"task: {task}",
            f"host: {host}",
            f"status: completed in {time.monotonic() - start:,.0f}s",
        ]
        if log_file is not None:
            body += [
                "",
                f"--- tail of {log_file} (last {tail_lines} lines) ---",
                _tail(log_file, tail_lines),
            ]
        _try_send(f"[Prefix] done: {task}", "\n".join(body))


__all__ = ["configured", "notify_on_exit", "send_email"]
