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

Notification failures (missing creds, SMTP down, blocked network egress) never crash the wrapped run: they print a warning to stderr.
"""

from __future__ import annotations

import os
import fcntl
import json
import smtplib
import socket
import ssl
import sys
import tempfile
import time
import traceback
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from email.message import EmailMessage
from pathlib import Path
from typing import Literal

from . import env

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_TIMEOUT = 30


class AmbiguousDeliveryError(RuntimeError):
    """The SMTP attempt may have been accepted and must not be retried."""


_WORKFLOW_STATUSES = frozenset({"completed", "failed"})
_DELIVERY_STATUSES = frozenset({"pending", "claimed", "failed", "sent"})


def _dotenv_path() -> Path:
    return env.dotenv_path()


def _load_dotenv(path: Path | None = None) -> dict[str, str]:
    return env.load_dotenv(path if path is not None else _dotenv_path())


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


def send_email(subject: str, body: str, *, to: list[str] | None = None) -> bool | None:
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

    stage = "before-send"
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=SMTP_TIMEOUT) as server:
            server.starttls(context=ssl.create_default_context())
            server.login(user, password)
            stage = "smtp-send"
            result: object = server.send_message(msg)
    except Exception as exc:
        if stage == "smtp-send":
            raise AmbiguousDeliveryError(
                "SMTP send outcome is unknown; refusing an automatic retry"
            ) from exc
        raise
    return result if isinstance(result, bool) else None


def _try_send(subject: str, body: str) -> bool | None:
    try:
        result = send_email(subject, body)
    except AmbiguousDeliveryError as exc:
        print(f"[notify] AMBIGUOUS send for '{subject}': {exc}", file=sys.stderr)
        return None
    except Exception as exc:  # notification must never kill the run
        print(f"[notify] FAILED to send '{subject}': {exc}", file=sys.stderr)
        return False
    if result is False:
        print(
            f"[notify] FAILED to send '{subject}': sender returned false",
            file=sys.stderr,
        )
        return False
    print(f"[notify] sent: {subject}", file=sys.stderr)
    return True


def send_batch_notification(
    task: str,
    event: Literal["completed", "decision_required"] | str,
    *,
    details: str | None = None,
) -> bool:
    """Best-effort notification for an explicitly reported batch event."""
    if event not in {"completed", "decision_required"}:
        raise ValueError("event must be 'completed' or 'decision_required'")
    body = [f"task: {task}", f"event: {event}"]
    if details:
        body += ["", details]
    attempt = _try_send(f"[Prefix] {event}: {task}", "\n".join(body))
    if attempt is None:
        raise AmbiguousDeliveryError(
            "SMTP send outcome is unknown; refusing an automatic retry"
        )
    return attempt is True


def _sync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_notification_state(path: Path, payload: Mapping[str, object]) -> None:
    _reject_persistence_symlinks(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _reject_persistence_symlinks(path)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _claim_notification_state(path: Path, payload: Mapping[str, object]) -> bool:
    _reject_persistence_symlinks(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _reject_persistence_symlinks(path)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return False
        _sync_directory(path.parent)
        os.unlink(temporary)
        _sync_directory(path.parent)
        return True
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _notification_status(path: Path) -> str:
    _reject_persistence_symlinks(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return "claimed"
    if not isinstance(payload, dict):
        return "claimed"
    status = payload.get("delivery_status", payload.get("status", "claimed"))
    return (
        status
        if isinstance(status, str) and status in _DELIVERY_STATUSES
        else "claimed"
    )


def _claim_failed_notification(path: Path, payload: Mapping[str, object]) -> bool:
    return _claim_retryable_notification(path, payload, expected="failed")


@contextmanager
def _notification_lock(path: Path) -> Iterator[None]:
    _reject_persistence_symlinks(path)
    lock_path = path.with_name(f".{path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    _reject_persistence_symlinks(lock_path)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    with os.fdopen(descriptor, "a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _claim_retryable_notification(
    path: Path,
    payload: Mapping[str, object],
    *,
    expected: Literal["failed", "pending"],
) -> bool:
    _reject_persistence_symlinks(path)
    with _notification_lock(path):
        if _notification_status(path) != expected:
            return False
        existing = _read_notification_payload(path)
        merged = dict(existing) if existing is not None else {}
        merged.update(payload)
        _write_notification_state(path, merged)
        return True


def _update_terminal_workflow(path: Path, payload: Mapping[str, object]) -> None:
    _reject_persistence_symlinks(path)
    with _notification_lock(path):
        existing = _read_notification_payload(path)
        if existing is None:
            return
        delivery = existing.get("delivery_status", existing.get("status"))
        if delivery not in {"sent", "claimed"}:
            return
        merged = dict(existing)
        for key in ("task", "event", "workflow_status", "details"):
            if key in payload:
                merged[key] = payload[key]
        _write_notification_state(path, merged)


def _reject_persistence_symlinks(path: Path) -> None:
    current = Path(path.anchor) if path.is_absolute() else Path.cwd()
    components = path.parts[1:] if path.is_absolute() else path.parts
    for component in components:
        current /= component
        if current.is_symlink():
            raise ValueError(
                f"refusing symlinked persistence path component: {current}"
            )


def _read_notification_payload(path: Path) -> dict[str, object] | None:
    _reject_persistence_symlinks(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def _delivery_status(payload: Mapping[str, object] | None) -> str:
    if payload is None:
        return "claimed"
    value = payload.get("delivery_status", payload.get("status", "claimed"))
    return value if isinstance(value, str) else "claimed"


def _workflow_status(payload: Mapping[str, object] | None) -> str | None:
    if payload is None:
        return None
    value = payload.get("workflow_status")
    if isinstance(value, str) and value in _WORKFLOW_STATUSES:
        return value
    event = payload.get("event")
    return event if isinstance(event, str) and event in _WORKFLOW_STATUSES else None


def record_terminal_workflow(
    task: str,
    *,
    status: Literal["completed", "failed"],
    state_path: str | Path,
) -> None:
    """Persist workflow completion without changing an existing delivery state."""
    state = Path(state_path)
    with _notification_lock(state):
        existing = _read_notification_payload(state)
        if existing is None and state.exists():
            raise RuntimeError("ambiguous notification state cannot be replaced")
        payload = dict(existing) if existing is not None else {}
        payload["task"] = task
        payload["event"] = status
        payload["workflow_status"] = status
        delivery = _delivery_status(existing) if existing is not None else "pending"
        if delivery not in _DELIVERY_STATUSES:
            delivery = "pending"
        payload["delivery_status"] = delivery
        payload["status"] = delivery
        payload["notification_attempted"] = delivery != "pending"
        _write_notification_state(state, payload)


def _terminal_claim(
    task: str,
    status: Literal["completed", "failed"],
    details: str | None,
) -> dict[str, object]:
    return {
        "event": status,
        "workflow_status": status,
        "delivery_status": "claimed",
        "status": "claimed",
        "notification_attempted": True,
        "task": task,
        "details": details or "",
    }


def claim_terminal_workflow(
    task: str,
    *,
    status: Literal["completed", "failed"],
    state_path: str | Path,
) -> bool:
    return _claim_retryable_notification(
        Path(state_path), _terminal_claim(task, status, None), expected="pending"
    )


def _attempt_custom_sender(sender: Callable[[], object]) -> bool | None:
    try:
        result = sender()
    except AmbiguousDeliveryError as exc:
        print(f"[notify] AMBIGUOUS custom send: {exc}", file=sys.stderr)
        return None
    except Exception as exc:
        print(f"[notify] FAILED custom send: {exc}", file=sys.stderr)
        return False
    return result is not False


def finalize_terminal_notification(
    task: str,
    *,
    status: Literal["completed", "failed"],
    state_path: str | Path,
    details: str | None = None,
    sender: Callable[[], object] | None = None,
) -> str:
    """Deliver one durable summary for a fully finalized workflow."""
    state = Path(state_path)
    claim = _terminal_claim(task, status, details)
    if not _claim_notification_state(state, claim):
        existing = _notification_status(state)
        if existing in {"sent", "claimed"}:
            _update_terminal_workflow(state, claim)
            return existing
        if existing == "failed":
            claimed = _claim_retryable_notification(state, claim, expected="failed")
        elif existing == "pending":
            claimed = _claim_retryable_notification(state, claim, expected="pending")
        else:
            claimed = False
        if not claimed:
            return _notification_status(state)

    subject_status = "done" if status == "completed" else "FAILED"
    body = [f"task: {task}", f"status: {status}"]
    if details:
        body += ["", details]
    delivered = (
        _attempt_custom_sender(sender)
        if sender is not None
        else _try_send(f"[Prefix] {subject_status}: {task}", "\n".join(body))
    )
    if delivered is None:
        return "claimed"

    current = _read_notification_payload(state)
    saved = dict(current) if current is not None else dict(claim)
    if delivered is True:
        saved["delivery_status"] = "sent"
        saved["status"] = "sent"
        _write_notification_state(state, saved)
        return "sent"
    saved["delivery_status"] = "failed"
    saved["status"] = "failed"
    _write_notification_state(state, saved)
    return "failed"


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
    enabled: bool = True,
) -> Iterator[None]:
    """Email ``task`` success/failure when the wrapped block exits.

    Set ``enabled=False`` to retain the context-manager boundary without sending
    an automatic notification.

    On exception the traceback (plus the tail of ``log_file`` when given) is
    emailed and the exception re-raised. On success a short summary email is
    sent. Notification failures are swallowed with a stderr warning.
    """
    if not enabled:
        yield
        return

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


__all__ = [
    "AmbiguousDeliveryError",
    "claim_terminal_workflow",
    "configured",
    "finalize_terminal_notification",
    "notify_on_exit",
    "record_terminal_workflow",
    "send_batch_notification",
    "send_email",
]
