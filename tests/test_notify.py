"""Tests for the Gmail notify component."""

from __future__ import annotations

import json
import smtplib
import threading
from pathlib import Path
from typing import Literal

import pytest

from prefix import notify


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for key in ("PREFIX_GMAIL_USER", "PREFIX_GMAIL_APP_PASSWORD", "PREFIX_NOTIFY_TO"):
        monkeypatch.delenv(key, raising=False)
    # isolate from the developer's real .env at the repo root
    monkeypatch.setattr(notify, "_dotenv_path", lambda: tmp_path / "absent.env")


@pytest.fixture
def fake_smtp(monkeypatch: pytest.MonkeyPatch) -> type:
    instances: list[object] = []

    class FakeSMTP:
        def __init__(self, host: str, port: int, timeout: float | None = None) -> None:
            self.host = host
            self.port = port
            self.ops: list[tuple[object, ...]] = []
            self.sent: list[object] = []
            instances.append(self)

        def __enter__(self) -> "FakeSMTP":
            return self

        def __exit__(self, *exc: object) -> bool:
            return False

        def starttls(self, context: object | None = None) -> None:
            self.ops.append(("starttls",))

        def login(self, user: str, password: str) -> None:
            self.ops.append(("login", user, password))

        def send_message(self, msg: object) -> None:
            self.sent.append(msg)

        def quit(self) -> None:
            self.ops.append(("quit",))

    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    setattr(FakeSMTP, "instances", instances)
    return FakeSMTP


def _set_creds(
    monkeypatch: pytest.MonkeyPatch, password: str = "abcdefghijklmnop"
) -> None:
    monkeypatch.setenv("PREFIX_GMAIL_USER", "sender@gmail.com")
    monkeypatch.setenv("PREFIX_GMAIL_APP_PASSWORD", password)


# --- dotenv parsing ---------------------------------------------------------


def test_load_dotenv_parses_keys_quotes_and_comments(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "# comment line\n"
        "PREFIX_GMAIL_USER=sender@gmail.com\n"
        'PREFIX_GMAIL_APP_PASSWORD="abcd efgh ijkl mnop"\n'
        "\n"
        "IGNORED_NO_EQUALS_LINE\n"
        "PREFIX_NOTIFY_TO='a@x.com, b@y.com'\n",
        encoding="utf-8",
    )
    values = notify._load_dotenv(env)
    assert values["PREFIX_GMAIL_USER"] == "sender@gmail.com"
    assert values["PREFIX_GMAIL_APP_PASSWORD"] == "abcd efgh ijkl mnop"
    assert values["PREFIX_NOTIFY_TO"] == "a@x.com, b@y.com"
    assert "IGNORED_NO_EQUALS_LINE" not in values
    assert len(values) == 3


def test_load_dotenv_ignores_process_environment_injection_keys(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "PREFIX_GMAIL_USER=sender@gmail.com\n"
        "LD_PRELOAD=/tmp/evil.so\n"
        "PYTHONPATH=/tmp/evil\n"
        "NOTIFICATION_STATE=/tmp/redirected.json\n",
        encoding="utf-8",
    )

    values = notify._load_dotenv(env)

    assert values == {"PREFIX_GMAIL_USER": "sender@gmail.com"}


@pytest.mark.parametrize("kind", ["mode", "symlink"])
def test_load_dotenv_rejects_unsafe_optional_file(tmp_path: Path, kind: str) -> None:
    target = tmp_path / "real.env"
    target.write_text("PREFIX_GMAIL_USER=sender@gmail.com\n", encoding="utf-8")
    if kind == "mode":
        target.chmod(0o666)
        env = target
    else:
        env = tmp_path / ".env"
        env.symlink_to(target)

    assert notify._load_dotenv(env) == {}


def test_load_dotenv_missing_file_returns_empty(tmp_path: Path) -> None:
    assert notify._load_dotenv(tmp_path / "nope.env") == {}


def test_ensure_env_populates_missing_keys_from_dotenv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = tmp_path / ".env"
    env.write_text("PREFIX_GMAIL_USER=dot@gmail.com\n", encoding="utf-8")
    monkeypatch.setattr(notify, "_dotenv_path", lambda: env)
    monkeypatch.setenv("PREFIX_GMAIL_USER", "real@gmail.com")  # real env wins
    notify._ensure_env()
    import os

    assert os.environ["PREFIX_GMAIL_USER"] == "real@gmail.com"


# --- configured / recipients ------------------------------------------------


def test_configured_false_without_credentials() -> None:
    assert notify.configured() is False


def test_configured_true_with_user_and_password(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_creds(monkeypatch)
    assert notify.configured() is True


def test_recipients_default_to_sender(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_creds(monkeypatch)
    assert notify._recipients() == ["sender@gmail.com"]


def test_recipients_split_on_commas(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_creds(monkeypatch)
    monkeypatch.setenv("PREFIX_NOTIFY_TO", "a@x.com, b@y.com ,c@z.com")
    assert notify._recipients() == ["a@x.com", "b@y.com", "c@z.com"]


# --- send_email -------------------------------------------------------------


def test_send_email_smtp_flow_and_headers(
    fake_smtp: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_creds(monkeypatch)
    monkeypatch.setenv("PREFIX_NOTIFY_TO", "rcpt@x.com")
    notify.send_email("hello subject", "hello body")
    (inst,) = fake_smtp.instances  # type: ignore[attr-defined]
    assert inst.host == "smtp.gmail.com" and inst.port == 587
    ops = [op[0] for op in inst.ops]
    assert ops.index("starttls") < ops.index("login")
    assert ("login", "sender@gmail.com", "abcdefghijklmnop") in inst.ops
    (msg,) = inst.sent
    assert msg["From"] == "sender@gmail.com"
    assert msg["To"] == "rcpt@x.com"
    assert msg["Subject"] == "hello subject"
    assert "hello body" in msg.get_content()


def test_send_email_strips_spaces_from_app_password(
    fake_smtp: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_creds(monkeypatch, password="abcd efgh ijkl mnop")
    notify.send_email("s", "b")
    (inst,) = fake_smtp.instances  # type: ignore[attr-defined]
    assert ("login", "sender@gmail.com", "abcdefghijklmnop") in inst.ops


def test_send_email_requires_configuration() -> None:
    with pytest.raises(RuntimeError, match="PREFIX_GMAIL_USER"):
        notify.send_email("s", "b")


# --- notify_on_exit ---------------------------------------------------------


def test_notify_on_exit_success_sends_email(
    fake_smtp: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_creds(monkeypatch)
    with notify.notify_on_exit("exp1"):
        pass
    (inst,) = fake_smtp.instances  # type: ignore[attr-defined]
    (msg,) = inst.sent
    assert "done" in msg["Subject"] and "exp1" in msg["Subject"]
    body = msg.get_content()
    assert "completed" in body
    assert "status: completed" in body or "status:" in body


def test_notify_on_exit_failure_sends_traceback_and_reraises(
    fake_smtp: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_creds(monkeypatch)
    with pytest.raises(RuntimeError, match="boom"):
        with notify.notify_on_exit("exp1"):
            raise RuntimeError("boom")
    (inst,) = fake_smtp.instances  # type: ignore[attr-defined]
    (msg,) = inst.sent
    assert "FAILED" in msg["Subject"] and "exp1" in msg["Subject"]
    body = msg.get_content()
    assert "RuntimeError: boom" in body
    assert "Traceback" in body


def test_notify_on_exit_includes_log_tail(
    fake_smtp: type, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_creds(monkeypatch)
    log = tmp_path / "run.log"
    log.write_text("\n".join(f"line{i}" for i in range(100)), encoding="utf-8")
    with pytest.raises(ValueError):
        with notify.notify_on_exit("exp", log_file=log, tail_lines=10):
            raise ValueError("x")
    (inst,) = fake_smtp.instances  # type: ignore[attr-defined]
    (msg,) = inst.sent
    body = msg.get_content()
    assert "line99" in body
    assert "line90" in body
    assert "line89" not in body


def test_notify_on_exit_unconfigured_is_silent_noop(fake_smtp: type) -> None:
    with notify.notify_on_exit("exp"):
        pass
    with pytest.raises(RuntimeError):
        with notify.notify_on_exit("exp"):
            raise RuntimeError("boom")
    assert fake_smtp.instances == []  # type: ignore[attr-defined]


def test_notify_on_exit_send_failure_never_masks_result(
    fake_smtp: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_creds(monkeypatch)
    monkeypatch.setattr(
        notify,
        "send_email",
        lambda *a, **k: (_ for _ in ()).throw(OSError("smtp down")),
    )
    with notify.notify_on_exit("exp"):
        pass  # must not raise despite smtp failure


def test_notify_on_exit_hostname_in_body(
    fake_smtp: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    import socket

    _set_creds(monkeypatch)
    with notify.notify_on_exit("exp"):
        pass
    (inst,) = fake_smtp.instances  # type: ignore[attr-defined]
    (msg,) = inst.sent
    assert socket.gethostname() in msg.get_content()


def test_notify_on_exit_disabled_sends_no_email_and_reraises(
    fake_smtp: type,
) -> None:
    with notify.notify_on_exit("child", enabled=False):
        pass
    with pytest.raises(RuntimeError, match="boom"):
        with notify.notify_on_exit("child", enabled=False):
            raise RuntimeError("boom")
    assert fake_smtp.instances == []  # type: ignore[attr-defined]


@pytest.mark.parametrize("event", ["completed", "decision_required"])
def test_send_batch_notification_sends_one_neutral_event_email(
    fake_smtp: type, monkeypatch: pytest.MonkeyPatch, event: str
) -> None:
    _set_creds(monkeypatch)
    notify.send_batch_notification("no-steering batch", event)
    (inst,) = fake_smtp.instances  # type: ignore[attr-defined]
    (msg,) = inst.sent
    assert len(inst.sent) == 1
    assert event in msg["Subject"]
    assert "no-steering batch" in msg.get_content()


def test_send_batch_notification_invalid_event_sends_no_email(
    fake_smtp: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_creds(monkeypatch)
    with pytest.raises(ValueError, match="completed"):
        notify.send_batch_notification("batch", "failed")
    assert fake_smtp.instances == []  # type: ignore[attr-defined]


def test_send_batch_notification_failure_is_non_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        notify,
        "send_email",
        lambda *a, **k: (_ for _ in ()).throw(OSError("smtp down")),
    )
    notify.send_batch_notification("batch", "completed")


def _terminal_summary(
    monkeypatch: pytest.MonkeyPatch,
    state_path: Path,
    *,
    status: Literal["completed", "failed"],
    calls: list[tuple[str, str]],
) -> None:
    def fake_sender(subject: str, body: str, **_: object) -> None:
        calls.append((subject, body))

    monkeypatch.setattr(notify, "send_email", fake_sender)
    notify.finalize_terminal_notification(
        "benchmark-investigation",
        status=status,
        state_path=state_path,
        details="whole workflow finalized",
    )


def test_terminal_summary_success_is_only_email_after_all_phases(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, str]] = []
    state = tmp_path / "checkpoints" / "notification.json"

    for phase in ("prepare", "generate", "analyze"):
        with notify.notify_on_exit(phase, enabled=False):
            pass
    _terminal_summary(monkeypatch, state, status="completed", calls=calls)

    assert len(calls) == 1
    assert "benchmark-investigation" in calls[0][0]
    assert "completed" in calls[0][1]
    assert json.loads(state.read_text(encoding="utf-8"))["status"] == "sent"


def test_terminal_summary_failure_is_one_terminal_email(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, str]] = []
    state = tmp_path / "notification.json"

    _terminal_summary(monkeypatch, state, status="failed", calls=calls)

    assert len(calls) == 1
    assert "FAILED" in calls[0][0]
    assert "failed" in calls[0][1]


def test_failed_managed_job_then_completed_scoring_uses_one_shared_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, str]] = []
    state = tmp_path / "generation" / "notification.json"
    attempts = 0

    def sender(subject: str, body: str, **_: object) -> None:
        nonlocal attempts
        attempts += 1
        calls.append((subject, body))
        if attempts == 1:
            raise OSError("temporary SMTP failure")

    monkeypatch.setattr(notify, "send_email", sender)

    assert (
        notify.finalize_terminal_notification(
            "no-steering", status="failed", state_path=state
        )
        == "failed"
    )
    assert (
        notify.finalize_terminal_notification(
            "no-steering", status="completed", state_path=state
        )
        == "sent"
    )
    assert len(calls) == 2
    saved = json.loads(state.read_text(encoding="utf-8"))
    assert saved["workflow_status"] == "completed"
    assert saved["delivery_status"] == "sent"


def test_sent_failed_workflow_then_completed_resume_updates_workflow_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, str]] = []
    state = tmp_path / "generation" / "notification.json"

    def sender(subject: str, body: str, **_: object) -> None:
        calls.append((subject, body))

    monkeypatch.setattr(notify, "send_email", sender)

    assert (
        notify.finalize_terminal_notification(
            "no-steering", status="failed", state_path=state
        )
        == "sent"
    )
    assert (
        notify.finalize_terminal_notification(
            "no-steering", status="completed", state_path=state
        )
        == "sent"
    )

    saved = json.loads(state.read_text(encoding="utf-8"))
    assert saved["workflow_status"] == "completed"
    assert saved["delivery_status"] == "sent"
    assert len(calls) == 1


def test_repeated_terminal_finalization_does_not_duplicate_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, str]] = []
    state = tmp_path / "notification.json"

    _terminal_summary(monkeypatch, state, status="completed", calls=calls)
    _terminal_summary(monkeypatch, state, status="completed", calls=calls)

    assert len(calls) == 1


def test_resume_after_sent_state_does_not_send_again(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, str]] = []
    state = tmp_path / "notification.json"
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(
        json.dumps(
            {
                "task": "benchmark-investigation",
                "status": "sent",
                "notification_attempted": True,
            }
        ),
        encoding="utf-8",
    )

    _terminal_summary(monkeypatch, state, status="completed", calls=calls)

    assert calls == []


def test_ambiguous_claimed_state_suppresses_duplicate_send(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, str]] = []
    state = tmp_path / "notification.json"
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(
        json.dumps(
            {
                "task": "benchmark-investigation",
                "status": "claimed",
                "notification_attempted": True,
            }
        ),
        encoding="utf-8",
    )

    _terminal_summary(monkeypatch, state, status="completed", calls=calls)

    assert calls == []


def test_initial_claim_fsync_failure_leaves_no_torn_claim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state = tmp_path / "notification.json"
    real_fsync = notify.os.fsync
    failed = False

    def fail_once(fd: int) -> None:
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("simulated crash during claim fsync")
        real_fsync(fd)

    monkeypatch.setattr(notify.os, "fsync", fail_once)
    with pytest.raises(OSError, match="claim fsync"):
        notify._claim_notification_state(state, {"status": "claimed"})

    assert not state.exists()
    assert notify._claim_notification_state(state, {"status": "claimed"}) is True
    assert json.loads(state.read_text(encoding="utf-8"))["status"] == "claimed"


@pytest.mark.parametrize(
    "contents",
    ['{"status":', "[]", '{"event": "completed"}'],
)
def test_malformed_or_ambiguous_claim_state_never_resends(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, contents: str
) -> None:
    calls: list[tuple[str, str]] = []
    state = tmp_path / "notification.json"
    state.write_text(contents, encoding="utf-8")

    _terminal_summary(monkeypatch, state, status="completed", calls=calls)

    assert calls == []
    assert state.read_text(encoding="utf-8") == contents


def test_concurrent_failed_retries_only_one_sends(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, str]] = []
    state = tmp_path / "notification.json"
    state.write_text(json.dumps({"status": "failed"}), encoding="utf-8")
    first_send_started = threading.Event()
    release_first_send = threading.Event()
    calls_lock = threading.Lock()

    def sender(subject: str, body: str, **_: object) -> None:
        with calls_lock:
            calls.append((subject, body))
        first_send_started.set()
        assert release_first_send.wait(timeout=5)

    monkeypatch.setattr(notify, "send_email", sender)
    results: list[str] = []

    def retry() -> None:
        results.append(
            notify.finalize_terminal_notification(
                "benchmark-investigation",
                status="completed",
                state_path=state,
            )
        )

    first = threading.Thread(target=retry)
    second = threading.Thread(target=retry)
    first.start()
    assert first_send_started.wait(timeout=5)
    second.start()
    second.join(timeout=5)
    release_first_send.set()
    first.join(timeout=5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert sorted(results) == ["claimed", "sent"]
    assert len(calls) == 1


def test_terminal_state_separates_workflow_and_delivery_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state = tmp_path / "notification.json"
    calls: list[tuple[str, str]] = []

    def sender(subject: str, body: str, **_: object) -> None:
        calls.append((subject, body))

    monkeypatch.setattr(notify, "send_email", sender)

    assert (
        notify.finalize_terminal_notification(
            "workflow", status="completed", state_path=state
        )
        == "sent"
    )
    saved = json.loads(state.read_text(encoding="utf-8"))
    assert saved["workflow_status"] == "completed"
    assert saved["delivery_status"] == "sent"
    assert saved["status"] == "sent"
    assert len(calls) == 1


def test_known_send_return_false_is_retryable_without_changing_workflow_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state = tmp_path / "notification.json"
    attempts = 0

    def sender(*_: object, **__: object) -> bool:
        nonlocal attempts
        attempts += 1
        return attempts > 1

    monkeypatch.setattr(notify, "send_email", sender)

    assert (
        notify.finalize_terminal_notification(
            "workflow", status="completed", state_path=state
        )
        == "failed"
    )
    first = json.loads(state.read_text(encoding="utf-8"))
    assert first["workflow_status"] == "completed"
    assert first["delivery_status"] == "failed"

    assert (
        notify.finalize_terminal_notification(
            "workflow", status="completed", state_path=state
        )
        == "sent"
    )
    second = json.loads(state.read_text(encoding="utf-8"))
    assert second["workflow_status"] == "completed"
    assert second["delivery_status"] == "sent"
    assert attempts == 2


def test_ambiguous_post_smtp_outcome_is_not_retried(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state = tmp_path / "notification.json"
    attempts = 0

    def sender(*_: object, **__: object) -> None:
        nonlocal attempts
        attempts += 1
        raise notify.AmbiguousDeliveryError("SMTP acceptance is unknown")

    monkeypatch.setattr(notify, "send_email", sender)

    assert (
        notify.finalize_terminal_notification(
            "workflow", status="completed", state_path=state
        )
        == "claimed"
    )
    saved = json.loads(state.read_text(encoding="utf-8"))
    assert saved["workflow_status"] == "completed"
    assert saved["delivery_status"] == "claimed"

    assert (
        notify.finalize_terminal_notification(
            "workflow", status="completed", state_path=state
        )
        == "claimed"
    )
    assert attempts == 1


def test_interrupted_claim_before_atomic_link_can_be_recovered(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state = tmp_path / "notification.json"
    real_link = notify.os.link
    interrupted = True

    def link_once(
        source: str | bytes,
        destination: str | bytes,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> None:
        nonlocal interrupted
        if interrupted:
            interrupted = False
            raise OSError("interrupted claim")
        real_link(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr(notify.os, "link", link_once)
    with pytest.raises(OSError, match="interrupted claim"):
        notify._claim_notification_state(state, {"delivery_status": "claimed"})
    assert not state.exists()
    assert (
        notify._claim_notification_state(state, {"delivery_status": "claimed"}) is True
    )


def test_record_terminal_workflow_does_not_replace_malformed_state(
    tmp_path: Path,
) -> None:
    state = tmp_path / "notification.json"
    contents = '{"workflow_status":'
    state.write_text(contents, encoding="utf-8")

    with pytest.raises(RuntimeError, match="ambiguous"):
        notify.record_terminal_workflow(
            "workflow", status="completed", state_path=state
        )

    assert state.read_text(encoding="utf-8") == contents


def test_unknown_delivery_status_is_ambiguous_and_never_resends(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state = tmp_path / "notification.json"
    original = {
        "task": "workflow",
        "workflow_status": "completed",
        "delivery_status": "bogus",
        "status": "bogus",
    }
    state.write_text(json.dumps(original), encoding="utf-8")
    calls: list[object] = []
    monkeypatch.setattr(
        notify,
        "send_email",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    assert (
        notify.finalize_terminal_notification(
            "workflow", status="completed", state_path=state
        )
        == "claimed"
    )
    assert calls == []
    assert json.loads(state.read_text(encoding="utf-8")) == original
