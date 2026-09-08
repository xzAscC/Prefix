"""Tests for the Gmail notify component."""

from __future__ import annotations

import smtplib
from pathlib import Path

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
            self.ops: list[tuple] = []
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
    FakeSMTP.instances = instances  # type: ignore[attr-defined]
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
