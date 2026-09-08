"""Contract tests for the Vertex AI Gemini judge."""

from __future__ import annotations

import json
import importlib
import threading
import time
from typing import Any, Callable

import pytest

judge_module = importlib.import_module("prefix.judge")
GeminiJudge = judge_module.GeminiJudge
JudgeBlocked = judge_module.JudgeBlocked
judge_batch = judge_module.judge_batch


def _response(text: str) -> dict[str, Any]:
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


class FakeTransport:
    def __init__(self, results: list[object]) -> None:
        self.results: list[object] = results
        self.calls: list[tuple[str, dict[str, Any], dict[str, Any]]] = []

    def __call__(
        self, url: str, headers: dict[str, Any], body: dict[str, Any]
    ) -> tuple[int, dict[str, Any]]:
        self.calls.append((url, headers, body))
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError("fake transport result must be a status/payload tuple")
        return result


def _judge(
    transport: FakeTransport,
    token_provider: Callable[[], str] = lambda: "test-token",
) -> GeminiJudge:
    return GeminiJudge(
        project="test-project",
        token_provider=token_provider,
        transport=transport,
    )


def test_request_url_headers_and_body_have_vertex_shape() -> None:
    transport = FakeTransport([(200, _response("LABEL: SAFE"))])
    _judge(transport).judge_safety("harmful request", "benign response")

    url, headers, body = transport.calls[0]
    assert url == (
        "https://aiplatform.googleapis.com/v1/projects/test-project/locations/"
        "global/publishers/google/models/gemini-3.7-flash:generateContent"
    )
    assert headers["Authorization"] == "Bearer test-token"
    assert headers["Content-Type"] == "application/json"
    assert body["contents"][0]["role"] == "user"
    assert isinstance(body["contents"][0]["parts"][0]["text"], str)
    assert body["generationConfig"]["temperature"] == 0.0


def test_safety_uses_last_label_and_maps_safe_and_unsafe() -> None:
    safe = FakeTransport(
        [
            (
                200,
                _response(
                    "The explanation mentions LABEL: UNSAFE as a decoy.\n`LABEL: SAFE`"
                ),
            )
        ]
    )
    unsafe = FakeTransport([(200, _response("LABEL: UNSAFE"))])

    assert _judge(safe).judge_safety("request", "response") is True
    assert _judge(unsafe).judge_safety("request", "response") is False


def test_unparseable_safety_answer_is_reasked() -> None:
    transport = FakeTransport(
        [(200, _response("unclear")), (200, _response("LABEL: SAFE"))]
    )
    assert _judge(transport).judge_safety("request", "response") is True
    assert len(transport.calls) == 2
    assert (
        "Answer with the final line"
        in transport.calls[1][2]["contents"][0]["parts"][0]["text"]
    )


def test_retryable_statuses_and_exceptions_back_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = FakeTransport(
        [(503, {}), RuntimeError("temporary"), (200, _response("LABEL: SAFE"))]
    )
    sleeps: list[float] = []
    monkeypatch.setattr(judge_module.time, "sleep", sleeps.append)

    assert _judge(transport).judge_safety("request", "response") is True
    assert sleeps == [1, 2]
    assert len(transport.calls) == 3


def test_retry_exhaustion_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = FakeTransport([(503, {}), (500, {}), (429, {})])
    monkeypatch.setattr(judge_module.time, "sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        _judge(transport).judge_safety("request", "response")


def test_math_json_parsing() -> None:
    result = {
        "format_boxed": True,
        "format_answer_is": False,
        "answer_correct": True,
    }
    transport = FakeTransport(
        [
            (
                200,
                _response(
                    'Reasoning JSON: {"answer_correct": false}\nJSON: '
                    + json.dumps(result)
                ),
            )
        ]
    )

    assert _judge(transport).judge_math("The answer is \\boxed{42}", "42") == result


def test_unparseable_math_answer_is_reasked() -> None:
    transport = FakeTransport(
        [
            (200, _response("not json")),
            (
                200,
                _response(
                    'JSON: {"format_boxed": false, "format_answer_is": true, "answer_correct": false}'
                ),
            ),
        ]
    )

    assert _judge(transport).judge_math("answer", "expected") == {
        "format_boxed": False,
        "format_answer_is": True,
        "answer_correct": False,
    }
    assert len(transport.calls) == 2


def test_token_provider_is_cached_across_judgments() -> None:
    calls: list[int] = []

    def provider() -> str:
        calls.append(1)
        return "cached-token"

    transport = FakeTransport(
        [(200, _response("LABEL: SAFE")), (200, _response("LABEL: SAFE"))]
    )
    client = _judge(transport, provider)
    client.judge_safety("request", "response")
    client.judge_safety("request", "response")

    assert len(calls) == 1


def test_project_falls_back_to_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "env-project")
    transport = FakeTransport([(200, _response("LABEL: SAFE"))])
    GeminiJudge(token_provider=lambda: "token", transport=transport).judge_safety(
        "request", "response"
    )
    assert "/projects/env-project/" in transport.calls[0][0]


def test_project_without_argument_or_environment_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    # isolate from the developer's real .env (see AGENTS.md rule 10)
    monkeypatch.setattr(judge_module, "ensure_env", lambda: None)
    with pytest.raises(RuntimeError, match="GOOGLE_CLOUD_PROJECT"):
        GeminiJudge(token_provider=lambda: "token", transport=FakeTransport([]))


class FakeBatchJudge:
    def __init__(self, delay: float = 0.1, failing_item: int | None = None) -> None:
        self.delay = delay
        self.failing_item = failing_item
        self._lock = threading.Lock()
        self.active = 0
        self.peak = 0

    def __call__(self, item: int) -> int:
        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            time.sleep(self.delay)
            if item == self.failing_item:
                raise RuntimeError(f"failed item {item}")
            return item * 2
        finally:
            with self._lock:
                self.active -= 1


def test_judge_batch_preserves_order_and_bounds_concurrency() -> None:
    judge = FakeBatchJudge()

    results = judge_batch(judge, range(8), max_workers=4)

    assert results == [0, 2, 4, 6, 8, 10, 12, 14]
    assert judge.peak <= 4


def test_judge_batch_runs_concurrently() -> None:
    judge = FakeBatchJudge()
    started = time.monotonic()

    judge_batch(judge, range(8), max_workers=4)

    assert time.monotonic() - started < 0.35


def test_judge_batch_propagates_item_exception() -> None:
    with pytest.raises(RuntimeError, match="failed item 3"):
        judge_batch(FakeBatchJudge(failing_item=3), range(8), max_workers=4)


def test_judge_batch_rejects_nonpositive_worker_count() -> None:
    with pytest.raises(ValueError, match="max_workers"):
        judge_batch(FakeBatchJudge(), [1], max_workers=0)


def test_request_body_sets_output_budget_and_safety_off() -> None:
    transport = FakeTransport([(200, _response("LABEL: SAFE"))])
    _judge(transport).judge_safety("harmful request", "benign response")
    _, _, body = transport.calls[0]
    assert body["generationConfig"]["maxOutputTokens"] == 8192
    categories = {item["category"] for item in body["safetySettings"]}
    assert categories == {
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
    }
    assert all(item["threshold"] == "BLOCK_NONE" for item in body["safetySettings"])


def test_blocked_prompt_raises_judge_blocked() -> None:
    blocked_payload = {
        "promptFeedback": {"blockReason": "SAFETY"},
        "candidates": [],
    }
    transport = FakeTransport([(200, blocked_payload)])
    with pytest.raises(JudgeBlocked, match="SAFETY"):
        _judge(transport).judge_safety("harmful request", "unsafe response")
