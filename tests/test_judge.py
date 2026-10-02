"""Contract tests for the Vertex AI Gemini judge."""

from __future__ import annotations

import json
import importlib
import threading
import time
from typing import Any, Callable, cast

import pytest

judge_module = importlib.import_module("prefix.judge")
GeminiJudge = judge_module.GeminiJudge
JudgeBlocked = judge_module.JudgeBlocked
JudgeParseError = judge_module.JudgeParseError
JudgeRetryableError = judge_module.JudgeRetryableError
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


def test_unparseable_safety_answer_has_explicit_parse_exception() -> None:
    transport = FakeTransport(
        [(200, _response("unclear")), (200, _response("still unclear"))]
    )

    with pytest.raises(
        JudgeParseError, match="Gemini safety judge returned an unparseable label"
    ):
        _judge(transport).judge_safety("request", "response")


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


@pytest.mark.parametrize("status", [401, 403, 404])
def test_provider_http_failures_are_retryable(status: int) -> None:
    with pytest.raises(
        JudgeRetryableError,
        match=f"Gemini request failed with HTTP status {status}",
    ):
        _judge(FakeTransport([(status, {})])).judge_safety("request", "response")


def test_missing_provider_candidate_is_retryable() -> None:
    with pytest.raises(
        JudgeRetryableError, match="Gemini response did not contain candidate text"
    ):
        _judge(FakeTransport([(200, {})])).judge_safety("request", "response")


def test_auth_token_provider_failure_is_retryable() -> None:
    def failing_provider() -> str:
        raise ValueError("credentials expired")

    with pytest.raises(
        JudgeRetryableError, match="Gemini authentication token provider failed"
    ):
        _judge(FakeTransport([]), failing_provider).judge_safety("request", "response")


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


def test_unparseable_math_answer_has_explicit_parse_exception() -> None:
    transport = FakeTransport(
        [(200, _response("not json")), (200, _response("still not json"))]
    )

    with pytest.raises(
        JudgeParseError, match="Gemini math judge returned an unparseable JSON result"
    ):
        _judge(transport).judge_math("answer", "expected")


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


class FakeFallbackJudge:
    def __init__(self, model: str, outcomes: dict[str, object]) -> None:
        self.model = model
        self.outcomes = outcomes
        self.calls: list[tuple[str, str]] = []

    def judge_safety(self, request: str, response: str) -> bool:
        self.calls.append((request, response))
        outcome = self.outcomes[self.model]
        if isinstance(outcome, BaseException):
            raise outcome
        return bool(outcome)


def _legacy_row(identifier: str = "pair-1") -> dict[str, Any]:
    return {
        "id": identifier,
        "status": "blocked",
        "label": None,
        "error": "original provider block",
        "model": "gemini-3.7-flash",
        "attempts": 3,
        "prompt_hash": "original-prompt-hash",
    }


def _run_fallback_judge(
    rows: list[dict[str, Any]],
    outcomes: dict[str, object],
    *,
    completed_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    judges: dict[str, FakeFallbackJudge] = {}

    def factory(model: str) -> FakeFallbackJudge:
        judges[model] = FakeFallbackJudge(model, outcomes)
        return judges[model]

    return cast(
        list[dict[str, Any]],
        judge_module.judge_with_fallback(
            rows,
            primary_model="gemini-3.7-flash",
            fallback_model="gemini-3.5-flash-lite",
            judge_factory=factory,
            completed_ids=completed_ids,
        ),
    )


def test_primary_success_does_not_call_fallback_and_records_provenance() -> None:
    rows = [_legacy_row()]
    original = dict(rows[0])

    result = _run_fallback_judge(
        rows,
        {
            "gemini-3.7-flash": True,
            "gemini-3.5-flash-lite": RuntimeError("fallback must not be called"),
        },
    )[0]

    assert rows[0] == original
    for key, value in original.items():
        assert result[key] == value
    assert result["final_source"] == "primary"
    assert result["disagreement"] is None
    assert result["primary"] == {
        "status": "ok",
        "label": "SAFE",
        "error": None,
        "model": "gemini-3.7-flash",
        "attempts": 1,
        "prompt_hash": result["primary"]["prompt_hash"],
    }
    assert result["fallback"] is None


@pytest.mark.parametrize(
    ("primary_error", "primary_status"),
    [
        (JudgeBlocked("prompt blocked by provider: SAFETY"), "blocked"),
        (JudgeRetryableError("Gemini request failed after 3 attempts"), "error"),
        (
            JudgeParseError("Gemini safety judge returned an unparseable label"),
            "unparseable",
        ),
    ],
    ids=["provider-block", "transport-exhausted", "unparseable"],
)
def test_primary_failure_uses_legacy_fallback_without_overwriting_original(
    primary_error: BaseException,
    primary_status: str,
) -> None:
    rows = [_legacy_row()]
    original = dict(rows[0])

    result = _run_fallback_judge(
        rows,
        {
            "gemini-3.7-flash": primary_error,
            "gemini-3.5-flash-lite": False,
        },
    )[0]

    assert rows[0] == original
    for key, value in original.items():
        assert result[key] == value
    assert result["final_source"] == "fallback"
    assert result["disagreement"] is None
    assert result["primary"]["status"] == primary_status
    assert result["primary"]["label"] is None
    assert result["primary"]["error"] == str(primary_error)
    assert result["primary"]["model"] == "gemini-3.7-flash"
    assert result["primary"]["attempts"] >= 1
    assert isinstance(result["primary"]["prompt_hash"], str)
    assert result["fallback"]["status"] == "ok"
    assert result["fallback"]["label"] == "UNSAFE"
    assert result["fallback"]["error"] is None
    assert result["fallback"]["model"] == "gemini-3.5-flash-lite"
    assert result["fallback"]["attempts"] == 1
    assert isinstance(result["fallback"]["prompt_hash"], str)


def test_both_models_fail_remains_unresolved_and_keeps_both_errors() -> None:
    rows = [_legacy_row()]

    result = _run_fallback_judge(
        rows,
        {
            "gemini-3.7-flash": JudgeBlocked("primary blocked"),
            "gemini-3.5-flash-lite": JudgeParseError("fallback unparseable"),
        },
    )[0]

    assert result["final_source"] == "unresolved"
    assert result["disagreement"] is None
    for key, value in _legacy_row().items():
        assert result[key] == value
    assert result["primary"]["status"] == "blocked"
    assert result["primary"]["error"] == "primary blocked"
    assert result["fallback"]["status"] == "unparseable"
    assert result["fallback"]["error"] == "fallback unparseable"
    assert result["label"] is None


def test_completed_ids_are_idempotent_and_do_not_invoke_any_provider() -> None:
    rows = [_legacy_row("already-done")]

    result = _run_fallback_judge(
        rows,
        {
            "gemini-3.7-flash": RuntimeError("network must not be called"),
            "gemini-3.5-flash-lite": RuntimeError("network must not be called"),
        },
        completed_ids={"already-done"},
    )

    assert result == rows
