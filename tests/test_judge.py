"""Contract tests for the Vertex AI Gemini judge."""

from __future__ import annotations

import json
import importlib
from typing import Any, Callable

import pytest

judge_module = importlib.import_module("prefix.judge")
GeminiJudge = judge_module.GeminiJudge


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
    with pytest.raises(RuntimeError, match="GOOGLE_CLOUD_PROJECT"):
        GeminiJudge(token_provider=lambda: "token", transport=FakeTransport([]))
