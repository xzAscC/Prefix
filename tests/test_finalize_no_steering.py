from __future__ import annotations

import importlib.util
import hashlib
import json
import threading
from pathlib import Path

import pytest

from prefix.judge import MATH_PROMPT, SAFETY_PROMPT


ROOT = Path(__file__).parents[1]


def _module():
    spec = importlib.util.spec_from_file_location(
        "finalize_no_steering", ROOT / "scripts" / "finalize_no_steering.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _patch_validators(monkeypatch: pytest.MonkeyPatch, module) -> None:
    monkeypatch.setattr(
        module,
        "validate_generation",
        lambda root: {
            "models": 4,
            "records": 4 * 12932,
        },
    )
    monkeypatch.setattr(
        module,
        "validate_scoring",
        lambda root, **kwargs: {
            "models": 4,
            "counts": {"harmbench": 400, "mmlu_pro": 12032, "math500": 500},
        },
    )


def test_finalizer_claims_once_and_sends_one_completion_email(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _module()
    _patch_validators(monkeypatch, module)
    calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        module,
        "send_batch_notification",
        lambda task, event, *, details=None: (
            calls.append((task, event, details or "")) or True
        ),
    )
    state = tmp_path / "checkpoints" / "finalizer.json"

    result = module.finalize(
        generation_root=tmp_path / "generation",
        scoring_root=tmp_path / "scores",
        state_path=state,
        batch_label="nightly",
        log_path=tmp_path / "logs" / "finalizer.log",
    )

    assert result == "sent"
    assert calls == [("no-steering nightly", "completed", calls[0][2])]
    assert "models=4" in calls[0][2]
    assert "harmbench=400" in calls[0][2]
    saved = json.loads(state.read_text(encoding="utf-8"))
    assert saved["status"] == "sent"
    assert saved["notification_attempted"] is True
    assert "finalization" in capsys.readouterr().out


def test_incomplete_artifacts_fail_before_claim_or_email(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    monkeypatch.setattr(
        module,
        "validate_generation",
        lambda root: (_ for _ in ()).throw(ValueError("incomplete generation")),
    )
    calls: list[object] = []
    monkeypatch.setattr(
        module,
        "send_batch_notification",
        lambda task, event, *, details=None: (
            calls.append((task, event, details)) or True
        ),
    )
    state = tmp_path / "state.json"

    with pytest.raises(ValueError, match="incomplete generation"):
        module.finalize(
            generation_root=tmp_path / "generation",
            scoring_root=tmp_path / "scores",
            state_path=state,
        )

    assert calls == []
    assert not state.exists()


def test_preexisting_claimed_state_suppresses_ambiguous_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    _patch_validators(monkeypatch, module)
    calls: list[object] = []
    monkeypatch.setattr(
        module,
        "send_batch_notification",
        lambda task, event, *, details=None: (
            calls.append((task, event, details)) or True
        ),
    )
    state = tmp_path / "state.json"
    kwargs = {
        "generation_root": tmp_path / "generation",
        "scoring_root": tmp_path / "scores",
        "state_path": state,
    }

    assert module.finalize(**kwargs) == "sent"
    state.write_text(
        json.dumps({"status": "claimed", "event": "completed"}),
        encoding="utf-8",
    )
    calls.clear()
    assert module.finalize(**kwargs) == "claimed"
    assert calls == []


def test_notification_failure_is_nonfatal_and_claim_remains_durable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    _patch_validators(monkeypatch, module)

    def fail(*args: object, **kwargs: object) -> None:
        raise OSError("smtp down")

    calls = 0

    def flaky(*args: object, **kwargs: object) -> bool:
        nonlocal calls
        calls += 1
        if calls == 1:
            fail(*args, **kwargs)
        return True

    monkeypatch.setattr(module, "send_batch_notification", flaky)
    state = tmp_path / "state.json"

    assert (
        module.finalize(
            generation_root=tmp_path / "generation",
            scoring_root=tmp_path / "scores",
            state_path=state,
        )
        == "failed"
    )
    saved = json.loads(state.read_text(encoding="utf-8"))
    assert saved["status"] == "failed"
    assert saved["notification_attempted"] is True
    assert (
        module.finalize(
            generation_root=tmp_path / "generation",
            scoring_root=tmp_path / "scores",
            state_path=state,
        )
        == "sent"
    )
    assert calls == 2


def test_notifier_false_result_does_not_mark_delivery_sent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    _patch_validators(monkeypatch, module)
    monkeypatch.setattr(
        module, "send_batch_notification", lambda *args, **kwargs: False
    )
    state = tmp_path / "state.json"

    assert (
        module.finalize(
            generation_root=tmp_path / "generation",
            scoring_root=tmp_path / "scores",
            state_path=state,
        )
        == "failed"
    )
    assert json.loads(state.read_text(encoding="utf-8"))["status"] == "failed"


def test_failed_notification_retries_once_and_sent_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    _patch_validators(monkeypatch, module)
    calls = 0

    def flaky(*args: object, **kwargs: object) -> bool:
        nonlocal calls
        calls += 1
        return calls > 1

    monkeypatch.setattr(module, "send_batch_notification", flaky)
    state = tmp_path / "state.json"
    kwargs = {
        "generation_root": tmp_path / "generation",
        "scoring_root": tmp_path / "scores",
        "state_path": state,
    }
    assert module.finalize(**kwargs) == "failed"
    assert module.finalize(**kwargs) == "sent"
    assert module.finalize(**kwargs) == "sent"
    assert calls == 2


def test_failed_retry_claims_before_smtp_acceptance_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    _patch_validators(monkeypatch, module)
    state = tmp_path / "state.json"
    attempts = 0

    def notifier(*args: object, **kwargs: object) -> bool:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("smtp unavailable")
        assert json.loads(state.read_text(encoding="utf-8"))["status"] == "claimed"
        raise KeyboardInterrupt("crash after SMTP acceptance")

    monkeypatch.setattr(module, "send_batch_notification", notifier)
    kwargs = {
        "generation_root": tmp_path / "generation",
        "scoring_root": tmp_path / "scores",
        "state_path": state,
    }

    assert module.finalize(**kwargs) == "failed"
    with pytest.raises(KeyboardInterrupt, match="SMTP acceptance"):
        module.finalize(**kwargs)
    assert json.loads(state.read_text(encoding="utf-8"))["status"] == "claimed"


def test_concurrent_failed_retries_only_one_can_claim_and_send(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    _patch_validators(monkeypatch, module)
    state = tmp_path / "state.json"
    first_send_started = threading.Event()
    release_first_send = threading.Event()
    calls = 0
    calls_lock = threading.Lock()

    def notifier(*args: object, **kwargs: object) -> bool:
        nonlocal calls
        with calls_lock:
            calls += 1
            call_number = calls
        if call_number == 2:
            first_send_started.set()
            assert release_first_send.wait(timeout=5)
        return call_number > 1

    monkeypatch.setattr(module, "send_batch_notification", notifier)
    kwargs = {
        "generation_root": tmp_path / "generation",
        "scoring_root": tmp_path / "scores",
        "state_path": state,
    }
    assert module.finalize(**kwargs) == "failed"

    results: list[str] = []

    def retry() -> None:
        results.append(module.finalize(**kwargs))

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
    assert calls == 2


def test_main_exits_nonzero_when_finalize_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    monkeypatch.setattr(module, "finalize", lambda **kwargs: "failed")

    with pytest.raises(SystemExit) as error:
        module.main(
            [
                "--generation-root",
                str(tmp_path / "generation"),
                "--scoring-root",
                str(tmp_path / "scores"),
                "--state-path",
                str(tmp_path / "state.json"),
                "--log-path",
                str(tmp_path / "finalizer.log"),
            ]
        )

    assert error.value.code != 0


def test_state_writes_fsync_file_and_parent_and_reject_symlinked_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    fsync_calls: list[int] = []
    real_fsync = module.os.fsync
    monkeypatch.setattr(
        module.os, "fsync", lambda fd: (fsync_calls.append(fd), real_fsync(fd))[1]
    )
    state = tmp_path / "nested" / "state.json"
    assert module._claim(state, {"status": "claimed"}) is True
    module._atomic_write(state, {"status": "sent"})
    assert len(fsync_calls) >= 4

    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        module._atomic_write(link / "state.json", {"status": "claimed"})


def test_validate_scoring_rejects_stale_or_unbound_score_manifest(
    tmp_path: Path,
) -> None:
    module = _module()
    model_id = "Qwen/Qwen3-4B"
    model_root = tmp_path / module.model_spec(model_id).slug
    for benchmark, scope in module.DATASET_SCOPES.items():
        path = model_root / benchmark / "scores.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(
                json.dumps(
                    {
                        "id": f"{benchmark}-{index}",
                        "benchmark": benchmark,
                        "status": "ok",
                    }
                )
                for index in range(int(scope["count"]))
            )
            + "\n"
        )
    (model_root / "scoring_manifest.json").write_text(
        json.dumps({"judge_model": "wrong-model"})
    )
    with pytest.raises(ValueError, match="judge model"):
        module.validate_scoring(tmp_path)


def _bound_scoring_fixture(tmp_path: Path):
    module = _module()
    model_id = "Qwen/Qwen3-4B"
    slug = module.model_spec(model_id).slug
    setattr(module, "MODEL_MATRIX", (model_id,))
    setattr(module, "DATASET_SCOPES", {"mmlu_pro": {"count": 1, "split": "test"}})
    generation_a = tmp_path / "generation-a" / "checkpoints"
    generation_b = tmp_path / "generation-b" / "checkpoints"
    scoring = tmp_path / "scores"
    response = generation_b / slug / "mmlu_pro" / "responses.jsonl"
    response.parent.mkdir(parents=True)
    response.write_text(
        json.dumps({"id": "m0", "benchmark": "mmlu_pro"}) + "\n",
        encoding="utf-8",
    )
    score = scoring / slug / "mmlu_pro" / "scores.jsonl"
    score.parent.mkdir(parents=True)
    score.write_text(
        json.dumps(
            {"id": "m0", "benchmark": "mmlu_pro", "status": "ok", "correct": True}
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "model_id": model_id,
        "model_slug": slug,
        "judge_model": "gemini-3.5-flash-lite",
        "response_root": str(generation_b.resolve()),
        "rubric_version": "gemini-rubric-v1",
        "prompt_sha256": {
            "safety": hashlib.sha256(SAFETY_PROMPT.encode()).hexdigest(),
            "math": hashlib.sha256(MATH_PROMPT.encode()).hexdigest(),
        },
        "response_schema": "no-steering-response-v1",
        "settings": {"temperature": 0.0, "max_output_tokens": 8192},
        "response_files": {
            "mmlu_pro": {
                "ids": ["m0"],
                "path": str(response.resolve()),
                "content_sha256": hashlib.sha256(response.read_bytes()).hexdigest(),
            }
        },
    }
    manifest_path = scoring / slug / "scoring_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    class Runner:
        def validate_score_completeness(self, rows: object, *, limited: bool) -> None:
            return None

        @staticmethod
        def _retryable_score_error(error: str) -> bool:
            return False

    setattr(module, "_runner", lambda: Runner())
    return module, generation_a, generation_b, scoring, manifest_path


def test_validate_scoring_rejects_cross_run_response_root(
    tmp_path: Path,
) -> None:
    module, generation_a, _generation_b, scoring, _manifest = _bound_scoring_fixture(
        tmp_path
    )
    with pytest.raises(ValueError, match="response root"):
        module.validate_scoring(scoring, generation_checkpoint_root=generation_a)


def test_validate_scoring_hashes_response_file_in_bounded_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, _generation_a, generation_b, scoring, _manifest = _bound_scoring_fixture(
        tmp_path
    )
    response = (
        generation_b
        / module.model_spec("Qwen/Qwen3-4B").slug
        / "mmlu_pro"
        / "responses.jsonl"
    )
    real_open = Path.open
    read_sizes: list[int] = []

    class TrackingReader:
        def __init__(self, handle) -> None:
            self.handle = handle

        def __enter__(self):
            self.handle.__enter__()
            return self

        def __exit__(self, *args: object) -> None:
            self.handle.__exit__(*args)

        def read(self, size: int = -1) -> bytes:
            read_sizes.append(size)
            return self.handle.read(size)

    def tracking_open(path: Path, mode: str = "r", *args, **kwargs):
        handle = real_open(path, mode, *args, **kwargs)
        if path == response and "b" in mode:
            return TrackingReader(handle)
        return handle

    monkeypatch.setattr(Path, "open", tracking_open)

    assert module.validate_scoring(scoring, generation_checkpoint_root=generation_b)
    assert read_sizes
    assert all(0 < size <= 1024 * 1024 for size in read_sizes)


def test_validate_scoring_collects_response_ids_without_materializing_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, _generation_a, generation_b, scoring, _manifest = _bound_scoring_fixture(
        tmp_path
    )
    response = (
        generation_b
        / module.model_spec("Qwen/Qwen3-4B").slug
        / "mmlu_pro"
        / "responses.jsonl"
    )
    real_read_jsonl = module._read_jsonl

    def reject_response_materialization(path: Path):
        if path == response:
            raise AssertionError("response rows must be iterated line by line")
        return real_read_jsonl(path)

    monkeypatch.setattr(module, "_read_jsonl", reject_response_materialization)

    assert module.validate_scoring(scoring, generation_checkpoint_root=generation_b)


def test_validate_scoring_rejects_changed_bound_response_path(
    tmp_path: Path,
) -> None:
    module, _generation_a, generation_b, scoring, manifest_path = (
        _bound_scoring_fixture(tmp_path)
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["response_files"]["mmlu_pro"]["path"] = str(
        (generation_b / "wrong" / "mmlu_pro" / "responses.jsonl").resolve()
    )
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="response path"):
        module.validate_scoring(scoring, generation_checkpoint_root=generation_b)


def test_validate_scoring_rejects_unexpected_error_status(
    tmp_path: Path,
) -> None:
    module, _generation_a, generation_b, scoring, manifest_path = (
        _bound_scoring_fixture(tmp_path)
    )
    score = (
        scoring / module.model_spec("Qwen/Qwen3-4B").slug / "mmlu_pro" / "scores.jsonl"
    )
    score.write_text(
        json.dumps(
            {
                "id": "m0",
                "benchmark": "mmlu_pro",
                "status": "error",
                "error": "permanent parser failure",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="error"):
        module.validate_scoring(scoring, generation_checkpoint_root=generation_b)


@pytest.mark.parametrize(
    "status",
    [
        "blocked",
        "unparseable",
        "error",
        "retryable",
        "incomplete",
        "unknown",
        None,
    ],
)
def test_finalize_rejects_non_ok_scoring_status_before_claim_or_notification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str | None
) -> None:
    module, _generation_a, generation_b, scoring, _manifest = _bound_scoring_fixture(
        tmp_path
    )
    monkeypatch.setattr(
        module,
        "validate_generation",
        lambda root: {"models": 1, "records": 1},
    )
    calls: list[object] = []
    monkeypatch.setattr(
        module,
        "send_batch_notification",
        lambda *args, **kwargs: calls.append((args, kwargs)) or True,
    )
    score = (
        scoring / module.model_spec("Qwen/Qwen3-4B").slug / "mmlu_pro" / "scores.jsonl"
    )
    row = {"id": "m0", "benchmark": "mmlu_pro", "correct": True}
    if status is not None:
        row["status"] = status
    score.write_text(json.dumps(row) + "\n", encoding="utf-8")
    state = tmp_path / "state.json"

    with pytest.raises(ValueError, match="status"):
        module.finalize(
            generation_root=generation_b.parent,
            scoring_root=scoring,
            state_path=state,
        )

    assert calls == []
    assert not state.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("judge_model", "wrong-judge"),
        ("rubric_version", "wrong-rubric"),
        ("prompt_sha256", {"safety": "0" * 64, "math": "0" * 64}),
        ("settings", {"temperature": 1.0, "max_output_tokens": 8192}),
    ],
)
def test_validate_scoring_rejects_contract_mismatch(
    tmp_path: Path, field: str, value: object
) -> None:
    module, _generation_a, generation_b, scoring, manifest_path = (
        _bound_scoring_fixture(tmp_path)
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="judge|rubric|prompt|settings"):
        module.validate_scoring(scoring, generation_checkpoint_root=generation_b)


def test_local_scoring_script_finalizes_only_after_all_models_succeed() -> None:
    script = (ROOT / "scripts" / "score_no_steering_local.sh").read_text(
        encoding="utf-8"
    )
    assert script.count("run_no_steering.py") == 1
    assert "finalize_no_steering.py" in script
    assert script.index("finalize_no_steering.py") > script.index("done")
    assert "--generation-root" in script
    assert "--scoring-root" in script
