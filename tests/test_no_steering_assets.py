from __future__ import annotations

import importlib
from types import ModuleType

import pytest


def pipeline() -> ModuleType:
    try:
        return importlib.import_module("prefix.no_steering")
    except ModuleNotFoundError as error:
        pytest.fail(f"missing standalone no-steering production module: {error}")


def test_dataset_scopes_have_required_counts() -> None:
    module = pipeline()
    assert module.DATASET_SCOPES == {
        "harmbench": {"split": "all", "count": 400},
        "mmlu_pro": {"split": "test", "count": 12032},
        "math500": {"split": "test", "count": 500},
    }


def test_failed_sample_does_not_block_later_samples() -> None:
    module = pipeline()
    seen: list[str] = []

    def evaluate(sample: dict[str, str]) -> dict[str, str]:
        seen.append(sample["id"])
        if sample["id"] == "failed":
            raise RuntimeError("synthetic sample failure")
        return {"id": sample["id"], "status": "ok"}

    rows = module.evaluate_samples(
        [{"id": "first"}, {"id": "failed"}, {"id": "later"}],
        evaluate=evaluate,
    )
    assert seen == ["first", "failed", "later"]
    assert [row["id"] for row in rows] == ["first", "failed", "later"]
    assert rows[1]["status"] == "error"


def test_completed_ids_are_skipped() -> None:
    module = pipeline()
    seen: list[str] = []
    rows = module.evaluate_samples(
        [{"id": "done"}, {"id": "new"}],
        completed_ids={"done"},
        evaluate=lambda sample: seen.append(sample["id"]) or {"id": sample["id"]},
    )
    assert seen == ["new"]
    assert [row["id"] for row in rows] == ["new"]
