from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_exp1_v2.py"


@pytest.fixture
def exp1_v2() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_exp1_v2", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def protocol() -> dict[str, Any]:
    return yaml.safe_load(
        (Path(__file__).parents[1] / "configs" / "exp1_v2.yaml").read_text()
    )


def _harmbench_rows() -> list[dict[str, Any]]:
    return [
        {
            "id": f"hb-{category}-{index:03d}",
            "category": category,
            "prompt": f"prompt {category} {index}",
            "source": "harmbench",
            "revision": "harmbench-test-revision",
        }
        for category in (
            "chemical_biological",
            "cybercrime_intrusion",
            "harassment_bullying",
            "illegal",
            "misinformation_disinformation",
            "harmful",
            "copyright",
        )
        for index in range(40)
    ]


def _llm_lat_rows() -> list[dict[str, Any]]:
    return [
        {
            "id": f"{label}-{index:03d}",
            "class": label,
            "prompt": f"{label} prompt {index}",
            "source": "llm-lat",
            "revision": "llm-lat-test-revision",
        }
        for label in ("harmful", "harmless")
        for index in range(180)
    ]


def _canonical_llm_lat_rows() -> list[dict[str, Any]]:
    return [
        {
            "id": f"{label}-{index:03d}",
            "class": label,
            "prompt": f"{label} prompt {index}",
            "source": "llm-lat",
            "revision": "llm-lat-canonical-pinned",
        }
        for label in ("harmful", "benign")
        for index in range(180)
    ]


class _Loader:
    def __init__(
        self, harmbench: list[dict[str, Any]], llm_lat: list[dict[str, Any]]
    ) -> None:
        self.harmbench = harmbench
        self.llm_lat = llm_lat
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def load_harmbench(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append(("harmbench", kwargs))
        return list(self.harmbench)

    def load_llm_lat(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append(("llm_lat", kwargs))
        return list(self.llm_lat)


class _ForbiddenFactory:
    def __init__(self, label: str) -> None:
        self.label = label
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> object:
        self.calls.append((args, kwargs))
        raise AssertionError(f"{self.label} must not be constructed during preflight")


def _prepare(
    exp1_v2: ModuleType,
    protocol: dict[str, Any],
    *,
    tmp_path: Path,
    loader: _Loader,
    **overrides: Any,
) -> Any:
    prepare = getattr(exp1_v2, "prepare_exp1_v2_preflight")
    inputs = tmp_path / "inputs"
    return prepare(
        protocol,
        cache_root=tmp_path / "cache",
        inputs_root=inputs,
        manifest_path=inputs / "manifest.json",
        marker_path=inputs / "preflight.ready",
        harmbench_loader=loader.load_harmbench,
        llm_lat_loader=loader.load_llm_lat,
        model_factory=_ForbiddenFactory("model"),
        provider_factory=_ForbiddenFactory("provider"),
        **overrides,
    )


def test_preflight_is_cpu_only_and_validates_offline_cache_identity(
    exp1_v2: ModuleType,
    protocol: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = _Loader(_harmbench_rows(), _llm_lat_rows())
    cuda_access = False

    class ForbiddenCuda:
        def __getattr__(self, name: str) -> Any:
            nonlocal cuda_access
            cuda_access = True
            raise AssertionError(f"CUDA access: {name}")

    monkeypatch.setattr("torch.cuda", ForbiddenCuda())
    result = _prepare(exp1_v2, protocol, tmp_path=tmp_path, loader=loader)

    assert not cuda_access
    assert result["cache"]["offline"] is True
    assert result["cache"]["identity"] == {
        "harmbench": "harmbench-test-revision",
        "llm_lat": {
            "benign": "llm-lat-test-revision",
            "harmful": "llm-lat-test-revision",
        },
    }


def test_preflight_accepts_both_pinned_production_llm_lat_revisions(
    exp1_v2: ModuleType,
    protocol: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from prefix import data

    monkeypatch.setattr(
        data,
        "load_llm_lat",
        lambda source, n, cache_dir=None: [
            f"{source} prompt {index}" for index in range(n)
        ],
    )
    rows = exp1_v2._cli_llm_lat_loader(
        dataset="llm-lat", n=180, cache_dir=tmp_path / "cache", offline=True
    )
    loader = _Loader(_harmbench_rows(), rows)
    result = _prepare(exp1_v2, protocol, tmp_path=tmp_path, loader=loader)
    calls_after_first_run = len(loader.calls)
    resumed = _prepare(exp1_v2, protocol, tmp_path=tmp_path, loader=loader)

    assert resumed == result
    assert len(loader.calls) == calls_after_first_run
    assert result["cache"]["identity"]["llm_lat"] == {
        "benign": "799694027732ac7b5633639690a2ea8ed8597f3e",
        "harmful": "8bfba31bc6d93a5b71808fee5275ef4b6330ed91",
    }


def test_prepare_rejects_missing_or_wrong_model_snapshot_before_dataset_load(
    exp1_v2: ModuleType,
    protocol: dict[str, Any],
    tmp_path: Path,
) -> None:
    loader = _Loader(_harmbench_rows(), _llm_lat_rows())
    calls: list[dict[str, Any]] = []

    def missing_or_wrong_snapshot(*args: Any, **kwargs: Any) -> object:
        calls.append({"args": args, **kwargs})
        raise FileNotFoundError("exact pinned revision snapshot is missing")

    with pytest.raises(FileNotFoundError, match="exact pinned revision"):
        _prepare(
            exp1_v2,
            protocol,
            tmp_path=tmp_path,
            loader=loader,
            model_snapshot_checker=missing_or_wrong_snapshot,
        )
    assert len(calls) == 1
    assert calls[0]["args"] == (("Qwen/Qwen3-4B",),)
    assert calls[0]["load"] is False
    assert loader.calls == []


def test_preflight_selects_exactly_154_harmbench_rows_with_stable_provenance(
    exp1_v2: ModuleType, protocol: dict[str, Any], tmp_path: Path
) -> None:
    loader = _Loader(_harmbench_rows(), _llm_lat_rows())
    first = _prepare(exp1_v2, protocol, tmp_path=tmp_path / "first", loader=loader)
    second_loader = _Loader(list(reversed(_harmbench_rows())), _llm_lat_rows())
    second = _prepare(
        exp1_v2, protocol, tmp_path=tmp_path / "second", loader=second_loader
    )

    first_rows = first["harmbench"]
    second_rows = second["harmbench"]
    assert len(first_rows) == len(second_rows) == 154
    assert [row["id"] for row in first_rows] == [row["id"] for row in second_rows]
    assert {row["category"] for row in first_rows} == {
        "chemical_biological",
        "cybercrime_intrusion",
        "harassment_bullying",
        "illegal",
        "misinformation_disinformation",
        "harmful",
        "copyright",
    }
    assert all(
        sum(row["category"] == category for row in first_rows) == 22
        for category in {
            "chemical_biological",
            "cybercrime_intrusion",
            "harassment_bullying",
            "illegal",
            "misinformation_disinformation",
            "harmful",
            "copyright",
        }
    )
    assert all(row["provenance"]["source"] == "harmbench" for row in first_rows)


def test_fresh_preflight_rejects_harmbench_selection_count_drift(
    exp1_v2: ModuleType, protocol: dict[str, Any], tmp_path: Path
) -> None:
    extra_category = [
        {
            "id": f"hb-extra-{index:03d}",
            "category": "extra",
            "prompt": f"prompt extra {index}",
            "source": "harmbench",
            "revision": "harmbench-test-revision",
        }
        for index in range(40)
    ]

    with pytest.raises(ValueError, match="categor|154"):
        _prepare(
            exp1_v2,
            protocol,
            tmp_path=tmp_path,
            loader=_Loader(_harmbench_rows() + extra_category, _llm_lat_rows()),
        )

    state = json.loads((tmp_path / "inputs" / "preflight.json").read_text())
    assert "harmbench" not in state["completed_units"]
    assert not (tmp_path / "inputs" / "preflight.ready").exists()


@pytest.mark.parametrize("drift", ["replacement", "missing", "extra"])
def test_preflight_rejects_harmbench_category_drift_at_exactly_154(
    drift: str, exp1_v2: ModuleType, protocol: dict[str, Any], tmp_path: Path
) -> None:
    rows = _harmbench_rows()
    if drift == "replacement":
        rows = [
            {**row, "category": "extra"} if row["category"] == "copyright" else row
            for row in rows
        ]
    elif drift == "missing":
        rows = [
            {**row, "category": "chemical_biological"}
            if row["category"] == "copyright"
            else row
            for row in rows
        ]
    else:
        rows[0] = {**rows[0], "category": "extra"}

    with pytest.raises(ValueError, match="categor"):
        _prepare(
            exp1_v2,
            protocol,
            tmp_path=tmp_path,
            loader=_Loader(rows, _llm_lat_rows()),
        )

    state = json.loads((tmp_path / "inputs" / "preflight.json").read_text())
    assert state["status"] != "complete"
    assert "harmbench" not in state["completed_units"]
    assert not (tmp_path / "inputs" / "preflight.ready").exists()


def test_preflight_splits_each_llm_lat_class_into_100_fit_and_50_holdout(
    exp1_v2: ModuleType, protocol: dict[str, Any], tmp_path: Path
) -> None:
    result = _prepare(
        exp1_v2,
        protocol,
        tmp_path=tmp_path,
        loader=_Loader(_harmbench_rows(), _llm_lat_rows()),
    )

    for label in ("harmful", "harmless"):
        split = result["llm_lat"][label]
        assert len(split["fit"]) == 100
        assert len(split["holdout"]) == 50
        assert set(row["id"] for row in split["fit"]).isdisjoint(
            row["id"] for row in split["holdout"]
        )
        assert all(
            row["provenance"]["source"] == "llm-lat"
            for row in split["fit"] + split["holdout"]
        )


def test_preflight_llm_lat_rows_carry_authoritative_pinned_revisions(
    exp1_v2: ModuleType, protocol: dict[str, Any], tmp_path: Path
) -> None:
    rows = _llm_lat_rows()
    for row in rows:
        row.pop("revision")
    result = _prepare(
        exp1_v2,
        protocol,
        tmp_path=tmp_path,
        loader=_Loader(_harmbench_rows(), rows),
    )

    revisions = {
        row["provenance"]["revision"]
        for label in ("harmless", "harmful")
        for split in ("fit", "holdout")
        for row in result["llm_lat"][label][split]
    }
    assert revisions == {
        "799694027732ac7b5633639690a2ea8ed8597f3e",
        "8bfba31bc6d93a5b71808fee5275ef4b6330ed91",
    }


def test_preflight_has_exactly_benign_and_harmful_canonical_classes(
    exp1_v2: ModuleType, protocol: dict[str, Any], tmp_path: Path
) -> None:
    result = _prepare(
        exp1_v2,
        protocol,
        tmp_path=tmp_path,
        loader=_Loader(_harmbench_rows(), _canonical_llm_lat_rows()),
    )

    assert set(result["llm_lat"]) == {"benign", "harmful"}


def test_preflight_selected_harmbench_rows_are_plain_json_rows_with_metadata(
    exp1_v2: ModuleType, protocol: dict[str, Any], tmp_path: Path
) -> None:
    rows = [
        {
            "id": f"hb-{category}-{index:03d}",
            "behavior": f"behavior {category} {index}",
            "category": category,
            "BehaviorID": f"HB{index:03d}",
            "ContextString": "context",
            "source": "harmbench",
            "revision": "harmbench-pinned",
        }
        for category in (
            "chemical_biological",
            "cybercrime_intrusion",
            "harassment_bullying",
            "illegal",
            "misinformation_disinformation",
            "harmful",
            "copyright",
        )
        for index in range(22)
    ]
    result = _prepare(
        exp1_v2,
        protocol,
        tmp_path=tmp_path,
        loader=_Loader(rows, _llm_lat_rows()),
    )

    selected = result["harmbench"]
    assert all(type(row) is dict for row in selected)
    json.dumps(selected)
    assert all(
        {"BehaviorID", "category", "ContextString", "source", "revision"} <= set(row)
        for row in selected
    )


def test_preflight_normalizes_pinned_cache_rows_and_preserves_selection_ids(
    exp1_v2: ModuleType,
    protocol: dict[str, Any],
    tmp_path: Path,
    harmbench_records: list[dict[str, Any]],
) -> None:
    pinned_rows = harmbench_records
    result = _prepare(
        exp1_v2,
        protocol,
        tmp_path=tmp_path,
        loader=_Loader(pinned_rows, _llm_lat_rows()),
    )

    selected = result["harmbench"]
    selected_ids = [row["id"] for row in selected]
    assert len(selected) == 154
    assert all(isinstance(prompt_id, str) and prompt_id for prompt_id in selected_ids)
    assert len(set(selected_ids)) == 154
    manifest = json.loads((tmp_path / "inputs" / "manifest.json").read_text())
    assert len(manifest["expected_generation_ids"]) == 12 * 154
    assert all("/None" not in value for value in manifest["expected_generation_ids"])


def test_preflight_persists_completed_units_atomically_before_marker(
    exp1_v2: ModuleType, protocol: dict[str, Any], tmp_path: Path
) -> None:
    loader = _Loader(_harmbench_rows(), _llm_lat_rows())
    _prepare(exp1_v2, protocol, tmp_path=tmp_path, loader=loader)

    preflight_path = tmp_path / "inputs" / "preflight.json"
    marker_path = tmp_path / "inputs" / "preflight.ready"
    assert preflight_path.is_file()
    assert marker_path.is_file()
    assert json.loads(preflight_path.read_text())["status"] == "complete"
    assert not list((tmp_path / "inputs").glob("*.tmp"))


def test_preflight_resumes_matching_prepared_state_without_provider_reload(
    exp1_v2: ModuleType, protocol: dict[str, Any], tmp_path: Path
) -> None:
    loader = _Loader(_harmbench_rows(), _llm_lat_rows())
    _prepare(exp1_v2, protocol, tmp_path=tmp_path, loader=loader)
    calls_after_first_run = len(loader.calls)

    _prepare(exp1_v2, protocol, tmp_path=tmp_path, loader=loader)

    assert len(loader.calls) == calls_after_first_run


def test_preflight_resume_skips_completed_unit_materialization(
    exp1_v2: ModuleType,
    protocol: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = _Loader(_harmbench_rows(), _llm_lat_rows())
    _prepare(exp1_v2, protocol, tmp_path=tmp_path, loader=loader)
    state_path = tmp_path / "inputs" / "preflight.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["status"] = "in_progress"
    state.pop("result")
    state_path.write_text(json.dumps(state), encoding="utf-8")
    (tmp_path / "inputs" / "preflight.ready").unlink()

    import prefix.data

    monkeypatch.setattr(
        prefix.data,
        "harmbench_selection",
        lambda *_args, **_kwargs: pytest.fail("completed HarmBench unit recomputed"),
    )
    result = _prepare(
        exp1_v2,
        protocol,
        tmp_path=tmp_path,
        loader=_Loader(_harmbench_rows(), _llm_lat_rows()),
    )

    assert len(result["harmbench"]) == 154


def test_completed_preflight_repairs_missing_marker_without_reloading(
    exp1_v2: ModuleType, protocol: dict[str, Any], tmp_path: Path
) -> None:
    loader = _Loader(_harmbench_rows(), _llm_lat_rows())
    expected = _prepare(exp1_v2, protocol, tmp_path=tmp_path, loader=loader)
    (tmp_path / "inputs" / "preflight.ready").unlink()
    calls_after_first_run = len(loader.calls)

    repaired = _prepare(
        exp1_v2,
        protocol,
        tmp_path=tmp_path,
        loader=_Loader([], []),
    )

    assert repaired == expected
    assert (tmp_path / "inputs" / "preflight.ready").is_file()
    assert calls_after_first_run == 2


def test_preflight_rejects_marker_without_complete_state(
    exp1_v2: ModuleType, protocol: dict[str, Any], tmp_path: Path
) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir(parents=True)
    (inputs / "preflight.json").write_text(
        json.dumps({"status": "in_progress"}), encoding="utf-8"
    )
    (inputs / "preflight.ready").write_text("marker", encoding="utf-8")

    with pytest.raises((RuntimeError, ValueError), match="preflight|complete"):
        _prepare(
            exp1_v2,
            protocol,
            tmp_path=tmp_path,
            loader=_Loader(_harmbench_rows(), _llm_lat_rows()),
        )


def test_preflight_fails_closed_on_selected_row_content_drift(
    exp1_v2: ModuleType, protocol: dict[str, Any], tmp_path: Path
) -> None:
    loader = _Loader(_harmbench_rows(), _llm_lat_rows())
    _prepare(exp1_v2, protocol, tmp_path=tmp_path, loader=loader)
    loader.harmbench[0]["prompt"] = "changed content"

    with pytest.raises((RuntimeError, ValueError), match="drift|source|stale"):
        _prepare(exp1_v2, protocol, tmp_path=tmp_path, loader=loader)


def test_preflight_marker_rejects_transformed_llm_lat_fit_tampering(
    exp1_v2: ModuleType, protocol: dict[str, Any], tmp_path: Path
) -> None:
    result = _prepare(
        exp1_v2,
        protocol,
        tmp_path=tmp_path,
        loader=_Loader(_harmbench_rows(), _llm_lat_rows()),
    )
    preflight_path = tmp_path / "inputs" / "preflight.json"
    paths = {
        "marker": tmp_path / "inputs" / "preflight.ready",
        "manifest": tmp_path / "inputs" / "manifest.json",
        "preflight": preflight_path,
        "inputs": tmp_path / "inputs",
    }
    assert exp1_v2._validate_preflight_marker(protocol, paths) == result

    state = json.loads(preflight_path.read_text(encoding="utf-8"))
    state["result"]["llm_lat"]["harmful"]["fit"][0]["prompt"] = "tampered"
    preflight_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises((RuntimeError, ValueError), match="LLM-LAT|split|drift"):
        exp1_v2._validate_preflight_marker(protocol, paths)


def test_manifest_is_canonical_over_reordered_selected_rows(
    protocol: dict[str, Any], exp1_v2: ModuleType, tmp_path: Path
) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text('{"id":"a"}\n{"id":"b"}\n', encoding="utf-8")
    rows = [{"id": "a", "prompt": "a"}, {"id": "b", "prompt": "b"}]
    kwargs = {
        "config": protocol,
        "sources": {"harmbench": source},
        "datasets": {"harmbench": {"id": "harmbench", "revision": "pinned"}},
        "selected_ids": {"harmbench": rows},
    }

    first = exp1_v2.build_exp1_v2_manifest(**kwargs)
    second = exp1_v2.build_exp1_v2_manifest(
        **{**kwargs, "selected_ids": {"harmbench": list(reversed(rows))}}
    )

    assert first == second
    assert first["selected_rows"] == {"harmbench": rows}


@pytest.mark.parametrize(
    "rows",
    [[{"id": ""}], [{"id": "same"}, {"id": "same"}], [{"prompt": "missing"}]],
)
def test_manifest_rejects_missing_or_duplicate_harmbench_ids(
    protocol: dict[str, Any],
    exp1_v2: ModuleType,
    tmp_path: Path,
    rows: list[dict[str, str]],
) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="ID"):
        exp1_v2.build_exp1_v2_manifest(
            protocol,
            sources={"harmbench": source},
            datasets={"harmbench": {"id": "harmbench", "revision": "pinned"}},
            selected_ids={"harmbench": rows},
        )


@pytest.mark.parametrize("drift", ["cache", "source", "manifest"])
def test_preflight_rejects_cache_source_and_manifest_drift(
    drift: str, exp1_v2: ModuleType, protocol: dict[str, Any], tmp_path: Path
) -> None:
    loader = _Loader(_harmbench_rows(), _llm_lat_rows())
    _prepare(exp1_v2, protocol, tmp_path=tmp_path, loader=loader)
    if drift == "cache":
        loader.harmbench[0]["revision"] = "different-cache-revision"
    elif drift == "source":
        loader.llm_lat[0]["source"] = "different-source"
    else:
        protocol = {**protocol, "alpha": 0.031}

    with pytest.raises((ValueError, RuntimeError)):
        _prepare(exp1_v2, protocol, tmp_path=tmp_path, loader=loader)


def test_preflight_never_touches_legacy_exp1_paths(
    exp1_v2: ModuleType, protocol: dict[str, Any], tmp_path: Path
) -> None:
    legacy_root = tmp_path / "legacy-exp1"
    legacy_root.mkdir()
    sentinel = legacy_root / "sentinel"
    sentinel.write_text("untouched")

    _prepare(
        exp1_v2,
        protocol,
        tmp_path=tmp_path / "new-exp1-v2",
        loader=_Loader(_harmbench_rows(), _llm_lat_rows()),
        legacy_root=legacy_root,
    )

    assert sentinel.read_text() == "untouched"
    assert list(legacy_root.iterdir()) == [sentinel]
