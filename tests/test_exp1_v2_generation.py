from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
import yaml

from prefix import runner


ROOT = Path(__file__).parents[1]
SCRIPT_PATH = ROOT / "scripts" / "run_exp1_v2.py"
CONFIG_PATH = ROOT / "configs" / "exp1_v2.yaml"
LAYERS = [1, 5, 8, 12, 16, 19, 20, 23, 27, 30, 34]
REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"


@pytest.fixture(scope="module")
def exp1_v2() -> Any:
    spec = importlib.util.spec_from_file_location("run_exp1_v2_generation", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def config() -> dict[str, object]:
    with CONFIG_PATH.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    assert isinstance(value, dict)
    return value


def _rows(prefix: str, count: int) -> list[dict[str, str]]:
    return [
        {"id": f"{prefix}-{index}", "prompt": f"{prefix} prompt {index}"}
        for index in range(count)
    ]


def _direction_metadata() -> dict[str, object]:
    return {
        "model_revision": REVISION,
        "layers": LAYERS,
        "fit_ids": {
            "benign": [f"benign-{i}" for i in range(100)],
            "harmful": [f"harmful-{i}" for i in range(100)],
        },
        "holdout_ids": {
            "benign": [f"benign-{i}" for i in range(100, 150)],
            "harmful": [f"harmful-{i}" for i in range(100, 150)],
        },
    }


def _direction_payload() -> dict[str, object]:
    return {
        str(layer): {"direction": [float(layer), 1.0], "mean_norm": 2.0}
        for layer in LAYERS
    }


def _direction_rows() -> tuple[
    dict[str, list[dict[str, str]]], dict[str, list[dict[str, str]]]
]:
    def rows(label: str, start: int, count: int) -> list[dict[str, str]]:
        return [
            {"id": f"{label}-{index}", "prompt": f"{label} prompt {index}"}
            for index in range(start, start + count)
        ]

    return (
        {
            "benign": rows("benign", 0, 100),
            "harmful": rows("harmful", 0, 100),
        },
        {
            "benign": rows("benign", 100, 50),
            "harmful": rows("harmful", 100, 50),
        },
    )


def _paired_generation(
    condition: str,
    prompt_id: str,
    *,
    trajectory_layers: list[int],
    manifest_sha256: str = "manifest-sha",
) -> dict[str, object]:
    generation_id = runner.condition_id(condition, prompt_id)
    return {
        "schema_version": 1,
        "manifest_sha256": manifest_sha256,
        "condition": condition,
        "prompt_id": prompt_id,
        "generation": {
            "id": generation_id,
            "condition": condition,
            "prompt_id": prompt_id,
            "text": f"response for {generation_id}",
        },
        "trajectory": [
            {
                "id": f"{generation_id}/capture_layer_{layer}",
                "condition": condition,
                "prompt_id": prompt_id,
                "layer": layer,
                "cosines": [0.1, 0.2],
            }
            for layer in trajectory_layers
        ],
    }


def _write_envelopes(path: Path, envelopes: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(envelope) + "\n" for envelope in envelopes),
        encoding="utf-8",
    )


def test_direction_build_uses_exact_fit_rows_all_layers_and_pinned_revision(
    exp1_v2: Any, config: dict[str, object], tmp_path: Path
) -> None:
    fit = {"benign": _rows("benign", 100), "harmful": _rows("harmful", 100)}
    holdout = {"benign": _rows("benign", 50), "harmful": _rows("harmful", 50)}
    captured: list[tuple[list[str], list[int]]] = []

    def engine_factory(*, model_id: str, revision: str) -> object:
        assert model_id == "Qwen/Qwen3-4B"
        assert revision == REVISION
        return object()

    def capture_hiddens(
        _llm: object, prompts: list[str], layers: list[int]
    ) -> dict[int, torch.Tensor]:
        captured.append((prompts, layers))
        return {
            layer: torch.tensor(
                [[float(layer), 1.0]] * 100 + [[float(layer) + 1.0, 1.0]] * 200
            )
            for layer in layers
        }

    result = exp1_v2.build_exp1_v2_directions(
        config,
        fit_rows=fit,
        holdout_rows=holdout,
        direction_path=tmp_path / "direction.json",
        metadata_path=tmp_path / "direction.metadata.json",
        engine_factory=engine_factory,
        capture_hiddens=capture_hiddens,
    )

    assert set(result) == set(LAYERS)
    assert captured == [
        ([row["prompt"] for row in fit["benign"] + fit["harmful"]], LAYERS)
    ]
    metadata = json.loads((tmp_path / "direction.metadata.json").read_text())
    assert metadata["model_revision"] == REVISION
    assert len(metadata["holdout_ids"]["benign"]) == 50
    assert len(metadata["holdout_ids"]["harmful"]) == 50
    assert not list(tmp_path.glob(".*.tmp*"))


def test_direction_zero_mean_difference_fails_before_publishing_artifacts(
    exp1_v2: Any, config: dict[str, object], tmp_path: Path
) -> None:
    fit = {"benign": _rows("benign", 100), "harmful": _rows("harmful", 100)}
    holdout = {"benign": _rows("benign", 50), "harmful": _rows("harmful", 50)}
    direction_path = tmp_path / "direction.json"
    metadata_path = tmp_path / "direction.metadata.json"

    with pytest.raises(ValueError, match="zero-norm"):
        exp1_v2.build_exp1_v2_directions(
            config,
            fit_rows=fit,
            holdout_rows=holdout,
            direction_path=direction_path,
            metadata_path=metadata_path,
            engine_factory=lambda **_: object(),
            capture_hiddens=lambda _engine, prompts, layers: {
                layer: torch.ones((len(prompts), 2)) for layer in layers
            },
        )

    assert not direction_path.exists()
    assert not metadata_path.exists()


def test_direction_persistence_is_atomic_and_resume_precedes_engine_construction(
    exp1_v2: Any, config: dict[str, object], tmp_path: Path
) -> None:
    direction_path = tmp_path / "direction.json"
    metadata_path = tmp_path / "direction.metadata.json"
    runner.write_json_atomic(direction_path, _direction_payload())
    runner.write_json_atomic(metadata_path, _direction_metadata())
    fit_rows, holdout_rows = _direction_rows()

    def forbidden_engine_factory(**_: object) -> object:
        raise AssertionError(
            "resume must load direction before constructing the engine"
        )

    loaded = exp1_v2.build_exp1_v2_directions(
        config,
        fit_rows=fit_rows,
        holdout_rows=holdout_rows,
        direction_path=direction_path,
        metadata_path=metadata_path,
        engine_factory=forbidden_engine_factory,
        capture_hiddens=lambda *_: {},
    )
    assert set(loaded) == set(LAYERS)


def test_direction_only_artifact_fails_closed_without_reconstructing_metadata(
    exp1_v2: Any, config: dict[str, object], tmp_path: Path
) -> None:
    direction_path = tmp_path / "direction.json"
    metadata_path = tmp_path / "direction.metadata.json"
    runner.write_json_atomic(direction_path, _direction_payload())
    fit_rows, holdout_rows = _direction_rows()

    with pytest.raises(RuntimeError, match="metadata|incomplete"):
        exp1_v2.build_exp1_v2_directions(
            config,
            fit_rows=fit_rows,
            holdout_rows=holdout_rows,
            direction_path=direction_path,
            metadata_path=metadata_path,
            engine_factory=lambda **_: pytest.fail("direction-only artifact rebound"),
            capture_hiddens=lambda *_: pytest.fail(
                "direction-only artifact recaptured"
            ),
        )

    assert not metadata_path.exists()


def test_direction_metadata_only_remains_fail_closed(
    exp1_v2: Any, config: dict[str, object], tmp_path: Path
) -> None:
    metadata_path = tmp_path / "direction.metadata.json"
    runner.write_json_atomic(metadata_path, _direction_metadata())
    fit_rows, holdout_rows = _direction_rows()

    with pytest.raises(RuntimeError, match="incomplete direction artifact"):
        exp1_v2.build_exp1_v2_directions(
            config,
            fit_rows=fit_rows,
            holdout_rows=holdout_rows,
            direction_path=tmp_path / "direction.json",
            metadata_path=metadata_path,
            engine_factory=lambda **_: object(),
            capture_hiddens=lambda *_: {},
        )


def test_direction_resume_rejects_current_fit_or_holdout_id_drift(
    exp1_v2: Any, config: dict[str, object], tmp_path: Path
) -> None:
    direction_path = tmp_path / "direction.json"
    metadata_path = tmp_path / "direction.metadata.json"
    runner.write_json_atomic(direction_path, _direction_payload())
    runner.write_json_atomic(metadata_path, _direction_metadata())
    fit_rows, holdout_rows = _direction_rows()

    changed_fit = {"benign": _rows("benign", 100), "harmful": _rows("harmful", 100)}
    changed_holdout = {
        "benign": _rows("benign", 50),
        "harmful": _rows("harmful", 50),
    }
    changed_fit["benign"][0]["id"] = "benign-replaced"

    with pytest.raises((RuntimeError, ValueError), match="metadata|fit|identity|stale"):
        exp1_v2.build_exp1_v2_directions(
            config,
            fit_rows=changed_fit,
            holdout_rows=changed_holdout,
            direction_path=direction_path,
            metadata_path=metadata_path,
            engine_factory=lambda **_: pytest.fail("resume constructed the engine"),
            capture_hiddens=lambda *_: {},
        )


@pytest.mark.parametrize("missing_layer", [1, 20])
def test_direction_rejects_missing_layers(
    exp1_v2: Any, config: dict[str, object], tmp_path: Path, missing_layer: int
) -> None:
    payload = _direction_payload()
    del payload[str(missing_layer)]
    direction_path = tmp_path / "direction.json"
    metadata_path = tmp_path / "direction.metadata.json"
    runner.write_json_atomic(direction_path, payload)
    runner.write_json_atomic(metadata_path, _direction_metadata())
    fit_rows, holdout_rows = _direction_rows()

    with pytest.raises((RuntimeError, ValueError), match="layer"):
        exp1_v2.build_exp1_v2_directions(
            config,
            fit_rows=fit_rows,
            holdout_rows=holdout_rows,
            direction_path=direction_path,
            metadata_path=metadata_path,
            engine_factory=lambda **_: object(),
            capture_hiddens=lambda *_: {},
        )


@pytest.mark.parametrize(
    "bad_metadata",
    [
        {**_direction_metadata(), "model_revision": "stale"},
        {**_direction_metadata(), "holdout_ids": {"benign": [], "harmful": []}},
    ],
)
def test_direction_rejects_stale_metadata_and_legacy_artifact_paths(
    exp1_v2: Any,
    config: dict[str, object],
    tmp_path: Path,
    bad_metadata: dict[str, object],
) -> None:
    direction_path = tmp_path / "direction.json"
    metadata_path = tmp_path / "direction.metadata.json"
    runner.write_json_atomic(direction_path, _direction_payload())
    runner.write_json_atomic(metadata_path, bad_metadata)
    fit_rows, holdout_rows = _direction_rows()
    with pytest.raises((RuntimeError, ValueError), match="metadata|stale|holdout"):
        exp1_v2.build_exp1_v2_directions(
            config,
            fit_rows=fit_rows,
            holdout_rows=holdout_rows,
            direction_path=direction_path,
            metadata_path=metadata_path,
            engine_factory=lambda **_: object(),
            capture_hiddens=lambda *_: {},
        )

    legacy_path = tmp_path / "exp1_direction.json"
    runner.write_json_atomic(legacy_path, _direction_payload())
    runner.write_json_atomic(metadata_path, _direction_metadata())
    with pytest.raises((RuntimeError, ValueError), match="legacy|exp1_direction"):
        exp1_v2.build_exp1_v2_directions(
            config,
            fit_rows=_rows("unused", 100),
            holdout_rows={
                "benign": _rows("unused", 50),
                "harmful": _rows("unused", 50),
            },
            direction_path=legacy_path,
            metadata_path=tmp_path / "direction.metadata.json",
            engine_factory=lambda **_: object(),
            capture_hiddens=lambda *_: {},
        )


def test_generation_is_baseline_plus_each_layer_with_composite_ids_and_full_contract(
    exp1_v2: Any, config: dict[str, object], tmp_path: Path
) -> None:
    prompts = _rows("hb", 154)
    calls: list[tuple[str, list[dict[str, object]], object]] = []
    directions = {
        layer: SimpleNamespace(direction=torch.tensor([1.0, 0.0]), mean_norm=2.0)
        for layer in LAYERS
    }

    def generate(
        condition: str, rows: list[dict[str, object]], spec: object, **kwargs: object
    ) -> list[dict[str, object]]:
        calls.append((condition, rows, spec))
        assert kwargs == {
            "max_new_tokens": 512,
            "temperature": 0.0,
            "top_p": 1.0,
            "enable_thinking": False,
            "capture_layers": (
                LAYERS
                if condition == "baseline"
                else [int(condition.removeprefix("layer_"))]
            ),
        }
        return [
            {"id": runner.condition_id(condition, row["id"]), "text": "deterministic"}
            for row in rows
        ]

    result = exp1_v2.generate_exp1_v2(
        config,
        prompts,
        directions=directions,
        checkpoint_root=tmp_path,
        generate=generate,
    )

    assert len(calls) == 12
    assert [condition for condition, _, _ in calls] == [
        "baseline",
        *[f"layer_{layer}" for layer in LAYERS],
    ]
    assert calls[0][2] is None
    for condition, _, spec in calls[1:]:
        spec = cast(Any, spec)
        assert spec.layer == int(condition.removeprefix("layer_"))
        assert spec.alpha == pytest.approx(0.03)
        assert spec.mean_norm == pytest.approx(2.0)
        assert spec.schedule.kind == "full"
    assert len(result) == 12 * 154
    assert len({row["id"] for row in result}) == 12 * 154
    assert directions[19].direction is not directions[20].direction


def test_generation_persists_each_prompt_repairs_torn_tail_and_resumes_missing_ids(
    exp1_v2: Any,
    config: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts = _rows("hb", 154)
    checkpoint = tmp_path / "baseline.jsonl"
    runner.append_jsonl(
        checkpoint,
        [
            {"id": runner.condition_id("baseline", row["id"]), "text": "done"}
            for row in prompts[:17]
        ],
    )
    with checkpoint.open("a", encoding="utf-8") as stream:
        stream.write('{"id":"torn"')
    generated_ids: list[str] = []
    persisted_batches: list[list[dict[str, object]]] = []
    real_append_jsonl = runner.append_jsonl

    def append_spy(path: str | Path, records: list[dict[str, object]]) -> None:
        persisted_batches.append(records)
        real_append_jsonl(path, records)

    monkeypatch.setattr(runner, "append_jsonl", append_spy)

    def generate(
        condition: str, rows: list[dict[str, object]], spec: object, **_: object
    ) -> list[dict[str, object]]:
        assert condition == "baseline"
        assert spec is None
        generated_ids.extend(str(row["id"]) for row in rows)
        return [
            {"id": runner.condition_id(condition, row["id"]), "text": "new"}
            for row in rows
        ]

    exp1_v2.generate_exp1_v2(
        config,
        prompts,
        directions={
            layer: SimpleNamespace(direction=torch.ones(2), mean_norm=1.0)
            for layer in LAYERS
        },
        checkpoint_root=tmp_path,
        conditions=["baseline"],
        generate=generate,
    )
    assert len(generated_ids) == 154 - 17
    assert len(persisted_batches) == 154 - 17
    assert all(len(batch) == 1 for batch in persisted_batches)
    assert len(runner.read_jsonl(checkpoint)) == 154
    assert len(runner.completed_ids(checkpoint)) == 154


def test_generation_rejects_dataset_drift_and_replacement_mode(
    exp1_v2: Any, config: dict[str, object], tmp_path: Path
) -> None:
    prompts = _rows("hb", 154)
    with pytest.raises((RuntimeError, ValueError), match="drift|replacement"):
        exp1_v2.generate_exp1_v2(
            config,
            prompts[:-1],
            directions={
                layer: SimpleNamespace(direction=torch.ones(2), mean_norm=1.0)
                for layer in LAYERS
            },
            checkpoint_root=tmp_path,
            generate=lambda **_: [],
        )

    changed = json.loads(json.dumps(config))
    cast(dict[str, object], changed["steering"])["replace_hidden_state"] = True
    with pytest.raises((RuntimeError, ValueError), match="replacement|protocol"):
        exp1_v2.generate_exp1_v2(
            changed,
            prompts,
            directions={
                layer: SimpleNamespace(direction=torch.ones(2), mean_norm=1.0)
                for layer in LAYERS
            },
            checkpoint_root=tmp_path,
            generate=lambda **_: [],
        )


def test_generation_commits_bounded_paired_envelopes_per_example(
    exp1_v2: Any, config: dict[str, object], tmp_path: Path
) -> None:
    prompts = _rows("hb", 154)
    calls: list[tuple[str, list[str], list[int]]] = []
    directions = {
        layer: SimpleNamespace(direction=torch.ones(2), mean_norm=1.0)
        for layer in LAYERS
    }

    def generate(
        condition: str,
        rows: list[dict[str, object]],
        _spec: object,
        **kwargs: object,
    ) -> list[dict[str, object]]:
        capture_layers = list(cast(list[int], kwargs["capture_layers"]))
        calls.append((condition, [str(row["id"]) for row in rows], capture_layers))
        return [
            _paired_generation(
                condition,
                str(row["id"]),
                trajectory_layers=capture_layers,
            )
            for row in rows
        ]

    result = exp1_v2.generate_exp1_v2(
        config,
        prompts,
        directions=directions,
        checkpoint_root=tmp_path,
        conditions=["baseline", "layer_1"],
        generate=generate,
        batch_size=3,
        manifest_sha256="manifest-sha",
    )

    assert len(calls) == math.ceil(154 / 3) * 2
    assert all(0 < len(prompt_ids) <= 3 for _, prompt_ids, _ in calls)
    assert calls[0][2] == LAYERS
    assert all(capture_layers == [1] for _, _, capture_layers in calls[52:])
    assert len(result) == 2 * 154

    for condition, expected_layers in (("baseline", LAYERS), ("layer_1", [1])):
        checkpoint = exp1_v2.exp1_v2_checkpoint_path(tmp_path, condition)
        envelopes = runner.read_jsonl(checkpoint)
        assert len(envelopes) == 154
        assert all(envelope["schema_version"] == 1 for envelope in envelopes)
        assert all(
            envelope["manifest_sha256"] == "manifest-sha" for envelope in envelopes
        )
        assert all(
            len(envelope["trajectory"]) == len(expected_layers)
            for envelope in envelopes
        )
        for envelope in envelopes:
            generation_id = str(cast(dict[str, object], envelope["generation"])["id"])
            trajectory_ids = {
                str(row["id"])
                for row in cast(list[dict[str, object]], envelope["trajectory"])
            }
            assert trajectory_ids == {
                f"{generation_id}/capture_layer_{layer}" for layer in expected_layers
            }
    assert not list((tmp_path / "exp1_v2").glob(".*.tmp*"))


def test_generation_backfills_jsonl_after_sidecar_only_crash(
    exp1_v2: Any,
    config: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts = _rows("hb", 154)
    directions = {
        layer: SimpleNamespace(direction=torch.ones(2), mean_norm=1.0)
        for layer in LAYERS
    }
    real_write = runner.write_json_atomic
    crashed = False

    def write_then_crash(path: str | Path, value: object) -> None:
        nonlocal crashed
        real_write(path, value)
        if not crashed and Path(path).parent.name == "baseline":
            crashed = True
            raise RuntimeError("injected sidecar-only crash")

    monkeypatch.setattr(runner, "write_json_atomic", write_then_crash)

    def generate(
        condition: str,
        rows: list[dict[str, object]],
        _spec: object,
        **kwargs: object,
    ) -> list[dict[str, object]]:
        return [
            _paired_generation(
                condition,
                str(row["id"]),
                trajectory_layers=list(cast(list[int], kwargs["capture_layers"])),
            )
            for row in rows
        ]

    with pytest.raises(RuntimeError, match="sidecar-only"):
        exp1_v2.generate_exp1_v2(
            config,
            prompts,
            directions=directions,
            checkpoint_root=tmp_path,
            conditions=["baseline"],
            generate=generate,
            batch_size=1,
            manifest_sha256="manifest-sha",
        )

    monkeypatch.setattr(runner, "write_json_atomic", real_write)
    resumed_calls: list[str] = []

    def resume_generate(
        condition: str,
        rows: list[dict[str, object]],
        _spec: object,
        **kwargs: object,
    ) -> list[dict[str, object]]:
        resumed_calls.extend(str(row["id"]) for row in rows)
        return generate(condition, rows, _spec, **kwargs)

    result = exp1_v2.generate_exp1_v2(
        config,
        prompts,
        directions=directions,
        checkpoint_root=tmp_path,
        conditions=["baseline"],
        generate=resume_generate,
        batch_size=1,
        manifest_sha256="manifest-sha",
    )

    checkpoint = exp1_v2.exp1_v2_checkpoint_path(tmp_path, "baseline")
    assert len(resumed_calls) == 153
    assert len(runner.read_jsonl(checkpoint)) == 154
    assert len(result) == 154


def test_generation_materializes_missing_sidecar_from_jsonl(
    exp1_v2: Any, config: dict[str, object], tmp_path: Path
) -> None:
    prompts = _rows("hb", 154)
    directions = {
        layer: SimpleNamespace(direction=torch.ones(2), mean_norm=1.0)
        for layer in LAYERS
    }
    checkpoint = exp1_v2.exp1_v2_checkpoint_path(tmp_path, "baseline")
    first = _paired_generation("baseline", "hb-0", trajectory_layers=LAYERS)
    runner.append_jsonl(checkpoint, [first])

    resumed_calls: list[str] = []

    def generate(
        condition: str,
        rows: list[dict[str, object]],
        _spec: object,
        **kwargs: object,
    ) -> list[dict[str, object]]:
        resumed_calls.extend(str(row["id"]) for row in rows)
        return [
            _paired_generation(
                condition,
                str(row["id"]),
                trajectory_layers=list(cast(list[int], kwargs["capture_layers"])),
            )
            for row in rows
        ]

    exp1_v2.generate_exp1_v2(
        config,
        prompts,
        directions=directions,
        checkpoint_root=tmp_path,
        conditions=["baseline"],
        generate=generate,
        batch_size=1,
        manifest_sha256="manifest-sha",
    )

    assert len(resumed_calls) == 153
    assert (
        len(list((checkpoint.parent / "envelopes" / "baseline").glob("*.json"))) == 154
    )


def test_generation_recovery_skips_only_durable_envelopes_after_failure(
    exp1_v2: Any, config: dict[str, object], tmp_path: Path
) -> None:
    prompts = _rows("hb", 154)
    directions = {
        layer: SimpleNamespace(direction=torch.ones(2), mean_norm=1.0)
        for layer in LAYERS
    }
    calls: list[list[str]] = []
    attempts = 0

    def generate(
        condition: str,
        rows: list[dict[str, object]],
        _spec: object,
        **kwargs: object,
    ) -> list[dict[str, object]]:
        nonlocal attempts
        attempts += 1
        calls.append([str(row["id"]) for row in rows])
        if attempts == 2:
            raise RuntimeError("injected generation failure")
        return [
            _paired_generation(
                condition,
                str(row["id"]),
                trajectory_layers=list(cast(list[int], kwargs["capture_layers"])),
            )
            for row in rows
        ]

    with pytest.raises(RuntimeError, match="injected generation failure"):
        exp1_v2.generate_exp1_v2(
            config,
            prompts,
            directions=directions,
            checkpoint_root=tmp_path,
            conditions=["baseline"],
            generate=generate,
            batch_size=1,
            manifest_sha256="manifest-sha",
        )

    checkpoint = exp1_v2.exp1_v2_checkpoint_path(tmp_path, "baseline")
    assert len(runner.read_jsonl(checkpoint)) == 1
    first_id = runner.condition_id("baseline", prompts[0]["id"])

    resumed_calls: list[list[str]] = []

    def resume_generate(
        condition: str,
        rows: list[dict[str, object]],
        _spec: object,
        **kwargs: object,
    ) -> list[dict[str, object]]:
        resumed_calls.append([str(row["id"]) for row in rows])
        return [
            _paired_generation(
                condition,
                str(row["id"]),
                trajectory_layers=list(cast(list[int], kwargs["capture_layers"])),
            )
            for row in rows
        ]

    resumed = exp1_v2.generate_exp1_v2(
        config,
        prompts,
        directions=directions,
        checkpoint_root=tmp_path,
        conditions=["baseline"],
        generate=resume_generate,
        batch_size=2,
        manifest_sha256="manifest-sha",
    )
    assert len(resumed) == 154
    assert first_id not in {prompt_id for batch in resumed_calls for prompt_id in batch}
    assert sum(len(batch) for batch in resumed_calls) == 153


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda envelope: {**envelope, "manifest_sha256": "stale"}, "manifest"),
        (
            lambda envelope: [envelope, envelope],
            "duplicate",
        ),
        (
            lambda envelope: {**envelope, "condition": "layer_1"},
            "condition",
        ),
        (
            lambda envelope: {**envelope, "prompt_id": "hb-wrong"},
            "prompt",
        ),
        (
            lambda envelope: {
                **envelope,
                "trajectory": envelope["trajectory"][:-1],
            },
            "trajectory",
        ),
        (
            lambda envelope: {
                **envelope,
                "trajectory": [
                    {**envelope["trajectory"][0], "cosines": [float("nan")]}
                ],
            },
            "finite",
        ),
        (
            lambda envelope: [
                envelope,
                _paired_generation("layer_1", "hb-0", trajectory_layers=[1]),
            ],
            "mixed",
        ),
    ],
)
def test_generation_rejects_invalid_orphaned_checkpoint_envelopes(
    exp1_v2: Any,
    config: dict[str, object],
    tmp_path: Path,
    mutator: Any,
    message: str,
) -> None:
    baseline = _paired_generation("baseline", "hb-0", trajectory_layers=LAYERS)
    checkpoint = exp1_v2.exp1_v2_checkpoint_path(tmp_path, "baseline")
    mutated = mutator(baseline)
    _write_envelopes(checkpoint, mutated if isinstance(mutated, list) else [mutated])
    checkpoint.with_name(f".{checkpoint.name}.tmp.orphan").write_text(
        "not a committed envelope", encoding="utf-8"
    )

    with pytest.raises((RuntimeError, ValueError), match=message):
        exp1_v2.generate_exp1_v2(
            config,
            _rows("hb", 154),
            directions={
                layer: SimpleNamespace(direction=torch.ones(2), mean_norm=1.0)
                for layer in LAYERS
            },
            checkpoint_root=tmp_path,
            conditions=["baseline"],
            generate=lambda **_: pytest.fail("invalid checkpoint was accepted"),
            batch_size=2,
            manifest_sha256="manifest-sha",
        )


def test_cli_generate_reuses_engine_and_serializes_canonical_capture_artifacts(
    exp1_v2: Any,
    config: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = exp1_v2._cli_paths(tmp_path)
    prompts = [
        {
            "id": f"hb-{index}",
            "behavior": f"harmful request hb-{index}",
            "category": f"category-{index % 7}",
            "provenance": {"source": "harmbench", "revision": "hb-revision"},
        }
        for index in range(154)
    ]
    provenance = {
        "source_sha256": "source-sha",
        "config_sha256": "config-sha",
        "model_revision": REVISION,
    }
    manifest = {
        "manifest_sha256": "manifest-sha",
        "config_sha256": "config-sha",
        "model_revision": REVISION,
        "selected_rows": {"harmbench": prompts},
        "provenance": provenance,
    }
    paths["inputs"].mkdir(parents=True)
    runner.write_json_atomic(paths["manifest"], manifest)
    runner.write_json_atomic(
        paths["preflight"],
        {"status": "complete", "result": {"harmbench": prompts}},
    )
    runner.write_json_atomic(
        paths["directions"],
        {
            str(layer): {
                "direction": [1.0, 0.0] if layer != 20 else [0.0, 2.0],
                "mean_norm": 2.0,
            }
            for layer in LAYERS
        },
    )

    engine_calls: list[tuple[str, dict[str, object]]] = []
    steering_calls: list[dict[str, object]] = []

    class FakeTokenizer:
        def apply_chat_template(
            self, messages: list[dict[str, str]], **_: object
        ) -> str:
            return f"request::{messages[0]['content']}"

    class FakeEngine:
        def get_tokenizer(self) -> FakeTokenizer:
            return FakeTokenizer()

    def get_engine(model_id: str, **kwargs: object) -> FakeEngine:
        engine_calls.append((model_id, kwargs))
        return FakeEngine()

    def steered_generate(
        engine: FakeEngine,
        requests: list[str],
        max_tokens: int,
        spec: object,
        **kwargs: object,
    ) -> list[runner.GenerateResult]:
        assert isinstance(engine, FakeEngine)
        sink = kwargs.get("sink")
        assert isinstance(sink, runner.CaptureSink)
        assert kwargs["batch_prompts"] == 32
        scalar_directions = kwargs["scalar_directions"]
        assert isinstance(scalar_directions, dict)
        capture_layers = list(cast(list[int], kwargs["capture_layers"]))
        steering_calls.append(
            {
                "requests": requests,
                "max_tokens": max_tokens,
                "spec": spec,
                "capture_layers": capture_layers,
                "scalar_directions": scalar_directions,
                "top_p": kwargs["top_p"],
            }
        )
        for request in requests:
            prompt_id = request.rsplit(" ", 1)[-1]
            request_id = f"request/{prompt_id}"
            for layer in capture_layers:
                direction = cast(list[torch.Tensor], scalar_directions[layer])[0]
                hidden = torch.tensor([float(layer), 2.0])
                norm = float(torch.linalg.vector_norm(hidden).item())
                sink.rows.append(
                    {
                        "phase": "prefill",
                        "layer": layer,
                        "slot": 0,
                        "request_id": request_id,
                        "k": None,
                        "dots": [float(torch.dot(hidden, direction).item())],
                        "norm": norm,
                    }
                )
                for token_index in (1, 2):
                    dot = float(torch.dot(hidden, direction).item()) + token_index
                    sink.rows.append(
                        {
                            "phase": "decode",
                            "layer": layer,
                            "slot": 0,
                            "request_id": request_id,
                            "k": token_index,
                            "dots": [dot],
                            "norm": norm,
                        }
                    )
        return [
            runner.GenerateResult(
                text=f"response for {request}",
                request_id=f"request/{request.rsplit(' ', 1)[-1]}",
            )
            for request in requests
        ]

    monkeypatch.setattr(runner, "get_engine", get_engine)
    monkeypatch.setattr(runner, "steered_generate", steered_generate)
    exp1_v2._cli_generate(config, paths)

    assert engine_calls == [
        (
            "Qwen/Qwen3-4B",
            {
                "revision": REVISION,
                "max_model_len": 8192,
                "gpu_memory_utilization": 0.8,
            },
        )
    ]
    batches_per_condition = math.ceil(154 / 32)
    assert len(steering_calls) == batches_per_condition * len(exp1_v2.CONDITIONS)
    assert all(len(cast(list[str], call["requests"])) <= 32 for call in steering_calls)
    assert all(
        call["capture_layers"] == LAYERS and call["spec"] is None
        for call in steering_calls[:batches_per_condition]
    )
    for call in steering_calls[batches_per_condition:]:
        spec = cast(Any, call["spec"])
        assert call["capture_layers"] == [int(spec.layer)]
    assert all(
        call["top_p"] == 1.0 and call["max_tokens"] == 512 for call in steering_calls
    )
    layer_19_spec = cast(Any, steering_calls[batches_per_condition * 6]["spec"])
    layer_20_spec = cast(Any, steering_calls[batches_per_condition * 7]["spec"])
    assert layer_19_spec.layer == 19
    assert layer_20_spec.layer == 20
    assert layer_19_spec.direction is not layer_20_spec.direction
    assert layer_19_spec.alpha == pytest.approx(0.03)
    assert layer_19_spec.mean_norm == pytest.approx(2.0)
    assert layer_19_spec.schedule.kind == "full"

    envelopes = [
        envelope
        for condition in exp1_v2.CONDITIONS
        for envelope in runner.read_jsonl(paths["generations"] / f"{condition}.jsonl")
    ]
    assert len(envelopes) == 1848
    assert len({str(envelope["generation"]["id"]) for envelope in envelopes}) == 1848
    trajectory_rows = [
        row
        for envelope in envelopes
        for row in cast(list[dict[str, object]], envelope["trajectory"])
    ]
    assert len({str(row["id"]) for row in trajectory_rows}) == 3388
    assert {int(cast(int, row["layer"])) for row in trajectory_rows} >= {19, 20}
    for layer in (19, 20):
        physical_rows = [row for row in trajectory_rows if row["layer"] == layer]
        baseline_rows = [row for row in physical_rows if row["condition"] == "baseline"]
        treatment_rows = [
            row for row in physical_rows if row["condition"] == f"layer_{layer}"
        ]
        assert len(baseline_rows) == 154
        assert len(treatment_rows) == 154
        assert all(row["capture_layer"] == layer for row in physical_rows)
        assert all(row["steering_layer"] is None for row in baseline_rows)
        assert all(row["steering_layer"] == layer for row in treatment_rows)
        assert {str(row["id"]) for row in baseline_rows}.isdisjoint(
            {str(row["id"]) for row in treatment_rows}
        )

    expected_generation_keys = {
        "id",
        "prompt_id",
        "condition",
        "category",
        "request",
        "response",
        "generation_status",
        "provenance",
    }
    for envelope in envelopes:
        generation = cast(dict[str, object], envelope["generation"])
        assert set(generation) == expected_generation_keys
        assert generation["generation_status"] == "ok"
        assert generation["provenance"] == provenance
        assert generation["request"] and generation["response"]
        assert not {"text", "judge", "label", "final_source"} & set(generation)
        for row in cast(list[dict[str, object]], envelope["trajectory"]):
            assert all(
                math.isfinite(float(cast(Any, value)))
                for value in cast(list[object], row["cosines"])
            )

    layer_19 = next(row for row in trajectory_rows if row["layer"] == 19)
    layer_20 = next(row for row in trajectory_rows if row["layer"] == 20)
    hidden_19_norm = math.sqrt(19.0**2 + 2.0**2)
    hidden_20_norm = math.sqrt(20.0**2 + 2.0**2)
    assert layer_19["cosines"] == pytest.approx(
        [20.0 / hidden_19_norm, 21.0 / hidden_19_norm]
    )
    assert layer_20["cosines"] == pytest.approx(
        [5.0 / (hidden_20_norm * 2.0), 6.0 / (hidden_20_norm * 2.0)]
    )
    assert layer_19["cosines"] != layer_20["cosines"]


def test_cli_generation_writes_physical_capture_layers_and_manifest_trajectory_universe(
    exp1_v2: Any,
    config: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = exp1_v2._cli_paths(tmp_path)
    prompts = [
        {
            "id": f"hb-{index}",
            "behavior": f"harmful request hb-{index}",
            "category": f"category-{index % 7}",
            "provenance": {"source": "harmbench", "revision": "hb-revision"},
        }
        for index in range(154)
    ]
    provenance = {
        "source_sha256": "source-sha",
        "config_sha256": "config-sha",
        "model_revision": REVISION,
    }
    paths["inputs"].mkdir(parents=True)
    runner.write_json_atomic(
        paths["manifest"],
        {
            "manifest_sha256": "manifest-sha",
            "config_sha256": "config-sha",
            "model_revision": REVISION,
            "selected_rows": {"harmbench": prompts},
            "provenance": provenance,
        },
    )
    runner.write_json_atomic(
        paths["preflight"], {"status": "complete", "result": {"harmbench": prompts}}
    )
    runner.write_json_atomic(
        paths["directions"],
        {str(layer): {"direction": [1.0, 0.0], "mean_norm": 2.0} for layer in LAYERS},
    )

    class FakeTokenizer:
        def apply_chat_template(
            self, messages: list[dict[str, str]], **_: object
        ) -> str:
            return f"request::{messages[0]['content']}"

    class FakeEngine:
        def get_tokenizer(self) -> FakeTokenizer:
            return FakeTokenizer()

    monkeypatch.setattr(runner, "get_engine", lambda *_args, **_kwargs: FakeEngine())

    def steered_generate(
        _engine: FakeEngine,
        requests: list[str],
        **kwargs: object,
    ) -> list[runner.GenerateResult]:
        sink = cast(runner.CaptureSink, kwargs["sink"])
        capture_layers = cast(list[int], kwargs["capture_layers"])
        for request in requests:
            request_id = f"request/{request.rsplit(' ', 1)[-1]}"
            for layer in capture_layers:
                for token_index in (1, 2):
                    sink.rows.append(
                        {
                            "layer": layer,
                            "request_id": request_id,
                            "k": token_index,
                            "dots": [1.0],
                            "norm": 1.0,
                        }
                    )
        return [
            runner.GenerateResult(
                text="deterministic",
                request_id=f"request/{request.rsplit(' ', 1)[-1]}",
            )
            for request in requests
        ]

    monkeypatch.setattr(runner, "steered_generate", steered_generate)
    exp1_v2._cli_generate(config, paths)

    envelopes = [
        envelope
        for condition in exp1_v2.CONDITIONS
        for envelope in runner.read_jsonl(paths["generations"] / f"{condition}.jsonl")
    ]
    trajectory_rows = [
        row
        for envelope in envelopes
        for row in cast(list[dict[str, object]], envelope["trajectory"])
    ]
    assert len(trajectory_rows) == 3388
    manifest_prompt_ids = [str(row["id"]) for row in prompts]
    expected_ids = {
        f"{condition}/{prompt_id}/capture_layer_{layer}"
        for condition in exp1_v2.CONDITIONS
        for prompt_id in manifest_prompt_ids
        for layer in (LAYERS if condition == "baseline" else [int(condition[6:])])
    }
    assert {str(row["id"]) for row in trajectory_rows} == expected_ids
    assert len(expected_ids) == 3388
    assert all(row["layer"] == row["capture_layer"] for row in trajectory_rows)
    assert all(
        row["steering_layer"] is None
        for row in trajectory_rows
        if row["condition"] == "baseline"
    )
    assert all(
        row["steering_layer"] == int(str(row["condition"])[6:])
        and row["capture_layer"] == int(str(row["condition"])[6:])
        for row in trajectory_rows
        if row["condition"] != "baseline"
    )
