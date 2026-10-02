from __future__ import annotations

from pathlib import Path
from typing import Any, cast
import importlib.util
import json
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import yaml

from prefix.no_steering import model_spec
from prefix import runner


ROOT = Path(__file__).parents[1]
CONFIG_PATH = ROOT / "configs" / "exp1_v2.yaml"
SCRIPT_PATH = ROOT / "scripts" / "run_exp1_v2.py"
LAYERS = [1, 5, 8, 12, 16, 19, 20, 23, 27, 30, 34]


@pytest.fixture(scope="module")
def exp1_v2() -> Any:
    spec = importlib.util.spec_from_file_location("run_exp1_v2", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def config() -> dict[str, object]:
    assert CONFIG_PATH == ROOT / "configs" / "exp1_v2.yaml"
    with CONFIG_PATH.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    assert isinstance(value, dict)
    return value


def test_exp1_v2_freezes_model_namespace_and_direction(
    config: dict[str, object],
) -> None:
    model = config["model"]
    direction = config["direction"]
    assert isinstance(model, dict) and isinstance(direction, dict)
    spec = model_spec("Qwen/Qwen3-4B")
    assert model["id"] == spec.model_id == "Qwen/Qwen3-4B"
    assert model["revision"] == spec.revision
    assert model["checkpoint_namespace"] == "exp1_v2"
    assert direction["semantics"] == "unsafe"
    assert direction["operation"] == "harmful_minus_benign"
    assert direction["fit_per_class"] == 100
    assert direction["holdout_per_class"] == 50


def test_exp1_v2_is_baseline_plus_additive_full_grid(config: dict[str, object]) -> None:
    grid = config["grid"]
    assert isinstance(grid, dict)
    assert grid["layers"] == LAYERS
    assert 19 in grid["layers"] and 20 in grid["layers"]
    assert grid["alpha"] == pytest.approx(0.03)
    assert grid["schedule"] == "full"
    assert config["conditions"] == ["baseline", *[f"layer_{layer}" for layer in LAYERS]]
    steering = cast(dict[str, Any], config["steering"])
    assert steering["mode"] == "additive"
    assert steering["replace_hidden_state"] is False


def test_exp1_v2_generation_and_no_tuning_contract(config: dict[str, object]) -> None:
    generation = config["generation"]
    assert isinstance(generation, dict)
    assert generation["n_prompts"] == 154
    assert generation["max_new_tokens"] == 512
    assert config["tuning"] is False
    assert config["replacement"] is False


def test_exp1_v2_freezes_distinct_global_zero_temperature_judges(
    config: dict[str, object],
) -> None:
    judges = config["judges"]
    assert isinstance(judges, dict)
    primary = judges["primary"]
    fallback = judges["fallback"]
    assert isinstance(primary, dict) and isinstance(fallback, dict)
    assert primary == {
        "model": "gemini-3.7-flash",
        "region": "global",
        "temperature": 0.0,
    }
    assert fallback == {
        "model": "gemini-3.5-flash-lite",
        "region": "global",
        "temperature": 0.0,
    }
    assert primary["model"] != fallback["model"]


def test_exp1_v2_script_uses_new_namespace_without_legacy_or_notifications() -> None:
    assert SCRIPT_PATH == ROOT / "scripts" / "run_exp1_v2.py"
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "exp1_v2" in source
    assert "exp1_direction" not in source
    assert "exp1_generations" not in source
    assert "exp1_trajectories" not in source
    assert "replace_hidden_state" not in source
    assert "notify_on_exit" in source


def test_exp1_v2_standalone_phase_block_notifies_on_success(
    config: dict[str, object],
    exp1_v2: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, bool, Path | None, str]] = []

    @contextmanager
    def fake_notify(task: str, *, log_file: Path, enabled: bool):
        try:
            yield
        except BaseException:
            calls.append((task, enabled, log_file, "failed"))
            raise
        else:
            calls.append((task, enabled, log_file, "completed"))

    monkeypatch.setattr(exp1_v2, "notify_on_exit", fake_notify)
    monkeypatch.setattr(runner, "load_config", lambda _path: config)
    monkeypatch.setattr(exp1_v2, "_cli_prepare", lambda *_args: None)

    exp1_v2.main(["--root", str(tmp_path), "--phase", "prepare"])

    assert calls == [("exp1-v2", True, tmp_path / "logs" / "exp1_v2.log", "completed")]


def test_exp1_v2_standalone_phase_block_notifies_then_reraises_exception(
    config: dict[str, object],
    exp1_v2: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, bool, Path | None, str]] = []

    @contextmanager
    def fake_notify(task: str, *, log_file: Path, enabled: bool):
        try:
            yield
        except BaseException:
            calls.append((task, enabled, log_file, "failed"))
            raise
        else:
            calls.append((task, enabled, log_file, "completed"))

    monkeypatch.setattr(exp1_v2, "notify_on_exit", fake_notify)
    monkeypatch.setattr(runner, "load_config", lambda _path: config)

    def fail(*_args: object) -> None:
        raise RuntimeError("phase failed")

    monkeypatch.setattr(exp1_v2, "_cli_prepare", fail)

    with pytest.raises(RuntimeError, match="phase failed"):
        exp1_v2.main(["--root", str(tmp_path), "--phase", "prepare"])

    assert calls == [("exp1-v2", True, tmp_path / "logs" / "exp1_v2.log", "failed")]


def test_exp1_v2_cli_contention_fails_before_phase_or_log_mutation(
    config: dict[str, object],
    exp1_v2: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []

    @contextmanager
    def rejecting_lock(root: Path):
        events.append(f"lock:{root}")
        raise runner.MutationLockError("already locked")
        yield root

    monkeypatch.setattr(runner, "load_config", lambda _path: config)
    monkeypatch.setattr(runner, "exclusive_mutation_lock", rejecting_lock)
    monkeypatch.setattr(exp1_v2, "_cli_prepare", lambda *_args: events.append("phase"))

    with pytest.raises(runner.MutationLockError, match="already locked"):
        exp1_v2.main(["--root", str(tmp_path), "--phase", "prepare"])

    assert events == [f"lock:{tmp_path}"]
    assert not (tmp_path / "logs").exists()


def test_exp1_v2_cli_lock_wraps_chained_phases_before_handlers(
    config: dict[str, object],
    exp1_v2: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []

    @contextmanager
    def recording_lock(root: Path):
        events.append("lock-enter")
        yield root
        events.append("lock-exit")

    monkeypatch.setattr(runner, "load_config", lambda _path: config)
    monkeypatch.setattr(runner, "exclusive_mutation_lock", recording_lock)
    monkeypatch.setattr(
        exp1_v2, "_cli_prepare", lambda *_args: events.append("prepare")
    )
    monkeypatch.setattr(exp1_v2, "_validate_preflight_marker", lambda *_args: None)
    monkeypatch.setattr(
        exp1_v2,
        "_cli_direction",
        lambda *_args: events.append("direction"),
    )
    monkeypatch.setattr(
        exp1_v2,
        "_write_phase_marker",
        lambda *_args: events.append("mark-direction"),
    )

    exp1_v2.main(["--root", str(tmp_path), "--phase", "prepare", "direction"])

    assert events == [
        "lock-enter",
        "prepare",
        "direction",
        "mark-direction",
        "lock-exit",
    ]


def test_exp1_v2_workflow_contention_fails_before_callbacks(
    exp1_v2: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []

    @contextmanager
    def rejecting_lock(root: Path):
        events.append(f"lock:{root}")
        raise runner.MutationLockError("already locked")
        yield root

    monkeypatch.setattr(runner, "exclusive_mutation_lock", rejecting_lock)

    with pytest.raises(runner.MutationLockError, match="already locked"):
        exp1_v2.run_exp1_v2_workflow(
            config={},
            paths={"root": tmp_path, "logs": tmp_path / "logs"},
            preflight=lambda: events.append("preflight"),
            direction=lambda: events.append("direction"),
            generate=lambda: events.append("generate"),
            judge=lambda: events.append("judge"),
            analyze=lambda: events.append("analyze"),
            engine_factory=lambda: None,
        )

    assert events == [f"lock:{tmp_path}"]


def test_exp1_v2_workflow_same_root_nested_lock_is_not_reacquired(
    exp1_v2: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    lock_calls: list[Path] = []

    @contextmanager
    def recording_lock(root: Path):
        lock_calls.append(root)
        yield root

    monkeypatch.setattr(runner, "exclusive_mutation_lock", recording_lock)

    def phase() -> None:
        with exp1_v2._exclusive_mutation_lock(tmp_path):
            pass

    exp1_v2.run_exp1_v2_workflow(
        config={},
        paths={"root": tmp_path, "logs": tmp_path / "logs"},
        preflight=phase,
        direction=lambda: None,
        generate=lambda: None,
        judge=lambda: None,
        analyze=lambda: None,
        engine_factory=lambda: None,
    )

    assert lock_calls == [tmp_path]


@pytest.mark.parametrize(
    ("section", "key", "value", "reason"),
    [
        ("model", "revision", "0" * 40, "model revision"),
        ("grid", "layers", [1, 5, 8], "layer list"),
        ("grid", "alpha", 0.04, "alpha"),
        ("grid", "schedule", "first_token", "schedule"),
        ("direction", "fit_per_class", 99, "fit count"),
        ("direction", "holdout_per_class", 49, "holdout count"),
        ("conditions", None, ["baseline"], "condition count"),
        ("generation", "n_prompts", 153, "HarmBench total"),
        ("generation", "max_new_tokens", 511, "decode max tokens"),
        ("generation", "decoding", "sampling", "decode mode"),
        ("generation", "temperature", 0.1, "decode temperature"),
        ("generation", "top_p", 0.9, "decode top-p"),
        ("generation", "enable_thinking", True, "decode thinking"),
        ("bootstrap", "n_resamples", 9999, "bootstrap resamples"),
        ("bootstrap", "confidence", 0.90, "bootstrap confidence"),
        ("bootstrap", "seed", 43, "bootstrap seed"),
    ],
)
def test_exp1_v2_protocol_rejects_any_drift(
    config: dict[str, object],
    exp1_v2: Any,
    section: str,
    key: str | None,
    value: object,
    reason: str,
) -> None:
    changed = json.loads(json.dumps(config))
    if section == "conditions":
        changed[section] = value
    else:
        assert key is not None
        cast(dict[str, object], changed[section])[key] = value

    with pytest.raises((AssertionError, ValueError), match=reason):
        exp1_v2.validate_exp1_v2_protocol(changed)


def test_exp1_v2_protocol_accepts_the_frozen_config(
    config: dict[str, object], exp1_v2: Any
) -> None:
    exp1_v2.validate_exp1_v2_protocol(config)


def test_exp1_v2_protocol_rejects_unexpected_top_level_keys(
    config: dict[str, object], exp1_v2: Any
) -> None:
    changed = json.loads(json.dumps(config))
    changed["unexpected_behavior"] = {"enabled": True}

    with pytest.raises(
        ValueError, match="protocol drift: top-level keys.*unexpected_behavior"
    ):
        exp1_v2.validate_exp1_v2_protocol(changed)


def test_exp1_v2_checkpoint_paths_are_namespace_isolated(
    exp1_v2: Any, tmp_path: Path
) -> None:
    baseline = exp1_v2.exp1_v2_checkpoint_path(tmp_path, "baseline")
    layered = exp1_v2.exp1_v2_checkpoint_path(tmp_path, "layer_1")

    assert baseline.parent == tmp_path / "exp1_v2"
    assert layered.parent == tmp_path / "exp1_v2"
    assert "exp1_direction" not in str(baseline)
    assert "exp1_generations" not in str(baseline)
    assert "exp1_trajectories" not in str(baseline)


def test_exp1_v2_cli_paths_are_confined_to_canonical_artifact_directories(
    exp1_v2: Any, tmp_path: Path
) -> None:
    paths = exp1_v2._cli_paths(tmp_path)

    assert paths["checkpoints"].is_relative_to(tmp_path / "checkpoints")
    assert paths["results"].is_relative_to(tmp_path / "results")
    assert paths["logs"].is_relative_to(tmp_path / "logs")
    assert paths["notification"].is_relative_to(tmp_path / "logs")
    assert paths["inputs"] != tmp_path / "inputs"
    assert paths["notification"] != tmp_path / "notification.json"


def test_exp1_v2_generation_uses_only_target_capture_layer_for_treatments(
    config: dict[str, object], exp1_v2: Any, tmp_path: Path
) -> None:
    prompts = [{"id": f"hb-{index}", "prompt": "prompt"} for index in range(154)]
    directions = {
        layer: SimpleNamespace(direction=[1.0], mean_norm=1.0) for layer in LAYERS
    }
    captures: list[tuple[str, list[int]]] = []

    def generate(
        condition: str, rows: list[dict[str, object]], spec: object, **kwargs: object
    ):
        del rows, spec
        captures.append((condition, list(cast(list[int], kwargs["capture_layers"]))))
        return [
            {"id": runner.condition_id(condition, prompts[index]["id"])}
            for index in range(154)
        ]

    exp1_v2.generate_exp1_v2(
        config,
        prompts,
        directions=directions,
        checkpoint_root=tmp_path,
        generate=generate,
    )

    assert captures[0] == ("baseline", LAYERS)
    assert all(
        layer_capture == [int(condition.removeprefix("layer_"))]
        for condition, layer_capture in captures[1:]
    )


def test_exp1_v2_id_denominators_are_generation_judge_1848_and_trajectory_3388(
    exp1_v2: Any,
) -> None:
    prompt_ids = [f"hb-{index}" for index in range(154)]
    generation_ids = {
        runner.condition_id(condition, prompt_id)
        for condition in exp1_v2.CONDITIONS
        for prompt_id in prompt_ids
    }
    trajectory_ids = {
        (generation_id, layer)
        for generation_id in generation_ids
        for layer in LAYERS
        if generation_id.split("/", 1)[0] == "baseline"
        or layer == int(generation_id.split("/", 1)[0].removeprefix("layer_"))
    }
    judge_ids = set(generation_ids)

    assert len(generation_ids) == 1848
    assert len(set(generation_ids)) == 1848
    assert len(judge_ids) == 1848
    assert len(trajectory_ids) == 3388


def test_exp1_v2_manifest_binds_protocol_sources_and_selected_dataset_ids(
    config: dict[str, object], exp1_v2: Any, tmp_path: Path
) -> None:
    source_a = tmp_path / "benign.jsonl"
    source_b = tmp_path / "harmful.jsonl"
    source_a.write_text('{"id":"b0"}\n', encoding="utf-8")
    source_b.write_text('{"id":"h0"}\n', encoding="utf-8")

    manifest = exp1_v2.build_exp1_v2_manifest(
        config,
        sources={"benign": source_a, "harmful": source_b},
        datasets={
            "benign": {"id": "LLM-LAT/benign-dataset", "revision": "pinned"},
            "harmful": {"id": "LLM-LAT/harmful-dataset", "revision": "pinned"},
            "harmbench": {"id": "harmbench", "revision": "sha"},
        },
        selected_ids={"benign": ["b0"], "harmful": ["h0"], "harmbench": ["hb0"]},
    )

    assert manifest["config_sha256"]
    assert set(manifest["source_sha256"]) == {"benign", "harmful"}
    assert manifest["aggregate_source_sha256"]
    assert manifest["datasets"]["benign"]["id"] == "LLM-LAT/benign-dataset"
    assert manifest["selected_ids"]["harmbench"] == ["hb0"]
    assert manifest["protocol"] == config


@pytest.mark.parametrize("manifest_state", ["missing", "stale"])
def test_exp1_v2_preflight_fails_closed_before_model_or_provider_construction(
    config: dict[str, object], exp1_v2: Any, tmp_path: Path, manifest_state: str
) -> None:
    manifest_path = tmp_path / "manifest.json"
    checkpoint = tmp_path / "exp1_v2" / "baseline.jsonl"
    checkpoint.parent.mkdir()
    checkpoint.write_text('{"id":"already-computed"}\n', encoding="utf-8")
    if manifest_state == "stale":
        manifest_path.write_text(
            json.dumps({"config_sha256": "0" * 64}), encoding="utf-8"
        )

    constructed: list[str] = []

    def model_factory() -> object:
        constructed.append("model")
        return object()

    def provider_factory() -> object:
        constructed.append("provider")
        return object()

    with pytest.raises((FileNotFoundError, RuntimeError, ValueError), match="manifest"):
        exp1_v2.preflight_exp1_v2(
            config,
            manifest_path=manifest_path,
            checkpoint_paths=[checkpoint],
            model_factory=model_factory,
            provider_factory=provider_factory,
        )
    assert constructed == []


def test_exp1_v2_phase_prerequisites_are_checked_at_consumer_boundary(
    exp1_v2: Any,
    config: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(runner, "load_config", lambda _path: config)
    monkeypatch.setattr(
        exp1_v2,
        "_cli_paths",
        lambda _root, _cache=None: {
            "logs": tmp_path / "logs",
            "log": tmp_path / "logs" / "run.log",
        },
    )
    monkeypatch.setattr(
        exp1_v2, "_validate_preflight_marker", lambda *_args: events.append("preflight")
    )
    monkeypatch.setattr(
        exp1_v2,
        "_validate_phase_marker",
        lambda _c, _p, phase: events.append(f"validate:{phase}"),
    )
    monkeypatch.setattr(
        exp1_v2, "_cli_direction", lambda *_args: events.append("direction")
    )
    monkeypatch.setattr(
        exp1_v2, "_cli_generate", lambda *_args: events.append("generate")
    )
    monkeypatch.setattr(
        exp1_v2,
        "_write_phase_marker",
        lambda _c, _p, phase: events.append(f"mark:{phase}"),
    )

    exp1_v2.main(["--root", str(tmp_path), "--phase", "direction", "generate"])

    assert events == [
        "preflight",
        "preflight",
        "direction",
        "mark:direction",
        "preflight",
        "validate:direction",
        "generate",
        "mark:generate",
    ]


def test_exp1_v2_cli_prepare_always_passes_exact_snapshot_checker_without_hub(
    config: dict[str, object],
    exp1_v2: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[object] = []

    def checker(model_ids: tuple[str, ...], *, cache_root: Path, load: bool) -> None:
        events.append(("checker", model_ids, cache_root, load))

    def fake_prepare(_config: dict[str, object], **kwargs: object) -> None:
        snapshot_checker = kwargs["model_snapshot_checker"]
        assert callable(snapshot_checker)
        cast(Any, snapshot_checker)(
            ("Qwen/Qwen3-4B",), cache_root=tmp_path / "cache", load=False
        )
        events.append("dataset")

    monkeypatch.setattr(exp1_v2, "_exp1_v2_model_snapshot_checker", lambda: checker)
    monkeypatch.setattr(exp1_v2, "prepare_exp1_v2_preflight", fake_prepare)
    paths = exp1_v2._cli_paths(tmp_path, tmp_path / "cache")

    exp1_v2._cli_prepare(config, paths)

    assert not (tmp_path / "cache" / "hub").exists()
    assert events == [
        ("checker", ("Qwen/Qwen3-4B",), tmp_path / "cache", False),
        "dataset",
    ]
