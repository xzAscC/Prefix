from __future__ import annotations

from pathlib import Path
from typing import Any
import importlib.util
import json
import math
import re
import sys

import pytest
from prefix import judge as judge_module
from prefix import metrics


ROOT = Path(__file__).parents[1]
SCRIPT_PATH = ROOT / "scripts" / "run_exp1_v2.py"
CONDITIONS = (
    "baseline",
    *(f"layer_{layer}" for layer in (1, 5, 8, 12, 16, 19, 20, 23, 27, 30, 34)),
)
LAYERS = (1, 5, 8, 12, 16, 19, 20, 23, 27, 30, 34)
N_PROMPTS = 154
EXPECTED_IDS = {
    f"{condition}:prompt-{index}"
    for condition in CONDITIONS
    for index in range(N_PROMPTS)
}


@pytest.fixture(scope="module")
def exp1_v2() -> Any:
    spec = importlib.util.spec_from_file_location(
        "run_exp1_v2_postprocess", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _generated_rows(*, missing: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for identifier in sorted(EXPECTED_IDS):
        if identifier == missing:
            continue
        condition, prompt_id = identifier.split(":", 1)
        rows.append(
            {
                "id": identifier,
                "condition": condition,
                "prompt_id": prompt_id,
                "category": "cat-a" if prompt_id.endswith(tuple("02468")) else "cat-b",
                "request": f"harmful request {prompt_id}",
                "response": f"response {condition} {prompt_id}",
                "generation_status": "ok",
                "source_sha256": "source-sha",
                "config_sha256": "config-sha",
                "model_revision": "model-revision",
            }
        )
    return rows


def _judge_factory(outcomes: dict[str, object], calls: list[str]):
    class FakeJudge:
        def __init__(self, model: str) -> None:
            self.model = model

        def judge_safety(self, request: str, response: str) -> bool:
            calls.append(self.model)
            outcome = outcomes[self.model]
            if isinstance(outcome, BaseException):
                raise outcome
            return bool(outcome)

    return FakeJudge


def _write_cli_generation(
    paths: dict[str, Path],
    rows: list[dict[str, Any]],
) -> None:
    by_condition: dict[str, list[dict[str, Any]]] = {
        condition: [] for condition in CONDITIONS
    }
    for row in rows:
        by_condition[str(row["condition"])].append(row)
    paths["generations"].mkdir(parents=True)
    for condition, condition_rows in by_condition.items():
        (paths["generations"] / f"{condition}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in condition_rows),
            encoding="utf-8",
        )


def _prepared_manifest() -> dict[str, Any]:
    return {
        "expected_generation_ids": sorted(EXPECTED_IDS),
        "expected_judge_ids": sorted(EXPECTED_IDS),
        "provenance": {
            "manifest_sha256": "manifest-sha",
            "source_sha256": "source-sha",
            "config_sha256": "config-sha",
            "model_revision": "model-revision",
        },
    }


def _complete_pdf_report() -> dict[str, Any]:
    trajectory: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    supports = (154, 154, 153, 153, 151, 151, 148, 148, 143, 143, 137, 137)
    condition_layers = (("baseline", 19),) + tuple(
        (f"layer_{layer}", layer) for layer in LAYERS
    )
    for condition, layer in condition_layers:
        offset = -0.3 + layer / 1000 if condition == "baseline" else layer / 100
        for token_index, support in enumerate(supports, start=1):
            trajectory.append(
                {
                    "condition": condition,
                    "layer": layer,
                    "token_index": token_index,
                    "mean_cosine": offset + token_index / 1000,
                    "support": support,
                    "category_support": {
                        "cat-a": support // 2,
                        "cat-b": support - support // 2,
                    },
                }
            )
    for layer in LAYERS:
        for token_index, support in enumerate(supports, start=1):
            mean = layer / 1000 + token_index / 10000
            bootstrap_rows.append(
                {
                    "layer": layer,
                    "token_index": token_index,
                    "mean_difference": mean,
                    "lower": mean - 0.02,
                    "upper": mean + 0.02,
                    "support": support,
                    "category_support": {
                        "cat-a": support // 2,
                        "cat-b": support - support // 2,
                    },
                }
            )
    return {
        "layers": list(LAYERS),
        "trajectory": trajectory,
        "bootstrap": {"n_resamples": 10000, "seed": 42, "confidence": 0.95},
        "bootstrap_rows": bootstrap_rows,
    }


@pytest.fixture
def retained_pdf_figures(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, list[Any]]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figures: list[Any] = []
    subplots = plt.subplots

    def retain_figure(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
        figure, axes = subplots(*args, **kwargs)
        figures.append(figure)
        return figure, axes

    monkeypatch.setattr(plt, "subplots", retain_figure)
    return plt, figures


def _axis_with_ylabel(figure: Any, text: str) -> Any:
    matches = [axis for axis in figure.axes if text in axis.get_ylabel()]
    assert len(matches) == 1
    return matches[0]


def _line_with_ydata(axis: Any, expected: list[float]) -> Any:
    matches = [
        line for line in axis.lines if list(line.get_ydata()) == pytest.approx(expected)
    ]
    assert len(matches) == 1
    return matches[0]


def _non_color_encoding(line: Any) -> tuple[Any, Any]:
    return line.get_linestyle(), line.get_marker()


def _canonical_trajectory_rows(
    generated: list[dict[str, Any]], *, provenance: dict[str, str] | None = None
) -> list[dict[str, Any]]:
    bound = provenance or {
        "source_sha256": "source-sha",
        "config_sha256": "config-sha",
        "model_revision": "model-revision",
    }
    rows: list[dict[str, Any]] = []
    for generation in generated:
        condition = str(generation["condition"])
        prompt_id = str(generation["prompt_id"])
        capture_layers = LAYERS if condition == "baseline" else (int(condition[6:]),)
        for layer in capture_layers:
            rows.append(
                {
                    "id": f"{generation['id']}/capture_layer_{layer}",
                    "generation_id": generation["id"],
                    "prompt_id": prompt_id,
                    "condition": condition,
                    "category": generation["category"],
                    "layer": layer,
                    "capture_layer": layer,
                    "steering_layer": None
                    if condition == "baseline"
                    else int(condition[6:]),
                    "cosines": [0.1 + layer / 1000.0],
                    **bound,
                }
            )
    return rows


def test_analysis_accepts_manifest_derived_3388_trajectory_ids_and_emits_all_layer_cis(
    exp1_v2: Any, tmp_path: Path
) -> None:
    generated = _generated_rows()
    trajectories = _canonical_trajectory_rows(generated)
    judged = [
        {**row, "final_source": "primary", "label": "UNSAFE"} for row in generated
    ]

    first = exp1_v2.analyze_exp1_v2(
        generated_rows=generated,
        trajectory_rows=trajectories,
        judge_rows=judged,
        expected_ids=EXPECTED_IDS,
        output_dir=tmp_path / "first",
        bootstrap={"n_resamples": 10000, "seed": 42, "confidence": 0.95},
    )
    second = exp1_v2.analyze_exp1_v2(
        generated_rows=generated,
        trajectory_rows=trajectories,
        judge_rows=judged,
        expected_ids=EXPECTED_IDS,
        output_dir=tmp_path / "second",
        bootstrap={"n_resamples": 10000, "seed": 42, "confidence": 0.95},
    )

    bootstrap_rows = first["bootstrap_rows"]
    assert isinstance(bootstrap_rows, list)
    assert len(bootstrap_rows) == 11
    assert {row["layer"] for row in bootstrap_rows} == set(LAYERS)
    assert {row["layer"] for row in bootstrap_rows} >= {19, 20}
    assert all(
        all(
            math.isfinite(float(row[key]))
            for key in ("mean_difference", "lower", "upper")
        )
        for row in bootstrap_rows
    )
    assert bootstrap_rows == second["bootstrap_rows"]
    assert first["trajectory_denominator"] == 3388


@pytest.mark.parametrize(
    "mutation", ["missing", "extra", "duplicate", "mixed", "provenance"]
)
def test_analysis_rejects_invalid_trajectory_universe_before_metrics(
    exp1_v2: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    generated = _generated_rows()
    trajectories = _canonical_trajectory_rows(generated)
    if mutation == "missing":
        trajectories = trajectories[:-1]
    elif mutation == "extra":
        trajectories = [*trajectories, {**trajectories[0], "id": "stale/trajectory"}]
    elif mutation == "duplicate":
        trajectories = [*trajectories, trajectories[0]]
    elif mutation == "mixed":
        trajectories = [
            *trajectories[:-1],
            {**trajectories[0], "id": "layer_20/prompt-999/capture_layer_20"},
        ]
    else:
        trajectories[0] = {**trajectories[0], "source_sha256": "different-source"}
    judged = [
        {**row, "final_source": "primary", "label": "UNSAFE"} for row in generated
    ]

    monkeypatch.setattr(
        metrics,
        "aggregate_token_cosine_trajectories",
        lambda *_args, **_kwargs: pytest.fail(
            "metrics ran before trajectory validation"
        ),
    )
    with pytest.raises(
        ValueError, match="trajectory|3388|duplicate|provenance|complete"
    ):
        exp1_v2.analyze_exp1_v2(
            generated_rows=generated,
            trajectory_rows=trajectories,
            judge_rows=judged,
            expected_ids=EXPECTED_IDS,
            output_dir=tmp_path / mutation,
            bootstrap={"n_resamples": 10000, "seed": 42, "confidence": 0.95},
        )


def test_judge_requires_all_1848_composite_ids_before_any_provider_call(
    exp1_v2: Any, tmp_path: Path
) -> None:
    calls: list[str] = []
    with pytest.raises(ValueError, match="1848|composite|complete"):
        exp1_v2.judge_exp1_v2(
            _generated_rows(missing="layer_20:prompt-153"),
            expected_ids=EXPECTED_IDS,
            output_path=tmp_path / "judged.jsonl",
            checkpoint_path=tmp_path / "judge.checkpoint.json",
            primary_model="primary-model",
            fallback_model="fallback-model",
            judge_factory=_judge_factory(
                {"primary-model": True, "fallback-model": True}, calls
            ),
        )
    assert calls == []


def test_judge_fallback_keeps_immutable_model_config_provenance_and_resumes_per_result(
    exp1_v2: Any, tmp_path: Path
) -> None:
    rows = _generated_rows()
    calls: list[str] = []
    output = tmp_path / "judged.jsonl"
    checkpoint = tmp_path / "judge.checkpoint.json"
    config = {
        "config_sha256": "config-sha",
        "source_sha256": "source-sha",
        "model_revision": "model-revision",
    }

    first = exp1_v2.judge_exp1_v2(
        rows,
        expected_ids=EXPECTED_IDS,
        output_path=output,
        checkpoint_path=checkpoint,
        provenance=config,
        primary_model="primary-model",
        fallback_model="fallback-model",
        judge_factory=_judge_factory(
            {"primary-model": RuntimeError("blocked"), "fallback-model": True}, calls
        ),
    )
    assert len(first) == 1848
    assert all(row["final_source"] == "fallback" for row in first)
    assert all(row["primary"]["model"] == "primary-model" for row in first)
    assert all(row["fallback"]["model"] == "fallback-model" for row in first)
    assert all(row["provenance"] == config for row in first)

    calls.clear()
    resumed = exp1_v2.judge_exp1_v2(
        rows,
        expected_ids=EXPECTED_IDS,
        output_path=output,
        checkpoint_path=checkpoint,
        provenance=config,
        primary_model="primary-model",
        fallback_model="fallback-model",
        judge_factory=_judge_factory(
            {
                "primary-model": AssertionError("rejudged"),
                "fallback-model": AssertionError("rejudged"),
            },
            calls,
        ),
    )
    assert resumed == first
    assert calls == []


def test_judge_reconstructs_checkpoint_after_output_only_crash(
    exp1_v2: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = _generated_rows()
    output = tmp_path / "judged.jsonl"
    checkpoint = tmp_path / "judge.checkpoint.json"
    config = {
        "config_sha256": "config-sha",
        "source_sha256": "source-sha",
        "model_revision": "model-revision",
    }
    import prefix.runner as runner

    real_write = runner.write_json_atomic
    crashed = False

    def write_then_crash(path: str | Path, value: object) -> None:
        nonlocal crashed
        if not crashed and Path(path) == checkpoint:
            crashed = True
            raise RuntimeError("injected checkpoint crash")
        real_write(path, value)

    monkeypatch.setattr(runner, "write_json_atomic", write_then_crash)
    with pytest.raises(RuntimeError, match="checkpoint crash"):
        exp1_v2.judge_exp1_v2(
            rows,
            expected_ids=EXPECTED_IDS,
            output_path=output,
            checkpoint_path=checkpoint,
            provenance=config,
            primary_model="primary-model",
            fallback_model="fallback-model",
            judge_factory=_judge_factory(
                {"primary-model": True, "fallback-model": True}, []
            ),
        )

    assert output.is_file()
    assert not checkpoint.exists()
    calls: list[str] = []
    resumed = exp1_v2.judge_exp1_v2(
        rows,
        expected_ids=EXPECTED_IDS,
        output_path=output,
        checkpoint_path=checkpoint,
        provenance=config,
        primary_model="primary-model",
        fallback_model="fallback-model",
        judge_factory=_judge_factory(
            {"primary-model": True, "fallback-model": True}, calls
        ),
    )
    assert len(resumed) == 1848
    assert len(calls) == 1847
    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert state["status"] == "complete"
    assert len(state["completed_ids"]) == 1848


@pytest.mark.parametrize("change", ["primary", "fallback", "provenance"])
def test_judge_rejects_stale_checkpoint_bindings_before_provider_call(
    exp1_v2: Any, tmp_path: Path, change: str
) -> None:
    rows = _generated_rows()
    output = tmp_path / "judged.jsonl"
    checkpoint = tmp_path / "judge.checkpoint.json"
    config = {
        "config_sha256": "config-sha",
        "source_sha256": "source-sha",
        "model_revision": "model-revision",
    }
    exp1_v2.judge_exp1_v2(
        rows,
        expected_ids=EXPECTED_IDS,
        output_path=output,
        checkpoint_path=checkpoint,
        provenance=config,
        primary_model="primary-model",
        fallback_model="fallback-model",
        judge_factory=_judge_factory(
            {"primary-model": True, "fallback-model": True}, []
        ),
    )
    calls: list[str] = []
    kwargs: dict[str, object] = {
        "primary_model": "primary-model",
        "fallback_model": "fallback-model",
        "provenance": config,
    }
    if change == "primary":
        kwargs["primary_model"] = "changed-primary"
    elif change == "fallback":
        kwargs["fallback_model"] = "changed-fallback"
    else:
        kwargs["provenance"] = {**config, "config_sha256": "changed"}

    with pytest.raises(RuntimeError, match="stale judge"):
        exp1_v2.judge_exp1_v2(
            rows,
            expected_ids=EXPECTED_IDS,
            output_path=output,
            checkpoint_path=checkpoint,
            judge_factory=_judge_factory(
                {"changed-primary": pytest.fail, "changed-fallback": pytest.fail},
                calls,
            ),
            **kwargs,
        )
    assert calls == []


def test_judge_rejects_generation_content_drift_before_provider_call(
    exp1_v2: Any, tmp_path: Path
) -> None:
    rows = _generated_rows()
    output = tmp_path / "judged.jsonl"
    checkpoint = tmp_path / "judge.checkpoint.json"
    config = {
        "config_sha256": "config-sha",
        "source_sha256": "source-sha",
        "model_revision": "model-revision",
    }
    exp1_v2.judge_exp1_v2(
        rows,
        expected_ids=EXPECTED_IDS,
        output_path=output,
        checkpoint_path=checkpoint,
        provenance=config,
        primary_model="primary-model",
        fallback_model="fallback-model",
        judge_factory=_judge_factory(
            {"primary-model": True, "fallback-model": True}, []
        ),
    )
    mutated = [dict(row) for row in rows]
    mutated[0]["generation"] = "mutated generation"
    calls: list[str] = []
    with pytest.raises(RuntimeError, match="stale judge"):
        exp1_v2.judge_exp1_v2(
            mutated,
            expected_ids=EXPECTED_IDS,
            output_path=output,
            checkpoint_path=checkpoint,
            provenance=config,
            primary_model="primary-model",
            fallback_model="fallback-model",
            judge_factory=_judge_factory(
                {"primary-model": pytest.fail, "fallback-model": pytest.fail}, calls
            ),
        )
    assert calls == []


def test_judge_reports_unresolved_when_primary_and_fallback_fail(
    exp1_v2: Any, tmp_path: Path
) -> None:
    calls: list[str] = []
    result = exp1_v2.judge_exp1_v2(
        _generated_rows(),
        expected_ids=EXPECTED_IDS,
        output_path=tmp_path / "judged.jsonl",
        checkpoint_path=tmp_path / "judge.checkpoint.json",
        primary_model="primary-model",
        fallback_model="fallback-model",
        judge_factory=_judge_factory(
            {
                "primary-model": RuntimeError("primary error"),
                "fallback-model": RuntimeError("fallback error"),
            },
            calls,
        ),
    )
    assert all(row["final_source"] == "unresolved" for row in result)
    assert all(row["primary"]["error"] == "primary error" for row in result)
    assert all(row["fallback"]["error"] == "fallback error" for row in result)


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate", "mixed"])
def test_cli_judge_validates_manifest_universe_before_constructing_judges(
    exp1_v2: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    rows = _generated_rows()
    manifest = _prepared_manifest()
    if mutation == "missing":
        rows = rows[:-1]
    elif mutation == "extra":
        rows = [
            *rows,
            {**rows[0], "id": "baseline:prompt-999", "condition": "baseline"},
        ]
    elif mutation == "duplicate":
        rows = [*rows, rows[0]]
    elif mutation == "mixed":
        rows = [row for row in rows if row["id"] != "layer_20:prompt-153"] + [
            {**rows[0], "id": "layer_20:prompt-999", "condition": "layer_20"}
        ]

    paths = exp1_v2._cli_paths(tmp_path)
    _write_cli_generation(paths, rows)
    paths["manifest"].parent.mkdir(parents=True, exist_ok=True)
    paths["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    constructed: list[str] = []

    class FakeGeminiJudge:
        def __init__(self, model: str, **kwargs: object) -> None:
            constructed.append(model)

        def judge_safety(self, request: str, response: str) -> bool:
            del request, response
            return True

    monkeypatch.setattr(judge_module, "GeminiJudge", FakeGeminiJudge)
    with pytest.raises(ValueError, match="1848|manifest|complete|duplicate"):
        exp1_v2._cli_judge(
            {
                "judges": {
                    "primary": {"model": "gemini-3.7-flash", "region": "global"},
                    "fallback": {
                        "model": "gemini-3.5-flash-lite",
                        "region": "global",
                    },
                }
            },
            paths,
        )
    assert constructed == []


def test_cli_judge_uses_prepared_manifest_and_preserves_envelope_provenance(
    exp1_v2: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = _generated_rows()
    for row in rows:
        row["prompt"] = f"original prompt {row['id']}"
    paths = exp1_v2._cli_paths(tmp_path)
    _write_cli_generation(paths, rows)
    manifest = _prepared_manifest()
    paths["manifest"].parent.mkdir(parents=True, exist_ok=True)
    paths["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    calls: list[tuple[str, str, str, str]] = []

    class FakeGeminiJudge:
        def __init__(
            self, model: str, *, region: str, temperature: float = 0.0
        ) -> None:
            self.model = model
            self.region = region
            self.temperature = temperature

        def judge_safety(self, request: str, response: str) -> bool:
            assert request and response
            calls.append((self.model, self.region, request, response))
            if self.model == "gemini-3.7-flash":
                raise RuntimeError("primary blocked")
            return True

    monkeypatch.setattr(judge_module, "GeminiJudge", FakeGeminiJudge)
    config = {
        "judges": {
            "primary": {
                "model": "gemini-3.7-flash",
                "region": "global",
                "temperature": 0.0,
            },
            "fallback": {
                "model": "gemini-3.5-flash-lite",
                "region": "global",
                "temperature": 0.0,
            },
        }
    }
    exp1_v2._cli_judge(config, paths)

    judged = [json.loads(line) for line in paths["judged"].read_text().splitlines()]
    assert len(judged) == 1848
    assert len(calls) == 3696
    assert {call[0] for call in calls} == {
        "gemini-3.7-flash",
        "gemini-3.5-flash-lite",
    }
    assert {call[1] for call in calls} == {"global"}
    for result in judged:
        original = next(row for row in rows if row["id"] == result["id"])
        assert result["id"] == original["id"]
        assert result["prompt"] == original["prompt"]
        assert result["category"] == original["category"]
        assert result["condition"] == original["condition"]
        assert result["provenance"] == manifest["provenance"]
        assert result["primary"]["model"] == "gemini-3.7-flash"
        assert result["primary"]["region"] == "global"
        assert result["primary"]["attempts"] == 1
        assert result["primary"]["region"] == "global"
        assert result["primary"]["attempts"] == 1
        assert result["fallback"]["model"] == "gemini-3.5-flash-lite"
        assert result["fallback"]["region"] == "global"
        assert result["fallback"]["attempts"] == 1
        assert result["final_source"] == "fallback"


def test_analysis_requires_complete_generation_trajectory_and_judge_denominators(
    exp1_v2: Any, tmp_path: Path
) -> None:
    generated = _generated_rows()
    trajectories = [
        {
            **row,
            "layer": 19 if row["condition"] == "baseline" else 20,
            "cosines": [0.1, 0.2],
        }
        for row in generated
    ]
    judged = [
        {**row, "final_source": "primary", "label": "UNSAFE"} for row in generated
    ]
    for name, broken in (
        ("generation", generated[:-1]),
        ("trajectory", trajectories[:-1]),
        ("judge", judged[:-1]),
    ):
        with pytest.raises(ValueError, match=f"{name}|denominator|complete"):
            exp1_v2.analyze_exp1_v2(
                generated_rows=broken if name == "generation" else generated,
                trajectory_rows=broken if name == "trajectory" else trajectories,
                judge_rows=broken if name == "judge" else judged,
                expected_ids=EXPECTED_IDS,
                output_dir=tmp_path / name,
                bootstrap={"n_resamples": 10000, "seed": 42, "confidence": 0.95},
            )


def test_analysis_uses_category_macro_paired_bootstrap_persists_units_and_preserves_layers(
    exp1_v2: Any, tmp_path: Path
) -> None:
    generated = _generated_rows()
    trajectories = [
        {
            **row,
            "layer": int(row["condition"].split("_")[-1])
            if row["condition"] != "baseline"
            else 0,
            "cosines": [0.1, 0.2],
        }
        for row in generated
    ]
    judged = [
        {
            **row,
            "final_source": "unresolved"
            if row["prompt_id"] == "prompt-153"
            else "primary",
            "label": None if row["prompt_id"] == "prompt-153" else "UNSAFE",
        }
        for row in generated
    ]
    report = exp1_v2.analyze_exp1_v2(
        generated_rows=generated,
        trajectory_rows=trajectories,
        judge_rows=judged,
        expected_ids=EXPECTED_IDS,
        output_dir=tmp_path,
        bootstrap={"n_resamples": 10000, "seed": 42, "confidence": 0.95},
    )
    assert report["unresolved_count"] == 12
    assert report["bootstrap"] == {"n_resamples": 10000, "seed": 42, "confidence": 0.95}
    assert {19, 20} <= {row["layer"] for row in report["trajectory"]}
    assert all("category_support" in row for row in report["trajectory"])
    assert all((tmp_path / f"{condition}.json").exists() for condition in CONDITIONS)


def test_analysis_aggregates_selected_nested_safety_labels_and_baseline_deltas(
    exp1_v2: Any, tmp_path: Path
) -> None:
    generated = _generated_rows()
    trajectories = _canonical_trajectory_rows(generated)
    judged: list[dict[str, Any]] = []
    for row in generated:
        prompt_index = int(str(row["prompt_id"]).removeprefix("prompt-"))
        unresolved = prompt_index == 153
        fallback_selected = prompt_index == 152
        if unresolved:
            final_source = "unresolved"
            primary = {"status": "error", "label": None}
            fallback = {"status": "error", "label": None}
        elif fallback_selected:
            final_source = "fallback"
            primary = {"status": "error", "label": None}
            fallback = {"status": "ok", "label": "SAFE"}
        else:
            final_source = "primary"
            label = (
                "SAFE"
                if row["condition"] == "baseline" and prompt_index % 2 == 0
                else "UNSAFE"
            )
            primary = {"status": "ok", "label": label}
            fallback = None
        judged.append(
            {
                **row,
                "primary": primary,
                "fallback": fallback,
                "final_source": final_source,
            }
        )

    report = exp1_v2.analyze_exp1_v2(
        generated_rows=generated,
        trajectory_rows=trajectories,
        judge_rows=judged,
        expected_ids=EXPECTED_IDS,
        output_dir=tmp_path,
        bootstrap={"n_resamples": 10000, "seed": 42, "confidence": 0.95},
    )

    safety = report["safety"]
    assert isinstance(safety, dict)
    by_condition = safety["by_condition"]
    assert by_condition["baseline"] == {
        "denominator": 154,
        "resolved_count": 153,
        "unresolved_count": 1,
        "safe_count": 77,
        "unsafe_count": 76,
        "safe_rate": pytest.approx(77 / 154),
        "unsafe_rate": pytest.approx(76 / 154),
        "unsafe_delta": 0,
        "unsafe_rate_delta": 0,
    }
    assert by_condition["layer_1"]["unsafe_count"] == 152
    assert by_condition["layer_1"]["unsafe_rate"] == pytest.approx(152 / 154)
    assert by_condition["layer_1"]["unsafe_delta"] == 76
    assert by_condition["layer_1"]["unsafe_rate_delta"] == pytest.approx(76 / 154)
    assert report["unresolved_count"] == 12


def test_analysis_rejects_stale_schema_version_checkpoint(
    exp1_v2: Any, tmp_path: Path
) -> None:
    generated = _generated_rows()
    trajectories = _canonical_trajectory_rows(generated)
    judged = [
        {
            **row,
            "primary": {"status": "ok", "label": "UNSAFE"},
            "fallback": None,
            "final_source": "primary",
        }
        for row in generated
    ]
    checkpoint = tmp_path / "analysis.checkpoint.json"
    kwargs = {
        "generated_rows": generated,
        "trajectory_rows": trajectories,
        "judge_rows": judged,
        "expected_ids": EXPECTED_IDS,
        "output_dir": tmp_path / "results",
        "checkpoint_path": checkpoint,
        "bootstrap": {"n_resamples": 10000, "seed": 42, "confidence": 0.95},
    }
    exp1_v2.analyze_exp1_v2(**kwargs)
    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    state["version"] = 1
    checkpoint.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(RuntimeError, match="stale analysis checkpoint"):
        exp1_v2.analyze_exp1_v2(**kwargs)


def test_analysis_checkpoints_each_condition_and_resumes_without_recomputing(
    exp1_v2: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generated = _generated_rows()
    trajectories = _canonical_trajectory_rows(generated)
    judged = [
        {**row, "final_source": "primary", "label": "UNSAFE"} for row in generated
    ]
    checkpoint = tmp_path / "analysis.checkpoint.json"
    calls: list[str] = []
    real_bootstrap = metrics.paired_bootstrap_ci

    def interrupt_after_baseline(*args: Any, **kwargs: Any) -> Any:
        condition = str(kwargs["treatment"])
        calls.append(condition)
        if len(calls) == 2:
            raise RuntimeError("injected analysis interruption")
        return real_bootstrap(*args, **kwargs)

    monkeypatch.setattr(metrics, "paired_bootstrap_ci", interrupt_after_baseline)
    with pytest.raises(RuntimeError, match="interruption"):
        exp1_v2.analyze_exp1_v2(
            generated_rows=generated,
            trajectory_rows=trajectories,
            judge_rows=judged,
            expected_ids=EXPECTED_IDS,
            output_dir=tmp_path / "results",
            checkpoint_path=checkpoint,
            bootstrap={"n_resamples": 10000, "seed": 42, "confidence": 0.95},
        )

    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert state["completed_conditions"] == ["baseline", "layer_1"]
    assert (tmp_path / "results" / "baseline.json").is_file()
    assert not (tmp_path / "results" / "analysis.json").exists()
    assert not (tmp_path / "results" / "exp1_v2_report.pdf").exists()

    monkeypatch.setattr(metrics, "paired_bootstrap_ci", real_bootstrap)
    report = exp1_v2.analyze_exp1_v2(
        generated_rows=generated,
        trajectory_rows=trajectories,
        judge_rows=judged,
        expected_ids=EXPECTED_IDS,
        output_dir=tmp_path / "results",
        checkpoint_path=checkpoint,
        bootstrap={"n_resamples": 10000, "seed": 42, "confidence": 0.95},
    )
    assert report["generation_denominator"] == 1848
    assert json.loads(checkpoint.read_text(encoding="utf-8"))["status"] == "complete"


@pytest.mark.parametrize(
    "artifact", ["generation", "trajectory", "judge", "checkpoint"]
)
def test_analysis_rejects_tampered_content_bound_resume_inputs(
    exp1_v2: Any, tmp_path: Path, artifact: str
) -> None:
    generated = _generated_rows()
    trajectories = _canonical_trajectory_rows(generated)
    judged = [
        {**row, "final_source": "primary", "label": "UNSAFE"} for row in generated
    ]
    checkpoint = tmp_path / "analysis.checkpoint.json"
    output = tmp_path / "results"
    exp1_v2.analyze_exp1_v2(
        generated_rows=generated,
        trajectory_rows=trajectories,
        judge_rows=judged,
        expected_ids=EXPECTED_IDS,
        output_dir=output,
        checkpoint_path=checkpoint,
        bootstrap={"n_resamples": 10000, "seed": 42, "confidence": 0.95},
    )
    if artifact == "generation":
        generated[0]["response"] = "tampered"
    elif artifact == "trajectory":
        trajectories[0]["cosines"] = [9.9]
    elif artifact == "judge":
        judged[0]["label"] = "SAFE"
    else:
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        state["input_sha256"] = "tampered"
        checkpoint.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises((RuntimeError, ValueError), match="stale|drift|tamper|binding"):
        exp1_v2.analyze_exp1_v2(
            generated_rows=generated,
            trajectory_rows=trajectories,
            judge_rows=judged,
            expected_ids=EXPECTED_IDS,
            output_dir=output,
            checkpoint_path=checkpoint,
            bootstrap={"n_resamples": 10000, "seed": 42, "confidence": 0.95},
        )


def test_pdf_is_written_before_any_png_and_contains_layer_19_and_20(
    exp1_v2: Any, tmp_path: Path
) -> None:
    pdf = tmp_path / "exp1_v2_report.pdf"
    exp1_v2.write_exp1_v2_pdf(
        {"layers": [19, 20], "trajectory": [{"layer": 19}, {"layer": 20}]},
        pdf_path=pdf,
        output_dir=tmp_path,
    )
    assert pdf.exists()
    assert pdf.read_bytes().startswith(b"%PDF")
    assert not list(tmp_path.glob("*.png"))


def test_complete_pdf_has_three_nonempty_one_based_panels_and_cleans_up(
    exp1_v2: Any,
    tmp_path: Path,
    retained_pdf_figures: tuple[Any, list[Any]],
) -> None:
    plt, figures = retained_pdf_figures
    report = _complete_pdf_report()
    assert {row["condition"] for row in report["trajectory"]} == set(CONDITIONS)
    assert {row["layer"] for row in report["bootstrap_rows"]} == set(LAYERS)
    assert all(row["lower"] < row["upper"] for row in report["bootstrap_rows"])

    pdf = tmp_path / "complete-report.pdf"
    exp1_v2.write_exp1_v2_pdf(report, pdf_path=pdf, output_dir=tmp_path)

    assert pdf.read_bytes().startswith(b"%PDF")
    assert not list(tmp_path.rglob("*.png"))
    assert {path for path in tmp_path.rglob("*") if path.is_file()} == {pdf}
    assert len(figures) == 1
    figure = figures[0]
    assert len(figure.axes) == 3
    trajectory_axis = _axis_with_ylabel(figure, "mean cosine")
    interval_axis = _axis_with_ylabel(figure, "cosine difference")
    support_axis = _axis_with_ylabel(figure, "support")
    assert trajectory_axis.lines
    assert interval_axis.lines and interval_axis.collections
    assert support_axis.lines or support_axis.texts
    lower, upper = support_axis.get_xlim()
    assert lower <= 1 <= upper
    assert not plt.fignum_exists(figure.number)
    assert any(math.isclose(float(tick), 1.0) for tick in support_axis.get_xticks())


def test_complete_pdf_styles_and_legend_are_accessible_without_color(
    exp1_v2: Any,
    tmp_path: Path,
    retained_pdf_figures: tuple[Any, list[Any]],
) -> None:
    _, figures = retained_pdf_figures
    report = _complete_pdf_report()
    exp1_v2.write_exp1_v2_pdf(
        report,
        pdf_path=tmp_path / "accessible-styles.pdf",
        output_dir=tmp_path,
    )

    figure = figures[0]
    trajectory_axis = _axis_with_ylabel(figure, "mean cosine")

    def values(condition: str, layer: int) -> list[float]:
        offset = -0.3 + layer / 1000 if condition == "baseline" else layer / 100
        return [offset + token_index / 1000 for token_index in range(1, 13)]

    plotted = {
        "Baseline": _line_with_ydata(trajectory_axis, values("baseline", 19)),
        "Layer 19": _line_with_ydata(trajectory_axis, values("layer_19", 19)),
        "Layer 20": _line_with_ydata(trajectory_axis, values("layer_20", 20)),
    }
    assert len({_non_color_encoding(line) for line in plotted.values()}) == 3

    assert len(figure.legends) == 1
    legend = figure.legends[0]
    labels = [text.get_text() for text in legend.get_texts()]
    handles = dict(zip(labels, legend.legend_handles, strict=True))
    assert set(labels) == {"Baseline", *(f"Layer {layer}" for layer in LAYERS)}
    for label, line in plotted.items():
        assert _non_color_encoding(handles[label]) == _non_color_encoding(line)


def test_complete_pdf_summarizes_support_shared_by_all_conditions(
    exp1_v2: Any,
    tmp_path: Path,
    retained_pdf_figures: tuple[Any, list[Any]],
) -> None:
    _, figures = retained_pdf_figures
    exp1_v2.write_exp1_v2_pdf(
        _complete_pdf_report(),
        pdf_path=tmp_path / "shared-support.pdf",
        output_dir=tmp_path,
    )

    support_axis = _axis_with_ylabel(figures[0], "support")
    support_lines = [line for line in support_axis.lines if len(line.get_xdata())]
    annotation = " ".join(text.get_text() for text in support_axis.texts)
    explicitly_shared = re.search(
        r"\b(all conditions|identical|same|shared)\b", annotation, re.IGNORECASE
    )
    assert len(support_lines) == 1 or explicitly_shared, (
        "identical condition support must be one summary trace or explicitly annotated"
    )


@pytest.mark.parametrize("bootstrap_rows", [None, []], ids=["missing", "empty"])
def test_enabled_bootstrap_without_intervals_is_rejected_or_disclosed(
    exp1_v2: Any,
    tmp_path: Path,
    retained_pdf_figures: tuple[Any, list[Any]],
    bootstrap_rows: list[dict[str, Any]] | None,
) -> None:
    plt, figures = retained_pdf_figures
    report = _complete_pdf_report()
    if bootstrap_rows is None:
        report.pop("bootstrap_rows")
    else:
        report["bootstrap_rows"] = bootstrap_rows
    pdf = tmp_path / "missing-intervals.pdf"

    try:
        exp1_v2.write_exp1_v2_pdf(report, pdf_path=pdf, output_dir=tmp_path)
    except ValueError as error:
        assert re.search(
            r"bootstrap|confidence|interval|uncertainty|\bCI\b",
            str(error),
            re.IGNORECASE,
        )
        assert not list(tmp_path.rglob("*.png"))
        assert not [path for path in tmp_path.rglob("*") if path.is_file()]
        return

    assert pdf.read_bytes().startswith(b"%PDF")
    assert len(figures) == 1
    figure = figures[0]
    visible_text = " ".join(
        [text.get_text() for text in figure.texts]
        + [
            text.get_text()
            for axis in figure.axes
            for text in (*axis.texts, axis.title, axis.yaxis.label)
        ]
    )
    assert re.search(
        r"(bootstrap|confidence|interval|uncertainty|\bCI\b).*"
        r"(missing|empty|unavailable|not available|no )"
        r"|(missing|empty|unavailable|not available|no ).*"
        r"(bootstrap|confidence|interval|uncertainty|\bCI\b)",
        visible_text,
        re.IGNORECASE,
    )
    assert not plt.fignum_exists(figure.number)
    assert not list(tmp_path.rglob("*.png"))
    assert {path for path in tmp_path.rglob("*") if path.is_file()} == {pdf}


def test_terminal_workflow_orders_phases_avoids_cpu_engines_logs_under_logs_and_notifies_once(
    exp1_v2: Any, tmp_path: Path
) -> None:
    events: list[str] = []
    notifications: list[str] = []
    exp1_v2.run_exp1_v2_workflow(
        config={"conditions": list(CONDITIONS)},
        paths={"root": tmp_path, "logs": tmp_path / "logs"},
        preflight=lambda: events.append("preflight"),
        direction=lambda: events.append("direction"),
        generate=lambda: events.append("generate"),
        judge=lambda: events.append("judge"),
        analyze=lambda: events.append("analyze"),
        engine_factory=lambda: (_ for _ in ()).throw(
            AssertionError("CPU phase constructed engine")
        ),
        notify=lambda status, state_path: notifications.append(status),
    )
    assert events == ["preflight", "direction", "generate", "judge", "analyze"]
    assert notifications == ["completed"]
    assert all(
        path.parent == tmp_path / "logs" for path in (tmp_path / "logs").glob("*")
    )


def test_terminal_workflow_sends_exactly_one_failure_notification(
    exp1_v2: Any, tmp_path: Path
) -> None:
    notifications: list[str] = []
    with pytest.raises(RuntimeError, match="generate failed"):
        exp1_v2.run_exp1_v2_workflow(
            config={"conditions": list(CONDITIONS)},
            paths={"root": tmp_path, "logs": tmp_path / "logs"},
            preflight=lambda: None,
            direction=lambda: None,
            generate=lambda: (_ for _ in ()).throw(RuntimeError("generate failed")),
            judge=lambda: pytest.fail("judge ran after generation failure"),
            analyze=lambda: pytest.fail("analyze ran after generation failure"),
            engine_factory=lambda: object(),
            notify=lambda status, state_path: notifications.append(status),
        )
    assert notifications == ["failed"]


def test_terminal_workflow_uses_run_scoped_notification_state_and_finalizes_once(
    exp1_v2: Any, tmp_path: Path
) -> None:
    events: list[str] = []
    notifications: list[tuple[str, Path]] = []
    state_path = tmp_path / "checkpoints" / "exp1_v2" / "notification.json"

    def finalize(status: str, state: Path) -> str:
        notifications.append((status, state))
        return "sent"

    exp1_v2.run_exp1_v2_workflow(
        config={},
        paths={
            "root": tmp_path,
            "logs": tmp_path / "logs",
            "notification_state": state_path,
        },
        preflight=lambda: events.append("prepare"),
        direction=lambda: events.append("direction"),
        generate=lambda: events.append("generate"),
        judge=lambda: events.append("judge"),
        analyze=lambda: events.append("analyze"),
        engine_factory=lambda: pytest.fail("engine constructed"),
        notify=finalize,
    )

    assert events == ["prepare", "direction", "generate", "judge", "analyze"]
    assert notifications == [("completed", state_path)]
    assert state_path.parent == tmp_path / "checkpoints" / "exp1_v2"
    assert not (tmp_path / "notification.json").exists()


def test_terminal_workflow_does_not_run_downstream_phases_after_upstream_failure(
    exp1_v2: Any, tmp_path: Path
) -> None:
    events: list[str] = []
    notifications: list[tuple[str, Path]] = []
    state_path = tmp_path / "checkpoints" / "exp1_v2" / "notification.json"

    with pytest.raises(RuntimeError, match="direction failed"):
        exp1_v2.run_exp1_v2_workflow(
            config={},
            paths={
                "root": tmp_path,
                "logs": tmp_path / "logs",
                "notification_state": state_path,
            },
            preflight=lambda: events.append("prepare"),
            direction=lambda: (
                events.append("direction"),
                (_ for _ in ()).throw(RuntimeError("direction failed")),
            )[-1],
            generate=lambda: pytest.fail("generate ran after direction failure"),
            judge=lambda: pytest.fail("judge ran after direction failure"),
            analyze=lambda: pytest.fail("analyze ran after direction failure"),
            engine_factory=lambda: pytest.fail("engine constructed"),
            notify=lambda status, state: (
                notifications.append((status, state)) or "failed"
            ),
        )

    assert events == ["prepare", "direction"]
    assert notifications == [("failed", state_path)]


def test_terminal_workflow_does_not_overwrite_completed_workflow_delivery_failure(
    exp1_v2: Any, tmp_path: Path
) -> None:
    state_path = tmp_path / "checkpoints" / "exp1_v2" / "notification.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps(
            {
                "task": "exp1-v2",
                "workflow_status": "completed",
                "delivery_status": "failed",
                "status": "failed",
            }
        ),
        encoding="utf-8",
    )
    events: list[str] = []
    notifications: list[tuple[str, Path]] = []

    def retry(status: str, path: Path) -> str:
        notifications.append((status, path))
        return "sent"

    exp1_v2.run_exp1_v2_workflow(
        config={},
        paths={
            "root": tmp_path,
            "logs": tmp_path / "logs",
            "notification_state": state_path,
        },
        preflight=lambda: events.append("prepare"),
        direction=lambda: events.append("direction"),
        generate=lambda: events.append("generate"),
        judge=lambda: events.append("judge"),
        analyze=lambda: events.append("analyze"),
        engine_factory=lambda: pytest.fail("engine constructed"),
        notify=retry,
    )

    assert events == []
    assert notifications == [("completed", state_path)]
    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved["workflow_status"] == "completed"
    assert saved["delivery_status"] == "failed"


def test_terminal_workflow_persists_workflow_status_when_delivery_fails(
    exp1_v2: Any, tmp_path: Path
) -> None:
    state_path = tmp_path / "checkpoints" / "exp1_v2" / "notification.json"

    def failed_delivery(status: str, path: Path) -> str:
        del status, path
        return "failed"

    exp1_v2.run_exp1_v2_workflow(
        config={},
        paths={
            "root": tmp_path,
            "logs": tmp_path / "logs",
            "notification_state": state_path,
        },
        preflight=lambda: None,
        direction=lambda: None,
        generate=lambda: None,
        judge=lambda: None,
        analyze=lambda: None,
        engine_factory=lambda: pytest.fail("engine constructed"),
        notify=failed_delivery,
    )

    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved["workflow_status"] == "completed"
    assert saved["delivery_status"] == "failed"


def test_terminal_workflow_rejects_unhashable_persisted_workflow_status(
    exp1_v2: Any, tmp_path: Path
) -> None:
    state_path = tmp_path / "checkpoints" / "exp1_v2" / "notification.json"
    state_path.parent.mkdir(parents=True)
    original = json.dumps({"workflow_status": [], "delivery_status": "pending"})
    state_path.write_text(original, encoding="utf-8")
    events: list[str] = []

    with pytest.raises(RuntimeError, match="terminal workflow state"):
        exp1_v2.run_exp1_v2_workflow(
            config={},
            paths={
                "root": tmp_path,
                "logs": tmp_path / "logs",
                "notification_state": state_path,
            },
            preflight=lambda: events.append("prepare"),
            direction=lambda: events.append("direction"),
            generate=lambda: events.append("generate"),
            judge=lambda: events.append("judge"),
            analyze=lambda: events.append("analyze"),
            engine_factory=lambda: pytest.fail("engine constructed"),
            notify=lambda status, path: pytest.fail("notification attempted"),
        )

    assert events == []
    assert state_path.read_text(encoding="utf-8") == original


def test_terminal_workflow_preserves_original_failure_when_state_persistence_fails(
    exp1_v2: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_path = tmp_path / "checkpoints" / "exp1_v2" / "notification.json"
    from prefix import notify as notify_module

    monkeypatch.setattr(
        notify_module,
        "record_terminal_workflow",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            OSError("notification disk failure")
        ),
    )

    with pytest.raises(RuntimeError, match="workflow failed"):
        exp1_v2.run_exp1_v2_workflow(
            config={},
            paths={
                "root": tmp_path,
                "logs": tmp_path / "logs",
                "notification_state": state_path,
            },
            preflight=lambda: None,
            direction=lambda: None,
            generate=lambda: (_ for _ in ()).throw(RuntimeError("workflow failed")),
            judge=lambda: pytest.fail("judge ran after generation failure"),
            analyze=lambda: pytest.fail("analyze ran after generation failure"),
            engine_factory=lambda: pytest.fail("engine constructed"),
            notify=lambda status, path: pytest.fail("notification attempted"),
        )


def test_terminal_workflow_rejects_unknown_delivery_status_before_retry(
    exp1_v2: Any, tmp_path: Path
) -> None:
    state_path = tmp_path / "checkpoints" / "exp1_v2" / "notification.json"
    state_path.parent.mkdir(parents=True)
    original = json.dumps(
        {
            "task": "exp1-v2",
            "workflow_status": "completed",
            "delivery_status": "bogus",
        }
    )
    state_path.write_text(original, encoding="utf-8")
    events: list[str] = []

    with pytest.raises(RuntimeError, match="delivery state"):
        exp1_v2.run_exp1_v2_workflow(
            config={},
            paths={
                "root": tmp_path,
                "logs": tmp_path / "logs",
                "notification_state": state_path,
            },
            preflight=lambda: events.append("prepare"),
            direction=lambda: events.append("direction"),
            generate=lambda: events.append("generate"),
            judge=lambda: events.append("judge"),
            analyze=lambda: events.append("analyze"),
            engine_factory=lambda: pytest.fail("engine constructed"),
            notify=lambda status, path: pytest.fail("notification attempted"),
        )

    assert events == []
    assert state_path.read_text(encoding="utf-8") == original


def test_terminal_workflow_rejects_unknown_delivery_without_workflow_status(
    exp1_v2: Any, tmp_path: Path
) -> None:
    state_path = tmp_path / "checkpoints" / "exp1_v2" / "notification.json"
    state_path.parent.mkdir(parents=True)
    original = json.dumps({"delivery_status": "bogus"})
    state_path.write_text(original, encoding="utf-8")
    events: list[str] = []

    with pytest.raises(RuntimeError, match="delivery state"):
        exp1_v2.run_exp1_v2_workflow(
            config={},
            paths={
                "root": tmp_path,
                "logs": tmp_path / "logs",
                "notification_state": state_path,
            },
            preflight=lambda: events.append("prepare"),
            direction=lambda: events.append("direction"),
            generate=lambda: events.append("generate"),
            judge=lambda: events.append("judge"),
            analyze=lambda: events.append("analyze"),
            engine_factory=lambda: pytest.fail("engine constructed"),
            notify=lambda status, path: pytest.fail("notification attempted"),
        )

    assert events == []
    assert state_path.read_text(encoding="utf-8") == original


def test_terminal_workflow_keeps_claim_when_callback_result_persistence_fails(
    exp1_v2: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import prefix.runner as runner

    state_path = tmp_path / "checkpoints" / "exp1_v2" / "notification.json"
    real_write = runner.write_json_atomic

    def fail_delivery_write(path: str | Path, value: object) -> None:
        if Path(path) == state_path:
            raise OSError("delivery state write failed")
        real_write(path, value)

    monkeypatch.setattr(runner, "write_json_atomic", fail_delivery_write)

    with pytest.raises(OSError, match="delivery state write failed"):
        exp1_v2.run_exp1_v2_workflow(
            config={},
            paths={
                "root": tmp_path,
                "logs": tmp_path / "logs",
                "notification_state": state_path,
            },
            preflight=lambda: None,
            direction=lambda: None,
            generate=lambda: None,
            judge=lambda: None,
            analyze=lambda: None,
            engine_factory=lambda: pytest.fail("engine constructed"),
            notify=lambda status, path: "sent",
        )

    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved["workflow_status"] == "completed"
    assert saved["delivery_status"] == "claimed"
