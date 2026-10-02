from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest
import torch
import yaml

from prefix import runner


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "run_exp1_v2.py"
LAYERS = (1, 5, 8, 12, 16, 19, 20, 23, 27, 30, 34)
CONDITIONS = ("baseline", *(f"layer_{layer}" for layer in LAYERS))
REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"


@pytest.fixture
def exp1_v2() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_exp1_v2_e2e", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _harmbench_rows() -> list[dict[str, str]]:
    return [
        {
            "id": f"hb-{category}-{index:03d}",
            "category": category,
            "behavior": f"request {category} {index:03d}",
            "source": "harmbench",
            "revision": "harmbench-fake-revision",
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


def _llm_lat_rows() -> list[dict[str, str]]:
    return [
        {
            "id": f"{label}-{index:03d}",
            "class": label,
            "prompt": f"{label} prompt {index:03d}",
            "source": "llm-lat-fake-cache",
            "revision": "llm-lat-fake-cache-revision",
        }
        for label in ("harmful", "benign")
        for index in range(180)
    ]


def test_exp1_v2_cli_all_fake_adapters_is_complete_and_resumable(
    exp1_v2: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = ROOT / "configs" / "exp1_v2.yaml"
    config = cast(dict[str, object], yaml.safe_load(config_path.read_text()))
    calls = {"capture": 0, "generate": 0, "judge": 0}

    def fake_snapshot_checker(
        model_ids: tuple[str, ...], *, cache_root: Path, load: bool
    ) -> dict[str, Path]:
        assert model_ids == ("Qwen/Qwen3-4B",)
        assert cache_root == tmp_path / "cache"
        assert load is False
        return {}

    monkeypatch.setattr(
        exp1_v2, "_exp1_v2_model_snapshot_checker", lambda: fake_snapshot_checker
    )

    from prefix import data, judge, runner as runner_module

    monkeypatch.setattr(
        data,
        "load_harmbench",
        lambda _cache_dir, *, offline: (
            list(_harmbench_rows())
            if offline
            else pytest.fail("fake E2E must remain offline")
        ),
    )
    monkeypatch.setattr(
        exp1_v2,
        "_cli_llm_lat_loader",
        lambda **kwargs: (
            list(_llm_lat_rows())
            if kwargs
            == {
                "dataset": "llm-lat",
                "n": 180,
                "cache_dir": tmp_path / "cache",
                "offline": True,
            }
            else pytest.fail(f"unexpected LLM-LAT loader arguments: {kwargs}")
        ),
    )

    class FakeTokenizer:
        def apply_chat_template(
            self, messages: list[dict[str, str]], **_: object
        ) -> str:
            return f"fake::{messages[0]['content']}"

    class FakeEngine:
        def get_tokenizer(self) -> FakeTokenizer:
            return FakeTokenizer()

    monkeypatch.setattr(
        runner_module,
        "get_engine",
        lambda *_args, **_kwargs: FakeEngine(),
    )

    def capture_prompt_hiddens(
        _engine: object, prompts: list[str], layers: list[int]
    ) -> dict[int, torch.Tensor]:
        calls["capture"] += 1
        return {
            layer: torch.tensor(
                [
                    [float(layer), 1.0 if index < 100 else 2.0]
                    for index in range(len(prompts))
                ]
            )
            for layer in layers
        }

    monkeypatch.setattr(runner_module, "capture_prompt_hiddens", capture_prompt_hiddens)

    def steered_generate(
        _engine: object,
        requests: list[str],
        max_tokens: int,
        spec: object,
        **kwargs: object,
    ) -> list[runner.GenerateResult]:
        del max_tokens, spec
        calls["generate"] += 1
        sink = cast(runner.CaptureSink, kwargs["sink"])
        capture_layers = [
            int(layer) for layer in cast(list[int], kwargs["capture_layers"])
        ]
        directions = cast(dict[int, list[torch.Tensor]], kwargs["scalar_directions"])
        results = []
        for request in requests:
            prompt_id = request.removeprefix("fake::request ")
            request_id = f"request/{prompt_id}"
            results.append(
                runner.GenerateResult(
                    text=f"deterministic {prompt_id}", request_id=request_id
                )
            )
            for layer in capture_layers:
                hidden = torch.tensor([float(layer), 1.0])
                norm = float(torch.linalg.vector_norm(hidden).item())
                dot = float(torch.dot(hidden, directions[layer][0]).item())
                for token_index in (1, 2):
                    sink.rows.append(
                        {
                            "request_id": request_id,
                            "layer": layer,
                            "k": token_index,
                            "dots": [dot + token_index],
                            "norm": norm,
                        }
                    )
        return results

    monkeypatch.setattr(runner_module, "steered_generate", steered_generate)

    class FakeJudge:
        def __init__(self, **_: object) -> None:
            pass

        def judge_safety(self, _request: str, _response: str) -> bool:
            calls["judge"] += 1
            return False

    monkeypatch.setattr(judge, "GeminiJudge", FakeJudge)

    argv = [
        "--config",
        str(config_path),
        "--root",
        str(tmp_path),
        "--cache-root",
        str(tmp_path / "cache"),
        "--phase",
        "all",
    ]
    exp1_v2.main(argv)

    paths = exp1_v2._cli_paths(tmp_path, tmp_path / "cache")
    manifest = runner.read_json(paths["manifest"])
    assert isinstance(manifest, dict)
    assert manifest["config_sha256"] == exp1_v2._preflight_config_sha256(config)
    assert manifest["manifest_sha256"] == exp1_v2._preflight_manifest_sha256(manifest)
    assert len(cast(list[object], manifest["expected_generation_ids"])) == 1848
    manifest_trajectory_ids = {
        str(identifier)
        for identifier in cast(list[object], manifest["expected_trajectory_ids"])
    }
    preflight = runner.read_json(paths["preflight"])
    assert isinstance(preflight, dict)
    assert preflight["status"] == "complete"
    assert paths["marker"].is_file()
    assert paths["directions"].is_file()
    assert paths["direction_metadata"].is_file()

    envelopes = [
        envelope
        for condition in CONDITIONS
        for envelope in runner.read_jsonl(paths["generations"] / f"{condition}.jsonl")
    ]
    assert len(envelopes) == 1848
    assert len({str(row["generation"]["id"]) for row in envelopes}) == 1848
    trajectories = runner.read_jsonl(paths["trajectories"])
    assert len(trajectories) == 3388
    materialized_trajectory_ids = {str(row["id"]) for row in trajectories}
    assert len(materialized_trajectory_ids) == 3388
    assert manifest_trajectory_ids == materialized_trajectory_ids
    assert all(row["layer"] == row["capture_layer"] for row in trajectories)
    assert all(
        row["steering_layer"] is None
        if row["condition"] == "baseline"
        else row["steering_layer"] == int(str(row["condition"])[6:])
        for row in trajectories
    )
    judged = runner.read_jsonl(paths["judged"])
    assert len(judged) == 1848
    analysis = runner.read_json(paths["results"] / "analysis.json")
    assert isinstance(analysis, dict)
    assert analysis["generation_denominator"] == 1848
    assert analysis["trajectory_denominator"] == 3388
    assert analysis["judge_denominator"] == 1848
    assert (
        paths["results"].joinpath("exp1_v2_report.pdf").read_bytes().startswith(b"%PDF")
    )
    assert not list(tmp_path.rglob("*.png"))
    assert calls["capture"] == 1
    assert calls["generate"] == len(CONDITIONS) * 5
    assert calls["judge"] == 1848

    before = dict(calls)
    exp1_v2.main(argv)
    assert calls == before
    assert len(runner.read_jsonl(paths["trajectories"])) == 3388
    assert len(runner.read_jsonl(paths["judged"])) == 1848
