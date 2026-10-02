from __future__ import annotations

import importlib
from types import ModuleType
from pathlib import Path

import yaml

import pytest


MODEL_IDS = (
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-14B",
    "allenai/Olmo-3-7B-Think",
    "allenai/Olmo-3-32B-Think",
)

EXPECTED_MODELS = (
    {
        "id": "Qwen/Qwen3-4B",
        "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        "slug": "Qwen--Qwen3-4B",
    },
    {
        "id": "Qwen/Qwen3-14B",
        "revision": "40c069824f4251a91eefaf281ebe4c544efd3e18",
        "slug": "Qwen--Qwen3-14B",
    },
    {
        "id": "allenai/Olmo-3-7B-Think",
        "revision": "d97e442d7cc678210054dbcc9b440894d62c89a4",
        "slug": "allenai--Olmo-3-7B-Think",
    },
    {
        "id": "allenai/Olmo-3-32B-Think",
        "revision": "f2edda15216e738ef2bb73771e11890e152b2112",
        "slug": "allenai--Olmo-3-32B-Think",
    },
)


def pipeline() -> ModuleType:
    try:
        return importlib.import_module("prefix.no_steering")
    except ModuleNotFoundError as error:
        pytest.fail(f"missing standalone no-steering production module: {error}")


def test_model_matrix_has_exact_no_steering_schema() -> None:
    module = pipeline()
    assert tuple(module.MODEL_MATRIX) == MODEL_IDS

    for model_id, config in module.MODEL_CONFIGS.items():
        assert model_id in MODEL_IDS
        assert config["model_id"] == model_id
        assert config.get("steering") is False
        assert config.get("quantization") in (None, False)


def test_yaml_and_source_have_identical_pinned_model_contract() -> None:
    module = pipeline()
    config_path = Path(__file__).parents[1] / "configs" / "no_steering.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert [
        {
            "id": spec.model_id,
            "revision": spec.revision,
            "slug": spec.slug,
        }
        for spec in module.MODEL_SPECS.values()
    ] == list(EXPECTED_MODELS)
    assert [
        {key: model[key] for key in ("id", "revision", "slug")}
        for model in config["models"]
    ] == list(EXPECTED_MODELS)
    assert all(
        model["steering"] is False and model["quantization"] is None
        for model in config["models"]
    )


def test_outputs_are_scoped_by_model() -> None:
    module = pipeline()
    outputs = module.output_paths("mmlu_pro", MODEL_IDS[0], root="results")
    assert outputs == {
        "responses": Path("results/Qwen--Qwen3-4B/mmlu_pro/responses.jsonl"),
        "scores": Path("results/Qwen--Qwen3-4B/mmlu_pro/scores.jsonl"),
        "ppl": Path("results/Qwen--Qwen3-4B/mmlu_pro/conditional_ppl.json"),
    }
    assert all(MODEL_IDS[0] not in str(path) for path in outputs.values())
    assert all(".." not in path.parts for path in outputs.values())


@pytest.mark.parametrize(
    "dataset", ["../escape", "nested/name", r"nested\\name", ".", ""]
)
def test_output_paths_reject_dataset_traversal(dataset: str) -> None:
    module = pipeline()
    with pytest.raises(ValueError, match="single non-empty path component"):
        module.output_paths(dataset, MODEL_IDS[0])


def test_evaluation_contract_does_not_enable_steering_or_quantization() -> None:
    module = pipeline()
    contract = module.evaluation_contract()
    assert contract["steering"] is False
    assert contract["quantization"] is False
