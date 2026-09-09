from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TypeVar


_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


_MODEL_ROWS = (
    ("Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c", "Qwen--Qwen3-4B"),
    ("Qwen/Qwen3-14B", "40c069824f4251a91eefaf281ebe4c544efd3e18", "Qwen--Qwen3-14B"),
    (
        "allenai/Olmo-3-7B-Think",
        "d97e442d7cc678210054dbcc9b440894d62c89a4",
        "allenai--Olmo-3-7B-Think",
    ),
    (
        "allenai/Olmo-3-32B-Think",
        "f2edda15216e738ef2bb73771e11890e152b2112",
        "allenai--Olmo-3-32B-Think",
    ),
)


def validate_model_spec(spec: ModelSpec) -> None:
    """Reject an unpinned, unsafe, or accidentally steered model specification."""
    if not spec.model_id or "/" not in spec.model_id or ".." in spec.model_id:
        raise ValueError("model_id must be a non-empty repository identifier")
    if not _REVISION_RE.fullmatch(spec.revision):
        raise ValueError("revision must be a 40-character lowercase commit SHA")
    if not _SLUG_RE.fullmatch(spec.slug) or ".." in spec.slug:
        raise ValueError("slug must be a safe single path component")
    if spec.steering:
        raise ValueError("no-steering specifications cannot enable steering")
    if spec.quantization not in (None, False):
        raise ValueError("no-steering specifications cannot enable quantization")


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Immutable, pinned model identity and baseline execution settings."""

    model_id: str
    revision: str
    slug: str
    steering: bool = False
    quantization: str | bool | None = None

    def __post_init__(self) -> None:
        validate_model_spec(self)


MODEL_SPECS: Mapping[str, ModelSpec] = MappingProxyType(
    {
        model_id: ModelSpec(model_id, revision, slug)
        for model_id, revision, slug in _MODEL_ROWS
    }
)
MODEL_SPECIFICATIONS = MODEL_SPECS
MODEL_MATRIX = tuple(MODEL_SPECS)

MODEL_CONFIGS: Mapping[str, Mapping[str, object]] = MappingProxyType(
    {
        model_id: MappingProxyType(
            {
                "model_id": spec.model_id,
                "revision": spec.revision,
                "slug": spec.slug,
                "steering": spec.steering,
                "quantization": spec.quantization,
            }
        )
        for model_id, spec in MODEL_SPECS.items()
    }
)

DATASET_SCOPES: dict[str, dict[str, object]] = {
    "harmbench": {"split": "all", "count": 400},
    "mmlu_pro": {"split": "test", "count": 12032},
    "math500": {"split": "test", "count": 500},
}


def validate_model_matrix(specs: Mapping[str, ModelSpec]) -> None:
    """Validate an evaluator's model matrix and its exact expected membership."""
    if tuple(specs) != MODEL_MATRIX:
        raise ValueError("model matrix does not contain the exact required models")
    for model_id, spec in specs.items():
        if model_id != spec.model_id:
            raise ValueError("model matrix key does not match model_id")
        validate_model_spec(spec)


def model_spec(model_id: str) -> ModelSpec:
    """Return the pinned specification for ``model_id``."""
    try:
        return MODEL_SPECS[model_id]
    except KeyError as error:
        raise ValueError(f"unsupported model: {model_id!r}") from error


def safe_model_slug(model_id: str) -> str:
    """Return the configured safe path component for a supported model."""
    return model_spec(model_id).slug


def output_paths(
    dataset: str, model_id: str, *, root: str | Path = "results"
) -> dict[str, Path]:
    """Build model-scoped output locations without allowing path traversal."""
    if not dataset or "/" in dataset or "\\" in dataset or dataset in {".", ".."}:
        raise ValueError("dataset must be a single non-empty path component")
    spec = model_spec(model_id)
    base = Path(root) / spec.slug / dataset
    return {
        "responses": base / "responses.jsonl",
        "scores": base / "scores.jsonl",
        "ppl": base / "conditional_ppl.json",
    }


def evaluation_contract() -> dict[str, object]:
    """Return the immutable-baseline execution contract as a fresh mapping."""
    return {"steering": False, "quantization": False}


def _selected_logprobs(sample: Mapping[str, object]) -> list[float]:
    values = sample.get("selected_logprobs")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError("selected_logprobs must be a sequence of numbers")
    result: list[float] = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError("logprobs must be finite real numbers")
        if not isinstance(value, (int, float, str)):
            raise ValueError("logprobs must be finite real numbers")
        try:
            number = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError("logprobs must be finite real numbers") from error
        if not math.isfinite(number) or number > 0.0:
            raise ValueError("logprobs must be finite real numbers")
        result.append(number)
    return result


def conditional_ppl(samples: Iterable[Mapping[str, object]]) -> float:
    """Compute response-only PPL as ``exp(total NLL / selected token count)``."""
    total_nll = 0.0
    token_count = 0
    for sample in samples:
        logprobs = _selected_logprobs(sample)
        total_nll -= sum(logprobs)
        token_count += len(logprobs)
    if token_count == 0:
        raise ValueError("conditional PPL requires at least one selected token")
    try:
        result = math.exp(total_nll / token_count)
    except OverflowError as error:
        raise ValueError("conditional PPL is nonfinite") from error
    if not math.isfinite(result):
        raise ValueError("conditional PPL is nonfinite")
    return result


Sample = TypeVar("Sample", bound=Mapping[str, object])
Result = TypeVar("Result", bound=Mapping[str, object])


def evaluate_samples(
    samples: Iterable[Sample],
    *,
    evaluate: Callable[[Sample], Result],
    completed_ids: set[str] | frozenset[str] | None = None,
) -> list[dict[str, object]]:
    """Evaluate each pending sample independently, preserving incremental order."""
    rows: list[dict[str, object]] = []
    for sample in samples:
        sample_id = sample.get("id")
        if sample_id is None:
            raise ValueError("each sample must contain an id")
        identifier = str(sample_id)
        if completed_ids is not None and identifier in completed_ids:
            continue
        row: dict[str, object]
        try:
            result = evaluate(sample)
            row = dict(result)
        except Exception as error:
            row = {"id": identifier, "status": "error", "error": str(error)}
        rows.append(row)
    return rows


__all__ = [
    "DATASET_SCOPES",
    "MODEL_CONFIGS",
    "MODEL_MATRIX",
    "MODEL_SPECS",
    "MODEL_SPECIFICATIONS",
    "ModelSpec",
    "conditional_ppl",
    "evaluation_contract",
    "evaluate_samples",
    "model_spec",
    "output_paths",
    "safe_model_slug",
    "validate_model_matrix",
    "validate_model_spec",
]
