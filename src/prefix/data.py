from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import random
import unicodedata
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import cast
from urllib.request import urlopen

HARMBENCH_URL = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench/"
    "8e1604d1171fe8a48d8febecd22f600e462bdcdd/"
    "data/behavior_datasets/harmbench_behaviors_text_all.csv"
)
_MMLU_PRO_DATASET = "TIGER-Lab/MMLU-Pro"
_MATH500_DATASET = "HuggingFaceH4/MATH-500"
_MMLU_PRO_REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
_MATH500_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
_LLM_LAT_REVISIONS = {
    "LLM-LAT/benign-dataset": "799694027732ac7b5633639690a2ea8ed8597f3e",
    "LLM-LAT/harmful-dataset": "8bfba31bc6d93a5b71808fee5275ef4b6330ed91",
}
MMLU_PRO_TEST_SIZE = 12032
_ANSWER_LETTERS = "ABCDEFGHIJ"


def llm_lat_revision(dataset: str) -> str:
    try:
        return _LLM_LAT_REVISIONS[dataset]
    except KeyError as error:
        raise ValueError(f"unsupported LLM-LAT dataset: {dataset}") from error


def _default_cache_dir(cache_dir: Path | None) -> Path:
    directory = (
        cache_dir
        if cache_dir is not None
        else Path(os.environ.get("PREFIX_DATA_CACHE", "data"))
    )
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _field(row: Mapping[str, object], name: str) -> object:
    wanted = name.lower()
    for key, value in row.items():
        if str(key).lower() == wanted:
            return value
    raise KeyError(f"Dataset row has no {name!r} field")


def _field_or(row: Mapping[str, object], name: str, fallback: str) -> object:
    try:
        return _field(row, name)
    except KeyError:
        return _field(row, fallback)


def _optional_field(row: Mapping[str, object], *names: str) -> object | None:
    for name in names:
        try:
            return _field(row, name)
        except KeyError:
            continue
    return None


class _HarmBenchRecord(dict[str, object]):
    def __init__(self, behavior: str, category: str, **metadata: object) -> None:
        super().__init__(behavior=behavior, category=category)
        self._metadata = {
            key: value for key, value in metadata.items() if value is not None
        }

    def __getitem__(self, key: str) -> object:
        try:
            return super().__getitem__(key)
        except KeyError:
            if key in self._metadata:
                return self._metadata[key]
            raise

    def get(self, key: str, default: object = None) -> object:
        if key in self._metadata:
            return self._metadata[key]
        return super().get(key, default)

    def __contains__(self, key: object) -> bool:
        return super().__contains__(key) or key in self._metadata

    @property
    def metadata(self) -> dict[str, object]:
        return dict(self._metadata)


def _harmbench_record(row: Mapping[str, object]) -> _HarmBenchRecord:
    behavior = str(_field(row, "behavior"))
    category = str(
        _optional_field(row, "semanticcategory", "category", "functionalcategory")
    )
    metadata = {
        "behavior_id": _optional_field(row, "behaviorid", "behavior_id"),
        "functional_category": _optional_field(
            row, "functionalcategory", "functional_category"
        ),
        "context": _optional_field(row, "contextstring", "context"),
        "tags": _optional_field(row, "tags"),
    }
    return _HarmBenchRecord(behavior, category, **metadata)


def _normalized_harmbench_behavior(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def _harmbench_cache_row(record: Mapping[str, object]) -> dict[str, object]:
    cached = dict(record)
    metadata = getattr(record, "metadata", {})
    if isinstance(metadata, dict):
        cached.update(metadata)
    return cached


def _fetch_harmbench(url: str) -> str:
    with urlopen(url) as response:  # noqa: S310 - URL is a module constant or injected.
        return response.read().decode("utf-8")


def load_harmbench(
    cache_dir: Path | None = None,
    fetch: Callable[[str], str] | None = None,
    *,
    offline: bool = False,
) -> list[dict[str, object]]:
    directory = _default_cache_dir(cache_dir)
    digest = hashlib.sha256(HARMBENCH_URL.encode("utf-8")).hexdigest()
    cache_path = directory / f"harmbench_{digest}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        return [_harmbench_record(row) for row in cached]

    if offline:
        raise FileNotFoundError(f"offline HarmBench cache is missing: {cache_path}")

    text = (fetch or _fetch_harmbench)(HARMBENCH_URL)
    rows = csv.DictReader(io.StringIO(text))
    records: list[_HarmBenchRecord] = []
    for row in rows:
        records.append(_harmbench_record(row))
    if len(records) != 400:
        raise ValueError(f"HarmBench CSV yielded {len(records)} records, expected 400")
    records.sort(key=lambda record: (record["category"], record["behavior"]))
    cache_path.write_text(
        json.dumps([_harmbench_cache_row(record) for record in records]),
        encoding="utf-8",
    )
    return cast(list[dict[str, object]], records)


def harmbench_selection(
    records: Iterable[Mapping[str, object]], n_per_category: int = 22, seed: int = 42
) -> list[dict[str, object]]:
    if n_per_category < 0:
        raise ValueError("n_per_category must be non-negative")

    candidates = sorted(
        (_harmbench_cache_row(record) for record in records),
        key=lambda record: (
            _normalized_harmbench_behavior(str(_field(record, "behavior"))),
            str(_field(record, "category")),
            str(_optional_field(record, "behaviorid", "behavior_id")),
            str(_field(record, "behavior")),
        ),
    )
    unique: dict[str, dict[str, object]] = {}
    for record in candidates:
        normalized = _normalized_harmbench_behavior(str(_field(record, "behavior")))
        unique.setdefault(normalized, record)

    ranked: list[tuple[str, str, dict[str, object]]] = []
    for normalized, record in unique.items():
        category = str(_field(record, "category"))
        digest = hashlib.sha256(f"{seed}\0{normalized}".encode("utf-8")).hexdigest()
        ranked.append((category, digest, record))

    by_category: dict[str, list[tuple[str, dict[str, object]]]] = {}
    for category, digest, record in ranked:
        by_category.setdefault(category, []).append((digest, record))
    for category, category_records in by_category.items():
        if len(category_records) < n_per_category:
            raise ValueError(
                f"HarmBench category {category!r} has {len(category_records)} "
                f"unique behaviors; need {n_per_category}"
            )
        category_records.sort(key=lambda item: item[0])

    selected: list[dict[str, object]] = []
    for category in sorted(by_category):
        for digest, record in by_category[category][:n_per_category]:
            output = dict(record)
            output["id"] = f"harmbench-{seed}-{digest}"
            selected.append(output)
    return selected


def validate_offline_dataset_caches(cache_dir: Path) -> dict[str, int]:
    """Load every configured benchmark from an existing cache only.

    The caller supplies the shared cache root.  The
    HarmBench file is checked before loading so its URL fallback can never be
    reached by this gate; Hugging Face datasets obey the offline environment
    variables while resolving their cached revisions.
    """
    cache_dir = Path(cache_dir)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    harmbench_digest = hashlib.sha256(HARMBENCH_URL.encode("utf-8")).hexdigest()
    harmbench_cache = cache_dir / f"harmbench_{harmbench_digest}.json"
    if not harmbench_cache.is_file():
        raise ValueError(f"harmbench cache is missing: {harmbench_cache}")

    counts = {
        "harmbench": len(load_harmbench(cache_dir)),
        "mmlu_pro": len(load_mmlu_pro("test", cache_dir=cache_dir)),
        "math500": len(load_math500(cache_dir=cache_dir)),
    }
    expected = {"harmbench": 400, "mmlu_pro": 12032, "math500": 500}
    for name, expected_count in expected.items():
        if counts[name] != expected_count:
            raise ValueError(
                f"{name} cache has {counts[name]} rows, expected {expected_count}"
            )
    return counts


def harmbench_split(n_val: int = 50, seed: int = 42) -> tuple[list[int], list[int]]:
    if not 0 <= n_val <= 400:
        raise ValueError("n_val must be between 0 and 400")
    validation = set(random.Random(seed).sample(range(400), n_val))
    val_indices = sorted(validation)
    test_indices = sorted(set(range(400)) - validation)
    return val_indices, test_indices


def _dataset_loader() -> Callable[..., Iterable[Mapping[str, object]]]:
    from datasets import load_dataset

    return load_dataset


def load_llm_lat(
    dataset: str,
    n: int,
    cache_dir: Path | None = None,
    loader: Callable[..., Iterable[Mapping[str, object]]] | None = None,
) -> list[str]:
    if dataset not in _LLM_LAT_REVISIONS:
        raise ValueError(f"unsupported LLM-LAT dataset: {dataset}")
    if n < 0:
        raise ValueError("n must be non-negative")
    rows = (loader or _dataset_loader())(
        dataset,
        split="train",
        cache_dir=cache_dir,
        revision=_LLM_LAT_REVISIONS[dataset],
    )
    records: list[str] = []
    for row in rows:
        try:
            value = _field_or(row, "prompt", "text")
        except KeyError:
            value = _field(row, "behavior")
        records.append(str(value))
        if len(records) == n:
            break
    if len(records) < n:
        raise ValueError(f"LLM-LAT dataset has fewer than {n} available rows")
    return records


def load_mmlu_pro(
    split: str,
    cache_dir: Path | None = None,
    loader: Callable[..., Iterable[Mapping[str, object]]] | None = None,
) -> list[dict[str, object]]:
    rows = (loader or _dataset_loader())(
        _MMLU_PRO_DATASET,
        split=split,
        cache_dir=cache_dir,
        revision=_MMLU_PRO_REVISION,
    )
    records: list[dict[str, object]] = []
    for row in rows:
        answer_index = int(str(_field(row, "answer_index")))
        records.append(
            {
                "question": str(_field(row, "question")),
                "options": list(cast(Iterable[str], _field(row, "options"))),
                "answer_index": answer_index,
                "answer_letter": _ANSWER_LETTERS[answer_index],
                "category": str(_field(row, "category")),
            }
        )
    records.sort(key=lambda record: (record["category"], record["question"]))
    return records


def mmlu_exp2_subset(
    n: int = 500, seed: int = 42, n_total: int | None = None
) -> list[int]:
    """Return a seeded sorted sample from the MMLU-Pro test split.

    Callers may pass ``len(records)`` for the actually loaded test split via
    ``n_total`` when it differs from the official split size.
    """
    total = MMLU_PRO_TEST_SIZE if n_total is None else n_total
    if not 0 <= n <= total:
        raise ValueError("n must be between 0 and n_total")
    return sorted(random.Random(seed).sample(range(total), n))


def load_math500(
    cache_dir: Path | None = None,
    loader: Callable[..., Iterable[Mapping[str, object]]] | None = None,
) -> list[dict[str, object]]:
    rows = (loader or _dataset_loader())(
        _MATH500_DATASET,
        split="test",
        cache_dir=cache_dir,
        revision=_MATH500_REVISION,
    )
    records: list[dict[str, object]] = [
        {
            "problem": str(_field(row, "problem")),
            "answer": str(_field(row, "answer")),
            "level": _field(row, "level"),
            "type": str(_field_or(row, "type", "subject")),
        }
        for row in rows
    ]
    records.sort(key=lambda record: (record["problem"], record["answer"]))
    return records


def math500_partition(
    n_direction: int = 50,
    n_val: int = 50,
    seed: int = 42,
    n_total: int | None = None,
) -> dict[str, list[int]]:
    total = 500 if n_total is None else n_total
    if n_direction < 0 or n_val < 0 or n_direction + n_val > total:
        raise ValueError("partition sizes must be non-negative and fit n_total")
    indices = list(range(total))
    random.Random(seed).shuffle(indices)
    direction = sorted(indices[:n_direction])
    validation = sorted(indices[n_direction : n_direction + n_val])
    test = sorted(indices[n_direction + n_val :])
    result = {"direction": direction, "val": validation, "test": test}
    sets = [set(values) for values in result.values()]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert set().union(*sets) == set(range(total))
    return result


def direction_prompt(problem: str, positive: bool) -> str:
    problem = problem.rstrip()
    if positive:
        return f"{problem} Reason step by step and end your response with \\boxed{{}}."
    return f"{problem} Reason step by step and end your response with ``The answer is {{}}.''."


def neutral_math_prompt(problem: str) -> str:
    return f"{problem.rstrip()} Reason step by step."
