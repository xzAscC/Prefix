from __future__ import annotations

import csv
import hashlib
import io
import json
import random
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import cast
from urllib.request import urlopen

HARMBENCH_URL = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/"
    "data/behavior_datasets/harmbench_behaviors_text_all.csv"
)
_MMLU_PRO_DATASET = "TIGER-Lab/MMLU-Pro"
_MATH500_DATASET = "HuggingFaceH4/MATH-500"
_ANSWER_LETTERS = "ABCDEFGHIJ"


def _default_cache_dir(cache_dir: Path | None) -> Path:
    directory = cache_dir if cache_dir is not None else Path("data")
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


def _fetch_harmbench(url: str) -> str:
    with urlopen(url) as response:  # noqa: S310 - URL is a module constant or injected.
        return response.read().decode("utf-8")


def load_harmbench(
    cache_dir: Path | None = None, fetch: Callable[[str], str] | None = None
) -> list[dict[str, object]]:
    directory = _default_cache_dir(cache_dir)
    digest = hashlib.sha256(HARMBENCH_URL.encode("utf-8")).hexdigest()
    cache_path = directory / f"harmbench_{digest}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    text = (fetch or _fetch_harmbench)(HARMBENCH_URL)
    rows = csv.DictReader(io.StringIO(text))
    records: list[dict[str, object]] = []
    for row in rows:
        behavior = str(_field(row, "behavior"))
        category = str(_field_or(row, "semanticcategory", "functionalcategory"))
        records.append({"behavior": behavior, "category": category})
    if len(records) != 400:
        raise ValueError(f"HarmBench CSV yielded {len(records)} records, expected 400")
    records.sort(key=lambda record: (record["category"], record["behavior"]))
    cache_path.write_text(json.dumps(records), encoding="utf-8")
    return records


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


def load_mmlu_pro(
    split: str,
    cache_dir: Path | None = None,
    loader: Callable[..., Iterable[Mapping[str, object]]] | None = None,
) -> list[dict[str, object]]:
    rows = (loader or _dataset_loader())(
        _MMLU_PRO_DATASET, split=split, cache_dir=cache_dir
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
    total = 500 if n_total is None else n_total
    if not 0 <= n <= total:
        raise ValueError("n must be between 0 and n_total")
    return sorted(random.Random(seed).sample(range(total), n))


def load_math500(
    cache_dir: Path | None = None,
    loader: Callable[..., Iterable[Mapping[str, object]]] | None = None,
) -> list[dict[str, object]]:
    rows = (loader or _dataset_loader())(
        _MATH500_DATASET, split="test", cache_dir=cache_dir
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
