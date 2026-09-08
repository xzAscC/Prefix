from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from prefix import data  # pyright: ignore[reportAttributeAccessIssue]


def _harmbench_csv(n_rows: int = 400) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=[
            "Behavior",
            "FunctionalCategory",
            "SemanticCategory",
            "Tags",
            "ContextString",
            "BehaviorID",
        ],
    )
    writer.writeheader()
    for index in range(n_rows):
        writer.writerow(
            {
                "Behavior": f"behavior {index:03d}",
                "FunctionalCategory": "standard",
                "SemanticCategory": f"category {index % 4}",
                "Tags": "",
                "ContextString": "none",
                "BehaviorID": f"HB{index:04d}",
            }
        )
    return output.getvalue()


def test_load_harmbench_filters_sorts_and_reuses_cache(tmp_path: Path) -> None:
    calls: list[str] = []

    def fetch(url: str) -> str:
        calls.append(url)
        return _harmbench_csv()

    records = data.load_harmbench(tmp_path, fetch=fetch)
    assert len(records) == 400
    assert all(set(record) == {"behavior", "category"} for record in records)
    assert str(records[0]["category"]).startswith("category")
    assert records == sorted(
        records, key=lambda record: (record["category"], record["behavior"])
    )

    cached = data.load_harmbench(
        tmp_path, fetch=lambda _: (_ for _ in ()).throw(AssertionError())
    )
    assert cached == records
    assert len(calls) == 1
    assert list(tmp_path.glob("*.json"))


def test_load_harmbench_rejects_unexpected_count(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(ValueError, match="expected 400"):
        data.load_harmbench(tmp_path, fetch=lambda _: _harmbench_csv(399))


def test_harmbench_split_is_reproducible_sorted_and_disjoint() -> None:
    validation, test = data.harmbench_split()
    assert len(validation) == 50
    assert len(test) == 350
    assert validation == sorted(validation)
    assert test == sorted(test)
    assert set(validation).isdisjoint(test)
    assert set(validation) | set(test) == set(range(400))
    assert data.harmbench_split() == data.harmbench_split()


def test_load_mmlu_pro_normalizes_and_sorts() -> None:
    rows: list[dict[str, object]] = [
        {"question": "z", "options": ["a"] * 10, "answer_index": 9, "category": "b"},
        {"question": "a", "options": ["b"] * 10, "answer_index": 1, "category": "a"},
    ]
    calls: list[tuple[str, str]] = []

    def loader(
        name: str, *, split: str, cache_dir: Path | None = None
    ) -> list[dict[str, object]]:
        calls.append((name, split))
        return rows

    result = data.load_mmlu_pro("test", loader=loader)
    assert result == [
        {
            "question": "a",
            "options": ["b"] * 10,
            "answer_index": 1,
            "answer_letter": "B",
            "category": "a",
        },
        {
            "question": "z",
            "options": ["a"] * 10,
            "answer_index": 9,
            "answer_letter": "J",
            "category": "b",
        },
    ]
    assert calls == [("TIGER-Lab/MMLU-Pro", "test")]


def test_mmlu_exp2_subset_is_sorted_reproducible_and_supports_override() -> None:
    subset = data.mmlu_exp2_subset(n=10, seed=7, n_total=30)
    assert len(subset) == 10
    assert subset == sorted(subset)
    assert subset == data.mmlu_exp2_subset(n=10, seed=7, n_total=30)
    assert max(subset) < 30


def test_load_math500_normalizes_and_sorts() -> None:
    rows: list[dict[str, object]] = [
        {"problem": "z", "answer": "2", "level": 2, "subject": "Algebra"},
        {"problem": "a", "answer": "1", "level": 1, "subject": "Geometry"},
    ]

    def loader(
        name: str, *, split: str, cache_dir: Path | None = None
    ) -> list[dict[str, object]]:
        assert name == "HuggingFaceH4/MATH-500"
        assert split == "test"
        return rows

    assert data.load_math500(loader=loader) == [
        {"problem": "a", "answer": "1", "level": 1, "type": "Geometry"},
        {"problem": "z", "answer": "2", "level": 2, "type": "Algebra"},
    ]


def test_math500_partition_is_disjoint_covering_and_sorted() -> None:
    partition = data.math500_partition(n_direction=3, n_val=4, seed=9, n_total=20)
    assert {key: len(value) for key, value in partition.items()} == {
        "direction": 3,
        "val": 4,
        "test": 13,
    }
    assert all(indices == sorted(indices) for indices in partition.values())
    assert set(partition["direction"]).isdisjoint(partition["val"])
    assert set(partition["direction"]).isdisjoint(partition["test"])
    assert set(partition["val"]).isdisjoint(partition["test"])
    assert set().union(*map(set, partition.values())) == set(range(20))
    assert partition == data.math500_partition(
        n_direction=3, n_val=4, seed=9, n_total=20
    )


def test_prompt_builders_strip_problem_whitespace() -> None:
    problem = "  Solve this. \n"
    assert (
        data.direction_prompt(problem, True)
        == "  Solve this. Reason step by step and end your response with \\boxed{}."
    )
    assert (
        data.direction_prompt(problem, False)
        == "  Solve this. Reason step by step and end your response with ``The answer is {}.''."
    )
    assert data.neutral_math_prompt(problem) == "  Solve this. Reason step by step."
