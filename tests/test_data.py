from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, cast

import pytest

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


def test_load_harmbench_offline_cache_miss_fails_before_fetch(tmp_path: Path) -> None:
    fetched = False

    def fetch(_: str) -> str:
        nonlocal fetched
        fetched = True
        raise AssertionError("offline cache miss must not fetch")

    load_harmbench = cast(Any, data.load_harmbench)
    with pytest.raises((FileNotFoundError, ValueError), match="cache|offline"):
        load_harmbench(tmp_path, fetch=fetch, offline=True)
    assert not fetched


def test_harmbench_rows_are_json_serializable_and_retain_all_available_metadata(
    tmp_path: Path,
) -> None:
    csv_text = _harmbench_csv()
    records = data.load_harmbench(tmp_path, fetch=lambda _: csv_text)

    json.dumps(records)
    assert records[0]["behavior_id"] == "HB0000"
    assert records[0]["category"] == "category 0"
    assert records[0]["context"] == "none"


def test_load_harmbench_rejects_unexpected_count(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(ValueError, match="expected 400"):
        data.load_harmbench(tmp_path, fetch=lambda _: _harmbench_csv(399))


def test_validate_offline_dataset_caches_checks_all_exact_scopes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str, Path | None]] = []
    digest = hashlib.sha256(data.HARMBENCH_URL.encode("utf-8")).hexdigest()
    (tmp_path / f"harmbench_{digest}.json").write_text("[]", encoding="utf-8")

    monkeypatch.setattr(
        data,
        "load_harmbench",
        lambda cache_dir=None, fetch=None: (
            calls.append(("harmbench", "all", cache_dir)) or [{}] * 400
        ),
    )
    monkeypatch.setattr(
        data,
        "load_mmlu_pro",
        lambda split, cache_dir=None, loader=None: (
            calls.append(("mmlu_pro", split, cache_dir)) or [{}] * 12032
        ),
    )
    monkeypatch.setattr(
        data,
        "load_math500",
        lambda cache_dir=None, loader=None: (
            calls.append(("math500", "test", cache_dir)) or [{}] * 500
        ),
    )

    assert data.validate_offline_dataset_caches(tmp_path) == {
        "harmbench": 400,
        "mmlu_pro": 12032,
        "math500": 500,
    }
    assert calls == [
        ("harmbench", "all", tmp_path),
        ("mmlu_pro", "test", tmp_path),
        ("math500", "test", tmp_path),
    ]


def test_validate_offline_dataset_caches_does_not_fetch_harmbench(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(data, "load_mmlu_pro", lambda *args, **kwargs: [{}] * 12032)
    monkeypatch.setattr(data, "load_math500", lambda *args, **kwargs: [{}] * 500)
    monkeypatch.setattr(
        data,
        "load_harmbench",
        lambda cache_dir=None, fetch=None: (_ for _ in ()).throw(
            AssertionError("cache-only validation must not fetch")
        ),
    )

    with pytest.raises(ValueError, match="harmbench"):
        data.validate_offline_dataset_caches(tmp_path)


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
    calls: list[tuple[str, str, str]] = []

    def loader(
        name: str,
        *,
        split: str,
        cache_dir: Path | None = None,
        revision: str,
    ) -> list[dict[str, object]]:
        calls.append((name, split, revision))
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
    assert calls == [("TIGER-Lab/MMLU-Pro", "test", data._MMLU_PRO_REVISION)]


def test_data_sources_use_immutable_revisions() -> None:
    assert re.fullmatch(r"[0-9a-f]{40}", data._MMLU_PRO_REVISION)
    assert re.fullmatch(r"[0-9a-f]{40}", data._MATH500_REVISION)
    assert re.match(
        r"https://raw\.githubusercontent\.com/[^/]+/[^/]+/[0-9a-f]{40}/.+$",
        data.HARMBENCH_URL,
    )


def test_mmlu_exp2_subset_is_sorted_reproducible_and_supports_override() -> None:
    default_subset = data.mmlu_exp2_subset()
    assert len(default_subset) == 500
    assert default_subset == sorted(set(default_subset))
    assert max(default_subset) > 499
    assert default_subset == data.mmlu_exp2_subset()

    subset = data.mmlu_exp2_subset(n=10, seed=7, n_total=30)
    assert len(subset) == 10
    assert subset == sorted(subset)
    assert subset == data.mmlu_exp2_subset(n=10, seed=7, n_total=30)
    assert max(subset) < 30


@pytest.mark.parametrize("n", [-1, 31])
def test_mmlu_exp2_subset_rejects_invalid_n(n: int) -> None:
    with pytest.raises(ValueError, match="between"):
        data.mmlu_exp2_subset(n=n, n_total=30)


def test_load_math500_normalizes_and_sorts() -> None:
    rows: list[dict[str, object]] = [
        {"problem": "z", "answer": "2", "level": 2, "subject": "Algebra"},
        {"problem": "a", "answer": "1", "level": 1, "subject": "Geometry"},
    ]

    def loader(
        name: str,
        *,
        split: str,
        cache_dir: Path | None = None,
        revision: str,
    ) -> list[dict[str, object]]:
        assert name == "HuggingFaceH4/MATH-500"
        assert split == "test"
        assert revision == data._MATH500_REVISION
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


@pytest.mark.parametrize(
    "dataset,rows,expected",
    [
        ("LLM-LAT/benign-dataset", [{"TEXT": "a"}, {"behavior": "b"}], ["a", "b"]),
        ("LLM-LAT/harmful-dataset", [{"prompt": "a"}, {"text": "b"}], ["a", "b"]),
    ],
)
def test_load_llm_lat_uses_pinned_train_loader(
    dataset: str,
    rows: list[dict[str, object]],
    expected: list[str],
) -> None:
    calls: list[tuple[str, str, object, str]] = []

    def loader(name: str, *, split: str, cache_dir, revision: str):
        calls.append((name, split, cache_dir, revision))
        return rows

    assert (
        data.load_llm_lat(dataset, 2, cache_dir=Path("cache"), loader=loader)
        == expected
    )
    assert calls == [
        (dataset, "train", Path("cache"), data._LLM_LAT_REVISIONS[dataset])
    ]


def test_load_llm_lat_rejects_short_dataset() -> None:
    with pytest.raises(ValueError, match="fewer than 2"):
        data.load_llm_lat(
            "LLM-LAT/benign-dataset",
            2,
            loader=lambda *args, **kwargs: [{"prompt": "one"}],
        )


@pytest.mark.parametrize(
    ("dataset", "revision"),
    [
        (
            "LLM-LAT/benign-dataset",
            "799694027732ac7b5633639690a2ea8ed8597f3e",
        ),
        (
            "LLM-LAT/harmful-dataset",
            "8bfba31bc6d93a5b71808fee5275ef4b6330ed91",
        ),
    ],
)
def test_public_llm_lat_revision_accessor_returns_pinned_revision(
    dataset: str, revision: str
) -> None:
    assert data.llm_lat_revision(dataset) == revision


def _normalized_behavior(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def _select_harmbench(records: Any, *, n_per_category: int, seed: int) -> Any:
    return getattr(data, "harmbench_selection")(
        records, n_per_category=n_per_category, seed=seed
    )


def test_harmbench_cache_has_global_normalized_uniqueness_and_seven_categories(
    harmbench_records: list[dict[str, Any]],
) -> None:
    records = harmbench_records

    assert len(records) == 400
    assert len({_normalized_behavior(str(row["behavior"])) for row in records}) == 393
    assert len({str(row["category"]) for row in records}) == 7
    assert min(Counter(str(row["category"]) for row in records).values()) == 22


def test_harmbench_selection_deduplicates_globally_before_balancing() -> None:
    rows: list[dict[str, object]] = []
    for category in ("a", "b"):
        for index in range(22):
            rows.append(
                {
                    "behavior": f"{category} behavior {index}",
                    "category": category,
                    "behavior_id": f"{category}-{index}",
                }
            )
    rows.extend(
        [
            {
                "behavior": "  ＳHARED\u00a0BEHAVIOR \n",
                "category": "a",
                "behavior_id": "a-shared",
            },
            {"behavior": "shared behavior", "category": "b", "behavior_id": "b-shared"},
        ]
    )

    selected = _select_harmbench(rows, n_per_category=22, seed=11)
    normalized = [_normalized_behavior(str(row["behavior"])) for row in selected]

    assert len(selected) == 44
    assert len(normalized) == len(set(normalized))
    assert sum(value == "shared behavior" for value in normalized) == 1
    assert Counter(str(row["category"]) for row in selected) == {"a": 22, "b": 22}


def test_harmbench_selection_is_seeded_hash_stable_and_returns_stable_ids(
    harmbench_records: list[dict[str, Any]],
) -> None:
    records = harmbench_records

    first = _select_harmbench(records, n_per_category=22, seed=37)
    shuffled = _select_harmbench(list(reversed(records)), n_per_category=22, seed=37)
    other_seed = _select_harmbench(records, n_per_category=22, seed=38)

    assert first == shuffled
    assert [row["id"] for row in first] == [row["id"] for row in shuffled]
    assert [row["id"] for row in first] != [row["id"] for row in other_seed]
    assert len(first) == 154
    assert len({row["id"] for row in first}) == 154
    assert Counter(str(row["category"]) for row in first) == {
        str(category): 22 for category in {row["category"] for row in records}
    }


def test_harmbench_selection_rejects_a_category_shorter_than_requested(
    harmbench_records: list[dict[str, Any]],
) -> None:
    records = harmbench_records

    with pytest.raises(ValueError, match="harmful.*22"):
        _select_harmbench(records, n_per_category=23, seed=37)


def test_load_harmbench_preserves_source_provenance(tmp_path: Path) -> None:
    csv_text = _harmbench_csv()
    records = data.load_harmbench(tmp_path, fetch=lambda _: csv_text)

    assert records[0]["behavior_id"] == "HB0000"
    assert records[0]["context"] == "none"


def test_harmbench_selection_does_not_read_generated_or_judge_labels() -> None:
    class LabelForbiddenRecord(dict[str, object]):
        def __getitem__(self, key: str) -> object:
            if key in {"label", "judge_label", "generated", "response"}:
                raise AssertionError(f"selection accessed forbidden field: {key}")
            return super().__getitem__(key)

        def get(self, key: str, default: object = None) -> object:
            if key in {"label", "judge_label", "generated", "response"}:
                raise AssertionError(f"selection accessed forbidden field: {key}")
            return super().get(key, default)

    records = [
        LabelForbiddenRecord(
            behavior=f"behavior {index}", category="only", behavior_id=f"HB{index}"
        )
        for index in range(22)
    ]

    selected = _select_harmbench(records, n_per_category=22, seed=37)

    assert len(selected) == 22
