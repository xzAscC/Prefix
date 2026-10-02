"""Tests for the numpy-only steering metrics."""

from __future__ import annotations

import numpy as np
import pytest
import time

import prefix.metrics as metrics
from prefix.metrics import (
    BinResult,
    binned_success_rate,
    cosine_alignment,
    mean_trajectories_by_label,
    pareto_frontier,
    representation_preservation,
    select_operating_point,
)


def test_cosine_alignment_uses_rowwise_cosines_and_zero_is_safe() -> None:
    hiddens = np.array([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0], [-3.0, 0.0]])
    result = cosine_alignment(hiddens, np.array([1.0, 0.0]))
    np.testing.assert_allclose(result, [1.0, 0.0, 0.0, -1.0])


def test_representation_preservation_uses_paired_rowwise_cosines() -> None:
    steered = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    unsteered = np.array([[2.0, 0.0], [1.0, -1.0], [0.0, 4.0]])
    result = representation_preservation(steered, unsteered)
    np.testing.assert_allclose(result, [1.0, 0.0, 0.0])


def test_binned_success_rate_has_equal_width_bins_and_drops_empty_bins() -> None:
    c = np.array([0.0, 0.1, 0.8, 1.0])
    labels = np.array([True, False, True, True])
    result = binned_success_rate(c, labels, n_bins=4)
    assert result == [
        BinResult(bin_center=0.125, rate=0.5, count=2),
        BinResult(bin_center=0.875, rate=1.0, count=2),
    ]


def test_binned_success_rate_degenerate_input_is_one_bin() -> None:
    result = binned_success_rate(
        np.array([2.0, 2.0]), np.array([True, False]), n_bins=5
    )
    assert result == [BinResult(bin_center=2.0, rate=0.5, count=2)]


def test_mean_trajectories_by_label_averages_only_available_tokens() -> None:
    traces = [
        np.array([1.0, 2.0, 3.0]),
        np.array([5.0, 7.0]),
        np.array([10.0, 20.0, 30.0, 40.0]),
    ]
    labels = np.array([True, False, True])
    result = mean_trajectories_by_label(traces, labels)
    np.testing.assert_allclose(result[True], [5.5, 11.0, 16.5, 40.0])
    np.testing.assert_allclose(result[False], [5.0, 7.0])


def test_mean_trajectories_skips_labels_with_no_examples() -> None:
    result = mean_trajectories_by_label([np.array([1.0])], np.array([True]))
    assert set(result) == {True}
    np.testing.assert_allclose(result[True], [1.0])


@pytest.mark.parametrize(
    ("traces", "labels"),
    [
        ([np.array([1.0])], np.array([True, False])),
        ([np.array([1.0]), np.array([2.0])], np.array([True])),
    ],
)
def test_mean_trajectories_rejects_trace_label_length_mismatch(
    traces: list[np.ndarray], labels: np.ndarray
) -> None:
    with pytest.raises(ValueError, match="same length"):
        mean_trajectories_by_label(traces, labels)


def test_pareto_frontier_returns_maximizers_and_keeps_equal_points() -> None:
    points = [(1.0, 1.0), (2.0, 0.5), (1.5, 2.0), (1.0, 0.5), (1.5, 2.0)]
    assert pareto_frontier(points) == [1, 2, 4]


def test_select_operating_point_applies_cap_and_tie_breaks_second() -> None:
    points = [(0.9, 0.95), (1.2, 0.90), (1.2, 0.99), (1.5, 0.80)]
    assert select_operating_point(points, baseline_second=1.0) == 2


def test_select_operating_point_raises_when_cap_is_unmet() -> None:
    with pytest.raises(ValueError, match="cap"):
        select_operating_point([(1.0, 0.5)], baseline_second=1.0)


def test_token_cosine_trajectory_rejects_zero_norm_and_nonfinite_values() -> None:
    direction = np.array([1.0, 0.0])
    with pytest.raises(ValueError, match="zero norm"):
        getattr(metrics, "token_cosine_trajectory")(
            np.array([[1.0, 0.0], [0.0, 0.0]]), direction
        )
    with pytest.raises(ValueError, match="finite"):
        getattr(metrics, "token_cosine_trajectory")(
            np.array([[np.nan, 0.0]]), direction
        )


def test_token_cosine_trajectory_aggregation_is_category_macro_and_one_based() -> None:
    records = [
        {
            "condition": "control",
            "layer": 19,
            "prompt_id": "a1",
            "category": "A",
            "cosines": [1.0, 0.5],
        },
        {
            "condition": "control",
            "layer": 19,
            "prompt_id": "a2",
            "category": "A",
            "cosines": [1.0],
        },
        {
            "condition": "control",
            "layer": 19,
            "prompt_id": "b1",
            "category": "B",
            "cosines": [0.0, 0.25, 0.5],
        },
        {
            "condition": "control",
            "layer": 19,
            "prompt_id": "b2",
            "category": "B",
            "cosines": [0.0],
        },
        {
            "condition": "control",
            "layer": 20,
            "prompt_id": "a1",
            "category": "A",
            "cosines": [0.2],
        },
    ]

    result = getattr(metrics, "aggregate_token_cosine_trajectories")(records)

    layer_19 = [
        row for row in result if row["condition"] == "control" and row["layer"] == 19
    ]
    assert [row["token_index"] for row in layer_19] == [1, 2, 3]
    np.testing.assert_allclose(
        [row["mean_cosine"] for row in layer_19], [0.5, 0.375, 0.5]
    )
    assert [row["support"] for row in layer_19] == [4, 2, 1]
    assert layer_19[1]["category_support"] == {"A": 1, "B": 1}
    assert layer_19[2]["category_support"] == {"B": 1}

    layer_20 = [
        row for row in result if row["condition"] == "control" and row["layer"] == 20
    ]
    assert [(row["layer"], row["token_index"]) for row in layer_20] == [(20, 1)]
    assert layer_20[0]["mean_cosine"] == pytest.approx(0.2)


def test_paired_bootstrap_resamples_prompts_shared_between_conditions() -> None:
    records = [
        {
            "condition": "control",
            "layer": 19,
            "prompt_id": "a1",
            "category": "A",
            "cosines": [0.0],
        },
        {
            "condition": "steered",
            "layer": 19,
            "prompt_id": "a1",
            "category": "A",
            "cosines": [1.0],
        },
        {
            "condition": "control",
            "layer": 19,
            "prompt_id": "a2",
            "category": "A",
            "cosines": [10.0],
        },
        {
            "condition": "steered",
            "layer": 19,
            "prompt_id": "a2",
            "category": "A",
            "cosines": [11.0],
        },
        {
            "condition": "control",
            "layer": 19,
            "prompt_id": "b1",
            "category": "B",
            "cosines": [20.0],
        },
        {
            "condition": "steered",
            "layer": 19,
            "prompt_id": "b1",
            "category": "B",
            "cosines": [23.0],
        },
        {
            "condition": "control",
            "layer": 19,
            "prompt_id": "b2",
            "category": "B",
            "cosines": [30.0],
        },
        {
            "condition": "steered",
            "layer": 19,
            "prompt_id": "b2",
            "category": "B",
            "cosines": [33.0],
        },
    ]

    first = getattr(metrics, "paired_bootstrap_ci")(
        records,
        control="control",
        treatment="steered",
        n_bootstrap=31,
        seed=17,
    )
    second = getattr(metrics, "paired_bootstrap_ci")(
        records,
        control="control",
        treatment="steered",
        n_bootstrap=31,
        seed=17,
    )

    assert first == second
    assert first[0]["layer"] == 19
    assert first[0]["token_index"] == 1
    assert first[0]["mean_difference"] == pytest.approx(2.0)
    assert first[0]["lower"] == pytest.approx(2.0)
    assert first[0]["upper"] == pytest.approx(2.0)


def test_paired_bootstrap_pairs_only_same_physical_layer_and_is_finite_reproducible() -> (
    None
):
    records = []
    for layer, difference in ((19, 1.0), (20, 3.0)):
        for category in ("A", "B"):
            for prompt_index in range(8):
                control = float(layer + prompt_index)
                records.extend(
                    [
                        {
                            "condition": "baseline",
                            "layer": layer,
                            "prompt_id": f"{category}-{prompt_index}",
                            "category": category,
                            "cosines": [control],
                        },
                        {
                            "condition": f"layer_{layer}",
                            "layer": layer,
                            "prompt_id": f"{category}-{prompt_index}",
                            "category": category,
                            "cosines": [control + difference],
                        },
                    ]
                )

    first = metrics.paired_bootstrap_ci(
        records,
        control="baseline",
        treatment="layer_19",
        n_bootstrap=10000,
        seed=42,
        confidence=0.95,
    )
    second = metrics.paired_bootstrap_ci(
        records,
        control="baseline",
        treatment="layer_19",
        n_bootstrap=10000,
        seed=42,
        confidence=0.95,
    )

    assert [(row["layer"], row["token_index"]) for row in first] == [(19, 1)]
    assert [row["mean_difference"] for row in first] == pytest.approx([1.0])
    assert first == second
    assert all(
        np.isfinite([row["mean_difference"], row["lower"], row["upper"]]).all()
        for row in first
    )
    layer_20 = metrics.paired_bootstrap_ci(
        records,
        control="baseline",
        treatment="layer_20",
        n_bootstrap=10000,
        seed=42,
        confidence=0.95,
    )
    assert [(row["layer"], row["mean_difference"]) for row in layer_20] == [
        (20, pytest.approx(3.0))
    ]


def test_paired_bootstrap_10000_resamples_has_practical_cpu_runtime() -> None:
    records = []
    for layer in range(1, 12):
        for category in ("A", "B", "C", "D", "E", "F", "G"):
            for prompt_index in range(64):
                prompt_id = f"{category}-{prompt_index}"
                records.extend(
                    [
                        {
                            "condition": "baseline",
                            "layer": layer,
                            "prompt_id": prompt_id,
                            "category": category,
                            "cosines": [0.1, 0.2],
                        },
                        {
                            "condition": "steered",
                            "layer": layer,
                            "prompt_id": prompt_id,
                            "category": category,
                            "cosines": [0.2, 0.4],
                        },
                    ]
                )

    started = time.perf_counter()
    result = metrics.paired_bootstrap_ci(
        records,
        control="baseline",
        treatment="steered",
        n_bootstrap=10000,
        seed=42,
        confidence=0.95,
    )
    elapsed = time.perf_counter() - started

    assert len(result) == 22
    assert elapsed < 4.0
