"""Tests for the numpy-only steering metrics."""

from __future__ import annotations

import numpy as np
import pytest

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
