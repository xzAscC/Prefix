"""NumPy metrics for steering experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BinResult:
    """Success rate and occupancy for one alignment bin."""

    bin_center: float
    rate: float
    count: int


def cosine_alignment(hiddens: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Return each hidden state's cosine similarity with ``direction``."""
    hiddens = np.asarray(hiddens, dtype=float)
    direction = np.asarray(direction, dtype=float)
    direction_norm = np.linalg.norm(direction)
    if direction_norm == 0.0:
        return np.zeros(hiddens.shape[0], dtype=float)
    numerators = hiddens @ direction
    denominators = np.linalg.norm(hiddens, axis=1) * direction_norm
    return np.divide(
        numerators, denominators, out=np.zeros_like(numerators), where=denominators != 0
    )


def representation_preservation(
    h_steered: np.ndarray, h_unsteered: np.ndarray
) -> np.ndarray:
    """Return paired row-wise cosine similarities between representations."""
    h_steered = np.asarray(h_steered, dtype=float)
    h_unsteered = np.asarray(h_unsteered, dtype=float)
    numerators = np.sum(h_steered * h_unsteered, axis=1)
    denominators = np.linalg.norm(h_steered, axis=1) * np.linalg.norm(
        h_unsteered, axis=1
    )
    return np.divide(
        numerators, denominators, out=np.zeros_like(numerators), where=denominators != 0
    )


def binned_success_rate(
    c: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> list[BinResult]:
    """Estimate success probability by equal-width bins over ``c``."""
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    c = np.asarray(c, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    if c.size == 0:
        return []
    lower = float(np.min(c))
    upper = float(np.max(c))
    if lower == upper:
        return [BinResult(lower, float(np.mean(labels)), int(c.size))]

    edges = np.linspace(lower, upper, n_bins + 1)
    bin_indices = np.searchsorted(edges, c, side="right") - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    results: list[BinResult] = []
    for index in range(n_bins):
        members = bin_indices == index
        count = int(np.sum(members))
        if count:
            results.append(
                BinResult(
                    bin_center=float((edges[index] + edges[index + 1]) / 2),
                    rate=float(np.mean(labels[members])),
                    count=count,
                )
            )
    return results


def mean_trajectories_by_label(
    traces: list[np.ndarray], labels: np.ndarray
) -> dict[bool, np.ndarray]:
    """Average variable-length traces independently for each boolean label."""
    labels = np.asarray(labels, dtype=bool)
    if len(traces) != len(labels):
        raise ValueError("traces and labels must have the same length")
    result: dict[bool, np.ndarray] = {}
    for label in (False, True):
        selected = [
            np.asarray(trace, dtype=float)
            for trace, value in zip(traces, labels)
            if value == label
        ]
        if not selected:
            continue
        max_length = max(trace.shape[0] for trace in selected)
        totals = np.zeros(max_length, dtype=float)
        counts = np.zeros(max_length, dtype=int)
        for trace in selected:
            length = trace.shape[0]
            totals[:length] += trace
            counts[:length] += 1
        available = counts > 0
        last = int(np.flatnonzero(available)[-1]) + 1
        result[label] = totals[:last] / counts[:last]
    return result


def pareto_frontier(points: list[tuple[float, float]]) -> list[int]:
    """Return indices of points not strictly dominated in either coordinate."""
    frontier: list[int] = []
    for index, point in enumerate(points):
        dominated = any(
            other[0] >= point[0]
            and other[1] >= point[1]
            and (other[0] > point[0] or other[1] > point[1])
            for other in points
        )
        if not dominated:
            frontier.append(index)
    return frontier


def select_operating_point(
    points: list[tuple[float, float]], baseline_second: float, cap_ratio: float = 0.9
) -> int:
    """Select the best first coordinate subject to a second-coordinate cap."""
    cap = cap_ratio * baseline_second
    eligible = [index for index, point in enumerate(points) if point[1] >= cap]
    if not eligible:
        raise ValueError("no operating point satisfies the cap")
    return max(eligible, key=lambda index: (points[index][0], points[index][1]))
