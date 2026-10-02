"""NumPy metrics for steering experiments."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypedDict

import numpy as np


@dataclass(frozen=True)
class BinResult:
    """Success rate and occupancy for one alignment bin."""

    bin_center: float
    rate: float
    count: int


class TokenTrajectorySummary(TypedDict):
    """One category-macro trajectory point for PDF-ready reporting."""

    condition: str
    layer: int
    token_index: int
    mean_cosine: float
    support: int
    category_support: dict[str, int]


class BootstrapSummary(TypedDict):
    """One paired-bootstrap confidence-interval point."""

    layer: int
    token_index: int
    mean_difference: float
    lower: float
    upper: float
    support: int
    category_support: dict[str, int]


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


def token_cosine_trajectory(hiddens: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Return strict, finite cosine similarities for each token state.

    Unlike the legacy ``cosine_alignment`` helper, analysis trajectories reject
    invalid norms instead of silently turning them into zero similarities.
    """
    hiddens = np.asarray(hiddens, dtype=float)
    direction = np.asarray(direction, dtype=float)
    if hiddens.ndim != 2 or direction.ndim != 1:
        raise ValueError("hiddens must be 2-D and direction must be 1-D")
    if hiddens.shape[1] != direction.shape[0]:
        raise ValueError("hidden states and direction must have matching dimensions")
    if not np.all(np.isfinite(hiddens)) or not np.all(np.isfinite(direction)):
        raise ValueError("hidden states and direction must be finite")
    direction_norm = float(np.linalg.norm(direction))
    hidden_norms = np.linalg.norm(hiddens, axis=1)
    if direction_norm == 0.0 or np.any(hidden_norms == 0.0):
        raise ValueError("cosine inputs must not have zero norm")
    result = (hiddens @ direction) / (hidden_norms * direction_norm)
    if not np.all(np.isfinite(result)):
        raise ValueError("cosine trajectory must be finite")
    return result


def _finite_cosines(record: Mapping[str, object]) -> np.ndarray:
    """Read and validate one variable-length cosine trace."""
    values = np.asarray(record["cosines"], dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("cosines must be a non-empty 1-D trace")
    if not np.all(np.isfinite(values)):
        raise ValueError("cosine traces must be finite")
    return values


def _record_layer(record: Mapping[str, object]) -> int:
    value = record["layer"]
    if not isinstance(value, (int, float, str)):
        raise ValueError("layer must be numeric")
    return int(value)


def aggregate_token_cosine_trajectories(
    records: Iterable[Mapping[str, object]],
) -> list[TokenTrajectorySummary]:
    """Aggregate variable-length traces with equal weighting across categories.

    Token indices are one-based.  A prompt contributes only to the token
    indices present in its own trace; no padding or last-value carry is used.
    Within each category, available prompt values are averaged first, then the
    category means are averaged equally.
    """
    groups: dict[tuple[str, int], list[tuple[str, np.ndarray]]] = {}
    for record in records:
        condition = str(record["condition"])
        layer = _record_layer(record)
        category = str(record["category"])
        groups.setdefault((condition, layer), []).append(
            (category, _finite_cosines(record))
        )

    result: list[TokenTrajectorySummary] = []
    for (condition, layer), traces in sorted(groups.items(), key=lambda item: item[0]):
        max_length = max(trace.size for _, trace in traces)
        for token_index in range(1, max_length + 1):
            by_category: dict[str, list[float]] = {}
            for category, trace in traces:
                if token_index <= trace.size:
                    by_category.setdefault(category, []).append(
                        float(trace[token_index - 1])
                    )
            category_support = {
                category: len(values)
                for category, values in sorted(by_category.items())
            }
            category_means = [np.mean(values) for values in by_category.values()]
            result.append(
                {
                    "condition": condition,
                    "layer": layer,
                    "token_index": token_index,
                    "mean_cosine": float(np.mean(category_means)),
                    "support": int(sum(category_support.values())),
                    "category_support": category_support,
                }
            )
    return result


def paired_bootstrap_ci(
    records: Iterable[Mapping[str, object]],
    *,
    control: str,
    treatment: str,
    n_bootstrap: int = 1000,
    seed: int | None = 0,
    confidence: float = 0.95,
) -> list[BootstrapSummary]:
    """Compute category-stratified paired prompt bootstrap intervals.

    Each prompt is paired before resampling, and the same sampled prompt
    indices are used for both conditions.  Resampling happens independently
    within each category; category effects are then averaged equally.
    """
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")

    by_condition: dict[tuple[str, int, str, str], np.ndarray] = {}
    for record in records:
        condition = str(record["condition"])
        key = (
            condition,
            _record_layer(record),
            str(record["category"]),
            str(record["prompt_id"]),
        )
        if key in by_condition:
            raise ValueError("duplicate condition/layer/category/prompt record")
        by_condition[key] = _finite_cosines(record)

    paired: dict[tuple[int, str], list[tuple[np.ndarray, np.ndarray]]] = {}
    layers = sorted({layer for _, layer, _, _ in by_condition})
    categories = sorted({category for _, _, category, _ in by_condition})
    for layer in layers:
        for category in categories:
            prompt_ids = sorted(
                {
                    prompt_id
                    for condition, item_layer, item_category, prompt_id in by_condition
                    if item_layer == layer
                    and item_category == category
                    and condition in {control, treatment}
                }
            )
            pairs = []
            for prompt_id in prompt_ids:
                control_trace = by_condition.get((control, layer, category, prompt_id))
                treatment_trace = by_condition.get(
                    (treatment, layer, category, prompt_id)
                )
                if control_trace is not None and treatment_trace is not None:
                    pairs.append((control_trace, treatment_trace))
            if pairs:
                paired[(layer, category)] = pairs

    rng = np.random.default_rng(seed)
    alpha = (1.0 - confidence) / 2.0
    result: list[BootstrapSummary] = []
    for layer in layers:
        layer_pairs = [
            pairs for (item_layer, _), pairs in paired.items() if item_layer == layer
        ]
        max_length = max(
            (trace.size for pairs in layer_pairs for pair in pairs for trace in pair),
            default=0,
        )
        for token_index in range(1, max_length + 1):
            category_differences: dict[str, np.ndarray] = {}
            for category in categories:
                pairs = paired.get((layer, category), [])
                differences = [
                    treatment_trace[token_index - 1] - control_trace[token_index - 1]
                    for control_trace, treatment_trace in pairs
                    if token_index <= control_trace.size
                    and token_index <= treatment_trace.size
                ]
                if differences:
                    category_differences[category] = np.asarray(differences)
            if not category_differences:
                continue

            category_support = {
                category: int(values.size)
                for category, values in sorted(category_differences.items())
            }
            point = float(
                np.mean([np.mean(values) for values in category_differences.values()])
            )
            categories_for_bootstrap = tuple(category_differences.values())
            bootstrap_values = np.zeros(n_bootstrap, dtype=float)
            chunk_size = 1024
            for start in range(0, n_bootstrap, chunk_size):
                stop = min(start + chunk_size, n_bootstrap)
                chunk_means = np.empty((stop - start, len(categories_for_bootstrap)))
                for category_index, values in enumerate(categories_for_bootstrap):
                    indices = rng.integers(
                        0,
                        values.size,
                        size=(stop - start, values.size),
                    )
                    chunk_means[:, category_index] = np.mean(values[indices], axis=1)
                bootstrap_values[start:stop] = np.mean(chunk_means, axis=1)
            lower, upper = np.quantile(bootstrap_values, [alpha, 1.0 - alpha])
            result.append(
                {
                    "layer": layer,
                    "token_index": token_index,
                    "mean_difference": point,
                    "lower": float(lower),
                    "upper": float(upper),
                    "support": int(sum(category_support.values())),
                    "category_support": category_support,
                }
            )
    return result
