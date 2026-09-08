"""Tests for residual-stream activation steering utilities."""

from __future__ import annotations

import pytest
import torch

from prefix.steering import (
    SteeringSchedule,
    applied_perturbation,
    apply_steering,
    dim_direction,
    mean_hidden_norm,
)


def test_dim_direction_returns_normalized_mean_difference() -> None:
    pos = torch.tensor([[3.0, 0.0], [1.0, 0.0]])
    neg = torch.tensor([[0.0, 4.0], [0.0, 2.0]])

    result = dim_direction(pos, neg)

    expected = torch.tensor([2.0, -3.0]) / 13**0.5
    assert result.dtype == torch.float32
    assert torch.allclose(result, expected)
    assert torch.isclose(torch.linalg.vector_norm(result), torch.tensor(1.0))


@pytest.mark.parametrize(
    ("pos", "neg"),
    [
        (torch.empty((0, 2)), torch.ones((1, 2))),
        (torch.ones((1, 2)), torch.empty((0, 2))),
        (torch.ones((1, 2)), torch.ones((1, 3))),
        (torch.ones(2), torch.ones((1, 2))),
    ],
)
def test_dim_direction_rejects_empty_or_mismatched_inputs(
    pos: torch.Tensor, neg: torch.Tensor
) -> None:
    with pytest.raises(ValueError):
        dim_direction(pos, neg)


def test_mean_hidden_norm_matches_hand_computed_value() -> None:
    hiddens = torch.tensor([[3.0, 4.0], [0.0, 12.0]])

    assert mean_hidden_norm(hiddens) == pytest.approx(8.5)


def test_applied_perturbation_scales_direction_and_preserves_dtype() -> None:
    direction = torch.tensor([0.6, -0.8], dtype=torch.float32)

    result = applied_perturbation(direction, alpha=0.25, mean_norm=8.0)

    assert result.dtype == torch.float32
    assert torch.equal(result, torch.tensor([1.2, -1.6], dtype=torch.float32))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_apply_steering_is_out_of_place_and_preserves_dtype(
    dtype: torch.dtype,
) -> None:
    hidden = torch.tensor([1.0, 2.0], dtype=dtype)
    direction = torch.tensor([0.5, -1.0], dtype=dtype)

    result = apply_steering(hidden, direction, beta=2.0)

    assert result.dtype == dtype
    assert result.shape == hidden.shape
    assert torch.allclose(result.float(), torch.tensor([2.0, 0.0]))
    assert torch.equal(hidden, torch.tensor([1.0, 2.0], dtype=dtype))


def test_apply_steering_broadcasts_direction_over_batch() -> None:
    hidden = torch.zeros((2, 3), dtype=torch.float32)
    direction = torch.tensor([1.0, -2.0, 0.5])

    result = apply_steering(hidden, direction, beta=3.0)

    assert result.shape == hidden.shape
    assert torch.equal(result[0], torch.tensor([3.0, -6.0, 1.5]))
    assert torch.equal(result[1], result[0])


def test_prefix_schedule_uses_last_prompt_and_first_m_minus_one_decodes() -> None:
    schedule = SteeringSchedule.prefix(5)

    assert schedule.intervene_on_prefill() is True
    assert [schedule.intervene_on_decode(k) for k in range(1, 8)] == [
        True,
        True,
        True,
        True,
        False,
        False,
        False,
    ]
    assert schedule.positions() == "t=1..5"


def test_one_token_schedule_only_intervenes_on_last_prompt_token() -> None:
    schedule = SteeringSchedule.one_token()

    assert schedule.intervene_on_prefill() is True
    assert all(not schedule.intervene_on_decode(k) for k in range(1, 20))
    assert schedule.positions() == "t=1 only"


def test_full_schedule_intervenes_on_prefill_and_every_decode() -> None:
    schedule = SteeringSchedule.full()

    assert schedule.intervene_on_prefill() is True
    assert all(schedule.intervene_on_decode(k) for k in range(1, 20))
    assert schedule.positions() == "t=1 onward"


def test_schedule_rejects_invalid_prefix_and_decode_index() -> None:
    with pytest.raises(ValueError, match="m"):
        SteeringSchedule.prefix(0)
    with pytest.raises(ValueError, match="1-based"):
        SteeringSchedule.full().intervene_on_decode(0)
