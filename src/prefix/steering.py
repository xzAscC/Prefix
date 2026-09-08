"""Pure-torch utilities for residual-stream activation steering."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _require_matrix(value: torch.Tensor, name: str) -> None:
    if value.ndim != 2:
        raise ValueError(f"{name} must have shape [N, d]")
    if value.shape[0] == 0:
        raise ValueError(f"{name} must not be empty")


def dim_direction(pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    """Construct a unit direction from positive and negative activations."""
    _require_matrix(pos, "pos")
    _require_matrix(neg, "neg")
    if pos.shape[1] != neg.shape[1]:
        raise ValueError("pos and neg must have matching hidden dimensions")

    difference = pos.float().mean(dim=0) - neg.float().mean(dim=0)
    norm = torch.linalg.vector_norm(difference)
    if norm == 0:
        raise ValueError("mean difference must have nonzero norm")
    return difference / norm


def mean_hidden_norm(hiddens: torch.Tensor) -> float:
    """Return the mean L2 norm across rows of a hidden-state matrix."""
    _require_matrix(hiddens, "hiddens")
    return float(torch.linalg.vector_norm(hiddens.float(), dim=1).mean().item())


def applied_perturbation(
    direction: torch.Tensor, alpha: float, mean_norm: float
) -> torch.Tensor:
    """Scale a direction by the normalized steering strength."""
    return direction * (alpha * mean_norm)


def apply_steering(
    hidden: torch.Tensor, direction: torch.Tensor, beta: float
) -> torch.Tensor:
    """Return hidden states with a broadcast residual-stream perturbation."""
    return hidden + direction.to(dtype=hidden.dtype) * beta


@dataclass(frozen=True)
class SteeringSchedule:
    """Describe which prompt/decode positions receive an intervention."""

    kind: str
    length: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"full", "prefix", "one_token"}:
            raise ValueError(f"invalid steering schedule kind: {self.kind!r}")

    @classmethod
    def full(cls) -> SteeringSchedule:
        return cls("full")

    @classmethod
    def prefix(cls, m: int) -> SteeringSchedule:
        if m < 1:
            raise ValueError("prefix length m must be at least 1")
        return cls("prefix", m)

    @classmethod
    def one_token(cls) -> SteeringSchedule:
        return cls("one_token")

    def intervene_on_prefill(self) -> bool:
        return True

    def intervene_on_decode(self, k: int) -> bool:
        if k < 1:
            raise ValueError("decode index k is 1-based")
        if self.kind == "one_token":
            return False
        if self.kind == "prefix":
            assert self.length is not None
            return k <= self.length - 1
        return True

    def positions(self) -> str:
        if self.kind == "full":
            return "t=1 onward"
        if self.kind == "one_token":
            return "t=1 only"
        assert self.length is not None
        return f"t=1..{self.length}"


__all__ = [
    "SteeringSchedule",
    "applied_perturbation",
    "apply_steering",
    "dim_direction",
    "mean_hidden_norm",
]
