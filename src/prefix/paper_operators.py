"""Operators and coefficients defined in the ICLR manuscript.

COAST solves the quadratic objective on a sphere intersected with a plane.
The eigendecomposition is cached once per direction/reference second moment.
"""
from __future__ import annotations

import math

import torch


class CoastOperator:
    def __init__(self, direction: torch.Tensor, second_moment: torch.Tensor):
        r = direction.double()
        if r.ndim != 1 or r.numel() < 2 or not torch.isfinite(r).all() or r.norm() == 0:
            raise ValueError('direction must be a finite, nonzero vector of dimension >= 2')
        self.r = r / r.norm()
        sigma = second_moment.to(device=r.device, dtype=r.dtype)
        if sigma.shape != (r.numel(), r.numel()) or not torch.isfinite(sigma).all():
            raise ValueError('second moment must be a finite square matrix matching direction')
        if not torch.allclose(sigma, sigma.T, atol=1e-7, rtol=1e-6):
            raise ValueError('second moment must be symmetric')
        self.sigma = (sigma + sigma.T) / 2
        # Householder's remaining columns give an orthonormal basis of r-perp.
        v = self.r.clone()
        v[0] += 1. if v[0] >= 0 else -1.
        v /= v.norm()
        householder = torch.eye(r.numel(), device=r.device, dtype=r.dtype) - 2 * v[:, None] * v[None, :]
        basis = householder[:, 1:]
        eigenvalues, rotation = torch.linalg.eigh(basis.T @ self.sigma @ basis)
        self.eigenvalues = eigenvalues
        self.basis = basis @ rotation
        self.projected_metric = self.basis.T @ self.sigma

    def __call__(self, hidden: torch.Tensor, target: float) -> torch.Tensor:
        if not math.isfinite(target) or not -1 <= target <= 1:
            raise ValueError('target cosine must be in [-1, 1]')
        if hidden.ndim != 1 or hidden.numel() != self.r.numel():
            raise ValueError('hidden must match direction')
        h = hidden.to(device=self.r.device, dtype=torch.float64)
        norm = h.norm()
        if not torch.isfinite(h).all():
            raise ValueError('hidden must be finite')
        if norm == 0:
            return hidden.clone()
        radius = math.sqrt(max(0., 1 - target * target))
        if radius == 0:
            return (norm * target * self.r).to(hidden)
        b = self.projected_metric @ (target * self.r - h / norm)
        gaps = self.eigenvalues - self.eigenvalues[0]
        tolerance = 1e-12 * max(1., float(self.eigenvalues.abs().max()))
        minimum = gaps <= tolerance
        at_boundary = torch.where(minimum, 0., -b / gaps.clamp_min(tolerance))
        # Trust-region hard case: the linear term is orthogonal to the lowest
        # eigenspace, leaving the remaining radius in that eigenspace.
        if b[minimum].norm() <= tolerance and at_boundary.norm() <= radius:
            z = at_boundary
            first = int(torch.nonzero(minimum)[0].item())
            z[first] = math.sqrt(max(0., radius * radius - float(z.square().sum())))
        else:
            lower = 0.
            upper = max(1., float(b.norm()) / radius)
            while (-b / (gaps + upper)).norm() > radius:
                upper *= 2
            for _ in range(80):
                middle = (lower + upper) / 2
                if (-b / (gaps + middle)).norm() > radius:
                    lower = middle
                else:
                    upper = middle
            z = -b / (gaps + upper)
        return (norm * (target * self.r + self.basis @ z)).to(hidden)


def coast(hidden: torch.Tensor, direction: torch.Tensor, target: float,
          second_moment: torch.Tensor) -> torch.Tensor:
    return CoastOperator(direction, second_moment)(hidden, target)


def das_coefficient(base_logits: torch.Tensor, probe_logits: torch.Tensor,
                    *, top_p: float = .9, maximum: float = 2.) -> float:
    """KL(base || probe) on the union of nucleus supports, capped at c_max."""
    if not 0 < top_p <= 1 or maximum <= 0:
        raise ValueError('top_p must be in (0, 1] and maximum must be positive')
    if base_logits.ndim != 1 or base_logits.shape != probe_logits.shape:
        raise ValueError('logits must be matching vocabulary vectors')
    support = torch.zeros_like(base_logits, dtype=torch.bool)
    for logits in (base_logits, probe_logits):
        probabilities, indices = logits.double().softmax(-1).sort(descending=True)
        # Include the first token that reaches/exceeds the requested mass.
        keep = probabilities.cumsum(0) - probabilities < top_p
        support[indices[keep]] = True
    log_p = base_logits[support].double().log_softmax(0)
    log_q = probe_logits[support].double().log_softmax(0)
    kl = (log_p.exp() * (log_p - log_q)).sum()
    return min(maximum, max(0., float(kl)))


def strength_at(policy: str, position: int, initial: float, *, length: int = 5,
                tau: float = 128., concept_probability: float | None = None,
                act_amplitude: float = 12., act_bias: float = 0.) -> float:
    """Position zero is the final input token, followed by generated tokens."""
    if position < 0 or length < 1 or tau <= 0:
        raise ValueError('position >= 0, length >= 1, and tau > 0 are required')
    if policy == 'full':
        return initial
    if policy == 'prefix':
        return initial if position < length else 0.
    if policy == 'linear':
        return initial * max(1. - position / length, 0.)
    if policy == 'exponential':
        return initial * math.exp(-position / tau)
    if policy == 'act':
        if concept_probability is None or not 0 <= concept_probability <= 1:
            raise ValueError('ACT requires a concept probability in [0, 1]')
        return act_amplitude * (1. - concept_probability + act_bias)
    raise ValueError(f'unknown scalar policy: {policy}')
