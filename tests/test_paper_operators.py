"""Numerical checks against the manuscript's constrained objectives."""
import math

import pytest
import torch

from prefix.paper_operators import coast, das_coefficient, strength_at


def test_coast_is_global_minimum_on_feasible_circle():
    h = torch.tensor([0.3, -0.8, 0.4], dtype=torch.float64)
    r = torch.tensor([1., 0., 0.], dtype=torch.float64)
    sigma = torch.tensor([[2., .4, .1], [.4, 4., .3], [.1, .3, 1.]], dtype=torch.float64)
    alpha = 0.6
    actual = coast(h, r, alpha, sigma)
    assert torch.linalg.vector_norm(actual).item() == pytest.approx(h.norm().item())
    assert (actual @ r / actual.norm()).item() == pytest.approx(alpha)
    angle = torch.linspace(-math.pi, math.pi, 20001, dtype=torch.float64)
    candidates = torch.stack([torch.full_like(angle, alpha), .8 * angle.cos(), .8 * angle.sin()], -1)
    delta = candidates - h / h.norm()
    scores = torch.einsum('ni,ij,nj->n', delta, sigma, delta)
    diff = actual / h.norm() - h / h.norm()
    assert (diff @ sigma @ diff).item() <= scores.min().item() + 1e-9


def test_coast_handles_hard_case_and_target_endpoints():
    h = torch.tensor([1., 0., 0.], dtype=torch.float64)
    sigma = torch.diag(torch.tensor([1., 2., 3.], dtype=torch.float64))
    result = coast(h, h, 0., sigma)
    assert abs(result[1].item()) == pytest.approx(1.)
    assert result[2].item() == pytest.approx(0.)
    for alpha in [-1., 1.]:
        torch.testing.assert_close(coast(h, h, alpha, sigma), alpha * h)
    torch.testing.assert_close(coast(torch.zeros_like(h), h, .5, sigma), torch.zeros_like(h))


def test_coast_identity_metric_matches_nearest_feasible_vector():
    h = torch.tensor([2., 3., 4.], dtype=torch.float64)
    r = torch.tensor([1., 0., 0.], dtype=torch.float64)
    expected = h.norm() * torch.tensor([.6, .48, .64], dtype=torch.float64)
    torch.testing.assert_close(coast(h, r, .6, torch.eye(3)), expected)


def test_das_uses_union_and_renormalizes_before_kl():
    p = torch.tensor([.7, .2, .1], dtype=torch.float64)
    q = torch.tensor([.1, .8, .1], dtype=torch.float64)
    expected = ((p[:2] / .9) * ((p[:2] / .9).log() - (q[:2] / .9).log())).sum()
    assert das_coefficient(p.log(), q.log(), top_p=.65) == pytest.approx(expected.item())
    assert das_coefficient(p.log(), p.log()) == pytest.approx(0.)
    assert das_coefficient(p.log(), q.log(), maximum=.01) == pytest.approx(.01)


def test_policy_positions_include_final_prompt_token_as_zero():
    assert [strength_at('prefix', t, 2., length=1) for t in range(3)] == [2., 0., 0.]
    assert [strength_at('linear', t, 2., length=2) for t in range(4)] == [2., 1., 0., 0.]
    assert strength_at('exponential', 128, 2.) == pytest.approx(2. / math.e)
    assert strength_at('act', 0, 1., concept_probability=.25) == 9.
    with pytest.raises(ValueError):
        strength_at('act', 0, 1.)
