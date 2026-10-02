"""Independent numerical checks of Section 5 identities and interventions."""
import numpy as np
import pytest

from prefix.attention_cosines import attention_output, cosine_metrics, compare_preservation


def test_intervention_matches_explicit_modified_keys_and_values():
    rng = np.random.default_rng(41)
    keys, values, query = rng.normal(size=(3, 7, 4)), None, None
    keys, values, query = keys[0], keys[1], keys[2, 0]
    dk, dv = rng.normal(size=(2, 4))
    for alpha in [0, -1, 0.25, 1, 4]:
        k, v = keys.copy(), values.copy()
        k[[1, 3]] += alpha * dk
        v[[1, 3]] += alpha * dv
        scores = k @ query / 2
        weights = np.exp(scores - scores.max())
        expected = (weights / weights.sum()) @ v
        actual = attention_output(keys, values, query, [1, 3], dk, dv, alpha)
        np.testing.assert_allclose(actual, expected, atol=1e-14)


def test_baseline_and_exact_section5_decomposition():
    rng = np.random.default_rng(93)
    baseline, direction = rng.normal(size=(2, 6))
    zero = cosine_metrics(baseline, baseline, direction)
    assert zero['R'] == pytest.approx(1)
    assert zero['delta_C'] == pytest.approx(0)
    assert zero['delta_perp'] == pytest.approx(0)
    for output in rng.normal(size=(20, 6)):
        row = cosine_metrics(output, baseline, direction)
        assert row['R'] == pytest.approx(1 - (row['delta_C']**2 + row['delta_perp']) / 2)


def test_theorem_signed_identity_and_lower_bound_for_random_outputs():
    rng = np.random.default_rng(53)
    for _ in range(100):
        baseline, direction, short, long = rng.normal(size=(4, 5))
        a = cosine_metrics(short, baseline, direction)
        b = cosine_metrics(long, baseline, direction)
        comparison = compare_preservation(a, b)
        assert comparison['R_gap'] == pytest.approx(comparison['identity_rhs'])
        assert comparison['R_gap'] >= comparison['lower_bound'] - 1e-14


@pytest.mark.parametrize('which', range(3))
def test_undefined_cosines_fail_explicitly(which):
    vectors = [np.ones(3), np.ones(3), np.ones(3)]
    vectors[which] = np.zeros(3)
    with pytest.raises(ValueError, match='nonzero'):
        cosine_metrics(*vectors)


def test_attention_is_stable_under_large_common_score_offset():
    keys = np.array([[1e5, 1], [1e5, 2.]])
    values = np.eye(2)
    q = np.array([1., 1.])
    np.testing.assert_allclose(attention_output(keys, values, q),
                               attention_output(keys - [1e5, 0], values, q), atol=1e-11)


def test_numerically_zero_concept_retains_R_and_marks_C_undefined():
    from prefix.attention_cosines import measured_cosines
    row = measured_cosines(np.array([1.,1.]), np.array([1.,0.]), np.zeros(2))
    assert row['R'] == pytest.approx(1 / np.sqrt(2))
    assert row['C'] is None
    assert row['delta_C'] is None
    assert row['delta_perp'] is None
    assert row['concept_defined'] is False
