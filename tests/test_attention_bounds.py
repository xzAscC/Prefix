import numpy as np
import pytest

from prefix.attention_bounds import construct_shift, evaluate, jacobian_certificate


def fixture(seed=3):
    rng = np.random.default_rng(seed)
    h = rng.normal(size=(14, 12))
    wk, wv = rng.normal(size=(4, 12)), rng.normal(size=(3, 12))
    q0 = rng.normal(size=4)
    return h, wk, wv, q0


@pytest.mark.parametrize('k,m', [(1, 1), (1, 4), (4, 4)])
def test_reference_matching_and_random_query_bounds(k, m):
    h, wk, wv, q0 = fixture()
    selected, prompt = list(range(k)), list(range(6, 6 + m))
    shift = construct_shift(h, wk, wv, q0, selected, prompt)
    keys, values = h @ wk.T, h @ wv.T
    for q in [q0, q0 + np.array([0.7, -0.8, 1.0, 0.2])]:
        row = evaluate(keys, values, q0, q, selected, prompt, [10, 11],
                       wk @ shift, wv @ shift)
        assert row['error'] <= row['bound_decomp'] + 1e-10
        assert row['bound_decomp'] <= row['bound_log'] + 1e-10
        assert row['bound_log'] <= row['bound_certified'] + 1e-10
        assert row['bound_certified'] <= row['diameter'] + 1e-10
        if np.array_equal(q, q0):
            assert row['error'] < 1e-10
            assert row['beta'] < 1e-10


def test_generated_positions_need_reference_mismatch():
    # Scalar keys, values: original input=0, prompt=2, generated=0, context=0.
    keys = np.zeros((4, 1))
    values = np.array([[0.0], [2.0], [0.0], [0.0]])
    row = evaluate(keys, values, np.ones(1), np.ones(1), [0, 2], [1], [3],
                   np.array([np.log(2)]), np.ones(1))
    assert row['error'] == pytest.approx(0.3)
    assert row['epsilon_k'] == row['epsilon_r'] == 0
    assert row['beta'] > row['error']
    assert row['bound_without_beta'] == 0


def test_single_token_sensitivity_recovers_lemma2():
    h, wk, wv, q0 = fixture()
    r = construct_shift(h, wk, wv, q0, [0], [6])
    row = evaluate(h @ wk.T, h @ wv.T, q0, -q0, [0], [6], [10], wk @ r, wv @ r)
    assert row['bound_certified'] <= row['bound_lemma2'] + 1e-10


def test_jacobian_certificate_covers_dense_path_and_degenerate_values():
    rng = np.random.default_rng(17)
    values = rng.normal(size=(7, 5))
    s0, delta = rng.normal(size=7), rng.normal(size=7) * 4
    coarse = jacobian_certificate(values, s0, delta, 3, points=5)
    dense = jacobian_certificate(values, s0, delta, 3, points=501)
    assert coarse['upper'] >= dense['sample_lower'] - 1e-10
    zero = jacobian_certificate(np.ones((7, 5)), s0, delta, 3)
    assert zero['upper'] < 1e-12


def test_missing_null_direction_is_rejected():
    with pytest.raises(ValueError, match='null'):
        construct_shift(np.eye(3), np.eye(3), np.eye(3), np.ones(3), [0], [1])


@pytest.mark.parametrize('selected_count', [1, 4, 9])
def test_large_block_certificate_brackets_dense_signed_probes(selected_count):
    rng = np.random.default_rng(44)
    values = rng.normal(size=(12, 6))
    scores, delta = rng.normal(size=(2, 12))
    coarse = jacobian_certificate(values, scores, delta, selected_count, points=9)
    dense = jacobian_certificate(values, scores, delta, selected_count, points=301)
    assert coarse['sample_lower'] <= coarse['upper'] + 1e-10
    assert dense['sample_lower'] <= coarse['upper'] + 1e-10


def test_empty_shared_context_removes_weight_error():
    h, wk, wv, q0 = fixture()
    r = construct_shift(h, wk, wv, q0, [0, 1], [6, 7])
    row = evaluate(h @ wk.T, h @ wv.T, q0, -q0, [0, 1], [6, 7], [], wk @ r, wv @ r)
    assert row['ws'] == 1
    assert row['weight_error'] == 0
    assert row['error'] == pytest.approx(row['value_error'])
    assert row['error'] <= row['bound_certified'] + 1e-10


def test_jacobian_anchor_is_the_designated_input_token_not_block_order():
    h, wk, wv, q0 = fixture()
    r = construct_shift(h, wk, wv, q0, [0, 1, 2], [6, 7])
    keys, values, q = h @ wk.T, h @ wv.T, -q0
    row = evaluate(keys, values, q0, q, [0, 1, 2], [6, 7], [10], wk @ r, wv @ r, anchor=2)
    reordered = evaluate(keys, values, q0, q, [2, 0, 1], [6, 7], [10], wk @ r, wv @ r)
    assert row['jacobian_upper'] == pytest.approx(reordered['jacobian_upper'])
    assert row['jacobian_sample'] == pytest.approx(reordered['jacobian_sample'])
