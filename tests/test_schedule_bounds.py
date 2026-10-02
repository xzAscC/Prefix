import numpy as np
import pytest
from prefix.schedule_bounds import evaluate_schedules
from prefix.attention_cosines import attention_output


def test_bounds_and_decomposition_against_independent_replay():
    rng = np.random.default_rng(18)
    for _ in range(50):
        keys,values = rng.normal(size=(2,8,4))
        q,dk,dv = rng.normal(size=(3,4))
        row = evaluate_schedules(keys,values,q,[1],[1,3,6],dk,dv)
        diff = attention_output(keys,values,q,[1,3,6],dk,dv)-attention_output(keys,values,q,[1],dk,dv)
        assert row['error'] == pytest.approx(np.linalg.norm(diff))
        assert row['identity_residual'] < 1e-12
        assert row['error'] <= row['bound_shares'] + 1e-12
        assert row['bound_shares'] <= row['bound_score'] + 1e-12
        assert row['bound_shares'] <= row['bound_epsilon'] + 1e-12


def test_empty_extra_set_and_all_visible_extra_set():
    k,v = np.zeros((2,3)), np.eye(3)[:2]
    q,dk,dv = np.ones(3),np.ones(3),np.array([1.,0,0])
    assert evaluate_schedules(k,v,q,[0],[0],dk,dv)['error'] == pytest.approx(0)
    row = evaluate_schedules(k,v,q,[],[0,1],dk,dv)
    assert row['error'] == pytest.approx(1)
    assert row['w_short'] == row['w_long'] == 1


def test_non_nested_sets_rejected():
    with pytest.raises(ValueError,match='nested'):
        evaluate_schedules(np.zeros((2,2)),np.eye(2),np.ones(2),[0],[1],np.ones(2),np.ones(2))


def test_small_before_share_is_not_sufficient_after_large_key_shift():
    row = evaluate_schedules(np.array([[0.],[-20.]]),np.array([[0.],[1.]]),np.ones(1),[],[1],np.array([40.]),np.ones(1))
    assert row['w_short'] < 1e-8
    assert row['w_long'] > .99
    assert row['error'] > 1.9


def test_short_diameter_includes_existing_interventions():
    # Originally all values coincide. The already-steered position separates
    # them, and using the original zero diameter would invalidate the bound.
    row = evaluate_schedules(np.zeros((3,1)),np.zeros((3,1)),np.ones(1),
                             [0],[0,1],np.array([-2.]),np.ones(1))
    assert row['diameter_short'] == pytest.approx(1.)
    assert row['error'] > row['w_long'] * row['value_shift_norm']
    assert row['error'] <= row['bound_shares']
