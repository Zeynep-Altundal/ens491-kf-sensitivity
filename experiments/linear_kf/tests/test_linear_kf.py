"""Consistency and reproducibility tests for the linear Kalman filter study.

Run from the repository root with ``python -m pytest -v -s`` to see the
measured values.
"""

import numpy as np
import pytest

from experiments.linear_kf.kf import kalman_filter
from experiments.linear_kf.model import DEFAULT_STEPS, constant_velocity_model, simulate
from experiments.linear_kf.sweep import DEFAULT_SEED, N_TRAJ, evaluate, run_sweeps

REL_TOL = 0.10


@pytest.fixture(scope="module")
def model():
    return constant_velocity_model()


@pytest.fixture(scope="module")
def data(model):
    return simulate(model, DEFAULT_STEPS, N_TRAJ, np.random.default_rng(DEFAULT_SEED))


def _true_params(model):
    return dict(F=model.F, H=model.H, Q=model.Q, R=model.R, x0=model.x0, P0=model.P0)


def test_nees_nis_consistent_with_true_noise(model, data):
    x_true, y = data
    metrics = evaluate(x_true, y, **_true_params(model))
    print(
        f"\nmean NEES = {metrics['mean_nees']:.4f} (expected {model.n}), "
        f"mean NIS = {metrics['mean_nis']:.4f} (expected {model.m})"
    )
    assert abs(metrics["mean_nees"] - model.n) <= REL_TOL * model.n
    assert abs(metrics["mean_nis"] - model.m) <= REL_TOL * model.m


def test_empirical_error_covariance_matches_steady_state(model, data):
    x_true, y = data
    res = kalman_filter(y, **_true_params(model))
    k0 = DEFAULT_STEPS // 2
    P_ss = res.covariances[-1]

    # The second half of the run is treated as steady state; verify that P has converged.
    drift = np.max(np.abs(res.covariances[k0:] - P_ss) / np.abs(P_ss))
    assert drift < 1e-3

    err = (res.estimates - x_true[:, 1:, :])[:, k0:, :].reshape(-1, model.n)
    C_emp = err.T @ err / err.shape[0]
    scale = np.sqrt(np.outer(np.diag(P_ss), np.diag(P_ss)))
    rel = np.abs(C_emp - P_ss) / scale
    print(f"\nsteady-state P =\n{P_ss}\nempirical covariance =\n{C_emp}")
    print(f"|C_emp - P| / sqrt(P_ii P_jj) =\n{rel}")
    assert np.all(rel <= REL_TOL)


def test_same_seed_gives_identical_results(model):
    kwargs = dict(steps=50, n_traj=20, s_grid=np.array([1e-3, 1.0, 1e3]))
    first = run_sweeps(model, seed=123, **kwargs)
    second = run_sweeps(model, seed=123, **kwargs)
    other = run_sweeps(model, seed=124, **kwargs)
    for sweep in first:
        assert np.array_equal(first[sweep], second[sweep])
    assert any(not np.array_equal(first[s], other[s]) for s in first)


@pytest.mark.parametrize("c", [1e-3, 1e-1, 1e1, 1e3])
def test_common_scale_invariance(model, data, c):
    """Scaling (Q_f, R_f, P0_f) by c leaves RMSE unchanged and divides NEES and NIS by c."""
    x_true, y = data
    base_params = _true_params(model)
    scaled = dict(base_params, Q=c * model.Q, R=c * model.R, P0=c * model.P0)
    base = evaluate(x_true, y, **base_params)
    out = evaluate(x_true, y, **scaled)
    print(
        f"\nc = {c:g}: pos RMSE {out['pos_rmse']:.12g} vs {base['pos_rmse']:.12g}, "
        f"vel RMSE {out['vel_rmse']:.12g} vs {base['vel_rmse']:.12g}, "
        f"c*NEES {c * out['mean_nees']:.12g} vs {base['mean_nees']:.12g}, "
        f"c*NIS {c * out['mean_nis']:.12g} vs {base['mean_nis']:.12g}"
    )
    rtol = 1e-9
    assert out["pos_rmse"] == pytest.approx(base["pos_rmse"], rel=rtol)
    assert out["vel_rmse"] == pytest.approx(base["vel_rmse"], rel=rtol)
    assert c * out["mean_nees"] == pytest.approx(base["mean_nees"], rel=rtol)
    assert c * out["mean_nis"] == pytest.approx(base["mean_nis"], rel=rtol)
