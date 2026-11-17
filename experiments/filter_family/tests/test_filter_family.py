"""Tests for the filter family.

Run from the repository root with ``python -m pytest -v -s experiments/filter_family`` to see
the measured values.
"""

import numpy as np
import pytest

from experiments.filter_family.filters import EKF, IEKF, KF, UKF, AdaptiveEKF, run_filter
from experiments.filter_family.models import (
    linear_cv_model,
    range_bearing_model,
    simulate,
    wrap_angle,
)
from experiments.linear_kf.kf import kalman_filter

N_TRAJ = 20
STEPS = 200
SEED = 0
# On a linear model EKF and IEKF are algebraically identical to the KF, and the unscented
# transform with alpha = 1, beta = 2, kappa = 0 (non-negative weights) is exact for linear maps.
# Remaining differences are floating-point rounding (order 1e-13 relative); an error in a
# formula gives differences of order one.
EQUIV_REL_TOL = 1e-9


@pytest.fixture(scope="module")
def linear_data():
    model = linear_cv_model()
    x, y = simulate(model, STEPS, N_TRAJ, np.random.default_rng(SEED))
    ref = kalman_filter(y, model.F_jacobian(None), model.H_jacobian(None), model.Q, model.R,
                        model.x0, model.P0)
    return model, x, y, ref


@pytest.mark.parametrize("cls", [EKF, IEKF, UKF])
def test_nonlinear_filters_reproduce_kf_on_linear_model(linear_data, cls):
    model, _, y, ref = linear_data
    x_tol = EQUIV_REL_TOL * max(1.0, float(np.max(np.abs(ref.estimates))))
    p_tol = EQUIV_REL_TOL * float(np.max(np.abs(ref.covariances)))
    dx = dp = 0.0
    for i in range(N_TRAJ):
        out = run_filter(cls(model, model.x0, model.P0), y[i])
        dx = max(dx, float(np.max(np.abs(out["x"] - ref.estimates[i]))))
        dp = max(dp, float(np.max(np.abs(out["P"] - ref.covariances))))
    print(f"\n{cls.name}: max |x - x_KF| = {dx:.3e} (tolerance {x_tol:.3e}), "
          f"max |P - P_KF| = {dp:.3e} (tolerance {p_tol:.3e}), "
          f"{N_TRAJ} trajectories x {STEPS} steps")
    assert dx <= x_tol
    assert dp <= p_tol


def test_adaptive_off_equals_kf_exactly(linear_data):
    model, _, y, _ = linear_data
    for i in range(N_TRAJ):
        a = run_filter(AdaptiveEKF(model, model.x0, model.P0, adapt=False), y[i])
        b = run_filter(KF(model, model.x0, model.P0), y[i])
        for key in ("x", "P", "nu", "S"):
            assert np.array_equal(a[key], b[key]), key
    print(f"\nadaptive (off) vs KF: bitwise identical for {N_TRAJ} trajectories x {STEPS} steps")


# Mean over runs of the estimated R at the final step must be within 10 % of R_true. With a
# window of 20 residuals a single estimate has a relative standard deviation of about
# sqrt(2/20) = 32 %; the mean over 200 runs reduces this to about 2.3 %.
ADAPT_RUNS = 200
ADAPT_WINDOW = 20
ADAPT_REL_TOL = 0.10


def test_adaptive_on_recovers_R():
    model = linear_cv_model()
    _, y = simulate(model, STEPS, ADAPT_RUNS, np.random.default_rng(SEED + 1))
    r_true = float(model.R[0, 0])
    hist = []
    for i in range(ADAPT_RUNS):
        f = AdaptiveEKF(model, model.x0, model.P0, R=100.0 * model.R, window=ADAPT_WINDOW)
        run_filter(f, y[i])
        hist.append([r[0, 0] for r in f.R_history])
    hist = np.array(hist)  # (runs, steps): R used at update k = 1..STEPS
    print(f"\nR_true = {r_true}, initial R_filter = {100.0 * r_true}, window = {ADAPT_WINDOW}, "
          f"{ADAPT_RUNS} runs")
    for k in (1, 20, 21, 22, 25, 30, 40, 50, 100, 150, 200):
        col = hist[:, k - 1]
        print(f"  update k = {k:3d}: mean R = {col.mean():9.4f}, std over runs = {col.std():8.4f}, "
              f"mean / R_true = {col.mean() / r_true:.4f}")
    final = hist[:, -1].mean()
    assert abs(final / r_true - 1.0) <= ADAPT_REL_TOL


# Central differences with step 1e-6 have truncation and rounding errors far below 1e-6
# relative for states at ranges of metres to hundreds of metres.
JAC_REL_TOL = 1e-6


def test_jacobians_match_finite_differences():
    model = range_bearing_model()
    rng = np.random.default_rng(SEED)
    states = np.column_stack([rng.uniform(-100, 100, (50, 2)), rng.uniform(-3, 3, (50, 2))])
    states = states[np.hypot(states[:, 0], states[:, 1]) > 5.0]
    h_step = 1e-6
    worst_h = worst_f = 0.0
    for x in states:
        for fun, jac, name in ((model.h, model.H_jacobian, "h"), (model.f, model.F_jacobian, "f")):
            J = jac(x)
            J_fd = np.empty_like(J)
            for j in range(4):
                d = np.zeros(4)
                d[j] = h_step
                diff = fun(x + d) - fun(x - d)
                if name == "h":
                    diff[1] = wrap_angle(diff[1])
                J_fd[:, j] = diff / (2.0 * h_step)
            rel = np.max(np.abs(J - J_fd)) / max(1e-12, np.max(np.abs(J)))
            if name == "h":
                worst_h = max(worst_h, rel)
            else:
                worst_f = max(worst_f, rel)
    print(f"\n{len(states)} states: max relative |J - J_fd| for h = {worst_h:.3e}, "
          f"for f = {worst_f:.3e} (tolerance {JAC_REL_TOL:.0e})")
    assert worst_h <= JAC_REL_TOL
    assert worst_f <= JAC_REL_TOL


def test_bearing_wrap_handling():
    model = range_bearing_model()
    eps = 0.01
    # Residual across the +pi / -pi boundary.
    d = model.residual(np.array([10.0, -np.pi + eps]), np.array([10.0, np.pi - eps]))
    assert d[1] == pytest.approx(2 * eps)
    # Weighted mean of bearings straddling the boundary must lie near pi, not near 0.
    Z = np.array([[10.0, np.pi - eps], [10.0, -np.pi + eps]])
    mean = model.measurement_mean(Z, np.array([0.5, 0.5]))
    assert abs(wrap_angle(mean[1] - np.pi)) < 1e-12
    print(f"\nresidual across boundary = {d[1]:.6f} (expected {2 * eps}), "
          f"mean bearing = {mean[1]:.6f}")
