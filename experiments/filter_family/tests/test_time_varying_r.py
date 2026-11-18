"""Tests for the time-varying measurement noise benchmark and the adaptive-R diagnostics."""

import numpy as np

from scipy.stats import multivariate_normal

from experiments.filter_family.adaptive_r import per_step_metrics, run_recorded, windowed_estimate
from experiments.filter_family.filters import EKF, AdaptiveEKF, run_filter
from experiments.filter_family.models import linear_cv_model, range_bearing_model, simulate
from experiments.filter_family.time_varying_r import noise_scale, simulate_tv


def test_simulate_tv_with_unit_scale_matches_simulate():
    for model in (linear_cv_model(), range_bearing_model()):
        c = noise_scale("constant", steps=50, n_runs=5)
        x_a, y_a = simulate(model, 50, 5, np.random.default_rng(3))
        x_b, y_b = simulate_tv(model, c, np.random.default_rng(3))
        assert np.array_equal(x_a, x_b)
        assert np.array_equal(y_a, y_b)


def test_patterns():
    step = noise_scale("step")[0]
    assert np.all(step[:200] == 1.0) and np.all(step[200:] == 10.0)
    ramp = noise_scale("ramp")[0]
    assert ramp[0] == 1.0 and np.isclose(ramp[-1], 10.0)
    burst = noise_scale("burst")
    assert np.allclose(burst.mean(axis=1), 3.4)
    for row in burst:
        for b in range(4):
            block = row[100 * b:100 * (b + 1)]
            assert np.sum(block == 25.0) == 10
            idx = np.flatnonzero(block == 25.0)
            assert np.all(np.diff(idx) == 1)  # one contiguous burst per block


def test_true_covariance_filter_with_constant_r_equals_ekf():
    model = range_bearing_model()
    _, y = simulate(model, 60, 1, np.random.default_rng(0))
    ref = run_filter(EKF(model, model.x0, model.P0), y[0])
    R_seq = np.repeat(model.R[None], 60, axis=0)
    o = run_recorded(EKF(model, model.x0, model.P0), y[0], R_seq)
    assert np.array_equal(ref["x"], o["x"])


def test_recorded_r_matches_adaptive_filter_history_and_estimator():
    model = range_bearing_model()
    _, y = simulate(model, 60, 1, np.random.default_rng(0))
    filt = AdaptiveEKF(model, model.x0, model.P0, window=10)
    o = run_recorded(filt, y[0])
    assert np.array_equal(np.array(filt.R_history), o["R_used"])
    est = windowed_estimate(o["e"], o["HPH_post"], 10, +1.0)
    # The R used at update k + 1 is the estimate formed after update k.
    assert np.allclose(est[9:-1], o["R_used"][10:], rtol=1e-12, atol=0.0)
    assert np.all(np.linalg.eigvalsh(o["R_used"][10:])[:, 0] >= 0.0)


def test_innovation_nll_uses_innovation_covariance():
    """NLL per step equals -log N(nu; 0, S) with S = H P- H^T + R the filter's innovation covariance."""
    model = range_bearing_model()
    x, y = simulate(model, 60, 1, np.random.default_rng(0))
    o = run_recorded(AdaptiveEKF(model, model.x0, model.P0, window=10), y[0])
    nll = per_step_metrics(model, x[0, 1:], o)["nll"]
    ref = [-multivariate_normal(mean=np.zeros(2), cov=S).logpdf(nu) for nu, S in zip(o["nu"], o["S"])]
    assert np.allclose(nll, ref, rtol=1e-12, atol=1e-12)
