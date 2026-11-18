"""Equivalence of the differentiable Kalman filter with the validated NumPy filters.

Skipped when PyTorch is not installed (``experiments/neural_r/requirements.txt``).
"""

import importlib.util

import numpy as np
import pytest

from experiments.filter_family.adaptive_r import per_step_metrics, run_recorded
from experiments.filter_family.filters import EKF
from experiments.filter_family.models import linear_cv_model, simulate
from experiments.filter_family.time_varying_r import noise_scale, simulate_tv
from experiments.linear_kf.kf import kalman_filter

# A marker instead of a module-level skip: some installed pytest plugins turn a module-level
# skip during collection into a skip of the whole session.
HAS_TORCH = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is not installed")
if HAS_TORCH:
    import torch

    from experiments.neural_r.dkf import MODELS, as_tensor, innovation_nll, run_kf

# Same operations in float64; differences are rounding only (order 1e-15 relative).
REL_TOL = 1e-10


def _rel(a, b):
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def test_dkf_reproduces_numpy_kf_with_fixed_r():
    model = linear_cv_model()
    _, y = simulate(model, 400, 20, np.random.default_rng(0))
    ref = kalman_filter(y, model.F_jacobian(None), model.H_jacobian(None), model.Q, model.R,
                        model.x0, model.P0)
    with torch.no_grad():
        o = run_kf(model, as_tensor(y))
    errs = {"x": _rel(o["x"].numpy(), ref.estimates),
            "P": _rel(o["P"].numpy(), np.broadcast_to(ref.covariances, o["P"].shape)),
            "nu": _rel(o["nu"].numpy(), ref.innovations),
            "S": _rel(o["S"].numpy(), np.broadcast_to(ref.innovation_covariances, o["S"].shape))}
    print("max relative differences:", errs)
    assert all(e < REL_TOL for e in errs.values())


def test_dkf_reproduces_true_covariance_filter_with_time_varying_r():
    model = linear_cv_model()
    c = noise_scale("burst", n_runs=10)
    x, y = simulate_tv(model, c, np.random.default_rng(0))
    R_seq = c[..., None, None] * model.R
    with torch.no_grad():
        o = run_kf(model, as_tensor(y), R_seq=as_tensor(R_seq))
    for i in range(10):
        ref = run_recorded(EKF(model, model.x0, model.P0), y[i], R_seq[i])
        assert _rel(o["x"][i].numpy(), ref["x"]) < REL_TOL
        assert _rel(o["P"][i].numpy(), ref["P"]) < REL_TOL
        m_ref = per_step_metrics(model, x[i, 1:], ref)
        nll = innovation_nll(o["nu"][i:i + 1], o["S"][i:i + 1])[0].numpy()
        assert _rel(nll, m_ref["nll"]) < REL_TOL


@pytest.mark.parametrize("kind", ["gru", "mlp"])
def test_learned_covariance_is_causal_and_positive(kind):
    """R_t depends only on measurements before t, and is positive."""
    model = linear_cv_model()
    _, y = simulate(model, 50, 2, np.random.default_rng(1))
    torch.manual_seed(0)
    net = MODELS[kind](model.m)
    y1 = as_tensor(y)
    y2 = y1.clone()
    y2[:, 30:] += 5.0  # change measurements from step 31 on
    with torch.no_grad():
        R1 = run_kf(model, y1, net=net)["R"]
        R2 = run_kf(model, y2, net=net)["R"]
    assert torch.equal(R1[:, :31], R2[:, :31])  # R at steps 1..31 uses data up to step 30 only
    assert not torch.equal(R1[:, 31:], R2[:, 31:])
    assert torch.all(R1 > 0)
