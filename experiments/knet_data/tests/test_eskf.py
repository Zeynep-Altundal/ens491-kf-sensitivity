"""ESKF = EKF for a vector state.

Run from the repository root with ``python -m pytest -v experiments/knet_data``.
"""

import numpy as np

from experiments.knet_data import common, sinusoidal as sin


def test_eskf_equals_ekf_on_sinusoidal_model():
    # Small non-linear example: 5 trajectories of the sinusoidal benchmark at 1/r^2 = 0 dB.
    p = sin.PARAMS["full"]
    q2, r2 = sin.noise_levels(0.0)
    _, y = sin.simulate(p, q2, r2, 5, np.random.default_rng(0))
    model = sin.ekf_model(p, q2 * np.eye(2), r2 * np.eye(2))
    F_jac = lambda x: np.stack([model.F_jacobian(xi) for xi in x])
    H_jac = lambda x: np.stack([model.H_jacobian(xi) for xi in x])
    args = (y, model.f, F_jac, model.h, H_jac, model.Q, model.R, model.x0, model.P0)
    x_ekf, P_ekf = common.batched_ekf(*args)
    x_eskf, P_eskf = common.batched_eskf(*args)
    np.testing.assert_allclose(x_eskf, x_ekf, rtol=0, atol=1e-12)
    np.testing.assert_allclose(P_eskf, P_ekf, rtol=0, atol=1e-12)