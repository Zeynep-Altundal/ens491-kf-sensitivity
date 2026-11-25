"""Tests for the sinusoidal KalmanNet benchmark data.

Run from the repository root with ``python -m pytest -v experiments/knet_data``.
"""

import numpy as np
import pytest

from experiments.knet_data import common, sinusoidal as sin


@pytest.fixture(autouse=True)
def tmp_data_dir(tmp_path, monkeypatch):
    # Write datasets to a temporary folder instead of ~/data/knet_benchmarks.
    monkeypatch.setattr(sin, "DATA_DIR", tmp_path)


def test_shapes_and_noise_levels():
    arrays, meta = common.load_dataset(sin.generate(0.0))
    assert arrays["x_train"].shape == (common.N_TRAIN, sin.T, 2)
    assert arrays["y_test"].shape == (common.N_TEST, sin.T, 2)
    # 1/r^2 = 0 dB -> r^2 = 1; nu = -20 dB -> q^2 = 0.01
    np.testing.assert_allclose(np.diag(arrays["R"]), 1.0)
    np.testing.assert_allclose(np.diag(arrays["Q"]), 0.01)
    assert meta["params_true"] == sin.PARAMS["full"]


def test_same_seed_same_data():
    a, _ = common.load_dataset(sin.generate(20.0, seed=5))
    b, _ = common.load_dataset(sin.generate(20.0, seed=5))
    np.testing.assert_array_equal(a["y_val"], b["y_val"])


def test_model_functions():
    p = sin.PARAMS["full"]
    x = np.array([0.0, 1.0])
    np.testing.assert_allclose(sin.f(x, p), 0.9 * np.sin(1.1 * x + 0.1 * np.pi) + 0.01)
    np.testing.assert_allclose(sin.h(x, p), x ** 2)


def test_jacobians_match_finite_differences():
    p = sin.PARAMS["full"]
    model = sin.ekf_model(p, np.eye(2), np.eye(2))
    x, eps = np.array([0.3, -0.2]), 1e-6
    for fun, jac in ((model.f, model.F_jacobian), (model.h, model.H_jacobian)):
        num = np.column_stack([(fun(x + eps * e) - fun(x - eps * e)) / (2 * eps) for e in np.eye(2)])
        np.testing.assert_allclose(jac(x), num, atol=1e-6)


def test_ekf_full_model_matches_paper_table_iii():
    # Paper Table III, EKF at 1/r^2 = 0 dB: -19.58 dB. Allow 0.5 dB for a different noise draw.
    arrays, _ = common.load_dataset(sin.generate(0.0))
    x_hat = sin.run_ekf(arrays, sin.PARAMS["full"])
    assert abs(common.mse_db(x_hat, arrays["x_test"]) - (-19.58)) < 0.5