"""Tests for the Lorenz export and the batched EKF.

Run from the repository root with ``python -m pytest -v experiments/knet_data``.
"""

import numpy as np
import pytest

from experiments.filter_family import lorenz as zl
from experiments.filter_family.filters import EKF, run_filter
from experiments.knet_data import common, lorenz


@pytest.fixture(autouse=True)
def small_and_tmp(tmp_path, monkeypatch):
    # Temporary folder and small split sizes, so the tests run in a few seconds.
    monkeypatch.setattr(lorenz, "DATA_DIR", tmp_path)
    monkeypatch.setattr(lorenz, "N_TRAIN", 20)
    monkeypatch.setattr(lorenz, "N_VAL", 10)
    monkeypatch.setattr(lorenz, "N_TEST", 20)
    monkeypatch.setitem(lorenz.SCENARIOS, "full", (300, (10.0,), 0.0))


def test_rotation_is_orthonormal_and_small():
    Rm = lorenz.rotation_xyz(1.0)
    np.testing.assert_allclose(Rm @ Rm.T, np.eye(3), atol=1e-12)
    assert np.abs(Rm - np.eye(3)).max() < 0.02


def test_shapes_and_meta():
    arrays, meta = common.load_dataset(lorenz.generate("full", 10.0))
    assert arrays["x_train"].shape == (20, lorenz.T_TRAIN, 3)
    assert arrays["y_test"].shape == (20, 300, 3)
    assert meta["scenario"] == "full" and meta["J_data"] == zl.J_DATA


def test_data_follow_the_filter_family_model():
    # Without noise, our simulation must reproduce the filter-family f exactly.
    x, y = lorenz.simulate(np.eye(3), 0.0, 0.0, 2, 50, np.random.default_rng(0))
    f = zl.make_f(zl.J_DATA)
    np.testing.assert_allclose(x[:, 0], f(np.broadcast_to(zl.X0, (2, 3))))
    np.testing.assert_allclose(x[:, 1:], f(x[:, :-1]))
    np.testing.assert_allclose(y, x)


def test_batched_ekf_equals_filter_family_ekf():
    x, y = zl.simulate(0.1, 3, 200, np.random.default_rng(1))
    m = zl.lorenz_model(0.1, zl.J_DATA, "exact")
    ref = np.stack([run_filter(EKF(m, m.x0, m.P0), yy)["x"] for yy in y])
    f = zl.make_f(zl.J_DATA)
    H_b = np.broadcast_to(np.eye(3), (3, 3, 3))
    ours, _ = common.batched_ekf(y, f, lorenz.jacobian(f), lambda z: z, lambda z: H_b, m.Q, m.R, m.x0, m.P0)
    # Both use finite-difference Jacobians, so they agree up to rounding (1e-9 m here).
    np.testing.assert_allclose(ours, ref, atol=1e-6)


def test_ekf_full_model_close_to_paper():
    # Paper Table V, EKF at 1/r^2 = 10 dB: -20.37 dB. Small sample, so allow 1 dB.
    lorenz.generate("full", 10.0)
    row = next(r for r in lorenz.check("full", 10.0) if r["method"] == "EKF J=5")
    assert abs(row["mse_db"] - (-20.37)) < 1.0 and row["diverged"] == 0