"""Tests for the linear KalmanNet benchmark data.

Run from the repository root with ``python -m pytest -v experiments/knet_data``.
"""

import numpy as np
import pytest

from experiments.knet_data import common, linear


@pytest.fixture(autouse=True)
def tmp_data_dir(tmp_path, monkeypatch):
    # Write datasets to a temporary folder instead of ~/data/knet_benchmarks.
    monkeypatch.setattr(linear, "DATA_DIR", tmp_path)


def test_shapes_and_keys():
    arrays, meta = common.load_dataset(linear.generate("full"))
    assert arrays["x_train"].shape == (common.N_TRAIN, 100, 2)
    assert arrays["y_val"].shape == (common.N_VAL, 100, 2)
    assert arrays["x_test"].shape == (common.N_TEST, 100, 2)
    assert meta["scenario"] == "full" and meta["T"] == 100


def test_same_seed_same_data():
    a, _ = common.load_dataset(linear.generate("rotF10", seed=3))
    b, _ = common.load_dataset(linear.generate("rotF10", seed=3))
    np.testing.assert_array_equal(a["y_test"], b["y_test"])


def test_splits_are_different():
    arrays, _ = common.load_dataset(linear.generate("full"))
    # Each split has its own random stream, so the first trajectories must differ.
    assert not np.allclose(arrays["y_train"][:10], arrays["y_test"][:10])


def test_mismatch_only_in_true_matrices():
    m = linear.scenario_matrices("rotF20")
    np.testing.assert_allclose(m["F_model"], linear.F0)
    np.testing.assert_allclose(m["F_true"], common.rotation(20.0) @ linear.F0)
    np.testing.assert_allclose(m["H_true"], np.eye(2))


@pytest.mark.parametrize("name", list(linear.SCENARIOS))
def test_true_kf_matches_its_own_prediction(name):
    # The KF with the true model is optimal here, so its empirical MSE must agree with the MSE
    # predicted by its covariance (within the bootstrap interval, widened by 0.2 dB).
    linear.generate(name)
    row = next(r for r in linear.check(name) if r["method"] == "KF true model")
    assert row["ci_lo_db"] - 0.2 <= row["predicted_db"] <= row["ci_hi_db"] + 0.2


def test_full_model_has_no_mismatch():
    linear.generate("full")
    rows = {r["method"]: r["mse_db"] for r in linear.check("full")}
    assert rows["KF true model"] == rows["KF given model"]
