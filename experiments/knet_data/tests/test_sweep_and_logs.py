"""Tests for the sweep expansion and the filter-log export (need PyTorch, not the reference code).

Run from the repository root with ``python -m pytest -v experiments/knet_data``.
"""

import json

import numpy as np
import pytest

pytest.importorskip("torch")

from experiments.knet_data import common, export_logs, kalmannet_run, linear, sweep  # noqa: E402


@pytest.fixture
def small_linear(tmp_path, monkeypatch):
    # Two small linear datasets in a temporary data folder; outputs and logs also temporary.
    for mod in (linear, sweep, export_logs):
        monkeypatch.setattr(mod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(kalmannet_run, "OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(export_logs, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(linear, "N_TRAIN", 20)
    monkeypatch.setattr(linear, "N_VAL", 10)
    monkeypatch.setattr(linear, "N_TEST", 50)
    linear.generate("full")
    linear.generate("rotH10")
    return tmp_path


def test_sweep_expands_patterns_and_seeds(small_linear):
    jobs = sweep.expand({"seeds": [0, 1], "runs": [{"data": "linear_*"}, {"data": "nothing_*"}]})
    assert sorted((j["data"], j["seed"]) for j in jobs) == [
        ("linear_full", 0), ("linear_full", 1), ("linear_rotH10", 0), ("linear_rotH10", 1)]
    assert all(j["given"] == "model" for j in jobs)


def test_export_logs_linear(small_linear):
    paths = export_logs.export("linear_full")
    assert sorted(p.stem for p in paths) == ["EKF", "ESKF", "KF"]   # no KalmanNet run here
    with np.load(paths[0]) as f:
        meta = json.loads(str(f["meta"]))
        assert f["x_hat"].shape == (50, 100, 2) and f["P"].shape == (50, 100, 2, 2)
        x = common.load_dataset(small_linear / "linear_full.npz")[0]["x_test"]
        np.testing.assert_allclose(f["err"], x - f["x_hat"], atol=1e-4)   # truth - estimate
        # Correct model: the average NEES should be close to the state dimension (2).
        assert 1.5 < float(np.mean(f["nees"])) < 2.5
    assert meta["source"] == "kalmannet_linear" and meta["error_convention"] == "truth - estimate"


def test_kalmannet_log_has_nan_covariance(small_linear):
    # Fake a finished KalmanNet run: the log must carry innovations but NaN covariances.
    arrays, _ = common.load_dataset(small_linear / "linear_rotH10.npz")
    out = kalmannet_run.run_dir("linear_rotH10", "model", 0)
    out.mkdir(parents=True)
    np.savez(out / "results.npz", x_hat=arrays["x_test"] + 0.1, meta=json.dumps({"steps": 1}))
    paths = export_logs.export("linear_rotH10")
    knet = [p for p in paths if p.stem == "KalmanNet"][0]
    with np.load(knet) as f:
        assert np.isnan(f["P"]).all() and np.isnan(f["nees"]).all() and np.isfinite(f["innov"]).all()
