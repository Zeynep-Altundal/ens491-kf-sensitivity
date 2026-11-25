"""Tests for the NCLT dataset preparation.

The unit tests use small synthetic arrays and do not need the NCLT files. The last test runs the
full preparation only if the raw files are present in NCLT_DIR (default ~/data/nclt).
Run from the repository root with ``python -m pytest -v experiments/knet_data``.
"""

import numpy as np
import pytest

from experiments.knet_data import nclt


def test_resample_drops_nan_rows_and_interpolates():
    a = np.array([[0.0, np.nan], [1.0, 10.0], [3.0, 30.0]])
    np.testing.assert_allclose(nclt.resample(a, np.array([1.0, 2.0, 3.0]), (1,))[:, 0], [10, 20, 30])


def test_central_diff_of_constant_velocity():
    p = np.column_stack([2.0 * np.arange(10), -1.0 * np.arange(10)])
    np.testing.assert_allclose(nclt.central_diff(p), np.tile([2.0, -1.0], (10, 1)))


def test_odometry_is_rotated_into_ground_truth_frame():
    # Robot drives straight with 1 m/s along heading yaw0. In its odometry frame it drives along +x.
    yaw0 = 0.7
    t = np.arange(0.0, 50.0, 0.01)
    gt = np.column_stack([t, 5 + t * np.cos(yaw0), 3 + t * np.sin(yaw0), 0 * t, 0 * t, 0 * t, yaw0 + 0 * t])
    odom = np.column_stack([t, t, 0 * t, 0 * t, 0 * t, 0 * t, 0 * t])
    _, x, y = nclt.build_series(gt, odom)
    np.testing.assert_allclose(y, x[:, nclt.VEL], atol=1e-9)


def test_windows_do_not_overlap_and_x0_is_previous_state():
    k = 1 + 3 * nclt.T_TRAIN + nclt.N_VAL * nclt.T_VAL + nclt.T_TEST
    x = np.arange(k, dtype=float)[:, None].repeat(4, axis=1)   # state value = time index
    y = x[:, :2]
    out, n_train = nclt.split_windows(x, y)
    assert n_train == 3
    assert out["x_train"].shape == (3, nclt.T_TRAIN, 4)
    assert out["x_test"].shape == (1, nclt.T_TEST, 4)
    used = np.concatenate([out[f"x_{s}"][..., 0].ravel() for s in ("train", "val", "test")])
    assert len(np.unique(used)) == len(used)                   # no time step used twice
    np.testing.assert_allclose(out["x0_val"][:, 0], out["x_val"][:, 0, 0] - 1)


@pytest.mark.skipif(not (nclt.NCLT_DIR / f"groundtruth_{nclt.SESSION}.csv").exists(),
                    reason="NCLT files not found")
def test_real_session(tmp_path, monkeypatch):
    monkeypatch.setattr(nclt, "DATA_DIR", tmp_path)
    arrays, meta = nclt.load_dataset(nclt.generate())
    assert meta["steps_1hz"] > 5000
    assert arrays["x_test"].shape == (1, nclt.T_TEST, 4)
    assert np.isfinite(arrays["y_train"]).all() and np.isfinite(arrays["x_train"]).all()