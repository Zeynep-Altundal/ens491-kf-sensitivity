"""One estimator run on one recording: estimates, covariance diagonal, errors, NEES and NIS."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from experiments.estimators.data import initial
from experiments.estimators import filters, vendor
from experiments.estimators.filters import Tuning

# Reported estimators: the companion filters (pinned export, adapter); own_*: cross-checks.
ESTIMATORS = {"KF": vendor.factory("KF"), "EKF": vendor.factory("EKF"), "ESKF": vendor.factory("ESKF"),
              "own_KF": filters.KF, "own_EKF": filters.EKF, "own_ESKF": filters.ESKF}
REPORTED = ("KF", "EKF", "ESKF")
from synthetic.rotation import quat_conjugate, quat_log, quat_multiply


IU_POS = np.triu_indices(3)  # P_pos: xx, xy, xz, yy, yz, zz of the position block


def position_epochs(imu_t: np.ndarray, pos_t: np.ndarray) -> np.ndarray:
    """IMU index of every position epoch (position stamps are IMU stamps)."""
    idx = np.searchsorted(imu_t, pos_t)
    idx = np.clip(idx, 0, len(imu_t) - 1)
    if not np.allclose(imu_t[idx], pos_t, atol=1e-9, rtol=0):
        raise ValueError("position time stamps are not IMU time stamps")
    return idx


def run(name: str, rec: dict, tuning: Tuning) -> dict:
    """Run estimator ``name`` over the recording; returns float64 arrays and the wall time."""
    t_wall = time.perf_counter()
    t = rec["imu_t"]
    n = len(t)
    f = ESTIMATORS[name](initial(rec), tuning)
    k_pos = position_epochs(t, rec["pos_t"])
    upd_at = {int(k): j for j, k in enumerate(k_pos)}
    out = {"t": t, "p_hat": np.empty((n, 3)), "v_hat": np.empty((n, 3)), "q_hat": np.empty((n, 4)),
           "bg_hat": np.empty((n, 3)), "ba_hat": np.empty((n, 3)), "P_diag": np.empty((n, 15)),
           "P_pos": np.empty((n, 6)), "nees": np.empty(n)}
    upd_t, nis = [], []
    e15 = np.empty(15)
    for k in range(n):
        if k > 0:
            f.predict(t[k] - t[k - 1], rec["imu_gyro"][k], rec["imu_accel"][k])
        j = upd_at.get(k)
        if j is not None and rec["pos_valid"][j]:
            _, _, s = f.update_position(rec["pos_z"][j], rec["pos_R"][j])
            upd_t.append(t[k])
            nis.append(s)
        p, v, q, bg, ba = f.state()
        out["p_hat"][k], out["v_hat"][k], out["q_hat"][k], out["bg_hat"][k], out["ba_hat"][k] = p, v, q, bg, ba
        P, mask = f.cov15()
        out["P_diag"][k] = np.where(mask, np.diag(P), np.nan)
        out["P_pos"][k] = P[IU_POS]
        e15[0:3] = rec["truth_p"][k] - p
        e15[3:6] = rec["truth_v"][k] - v
        if mask[6]:
            e15[6:9] = quat_log(quat_multiply(quat_conjugate(q), rec["truth_q"][k]))
            e15[9:12] = rec["truth_bg"][k] - bg
            e15[12:15] = rec["truth_ba"][k] - ba
        e = e15[mask]
        out["nees"][k] = float(e @ np.linalg.solve(P[np.ix_(mask, mask)], e))
    out["err_p"] = rec["truth_p"] - out["p_hat"]
    out["err_v"] = rec["truth_v"] - out["v_hat"]
    if f.uses_imu:
        out["err_att"] = quat_log(quat_multiply(quat_conjugate(out["q_hat"]), rec["truth_q"]))
    else:
        out["err_att"] = np.full((n, 3), np.nan)
    out["upd_t"], out["nis"] = np.array(upd_t), np.array(nis)
    out["nees_dim"] = f.nees_dim
    out["runtime_s"] = time.perf_counter() - t_wall
    return out


def save(path: Path, out: dict, meta: dict) -> Path:
    """Compressed ``.npz`` with float32 series (format: ``README.md``, "Results file format")."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.asarray(v, dtype=np.float32) for k, v in out.items()
              if isinstance(v, np.ndarray) and k != "t"}
    arrays["t"] = np.asarray(out["t"], dtype=np.float64)
    arrays["upd_t"] = np.asarray(out["upd_t"], dtype=np.float64)
    m = dict(meta, nees_dim=out["nees_dim"], runtime_s=out["runtime_s"])
    np.savez_compressed(path, meta=np.array(json.dumps(m, sort_keys=True, default=str)), **arrays)
    return path
