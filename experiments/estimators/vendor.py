"""Thin adapter: the filters of the companion repository behind the ``Estimator`` interface.

Source: ``ens491-nkf``, ``nkf/compare/filters.py`` and its imports, exported read-only with
``git archive`` from the pinned commit ``VENDOR_COMMIT`` into ``_vendor/nkf_<commit>/`` (git-ignored;
``python -m experiments.estimators.vendor --export`` recreates it). The exported files are used
unchanged; nothing here modifies their behaviour. The adapter only

* converts our tuning and initial state to their ``Init`` and ``NoiseSpec``,
* calls ``predict(gyro_k, accel_k, dt)`` with the IMU sample at the start of the interval, as their
  runner does (``nkf/compare/runner.py``: ``f.predict(gyro[k], accel[k], t[k + 1] - t[k])``),
* reads the full covariance ``P`` of the filter object for the NEES (their ``cov_blocks`` returns
  only the diagonal blocks); for the 16-state EKF it maps the quaternion block with their own
  ``EKF._J_q_to_att``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np

from experiments.estimators.filters import Initial, Tuning

VENDOR_COMMIT = "bb29c92d30788bfdf06c205e4a85b73ac47e8765"
VENDOR_BRANCH = "origin/feat/W03-synthetic-scenarios"
VENDOR_REPO = Path.home() / "ens491-nkf"
VENDOR_FILES = ("nkf/__init__.py", "nkf/compare/__init__.py", "nkf/compare/filters.py",
                "nkf/compare/kinematics.py", "nkf/filters/__init__.py", "nkf/filters/conventions.py")
VENDOR_SHA256 = {
    "nkf/__init__.py": "43c1aa7b155d6ce7df494d5457807a9d1ae4228bb6dd4d2a627942821cb5a750",
    "nkf/compare/__init__.py": "ae4f99bb9cb5c773e0c0752b476a3bb7435ad18420777e4160d25dadc2331a3c",
    "nkf/compare/filters.py": "7ef5445a7b687cde8589b0ae0a203c8a3f6e939d613c086988efb3e5f64dde30",
    "nkf/compare/kinematics.py": "42cc9aae078ded35940186e34335ae0de5098e8039d5423c8c6c6026a537e598",
    "nkf/filters/__init__.py": "ec7a463f0efc61f5b8dca849601b14b56af5df391a70b25402e834b6aa20f03b",
    "nkf/filters/conventions.py": "7e81dfe37342e1aeb0901828f0feb4ad4bf3a5a6239fc8a69284441559871223",
}
VENDOR_DIR = Path(__file__).resolve().parent / "_vendor" / f"nkf_{VENDOR_COMMIT}"
NAMES = {"KF": "KF-CV", "EKF": "EKF", "ESKF": "ESKF"}  # our name -> their class name


def export() -> Path:
    """Export the pinned files read-only (``git archive``; the companion repository is not changed)."""
    VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    tar = subprocess.run(["git", "-C", str(VENDOR_REPO), "archive", "--format=tar", VENDOR_COMMIT,
                          *VENDOR_FILES], check=True, capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", str(VENDOR_DIR)], input=tar, check=True)
    verify()
    return VENDOR_DIR


def verify() -> None:
    for rel, digest in VENDOR_SHA256.items():
        got = hashlib.sha256((VENDOR_DIR / rel).read_bytes()).hexdigest()
        if got != digest:
            raise RuntimeError(f"vendored file {rel} differs from commit {VENDOR_COMMIT[:12]}")


def module():
    """``nkf.compare.filters`` of the pinned export (checked against the recorded hashes)."""
    verify()
    if str(VENDOR_DIR) not in sys.path:
        sys.path.insert(0, str(VENDOR_DIR))
    mod = importlib.import_module("nkf.compare.filters")
    if Path(mod.__file__).resolve().parents[2] != VENDOR_DIR:
        raise RuntimeError(f"nkf imported from {mod.__file__}, not from the pinned export")
    return mod


class VendorEstimator:
    """One filter of the companion repository behind our interface."""

    def __init__(self, ours: str, init: Initial, tuning: Tuning):
        vf = module()
        self.vf = vf
        self.name = ours
        cls = vf.FILTERS[NAMES[ours]]
        z = np.zeros(3)
        self.f = cls(vf.Init(np.asarray(init.p, float), np.asarray(init.v, float), np.asarray(init.q, float),
                             z, z.copy(), tuning.std_p, tuning.std_v, tuning.std_att, tuning.std_bg, tuning.std_ba),
                     vf.NoiseSpec(tuning.gyro_noise_density, tuning.accel_noise_density, tuning.gyro_bias_rw,
                                  tuning.accel_bias_rw, sigma_cv=float(np.sqrt(tuning.q_cv))))
        self.uses_imu = bool(cls.uses_imu)
        self.nees_dim = 15 if self.uses_imu else 6
        self.g, self.a = np.asarray(init.gyro, float), np.asarray(init.accel, float)

    def predict(self, dt, gyro, accel) -> None:
        self.f.predict(self.g, self.a, dt)  # sample at the start of the interval, held
        self.g, self.a = np.asarray(gyro, float), np.asarray(accel, float)

    def update_position(self, z, R):
        info = self.f.update_position(np.asarray(z, float), np.asarray(R, float))
        return info.innov, info.S, info.nis

    def state(self):
        return tuple(np.array(x, dtype=float) for x in self.f.state())

    def cov15(self):
        P = np.asarray(self.f.P, float)
        if not self.uses_imu:
            full = np.full((15, 15), np.nan)
            full[:6, :6] = P
            mask = np.zeros(15, bool)
            mask[:6] = True
            return full, mask
        if P.shape == (16, 16):
            A = np.zeros((15, 16))
            A[0:6, 0:6] = np.eye(6)
            A[6:9, 6:10] = self.vf.EKF._J_q_to_att(self.f.x[6:10])
            A[9:15, 10:16] = np.eye(6)
            P = A @ P @ A.T
        return P, np.ones(15, bool)


def factory(ours: str):
    return lambda init, tuning: VendorEstimator(ours, init, tuning)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="export or verify the pinned companion filters")
    ap.add_argument("--export", action="store_true")
    args = ap.parse_args()
    print(export() if args.export else (verify() or f"verified {VENDOR_DIR}"))
