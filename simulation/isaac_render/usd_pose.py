"""Conversion of the simulator pose (ENU world, FLU body, Hamilton quaternion) to USD.

Conventions of the simulator (``../rviz_drone/``):

* World frame ENU: x east, y north, z up, metres.
* Body frame FLU: x forward, y left, z up.
* Attitude: Hamilton quaternion ``[w, x, y, z]``, body-to-world, i.e. ``v_w = R(q) v_b``.

Conventions of the USD stage built by ``render_flight.py``:

* ``upAxis = Z`` and ``metersPerUnit = 1``, so the stage axes are the ENU axes without change:
  ``p_usd = p_enu``.
* A prim with the ops ``xformOp:translate`` then ``xformOp:orient`` (``GfQuatd``, also Hamilton,
  real part first) maps local points to its parent. USD uses row vectors:
  ``[p_parent, 1] = [p_local, 1] M`` with ``M = [[R^T, 0], [t, 1]]``.
* The quadrotor prim is the body frame: its orient op is the simulator quaternion itself,
  normalised; no axis permutation and no conjugation is needed. The fixed rotation of the drone
  asset relative to the body frame is not part of this conversion (``asset_alignment.py``).

The functions here use numpy only, so that they are tested without Isaac Sim.
"""

from __future__ import annotations

import numpy as np


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if not np.isfinite(n) or n < 1e-12:
        raise ValueError("quaternion with zero or non-finite norm")
    return q / n


def rotmat(q: np.ndarray) -> np.ndarray:
    """Column-vector rotation matrix of a Hamilton quaternion [w, x, y, z] (body to world)."""
    w, x, y, z = quat_normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def enu_flu_to_usd(p_enu: np.ndarray, q_wb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Translate and orient values for the body prim: (t [3], q [w, x, y, z])."""
    return np.asarray(p_enu, dtype=float).copy(), quat_normalize(q_wb)


def usd_matrix(t: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Row-vector 4x4 local-to-parent matrix of the ops translate then orient."""
    m = np.eye(4)
    m[:3, :3] = rotmat(q).T
    m[3, :3] = t
    return m
