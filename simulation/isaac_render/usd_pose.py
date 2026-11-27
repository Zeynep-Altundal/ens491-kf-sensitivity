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
* The quadrotor prim is the body frame. With the asset aligned to FLU (``ASSET_TO_BODY`` identity)
  the orient op is the simulator quaternion itself, normalised; no axis permutation and no
  conjugation is needed.

The functions here use numpy only, so that they are tested without Isaac Sim.
"""

from __future__ import annotations

import numpy as np

# Rotation from the asset frame to the FLU body frame, Hamilton [w, x, y, z]. Identity: the
# forward axis of the asset is +x and its up axis +z (checked in render_flight.py at run time).
ASSET_TO_BODY = np.array([1.0, 0.0, 0.0, 0.0])


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if not np.isfinite(n) or n < 1e-12:
        raise ValueError("quaternion with zero or non-finite norm")
    return q / n


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a * b, both [w, x, y, z]."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def rotmat(q: np.ndarray) -> np.ndarray:
    """Column-vector rotation matrix of a Hamilton quaternion [w, x, y, z] (body to world)."""
    w, x, y, z = quat_normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def enu_flu_to_usd(p_enu: np.ndarray, q_wb: np.ndarray,
                   asset_to_body: np.ndarray = ASSET_TO_BODY) -> tuple[np.ndarray, np.ndarray]:
    """Translate and orient values for the quadrotor prim: (t [3], q [w, x, y, z])."""
    t = np.asarray(p_enu, dtype=float).copy()
    q = quat_normalize(quat_multiply(quat_normalize(q_wb), quat_normalize(asset_to_body)))
    return t, q


def usd_matrix(t: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Row-vector 4x4 local-to-parent matrix of the ops translate then orient."""
    m = np.eye(4)
    m[:3, :3] = rotmat(q).T
    m[3, :3] = t
    return m
