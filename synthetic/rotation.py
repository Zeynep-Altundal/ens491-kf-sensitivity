"""Quaternion and rotation utilities.

Conventions
-----------
* Hamilton quaternions stored scalar-first as ``[w, x, y, z]``.
* A quaternion ``q_wb`` represents the body-to-world rotation ``R_wb``, i.e.
  ``v_w = R_wb @ v_b``.
* Euler angles follow the intrinsic Z-Y-X (yaw-pitch-roll) sequence:
  ``R_wb = Rz(yaw) @ Ry(pitch) @ Rx(roll)``.

All functions accept arrays with arbitrary leading dimensions.
"""

from __future__ import annotations

import numpy as np


def quat_multiply(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Hamilton product ``p ⊗ q`` for arrays of shape (..., 4)."""
    pw, px, py, pz = np.moveaxis(np.asarray(p, dtype=float), -1, 0)
    qw, qx, qy, qz = np.moveaxis(np.asarray(q, dtype=float), -1, 0)
    return np.stack(
        [
            pw * qw - px * qx - py * qy - pz * qz,
            pw * qx + px * qw + py * qz - pz * qy,
            pw * qy - px * qz + py * qw + pz * qx,
            pw * qz + px * qy - py * qx + pz * qw,
        ],
        axis=-1,
    )


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Return the unit quaternion(s) in the direction of ``q``."""
    q = np.asarray(q, dtype=float)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Rotation matrix (..., 3, 3) of unit quaternion(s) (..., 4)."""
    w, x, y, z = np.moveaxis(quat_normalize(q), -1, 0)
    r = np.empty(np.shape(w) + (3, 3))
    r[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    r[..., 0, 1] = 2.0 * (x * y - w * z)
    r[..., 0, 2] = 2.0 * (x * z + w * y)
    r[..., 1, 0] = 2.0 * (x * y + w * z)
    r[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    r[..., 1, 2] = 2.0 * (y * z - w * x)
    r[..., 2, 0] = 2.0 * (x * z - w * y)
    r[..., 2, 1] = 2.0 * (y * z + w * x)
    r[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return r


def quat_exp(rotvec: np.ndarray) -> np.ndarray:
    """Unit quaternion of a rotation vector (axis * angle), shape (..., 3) -> (..., 4)."""
    rotvec = np.asarray(rotvec, dtype=float)
    angle = np.linalg.norm(rotvec, axis=-1, keepdims=True)
    small = angle < 1e-8
    safe = np.where(small, 1.0, angle)
    # sin(angle/2)/angle, with its Taylor expansion near zero.
    scale = np.where(small, 0.5 - angle**2 / 48.0, np.sin(0.5 * angle) / safe)
    return np.concatenate([np.cos(0.5 * angle), scale * rotvec], axis=-1)


def euler_zyx_to_quat(roll, pitch, yaw) -> np.ndarray:
    """Quaternion ``q_wb = qz(yaw) ⊗ qy(pitch) ⊗ qx(roll)``."""
    roll, pitch, yaw = np.broadcast_arrays(
        np.asarray(roll, float), np.asarray(pitch, float), np.asarray(yaw, float)
    )
    zeros = np.zeros_like(roll)
    qx = np.stack([np.cos(roll / 2), np.sin(roll / 2), zeros, zeros], axis=-1)
    qy = np.stack([np.cos(pitch / 2), zeros, np.sin(pitch / 2), zeros], axis=-1)
    qz = np.stack([np.cos(yaw / 2), zeros, zeros, np.sin(yaw / 2)], axis=-1)
    return quat_multiply(quat_multiply(qz, qy), qx)


def euler_zyx_rates_to_body_rate(roll, pitch, roll_rate, pitch_rate, yaw_rate) -> np.ndarray:
    """Body angular rate ``w_b`` from Z-Y-X Euler angles and their time derivatives.

    Follows from ``[w_b]_x = R_wb^T dR_wb/dt`` with ``R_wb = Rz Ry Rx``.
    """
    sr, cr = np.sin(roll), np.cos(roll)
    sp, cp = np.sin(pitch), np.cos(pitch)
    return np.stack(
        [
            roll_rate - yaw_rate * sp,
            pitch_rate * cr + yaw_rate * sr * cp,
            -pitch_rate * sr + yaw_rate * cr * cp,
        ],
        axis=-1,
    )
