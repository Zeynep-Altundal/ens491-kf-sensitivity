"""Batched rotation and IMU kinematics used by the comparison filters (NumPy, float64).

Same equations as nkf/filters/conventions.py (propagate_nominal), written for stacks of states,
so that the UKF sigma points and the finite-difference Jacobians of the EKF run in one call.
"""

from __future__ import annotations

import numpy as np

from experiments.nav_compare.filters import conventions as c

G_W = np.asarray(c.GRAVITY_W, dtype=float)


def skew_b(w):
    z = np.zeros(w.shape[:-1])
    return np.stack(
        [
            np.stack([z, -w[..., 2], w[..., 1]], -1),
            np.stack([w[..., 2], z, -w[..., 0]], -1),
            np.stack([-w[..., 1], w[..., 0], z], -1),
        ],
        -2,
    )


def exp_b(phi):
    """Rotation matrices from rotation vectors, batched (..., 3) -> (..., 3, 3)."""
    th = np.linalg.norm(phi, axis=-1)[..., None, None]
    k = skew_b(phi)
    small = th < 1e-8
    ths = np.where(small, 1.0, th)
    a = np.where(small, 1.0, np.sin(ths) / ths)
    b = np.where(small, 0.5, (1 - np.cos(ths)) / ths**2)
    return np.eye(3) + a * k + b * (k @ k)


def log_b(R):
    """Rotation vectors from rotation matrices, batched."""
    tr = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1.0, 1.0)
    th = np.arccos(tr)
    v = np.stack(
        [
            R[..., 2, 1] - R[..., 1, 2],
            R[..., 0, 2] - R[..., 2, 0],
            R[..., 1, 0] - R[..., 0, 1],
        ],
        -1,
    )
    s = np.sin(th)
    f = np.where(th < 1e-8, 0.5, th / (2 * np.where(th < 1e-8, 1.0, s)))
    return f[..., None] * v


def quat_mul_b(a, b):
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        -1,
    )


def quat_from_rotvec_b(phi):
    th = np.linalg.norm(phi, axis=-1, keepdims=True)
    half = 0.5 * th
    s = np.where(
        th < 1e-10, 0.5 - th**2 / 48.0, np.sin(half) / np.where(th < 1e-10, 1.0, th)
    )
    return np.concatenate([np.cos(half), s * phi], -1)


def quat_to_rot_b(q):
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack(
        [
            np.stack(
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1
            ),
            np.stack(
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1
            ),
            np.stack(
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1
            ),
        ],
        -2,
    )


def prop_R(p, v, R, bg, ba, gyro, accel, dt):
    """Same equations as conventions.propagate_nominal, batched, attitude as matrices."""
    w = gyro - bg
    a = (R @ (accel - ba)[..., None])[..., 0] + G_W
    return p + v * dt + 0.5 * a * dt * dt, v + a * dt, R @ exp_b(w * dt), bg, ba


def f_full(x, gyro, accel, dt):
    """Full-state propagation for the EKF, x = [p v q bg ba] (16), batched, q normalised."""
    p, v, q, bg, ba = (
        x[..., 0:3],
        x[..., 3:6],
        x[..., 6:10],
        x[..., 10:13],
        x[..., 13:16],
    )
    gy, ac = gyro, accel
    qn = q / np.linalg.norm(q, axis=-1, keepdims=True)
    R = quat_to_rot_b(qn)
    w = gy - bg
    a = (R @ (ac - ba)[..., None])[..., 0] + G_W
    q2 = quat_mul_b(qn, quat_from_rotvec_b(w * dt))
    q2 = q2 / np.linalg.norm(q2, axis=-1, keepdims=True)
    return np.concatenate([p + v * dt + 0.5 * a * dt * dt, v + a * dt, q2, bg, ba], -1)
