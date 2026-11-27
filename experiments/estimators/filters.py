"""Three estimators behind one interface (protocol: ``README.md``).

* ``KF``: linear constant-velocity Kalman filter on position measurements only (system-level
  baseline; it never sees the IMU).
* ``EKF``: full-state EKF, state ``[p, v, q, b_g, b_a]`` (16), quaternion in the state, the same
  nominal propagation as the ESKF, analytic Jacobian of that discrete step, additive update and
  normalisation of the quaternion with projection of the covariance onto the tangent space.
* ``ESKF``: ``experiments/eskf_sensitivity/eskf.py`` (unchanged) through an adapter.

Interface (``Estimator``): ``predict(dt, gyro, accel)`` to the next IMU epoch,
``update_position(z, R) -> (innovation, S, nis)``, ``state() -> (p, v, q, b_g, b_a)`` (NaN where not
estimated), ``cov15() -> (P, mask)`` with ``P`` 15 x 15 in the order ``p, v, dtheta, b_g, b_a`` and
``mask`` the estimated components.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.eskf_sensitivity import eskf as es
from synthetic.imu_model import GRAVITY_W
from synthetic.rotation import quat_exp, quat_multiply, quat_normalize, quat_to_rotmat

NAN3 = np.full(3, np.nan)
NAN4 = np.full(4, np.nan)


@dataclass(frozen=True)
class Tuning:
    gyro_noise_density: float
    accel_noise_density: float
    gyro_bias_rw: float
    accel_bias_rw: float
    std_p: float
    std_v: float
    std_att: float
    std_bg: float
    std_ba: float
    q_cv: float  # KF: white acceleration PSD per axis [m^2/s^3]

    def imu_noise(self) -> es.ImuNoise:
        return es.ImuNoise(self.gyro_noise_density, self.accel_noise_density, self.gyro_bias_rw,
                           self.accel_bias_rw)

    def p0_15(self) -> np.ndarray:
        return np.diag(np.r_[np.full(3, self.std_p), np.full(3, self.std_v), np.full(3, self.std_att),
                             np.full(3, self.std_bg), np.full(3, self.std_ba)] ** 2)


@dataclass(frozen=True)
class Initial:
    p: np.ndarray
    v: np.ndarray
    q: np.ndarray
    gyro: np.ndarray  # IMU sample at t0 (the IMU-driven filters integrate from it)
    accel: np.ndarray


def _joseph(P, H, R, nu):
    S = H @ P @ H.T + R
    K = np.linalg.solve(S, H @ P).T
    A = np.eye(P.shape[0]) - K @ H
    P_new = A @ P @ A.T + K @ R @ K.T
    return K @ nu, 0.5 * (P_new + P_new.T), S


def _nis(nu, S) -> float:
    return float(nu @ np.linalg.solve(S, nu))


# ---------------------------------------------------------------- quaternion algebra (Hamilton, [w, x, y, z])

def xi(q: np.ndarray) -> np.ndarray:
    """4 x 3 matrix with ``q (x) [0, a] = xi(q) a``."""
    w, x, y, z = q
    return np.array([[-x, -y, -z], [w, -z, y], [z, w, -x], [-y, x, w]])


def left_matrix(q: np.ndarray) -> np.ndarray:
    """``q (x) e = left_matrix(q) e``."""
    w, v = q[0], q[1:]
    m = np.empty((4, 4))
    m[0, 0], m[0, 1:], m[1:, 0] = w, -v, v
    m[1:, 1:] = w * np.eye(3) + es.skew(v)
    return m


def right_matrix(e: np.ndarray) -> np.ndarray:
    """``q (x) e = right_matrix(e) q``."""
    w, v = e[0], e[1:]
    m = np.empty((4, 4))
    m[0, 0], m[0, 1:], m[1:, 0] = w, -v, v
    m[1:, 1:] = w * np.eye(3) - es.skew(v)
    return m


def tangent_projection(q: np.ndarray) -> np.ndarray:
    """Jacobian of ``q -> q / |q|`` at a unit quaternion: ``I - q q^T``."""
    return np.eye(4) - np.outer(q, q)


def d_rot_times(q: np.ndarray, f: np.ndarray) -> np.ndarray:
    """3 x 4 Jacobian of ``R(q / |q|) f`` with respect to ``q`` at a unit quaternion."""
    w, v = q[0], q[1:]
    d = np.empty((3, 4))
    d[:, 0] = 2.0 * (w * f + np.cross(v, f))
    d[:, 1:] = 2.0 * (np.dot(v, f) * np.eye(3) + np.outer(v, f) - np.outer(f, v) - w * es.skew(f))
    return d @ tangent_projection(q)


# ---------------------------------------------------------------- KF

class KF:
    """Constant-velocity KF on position only (system-level baseline, no IMU)."""

    name, uses_imu, nees_dim = "KF", False, 6

    def __init__(self, init: Initial, tuning: Tuning):
        self.x = np.r_[init.p, init.v].astype(float)
        self.P = np.diag(np.r_[np.full(3, tuning.std_p ** 2), np.full(3, tuning.std_v ** 2)])
        self.q = tuning.q_cv

    @staticmethod
    def transition(dt: float, q: float) -> tuple[np.ndarray, np.ndarray]:
        i3 = np.eye(3)
        F = np.block([[i3, dt * i3], [np.zeros((3, 3)), i3]])
        Q = q * np.block([[dt ** 3 / 3 * i3, dt ** 2 / 2 * i3], [dt ** 2 / 2 * i3, dt * i3]])
        return F, Q

    def predict(self, dt, gyro=None, accel=None) -> None:
        F, Q = self.transition(dt, self.q)
        self.x = F @ self.x
        P = F @ self.P @ F.T + Q
        self.P = 0.5 * (P + P.T)

    def update_position(self, z, R):
        H = np.hstack([np.eye(3), np.zeros((3, 3))])
        nu = np.asarray(z, float) - self.x[:3]
        dx, self.P, S = _joseph(self.P, H, R, nu)
        self.x = self.x + dx
        return nu, S, _nis(nu, S)

    def state(self):
        return self.x[:3].copy(), self.x[3:].copy(), NAN4, NAN3, NAN3

    def cov15(self):
        P = np.full((15, 15), np.nan)
        P[:6, :6] = self.P
        mask = np.zeros(15, bool)
        mask[:6] = True
        return P, mask


# ---------------------------------------------------------------- ESKF (adapter)

class ESKF:
    """Adapter to ``experiments.eskf_sensitivity.eskf.ErrorStateKalmanFilter``."""

    name, uses_imu, nees_dim = "ESKF", True, 15

    def __init__(self, init: Initial, tuning: Tuning):
        x0 = es.NominalState(p=np.asarray(init.p, float), v=np.asarray(init.v, float),
                             q=quat_normalize(np.asarray(init.q, float)), bg=np.zeros(3), ba=np.zeros(3))
        self.f = es.ErrorStateKalmanFilter(x0, tuning.p0_15(), tuning.imu_noise(), init.gyro, init.accel)

    def predict(self, dt, gyro, accel) -> None:
        self.f.propagate(gyro, accel, dt)

    def update_position(self, z, R):
        r = self.f.update_position(z, R)
        return r.innovation, r.innovation_cov, r.nis

    def state(self):
        x = self.f.state
        return x.p.copy(), x.v.copy(), x.q.copy(), x.bg.copy(), x.ba.copy()

    def cov15(self):
        return self.f.P.copy(), np.ones(15, bool)


# ---------------------------------------------------------------- EKF (16 states)

IP, IV, IQ, IBG, IBA = slice(0, 3), slice(3, 6), slice(6, 10), slice(10, 13), slice(13, 16)


def ekf_step(x: np.ndarray, g0, a0, g1, a1, dt: float) -> np.ndarray:
    """Discrete propagation of the 16-state vector: ``eskf.nominal_step`` (same scheme)."""
    n = es.nominal_step(es.NominalState(p=x[IP], v=x[IV], q=x[IQ], bg=x[IBG], ba=x[IBA]), g0, a0, g1, a1, dt)
    return np.r_[n.p, n.v, n.q, n.bg, n.ba]


def ekf_jacobian(x: np.ndarray, g0, a0, g1, a1, dt: float) -> np.ndarray:
    """16 x 16 Jacobian of ``ekf_step`` (quaternion columns valid for tangent perturbations)."""
    q, bg, ba = x[IQ], x[IBG], x[IBA]
    w0, w1 = g0 - bg, g1 - bg
    phi = 0.5 * dt * (w0 + w1) + dt ** 2 / 12.0 * np.cross(w0, w1)
    e = quat_exp(phi)
    q1 = quat_normalize(quat_multiply(q, e))
    proj1 = tangent_projection(q1)
    dq1_dq = proj1 @ right_matrix(e) @ tangent_projection(q)
    dexp = np.vstack([-0.25 * phi[None, :], 0.5 * np.eye(3)])  # d Exp(phi) / d phi, first order
    dq1_dphi = proj1 @ left_matrix(q) @ dexp
    dphi_dbg = -dt * np.eye(3) + dt ** 2 / 12.0 * (es.skew(w1) - es.skew(w0))
    dq1_dbg = dq1_dphi @ dphi_dbg
    f0, f1 = a0 - ba, a1 - ba
    r0, r1 = quat_to_rotmat(q), quat_to_rotmat(q1)
    da0_dq = d_rot_times(q, f0)
    da1_dq = d_rot_times(q1, f1) @ dq1_dq
    da1_dbg = d_rot_times(q1, f1) @ dq1_dbg
    F = np.eye(16)
    i3 = np.eye(3)
    F[IP, IV] = dt * i3
    F[IP, IQ] = dt ** 2 * (da0_dq / 3.0 + da1_dq / 6.0)
    F[IP, IBG] = dt ** 2 * da1_dbg / 6.0
    F[IP, IBA] = -dt ** 2 * (r0 / 3.0 + r1 / 6.0)
    F[IV, IQ] = 0.5 * dt * (da0_dq + da1_dq)
    F[IV, IBG] = 0.5 * dt * da1_dbg
    F[IV, IBA] = -0.5 * dt * (r0 + r1)
    F[IQ, IQ] = dq1_dq
    F[IQ, IBG] = dq1_dbg
    return F


def att_to_quat_map(q: np.ndarray) -> np.ndarray:
    """16 x 15 map from the 15-dim error ``[dp, dv, dtheta, db_g, db_a]`` to the 16-state
    perturbation (``dq = xi(q) dtheta / 2``, right attitude error)."""
    T = np.zeros((16, 15))
    T[0:6, 0:6] = np.eye(6)
    T[IQ, 6:9] = 0.5 * xi(q)
    T[10:16, 9:15] = np.eye(6)
    return T


def quat_to_att_map(q: np.ndarray) -> np.ndarray:
    """15 x 16 map back (``dtheta = 2 xi(q)^T dq``); the left inverse of ``att_to_quat_map``."""
    A = np.zeros((15, 16))
    A[0:6, 0:6] = np.eye(6)
    A[6:9, IQ] = 2.0 * xi(q).T
    A[9:15, 10:16] = np.eye(6)
    return A


class EKF:
    """Full-state EKF, 16 states, global attitude error handled by normalisation."""

    name, uses_imu, nees_dim = "EKF", True, 15

    def __init__(self, init: Initial, tuning: Tuning):
        q = quat_normalize(np.asarray(init.q, float))
        self.x = np.r_[init.p, init.v, q, np.zeros(3), np.zeros(3)].astype(float)
        T = att_to_quat_map(q)
        self.P = T @ tuning.p0_15() @ T.T
        self.t = tuning
        self.g, self.a = np.asarray(init.gyro, float), np.asarray(init.accel, float)

    def process_noise(self, dt: float) -> np.ndarray:
        t = self.t
        Q = np.zeros((16, 16))
        Q[IV, IV] = t.accel_noise_density ** 2 * dt * np.eye(3)
        G = 0.5 * xi(self.x[IQ])
        Q[IQ, IQ] = t.gyro_noise_density ** 2 * dt * G @ G.T
        Q[IBG, IBG] = t.gyro_bias_rw ** 2 * dt * np.eye(3)
        Q[IBA, IBA] = t.accel_bias_rw ** 2 * dt * np.eye(3)
        return Q

    def predict(self, dt, gyro, accel) -> None:
        gyro, accel = np.asarray(gyro, float), np.asarray(accel, float)
        F = ekf_jacobian(self.x, self.g, self.a, gyro, accel, dt)
        self.x = ekf_step(self.x, self.g, self.a, gyro, accel, dt)
        P = F @ self.P @ F.T + self.process_noise(dt)
        self.P = 0.5 * (P + P.T)
        self.g, self.a = gyro, accel

    def update_position(self, z, R):
        H = np.zeros((3, 16))
        H[:, IP] = np.eye(3)
        nu = np.asarray(z, float) - self.x[IP]
        dx, P, S = _joseph(self.P, H, R, nu)
        x = self.x + dx
        n = np.linalg.norm(x[IQ])
        x[IQ] = x[IQ] / n
        J = np.eye(16)
        J[IQ, IQ] = tangent_projection(x[IQ]) / n  # Jacobian of the normalisation
        P = J @ P @ J.T
        self.x, self.P = x, 0.5 * (P + P.T)
        return nu, S, _nis(nu, S)

    def state(self):
        x = self.x
        return x[IP].copy(), x[IV].copy(), x[IQ].copy(), x[IBG].copy(), x[IBA].copy()

    def cov15(self):
        A = quat_to_att_map(self.x[IQ])
        return A @ self.P @ A.T, np.ones(15, bool)


ESTIMATORS = {"KF": KF, "EKF": EKF, "ESKF": ESKF}
