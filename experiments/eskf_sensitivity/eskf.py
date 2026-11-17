"""Error-state Kalman filter (ESKF) for IMU propagation with position updates.

Conventions follow ``synthetic/README.md``: ENU world frame, gravity
``g_w = [0, 0, -9.81]``, Hamilton quaternion ``q`` (scalar first) for the
body-to-world rotation ``R = R(q)``, gyro measuring ``w_b`` and accelerometer
measuring ``f_b = R^T (a_w - g_w)``.

Nominal state ``x = (p, v, q, b_g, b_a)``. Error state (15)
``dx = [dp, dv, dtheta, db_g, db_a]`` with

    p_true  = p + dp,         v_true  = v + dv,
    q_true  = q ⊗ Exp(dtheta)  (local, body-frame attitude error),
    bg_true = b_g + db_g,      ba_true = b_a + db_a.

The equations are documented in ``experiments/eskf_sensitivity/README.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

from synthetic.imu_model import GRAVITY_W
from synthetic.rotation import (
    quat_conjugate,
    quat_exp,
    quat_log,
    quat_multiply,
    quat_normalize,
    quat_to_rotmat,
)

ERROR_DIM = 15
IDX_P = slice(0, 3)
IDX_V = slice(3, 6)
IDX_TH = slice(6, 9)
IDX_BG = slice(9, 12)
IDX_BA = slice(12, 15)
H_POS = np.hstack([np.eye(3), np.zeros((3, 12))])


def skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix with ``skew(a) @ b == cross(a, b)``."""
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


@dataclass(frozen=True)
class NominalState:
    p: np.ndarray  # (3,) position, world [m]
    v: np.ndarray  # (3,) velocity, world [m/s]
    q: np.ndarray  # (4,) body-to-world quaternion [w, x, y, z]
    bg: np.ndarray  # (3,) gyro bias [rad/s]
    ba: np.ndarray  # (3,) accelerometer bias [m/s^2]


@dataclass(frozen=True)
class ImuNoise:
    """Continuous-time IMU noise parameters used to build the process noise."""

    gyro_noise_density: float  # [rad/s/sqrt(Hz)]
    accel_noise_density: float  # [m/s^2/sqrt(Hz)]
    gyro_bias_rw: float  # [rad/s^2/sqrt(Hz)]
    accel_bias_rw: float  # [m/s^3/sqrt(Hz)]


def nominal_step(x: NominalState, gyro0, accel0, gyro1, accel1, dt: float) -> NominalState:
    """Propagate the nominal state over one IMU interval [t_k, t_k+1].

    Uses the bias-corrected samples at both ends of the interval: rotation
    vector with coning term for the attitude, trapezoidal velocity, and a
    position update that is exact for linearly varying world acceleration.
    Biases are constant.
    """
    w0, w1 = gyro0 - x.bg, gyro1 - x.bg
    phi = 0.5 * dt * (w0 + w1) + dt**2 / 12.0 * np.cross(w0, w1)
    q1 = quat_normalize(quat_multiply(x.q, quat_exp(phi)))
    a0 = quat_to_rotmat(x.q) @ (accel0 - x.ba) + GRAVITY_W
    a1 = quat_to_rotmat(q1) @ (accel1 - x.ba) + GRAVITY_W
    p1 = x.p + x.v * dt + dt**2 * (a0 / 3.0 + a1 / 6.0)
    v1 = x.v + 0.5 * dt * (a0 + a1)
    return NominalState(p=p1, v=v1, q=q1, bg=x.bg.copy(), ba=x.ba.copy())


def error_dynamics(x: NominalState, gyro0, accel0, gyro1, accel1) -> np.ndarray:
    """Continuous error-state matrix A (15 x 15) evaluated over the interval.

    Uses the interval means of the bias-corrected rate and specific force and
    the rotation at the start of the interval.
    """
    w = 0.5 * (gyro0 + gyro1) - x.bg
    f = 0.5 * (accel0 + accel1) - x.ba
    r = quat_to_rotmat(x.q)
    a = np.zeros((ERROR_DIM, ERROR_DIM))
    a[IDX_P, IDX_V] = np.eye(3)
    a[IDX_V, IDX_TH] = -r @ skew(f)
    a[IDX_V, IDX_BA] = -r
    a[IDX_TH, IDX_TH] = -skew(w)
    a[IDX_TH, IDX_BG] = -np.eye(3)
    return a


def error_transition(x: NominalState, gyro0, accel0, gyro1, accel1, dt: float) -> np.ndarray:
    """Discrete error-state transition ``F = I + A dt + (A dt)^2 / 2``."""
    adt = error_dynamics(x, gyro0, accel0, gyro1, accel1) * dt
    return np.eye(ERROR_DIM) + adt + 0.5 * adt @ adt


def process_noise(noise: ImuNoise, dt: float) -> np.ndarray:
    """Discrete process noise ``Q = diag(0, s_a^2 dt I, s_g^2 dt I, s_bg^2 dt I, s_ba^2 dt I)``.

    The velocity block uses ``R s_a^2 I R^T = s_a^2 I`` (isotropic noise).
    """
    q = np.zeros((ERROR_DIM, ERROR_DIM))
    q[IDX_V, IDX_V] = noise.accel_noise_density**2 * dt * np.eye(3)
    q[IDX_TH, IDX_TH] = noise.gyro_noise_density**2 * dt * np.eye(3)
    q[IDX_BG, IDX_BG] = noise.gyro_bias_rw**2 * dt * np.eye(3)
    q[IDX_BA, IDX_BA] = noise.accel_bias_rw**2 * dt * np.eye(3)
    return q


def inject(x: NominalState, dx: np.ndarray) -> NominalState:
    """Nominal state corrected by the error state ``dx``."""
    return NominalState(
        p=x.p + dx[IDX_P],
        v=x.v + dx[IDX_V],
        q=quat_normalize(quat_multiply(x.q, quat_exp(dx[IDX_TH]))),
        bg=x.bg + dx[IDX_BG],
        ba=x.ba + dx[IDX_BA],
    )


def difference(x_true: NominalState, x: NominalState) -> np.ndarray:
    """Error state ``dx`` such that ``inject(x, dx) == x_true``."""
    return np.concatenate([
        x_true.p - x.p,
        x_true.v - x.v,
        quat_log(quat_multiply(quat_conjugate(x.q), x_true.q)),
        x_true.bg - x.bg,
        x_true.ba - x.ba,
    ])


def reset_jacobian(dtheta: np.ndarray) -> np.ndarray:
    """Covariance reset Jacobian ``G = blkdiag(I6, I - [dtheta/2]x, I6)``."""
    g = np.eye(ERROR_DIM)
    g[IDX_TH, IDX_TH] = np.eye(3) - skew(0.5 * dtheta)
    return g


@dataclass(frozen=True)
class UpdateResult:
    innovation: np.ndarray  # (3,)
    innovation_cov: np.ndarray  # (3, 3)
    nis: float
    accepted: bool


class ErrorStateKalmanFilter:
    """ESKF with IMU propagation and position updates.

    Parameters
    ----------
    x0, P0
        Initial nominal state and error covariance (15 x 15).
    noise
        IMU noise parameters for the process noise.
    gyro0, accel0
        IMU sample at the initial time; ``propagate`` integrates from the
        previously stored sample to the new one.
    gate_prob
        If given, a position update is rejected when its NIS exceeds the
        ``gate_prob`` quantile of the chi-square distribution with 3 degrees
        of freedom. ``None`` (default) disables the gate.
    """

    def __init__(self, x0: NominalState, P0: np.ndarray, noise: ImuNoise,
                 gyro0: np.ndarray, accel0: np.ndarray, gate_prob: float | None = None):
        self._x = x0
        self._P = np.array(P0, dtype=float)
        self.noise = noise
        self._gyro = np.asarray(gyro0, dtype=float)
        self._accel = np.asarray(accel0, dtype=float)
        self.gate_threshold = None if gate_prob is None else float(chi2.ppf(gate_prob, df=3))

    @property
    def state(self) -> NominalState:
        return self._x

    @property
    def P(self) -> np.ndarray:
        return self._P

    def propagate(self, gyro: np.ndarray, accel: np.ndarray, dt: float) -> None:
        """Propagate to the time of the IMU sample ``(gyro, accel)``, ``dt`` after the previous one."""
        gyro = np.asarray(gyro, dtype=float)
        accel = np.asarray(accel, dtype=float)
        F = error_transition(self._x, self._gyro, self._accel, gyro, accel, dt)
        self._x = nominal_step(self._x, self._gyro, self._accel, gyro, accel, dt)
        P = F @ self._P @ F.T + process_noise(self.noise, dt)
        self._P = 0.5 * (P + P.T)
        self._gyro, self._accel = gyro, accel

    def update_position(self, z: np.ndarray, R: np.ndarray) -> UpdateResult:
        """Position update ``z = p_true + v``, ``v ~ N(0, R)``; returns innovation statistics."""
        z = np.asarray(z, dtype=float)
        R = np.asarray(R, dtype=float)
        nu = z - self._x.p
        S = self._P[IDX_P, IDX_P] + R
        nis = float(nu @ np.linalg.solve(S, nu))
        if self.gate_threshold is not None and nis > self.gate_threshold:
            return UpdateResult(nu, S, nis, accepted=False)

        K = np.linalg.solve(S, self._P[IDX_P, :]).T  # P H^T S^-1
        dx = K @ nu
        A = np.eye(ERROR_DIM) - K @ H_POS
        P = A @ self._P @ A.T + K @ R @ K.T
        self._x = inject(self._x, dx)
        G = reset_jacobian(dx[IDX_TH])
        P = G @ P @ G.T
        self._P = 0.5 * (P + P.T)
        return UpdateResult(nu, S, nis, accepted=True)
