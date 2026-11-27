"""The filters of the comparison, all behind the same interface (docs/common_interface.md).

  KF-CV   linear KF, state [p, v], constant-velocity model, position measurements only
  KF-IMU  linear KF, state [p, v, b_a]; the accelerometer is a known input and the orientation is
          GIVEN (taken from the truth), which keeps the model linear
  EKF     full-state EKF, state [p, v, q, b_g, b_a] (16), quaternion inside the state
  ESKF    error-state EKF, 15 error states, attitude error on the right
  IESKF   ESKF with an iterated (Gauss-Newton) update for nonlinear measurements
  UKF     error-state UKF: sigma points on the 15 error states, alpha = 1, beta = 2, kappa = 0
  InEKF   right-invariant EKF on SE_2(3) with biases ("imperfect" InEKF): error
          X = Exp(xi) X_hat in the world frame; its Jacobians do not depend on the estimate
          except through the biases

Measurements (each filter implements what its state allows, otherwise returns None):
  update_position(z, R)          position in w (GPS-like)
  update_body_velocity(rows, z)  rows of R^T v: NHC (lateral, vertical = 0), wheel speed (forward)
  update_yaw_rate(z, var, gz)    yaw rate from the steering model: gyro_z - b_gz
  update_heading(z, var)         heading (yaw) of the body x axis, e.g. from a magnetometer
  update_zupt(var)               zero velocity while the vehicle stands still
  update_zaru(gyro, var)         zero angular rate while standing still: gyro = b_g (calibration)

Notation: docs/notation.md (code names: P covariance, F transition, Q process noise, H measurement
matrix, R_meas measurement covariance, innov innovation, S innovation covariance, K gain).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.nav_compare.compare import kinematics as kin
from experiments.nav_compare.filters import conventions as c

NAN3 = np.full(3, np.nan)
NAN33 = np.full((3, 3), np.nan)


@dataclass
class NoiseSpec:
    gyro_density: float
    accel_density: float
    gyro_bias_rw: float
    accel_bias_rw: float
    sigma_cv: float = 1.0  # KF-CV only (tuning value)


@dataclass
class Init:
    p: np.ndarray
    v: np.ndarray
    q: np.ndarray  # [w, x, y, z]
    b_g: np.ndarray
    b_a: np.ndarray
    std_p: float
    std_v: float
    std_att: float
    std_bg: float
    std_ba: float


@dataclass
class UpdateInfo:
    innov: np.ndarray
    S: np.ndarray
    nis: float


def _joseph(P, H, R_meas, innov):
    S = H @ P @ H.T + R_meas
    K = np.linalg.solve(S, H @ P).T  # P H^T S^-1 (P and S symmetric)
    ikh = np.eye(P.shape[0]) - K @ H
    P_new = ikh @ P @ ikh.T + K @ R_meas @ K.T
    return K @ innov, 0.5 * (P_new + P_new.T), S


def _info(innov, S):
    return UpdateInfo(innov, S, float(innov @ np.linalg.solve(S, innov)))


E_X = np.array([[1.0, 0.0, 0.0]])
LAT_VERT = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


def heading_of(R):
    """Heading (yaw) of the body x axis in w, batched: atan2(R10, R00)."""
    return np.arctan2(R[..., 1, 0], R[..., 0, 0])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def left_jacobian(phi):
    th = np.linalg.norm(phi)
    K = kin.skew_b(phi)
    if th < 1e-8:
        return np.eye(3) + 0.5 * K
    return (
        np.eye(3) + (1 - np.cos(th)) / th**2 * K + (th - np.sin(th)) / th**3 * (K @ K)
    )


class KFCV:
    """Linear KF with a constant-velocity model; ignores the IMU."""

    name, uses_imu = "KF-CV", False

    def __init__(self, init: Init, noise: NoiseSpec, **_):
        self.x = np.r_[init.p, init.v].astype(float)
        self.P = np.diag(np.r_[np.full(3, init.std_p**2), np.full(3, init.std_v**2)])
        self.q = noise.sigma_cv**2

    def predict(self, gyro, accel, dt):
        F = np.eye(6)
        F[0:3, 3:6] = dt * np.eye(3)
        I3 = np.eye(3)
        Q = self.q * np.block(
            [[dt**3 / 3 * I3, dt**2 / 2 * I3], [dt**2 / 2 * I3, dt * I3]]
        )
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update_position(self, z, R_meas):
        H = np.zeros((3, 6))
        H[:, 0:3] = np.eye(3)
        innov = z - self.x[0:3]
        dx, self.P, S = _joseph(self.P, H, R_meas, innov)
        self.x = self.x + dx
        return _info(innov, S)

    def update_body_velocity(self, rows, z, R_meas):
        return None  # no orientation in the state: body-frame aiding is not possible

    def update_yaw_rate(self, z, var, gyro_z):
        return None

    def update_heading(self, z, var):
        return None

    def update_zaru(self, gyro, var):
        return None

    def update_zupt(self, var):
        H = np.zeros((3, 6))
        H[:, 3:6] = np.eye(3)
        innov = -self.x[3:6]
        dx, self.P, S = _joseph(self.P, H, var * np.eye(3), innov)
        self.x = self.x + dx
        return _info(innov, S)

    def state(self):
        return self.x[0:3], self.x[3:6], np.full(4, np.nan), NAN3, NAN3

    def cov_blocks(self):
        return self.P[0:3, 0:3], self.P[3:6, 3:6], NAN33, NAN33, NAN33


class KFIMU:
    """Linear KF with the accelerometer as input; the orientation comes from outside (truth)."""

    name, uses_imu = "KF-IMU", True

    def __init__(self, init: Init, noise: NoiseSpec, attitude=None, **_):
        if attitude is None:
            raise ValueError(
                "KF-IMU needs the given orientation (attitude=[N,3,3] or [N,4])"
            )
        att = np.asarray(attitude, dtype=float)
        self.R_given = kin.quat_to_rot_b(att) if att.shape[-1] == 4 else att
        self.k = 0
        self.x = np.r_[init.p, init.v, init.b_a].astype(float)
        self.P = np.diag(np.r_[np.full(3, init.std_p**2), np.full(3, init.std_v**2),
                               np.full(3, init.std_ba**2)])  # fmt: skip
        self.n = noise

    def predict(self, gyro, accel, dt):
        R = self.R_given[self.k]
        p, v, ba = self.x[0:3], self.x[3:6], self.x[6:9]
        a = R @ (accel - ba) + kin.G_W  # acceleration in w, linear in b_a for a given R
        self.x = np.r_[p + v * dt + 0.5 * a * dt * dt, v + a * dt, ba]
        F = np.eye(9)
        F[0:3, 3:6] = dt * np.eye(3)
        F[0:3, 6:9] = -0.5 * dt * dt * R
        F[3:6, 6:9] = -dt * R
        Q = np.zeros((9, 9))
        Q[3:6, 3:6] = self.n.accel_density**2 * dt * np.eye(3)
        Q[6:9, 6:9] = self.n.accel_bias_rw**2 * dt * np.eye(3)
        self.P = F @ self.P @ F.T + Q
        self.k += 1

    def update_position(self, z, R_meas):
        H = np.zeros((3, 9))
        H[:, 0:3] = np.eye(3)
        innov = z - self.x[0:3]
        dx, self.P, S = _joseph(self.P, H, R_meas, innov)
        self.x = self.x + dx
        return _info(innov, S)

    def update_body_velocity(self, rows, z, R_meas):
        """rows of R^T v (body velocity), linear in v because R is given."""
        R = self.R_given[min(self.k, len(self.R_given) - 1)]
        H = np.zeros((len(rows), 9))
        H[:, 3:6] = rows @ R.T
        innov = z - rows @ (R.T @ self.x[3:6])
        dx, self.P, S = _joseph(self.P, H, R_meas, innov)
        self.x = self.x + dx
        return _info(innov, S)

    def update_yaw_rate(self, z, var, gyro_z):
        return None  # KF-IMU does not use the gyroscope

    def update_heading(self, z, var):
        return None  # the orientation is given

    def update_zaru(self, gyro, var):
        return None

    def update_zupt(self, var):
        H = np.zeros((3, 9))
        H[:, 3:6] = np.eye(3)
        innov = -self.x[3:6]
        dx, self.P, S = _joseph(self.P, H, var * np.eye(3), innov)
        self.x = self.x + dx
        return _info(innov, S)

    def state(self):
        return self.x[0:3], self.x[3:6], np.full(4, np.nan), NAN3, self.x[6:9]

    def cov_blocks(self):
        return self.P[0:3, 0:3], self.P[3:6, 3:6], NAN33, NAN33, self.P[6:9, 6:9]


def _p0_15(init: Init):
    return np.diag(np.r_[np.full(3, init.std_p), np.full(3, init.std_v), np.full(3, init.std_att),
                         np.full(3, init.std_bg), np.full(3, init.std_ba)] ** 2)  # fmt: skip


class ESKF:
    """Error-state EKF (same equations as the repository ESKF)."""

    name, uses_imu = "ESKF", True

    def __init__(self, init: Init, noise: NoiseSpec, **_):
        self.p, self.v = init.p.astype(float).copy(), init.v.astype(float).copy()
        self.R = c.quat_to_rot(init.q)
        self.b_g, self.b_a = (
            init.b_g.astype(float).copy(),
            init.b_a.astype(float).copy(),
        )
        self.P = _p0_15(init)
        dens = [
            max(float(d), 1e-12)
            for d in (
                noise.gyro_density,
                noise.accel_density,
                noise.gyro_bias_rw,
                noise.accel_bias_rw,
            )
        ]
        self.Q_c = c.continuous_noise_cov(*dens)

    def predict(self, gyro, accel, dt):
        A, G = c.error_jacobians(self.R, accel - self.b_a, gyro - self.b_g)
        F, Q = c.discretise(A, G, self.Q_c, dt)
        self.p, self.v, self.R, _, _ = kin.prop_R(self.p, self.v, self.R, self.b_g, self.b_a,
                                                  gyro, accel, dt)  # fmt: skip
        self.P = c.propagate_cov(self.P, F, Q)

    def update_position(self, z, R_meas):
        H = np.zeros((3, 15))
        H[:, 0:3] = np.eye(3)
        innov = z - self.p
        dx, self.P, S = _joseph(self.P, H, R_meas, innov)
        self._inject(dx)
        return _info(innov, S)

    def _inject(self, dx):
        self.p = self.p + dx[0:3]
        self.v = self.v + dx[3:6]
        self.R = self.R @ kin.exp_b(dx[6:9])
        self.b_g = self.b_g + dx[9:12]
        self.b_a = self.b_a + dx[12:15]
        self.P = c.reset_cov(self.P, dx[6:9])

    def update_body_velocity(self, rows, z, R_meas):
        """rows of the body velocity R^T v (NHC: lateral, vertical = 0; odometer: forward)."""
        vb = self.R.T @ self.v
        H = np.zeros((len(rows), 15))
        H[:, 3:6] = rows @ self.R.T
        H[:, 6:9] = rows @ kin.skew_b(vb)
        innov = z - rows @ vb
        dx, self.P, S = _joseph(self.P, H, R_meas, innov)
        self._inject(dx)
        return _info(innov, S)

    def update_yaw_rate(self, z, var, gyro_z):
        """Yaw rate from the steering model: omega_z = gyro_z - b_gz (+ noise)."""
        H = np.zeros((1, 15))
        H[0, 11] = -1.0
        innov = np.array([z - (gyro_z - self.b_g[2])])
        dx, self.P, S = _joseph(self.P, H, np.array([[var]]), innov)
        self._inject(dx)
        return _info(innov, S)

    # --- states near the estimate, for numerical Jacobians and sigma points (batched)
    def retract(self, D):
        """Full states for error vectors D [n, 15] (this filter's error definition)."""
        return (self.p + D[:, 0:3], self.v + D[:, 3:6], self.R @ kin.exp_b(D[:, 6:9]),
                self.b_g + D[:, 9:12], self.b_a + D[:, 12:15])  # fmt: skip

    def num_jacobian(self, hfun, D0=None, eps=1e-6):
        D0 = np.zeros(15) if D0 is None else D0
        E = np.vstack([D0 + eps * np.eye(15), D0 - eps * np.eye(15), D0[None]])
        Z = hfun(*self.retract(E))
        H = (Z[:15] - Z[15:30]).T / (2 * eps)
        return H, Z[30]

    def _apply(self, H, R_meas, innov):
        dx, self.P, S = _joseph(self.P, H, R_meas, innov)
        self._inject(dx)
        return _info(innov, S)

    def update_heading(self, z, var):
        H, zh = self.num_jacobian(lambda p, v, R, bg, ba: heading_of(R)[:, None])
        return self._apply(H, np.array([[var]]), wrap(np.atleast_1d(z - zh)))

    def update_zupt(self, var):
        H = np.zeros((3, 15))
        H[:, 3:6] = np.eye(3)
        return self._apply(H, var * np.eye(3), -self.v)

    def update_zaru(self, gyro, var):
        H = np.zeros((3, 15))
        H[:, 9:12] = np.eye(3)
        return self._apply(H, var * np.eye(3), gyro - self.b_g)

    def state(self):
        q = c.quat_from_rotvec(c.so3_log(self.R))
        return self.p, self.v, q, self.b_g, self.b_a

    def cov_blocks(self):
        P = self.P
        return P[0:3, 0:3], P[3:6, 3:6], P[6:9, 6:9], P[9:12, 9:12], P[12:15, 12:15]


class IESKF(ESKF):
    """ESKF with an iterated update (Gauss-Newton) for the nonlinear measurements.

    Position, yaw rate, ZUPT and ZARU are linear in the error state, so the iteration changes
    nothing there; it is used for the body velocity (NHC, wheel speed) and the heading.
    """

    name = "IESKF"
    max_iter = 5

    def _iterate(self, hfun, z, R_meas, angle=False):
        P0 = self.P
        dx = np.zeros(15)
        for _ in range(self.max_iter):
            H, zh = self.num_jacobian(hfun, D0=dx)
            r = z - zh
            if angle:
                r = wrap(r)
            r = r + H @ dx
            S = H @ P0 @ H.T + R_meas
            K = np.linalg.solve(S, H @ P0).T
            new = K @ r
            done = np.max(np.abs(new - dx)) < 1e-10
            dx = new
            if done:
                break
        ikh = np.eye(15) - K @ H
        P = ikh @ P0 @ ikh.T + K @ R_meas @ K.T
        self.P = 0.5 * (P + P.T)
        H0, z0 = self.num_jacobian(hfun)
        innov = wrap(z - z0) if angle else z - z0
        S0 = H0 @ P0 @ H0.T + R_meas
        self._inject(dx)
        return _info(np.atleast_1d(innov), S0)

    def update_body_velocity(self, rows, z, R_meas):
        def hb(p, v, R, bg, ba):
            return (rows @ (np.swapaxes(R, -1, -2) @ v[..., None]))[..., 0]

        return self._iterate(hb, z, R_meas)

    def update_heading(self, z, var):
        return self._iterate(lambda p, v, R, bg, ba: heading_of(R)[:, None],
                             np.atleast_1d(z), np.array([[var]]), angle=True)  # fmt: skip


class UKF(ESKF):
    """Error-state UKF. Sigma points on the 15 error states (Merwe weights, alpha = 1)."""

    name = "UKF"

    def __init__(
        self, init: Init, noise: NoiseSpec, alpha=1.0, beta=2.0, kappa=0.0, **kw
    ):
        super().__init__(init, noise, **kw)
        n = 15
        lam = alpha**2 * (n + kappa) - n
        self.c = n + lam
        self.Wm = np.full(2 * n + 1, 0.5 / self.c)
        self.Wc = self.Wm.copy()
        self.Wm[0] = lam / self.c
        self.Wc[0] = lam / self.c + 1 - alpha**2 + beta

    def _sigma(self):
        L = np.linalg.cholesky(self.P + 1e-15 * np.eye(15)) * np.sqrt(self.c)
        return np.vstack([np.zeros(15), L.T, -L.T])

    def predict(self, gyro, accel, dt):
        A, G = c.error_jacobians(self.R, accel - self.b_a, gyro - self.b_g)
        _, Q = c.discretise(
            A, G, self.Q_c, dt
        )  # same additive process noise as the ESKF
        D = self._sigma()
        p, v, R, bg, ba = kin.prop_R(*self.retract(D), gyro, accel, dt)
        R0 = R[0]
        E = np.concatenate(
            [p - p[0], v - v[0], kin.log_b(R0.T @ R), bg - bg[0], ba - ba[0]], 1
        )
        m = self.Wm @ E
        dE = E - m
        P = (dE.T * self.Wc) @ dE + Q
        self.p, self.v = p[0] + m[0:3], v[0] + m[3:6]
        self.b_g, self.b_a = bg[0] + m[9:12], ba[0] + m[12:15]
        self.R = R0 @ kin.exp_b(m[6:9])
        self.P = 0.5 * (P + P.T)

    def _ut_update(self, hfun, z, R_meas, angle=False):
        D = self._sigma()
        Z = hfun(*self.retract(D))
        if angle:
            Z = Z[0] + wrap(Z - Z[0])
        zh = self.Wm @ Z
        dZ = Z - zh
        dX = D - self.Wm @ D
        S = (dZ.T * self.Wc) @ dZ + R_meas
        Pxz = (dX.T * self.Wc) @ dZ
        K = np.linalg.solve(S, Pxz.T).T
        innov = wrap(z - zh) if angle else z - zh
        dx = K @ innov
        P = self.P - K @ S @ K.T
        self.P = 0.5 * (P + P.T)
        self._inject(dx)
        return _info(np.atleast_1d(innov), S)

    def update_body_velocity(self, rows, z, R_meas):
        def hb(p, v, R, bg, ba):
            return (rows @ (np.swapaxes(R, -1, -2) @ v[..., None]))[..., 0]

        return self._ut_update(hb, z, R_meas)

    def update_heading(self, z, var):
        return self._ut_update(lambda p, v, R, bg, ba: heading_of(R)[:, None],
                               np.atleast_1d(z), np.array([[var]]), angle=True)  # fmt: skip


class InEKF:
    """Right-invariant EKF on SE_2(3) with biases.

    Error (world frame): R = Exp(xi_R) R_hat, v = Exp(xi_R) v_hat + J xi_v, p likewise,
    b = b_hat + zeta. Error order as in the ESKF: [xi_p, xi_v, xi_R, zeta_g, zeta_a].
    Error dynamics (first order):
      d xi_p = xi_v - [p]x R zeta_g,  d xi_v = [g]x xi_R - [v]x R zeta_g - R zeta_a,
      d xi_R = -R zeta_g
    so the navigation block does not depend on the estimate (only the bias columns do).
    """

    name, uses_imu = "InEKF", True

    def __init__(self, init: Init, noise: NoiseSpec, **_):
        self.p, self.v = init.p.astype(float).copy(), init.v.astype(float).copy()
        self.R = c.quat_to_rot(init.q)
        self.b_g, self.b_a = (
            init.b_g.astype(float).copy(),
            init.b_a.astype(float).copy(),
        )
        raw = (
            noise.gyro_density,
            noise.accel_density,
            noise.gyro_bias_rw,
            noise.accel_bias_rw,
        )
        dens = [max(float(d), 1e-12) for d in raw]
        self.Q_c = c.continuous_noise_cov(*dens)
        T = self._T_from_eskf()
        self.P = T @ _p0_15(init) @ T.T

    # maps between this error and the ESKF error (body-frame attitude error)
    def _T_from_eskf(self):
        T = np.eye(15)
        T[6:9, 6:9] = self.R
        T[3:6, 6:9] = kin.skew_b(self.v) @ self.R
        T[0:3, 6:9] = kin.skew_b(self.p) @ self.R
        return T

    def _T_to_eskf(self):
        T = np.eye(15)
        T[6:9, 6:9] = self.R.T
        T[3:6, 6:9] = -kin.skew_b(self.v)
        T[0:3, 6:9] = -kin.skew_b(self.p)
        return T

    def predict(self, gyro, accel, dt):
        R, v, p = self.R, self.v, self.p
        A = np.zeros((15, 15))
        A[0:3, 3:6] = np.eye(3)
        A[0:3, 9:12] = -kin.skew_b(p) @ R
        A[3:6, 6:9] = kin.skew_b(kin.G_W)
        A[3:6, 9:12] = -kin.skew_b(v) @ R
        A[3:6, 12:15] = -R
        A[6:9, 9:12] = -R
        G = np.zeros((15, 12))  # noise order: n_g, n_a, n_bg, n_ba
        G[0:3, 0:3] = -kin.skew_b(p) @ R
        G[3:6, 0:3] = -kin.skew_b(v) @ R
        G[3:6, 3:6] = -R
        G[6:9, 0:3] = -R
        G[9:12, 6:9] = np.eye(3)
        G[12:15, 9:12] = np.eye(3)
        F = np.eye(15) + A * dt + 0.5 * (A @ A) * dt * dt
        Q = F @ G @ self.Q_c @ G.T @ F.T * dt
        self.p, self.v, self.R, _, _ = kin.prop_R(
            p, v, R, self.b_g, self.b_a, gyro, accel, dt
        )
        P = F @ self.P @ F.T + Q
        self.P = 0.5 * (P + P.T)

    def retract(self, D):
        E = kin.exp_b(D[:, 6:9])
        Jl = np.stack([left_jacobian(d) for d in D[:, 6:9]])
        v = (E @ self.v) + (Jl @ D[:, 3:6, None])[..., 0]
        p = (E @ self.p) + (Jl @ D[:, 0:3, None])[..., 0]
        return p, v, E @ self.R, self.b_g + D[:, 9:12], self.b_a + D[:, 12:15]

    def _inject(self, dx):
        p, v, R, bg, ba = self.retract(dx[None])
        self.p, self.v, self.R, self.b_g, self.b_a = p[0], v[0], R[0], bg[0], ba[0]

    def _apply(self, H, R_meas, innov):
        dx, self.P, S = _joseph(self.P, H, R_meas, innov)
        self._inject(dx)
        return _info(innov, S)

    def update_position(self, z, R_meas):
        H = np.zeros((3, 15))
        H[:, 0:3] = np.eye(3)
        H[:, 6:9] = -kin.skew_b(self.p)
        return self._apply(H, R_meas, z - self.p)

    def update_body_velocity(self, rows, z, R_meas):
        H = np.zeros((len(rows), 15))
        H[:, 3:6] = rows @ self.R.T  # independent of v: the invariant property
        return self._apply(H, R_meas, z - rows @ (self.R.T @ self.v))

    def update_yaw_rate(self, z, var, gyro_z):
        H = np.zeros((1, 15))
        H[0, 11] = -1.0
        return self._apply(H, np.array([[var]]), np.array([z - (gyro_z - self.b_g[2])]))

    def update_heading(self, z, var):
        eps = 1e-6
        E = np.vstack([eps * np.eye(15), -eps * np.eye(15), np.zeros((1, 15))])
        Z = heading_of(self.retract(E)[2])
        H = ((Z[:15] - Z[15:30]) / (2 * eps))[None]
        return self._apply(H, np.array([[var]]), np.atleast_1d(wrap(z - Z[30])))

    def update_zupt(self, var):
        H = np.zeros((3, 15))
        H[:, 3:6] = np.eye(3)
        H[:, 6:9] = -kin.skew_b(self.v)
        return self._apply(H, var * np.eye(3), -self.v)

    def update_zaru(self, gyro, var):
        H = np.zeros((3, 15))
        H[:, 9:12] = np.eye(3)
        return self._apply(H, var * np.eye(3), gyro - self.b_g)

    def state(self):
        q = c.quat_from_rotvec(c.so3_log(self.R))
        return self.p, self.v, q, self.b_g, self.b_a

    def cov_blocks(self):
        T = self._T_to_eskf()
        P = T @ self.P @ T.T
        return P[0:3, 0:3], P[3:6, 3:6], P[6:9, 6:9], P[9:12, 9:12], P[12:15, 12:15]


class EKF:
    """Full-state EKF; Jacobians by central finite differences of the discrete model."""

    name, uses_imu = "EKF", True

    def __init__(self, init: Init, noise: NoiseSpec, **_):
        q = init.q / np.linalg.norm(init.q)
        self.x = np.r_[init.p, init.v, q, init.b_g, init.b_a].astype(float)
        J = self._J_att_to_q(q)
        T = np.zeros((16, 15))
        T[0:6, 0:6] = np.eye(6)
        T[6:10, 6:9] = J
        T[10:16, 9:15] = np.eye(6)
        self.P = T @ _p0_15(init) @ T.T + 1e-14 * np.eye(16)
        self.n = noise
        self.h = 1e-6

    @staticmethod
    def _J_att_to_q(q):
        """d q / d dtheta for q = q_hat (x) Exp(dtheta) at dtheta = 0 (4 x 3)."""
        J = np.zeros((4, 3))
        for i in range(3):
            e = np.zeros(4)
            e[1 + i] = 0.5
            J[:, i] = kin.quat_mul_b(q, e)
        return J

    @staticmethod
    def _J_q_to_att(q):
        """d dtheta / d q (3 x 4): pseudo-inverse of the map above on the unit sphere."""
        return np.linalg.pinv(EKF._J_att_to_q(q))

    def predict(self, gyro, accel, dt):
        n, h = 16, self.h
        X = np.vstack([self.x + h * np.eye(n), self.x - h * np.eye(n), self.x])
        U = np.vstack([h * np.eye(6), -h * np.eye(6)])
        Xall = np.vstack([X, np.repeat(self.x[None], 12, 0)])
        g = np.vstack([np.repeat(gyro[None], 2 * n + 1, 0), gyro + U[:, 0:3]])
        a = np.vstack([np.repeat(accel[None], 2 * n + 1, 0), accel + U[:, 3:6]])
        Y = kin.f_full(Xall, g, a, dt)
        F = (Y[0:n] - Y[n : 2 * n]).T / (2 * h)
        G = (Y[2 * n + 1 : 2 * n + 7] - Y[2 * n + 7 : 2 * n + 13]).T / (2 * h)
        nz = self.n
        Qu = np.diag(
            np.r_[
                np.full(3, nz.gyro_density**2 / dt),
                np.full(3, nz.accel_density**2 / dt),
            ]
        )
        Q = G @ Qu @ G.T
        Q[10:13, 10:13] += np.eye(3) * nz.gyro_bias_rw**2 * dt
        Q[13:16, 13:16] += np.eye(3) * nz.accel_bias_rw**2 * dt
        self.x = Y[2 * n]
        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)

    def update_position(self, z, R_meas):
        H = np.zeros((3, 16))
        H[:, 0:3] = np.eye(3)
        innov = z - self.x[0:3]
        dx, self.P, S = _joseph(self.P, H, R_meas, innov)
        self.x = self.x + dx
        self.x[6:10] /= np.linalg.norm(self.x[6:10])
        return _info(innov, S)

    def _update(self, H, R_meas, innov):
        dx, self.P, S = _joseph(self.P, H, R_meas, innov)
        self.x = self.x + dx
        self.x[6:10] /= np.linalg.norm(self.x[6:10])
        return _info(innov, S)

    def update_body_velocity(self, rows, z, R_meas):
        h = self.h

        def hb(X):
            R = kin.quat_to_rot_b(X[..., 6:10])
            return (rows @ (np.swapaxes(R, -1, -2) @ X[..., 3:6, None]))[..., 0]

        X = np.vstack([self.x + h * np.eye(16), self.x - h * np.eye(16), self.x])
        Z = hb(X)
        H = (Z[:16] - Z[16:32]).T / (2 * h)
        return self._update(H, R_meas, z - Z[32])

    def update_yaw_rate(self, z, var, gyro_z):
        H = np.zeros((1, 16))
        H[0, 12] = -1.0
        return self._update(H, np.array([[var]]), np.array([z - (gyro_z - self.x[12])]))

    def update_heading(self, z, var):
        h = self.h
        X = np.vstack([self.x + h * np.eye(16), self.x - h * np.eye(16), self.x])
        Z = heading_of(kin.quat_to_rot_b(X[:, 6:10]))
        H = ((Z[:16] - Z[16:32]) / (2 * h))[None]
        return self._update(H, np.array([[var]]), np.atleast_1d(wrap(z - Z[32])))

    def update_zupt(self, var):
        H = np.zeros((3, 16))
        H[:, 3:6] = np.eye(3)
        return self._update(H, var * np.eye(3), -self.x[3:6])

    def update_zaru(self, gyro, var):
        H = np.zeros((3, 16))
        H[:, 10:13] = np.eye(3)
        return self._update(H, var * np.eye(3), gyro - self.x[10:13])

    def state(self):
        x = self.x
        return x[0:3], x[3:6], x[6:10], x[10:13], x[13:16]

    def cov_blocks(self):
        J = self._J_q_to_att(self.x[6:10])
        P = self.P
        return (P[0:3, 0:3], P[3:6, 3:6], J @ P[6:10, 6:10] @ J.T, P[10:13, 10:13],
                P[13:16, 13:16])  # fmt: skip


FILTERS = {"KF-CV": KFCV, "KF-IMU": KFIMU, "EKF": EKF, "ESKF": ESKF, "IESKF": IESKF,
           "UKF": UKF, "InEKF": InEKF}  # fmt: skip
