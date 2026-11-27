"""Quadrotor rigid-body model, rotor allocation, ground contact and cascaded controller.

Conventions (as in ``synthetic/``): world frame ENU, body frame FLU, Hamilton quaternion
``[w, x, y, z]`` for the body-to-world rotation, gravity ``[0, 0, -9.81]``.
State vector ``x = [p (3), v (3), q (4), w_b (3)]`` (13,).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import repo_path  # noqa: F401  (makes ``synthetic`` importable)
from synthetic.imu_model import GRAVITY_W
from synthetic.rotation import quat_multiply, quat_normalize, quat_to_rotmat

E3 = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class QuadParams:
    # Mass, inertia and arm of the vehicle rendered in Isaac Sim (NTNU ARL-Robot-1 asset of the
    # Isaac Sim 6.0 assets): base link 1.2 kg with diag(0.013, 0.014, 0.013) kg m^2, four 0.01 kg
    # propellers at (+-0.1, +-0.1, 0) m; inertia about the centre with the parallel-axis theorem.
    mass: float = 1.24  # [kg]
    inertia: tuple[float, float, float] = (0.013404, 0.014404, 0.013804)  # [kg m^2], principal axes
    arm: float = 0.141421  # [m], centre to rotor (0.1 m * sqrt(2))
    f_max: float = 10.0  # [N] per rotor
    k_m: float = 0.016  # [m], yaw torque per unit thrust
    drag: float = 0.1  # [N s/m], linear translational drag

    @property
    def J(self) -> np.ndarray:
        return np.diag(self.inertia)

    def rotor_positions(self) -> np.ndarray:
        """Rotor positions in the body frame (4, 3): front-right, rear-left, front-left, rear-right."""
        a = self.arm / np.sqrt(2.0)
        return np.array([[a, -a, 0.0], [-a, a, 0.0], [a, a, 0.0], [-a, -a, 0.0]])

    def allocation(self) -> np.ndarray:
        """Matrix M with [T, tau_x, tau_y, tau_z] = M f for rotor thrusts f (4,)."""
        r = self.rotor_positions()
        spin = np.array([-1.0, -1.0, 1.0, 1.0])  # sign of the reaction yaw torque per rotor
        return np.vstack([np.ones(4), r[:, 1], -r[:, 0], spin * self.k_m])


@dataclass(frozen=True)
class ControllerGains:
    kp: tuple[float, float, float] = (6.0, 6.0, 8.0)  # [1/s^2]
    kd: tuple[float, float, float] = (4.5, 4.5, 5.5)  # [1/s]
    k_att: float = 100.0  # [1/s^2], roll/pitch attitude error, scaled by J
    k_rate: float = 16.0  # [1/s], roll/pitch angular-rate damping, scaled by J
    k_att_yaw: float = 30.0  # [1/s^2], yaw attitude error (low yaw authority of the rotors)
    k_rate_yaw: float = 10.0  # [1/s], yaw-rate damping
    max_tilt: float = np.deg2rad(45.0)  # [rad]


@dataclass
class Wrench:
    thrust: float
    torque: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotor_thrusts: np.ndarray = field(default_factory=lambda: np.zeros(4))


def hat(w: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def vee(m: np.ndarray) -> np.ndarray:
    return np.array([m[2, 1], m[0, 2], m[1, 0]])


def saturate(params: QuadParams, thrust: float, torque: np.ndarray) -> Wrench:
    """Map a commanded wrench to rotor thrusts, clip them to [0, f_max] and map back.

    The yaw torque has the lowest priority: if the rotor thrusts would leave their limits, the yaw
    torque is reduced (halved up to four times, then set to zero) before the thrusts are clipped.
    """
    m = params.allocation()
    torque = np.asarray(torque, dtype=float)
    for scale in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.0):
        cmd = np.array([thrust, torque[0], torque[1], scale * torque[2]])
        f = np.linalg.solve(m, cmd)
        if f.min() >= 0.0 and f.max() <= params.f_max:
            break
    f = np.clip(f, 0.0, params.f_max)
    w = m @ f
    return Wrench(thrust=float(w[0]), torque=w[1:].copy(), rotor_thrusts=f)


def on_ground(x: np.ndarray) -> bool:
    return x[2] <= 0.0


def derivative(x: np.ndarray, wrench: Wrench, params: QuadParams) -> np.ndarray:
    """Time derivative of the state for a constant applied wrench (ground contact included)."""
    v, q, w = x[3:6], x[6:10], x[10:13]
    r_wb = quat_to_rotmat(q)
    force = params.mass * GRAVITY_W + r_wb @ (wrench.thrust * E3) - params.drag * v
    acc = force / params.mass
    j = params.J
    w_dot = np.linalg.solve(j, wrench.torque - np.cross(w, j @ w))
    if on_ground(x) and acc[2] <= 0.0:
        # Supported by the ground: no motion until the thrust lifts the vehicle.
        return np.concatenate([np.zeros(3), np.zeros(3), np.zeros(4), np.zeros(3)])
    q_dot = 0.5 * quat_multiply(q, np.concatenate([[0.0], w]))
    return np.concatenate([v, acc, q_dot, w_dot])


def rk4_step(x: np.ndarray, wrench: Wrench, params: QuadParams, dt: float) -> np.ndarray:
    k1 = derivative(x, wrench, params)
    k2 = derivative(x + 0.5 * dt * k1, wrench, params)
    k3 = derivative(x + 0.5 * dt * k2, wrench, params)
    k4 = derivative(x + dt * k3, wrench, params)
    x_new = x + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    x_new[6:10] = quat_normalize(x_new[6:10])
    if x_new[2] < 0.0:
        # Touch-down: stop on the ground plane.
        x_new[2] = 0.0
        x_new[3:6] = 0.0
        x_new[10:13] = 0.0
    return x_new


def control(x: np.ndarray, p_ref: np.ndarray, v_ref: np.ndarray, a_ref: np.ndarray,
            params: QuadParams, gains: ControllerGains, yaw_ref: float = 0.0,
            yaw_rate_ref: float = 0.0) -> Wrench:
    """Cascaded controller: position PD -> desired force -> desired attitude -> torques.

    ``yaw_rate_ref`` [rad/s] is a feed-forward of the yaw rate (about world z) in the rate damping;
    with the default 0 the controller is the one used before it was added.
    """
    p, v, q, w = x[0:3], x[3:6], x[6:10], x[10:13]
    a_des = a_ref + np.asarray(gains.kp) * (p_ref - p) + np.asarray(gains.kd) * (v_ref - v)
    f_des = params.mass * (a_des - GRAVITY_W) + params.drag * v

    # Tilt limit: keep the vertical component and scale the horizontal one.
    f_des[2] = max(f_des[2], 0.1 * params.mass * 9.81)
    horiz = np.linalg.norm(f_des[:2])
    max_horiz = f_des[2] * np.tan(gains.max_tilt)
    if horiz > max_horiz:
        f_des[:2] *= max_horiz / horiz

    r_wb = quat_to_rotmat(q)
    z_d = f_des / np.linalg.norm(f_des)
    x_c = np.array([np.cos(yaw_ref), np.sin(yaw_ref), 0.0])
    y_d = np.cross(z_d, x_c)
    y_d /= np.linalg.norm(y_d)
    x_d = np.cross(y_d, z_d)
    r_d = np.column_stack([x_d, y_d, z_d])

    thrust = float(f_des @ (r_wb @ E3))
    e_r = 0.5 * vee(r_d.T @ r_wb - r_wb.T @ r_d)
    j = params.J
    k_att = np.array([gains.k_att, gains.k_att, gains.k_att_yaw])
    k_rate = np.array([gains.k_rate, gains.k_rate, gains.k_rate_yaw])
    w_d = r_wb.T @ np.array([0.0, 0.0, yaw_rate_ref])  # desired angular velocity, body frame
    torque = j @ (-k_att * e_r - k_rate * (w - w_d)) + np.cross(w, j @ w)
    return saturate(params, thrust, torque)


def hover_state(z: float = 2.0) -> np.ndarray:
    """State at rest at height ``z`` with identity attitude."""
    x = np.zeros(13)
    x[2] = z
    x[6] = 1.0
    return x


def hover_wrench(params: QuadParams) -> Wrench:
    return saturate(params, params.mass * 9.81, np.zeros(3))
