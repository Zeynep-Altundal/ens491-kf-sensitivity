"""Everything of the replay that does not need Isaac Sim (numpy only, tested in ``tests/``).

Camera paths, propeller angles, the trail behind the drone, the flight phase and the overlay text
are computed here from the simulator truth (``t``, ``p``, ``v``, ``rotor`` at 200 Hz).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Prims of the propellers in the ARL-Robot-1 asset, in the rotor order of
# ``QuadParams.rotor_positions``: front-right, rear-left, front-left, rear-right.
ROTOR_PRIMS = ("front_right_prop", "back_left_prop", "front_left_prop", "back_right_prop")

PHASE_LABELS = {"ground": "on ground", "takeoff": "take-off", "hover": "hover",
                "straight": "straight leg", "turn": "coordinated turn", "hover_turn": "hover, turning",
                "figure_eight": "figure-eight", "aggressive": "aggressive", "return": "return",
                "landing": "landing"}

CAMERAS = ("chase", "overview", "topdown")


@dataclass(frozen=True)
class CameraPose:
    eye: np.ndarray
    target: np.ndarray
    up: np.ndarray


def sample_index(t: np.ndarray, time: float) -> int:
    """Index of the truth sample nearest to ``time`` (uniform sampling)."""
    dt = (t[-1] - t[0]) / (len(t) - 1)
    return int(np.clip(np.round((time - t[0]) / dt), 0, len(t) - 1))


def low_pass(x: np.ndarray, t: np.ndarray, tau: float) -> np.ndarray:
    """Causal first-order low-pass filter along axis 0 (exact discretisation), x(0) as initial value."""
    y = np.empty_like(x, dtype=float)
    y[0] = x[0]
    alpha = 1.0 - np.exp(-np.diff(t) / tau)
    for k in range(1, len(t)):
        y[k] = y[k - 1] + alpha[k - 1] * (x[k] - y[k - 1])
    return y


def rotor_spin_direction(allocation: np.ndarray) -> np.ndarray:
    """Sign of the rotor angular velocity about body z (+1 counter-clockwise seen from above).

    Row 3 of the allocation matrix is the reaction yaw torque per unit thrust; the rotor spins in
    the opposite direction.
    """
    return -np.sign(allocation[3])


def prop_angles(t: np.ndarray, rotor_thrust: np.ndarray, hover_thrust: float, spin: np.ndarray,
                hover_rev_s: float = 4.0, max_rev_s: float = 7.0) -> np.ndarray:
    """Propeller angles [rad] (n, 4) for display.

    The displayed speed is ``hover_rev_s * sqrt(f / f_hover)`` revolutions per second, limited to
    ``max_rev_s`` so that it stays below half the frame rate at 30 frames/s (no backward-turning
    aliasing). It is a visual cue, not the physical rotor speed (thousands of rpm).
    """
    rate = hover_rev_s * np.sqrt(np.clip(rotor_thrust, 0.0, None) / hover_thrust)
    rate = np.minimum(rate, max_rev_s) * 2.0 * np.pi
    dt = np.diff(t, prepend=t[0])
    return np.cumsum(rate * dt[:, None], axis=0) * spin[None, :]


def yaw_from_quat(q: np.ndarray) -> np.ndarray:
    """Yaw [rad] (rotation about world z of the body x axis) of Hamilton quaternions (n, 4) [w, x, y, z];
    independent of roll and pitch below 90 deg."""
    q = np.atleast_2d(q)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def chase_path(t: np.ndarray, p: np.ndarray, yaw: np.ndarray, distance: float = 2.6, height: float = 0.45,
               tau_pos: float = 0.35, tau_yaw: float = 0.8, tau_target: float = 0.12,
               min_height: float = 0.25, min_distance: float = 2.0, target_lift: float = 0.25
               ) -> tuple[np.ndarray, np.ndarray]:
    """Eye and target of the chase camera.

    The camera follows the position and the yaw of the drone only (never roll or pitch): it sits
    ``distance`` behind and ``height`` above the low-pass filtered position, along the low-pass
    filtered (unwrapped) yaw; the target is ``target_lift`` above the low-pass filtered position, so
    that the line of sight is nearly level (about 4 deg down at the defaults) and the horizon and
    the sky stay in the upper part of the frame; the drone is seen slightly below the image centre. The up vector of the
    camera is world z (``camera_pose``). If the drone comes closer than ``min_distance`` (e.g. when
    it brakes), the camera is pushed back along the line of sight (push smoothed as well).
    """
    ps = low_pass(p, t, tau_pos)
    ys = low_pass(np.unwrap(yaw)[:, None], t, tau_yaw)[:, 0]
    eye = ps + np.stack([-distance * np.cos(ys), -distance * np.sin(ys), np.full(len(t), height)], axis=1)
    target = low_pass(p + np.array([0.0, 0.0, target_lift]), t, tau_target)
    los = eye - p
    dist = np.linalg.norm(los, axis=1, keepdims=True)
    push = np.where(dist < min_distance, (min_distance - dist) / np.maximum(dist, 1e-9), 0.0) * los
    eye = eye + low_pass(push, t, tau_pos / 3.0)
    eye[:, 2] = np.maximum(eye[:, 2], min_height)
    return eye, target


def fixed_cameras(p: np.ndarray, sun_azimuth_deg: float = 146.5, hfov_deg: float = 47.18,
                  aspect: float = 16 / 9) -> dict[str, CameraPose]:
    """Overview and top-down cameras that frame the whole flight ``p`` (n, 3).

    Overview: from the side of the sun (the drone is lit from the front), 35 deg above the
    horizon. Top-down: straight above the centre, north up (east to the right)."""
    lo, hi = p.min(axis=0), p.max(axis=0)
    centre = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, 1.0])
    half = np.array([(hi[0] - lo[0]) / 2 + 3.0, (hi[1] - lo[1]) / 2 + 3.0])
    tan_h = np.tan(np.radians(hfov_deg) / 2)
    radius = float(np.hypot(*half))
    az, el = np.radians(sun_azimuth_deg), np.radians(35.0)
    direction = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    overview = CameraPose(eye=centre + direction * (1.15 * radius / tan_h), target=centre,
                          up=np.array([0.0, 0.0, 1.0]))
    height = max(half[0], half[1] * aspect) / tan_h
    topdown = CameraPose(eye=np.array([centre[0], centre[1], height]), target=np.array([centre[0], centre[1], 0.0]),
                         up=np.array([0.0, 1.0, 0.0]))
    return {"overview": overview, "topdown": topdown}


def fill_direction(eye: np.ndarray, target: np.ndarray, down_deg: float = 30.0) -> np.ndarray:
    """Unit direction of travel of the fill light: the horizontal view direction of the camera,
    tilted ``down_deg`` towards the ground (light from behind and above the camera). For a camera
    looking straight down it is straight down."""
    d = np.asarray(target, float) - np.asarray(eye, float)
    h = np.array([d[0], d[1], 0.0])
    n = np.linalg.norm(h)
    if n < 1e-6 * max(np.linalg.norm(d), 1e-12):
        return np.array([0.0, 0.0, -1.0])
    a = np.radians(down_deg)
    return np.cos(a) * h / n + np.array([0.0, 0.0, -np.sin(a)])


def camera_pose(name: str, k: int, chase_eye: np.ndarray, chase_target: np.ndarray,
                fixed: dict[str, CameraPose]) -> CameraPose:
    if name == "chase":
        return CameraPose(eye=chase_eye[k], target=chase_target[k], up=np.array([0.0, 0.0, 1.0]))
    if name in fixed:
        return fixed[name]
    raise ValueError(f"unknown camera {name!r}; choose from {CAMERAS}")


def upcoming_route(t: np.ndarray, p_ref: np.ndarray, k: int, horizon_s: float = 8.0, n: int = 81) -> np.ndarray:
    """Points of the reference route from now to ``horizon_s`` ahead (clipped at the end)."""
    times = np.minimum(t[k] + np.linspace(0.0, horizon_s, n), t[-1])
    return p_ref[[sample_index(t, x) for x in times]]


def trail(t: np.ndarray, p: np.ndarray, k: int, length_s: float = 2.0, n: int = 41
          ) -> tuple[np.ndarray, np.ndarray]:
    """Points of the trail behind the drone (oldest first) and their age fraction in [0, 1]."""
    times = t[k] - np.linspace(length_s, 0.0, n)
    times = np.maximum(times, t[0])
    idx = np.array([sample_index(t, x) for x in times])
    age = (t[k] - t[idx]) / length_s
    return p[idx], np.clip(age, 0.0, 1.0)


def overlay_lines(time: float, phase: str, speed: float, camera: str) -> list[str]:
    return [f"t = {time:6.2f} s", f"phase: {PHASE_LABELS.get(phase, phase)}",
            f"speed: {speed:5.2f} m/s", f"camera: {camera}"]


def project(points: np.ndarray, eye: np.ndarray, target: np.ndarray, up: np.ndarray, focal_mm: float,
            h_aperture_mm: float, width: int, height: int) -> np.ndarray:
    """Pixel coordinates (n, 2) of world points for a pinhole camera built like ``SetLookAt``
    (right = forward x up), square pixels, principal point at the image centre."""
    f = np.asarray(target, float) - np.asarray(eye, float)
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    d = np.atleast_2d(points) - np.asarray(eye, float)
    z = d @ f
    f_px = focal_mm / h_aperture_mm * width
    return np.stack([width / 2 + f_px * (d @ r) / z, height / 2 - f_px * (d @ u) / z], axis=1)


def tilt_deg(rotmat_wb: np.ndarray) -> float:
    """Angle between body z and world z [deg]."""
    return float(np.degrees(np.arccos(np.clip(rotmat_wb[2, 2], -1.0, 1.0))))


def principal_angle_deg(mask: np.ndarray) -> float:
    """Angle of the major axis of a pixel mask against the image horizontal [deg], counter-clockwise
    positive (image y points down), in (-90, 90]."""
    vs, us = np.nonzero(mask)
    c = np.cov(np.stack([us - us.mean(), -(vs - vs.mean())]))
    w, v = np.linalg.eigh(c)
    major = v[:, np.argmax(w)]
    a = np.degrees(np.arctan2(major[1], major[0]))
    return float((a + 90.0) % 180.0 - 90.0) if a != -90.0 else 90.0
