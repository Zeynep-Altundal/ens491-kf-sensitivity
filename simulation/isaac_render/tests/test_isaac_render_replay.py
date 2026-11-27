"""Tests of the Isaac-independent part of the replay (cameras, propellers, trail, overlay)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parents[1] / "rviz_drone"))

import replay  # noqa: E402
from rviz_drone.quadrotor import QuadParams  # noqa: E402


def test_rotor_prims_match_the_rotor_positions_of_the_model():
    # front/back and left/right in the prim names must match the signs of the rotor positions.
    for name, (x, y, _) in zip(replay.ROTOR_PRIMS, QuadParams().rotor_positions()):
        assert ("front" in name) == (x > 0)
        assert ("left" in name) == (y > 0)


def test_spin_directions_oppose_the_reaction_torque_and_cancel():
    spin = replay.rotor_spin_direction(QuadParams().allocation())
    assert spin.sum() == 0
    # Diagonal pairs spin in the same direction (front-right with rear-left).
    assert spin[0] == spin[1] and spin[2] == spin[3] and spin[0] != spin[2]


def test_prop_angles_hover_rate_and_limit():
    t = np.arange(0.0, 2.0 + 1e-12, 0.005)
    hover = 3.0
    f = np.tile([hover, hover, 4 * hover, 100 * hover], (len(t), 1))
    ang = replay.prop_angles(t, f, hover, np.array([1, -1, 1, 1]), hover_rev_s=4.0, max_rev_s=7.0)
    rev_s = (ang[-1] - ang[0]) / (2 * np.pi) / (t[-1] - t[0])
    np.testing.assert_allclose(rev_s, [4.0, -4.0, 7.0, 7.0], rtol=1e-9)


def test_low_pass_converges_and_is_causal():
    t = np.arange(0.0, 5.0, 0.005)
    x = np.where(t[:, None] >= 1.0, 1.0, 0.0) * np.ones((1, 3))
    y = replay.low_pass(x, t, tau=0.2)
    assert np.all(y[t < 1.0] == 0.0)
    np.testing.assert_allclose(y[-1], 1.0, atol=1e-6)
    k = replay.sample_index(t, 1.2)
    np.testing.assert_allclose(y[k], 1.0 - np.exp(-1.0), atol=0.02)


def _quat(yaw_deg, pitch_deg=0.0, roll_deg=0.0):
    """Hamilton quaternion of R = Rz(yaw) Ry(pitch) Rx(roll)."""
    def q(axis, deg):
        a = np.radians(deg) / 2
        v = np.zeros(4)
        v[0], v[axis] = np.cos(a), np.sin(a)
        return v

    def mul(a, b):
        aw, ax, ay, az = a
        bw, bx, by, bz = b
        return np.array([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                         aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw])
    return mul(mul(q(3, yaw_deg), q(2, pitch_deg)), q(1, roll_deg))


def test_yaw_is_independent_of_roll_and_pitch():
    for yaw in (-170.0, -30.0, 0.0, 45.0, 120.0):
        for pitch, roll in ((0, 0), (30, 0), (0, 40), (-35, 25)):
            got = np.degrees(replay.yaw_from_quat(_quat(yaw, pitch, roll))[0])
            assert abs((got - yaw + 180) % 360 - 180) < 1e-9


def test_chase_camera_follows_position_and_yaw_only():
    t = np.arange(0.0, 6.0, 0.005)
    p = np.stack([2.0 * t, 0 * t, 3.0 + 0 * t], axis=1)
    level = np.array([_quat(0.0) for _ in t])
    rolling = np.array([_quat(0.0, 20 * np.sin(x), 35 * np.sin(3 * x)) for x in t])
    e1, g1 = replay.chase_path(t, p, replay.yaw_from_quat(level))
    e2, g2 = replay.chase_path(t, p, replay.yaw_from_quat(rolling))
    np.testing.assert_allclose(e1, e2, atol=1e-12)  # roll and pitch do not move the camera
    np.testing.assert_allclose(g1, g2, atol=1e-12)
    # Behind the drone (heading +x) and above it; target near the drone.
    k = replay.sample_index(t, 5.0)
    assert e1[k, 0] < p[k, 0] - 2.0 and e1[k, 2] > p[k, 2]
    assert np.linalg.norm(g1[k] - p[k]) < 0.5
    # A heading of +90 deg puts the camera south of the drone; yaw is smoothed (no jump).
    yaw = np.where(t < 3.0, 0.0, np.pi / 2)
    e3, _ = replay.chase_path(t, p * 0, yaw)
    assert e3[-1, 1] < -2.0 and abs(e3[-1, 0]) < 0.1
    assert np.abs(np.diff(e3, axis=0)).max() < 0.05


def test_chase_camera_keeps_its_distance_and_height():
    t = np.arange(0.0, 10.0, 0.005)
    p = np.stack([3 * np.sin(t), 1.5 * np.sin(2 * t), 0.1 + 0 * t], axis=1)
    yaw = np.arctan2(np.gradient(p[:, 1]), np.gradient(p[:, 0]))
    eye, target = replay.chase_path(t, p, yaw)
    assert eye[:, 2].min() >= 0.25
    dist = np.linalg.norm(eye - p, axis=1)
    assert dist.min() > 1.5 and dist.max() < 5.0
    assert np.linalg.norm(target - p, axis=1).max() < 1.0


def test_chase_camera_keeps_the_horizon_in_frame():
    from rviz_drone import reference
    t = np.arange(0.0, reference.DURATION, 0.01)
    p = np.array([reference.reference(x)[0] for x in t])
    yaw = np.array([reference.yaw_reference(x)[0] for x in t])
    eye, target = replay.chase_path(t, p, yaw)
    d = target - eye
    pitch = np.degrees(np.arctan2(d[:, 2], np.linalg.norm(d[:, :2], axis=1)))
    half_vfov = np.degrees(np.arctan(np.tan(np.radians(47.18 / 2)) * 9 / 16))
    # Horizon (elevation 0) inside the upper part of the frame: the line of sight is at most a few
    # degrees below level, so the horizon is above the image centre and below the top edge.
    in_frame = (pitch < 0.0) & (pitch > -0.7 * half_vfov)
    assert in_frame.mean() > 0.9
    # The drone stays in the frame.
    to_drone = p - eye
    below = np.degrees(np.arctan2(to_drone[:, 2], np.linalg.norm(to_drone[:, :2], axis=1))) - pitch
    assert np.abs(below).max() < half_vfov


def test_online_chase_filter_equals_the_offline_chase_path():
    t = np.arange(0.0, 20.0, 0.005)
    p = np.stack([3 * np.sin(t), 1.5 * np.sin(2 * t), 1.0 + 0.5 * np.sin(0.3 * t)], axis=1)
    yaw = np.arctan2(np.gradient(p[:, 1]), np.gradient(p[:, 0]))  # wraps at +-pi
    eye, target = replay.chase_path(t, p, yaw)
    f = replay.ChaseFilter()
    online = np.array([np.concatenate(f.step(t[k], p[k], yaw[k])) for k in range(len(t))])
    np.testing.assert_allclose(online[:, :3], eye, atol=1e-9)
    np.testing.assert_allclose(online[:, 3:], target, atol=1e-9)


def test_fill_light_shines_along_the_view_from_above():
    d = replay.fill_direction(np.array([0.0, 0.0, 1.0]), np.array([3.0, 4.0, 0.5]))
    assert abs(np.linalg.norm(d) - 1.0) < 1e-12 and d[2] < 0
    np.testing.assert_allclose(d[:2] / np.linalg.norm(d[:2]), [0.6, 0.8])
    np.testing.assert_allclose(replay.fill_direction(np.array([0, 0, 50.0]), np.zeros(3)), [0, 0, -1.0])


def test_fixed_cameras_frame_the_flight():
    p = np.array([[0.0, 0.0, 0.0], [40.0, 0.0, 3.0], [40.0, 28.0, 3.0]])
    cams = replay.fixed_cameras(p)
    top = cams["topdown"]
    look = top.target - top.eye
    right = np.cross(look, top.up)
    np.testing.assert_allclose(right / np.linalg.norm(right), [1.0, 0.0, 0.0])  # north up, east right
    uv = replay.project(p, top.eye, top.target, top.up, 24.0, 20.955, 1280, 720)
    assert np.all((uv >= 0) & (uv <= [1280, 720]))
    ov = cams["overview"]
    uv = replay.project(p, ov.eye, ov.target, ov.up, 24.0, 20.955, 1280, 720)
    assert np.all((uv >= 0) & (uv <= [1280, 720]))
    with pytest.raises(ValueError):
        replay.camera_pose("side", 0, np.zeros((1, 3)), np.zeros((1, 3)), cams)


def test_upcoming_route_starts_at_the_reference_and_looks_ahead():
    t = np.arange(0.0, 20.0, 0.005)
    p_ref = np.stack([t, 0 * t, 0 * t], axis=1)
    k = replay.sample_index(t, 5.0)
    pts = replay.upcoming_route(t, p_ref, k, horizon_s=8.0, n=81)
    np.testing.assert_allclose(pts[0], [5.0, 0, 0])
    np.testing.assert_allclose(pts[-1], [13.0, 0, 0], atol=0.005)
    end = replay.upcoming_route(t, p_ref, len(t) - 10)
    assert np.all(end[:, 0] <= t[-1] + 1e-12)


def test_trail_is_ordered_and_bounded():
    t = np.arange(0.0, 10.0, 0.005)
    p = np.stack([t, 0 * t, 0 * t], axis=1)
    pts, age = replay.trail(t, p, replay.sample_index(t, 5.0), length_s=1.5, n=31)
    assert np.all(np.diff(pts[:, 0]) > 0)
    np.testing.assert_allclose(pts[-1], [5.0, 0, 0])
    np.testing.assert_allclose(pts[0], [3.5, 0, 0], atol=0.005)
    assert age[0] == pytest.approx(1.0) and age[-1] == 0.0
    # Near the start the trail is clipped to t = 0.
    pts0, _ = replay.trail(t, p, 0)
    np.testing.assert_allclose(pts0, 0.0)


def test_overlay_lines():
    lines = replay.overlay_lines(12.345, "figure_eight", 2.5, "chase")
    assert lines == ["t =  12.35 s", "phase: figure-eight", "speed:  2.50 m/s", "camera: chase"]


def test_projection_of_a_side_camera():
    eye, target, up = np.array([0.0, -3.0, 1.5]), np.array([0.0, 0.0, 1.5]), np.array([0.0, 0.0, 1.0])
    uv = replay.project(np.array([[0.0, 0.0, 1.5], [1.0, 0.0, 1.5], [0.0, 0.0, 2.5]]), eye, target, up,
                        24.0, 20.955, 1280, 720)
    np.testing.assert_allclose(uv[0], [640.0, 360.0])
    f_px = 24.0 / 20.955 * 1280
    np.testing.assert_allclose(uv[1], [640.0 + f_px / 3.0, 360.0])  # +x is to the right
    np.testing.assert_allclose(uv[2], [640.0, 360.0 - f_px / 3.0])  # +z is up


def test_principal_angle_of_a_tilted_bar():
    mask = np.zeros((200, 200), bool)
    for s in np.linspace(-60, 60, 400):
        for w in np.linspace(-4, 4, 9):
            a = np.radians(30.0)
            u = 100 + s * np.cos(a) - w * np.sin(a)
            v = 100 - (s * np.sin(a) + w * np.cos(a))
            mask[int(round(v)), int(round(u))] = True
    assert abs(replay.principal_angle_deg(mask) - 30.0) < 1.0
