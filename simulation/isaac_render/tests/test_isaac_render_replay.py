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


def test_chase_camera_stays_above_ground_and_near_the_drone():
    t = np.arange(0.0, 10.0, 0.005)
    p = np.stack([3 * np.sin(t), 1.5 * np.sin(2 * t), 0.1 + 0 * t], axis=1)
    eye, target = replay.chase_path(t, p)
    assert eye[:, 2].min() >= 0.25
    dist = np.linalg.norm(eye - p, axis=1)
    assert dist.max() < 4.0
    assert dist.min() > 1.5  # minimum distance (2 m, with a smoothed push-back)
    assert np.linalg.norm(target - p, axis=1).max() < 1.0


def test_topdown_camera_has_north_up():
    cam = replay.camera_pose("topdown", 0, np.zeros((1, 3)), np.zeros((1, 3)))
    look = cam.target - cam.eye
    right = np.cross(look, cam.up)
    np.testing.assert_allclose(right / np.linalg.norm(right), [1.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        replay.camera_pose("side", 0, np.zeros((1, 3)), np.zeros((1, 3)))


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
