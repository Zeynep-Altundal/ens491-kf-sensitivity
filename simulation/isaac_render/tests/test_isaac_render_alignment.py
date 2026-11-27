"""Tests of the fixed asset alignment (ARL-Robot-1 asset frame to FLU body frame)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parents[1] / "rviz_drone"))

import asset_alignment as al  # noqa: E402
import replay  # noqa: E402
from rviz_drone.quadrotor import QuadParams  # noqa: E402


def test_alignment_is_a_unit_quaternion_and_a_proper_rotation():
    assert np.linalg.norm(al.ASSET_TO_BODY) == 1.0
    r = al.rotmat(al.ASSET_TO_BODY)
    np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(r) == 1.0


def test_aligned_asset_rotors_coincide_with_the_model_rotors():
    np.testing.assert_allclose(al.asset_to_body(al.ASSET_ROTOR_POSITIONS), QuadParams().rotor_positions(),
                               atol=1e-6)


def test_rotor_names_match_their_measured_positions():
    for name, (x, y, _) in zip(replay.ROTOR_PRIMS, al.ASSET_ROTOR_POSITIONS):
        assert ("front" in name) == (x > 0)
        assert ("left" in name) == (y > 0)


def test_spin_axis_maps_to_body_up():
    np.testing.assert_allclose(al.asset_to_body(al.ASSET_SPIN_AXIS[None, :])[0], [0.0, 0.0, 1.0])


def test_alignment_is_not_part_of_the_pose_conversion():
    import usd_pose
    import inspect
    assert "asset" not in inspect.signature(usd_pose.enu_flu_to_usd).parameters
