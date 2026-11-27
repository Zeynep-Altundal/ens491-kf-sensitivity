"""Tests of the ENU/FLU/Hamilton to USD pose conversion (numpy only, no Isaac Sim)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parents[2]))

from usd_pose import enu_flu_to_usd, rotmat, usd_matrix  # noqa: E402
from synthetic.rotation import quat_to_rotmat  # noqa: E402


def _random_quats(n: int, seed: int = 7) -> np.ndarray:
    q = np.random.default_rng(seed).normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def test_rotmat_matches_synthetic_convention():
    for q in _random_quats(50):
        np.testing.assert_allclose(rotmat(q), quat_to_rotmat(q), atol=1e-12)


def test_usd_row_vector_matrix_maps_body_points_like_the_simulator():
    rng = np.random.default_rng(3)
    for q in _random_quats(50):
        p = rng.normal(size=3) * 10.0
        t, q_usd = enu_flu_to_usd(p, q)
        m = usd_matrix(t, q_usd)
        for b in rng.normal(size=(5, 3)):
            world_sim = p + rotmat(q) @ b
            world_usd = (np.append(b, 1.0) @ m)[:3]
            np.testing.assert_allclose(world_usd, world_sim, atol=1e-12)


@pytest.mark.parametrize("q, body, world", [
    # Yaw +90 deg about up: forward (east at zero yaw) points north.
    ([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)], [1, 0, 0], [0, 1, 0]),
    # Roll +90 deg about forward: left points up.
    ([np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0], [0, 1, 0], [0, 0, 1]),
    # Pitch +90 deg about left (nose down in FLU): forward points down.
    ([np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0], [1, 0, 0], [0, 0, -1]),
])
def test_known_rotations(q, body, world):
    t, q_usd = enu_flu_to_usd(np.zeros(3), np.array(q, dtype=float))
    got = (np.append(np.array(body, dtype=float), 1.0) @ usd_matrix(t, q_usd))[:3]
    np.testing.assert_allclose(got, world, atol=1e-12)


def test_zero_quaternion_is_rejected():
    with pytest.raises(ValueError):
        enu_flu_to_usd(np.zeros(3), np.zeros(4))
