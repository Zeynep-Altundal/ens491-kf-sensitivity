"""Fixed alignment of the drone asset (NTNU ARL-Robot-1) to the FLU body frame of the simulator.

Kept separate from the pose conversion (``usd_pose.py``): the pose conversion maps the simulator
state to the body prim ``/World/Quadrotor``; this constant is a fixed rotation of the asset prim
below it (``/World/Quadrotor/asset``, ``xformOp:orient``), from asset coordinates to body
coordinates.

Evidence for the identity rotation (asset frame = FLU body frame):

* Propeller prims measured in the asset (``render_flight.py`` prints them at every start):
  ``front_right_prop`` (0.1, -0.1, 0), ``back_left_prop`` (-0.1, 0.1, 0), ``front_left_prop``
  (0.1, 0.1, 0), ``back_right_prop`` (-0.1, -0.1, 0) m: "front" is +x and "left" is +y, so x is
  forward and y is left; the propellers lie in the plane z = 0 and spin about their z axis
  (``xformOp:orient`` identity), so z is the thrust axis (up).
* Asset stage: ``upAxis = Z``, ``metersPerUnit = 1``.
* Rendered checks (``render_flight.py --verify-heading`` and ``--inspect``): heading of the front
  marker from above, and side views of hover, +x translation, +90 deg yaw and t = 36 s.
"""

from __future__ import annotations

import numpy as np

# Hamilton [w, x, y, z], asset coordinates to body coordinates.
ASSET_TO_BODY = np.array([1.0, 0.0, 0.0, 0.0])

# Propeller centres in asset coordinates [m], measured from the asset, in the rotor order of
# ``QuadParams.rotor_positions`` (front-right, rear-left, front-left, rear-right).
ASSET_ROTOR_POSITIONS = np.array([[0.1, -0.1, 0.0], [-0.1, 0.1, 0.0], [0.1, 0.1, 0.0], [-0.1, -0.1, 0.0]])
# Spin axis of every propeller prim in asset coordinates.
ASSET_SPIN_AXIS = np.array([0.0, 0.0, 1.0])


def rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def asset_to_body(points: np.ndarray) -> np.ndarray:
    """Asset coordinates (n, 3) to body coordinates."""
    return np.asarray(points, dtype=float) @ rotmat(ASSET_TO_BODY).T
