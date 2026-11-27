"""Simulated magnetometer heading (compass) for both platforms.

The magnetometer measures the Earth's magnetic field in the body frame; after tilt compensation it
gives the heading (yaw) of the body x axis. Here we simulate that heading directly:
  heading_m = true heading + constant offset (residual hard-iron / declination error) + white noise.
Typical values: a few degrees of noise and offset on a vehicle with motors and steel nearby
(docs/synthetic_data_design.md).
"""

from __future__ import annotations

import numpy as np

MAG_STD_RAD = float(np.deg2rad(2.0))


def simulate_heading(seq, rate_hz=10.0, std_rad=MAG_STD_RAD, offset_rad=0.0, seed=0):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 5151]))
    ref = seq.reference
    t = np.arange(ref.t_s[0], ref.t_s[-1], 1.0 / rate_hz)
    q = ref.q_wb
    w, x, y, z = (np.interp(t, ref.t_s, q[:, i]) for i in range(4))
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    heading = np.arctan2(
        2 * (x * y + w * z), 1 - 2 * (y * y + z * z)
    )  # atan2(R10, R00)
    meas = heading + offset_rad + rng.normal(0.0, std_rad, len(t))
    meas = (meas + np.pi) % (2 * np.pi) - np.pi
    return {"mag_t": t, "mag_heading": meas,
            "mag_meta": {"rate_hz": rate_hz, "std_rad": float(std_rad),
                         "offset_rad": float(offset_rad)}}  # fmt: skip
