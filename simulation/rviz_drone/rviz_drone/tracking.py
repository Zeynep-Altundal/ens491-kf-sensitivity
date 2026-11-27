"""Tracking table of the closed-loop flight (position error, tilt and yaw per trajectory segment).

    python -m rviz_drone.tracking      (from simulation/rviz_drone/)

prints the Markdown table that is recorded in ``../README.md``.
"""

from __future__ import annotations

import numpy as np

from .flight import simulate_truth
from .reference import reference, segment_at
from synthetic.rotation import quat_to_rotmat

ROWS = (("take-off", ("takeoff",)), ("hover (both)", ("hover",)), ("figure-eight", ("figure_eight",)),
        ("aggressive", ("aggressive",)), ("landing", ("landing",)))


def tracking_rows(truth: dict[str, np.ndarray] | None = None) -> list[dict[str, float | str]]:
    truth = truth if truth is not None else simulate_truth()
    t, p, p_ref, q = truth["t"], truth["p"], truth["p_ref"], truth["q"]
    names = np.array([segment_at(x).name for x in t])
    err = np.linalg.norm(p - p_ref, axis=1)
    a_ref = np.array([np.linalg.norm(reference(x)[2]) for x in t])
    r = np.array([quat_to_rotmat(x) for x in q])
    tilt = np.degrees(np.arccos(np.clip(r[:, 2, 2], -1.0, 1.0)))
    yaw = np.degrees(np.abs(np.arctan2(r[:, 1, 0], r[:, 0, 0])))
    rows = []
    for label, segs in ROWS:
        m = np.isin(names, segs)
        rows.append({"segment": label, "a_ref": a_ref[m].max(), "max_err": err[m].max(),
                     "rms_err": float(np.sqrt(np.mean(err[m] ** 2))), "tilt": tilt[m].max(),
                     "yaw": yaw[m].max()})
    return rows


def markdown(rows: list[dict[str, float | str]]) -> str:
    def e(x: float) -> str:
        return "< 0.001" if x < 0.001 else f"{x:.3f}"

    lines = ["| Segment | max `|a_ref|` [m/s^2] | max error [m] | RMS error [m] | max tilt [deg] | max yaw [deg] |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['segment']} | {r['a_ref']:.2f} | {e(r['max_err'])} | {e(r['rms_err'])} | "
                     f"{r['tilt']:.1f} | {r['yaw']:.1f} |")
    return "\n".join(lines)


if __name__ == "__main__":
    print(markdown(tracking_rows()))
