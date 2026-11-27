"""Tracking table of the closed-loop flight (position error, tilt and yaw error per segment).

    python -m rviz_drone.tracking [mission|figure_eights]      (from simulation/rviz_drone/)

prints the Markdown table that is recorded in ``../README.md``.
"""

from __future__ import annotations

import sys

import numpy as np

from . import reference as references
from .flight import simulate_truth
from synthetic.rotation import quat_to_rotmat

LABELS = {"takeoff": "take-off", "hover": "hover (all)", "hover_turn": "hover with turn",
          "figure_eight": "figure-eight", "straight": "straight leg", "turn": "coordinated turn"}


def tracking_rows(trajectory: str | None = None, truth: dict[str, np.ndarray] | None = None
                  ) -> list[dict[str, float | str]]:
    traj = references.get(trajectory)
    truth = truth if truth is not None else simulate_truth(trajectory=trajectory)
    t, p, p_ref, q, yaw_ref = truth["t"], truth["p"], truth["p_ref"], truth["q"], truth["yaw_ref"]
    names = np.array([traj.segment_at(x).name for x in t])
    err = np.linalg.norm(p - p_ref, axis=1)
    a_ref = np.array([np.linalg.norm(traj.reference(x)[2]) for x in t])
    r = np.array([quat_to_rotmat(x) for x in q])
    tilt = np.degrees(np.arccos(np.clip(r[:, 2, 2], -1.0, 1.0)))
    yaw = np.arctan2(r[:, 1, 0], r[:, 0, 0])
    yaw_err = np.degrees(np.abs((yaw - yaw_ref + np.pi) % (2 * np.pi) - np.pi))
    rows = []
    for name in dict.fromkeys(s.name for s in traj.segments):
        if name == "ground":
            continue
        m = names == name
        rows.append({"segment": LABELS.get(name, name), "a_ref": a_ref[m].max(), "max_err": err[m].max(),
                     "rms_err": float(np.sqrt(np.mean(err[m] ** 2))), "tilt": tilt[m].max(),
                     "yaw_err": yaw_err[m].max()})
    return rows


def markdown(rows: list[dict[str, float | str]]) -> str:
    def e(x: float) -> str:
        return "< 0.001" if x < 0.001 else f"{x:.3f}"

    lines = ["| Segment | max `|a_ref|` [m/s^2] | max error [m] | RMS error [m] | max tilt [deg] | max yaw error [deg] |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['segment']} | {r['a_ref']:.2f} | {e(r['max_err'])} | {e(r['rms_err'])} | "
                     f"{r['tilt']:.1f} | {r['yaw_err']:.1f} |")
    return "\n".join(lines)


if __name__ == "__main__":
    print(markdown(tracking_rows(sys.argv[1] if len(sys.argv) > 1 else None)))
