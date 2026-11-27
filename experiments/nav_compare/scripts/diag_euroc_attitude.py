"""Where does the ~5 deg attitude error on EuRoC come from: the filters, or the IMU / ground truth?

Three checks, printed as a short report:
  1. Gravity: at rest the accelerometer (minus the reference bias), rotated with the ground-truth
     attitude, must point along +z; the angle between them is the tilt mismatch of the reference.
  2. Gyro vs ground truth: integrate the gyro (minus the reference bias) over 5 s windows from the
     ground-truth attitude; the rotation error at the end of each window (median over windows) shows
     whether gyro and ground-truth attitude agree (frames and time stamps).
  3. ESKF attitude error split into roll, pitch and yaw (body frame), and its value over time.

    python -m experiments.nav_compare.scripts.diag_euroc_attitude ~/data/euroc/V1_01_easy
"""

from __future__ import annotations

import sys

import numpy as np

from experiments.nav_compare.compare import kinematics as kin
from experiments.nav_compare.compare.runner import errors, run_filter
from experiments.nav_compare.data.euroc import euroc_recording


def main(argv=None):
    folder = (argv or sys.argv[1:])[0]
    rec = euroc_recording(folder)
    t, gyro, acc = rec["imu_t"], rec["imu_gyro"], rec["imu_accel"]
    C = kin.quat_to_rot_b(rec["truth_q"])
    bg, ba = rec["truth_bg"], rec["truth_ba"]

    # 1. gravity direction at rest (speed < 0.02 m/s)
    still = np.linalg.norm(rec["truth_v"], axis=1) < 0.02
    if still.sum() > 50:
        f_w = np.einsum("nij,nj->ni", C[still], acc[still] - ba[still]).mean(axis=0)
        tilt = np.degrees(np.arccos(f_w[2] / np.linalg.norm(f_w)))
        print(f"1. at rest ({still.sum()} samples): |f| = {np.linalg.norm(f_w):.3f} m/s^2, "
              f"tilt between C*(a_m - b_a) and +z = {tilt:.2f} deg")  # fmt: skip
    else:
        print("1. no rest period found")

    # 2. gyro integration vs ground-truth attitude, 5 s windows
    n_win = int(5.0 / np.median(np.diff(t)))
    errs = []
    for s in range(0, len(t) - n_win, n_win):
        R = C[s].copy()
        for k in range(s, s + n_win):
            R = R @ kin.exp_b((gyro[k] - bg[k]) * (t[k + 1] - t[k]))
        errs.append(np.degrees(np.linalg.norm(kin.log_b(R.T @ C[s + n_win]))))
    print(f"2. gyro integrated over 5 s vs ground truth: median {np.median(errs):.2f} deg, "
          f"max {np.max(errs):.2f} deg ({len(errs)} windows)")  # fmt: skip

    # 3. ESKF attitude error per axis
    E = errors(rec, run_filter(rec, "ESKF"))
    th = np.degrees(E["e_th"])
    print("3. ESKF attitude error [deg], body frame (roll x, pitch y, yaw z):")
    print(f"   mean  {th.mean(axis=0).round(2)}   std {th.std(axis=0).round(2)}")
    for frac in (0.1, 0.25, 0.5, 0.75, 1.0):
        i = min(int(frac * len(th)) - 1, len(th) - 1)
        print(f"   t = {E['t'][i]:6.1f} s: {th[i].round(2)}")
    print("Reading: 1 and 2 small and 3 large -> the filters; 1 or 2 large -> frames / time stamps "
          "of the data (the filter result is then not the cause).")  # fmt: skip


if __name__ == "__main__":
    main()
