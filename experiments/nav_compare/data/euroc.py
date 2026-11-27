"""EuRoC MAV (ASL format) -> common recording, so that the filters of compare/ run on real IMU data.

Files read (inside the sequence folder, e.g. ~/data/euroc/V1_01_easy):
  mav0/imu0/data.csv                       t [ns], gyro x y z [rad/s], accel x y z [m/s^2] (200 Hz)
  mav0/state_groundtruth_estimate0/data.csv t [ns], p, q (w x y z), v, b_w, b_a (200 Hz)
The ground-truth estimate is given for the IMU frame (body = IMU) in a gravity-aligned world frame
with z up, so it already follows the conventions of docs/common_interface.md (Hamilton quaternion
[w, x, y, z], body to world). It is interpolated to the IMU time stamps (the two clocks are the
same; quaternions are interpolated linearly and normalised, which is accurate at 200 Hz).

EuRoC has no GNSS. Position fixes are simulated from the ground truth (rate and std below); an
outage can be cut out. Noise values given to the filters: the shipped imu0/sensor.yaml.

The three sequences kept for the final test (V1_03_difficult, MH_05_difficult, V2_03_difficult)
are refused.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

LOCKED = ("V1_03", "MH_05", "V2_03")
# imu0/sensor.yaml of EuRoC (ADIS16448)
SENSOR_YAML = {"gyro_density": 1.6968e-4, "accel_density": 2.0e-3,
               "gyro_bias_rw": 1.9393e-5, "accel_bias_rw": 3.0e-3}  # fmt: skip


def _check_unlocked(folder: Path):
    if any(name in folder.name for name in LOCKED) or any(
        name in str(folder) for name in LOCKED
    ):
        raise PermissionError(
            f"{folder.name} is a locked test sequence and must not be opened"
        )


def read_csv(path):
    """Time stamps [ns] as int64 (float64 would lose ~256 ns at 1.4e18) and the data columns."""
    t_ns = np.loadtxt(
        path, delimiter=",", comments="#", usecols=0, dtype=np.int64, ndmin=1
    )
    data = np.loadtxt(path, delimiter=",", comments="#", ndmin=2)
    return t_ns, data


def euroc_recording(folder, pos_rate_hz=10.0, pos_std_m=0.1, outage=None, t_max_s=None,
                    seed=0, init_bias="truth") -> dict:  # fmt: skip
    """Recording dict (same keys as data/recording.py) of one EuRoC sequence.

    outage: (t_start, t_end) in seconds from the start, without position fixes.
    init_bias: "truth" starts the filters from the ground-truth bias estimate at t0 (P0 small),
    "zero" from zero bias (P0 large enough for the EuRoC biases).
    """
    folder = Path(folder).expanduser()
    _check_unlocked(folder)
    ti_ns, imu = read_csv(folder / "mav0" / "imu0" / "data.csv")
    tg_ns, gt = read_csv(folder / "mav0" / "state_groundtruth_estimate0" / "data.csv")
    ref = min(ti_ns[0], tg_ns[0])
    t_imu = (ti_ns - ref) * 1e-9
    t_gt = (tg_ns - ref) * 1e-9
    keep = (t_imu >= t_gt[0]) & (t_imu <= t_gt[-1])
    t_imu, imu = t_imu[keep], imu[keep]
    t0 = t_imu[0]
    if t_max_s is not None:
        m = t_imu - t0 <= t_max_s
        t_imu, imu = t_imu[m], imu[m]

    def interp(cols):
        return np.stack([np.interp(t_imu, t_gt, gt[:, c]) for c in cols], axis=1)

    p, q, v = interp((1, 2, 3)), interp((4, 5, 6, 7)), interp((8, 9, 10))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    bg, ba = interp((11, 12, 13)), interp((14, 15, 16))

    rng = np.random.default_rng(np.random.SeedSequence([seed, 9191]))
    t = t_imu - t0
    pos_t = np.arange(0.0, t[-1], 1.0 / pos_rate_hz)
    pos_p = np.stack([np.interp(pos_t, t, p[:, i]) for i in range(3)], axis=1)
    pos_z = pos_p + rng.normal(0.0, pos_std_m, pos_p.shape)
    valid = np.ones(len(pos_t), bool)
    events = []
    if outage is not None:
        a, b = outage
        valid &= ~((pos_t >= a) & (pos_t < b))
        events.append(
            {"type": "outage", "sensor": "pos", "t_start": float(a), "t_end": float(b)}
        )
    meta = {
        "version": "v1", "id": folder.name, "platform": "uav", "source": "euroc",
        "imu_rate_hz": float(1.0 / np.median(np.diff(t))), "true_noise": dict(SENSOR_YAML),
        "scenario": {"imu_grade": "euroc"}, "pos_rate_hz": pos_rate_hz, "pos_std_m": pos_std_m,
        "init_bias": init_bias,
        "init_bias_std": (2e-3, 5e-2) if init_bias == "truth" else (0.1, 0.3),
        "frames": {"world": "EuRoC ground-truth frame (gravity aligned, z up)", "body": "IMU (S)",
                   "quaternion": "Hamilton [w,x,y,z], b->w"},
        "position_fixes": "simulated from the ground truth (EuRoC has no GNSS)",
    }  # fmt: skip
    return {
        "imu_t": t, "imu_gyro": imu[:, 1:4], "imu_accel": imu[:, 4:7],
        "imu_gyro_clean": imu[:, 1:4] - bg, "imu_accel_clean": imu[:, 4:7] - ba,
        "truth_t": t, "truth_p": p, "truth_v": v, "truth_q": q, "truth_bg": bg, "truth_ba": ba,
        "pos_t": pos_t, "pos_z": pos_z, "pos_R": np.broadcast_to(pos_std_m**2 * np.eye(3),
                                                                 (len(pos_t), 3, 3)).copy(),
        "pos_valid": valid, "meta": meta, "events": events,
    }  # fmt: skip
