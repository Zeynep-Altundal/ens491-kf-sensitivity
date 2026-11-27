"""Write and read one flight as a single .npz file (common recording format, draft).

Layout follows docs/common_interface_draft.md, section 2. The filter side may read only the
`imu_*` and `pos_*` arrays and the initial state; `truth_*` and the `*_clean` arrays are for
evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, tuple):
        return list(o)
    return str(o)


def to_recording(
    seq,
    source: str = "scenario_gen",
    extra_meta: dict | None = None,
    odometry: dict | None = None,
    magnetometer: dict | None = None,
) -> dict:
    """The recording as an in-memory dict, same keys as the file (meta/events already parsed).

    odometry: output of experiments.nav_compare.data.odometry.simulate_odometry (UGV only), stored as odom_* keys.
    """
    arrays, meta_json, events_json = _arrays(seq, source, extra_meta)
    rec = {**arrays, "meta": json.loads(meta_json), "events": json.loads(events_json)}
    if odometry:
        rec.update(odometry)
    if magnetometer:
        rec.update(magnetometer)
    return rec


def save_recording(
    seq, path, source: str = "scenario_gen", extra_meta: dict | None = None
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays, meta_json, events_json = _arrays(seq, source, extra_meta)
    np.savez_compressed(path, meta=meta_json, events=events_json, **arrays)
    return path


def _arrays(seq, source, extra_meta):
    tr = seq.truth
    obs = seq.observations
    pos_t = np.array([o.t_s for o in obs], dtype=float)
    pos_z = np.array([o.z for o in obs], dtype=float).reshape(-1, 3)
    pos_R = np.array([o.r_meas for o in obs], dtype=float).reshape(-1, 3, 3)
    meta = {
        "version": "v1",
        "imu_sample": "interval",
        "id": seq.id,
        "platform": seq.platform,
        "source": source,
        "imu_rate_hz": float(tr.get("imu_hz", np.nan)),
        "seed": tr.get("seed"),
        "true_noise": tr.get("noise_params", {}),
        "scenario": tr.get("scenario", {}),
        "outage_s": list(seq.outage_s) if seq.outage_s else None,
        "gyro_scale_factor": tr.get("gyro_scale_factor"),
        "accel_scale_factor": tr.get("accel_scale_factor"),
        "gyro_misalignment_rad": tr.get("gyro_misalignment_rad"),
        "accel_misalignment_rad": tr.get("accel_misalignment_rad"),
        "frames": {
            "world": "ENU",
            "body": "IMU frame",
            "quaternion": "Hamilton [w,x,y,z], b->w",
        },
        **(extra_meta or {}),
    }
    ref = seq.reference
    arrays = dict(
        imu_t=seq.imu.t_s,
        imu_gyro=seq.imu.gyro_rad_s,
        imu_accel=seq.imu.accel_m_s2,
        imu_gyro_clean=tr.get("gyro_clean", np.full_like(seq.imu.gyro_rad_s, np.nan)),
        imu_accel_clean=tr.get("accel_clean", np.full_like(seq.imu.accel_m_s2, np.nan)),
        truth_t=ref.t_s,
        truth_p=ref.p_w,
        truth_v=ref.v_w,
        truth_q=ref.q_wb,
        truth_bg=tr.get("gyro_bias_series", np.full_like(ref.p_w, np.nan)),
        truth_ba=tr.get("accel_bias_series", np.full_like(ref.p_w, np.nan)),
        pos_t=pos_t,
        pos_z=pos_z,
        pos_R=pos_R,
        pos_valid=np.ones(len(pos_t), dtype=bool),
    )
    meta_json = json.dumps(meta, default=_json_default)
    events_json = json.dumps(tr.get("events", []), default=_json_default)
    return arrays, meta_json, events_json


def load_recording(path) -> dict:
    with np.load(path, allow_pickle=False) as d:
        out = {k: d[k] for k in d.files}
    out["meta"] = json.loads(str(out["meta"]))
    out["events"] = json.loads(str(out["events"]))
    return out
