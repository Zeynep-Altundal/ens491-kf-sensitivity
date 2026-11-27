"""Run one filter on one recording, write the common run log, and evaluate it.

A recording is the dict of nkf/data/recording.py (from `load_recording` or `to_recording`).
The filter sees only imu_* and pos_* and the initial state; truth_* is used for the initial state
and for evaluation only (and, for KF-IMU, for the given orientation).
"""

from __future__ import annotations

import numpy as np

from experiments.nav_compare.compare import kinematics as kin
from experiments.nav_compare.compare.filters import (
    E_X,
    FILTERS,
    LAT_VERT,
    Init,
    NoiseSpec,
)
from experiments.nav_compare.data.odometry import yaw_rate_from_odometry, yaw_rate_std
from experiments.nav_compare.data.scenarios import IMU_GRADES

# Initial uncertainty used for every filter (same blocks for all)
INIT_STD = {"p": 0.5, "v": 0.2, "att": np.deg2rad(1.0)}
ODOM_MIN_SPEED = 1.0  # m/s, no yaw-rate update below this speed
YAW_RATE_STD_FLOOR = 0.005  # rad/s, model error of the bicycle model
SIGMA_CV = {
    "ugv": 1.0,
    "uav": 1.0,
}  # KF-CV tuning value, set by tune_sigma_cv on a separate seed


def make_init(rec, init_att_err=None, init_seed=0) -> Init:
    """Initial state from the truth at t0; biases start at 0 (unknown to the filter).

    init_att_err [rad]: start with an attitude error of this size about a random axis (same axis
    for every filter for a given init_seed), and tell the filter std_att = init_att_err.
    """
    meta = rec["meta"]
    grade = meta.get("scenario", {}).get("imu_grade", "euroc")
    _, _, bg_std, ba_std, _, _ = IMU_GRADES[grade]
    bias_scale = meta.get("scenario", {}).get("bias_scale", 1.0)
    z = np.zeros(3)
    q0 = rec["truth_q"][0].copy()
    std_att = INIT_STD["att"]
    if init_att_err:
        axis = np.random.default_rng(np.random.SeedSequence([init_seed, 777])).normal(
            size=3
        )
        axis /= np.linalg.norm(axis)
        R0 = kin.quat_to_rot_b(q0) @ kin.exp_b(axis * init_att_err)
        from experiments.nav_compare.filters import conventions as c

        q0 = c.quat_from_rotvec(c.so3_log(R0))
        std_att = float(init_att_err)
    bg0, ba0 = z, z.copy()
    bg_std, ba_std = bg_std * bias_scale, ba_std * bias_scale
    if (
        meta.get("init_bias") == "truth"
    ):  # e.g. EuRoC: start from the reference bias estimate
        bg0, ba0 = rec["truth_bg"][0].copy(), rec["truth_ba"][0].copy()
        bg_std, ba_std = meta.get("init_bias_std", (bg_std, ba_std))
    return Init(rec["truth_p"][0].copy(), rec["truth_v"][0].copy(), q0,
                bg0, ba0, INIT_STD["p"], INIT_STD["v"], std_att,
                bg_std, ba_std)  # fmt: skip


def make_noise(rec, sigma_cv=None) -> NoiseSpec:
    tn = rec["meta"]["true_noise"]
    sc = (
        sigma_cv if sigma_cv is not None else SIGMA_CV.get(rec["meta"]["platform"], 1.0)
    )
    return NoiseSpec(tn["gyro_density"], tn["accel_density"], tn["gyro_bias_rw"],
                     tn["accel_bias_rw"], sc)  # fmt: skip


STATIC_SPEED = 0.05  # m/s; wheel speed below this = standing still (ZUPT and ZARU)
ZUPT_STD = 0.02  # m/s


def run_filter(rec, name, log_every=10, sigma_cv=None, nhc_std=None, odometry=False,
               magnetometer=False, static=False, init_att_err=None, init_seed=0):  # fmt: skip
    """Run filter `name` on recording `rec`; return the run log (docs/common_interface.md, 5).

    nhc_std: apply NHC (body lateral and vertical velocity = 0) at every IMU step with this std
    [m/s]; odometry: False, "speed" (wheel speed only) or "both" / True (wheel speed and the
    yaw rate from the steering angle), from the odom_* signals of the recording.
    magnetometer: heading updates from the mag_* signals.
    static: while the wheel speed says the car stands still, apply ZUPT (v = 0) and ZARU
    (gyro = b_g, i.e. gyro bias calibration). UGV only (needs odom_*).
    Filters that cannot use a measurement ignore it (KF-CV: only ZUPT; KF-IMU: no gyro, no heading).
    """
    init = make_init(rec, init_att_err, init_seed)
    f = FILTERS[name](init, make_noise(rec, sigma_cv), attitude=rec["truth_q"])
    t, gyro, accel = rec["imu_t"], rec["imu_gyro"], rec["imu_accel"]
    pos_t, pos_z, pos_R = rec["pos_t"], rec["pos_z"], rec["pos_R"]
    valid = rec["pos_valid"]
    order = np.argsort(pos_t)
    j = 0
    while j < len(order) and pos_t[order[j]] <= t[0] + 1e-9:
        j += 1
    idx = list(range(0, len(t), log_every))
    if idx[-1] != len(t) - 1:
        idx.append(len(t) - 1)
    keep = set(idx)
    L = {k: [] for k in ("k", "p_hat", "v_hat", "q_hat", "bg_hat", "ba_hat",
                         "P_pp", "P_vv", "P_thth", "P_bg", "P_ba")}  # fmt: skip
    U = {"upd_t": [], "innov": [], "S": [], "nis": []}

    def log(k):
        p, v, q, bg, ba = f.state()
        Ppp, Pvv, Ptt, Pbg, Pba = f.cov_blocks()
        for key, val in zip(
            L, (k, p, v, q, bg, ba, Ppp, Pvv, Ptt, Pbg, Pba), strict=True
        ):
            L[key].append(np.array(val, dtype=float, copy=True))

    use_odom = bool(odometry) and "odom_t" in rec
    use_yaw = odometry in (True, "both")
    if use_odom:
        o_t, o_v, o_d = rec["odom_t"], rec["odom_speed"], rec["odom_steer"]
        om = rec["odom_meta"] if isinstance(rec["odom_meta"], dict) else {}
        L_wb, s_v, s_d = om["wheelbase_m"], om["speed_std"], om["steer_std_rad"]
        jo = int(np.searchsorted(o_t, t[0], side="right"))
    nhc_R = None if nhc_std is None else nhc_std**2 * np.eye(2)
    use_mag = magnetometer and "mag_t" in rec
    if use_mag:
        m_t, m_h = rec["mag_t"], rec["mag_heading"]
        m_var = rec["mag_meta"]["std_rad"] ** 2
        jm = int(np.searchsorted(m_t, t[0], side="right"))
    use_static = static and "odom_t" in rec
    if use_static:
        st_t, st_speed = rec["odom_t"], rec["odom_speed"]
        js = int(np.searchsorted(st_t, t[0], side="right"))
        tn = rec["meta"]["true_noise"]
        zaru_var = tn["gyro_density"] ** 2 * float(
            rec["meta"].get("imu_rate_hz") or 100.0
        )

    log(0)
    for k in range(len(t) - 1):
        f.predict(gyro[k], accel[k], t[k + 1] - t[k])
        if nhc_R is not None:
            f.update_body_velocity(LAT_VERT, np.zeros(2), nhc_R)
        if use_odom:
            while jo < len(o_t) and o_t[jo] <= t[k + 1] + 1e-9:
                f.update_body_velocity(E_X, np.array([o_v[jo]]), np.array([[s_v**2]]))
                if (
                    use_yaw and o_v[jo] > ODOM_MIN_SPEED
                ):  # bicycle model invalid at low speed
                    yr = yaw_rate_from_odometry(o_v[jo], o_d[jo], L_wb)
                    var = yaw_rate_std(o_v[jo], o_d[jo], s_v, s_d, L_wb) ** 2
                    f.update_yaw_rate(yr, var + YAW_RATE_STD_FLOOR**2, gyro[k][2])
                jo += 1
        if use_static:
            while js < len(st_t) and st_t[js] <= t[k + 1] + 1e-9:
                if abs(st_speed[js]) < STATIC_SPEED:
                    f.update_zupt(ZUPT_STD**2)
                    f.update_zaru(gyro[k], zaru_var)
                js += 1
        if use_mag:
            while jm < len(m_t) and m_t[jm] <= t[k + 1] + 1e-9:
                f.update_heading(m_h[jm], m_var)
                jm += 1
        while j < len(order) and pos_t[order[j]] <= t[k + 1] + 1e-9:
            m = order[j]
            if valid[m]:
                info = f.update_position(pos_z[m], pos_R[m])
                U["upd_t"].append(pos_t[m])
                U["innov"].append(info.innov)
                U["S"].append(info.S)
                U["nis"].append(info.nis)
            j += 1
        if k + 1 in keep:
            log(k + 1)
    out = {key: np.asarray(val) for key, val in {**L, **U}.items()}
    out["k"] = out["k"].astype(int)
    out["t"] = t[out["k"]]
    out["filter"] = name
    return out


# ------------------------------------------------------------------------------- evaluation


def errors(rec, log):
    """Errors true minus estimate (attitude: Log(R_hat^T R_true)) and NEES per block."""
    k = log["k"]
    e_p = rec["truth_p"][k] - log["p_hat"]
    e_v = rec["truth_v"][k] - log["v_hat"]
    out = {"t": log["t"], "e_p": e_p, "e_v": e_v,
           "err_p": np.linalg.norm(e_p, axis=1)}  # fmt: skip
    out["nees_p"] = np.einsum(
        "ni,ni->n", e_p, np.linalg.solve(log["P_pp"], e_p[..., None])[..., 0]
    )
    out["nees_v"] = np.einsum(
        "ni,ni->n", e_v, np.linalg.solve(log["P_vv"], e_v[..., None])[..., 0]
    )
    if np.all(np.isfinite(log["q_hat"])):
        R_true = kin.quat_to_rot_b(rec["truth_q"][k])
        R_hat = kin.quat_to_rot_b(log["q_hat"])
        e_th = kin.log_b(np.swapaxes(R_hat, -1, -2) @ R_true)
        out["e_th"] = e_th
        out["err_att_deg"] = np.rad2deg(np.linalg.norm(e_th, axis=1))
        sol = np.linalg.solve(log["P_thth"], e_th[..., None])[..., 0]
        out["nees_att"] = np.einsum("ni,ni->n", e_th, sol)
    if np.all(np.isfinite(log["ba_hat"])):
        out["e_ba"] = rec["truth_ba"][k] - log["ba_hat"]
    if np.all(np.isfinite(log["bg_hat"])):
        out["e_bg"] = rec["truth_bg"][k] - log["bg_hat"]
    return out


def event_window(rec):
    """Union of the event windows of the recording, or None."""
    ev = rec["events"]
    if not ev:
        return None
    return min(e["t_start"] for e in ev), max(e["t_end"] for e in ev)


def summarize(rec, log, t_skip=10.0):
    """Scalar metrics of one run. The first `t_skip` s (start-up) are left out."""
    E = errors(rec, log)
    t = E["t"]
    m = t >= t_skip
    s = {"filter": log["filter"],
         "rmse_p": float(np.sqrt(np.mean(E["err_p"][m] ** 2))),
         "max_p": float(E["err_p"][m].max()),
         "final_p": float(E["err_p"][-1]),
         "nees_p": float(np.mean(E["nees_p"][m])),
         "nees_v": float(np.mean(E["nees_v"][m])),
         "nis": float(np.mean(log["nis"])) if len(log["nis"]) else np.nan}  # fmt: skip
    if "nees_att" in E:
        s["rmse_att_deg"] = float(np.sqrt(np.mean(E["err_att_deg"][m] ** 2)))
        s["nees_att"] = float(np.mean(E["nees_att"][m]))
    w = event_window(rec)
    if w is not None and w[1] - w[0] < 0.5 * (t[-1] - t[0]):
        a, b = w
        pre = (t >= a - 10.0) & (t < a)
        level = float(np.median(E["err_p"][pre])) if pre.any() else np.nan
        during = (t >= a) & (t <= b + 10.0)
        s["pre_err"] = level
        s["peak_err"] = float(E["err_p"][during].max())
        s["peak_t"] = float(t[during][np.argmax(E["err_p"][during])])
        after = t > b
        ok = after & (E["err_p"] <= max(1.5 * level, level + 0.2))
        s["recovery_s"] = float(t[ok][0] - b) if ok.any() else np.nan
    return s
