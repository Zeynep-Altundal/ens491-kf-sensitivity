"""Configurable synthetic scenarios for comparing filters under controlled conditions.

One `Scenario` describes the trajectory, the IMU error model, the timing defects, the
position-update (GPS-like) defects and the initial-state error. `generate(scenario, seed)`
returns the same `SyntheticSequence` layout as `experiments.nav_compare.data.synthetic.make_synthetic`, so every
filter and runner that accepts that object accepts this one.

Design rules
  - Truth first: the trajectory (position, attitude) is built, then specific force and body
    rates are derived from it, then errors are added to the *measurements* only.
  - Independent random streams (SeedSequence.spawn): changing one factor (for example the
    vibration level) does not reshuffle the noise of the other factors. Scenarios that differ
    in one setting are therefore a paired comparison for the same seed.
  - IMU grades: "euroc" (default) uses the densities of synthetic/ (EuRoC ADIS16448 level); the
    other presets are illustrative orders of magnitude, not values from a datasheet.
  - Same conventions as docs/common_interface.md (ENU, z up, Hamilton [w,x,y,z], R_wb, g_w=[0,0,-9.81]).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from experiments.nav_compare.data.sequence import (
    ImuSequence,
    Reference,
    simulate_position_updates,
)
from experiments.nav_compare.data.synthetic import (
    GRAVITY_W,
    SyntheticSequence,
    body_rates,
    hash_arrays,
    smoothstep,
)

# (gyro noise density rad/s/sqrt(Hz), accel noise density m/s^2/sqrt(Hz),
#  gyro bias std rad/s, accel bias std m/s^2,
#  gyro bias random walk rad/s/sqrt(s), accel bias random walk m/s^2/sqrt(s))
IMU_GRADES = {
    # Default since week 3: same white-noise and random-walk densities as synthetic/ScenarioConfig,
    # which equal (accel) or are within 18 % (gyro) of the EuRoC imu0/sensor.yaml (ADIS16448);
    # turn-on bias std of the order of the fixed turn-on biases of synthetic/ (per axis).
    "euroc": (2.0e-4, 2.0e-3, 1.5e-3, 4.0e-2, 2.0e-5, 3.0e-3),
    "low": (
        1e-5,
        1e-4,
        5e-5,
        2e-3,
        1e-7,
        1e-6,
    ),  # low-noise, tactical-like (illustrative)
    "medium": (1e-4, 1e-3, 5e-4, 1e-2, 1e-6, 1e-5),
    "high": (5e-4, 5e-3, 3e-3, 5e-2, 1e-5, 1e-4),  # consumer MEMS-like (illustrative)
}


@dataclass(frozen=True)
class Scenario:
    """All knobs. The defaults give a mild, realistic baseline; every field can be swept."""

    name: str = "baseline"
    platform: str = "ugv"  # "ugv" | "uav"
    duration_s: float | None = None  # ugv 120, uav 60 when None
    imu_hz: float | None = None  # ugv 100, uav 200 when None

    # --- trajectory, UGV ---
    speed_m_s: float = 8.0
    stops: tuple = ((40.0, 50.0),)  # (start_s, end_s) full stops, ZUPT opportunities
    turn_rate_rad_s: float = 0.15  # yaw-rate amplitude
    turn_period_s: float = 30.0
    slip_rad: float = (
        0.0  # side-slip angle amplitude: velocity not along body x (breaks NHC)
    )
    grade_rad: float = (
        0.0  # road slope amplitude (pitch), breaks "vertical velocity is zero"
    )
    grade_period_s: float = 40.0
    # --- trajectory, UAV ---
    uav_amp_xy_m: tuple = (2.0, 1.5)
    uav_amp_z_m: float = 0.3
    uav_period_s: float = 20.0  # shorter period = more aggressive (accel ~ 1/period^2)
    uav_yaw_amp_rad: float = 0.3

    # --- IMU error model ---
    imu_grade: str = "euroc"
    noise_scale: float = (
        1.0  # multiplies all four noise terms of the grade (white + walk)
    )
    bias_scale: float = 1.0  # multiplies the initial bias std
    bias_random_walk: bool = True  # False: constant bias only
    scale_factor_std: float = 0.0  # per-axis, dimensionless (0.001 = 1000 ppm)
    misalignment_std_rad: float = 0.0  # small fixed rotation of the sensor axes
    quantization_gyro: float = 0.0  # rad/s step, 0 = off
    quantization_accel: float = 0.0  # m/s^2 step, 0 = off
    saturation_gyro: float = 0.0  # rad/s, 0 = off
    saturation_accel: float = 0.0  # m/s^2, 0 = off
    vibration_accel: float = (
        0.0  # m/s^2 amplitude, sinusoid on every axis (rotor / engine)
    )
    vibration_gyro: float = 0.0  # rad/s amplitude
    vibration_hz: float = 0.0  # 0 = platform default (uav 80 Hz, ugv 20 Hz)
    gravity_error_m_s2: float = (
        0.0  # true local g differs from the 9.81 the filter assumes
    )

    # --- timing ---
    timestamp_jitter_s: float = 0.0  # std, limited to 0.4 sample periods
    dropout_bursts_per_min: float = 0.0
    dropout_burst_s: float = 0.0

    # --- position updates (GPS-like) ---
    obs_rate_hz: float = 1.0
    obs_std_m: float = 0.5  # true noise
    obs_assumed_factor: float = (
        1.0  # filter is told std * factor (measurement-side mismatch)
    )
    obs_outlier_prob: float = 0.0
    obs_outlier_std_m: float = 20.0
    outage_s: tuple | None = (
        None  # (start, end); None = platform default, () = no outage
    )

    # --- initial state given to the filter (non-oracle when > 0) ---
    init_pos_std_m: float = 0.0
    init_vel_std_m_s: float = 0.0
    init_att_std_rad: float = 0.0
    init_bias_known: bool = True  # False: filter starts with zero bias estimate

    extra: dict = field(default_factory=dict)

    def replace(self, **kw):
        return dataclasses.replace(self, **kw)


def _stream(seed, n=8):
    return [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(n)]


# ----------------------------------------------------------------------------------- truth


def _ugv_truth(sc, t):
    dt = np.diff(t, prepend=t[0])
    stop = np.ones_like(t)
    for s, e in sc.stops:
        stop *= np.clip(
            1.0 - smoothstep((t - (s - 2.0)) / 2.0) + smoothstep((t - e) / 2.0), 0, 1
        )
    speed = sc.speed_m_s * smoothstep(t / 5.0) * stop
    moving = speed > 0.1
    yaw_rate = sc.turn_rate_rad_s * np.sin(2 * np.pi * t / sc.turn_period_s) * moving
    yaw = np.cumsum(yaw_rate * dt)
    pitch = (
        -sc.grade_rad * np.sin(2 * np.pi * t / sc.grade_period_s) * moving
    )  # up = nose up
    beta = sc.slip_rad * np.sin(2 * np.pi * t / sc.turn_period_s) * moving
    euler = np.stack([yaw, pitch, np.zeros_like(t)], axis=1)
    R_wb = Rotation.from_euler("ZYX", euler)
    v_b = np.stack(
        [speed * np.cos(beta), speed * np.sin(beta), np.zeros_like(t)], axis=1
    )
    v_w = R_wb.apply(v_b)
    p_w = np.cumsum(v_w * dt[:, None], axis=0)
    return p_w, R_wb


def _uav_attitude(t, a_w, yaw_amp):
    thrust = a_w - GRAVITY_W
    z_b = thrust / np.linalg.norm(thrust, axis=1, keepdims=True)
    yaw = yaw_amp * np.sin(2 * np.pi * t / 25.0)
    x_c = np.stack([np.cos(yaw), np.sin(yaw), np.zeros_like(yaw)], axis=1)
    y_b = np.cross(z_b, x_c)
    y_b /= np.linalg.norm(y_b, axis=1, keepdims=True)
    x_b = np.cross(y_b, z_b)
    return Rotation.from_matrix(np.stack([x_b, y_b, z_b], axis=2))


def _smootherstep(x):
    """Quintic ramp: zero first AND second derivative at both ends (no acceleration jump)."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * x * (x * (6.0 * x - 15.0) + 10.0)


def _uav_truth(sc, t):
    # smoothstep (as in synthetic.py) leaves a jump in acceleration at the ramp start, which
    # makes the thrust direction, and so the attitude, jump (~3.5 rad/s spike at t = 5 s).
    ramp = _smootherstep((t - 5.0) / 5.0)
    w = 2 * np.pi / sc.uav_period_s
    ax, ay = sc.uav_amp_xy_m
    p_w = np.stack(
        [
            ax * np.sin(w * t) * ramp,
            ay * np.sin(2 * w * t) * ramp,
            1.5 + sc.uav_amp_z_m * np.sin(0.5 * w * t) * ramp,
        ],
        axis=1,
    )
    v_w = np.gradient(p_w, t, axis=0)
    a_w = np.gradient(v_w, t, axis=0)
    return p_w, _uav_attitude(t, a_w, sc.uav_yaw_amp_rad)


# ------------------------------------------------------------------------------ IMU errors


def _corrupt_imu(sc, t, gyro_true, accel_true, rng):
    nd_g, nd_a, b_g, b_a, rw_g, rw_a = IMU_GRADES[sc.imu_grade]
    k = sc.noise_scale
    fs = 1.0 / float(np.median(np.diff(t)))
    dt = 1.0 / fs
    n = len(t)
    bg0 = rng["bias"].normal(0.0, b_g * sc.bias_scale, 3)
    ba0 = rng["bias"].normal(0.0, b_a * sc.bias_scale, 3)
    if sc.bias_random_walk:
        bg = bg0 + np.cumsum(
            rng["walk"].normal(0, rw_g * k * np.sqrt(dt), (n, 3)), axis=0
        )
        ba = ba0 + np.cumsum(
            rng["walk"].normal(0, rw_a * k * np.sqrt(dt), (n, 3)), axis=0
        )
    else:
        bg, ba = np.tile(bg0, (n, 1)), np.tile(ba0, (n, 1))

    info = {"gyro_bias": bg, "accel_bias": ba}

    def sensor(true, bias, nd, sf_std, mis_std, vib, quant, sat, name):
        sf = rng["sf"].normal(0.0, sf_std, 3) if sf_std > 0 else np.zeros(3)
        mis = rng["mis"].normal(0.0, mis_std, 3) if mis_std > 0 else np.zeros(3)
        info[f"{name}_scale_factor"] = sf
        info[f"{name}_misalignment_rad"] = mis
        m = Rotation.from_rotvec(mis).as_matrix() @ np.diag(1.0 + sf)
        meas = (
            true @ m.T
            + bias
            + rng["white"].normal(0.0, k * nd * np.sqrt(fs), true.shape)
        )
        if vib > 0:
            f = sc.vibration_hz or (80.0 if sc.platform == "uav" else 20.0)
            phase = rng["vib"].uniform(0, 2 * np.pi, 3)
            meas = meas + vib * np.sin(2 * np.pi * f * t[:, None] + phase)
        if quant > 0:
            meas = np.round(meas / quant) * quant
        if sat > 0:
            meas = np.clip(meas, -sat, sat)
        return meas

    gyro = sensor(
        gyro_true,
        bg,
        nd_g,
        sc.scale_factor_std,
        sc.misalignment_std_rad,
        sc.vibration_gyro,
        sc.quantization_gyro,
        sc.saturation_gyro,
        "gyro",
    )
    accel = sensor(
        accel_true,
        ba,
        nd_a,
        sc.scale_factor_std,
        sc.misalignment_std_rad,
        sc.vibration_accel,
        sc.quantization_accel,
        sc.saturation_accel,
        "accel",
    )
    noise = {  # what an ideal filter would use for Q (same names as scripts/run_eskf_kitti.py)
        "gyro_density": k * nd_g,
        "accel_density": k * nd_a,
        "gyro_bias_rw": k * rw_g if sc.bias_random_walk else 0.0,
        "accel_bias_rw": k * rw_a if sc.bias_random_walk else 0.0,
    }
    return gyro, accel, bg0, ba0, noise, info


# ------------------------------------------------------------------------------- timing


def _apply_timing(sc, t, gyro, accel, ref_arrays, rng):
    """Jitter the IMU timestamps and delete burst dropouts (reference rows are deleted too)."""
    dt = float(np.median(np.diff(t)))
    t_imu = t.copy()
    if sc.timestamp_jitter_s > 0:
        j = np.clip(
            rng["jitter"].normal(0, sc.timestamp_jitter_s, t.shape), -0.4 * dt, 0.4 * dt
        )
        t_imu = t + j
    keep = np.ones(len(t), dtype=bool)
    if sc.dropout_bursts_per_min > 0 and sc.dropout_burst_s > 0:
        n_bursts = rng["drop"].poisson(sc.dropout_bursts_per_min * t[-1] / 60.0)
        length = max(1, int(round(sc.dropout_burst_s / dt)))
        for s in rng["drop"].integers(1, max(2, len(t) - length - 1), n_bursts):
            keep[s : s + length] = False
    return t_imu[keep], gyro[keep], accel[keep], [a[keep] for a in ref_arrays], keep


# ----------------------------------------------------------------------------- generator


def generate(sc: Scenario, seed: int = 0) -> SyntheticSequence:
    """Build one sequence for scenario `sc` and random seed `seed` (deterministic)."""
    if sc.platform not in ("ugv", "uav"):
        raise ValueError(f"unknown platform {sc.platform}")
    if sc.imu_grade not in IMU_GRADES:
        raise ValueError(f"unknown imu_grade {sc.imu_grade}")
    ugv = sc.platform == "ugv"
    duration = sc.duration_s or (120.0 if ugv else 60.0)
    hz = sc.imu_hz or (100.0 if ugv else 200.0)
    t = np.arange(0.0, duration, 1.0 / hz)
    names = [
        "bias",
        "walk",
        "white",
        "sf",
        "mis",
        "vib",
        "jitter",
        "drop",
        "obs",
        "init",
        "out",
    ]
    rng = dict(zip(names, _stream(seed, len(names)), strict=True))

    p_w, R_wb = _ugv_truth(sc, t) if ugv else _uav_truth(sc, t)
    v_w = np.gradient(p_w, t, axis=0)
    a_w = np.gradient(v_w, t, axis=0)
    g_true = GRAVITY_W + np.array([0.0, 0.0, -sc.gravity_error_m_s2])
    gyro_true = body_rates(R_wb, 1.0 / hz)
    accel_true = R_wb.inv().apply(a_w - g_true)

    gyro, accel, bg0, ba0, noise, info = _corrupt_imu(sc, t, gyro_true, accel_true, rng)
    q_wb = np.roll(R_wb.as_quat(), 1, axis=1)
    valid = np.ones(len(t), dtype=bool)
    t_imu, gyro, accel, (t_ref, p_ref, v_ref, q_ref, ok), keep = _apply_timing(
        sc, t, gyro, accel, [t, p_w, v_w, q_wb, valid], rng
    )
    imu = ImuSequence(t_imu, gyro, accel)
    reference = Reference(t_ref, p_ref, ok, v_ref, q_ref)

    # GPS-like position updates (declared generator), then defects on top.
    outage = (
        sc.outage_s
        if sc.outage_s is not None
        else ((60.0, 90.0) if ugv else (30.0, 40.0))
    )
    gen = simulate_position_updates(
        reference,
        sc.obs_rate_hz,
        sc.obs_std_m,
        seed=seed,
        config_hash=f"scenario:{sc.name}",
    )
    r_told = np.eye(3) * (sc.obs_std_m * sc.obs_assumed_factor) ** 2
    obs = []
    for o in gen:
        if outage and outage[0] <= o.t_s < outage[1]:
            continue
        z = o.z
        outlier = sc.obs_outlier_prob > 0 and rng["obs"].random() < sc.obs_outlier_prob
        if outlier:
            z = z + rng["obs"].normal(0.0, sc.obs_outlier_std_m, 3)
        prov = {**o.provenance, "outlier": bool(outlier)}
        obs.append(dataclasses.replace(o, z=z, r_meas=r_told, provenance=prov))
    hidden = len(gen) - len(obs)

    truth = {
        "scenario": dataclasses.asdict(sc),
        "seed": seed,
        "imu_hz": hz,
        "gyro_bias_true": bg0,
        "accel_bias_true": ba0,
        "noise_params": noise,
        "dropped_samples": int((~keep).sum()),
        # time series of the true errors, aligned with the IMU samples that were kept
        "gyro_bias_series": info["gyro_bias"][keep],
        "accel_bias_series": info["accel_bias"][keep],
        "gyro_clean": gyro_true[keep],
        "accel_clean": accel_true[keep],
        "gyro_scale_factor": info["gyro_scale_factor"],
        "accel_scale_factor": info["accel_scale_factor"],
        "gyro_misalignment_rad": info["gyro_misalignment_rad"],
        "accel_misalignment_rad": info["accel_misalignment_rad"],
        "events": [],
    }
    return SyntheticSequence(
        id=f"{sc.platform}_{sc.name}_s{seed:02d}",
        platform=sc.platform,
        imu=imu,
        reference=reference,
        observations=obs,
        outage_s=tuple(outage),
        hidden_observations=hidden,
        truth=truth,
        source_hash=hash_arrays(t_imu, gyro, accel, p_ref),
        split_id="synthetic",
    )


def initial_state_for(
    sc: Scenario, seq: SyntheticSequence, seed: int = 0, index: int = 0
):
    """Initial state for a filter: oracle plus the perturbation asked for by the scenario.

    All init_* = 0 and init_bias_known=True gives the oracle state with true biases;
    otherwise the init is labelled 'operational'.
    """
    from experiments.nav_compare.baselines.strapdown import (
        InitialState,
        oracle_initial_state,
    )

    base = oracle_initial_state(seq.reference, index)
    rng = _stream(seed, 11)[9]
    p = base.p + rng.normal(0.0, sc.init_pos_std_m, 3) if sc.init_pos_std_m else base.p
    v = (
        base.v + rng.normal(0.0, sc.init_vel_std_m_s, 3)
        if sc.init_vel_std_m_s
        else base.v
    )
    q = base.q
    if sc.init_att_std_rad:
        d = Rotation.from_rotvec(rng.normal(0.0, sc.init_att_std_rad, 3))
        q = np.roll((Rotation.from_quat(np.roll(base.q, -1)) * d).as_quat(), 1)
    known = sc.init_bias_known
    bg = seq.truth["gyro_bias_true"] if known else np.zeros(3)
    ba = seq.truth["accel_bias_true"] if known else np.zeros(3)
    perturbed = bool(
        sc.init_pos_std_m or sc.init_vel_std_m_s or sc.init_att_std_rad or not known
    )
    return InitialState(np.array(p), np.array(v), np.array(q), bg, ba,
                        "operational" if perturbed else "oracle")  # fmt: skip


# --------------------------------------------------------------------------------- grids


def scenario_grid(platform: str) -> dict:
    """One-factor-at-a-time sweep around the baseline, plus platform-specific cases.

    Returns {name: Scenario}. Every scenario differs from 'baseline' in the factor named.
    """
    base = Scenario(name="baseline", platform=platform)
    g: dict[str, Scenario] = {"baseline": base}

    def add(name, **kw):
        g[name] = base.replace(name=name, **kw)

    for grade in ("low", "medium", "high"):
        add(f"grade_{grade}", imu_grade=grade)
    for s in (0.3, 3.0):
        add(f"noise_x{s:g}", noise_scale=s)
    add("bias_constant", bias_random_walk=False)
    for s in (0.3, 3.0):
        add(f"bias_x{s:g}", bias_scale=s)
    add("scale_factor_1e-3", scale_factor_std=1e-3)
    add("misalign_0.5deg", misalignment_std_rad=np.deg2rad(0.5))
    add("quantization", quantization_gyro=1e-3, quantization_accel=1e-2)
    if platform == "ugv":
        add("saturation", saturation_gyro=0.1)  # clips the yaw rate peaks (0.15 rad/s)
    else:
        add("saturation", saturation_gyro=0.15, saturation_accel=10.0)
    for a in (0.1, 0.5):
        add(f"vibration_{a:g}", vibration_accel=a, vibration_gyro=a / 20)
    add("gravity_err_0.02", gravity_error_m_s2=0.02)
    add(
        "jitter_20pct",
        timestamp_jitter_s=0.2
        / (base.imu_hz or (100.0 if platform == "ugv" else 200.0)),
    )
    add("dropouts", dropout_bursts_per_min=4.0, dropout_burst_s=0.1)
    for hz in (50.0, 400.0):
        add(f"imu_{hz:g}hz", imu_hz=hz)
    add("obs_noise_x4", obs_std_m=2.0)
    add("obs_mismatch_x0.25", obs_assumed_factor=0.25)  # filter trusts GPS 4x too much
    add("obs_mismatch_x4", obs_assumed_factor=4.0)  # filter trusts GPS 4x too little
    add("obs_outliers_5pct", obs_outlier_prob=0.05)
    add("obs_rate_5hz", obs_rate_hz=5.0)
    full = (10.0, 1e9)  # GPS lost from 10 s to the end
    d = 30.0 if platform == "ugv" else 15.0
    for dur in (10.0, 30.0, 60.0):
        add(f"outage_{dur:g}s", outage_s=(d, d + dur))
    add("outage_full", outage_s=full)
    add("init_operational", init_pos_std_m=1.0, init_vel_std_m_s=0.1, init_att_std_rad=0.02,
        init_bias_known=False)  # fmt: skip
    if platform == "ugv":
        add("ugv_straight", turn_rate_rad_s=0.0, stops=())
        add(
            "ugv_stop_go",
            stops=((20.0, 28.0), (40.0, 50.0), (70.0, 80.0), (100.0, 108.0)),
        )
        add("ugv_slow", speed_m_s=2.0)
        add("ugv_fast", speed_m_s=15.0)
        add("ugv_slalom", turn_rate_rad_s=0.4, turn_period_s=10.0)
        for s in (2.0, 5.0):
            add(f"ugv_slip_{s:g}deg", slip_rad=np.deg2rad(s))
        add("ugv_grade_5deg", grade_rad=np.deg2rad(5.0))
        add("ugv_bumpy", vibration_accel=1.0, vibration_hz=5.0)
    else:
        add("uav_hover", uav_amp_xy_m=(0.0, 0.0), uav_amp_z_m=0.0, uav_yaw_amp_rad=0.0)
        add("uav_gentle", uav_period_s=40.0)
        add("uav_aggressive", uav_period_s=10.0)
        add("uav_very_aggressive", uav_period_s=6.0, uav_amp_xy_m=(3.0, 2.0))
        add("uav_big_yaw", uav_yaw_amp_rad=1.2)
        add(
            "uav_rotor_vibration",
            vibration_accel=0.5,
            vibration_gyro=0.02,
            vibration_hz=120.0,
        )
    return g
