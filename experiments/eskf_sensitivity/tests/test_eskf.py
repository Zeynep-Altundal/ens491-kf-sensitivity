"""Validation tests for the error-state Kalman filter.

Run from the repository root with ``python -m pytest -v -s experiments/eskf_sensitivity``
to see the measured values. Thresholds and their justification are stated next
to each test.
"""

import numpy as np
import pytest
from scipy.stats import chi2

from experiments.eskf_sensitivity.eskf import (
    ERROR_DIM,
    NominalState,
    difference,
    error_transition,
    inject,
    nominal_step,
)
from experiments.eskf_sensitivity.run import (
    FilterSettings,
    exact_initial_state,
    initial_state,
    run_filter,
    scenario_config,
    summarize,
)
from synthetic.scenario import ScenarioConfig, generate

BLOCKS = {"p": slice(0, 3), "v": slice(3, 6), "th": slice(6, 9), "bg": slice(9, 12), "ba": slice(12, 15)}

# --- F versus finite differences -------------------------------------------------------------
# F = I + A dt + (A dt)^2 / 2 approximates the Jacobian of the discrete nominal step. Where
# F - I is non-zero, the difference to the exact Jacobian is O(dt |w|) relative (about 1.5 % for
# |w| = 3.7 rad/s, dt = 5 ms), whereas a sign or frame error gives a difference of order 100 %.
F_REL_TOL = 0.05
# Where F - I is zero, the exact Jacobian differs by terms of O(dt^3) or less.
F_ABS_TOL = 1e-6
FD_STEP = 1e-6


def test_F_matches_finite_differences():
    out = generate(ScenarioConfig(profile="aggressive", duration=20.0, enable_white_noise=False,
                                  enable_bias_rw=False))
    k, dt = 2000, 1.0 / 200.0  # t = 10 s
    x = NominalState(out["p_true"][k], out["v_true"][k], out["q_true"][k],
                     np.array([1e-3, -2e-3, 1.5e-3]), np.array([0.05, -0.04, 0.03]))
    imu = (out["gyro_meas"][k], out["accel_meas"][k], out["gyro_meas"][k + 1], out["accel_meas"][k + 1])
    F = error_transition(x, *imu, dt)
    x1 = nominal_step(x, *imu, dt)
    J = np.empty((ERROR_DIM, ERROR_DIM))
    for i in range(ERROR_DIM):
        d = np.zeros(ERROR_DIM)
        d[i] = FD_STEP
        plus = difference(nominal_step(inject(x, d), *imu, dt), x1)
        minus = difference(nominal_step(inject(x, -d), *imu, dt), x1)
        J[:, i] = (plus - minus) / (2.0 * FD_STEP)

    eye = np.eye(ERROR_DIM)
    print()
    failures = []
    for rn, rs in BLOCKS.items():
        for cn, cs in BLOCKS.items():
            dev = np.max(np.abs(F[rs, cs] - eye[rs, cs]))
            diff = np.max(np.abs(J[rs, cs] - F[rs, cs]))
            if dev > 0.0:
                rel = diff / dev
                print(f"  d{rn}/d{cn}: max|F-I| = {dev:.3e}, max|J-F| = {diff:.3e}, relative = {rel:.3e}")
                if rel > F_REL_TOL:
                    failures.append((rn, cn, rel))
            else:
                if diff > 1e-12:
                    print(f"  d{rn}/d{cn}: F-I = 0, max|J-F| = {diff:.3e}")
                if diff > F_ABS_TOL:
                    failures.append((rn, cn, diff))
    assert not failures, failures


# --- Noise-free scenario -----------------------------------------------------------------------
# With error-free data and an exact initial state the estimate may deviate from the truth only
# by the integration error of the nominal propagation. For "straight" and "turning" this error is
# below 1e-4 m over 60 s (synthetic/README.md); 1 cm and 1 mrad leave margin while a filter error
# produces deviations of the order of the measurement noise (0.5 m) or more. "aggressive" has a
# known 0.45 m discretisation error at 200 Hz and is reported without a threshold.
NOISE_FREE_POS_TOL = 0.01
NOISE_FREE_ATT_TOL = 1e-3


@pytest.mark.parametrize("profile", ["straight", "turning", "aggressive"])
def test_noise_free_estimate_stays_on_truth(profile):
    cfg = ScenarioConfig(profile=profile, duration=60.0, enable_white_noise=False,
                         enable_turn_on_bias=False, enable_bias_rw=False, enable_meas_noise=False)
    out = generate(cfg)
    hist = run_filter(out, cfg, FilterSettings(), exact_initial_state(out))
    print(f"\n{profile}: max position error = {hist['max_pos_err']:.3e} m, "
          f"max attitude error = {hist['max_att_err']:.3e} rad")
    if profile != "aggressive":
        assert hist["max_pos_err"] <= NOISE_FREE_POS_TOL
        assert hist["max_att_err"] <= NOISE_FREE_ATT_TOL


# --- Monte Carlo consistency -------------------------------------------------------------------
N_RUNS = 50
# Mean NEES / NIS over N runs at one epoch is chi2(3N)/N distributed for a consistent filter;
# the two-sided 95 % interval of that distribution is used for the means over all runs and
# epochs. Averaging additionally over time narrows the true spread, so this bound is lenient
# with respect to random variation but still rejects a filter that is over- or under-confident
# by more than about 20 %.
CONF = 0.95
# A filter estimate should be better than the raw measurement: 3D RMS of the measurement noise.
MEAS_RMS_3D = 0.5 * np.sqrt(3.0)
# Bias convergence: the RMS error over runs at the final epoch must be below half of the RMS
# initial error (the initial bias estimate is zero, so the initial error is the drawn bias).
BIAS_REDUCTION = 0.5


@pytest.fixture(scope="module")
def monte_carlo():
    settings = FilterSettings()
    hists = []
    for i in range(N_RUNS):
        cfg = scenario_config("turning", i, settings)
        out = generate(cfg)
        hists.append(run_filter(out, cfg, settings, initial_state(out, settings, i)))
    return hists


def test_monte_carlo_consistency(monte_carlo):
    nees = np.array([h["nees_pos"] for h in monte_carlo])  # (N, epochs)
    nis = np.array([h["nis"] for h in monte_carlo])
    err = np.array([h["err"][:, 0:3] for h in monte_carlo])
    lo, hi = chi2.ppf([(1 - CONF) / 2, (1 + CONF) / 2], df=3 * N_RUNS) / N_RUNS
    mean_nees, mean_nis = float(np.mean(nees)), float(np.nanmean(nis))
    rmse = float(np.sqrt(np.mean(np.sum(err**2, axis=2))))
    frac_nees = float(np.mean((nees.mean(0) >= lo) & (nees.mean(0) <= hi)))
    frac_nis = float(np.mean((np.nanmean(nis, 0) >= lo) & (np.nanmean(nis, 0) <= hi)))
    print(f"\nN = {N_RUNS} runs, {nees.shape[1]} measurement epochs each")
    print(f"position RMSE (3D) = {rmse:.4f} m (measurement 3D RMS = {MEAS_RMS_3D:.4f} m)")
    print(f"mean position NEES = {mean_nees:.4f} (expected 3, 95 % interval [{lo:.4f}, {hi:.4f}])")
    print(f"mean NIS           = {mean_nis:.4f} (expected 3, 95 % interval [{lo:.4f}, {hi:.4f}])")
    print(f"fraction of epochs with run-averaged NEES inside the interval = {frac_nees:.3f}")
    print(f"fraction of epochs with run-averaged NIS inside the interval  = {frac_nis:.3f}")
    assert rmse < MEAS_RMS_3D
    assert lo <= mean_nees <= hi
    assert lo <= mean_nis <= hi


def test_monte_carlo_bias_convergence(monte_carlo):
    err = np.array([h["err"] for h in monte_carlo])  # (N, epochs, 15)
    sigma = np.array([h["sigma"] for h in monte_carlo])
    print()
    failures = []
    for name, sl, unit in (("gyro bias", slice(9, 12), "rad/s"), ("accel bias", slice(12, 15), "m/s^2")):
        rms0 = np.sqrt(np.mean(err[:, 0, sl] ** 2, axis=0))
        rms_end = np.sqrt(np.mean(err[:, -1, sl] ** 2, axis=0))
        sig_end = np.mean(sigma[:, -1, sl], axis=0)
        for ax, a, b, s in zip("xyz", rms0, rms_end, sig_end):
            ratio = b / a
            print(f"  {name} {ax}: RMS error t=0 {a:.3e}, t=60 s {b:.3e} {unit}, ratio {ratio:.3f}, "
                  f"mean filter sigma at 60 s {s:.3e}")
            if ratio >= BIAS_REDUCTION:
                failures.append((name, ax, ratio))
    assert not failures, failures


def test_same_seed_identical():
    settings = FilterSettings()
    runs = []
    for _ in range(2):
        cfg = scenario_config("turning", 3, settings, duration=10.0)
        out = generate(cfg)
        runs.append(run_filter(out, cfg, settings, initial_state(out, settings, 3)))
    for key in ("err", "sigma", "nis"):
        assert np.array_equal(runs[0][key], runs[1][key], equal_nan=True), key


# --- Attitude-perturbation columns of F --------------------------------------------------------
# Same thresholds as test_F_matches_finite_differences, applied only to the columns of F that
# belong to the attitude error, on several states of two profiles.
@pytest.mark.parametrize("profile,t_s", [("aggressive", 5.0), ("aggressive", 10.0),
                                         ("aggressive", 15.0), ("turning", 10.0)])
def test_F_attitude_columns_match_finite_differences(profile, t_s):
    out = generate(ScenarioConfig(profile=profile, duration=20.0, enable_white_noise=False,
                                  enable_bias_rw=False))
    k, dt = int(round(t_s * 200)), 1.0 / 200.0
    x = NominalState(out["p_true"][k], out["v_true"][k], out["q_true"][k],
                     np.array([1e-3, -2e-3, 1.5e-3]), np.array([0.05, -0.04, 0.03]))
    imu = (out["gyro_meas"][k], out["accel_meas"][k], out["gyro_meas"][k + 1], out["accel_meas"][k + 1])
    F = error_transition(x, *imu, dt)
    x1 = nominal_step(x, *imu, dt)
    J = np.empty((ERROR_DIM, 3))
    for c, i in enumerate(range(6, 9)):
        d = np.zeros(ERROR_DIM)
        d[i] = FD_STEP
        plus = difference(nominal_step(inject(x, d), *imu, dt), x1)
        minus = difference(nominal_step(inject(x, -d), *imu, dt), x1)
        J[:, c] = (plus - minus) / (2.0 * FD_STEP)
    Fth = F[:, 6:9]
    eye = np.eye(ERROR_DIM)[:, 6:9]
    print(f"\n{profile} t = {t_s:g} s:")
    failures = []
    for rn, rs in BLOCKS.items():
        dev = np.max(np.abs(Fth[rs] - eye[rs]))
        diff = np.max(np.abs(J[rs] - Fth[rs]))
        if dev > 0.0:
            rel = diff / dev
            print(f"  d{rn}/dth: max|F-I| = {dev:.3e}, max|J-F| = {diff:.3e}, relative = {rel:.3e}")
            if rel > F_REL_TOL:
                failures.append((rn, rel))
        else:
            print(f"  d{rn}/dth: F-I = 0, max|J-F| = {diff:.3e}")
            if diff > F_ABS_TOL:
                failures.append((rn, diff))
    assert not failures, failures


# --- Covariance reset Jacobian -----------------------------------------------------------------
# After injecting the estimate dx_hat, the error with respect to the new nominal state is
# g(dx) = difference(inject(x, dx), inject(x, dx_hat)). Its Jacobian at dx = dx_hat is the
# identity except for the attitude block, which is the right Jacobian J_r(dtheta_hat)
# = I - [dtheta_hat/2]x + O(|dtheta_hat|^2). The filter uses G = I - [dtheta_hat/2]x, so the
# attitude block may differ by a second-order term of size about |dtheta_hat|^2 / 6; the threshold
# 0.5 |dtheta_hat|^2 allows a factor three on that. All other blocks must be the identity.
RESET_ABS_TOL = 1e-8


@pytest.mark.parametrize("angle", [1e-3, 1e-2, 1e-1])
def test_reset_jacobian_matches_finite_differences(angle):
    from experiments.eskf_sensitivity.eskf import reset_jacobian

    rng = np.random.default_rng(0)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    x = NominalState(np.array([1.0, 2.0, 3.0]), np.array([0.5, -0.2, 0.1]),
                     np.array([np.cos(0.3), 0.0, 0.0, np.sin(0.3)]),
                     np.array([1e-3, 0.0, -1e-3]), np.array([0.02, 0.01, -0.03]))
    dx_hat = np.concatenate([rng.normal(scale=0.1, size=6), angle * axis, rng.normal(scale=1e-3, size=6)])
    x_new = inject(x, dx_hat)
    J = np.empty((ERROR_DIM, ERROR_DIM))
    for i in range(ERROR_DIM):
        d = np.zeros(ERROR_DIM)
        d[i] = FD_STEP
        J[:, i] = (difference(inject(x, dx_hat + d), x_new)
                   - difference(inject(x, dx_hat - d), x_new)) / (2.0 * FD_STEP)
    G = reset_jacobian(dx_hat[6:9])
    th = slice(6, 9)
    diff_th = np.max(np.abs(J[th, th] - G[th, th]))
    diff_th_identity = np.max(np.abs(J[th, th] - np.eye(3)))
    mask = np.ones((ERROR_DIM, ERROR_DIM), dtype=bool)
    mask[th, th] = False
    diff_other = np.max(np.abs(J - np.eye(ERROR_DIM))[mask])
    print(f"\n|dtheta_hat| = {angle:g}: attitude block max|J - G| = {diff_th:.3e} "
          f"(threshold {0.5 * angle**2:.3e}), max|J - I| (no reset) = {diff_th_identity:.3e}, "
          f"other blocks max|J - I| = {diff_other:.3e}")
    assert diff_th <= 0.5 * angle**2
    assert diff_other <= RESET_ABS_TOL


def test_zero_misalignment_reproduces_nominal_history(tmp_path):
    """imu_misalignment_x_deg = 0 must reproduce results/turning_run0_history.csv byte for byte."""
    from pathlib import Path

    from experiments.eskf_sensitivity.run import (
        DEFAULT_PROFILE,
        DEFAULT_RUN_INDEX,
        RESULTS_DIR,
        write_history_csv,
    )

    settings = FilterSettings()
    cfg = scenario_config(DEFAULT_PROFILE, DEFAULT_RUN_INDEX, settings, imu_misalignment_x_deg=0.0)
    out = generate(cfg)
    hist = run_filter(out, cfg, settings, initial_state(out, settings, DEFAULT_RUN_INDEX))
    path = tmp_path / "history.csv"
    write_history_csv(hist, out, path)
    reference = Path(RESULTS_DIR) / f"{DEFAULT_PROFILE}_run{DEFAULT_RUN_INDEX}_history.csv"
    assert path.read_bytes() == reference.read_bytes()


def test_zero_gyro_scale_factor_reproduces_nominal_history(tmp_path):
    """gyro_scale_factor = 0 must reproduce results/turning_run0_history.csv byte for byte."""
    from pathlib import Path

    from experiments.eskf_sensitivity.run import (
        DEFAULT_PROFILE,
        DEFAULT_RUN_INDEX,
        RESULTS_DIR,
        write_history_csv,
    )

    settings = FilterSettings()
    cfg = scenario_config(DEFAULT_PROFILE, DEFAULT_RUN_INDEX, settings, gyro_scale_factor=0.0)
    out = generate(cfg)
    hist = run_filter(out, cfg, settings, initial_state(out, settings, DEFAULT_RUN_INDEX))
    path = tmp_path / "history.csv"
    write_history_csv(hist, out, path)
    reference = Path(RESULTS_DIR) / f"{DEFAULT_PROFILE}_run{DEFAULT_RUN_INDEX}_history.csv"
    assert path.read_bytes() == reference.read_bytes()
