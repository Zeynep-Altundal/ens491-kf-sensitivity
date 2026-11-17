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
