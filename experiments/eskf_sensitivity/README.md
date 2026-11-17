# experiments/eskf_sensitivity

Classical error-state Kalman filter (ESKF) for IMU propagation with low-rate position updates,
validated on the synthetic generator in `synthetic/`, and a one-parameter-at-a-time sensitivity
study of the filter (`sweep.py`, protocol below).

## Conventions

As in `synthetic/README.md`: ENU world frame, gravity `g = [0, 0, -9.81] m/s^2`, Hamilton unit
quaternion `q` (scalar first) for the body-to-world rotation `R = R(q)`, gyroscope measuring the
body rate `w_m = w_b + b_g + n_g`, accelerometer measuring the specific force
`f_m = R^T (a - g) + b_a + n_a`.

| Symbol | Meaning |
|--------|---------|
| `x = (p, v, q, b_g, b_a)` | nominal state (16 numbers) |
| `dx = [dp, dv, dtheta, db_g, db_a]` | error state (15) |
| `[u]x` | skew-symmetric matrix, `[u]x w = u x w` |
| `Exp(u)` / `Log(q)` | rotation vector to quaternion and back |

Error definition (local attitude error, expressed in the body frame):

    p_true = p + dp,   v_true = v + dv,   q_true = q ⊗ Exp(dtheta),
    b_g,true = b_g + db_g,   b_a,true = b_a + db_a.

## Propagation (`propagate`)

The filter stores the previous IMU sample `(w_0, f_0)`; a call with the new sample `(w_1, f_1)`
propagates over `dt`.

Nominal state (same second-order scheme as the generator tests):

    w_i  = w_m,i - b_g,                     i = 0, 1
    phi  = dt/2 (w_0 + w_1) + dt^2/12 (w_0 x w_1)
    q_1  = q_0 ⊗ Exp(phi)
    a_0  = R(q_0) (f_m,0 - b_a) + g,        a_1 = R(q_1) (f_m,1 - b_a) + g
    p_1  = p_0 + v_0 dt + dt^2 (a_0/3 + a_1/6)
    v_1  = v_0 + dt/2 (a_0 + a_1)
    b_g, b_a constant

Continuous error dynamics, with `w = (w_0 + w_1)/2`, `f = (f_m,0 + f_m,1)/2 - b_a` and
`R = R(q_0)`:

    d(dp)/dt     = dv
    d(dv)/dt     = -R [f]x dtheta - R db_a - R n_a
    d(dtheta)/dt = -[w]x dtheta - db_g - n_g
    d(db_g)/dt   = n_bg
    d(db_a)/dt   = n_ba

i.e. `d(dx)/dt = A dx + noise` with the non-zero blocks of `A`:
`A_p,v = I`, `A_v,theta = -R [f]x`, `A_v,ba = -R`, `A_theta,theta = -[w]x`, `A_theta,bg = -I`.

Discrete transition and covariance propagation:

    F = I + A dt + (A dt)^2 / 2
    P <- F P F^T + Q
    Q = diag(0, s_a^2 dt I, s_g^2 dt I, s_bg^2 dt I, s_ba^2 dt I)

`s_g`, `s_a` are the gyro and accelerometer noise densities and `s_bg`, `s_ba` the bias
random-walk densities, in the units of `synthetic/README.md`. The velocity block uses
`R (s_a^2 I) R^T = s_a^2 I`. The second-order term in `F` provides the `dt^2` couplings
(for example `dp/dtheta = -R [f]x dt^2/2`) that the nominal position step contains.

## Position update (`update_position`)

    z = p_true + v,   v ~ N(0, R_z),   H = [I 0 0 0 0]
    nu  = z - p                          (innovation)
    S   = H P H^T + R_z,   NIS = nu^T S^-1 nu
    K   = P H^T S^-1
    dx  = K nu
    P   <- (I - K H) P (I - K H)^T + K R_z K^T          (Joseph form)

Error injection: `p += dp`, `v += dv`, `q <- q ⊗ Exp(dtheta)`, `b_g += db_g`, `b_a += db_a`.
Reset: the error mean is set to zero and `P <- G P G^T` with
`G = blkdiag(I_6, I - [dtheta/2]x, I_6)` (switchable with `apply_reset`, default on; off is used
only in the consistency analysis).

Optional innovation gate (`gate_prob`, off by default): the update is skipped when
`NIS > chi2.ppf(gate_prob, 3)`.

## Initialisation used in the runs (`run.py`)

`P0 = diag(s_p0^2 I, s_v0^2 I, s_th0^2 I, s_bg0^2 I, s_ba0^2 I)` with
`s_p0 = 0.5 m`, `s_v0 = 0.1 m/s`, `s_th0 = 0.01 rad`, `s_bg0 = 2e-3 rad/s`, `s_ba0 = 0.05 m/s^2`.
For run index `i` the scenario uses seed `i`; the turn-on biases and the initial position,
velocity and attitude errors are drawn from these `P0` blocks, so the filter's initial
uncertainty matches the actual initial error. The bias estimates start at zero. The filter is
given the true IMU noise parameters and the true measurement covariance. Metrics are evaluated at
the measurement epochs (10 Hz) after the update.

## Sensitivity study protocol (`sweep.py`)

This protocol, including all grids, was fixed before any sweep was run; the grids are not changed
after seeing results.

### Common settings

* Profile `turning`, 60 s, IMU 200 Hz, position measurements 10 Hz, all other scenario values at
  the `ScenarioConfig` defaults, unless a sweep states otherwise.
* Monte Carlo over independent seeds: run index `i = 0 .. N-1`, **N = 30**, executed in parallel
  with `multiprocessing` (one process per CPU core minus two). Run `i` uses scenario seed `i` and
  draws the turn-on biases and the initial position, velocity and attitude errors from the true
  initial covariance exactly as `run.py` does. The same run indices are used for every setting
  (common random numbers), so differences between settings are not caused by different draws.
* Truth always uses the true initial covariance `P0_true` (the `FilterSettings` defaults); a sweep
  changes only the filter value where stated. The innovation gate is off except in sweep (g).
* **Evaluation epochs:** every 20th IMU sample (10 Hz, t = 0, 0.1, ..., 60 s), independent of the
  measurement rate; the state is taken after the update when a measurement is processed at that
  sample, otherwise after propagation.

### Metrics per setting

* Position, velocity and attitude RMSE: 3D, over all runs and evaluation epochs.
* Mean position NEES (expected 3) over all runs and evaluation epochs.
* Mean NIS (expected 3) over all accepted updates; for sweep (g) also the fraction of
  measurements rejected by the gate.
* Final bias estimation error: RMS over runs of the 3D norm of the gyro and accelerometer bias
  error at t = 60 s.
* Sweep (f): per run the maximum position error norm over the evaluation epochs inside the dropout
  window; the mean over runs and the maximum over runs are reported.
* Runs with non-finite estimates or numerical errors are counted and excluded from the averages.

### Mis-tuning sweeps (true scenario nominal; one filter quantity scaled)

`s` in `logspace(-2, 2, 7)` = 0.01, 0.0464, 0.215, 1, 4.64, 21.5, 100:

* (a) `q_imu`: the white-noise blocks of Q (velocity and attitude) scaled by `s` (noise densities
  scaled by `sqrt(s)`).
* (b) `q_bias`: the bias random-walk blocks of Q scaled by `s`.
* (c) `r`: R scaled by `s`.
* (d) `p0`: the filter P0 scaled by `s` (truth still drawn from `P0_true`).

### Scenario sweeps (filter given the true noise parameters of each setting)

* (e) `bias_estimation`: turn-on bias scale `c` in **0, 0.1, 0.3, 1, 3, 10** times the nominal
  standard deviations (gyro 2e-3 rad/s, accelerometer 0.05 m/s^2 per axis); the turn-on biases
  are drawn per run from `N(0, (c sigma_nominal)^2)`. Two variants: bias estimation **enabled**
  (filter P0 bias blocks `(c sigma_nominal)^2`, bias random-walk Q as true) and **disabled** (bias
  states removed from estimation by setting their P0 and Q blocks to zero, so the bias estimates
  stay at zero). Reasoning: `c = 1` is the nominal MEMS-grade value; `c = 0.1` corresponds to a
  much better (tactical-grade) sensor, `c = 10` (gyro 0.02 rad/s, about 1.1 deg/s, accelerometer
  0.5 m/s^2) to a poor consumer sensor; `c = 0` isolates the effect of the bias random walk alone.
  The bias random walk stays active in the truth for all `c`.
* (f) `dropout`: one measurement dropout window starting at **t = 15 s** with duration
  **0, 2, 5, 10, 20, 40 s**. Reasoning for the start time: the window must fit inside the 60 s run
  for the longest duration (15 + 40 = 55 s, leaving 5 s of recovery) and should start after the
  first part of the initial transient.
* (g) `outliers`: measurement outlier probability per epoch **0, 0.01, 0.02, 0.05, 0.1, 0.2** with
  magnitude 5 m (the generator default), gate **off** and **on**. The gate uses
  `gate_prob = 0.999` (threshold `chi2.ppf(0.999, 3) = 16.27`), so about 0.1 % of correct
  measurements are rejected. Reasoning: a 5 m offset is ten times the per-axis measurement
  standard deviation (0.5 m) and gives a NIS of the order of 100, far above the threshold; the
  probabilities range from occasional outliers (0.01: about 6 per run at 10 Hz) to heavy
  contamination (0.2: about 120 per run).
* (h) `imu_spikes`: IMU spike probability per sample and axis **0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2**
  for both sensors, magnitudes at the generator defaults (gyro 0.5 rad/s, accelerometer 5 m/s^2).
  Reasoning: a run has 12 000 samples per axis, so the grid spans about 1 to 120 spikes per axis
  and run. A single gyro spike changes the integrated attitude by about 0.5 rad/s x 5 ms = 2.5 mrad
  and a single accelerometer spike the velocity by about 0.025 m/s; the grid covers the range from
  isolated spikes to frequent ones whose effects accumulate.
* (i) `profile`: **straight, turning, aggressive**. Note: `aggressive` carries the known 0.45 m
  discretisation error of the 200 Hz integration (`synthetic/README.md`), and `straight` has
  constant attitude, which limits the observability of yaw and of the horizontal accelerometer
  biases.
* (j) `meas_rate`: position measurement rate **1, 2, 5, 10, 20 Hz** (all divide the IMU rate).

All settings are reported, including those without a visible effect. Outputs:
`results/sweep_<name>.csv`, `results/sweep_dropout_timeseries.csv` and `figures/sweep_<name>.png`,
`figures/sweep_dropout_error_over_time.png`.

## Statistical closure protocol (`closure.py`)

Fixed before running; not changed after seeing results.

* Settings: the nominal setting and the first and last grid point of every sweep (all variants
  of sweeps (e) and (g)), with the sweep protocol above (N = 30, turning unless varied, run
  indices 0..29).
* Per-run quantities as in the sweeps; reported: position, velocity and attitude RMSE, mean
  position NEES, mean NIS and the final bias errors with 95 % bootstrap intervals, and the paired
  difference of position RMSE and mean position NEES of each setting against the nominal setting
  (the same run indices, hence paired).
* Seed-to-seed spread: the nominal setting is repeated with **5 master seeds**; master seed `m`
  uses run indices `1000 m + i`, i = 0..29 (master seed 0 reproduces the sweep).
* Intervals: bootstrap over runs as defined in `experiments/common/README.md` (B = 10 000, 95 %
  percentile, resampling seed 2026).
* Previously reported statements re-examined: each "matters" / "small effect" statement of the
  sweep report is checked against the paired difference to the nominal setting.
* Outputs: `results/closure_summary.csv`, `results/closure_paired.csv`, `results/closure_seeds.csv`.

## Full-state consistency protocol (`consistency.py`)

Fixed before running; not changed after seeing results. No noise parameter or threshold is
changed to make a result consistent.

* Error definition exactly as in the filter (`eskf.difference`): `dp`, `dv`,
  `dtheta = Log(q_nom^-1 q_true)` (local attitude error), `db_g`, `db_a`.
* Full NEES `dx^T P^-1 dx` (expected 15) and sub-block NEES for position, velocity, attitude,
  gyro bias and accelerometer bias using the corresponding diagonal blocks of P (expected 3 each).
* Monte Carlo **N = 50** (run indices 0..49) on `straight`, `turning` and `aggressive`, 60 s,
  nominal noise, filter given the true parameters, initial errors and turn-on biases drawn from
  P0 as in `run.py`; evaluation epochs at 10 Hz.
* Reported: mean over runs and epochs with a 95 % bootstrap interval over runs; time-resolved
  run-averaged NEES with the two-sided 95 % bounds `chi2.ppf([0.025, 0.975], d N) / N`
  (d = 15 or 3); the fraction of epochs inside the bounds (about 95 % expected).
* Covariance reset: the filter applies `P <- G P G^T` with `G = blkdiag(I6, I - [dtheta/2]x, I6)`
  after each injection. Checked (i) by a finite-difference test of the map from the pre-injection
  error to the post-injection error, and (ii) by repeating the Monte Carlo with the reset disabled
  (`G = I`) and comparing full-state NEES.
* Attitude-perturbation blocks of F: finite-difference test using attitude perturbations only, on
  several states of the `aggressive` and `turning` profiles (thresholds as in the existing F test).
* Outputs: `results/consistency_summary.csv`, `results/consistency_time_<profile>.csv`,
  `figures/consistency_<profile>.png`.

## Structural mismatch and GNSS availability protocol (`misalignment.py`)

Fixed before running; not changed after seeing results. No filter parameter is tuned.

### Part 1: fixed IMU-to-body misalignment

* Generator: a constant sensor-to-body rotation `R_BS = Exp(dphi e_x)` about the body x axis
  (`ScenarioConfig.imu_misalignment_x_deg`). The ideal sensor measures `w_S = R_BS^T w_B` and
  `f_S = R_BS^T f_B`. White noise, turn-on bias, bias random walk and spikes are then generated
  in the **sensor frame** and added to these signals, as for a physical sensor whose errors belong
  to its own axes. The per-axis noise is isotropic (`sigma^2 I`), so the noise covariance is the
  same in any rotated frame (`R^T sigma^2 I R = sigma^2 I`); the misalignment therefore changes only
  the clean signal, not the noise statistics. The true biases are recorded in the sensor frame.
* Filter: unchanged. It treats the sensor frame as the body frame (it assumes `R_BS = I`). Errors
  are computed against the true **body** attitude, so an estimate that converges to the sensor
  attitude shows a constant attitude error equal to the misalignment.
* Levels: **0, 1, 3, 5 degrees**. Profile `turning`, 60 s, **N = 50**, run indices 0..49 at every
  level (same noise, bias and initial-error realisations; the levels are paired).
* Reported per level: position, velocity and attitude RMSE (3D); full-state NEES (expected 15)
  and the five sub-block NEES (expected 3); mean over runs and epochs with a 95 % bootstrap
  interval, the two-sided 95 % chi-square interval of the N-run average at one epoch,
  `chi2.ppf([0.025, 0.975], d N) / N`, and the fraction of epochs whose run-averaged NEES lies
  inside it; mean NIS (expected 3); final (t = 60 s) estimated and true biases per axis (mean over
  runs) and the mean bias error; paired differences to the 0 degree level (position RMSE, attitude
  RMSE, full NEES, NIS and final bias error per axis) with 95 % bootstrap intervals.
* Whether the bias estimates absorb part of the misalignment is stated as an observation based on
  the paired bias-error differences, not as a verified mechanism.
* Test: `imu_misalignment_x_deg = 0` reproduces the existing nominal single-run history
  (`results/turning_run0_history.csv`) byte for byte.

### Part 2: GNSS availability scenarios (reported separately)

At 0 degrees misalignment, profile `turning`, 60 s, N = 50, run indices 0..49:

* **S0**: position measurements at 10 Hz for the whole run.
* **S1**: outage from 20 s to 60 s inclusive (no measurement with `t >= 20 s`; dropout window
  `[20 s, 60.1 s)`).
* **S2**: no position measurement in the whole run (dropout window `[0 s, 60.1 s)`); the filter
  runs on the IMU alone from the P0 prior.
* **S1 at 1, 3 and 5 degrees** misalignment, to show the effect on dead-reckoning drift.

Reported: RMS position error over time with `3 sqrt(mean tr P_pos)`; final (t = 60 s) and maximum
position error (mean over runs with 95 % interval, and maximum over runs); mean full-state NEES
over the epochs inside and outside the outage (95 % intervals); drift growth: RMS position error
and `3 sqrt(mean tr P_pos)` at elapsed outage times 0, 1, 2, 5, 10, 20, 30 and 40 s (for S2 the
elapsed time is measured from t = 0).

### Part 3: clarifications (`clarify.py`)

* (a) The "4.2 % of epochs" figure of the turning consistency study is restated with its exact
  definition. The turning consistency runs (N = 50, run indices 0..49, reset on) are recomputed with
  the same seeds to obtain the errors behind the NEES; the recomputed run-averaged NEES must equal
  `results/consistency_time_turning.csv`. Shown for t = 5..25 s (window 10..18 s highlighted):
  run-averaged full, attitude and gyro-bias NEES with bounds, the RMS attitude error per axis and
  the RMS gyro-bias error per axis together with the mean filter standard deviations.
* (b) Attitude columns of F (`aggressive`, t = 10 s, as in the existing test): sensitivity of the
  finite-difference result to the step `h = 1e-4 .. 1e-8 rad` (central differences), and the
  linearisation error for finite attitude perturbations of magnitude `1e-4 .. 1e-1 rad`, measured
  as `|g(d) - d - (F - I) d| / |(F - I) d|`, where `g(d)` is the propagated error.
* (c) Every bootstrap in this repository resamples whole Monte Carlo runs, never epochs (see
  "Bootstrap unit" below).

### Bootstrap unit

All bootstrap intervals (`closure.py` in `linear_kf`, `filter_family` and `eskf_sensitivity`,
`consistency.py` and `misalignment.py`) resample whole Monte Carlo runs: each resample draws run
indices with replacement and recomputes the statistic from the per-run values (per-run time
averages or per-run final values). Epochs within a run are never resampled individually, because
they are correlated in time.

## Files

| File | Purpose |
|------|---------|
| `eskf.py` | Filter: nominal step, error dynamics, process noise, injection, reset, update, gate. |
| `run.py` | Scenario set-up, filter run, summary metrics and CSV output. |
| `sweep.py` | Sensitivity study (protocol above), parallel Monte Carlo; writes `results/sweep_*.csv`. |
| `closure.py` | Statistical closure of the sweeps: bootstrap intervals, paired differences to nominal, seed spread. |
| `consistency.py` | Full-state and sub-block NEES Monte Carlo, with and without covariance reset. |
| `misalignment.py` | IMU-to-body misalignment study and GNSS availability scenarios. |
| `clarify.py` | Clarifications on the consistency study and on the finite-difference check of F. |
| `plots.py` | Figures of a single run, of the sweeps, of the consistency, misalignment and GNSS studies and of the clarifications. |
| `tests/` | Validation tests. |
| `results/` | CSV output. |
| `figures/` | PNG output. |

## Running (from the repository root)

```bash
python -m pytest -v -s experiments/eskf_sensitivity
python -m experiments.eskf_sensitivity.run
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.eskf_sensitivity.sweep
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.eskf_sensitivity.closure
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.eskf_sensitivity.consistency
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.eskf_sensitivity.misalignment
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.eskf_sensitivity.clarify
python -m experiments.eskf_sensitivity.plots
```
