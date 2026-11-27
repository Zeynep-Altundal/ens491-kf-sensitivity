# experiments/estimators

Step 5 of 8: three estimators run on the mission recording and its corruption scenarios, all
through one interface that reads the common recording format v1
(`../../simulation/rviz_drone/README.md`, sections "Flight recorder" and "Corruption scenarios").

This protocol was written before any run of this package. Changes made after a run are listed
under "Protocol changes" with their reason.

## Labels for statements

* **Observed**: a number or behaviour read directly from the results of this package (with its
  run, seed set and interval).
* **Hypothesized**: a proposed explanation of an observation that has not been tested by a
  dedicated experiment.
* **Verified**: a statement confirmed by a dedicated check whose design was fixed before it was
  run (a test, a sanity check of this protocol or a targeted experiment).

The result tables of this step contain Observed statements only; no explanation is claimed yet.

## Estimators (`filters.py`, interface `Estimator`)

Every estimator implements `predict(t, gyro, accel)` at each IMU epoch, `update_position(z, R)` at
each position epoch with `pos_valid = True` (invalid epochs are skipped: prediction only), and
reports its state, covariance diagonal, errors against the truth, NEES and NIS. All estimators run
on the IMU time base (200 Hz), so that all errors are evaluated at the same epochs.

| Name | State | Inputs | Model |
|---|---|---|---|
| `KF` | `p, v` (6) | position measurements only, no IMU | linear constant-velocity model, continuous white acceleration noise with power spectral density `q_cv` per axis, exact discretisation; Joseph update |
| `EKF` | `p, v, q, b_g, b_a` (16) | gyro, accelerometer, position | full-state EKF with the quaternion in the state: the same nominal propagation as the ESKF (rotation vector with coning term, trapezoidal velocity, second-order position), analytic 16 x 16 Jacobian of that discrete step, additive update of all 16 states followed by normalisation of the quaternion; the covariance is projected onto the tangent space of the unit sphere after each normalisation (global attitude error handled by normalisation, no error-state reset) |
| `ESKF` | 15 error states `dp, dv, dtheta, db_g, db_a` | gyro, accelerometer, position | the existing filter `experiments/eskf_sensitivity/eskf.py` (unchanged), used through an adapter |

The `KF` is a system-level baseline: it answers "what does a position-only filter achieve on the
same recording", not "how does a linear filter compare with the nonlinear ones on equal terms".
It never sees the IMU, so it is not an estimator-level equal comparison with `EKF` and `ESKF`.

Full-state EKF from the companion repository: `nkf/compare/filters.py` (class `EKF`) of
`ens491-nkf`, branch `origin/feat/W03-synthetic-scenarios`, commit `bb29c92` (read with `git show`,
that repository unchanged). It is pure NumPy (dependencies `nkf.compare.kinematics`,
`nkf.filters.conventions`), uses the same frames (ENU, Hamilton `[w, x, y, z]`, body to world,
`g_w = [0, 0, -9.81]`) and could be used through an adapter from a pinned export of that commit
into a git-ignored folder. It is not used as the `EKF` of this step because its propagation is
first order with the IMU sample held over the interval (`f_full`: `p + v dt + a dt^2 / 2`,
`v + a dt`, `q (x) Exp(w dt)`) and its Jacobians are central finite differences, while the ESKF
uses a second-order scheme; an EKF-ESKF comparison would then mix the filter structure with the
integration scheme. It is run once on the nominal recording as a cross-check
(`cross_check_nkf.py`, export in `_external/`, git-ignored).

## Tuning (fixed once, never changed per scenario)

All values come from the nominal recording `mission_seed0.npz` (its `meta.true_noise` and its
truth), fixed before the sweep and used unchanged for every scenario and seed:

| Quantity | Value | Source |
|---|---|---|
| gyro noise density | `meta.true_noise.gyro_noise_density` (2.0e-4 rad/s/sqrt(Hz)) | generator |
| accelerometer noise density | `meta.true_noise.accel_noise_density` (2.0e-3 m/s^2/sqrt(Hz)) | generator |
| gyro bias random walk | `meta.true_noise.gyro_bias_rw` (2.0e-5 rad/s^2/sqrt(Hz)) | generator |
| accelerometer bias random walk | `meta.true_noise.accel_bias_rw` (3.0e-3 m/s^3/sqrt(Hz)) | generator |
| position measurement covariance | `pos_R` of each recording (the covariance told to the filter; equal to the nominal one in every scenario) | recording |
| initial state | truth at `t = 0` for `p, v, q`; biases 0 | common interface (same start for all) |
| initial standard deviations | `p` 0.5 m (the position sigma), `v` 0.1 m/s, attitude 1 deg, `b_g` 2.0e-3 rad/s and `b_a` 0.05 m/s^2 (largest turn-on bias component) | generator |
| `KF` acceleration noise `q_cv` | `a_rms^2 x 1 s` per axis, `a_rms` the per-axis RMS of the true acceleration of the nominal flight (`sim_a_w`), averaged over the three axes | nominal truth |
| gating | none | - |

## Seeds and runs

* Measurement-noise seeds: the sensor seed of the recorder, 0 to 9. Seed 0 is the recorded
  `mission_seed0.npz`; seeds 1 to 9 are recorded in memory from the same simulated truth
  (`rviz_drone.recorder.build_recording`; the truth is identical for all seeds).
* Each scenario of `rviz_drone.corruption.SCENARIOS` (31) is applied to each seed recording with
  its own fixed corruption seed, so the injected corruption is the same for all ten measurement
  seeds and only the sensor noise changes.
* Runs: 3 estimators x (1 nominal + 31 scenarios) x 10 seeds = 960. If the sweep would take more
  than 60 minutes, the number of seeds is reduced and the reduction is stated.

## Metrics (all at the IMU epochs, 200 Hz)

* Errors: position `e_p = p_true - p_hat` and velocity `e_v = v_true - v_hat` in the world frame;
  attitude `dtheta = Log(R_hat^T R_true)` in the body frame (common interface), reported as its
  angle; bias errors. `KF` has no attitude or biases (NaN).
* NEES: `e^T P^-1 e` over the full error state (15 for `EKF` and `ESKF`, the EKF covariance mapped
  to the attitude error by `dtheta = 2 Xi(q)^T dq`; 6 for `KF`). NIS at every position update
  (3 degrees of freedom).
* RMSE of position (main metric), velocity and attitude per mission phase (segments of
  `reference.mission`, repeated segment names pooled; the ground segments are included only in
  "overall") and overall.
* Peak position error over the flight and its time.
* Time to recover after each corruption event: for an event ending at `t_end`, the first time
  `t >= t_end` from which the estimate of the corrupted run stays within 0.1 m of the estimate of
  the nominal run with the same estimator and seed (`||p_hat_scenario - p_hat_nominal|| <= 0.1 m`
  for at least 2 s); the recovery time is `t - t_end`. "Not recovered" if this does not happen
  before the end of the flight; "n/a" for corruptions without an end inside the flight
  (whole-flight noise scale, held ramp). Spikes: one event per spike (`t_end` = spike time); the
  repeated-spike scenario reports the median and the largest recovery over its ten spikes.
* Intervals: mean over the 10 seeds with a 95 % percentile bootstrap interval over seeds
  (`experiments/common/bootstrap.py`: B = 10 000, seed 2026, runs as the unit of resampling).
  RMSE: square root of the mean over seeds of the per-run mean squared error.

## Sanity checks (before the sweep; criteria fixed here)

1. **EKF and ESKF agree on the nominal recording** (implementation check): for each of the 10
   seeds, the largest difference over the flight between the two estimates is at most 0.05 m in
   position, 0.05 m/s in velocity and 0.2 deg in attitude.
2. **Consistency on the nominal recording**: the 10-seed average NEES at each IMU epoch lies inside
   the two-sided 95 % bounds `chi2.ppf([0.025, 0.975], 10 d) / 10` in at least 90 % of the epochs,
   and the 10-seed average NIS at each position epoch inside `chi2.ppf([0.025, 0.975], 30) / 10`
   in at least 90 % of the position epochs; checked for each estimator. The result is reported in
   any case; a failure of `EKF` or `ESKF` stops the step before the sweep and is reported.
3. **KF matches the linear reference** (implementation check): on the nominal recording (no
   missing measurements) the `KF` estimate at the position epochs equals the output of
   `experiments/linear_kf/kf.py:kalman_filter` with `F(0.1 s)`, `Q(0.1 s)`, the same `R`, `x0` and
   `P0` within 1e-9 m (position) and 1e-9 m/s (velocity).

## Results file format (`results/`, git-ignored except its README)

One compressed `.npz` per run, `results/runs/<scenario>/<estimator>_seed<k>.npz`, `<scenario>` =
`nominal` or a scenario name:

| Key | Shape | Content |
|---|---|---|
| `t` | [N] | IMU time stamps [s] |
| `p_hat`, `v_hat` | [N,3] | estimate |
| `q_hat` | [N,4] | attitude estimate `[w, x, y, z]` (NaN for `KF`) |
| `bg_hat`, `ba_hat` | [N,3] | bias estimates (NaN where not estimated) |
| `P_diag` | [N,15] | covariance diagonal in the order `p, v, dtheta, b_g, b_a` (EKF attitude mapped as above; NaN where not estimated) |
| `err_p`, `err_v` | [N,3] | `truth - estimate` [m, m/s] |
| `err_att` | [N,3] | `dtheta` [rad] (NaN for `KF`) |
| `nees` | [N] | NEES (dimension in `meta.nees_dim`) |
| `upd_t`, `nis` | [K] | time and NIS of every position update |
| `meta` | JSON string | estimator, scenario, seed, tuning, recording id, `git_commit`, runtime |

The series are stored as float32 to keep the size small; all metrics are computed from the float64
values in memory before writing. Summary tables: `results/summary_*.csv` and the Markdown tables
below.

## Done criteria

1. Sanity checks 1 and 3 pass; check 2 reported (and passed for `EKF` and `ESKF`; evaluated after
   the convergence window of protocol change 5).
2. All 960 runs (or the stated reduced number) written; runtime reported.
3. Tables: RMSE per phase and overall, peak error and its time, recovery times, per estimator and
   scenario with bootstrap intervals over seeds.
4. Unit tests of the estimators (interface, missing measurements, EKF Jacobian against finite
   differences, KF against the linear reference) and the full test suite pass.

## Protocol changes

1. 2026-11-27, before the sweep, instruction of the project owner: the reported EKF must be the
   companion implementation (and the companion KF and ESKF if they run through the adapter); the
   own `KF`, `EKF` and `ESKF` of `filters.py` become cross-checks (`own_KF`, `own_EKF`,
   `own_ESKF`). If the companion code cannot run correctly on our recordings, it is not patched;
   that part stops and the sweep continues with the own filters marked provisional. The
   paragraph "Full-state EKF from the companion repository" above describes the earlier plan.
2. 2026-11-27, after the sanity checks (results below): check 2 failed for every estimator. Under
   this protocol a failure of the EKF or ESKF stops the step before the sweep; following the
   instruction in change 1, the sweep was run with the companion `KF` and the own `EKF` and `ESKF`
   (provisional). No criterion, tolerance or tuning value was changed.
3. The summary tables (`results/summary_*.csv`, `results/tables.md`) and `results/sanity.json` are
   version-controlled; only `results/runs/` is git-ignored.
4. 2026-11-27, decision of the project owner after the first sanity results: the adapter passes the
   interval mean `(u_k + u_k+1) / 2` of the two IMU samples bounding each interval to the companion
   `predict` (their code unchanged; `vendor.INPUT`). Checks 1 and 2 were rerun for the companion
   pair with this adapter. Rule fixed before the rerun: if both pass, the reported `KF`, `EKF` and
   `ESKF` are the companion filters and the label "provisional" is removed; otherwise the own `EKF`
   and `ESKF` stay reported (provisional) with the reason stated. Outcome: check 1 fails (below).
5. **Protocol amendment, 2026-11-27, decision of the project owner** (after the first sanity
   results, before the final sweep): check 2 is evaluated after a 10 s convergence window
   (epochs and position updates with `t >= 10 s`); the whole-flight fractions are still reported.
   Reason (Observed, own `EKF` and `ESKF`, first sanity run): every NEES violation over the whole
   flight is below the lower bound (none above); 88.2 % of the epochs are inside the bounds over the
   whole flight and 99.4 % from 10 s on. The violations before 10 s lie in the ground (0-1 s),
   takeoff (1-5 s), hover (5-7 s) and the first 3 s of the straight leg (own `ESKF`: 200, 800, 400
   and 483 epochs). The tolerance (90 %) and the bounds are unchanged. For the `KF` the NEES result
   is not a pass: it is reported as a known property of the constant-velocity model on a
   manoeuvring vehicle (its process noise is tuned for the whole flight, so the 6-state NEES is
   mostly below the bound in calm flight and above it in manoeuvres; Hypothesized, not tested).

## Companion filters (adapter `vendor.py`)

* Source: `ens491-nkf`, branch `origin/feat/W03-synthetic-scenarios`, pinned commit
  `bb29c92d30788bfdf06c205e4a85b73ac47e8765` (2026-11-26). Exported read-only with `git archive`
  into `_vendor/nkf_bb29c92d30788bfdf06c205e4a85b73ac47e8765/` (git-ignored; recreate with
  `.venv/bin/python -m experiments.estimators.vendor --export`); the adapter checks the SHA-256 of
  every exported file before importing it. Nothing in that repository was checked out, changed or
  committed, and the export is not committed here.
* Files (all unchanged): `nkf/__init__.py`, `nkf/compare/__init__.py`, `nkf/compare/filters.py`
  (classes `KFCV`, `EKF`, `ESKF`, data classes `Init`, `NoiseSpec`), `nkf/compare/kinematics.py`,
  `nkf/filters/__init__.py`, `nkf/filters/conventions.py`; they import only NumPy, `math` and each
  other.
* Adapter: our tuning and initial state are passed as their `Init` and `NoiseSpec`
  (`sigma_cv = sqrt(q_cv)`); `predict(u, t_k+1 - t_k)` is called with the interval mean
  `u = (u_k + u_k+1) / 2` (protocol change 4; their runner, `nkf/compare/runner.py` line 118, passes
  the start sample `u_k`, still available as `vendor.INPUT = "start"`); the full
  covariance `P` of the filter object is read for the NEES (their `cov_blocks` gives only the
  diagonal blocks), the 16-state EKF covariance mapped with their `EKF._J_q_to_att`.
* Conventions checked in their code: Hamilton `[w, x, y, z]`, `R_wb` body to world, attitude error
  on the right, `g_w = [0, 0, -9.81]`, position update `z - p`: the same as ours. Missing position
  measurements: the filters need no special handling (no update is called).

## Sanity checks: results (2026-11-27, 10 seeds, nominal recording; `results/sanity.json`)

Final run: companion `EKF` and `ESKF` with the interval-mean adapter (protocol change 4); check 2
after the 10 s convergence window (protocol change 5), whole-flight fractions in brackets.

| Check | Estimator | Result | Criterion | Pass |
|---|---|---|---|---|
| 1 EKF-ESKF agreement | own `EKF` vs own `ESKF` | 0.0022 m, 0.0016 m/s, 0.019 deg | 0.05 m, 0.05 m/s, 0.2 deg | yes |
| 1 EKF-ESKF agreement | companion `EKF` vs companion `ESKF` | 0.038 m, 0.028 m/s, **0.41 deg** | same | **no** |
| 2 NEES / NIS | companion `KF` (= own `KF`) | NEES inside 21.6 % (64.8 % below, 13.7 % above) [20.1 %]; NIS inside 90.6 % [90.5 %] | 90 % each | no (known property, change 5) |
| 2 NEES / NIS | companion `EKF` | NEES inside 96.2 % (3.5 % above, 0.3 % below) [85.3 %], mean 14.9; NIS inside 94.5 % [94.5 %] | 90 % each | yes |
| 2 NEES / NIS | companion `ESKF` | NEES inside 96.3 % (3.3 % above, 0.4 % below) [85.4 %], mean 14.8; NIS inside 94.5 % [94.5 %] | 90 % each | yes |
| 2 NEES / NIS | own `EKF` | NEES inside 99.45 % (0.55 % below, none above) [88.2 %], mean 14.1; NIS inside 94.4 % [94.3 %] | 90 % each | yes |
| 2 NEES / NIS | own `ESKF` | NEES inside 99.44 % (0.56 % below, none above) [88.2 %], mean 14.2; NIS inside 94.4 % [94.3 %] | 90 % each | yes |
| 3 KF vs linear reference | companion `KF` and own `KF` | 9.2e-14 m, 3.2e-13 m/s | 1e-9 | yes |

Observed (seed-averaged NEES): the own `ESKF` epochs below the lower bound after 10 s (82 epochs)
lie near 37 s (figure-eight) and 79 s (landing); the companion `EKF` epochs above the upper bound
after 10 s (511 epochs) lie in the aggressive segment. A previous version of this README stated
that every epoch below the lower bound lies in the first 10 s; that was wrong (corrected here).

**Outcome of protocol change 4.** With the interval-mean input the companion `EKF` and `ESKF` pass
check 2, but their pair fails check 1 on attitude: the largest difference over the flight and the
seeds is 0.41 deg against the 0.2 deg tolerance (position 0.038 m and velocity 0.028 m/s pass). The
same check failed with the start-sample input (0.41 deg), so the input convention does not cause it.
Observed on seed 0: the difference is almost all about the body z axis (0.219 of 0.220 deg at its
largest, 69.4 s), and it is largest in the hover, aggressive, return and landing phases (0.21, 0.21,
0.22, 0.20 deg) and at most 0.006 deg in the straight, turn and figure-eight phases. Observed in
their code: the companion `EKF` uses central finite-difference Jacobians of the discrete model and
the process noise `G diag(sigma^2 / dt) G^T` of the IMU-input Jacobian, the companion `ESKF` the
analytic error-state model discretised by `conventions.discretise`; which of these causes the yaw
difference is not tested (OPEN). Therefore the reported `EKF` and `ESKF` remain the own filters,
labelled provisional; the reported `KF` is the companion KF-CV.

Differences on the nominal recording (largest over the flight and 10 seeds), with the final adapter:

| Pair | Position [m] | Velocity [m/s] | Attitude [deg] | Mean NEES (own / companion) | Largest NEES difference |
|---|---|---|---|---|---|
| own `EKF` vs companion `EKF` | 0.0213 | 0.0353 | 0.127 | 13.54 / 14.16 | 16.5 |
| own `ESKF` vs companion `ESKF` | 0.0445 | 0.0511 | 0.490 | 13.55 / 14.12 | 15.7 |
| own `KF` vs companion `KF-CV` | 0 | 0 | - | 4.66 / 4.66 | 0 |

History (first sanity run, start-sample adapter, verified diagnostic on seed 0): the companion
filters held the IMU sample of the start of each interval while our recordings contain
instantaneous samples; mean NEES 45.6 (`EKF`) and 44.6 (`ESKF`) over the flight, 194.7 in the
aggressive segment (companion `ESKF`); the own `EKF` against the companion `EKF` differed by
0.0265 m, 0.0609 m/s and 0.757 deg.

Runtime per run (nominal recording, mean of 10): companion `KF` 0.60 s, `EKF` 3.8 s, `ESKF` 2.0 s;
own `KF` 0.59 s, `EKF` 5.0 s, `ESKF` 2.2 s.

## Open items

* **IMU sample convention of the common interface (OPEN).** The common recording format does not
  state whether IMU sample `k` is the instantaneous value at `t_k` or the mean over an interval.
  Our recordings hold instantaneous values; the companion runner assumes a value held over
  `[t_k, t_k+1)`. Needed: a statement in the common interface, agreed with the companion
  repository's owner role.
* **Companion EKF-ESKF yaw difference (OPEN).** See "Outcome of protocol change 4"; needed: a
  diagnostic run of the companion pair with a common process-noise discretisation.

## Sweep and results (2026-11-27, final estimator set)

960 runs (3 estimators x 32 recordings x 10 seeds; no seed reduction): wall time 211.4 s on 24
worker processes (sum of the run times 3020 s), code at commit `620049e` (plus the uncommitted
adapter change of protocol change 4, which does not affect the swept estimators). Run files: 1.5 GB
in `results/runs/` (git-ignored). Reported estimators: `KF` = companion KF-CV; `EKF`, `ESKF` = own
filters, provisional. All metrics are equal to those of the first sweep (largest difference 0; only
the runtimes differ). Tables with bootstrap intervals: `results/tables.md`.

Observed (selection; numbers from `results/tables.md`, mean over 10 seeds with 95 % interval):

* Nominal position RMSE: `KF` 0.571 m [0.563, 0.579], `EKF` 0.233 m [0.228, 0.238], `ESKF`
  0.233 m [0.228, 0.238].
* `EKF` and `ESKF` give equal RMSE to three decimals in every scenario; the largest difference of
  the means is 0.009 m (`gyro_bias_step_large`: 5.657 m and 5.648 m).
* `KF` RMSE does not change in the IMU scenarios (it does not use the IMU) and has a recovery time
  of 0 there by construction.
* Outages: 40 s calm outage, `KF` 25.5 m [21.3, 29.2], `EKF` 48.3 m [39.4, 56.8]; 40 s outage over
  the aggressive segment, `KF` 62.9 m [58.0, 67.9], `EKF` 7.15 m [4.36, 9.83].
* With the 0.1 m recovery tolerance, the `EKF` and `ESKF` estimates do not return to the nominal
  estimate before the end of the flight after the large accelerometer step and after every gyro
  step that starts in the figure-eight or aggressive segment (10 of 10 seeds).

Peak analysis of this sweep: `../peak_analysis/`.

## Files

| File | Purpose |
|---|---|
| `filters.py` | Interface, own `KF`, `EKF` (16 states) and `ESKF` adapter (cross-checks) |
| `vendor.py` | Adapter to the pinned companion filters (`KF-CV`, `EKF`, `ESKF`), export and hash check |
| `data.py` | Recordings per measurement seed, scenarios, fixed tuning, mission phases |
| `run.py` | One estimator run: estimates, covariance diagonal, errors, NEES, NIS; result file writer |
| `metrics.py` | RMSE per phase, peak error, corruption events, recovery time |
| `sanity.py` | Sanity checks 1-3 before the sweep |
| `sweep.py` | All runs in parallel |
| `report.py` | Summary tables with bootstrap intervals |
| `tests/` | Unit tests |
| `results/` | Outputs (see its README) |
| `_vendor/` | Pinned export of the companion files (git-ignored) |
