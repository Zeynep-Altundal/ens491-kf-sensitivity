# experiments/peak_analysis

Step 6 of 8, analysis half: error peaks of the estimator sweep (`../estimators/`), their
disagreement between estimators, the effect of the corruption onset phase, and one pre-declared
diagnostic of the outage reversal.

Input: the per-run files of the final sweep, `../estimators/results/runs/<scenario>/<estimator>_seed<k>.npz`
(git-ignored there; regenerate with `.venv/bin/python -m experiments.estimators.sweep`), the
recovery times `../estimators/results/recovery.csv`, and the recordings through
`experiments.estimators.data`. Estimators: `KF` (companion KF-CV), `EKF` and `ESKF` (own filters,
provisional; see `../estimators/README.md`).

The protocol below (sections "Peak rule" to "Figures") was written on 2026-11-27 before any part
of this analysis was run. Changes after a run are listed under "Protocol changes".

## Labels

The labels of `../estimators/README.md` (Observed, Hypothesized, Verified), plus **Rejected**: a
stated hypothesis contradicted by a diagnostic whose decision rule was fixed before it was run.
The result tables contain Observed statements only.

## Peak rule (per run)

For one estimator, seed and scenario:

* `eps_s(t) = ||e_p(t)||`, the position error norm of the scenario run; `eps_n(t)` the same for the
  nominal run of the same estimator and seed (same measurement noise, no corruption).
* Excess error `D(t) = eps_s(t) - eps_n(t)`.
* Threshold `theta = max(0.1 m, RMS of eps_n over the flight)` (per estimator and seed).
* Above-threshold epochs `D(t) > theta` are grouped into segments; segments separated by less than
  0.2 s are merged; a merged segment lasting at least 0.2 s (40 IMU epochs, two position epochs)
  is a **peak**.
* Per peak: start and end time, peak time `t_pk = argmax D` inside the segment, height
  `eps_s(t_pk)` and excess `D(t_pk)`.
* Attribution: a peak belongs to the corruption event with the latest onset at or before its start
  (onset = `t_start` of the corruption; for spikes, the spike time; whole-flight noise scales:
  0 s). A peak that starts before every onset is "unattributed" and counted separately.
* Delay from onset: `t_pk - onset` of the attributed event.
* A run "has a peak" if it has at least one attributed peak. An estimator **shows a peak** in a
  scenario if at least 5 of the 10 seeds have a peak.
* Recovery time: the metric of `../estimators/README.md` (estimate back within 0.1 m of the
  nominal run's estimate for 2 s after the event end), read from `recovery.csv`; for a scenario
  with several events (repeated spikes) the largest recovery time over its events per seed.

## Tables (`results/tables.md`, Observed only)

1. **Peaks per scenario and estimator**: seeds with a peak (of 10); peaks per run (median); largest
   peak of the run: height and excess (median over seeds with a peak, range); delay from onset
   (median, range); recovery time (mean over seeds, or the number of seeds not recovered).
2. **Disagreement**: every scenario in which at least one estimator shows a peak and at least one
   other does not, with the seed counts.
3. **Bias effect by onset phase**: the accelerometer and gyro small bias steps (10 s) with onset in
   the hover, straight, figure-eight and aggressive phases; per estimator: seeds with a peak, the
   largest excess `max D` over `[onset, end of flight]` (mean over seeds with the 95 % bootstrap
   interval, `experiments/common/bootstrap.py`), delay of the largest peak (median), recovery time.
4. **IMU noise burst (x10, 2 s) by phase and estimator**: as table 3.

## Outage reversal diagnostic (pre-declared)

Observation (sweep, position RMSE over the flight): 40 s calm outage (8-48 s): `KF` 25.5 m, `EKF`
48.3 m; 40 s aggressive outage (42-82 s): `KF` 62.9 m, `EKF` 7.15 m.

**Hypothesis H.** (a) The `KF` outage error is the constant-velocity extrapolation error of the true
trajectory (term `T(t) = p(t) - p(t0) - v(t0)(t - t0)`, truth only), not its state error at the
outage onset; `T` is larger over the aggressive outage than over the calm one. (b) The `EKF` outage
error is the open-loop integration of its attitude and IMU-bias estimation errors at the outage
onset; the calm outage starts at 8 s, inside the 10 s convergence window of the sanity checks,
where these errors are larger than at 42 s.

**Design.** Per seed (10 seeds), for `outage_40s_calm` and `outage_40s_aggressive`. Outage window:
the IMU epochs from the outage start to the epoch before the first valid position update after it.
The state at the last epoch before the window, `x0`, comes from an in-memory float64 run of the
estimator on the scenario recording.

* `KF`: exact split of the error during the window, `e(t) = T(t) + (dp0 + dv0 (t - t0))`, with the
  onset errors `dp0`, `dv0` of the KF estimate.
* `EKF`: open-loop propagation over the window with the filter's own nominal step
  (`filters.ekf_step`, the same as the ESKF nominal step) and the recorded IMU samples, from
  * `C0`: `x0` unchanged (must reproduce the run's estimate within 1e-6 m; otherwise the
    diagnostic is invalid and stops),
  * `C1`: `x0` with attitude, gyro bias and accelerometer bias set to the truth at the onset,
  * reported only, not part of the decision: `C1a` (biases only to truth), `C1b` (attitude only),
    `C2` (position and velocity only).
* Quantity: RMS position error over the window, per seed; mean over seeds.

**Decision rule** (all must hold for "Verified"; any failure gives "Rejected" with the failing
part named):

1. `KF`, both outages: `RMS(T) / RMS(e_KF)` within [0.75, 1.25].
2. `RMS(T)` aggressive > `RMS(T)` calm.
3. `EKF`, calm outage: `RMS(C1) <= 0.25 RMS(C0)` (attitude and bias errors at the onset carry at
   least 75 % of the error).
4. `EKF`: the attitude-and-bias part `RMS(C0) - RMS(C1)` is larger in the calm outage than in the
   aggressive one.

## Figures (`build/`, git-ignored; `python -m experiments.peak_analysis.figures`)

* Position error norm against time, mean over the 10 seeds, one line per estimator plus the
  nominal run of each estimator (dashed), corruption windows shaded, mission phases as bands, for
  six scenarios: `accel_bias_step_large`, `gyro_bias_step_small_onset_aggressive`,
  `imu_noise_burst_x10_onset_figure_eight`, `pos_spike_single`, `outage_40s_calm`,
  `outage_40s_aggressive`.
* Disagreement heat map: scenarios x estimators, seeds with a peak (0-10).

## Protocol changes

None. (The rule for the recovery time of scenarios with several events was added to "Peak rule"
before the first run.)

## Results (2026-11-27)

Run on the final sweep (`../estimators/README.md`, "Sweep and results"): 930 scenario runs, 2102
peaks, analysis 21 s; diagnostic 30 s; figures 22 s. Tables: `results/tables.md`; every peak:
`results/peaks.csv`; one row per run: `results/run_peaks.csv`; diagnostic:
`results/outage_diagnostic.json`.

Observed (selection from `results/tables.md`):

* No peak starts before the onset of its scenario's corruption (0 unattributed peaks).
* Every disagreement (14 of 31 scenarios) is between the `KF` (no peak) and `EKF`/`ESKF` (peak in
  10 of 10 seeds): the accelerometer and gyro bias steps (all onsets), the accelerometer ramp and
  the whole-flight IMU noise x10. `EKF` and `ESKF` never disagree; the closest case is
  `outage_5s_aggressive` (10 and 9 of 10 seeds).
* No estimator shows a peak for `imu_noise_x3` and for the 2 s IMU noise burst in any phase (the
  `EKF`/`ESKF` have a peak in 1 of 10 seeds with the aggressive onset); the largest excess after
  the burst is 0.07-0.11 m on average.
* Small bias steps by onset phase (`EKF`, mean largest excess): accelerometer 0.71, 0.69, 0.68 and
  0.81 m for the hover, straight, figure-eight and aggressive onsets, mean recovery 32.6, 53.4, 36.0
  and 15.1 s; gyro 2.42, 2.34, 1.74 and 2.17 m, not recovered in 9, 7, 10 and 10 of 10 seeds.
* Position spikes and bursts: the `KF` peak is higher (single spike: median 4.59 m against 0.81 m)
  and its recovery shorter (1.6 s against 4.0 s) than for the `EKF`.

Outage reversal diagnostic (10 seeds, means; window 8.0-48.0 s and 42.0-82.0 s):

| Quantity | Calm outage | Aggressive outage |
|---|---|---|
| `KF` RMS error over the window [m] | 35.4 | 89.9 |
| trajectory term `T` [m] | 19.9 | 70.8 |
| `KF` onset-error term `dp0 + dv0 tau` [m] | 23.1 | 26.6 |
| `EKF` run = `C0` [m] (reproduced exactly, difference 0) | 66.7 | 8.77 |
| `C1`: attitude and biases true at onset [m] | 6.92 | 5.65 |
| `C1a`: biases true [m] / `C1b`: attitude true [m] | 40.6 / 39.8 | 6.63 / 8.13 |
| `C2`: position and velocity true [m] | 63.6 | 7.90 |
| `EKF` onset errors: tilt, yaw [deg]; gyro bias [rad/s]; accelerometer bias [m/s^2] | 0.57, 0.68; 0.0021; 0.060 | 0.12, 0.50; 0.00035; 0.015 |

Decision rule: (1) `RMS(T) / RMS(e_KF)` = 0.56 (calm) and 0.79 (aggressive), required [0.75, 1.25]:
**fails** for the calm outage; (2) `T` aggressive - calm = 50.9 m > 0: holds; (3) `C1 / C0` calm =
0.10 <= 0.25: holds; (4) attitude-and-bias part 59.8 m (calm) > 3.1 m (aggressive): holds.

**Outcome: H Rejected** (part (a), rule 1). Over the calm outage the `KF` error is not the
trajectory term alone: its onset-state term (23.1 m) is as large as the trajectory term (19.9 m).
Parts 2-4 of the rule hold; they are Observed results of the diagnostic, not a verified
explanation of the reversal, since H was stated and rejected as a whole.

## Files

| File | Purpose |
|---|---|
| `peaks.py` | Peak rule: threshold, segments, attribution |
| `analyze.py` | Peaks of every run and the tables |
| `outage_diagnostic.py` | Pre-declared outage reversal diagnostic |
| `figures.py` | Error-against-time figures and the disagreement heat map (`build/`) |
| `tests/` | Unit tests |
| `results/` | Tables and per-run outputs (see its README) |
| `build/` | Figures (git-ignored) |
