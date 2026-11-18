# experiments/filter_family

Kalman filter family behind one interface (`filters.py`): KF, EKF, iterated EKF (IEKF), UKF and an
adaptive filter that estimates the measurement covariance R. The members are checked against each
other where theory requires agreement (linear model) and compared on a nonlinear range-bearing
benchmark.

## Interface

Each filter is built from a `StateSpaceModel` (`models.py`), an initial mean and covariance, and
optional filter values of Q and R. `predict()` returns the predicted `(x, P)`; `update(z)` returns
the posterior estimate, covariance, innovation and innovation covariance. Innovations and all
measurement differences use the model's residual function, which wraps the bearing to
`(-pi, pi]`.

## Models (`models.py`)

| Model | Description |
|-------|-------------|
| `linear_cv_model` | The two-state constant-velocity model of `experiments/linear_kf` (`dt = 0.1 s`, `q = 1`, `r = 1 m^2`). |
| `range_bearing_model` | State `[px, py, vx, vy]`, constant velocity with white-noise acceleration `q = 0.01 m^2/s^3` per axis, `dt = 1 s`. Measurement `[range, bearing]` from a sensor at the origin, `sigma_range = 0.5 m`, `sigma_bearing = 0.05 rad`, bearing `atan2(py, px)` in `(-pi, pi]`. Mean initial state `[-50, 20, 0.5, -1.0]`, `P0 = diag(25, 25, 1, 1)`. |

Range-bearing Jacobian, with `dx = px - sx`, `dy = py - sy`, `r = sqrt(dx^2 + dy^2)`:

    H = [[ dx/r,    dy/r,   0, 0],
         [-dy/r^2,  dx/r^2, 0, 0]]

These parameter values are chosen for illustration.

## Filters (`filters.py`)

* **KF**: the prediction and update steps of `experiments/linear_kf/kf.py` (`kf_predict`,
  `kf_update`, Joseph-form covariance update).
* **EKF**: `x <- f(x)`, `P <- F P F^T + Q` with `F` evaluated at the previous estimate; update
  with `H` evaluated at the prediction, innovation `nu = residual(z, h(x))`, then the same
  Joseph-form update as the KF.
* **IEKF**: Gauss-Newton iterations of the measurement update starting from `x_0 = x_pred`:
  `H_i = H(x_i)`, `r_i = residual(z, h(x_i)) - H_i (x_pred - x_i)`,
  `x_{i+1} = x_pred + K_i r_i` with `K_i = P_pred H_i^T (H_i P_pred H_i^T + R)^-1`, until
  `|x_{i+1} - x_i| <= tol` (default `1e-8`) or `max_iter` (default 10) iterations; the covariance
  uses the gain and Jacobian of the last iteration. The innovation and innovation covariance
  reported (and used for NIS) are those at the prior.
* **UKF**: scaled unscented transform with **alpha = 1, beta = 2, kappa = 0**, hence
  `lambda = alpha^2 (n + kappa) - n = 0`, weights `Wm_0 = 0`, `Wc_0 = 2`,
  `Wm_i = Wc_i = 1/(2n)`, sigma points `x +/- sqrt(n) L_i` with `L L^T = P`. With these values all
  mean weights are non-negative, which keeps the transform well conditioned (a small alpha such as
  `1e-3` gives a central weight of the order of `-1e6`). Sigma points are redrawn from the predicted
  mean and covariance before the update; the measurement mean uses the circular weighted mean
  `atan2(sum w sin(b), sum w cos(b))` for the bearing; the covariance update is
  `P <- P - K S K^T`.
* **Adaptive (AdaptiveEKF)**: an EKF (the KF on a linear model) whose filter value of R is
  re-estimated from post-fit residuals. After the update at step k, with posterior `(x+, P+)`, the
  residual `e_k = residual(z_k, h(x+_k))` is stored. Once `W` residuals are available
  (window `W = 20` by default), the R used for the next update is

      R_{k+1} = (1/W) sum_{j=k-W+1..k} e_j e_j^T + H(x+_k) P+_k H(x+_k)^T

  Before the window is full, the initial R is used. The rule follows from
  `E[e e^T] = R - H P+ H^T` for an optimal linear filter, and the estimate is positive
  semidefinite by construction. With `adapt = False` the filter is bitwise identical to the EKF
  (to the KF on the linear model).

## Sensitivity study protocol (`sweep.py`)

This protocol, including all grids, was fixed before any sweep was run; the grids are not changed
after seeing results.

### Common settings

* Range-bearing benchmark, 60 steps, Monte Carlo **N = 100**, seed 0.
* Filters: EKF, IEKF (`max_iter = 10`, `tol = 1e-8`), UKF (`alpha = 1`, `beta = 2`, `kappa = 0`),
  AdaptiveEKF (window 20, adaptation on, starting from the filter value of R of the setting).
* Every filter in a setting runs on the same trajectories and measurements. Within a sweep that
  changes the true system (nonlinearity sweeps), every setting is simulated with the same seed, so
  the underlying standard-normal draws are shared across settings (common random numbers).
* Filters start from the mean initial state of the setting with the filter value of P0; the true
  initial state is drawn from `N(x0_true, P0_true)`.

### Metrics per setting and filter

* **Divergence.** A run diverges if the filter raises a numerical error (for example a failed
  Cholesky factorisation in the UKF), produces a non-finite estimate, or its final position error
  exceeds **10 times the error of a single measurement fix** at the final step,
  `e_fix = sqrt(sigma_range^2 + (r_T sigma_bearing)^2)`, where `r_T` is the true final range.
  A tracking filter should do better than one measurement; an error ten times larger indicates
  that the track is lost. The criterion scales with the measurement geometry, so it remains
  meaningful when the bearing noise or the range changes.
* **Divergence rate:** fraction of diverged runs.
* **Position RMSE** (2D, over all steps), **mean NEES** (4 states, expected 4), **mean NIS**
  (2 measurements, expected 2) and **mean IEKF iterations** are computed over the non-diverged
  runs only; they are therefore always read together with the divergence rate.

### Mis-tuning sweeps (true system nominal; one filter quantity scaled)

`s` in `logspace(-2, 2, 9)` = 0.01, 0.0316, 0.1, 0.316, 1, 3.16, 10, 31.6, 100:

* `Q_filter = s Q_true`
* `R_filter = s R_true` (the adaptive filter starts from this value and then adapts)
* `P0_filter = s P0_true`

### Nonlinearity sweeps (filters given the true parameters of each setting)

1. **Bearing noise standard deviation** `sigma_bearing` [rad]:
   **0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5**.
   Reasoning: converting range and bearing to position is nonlinear through the bearing; a
   bearing error `e` shortens the along-range component by `r (1 - cos e)`, whose mean is about
   `r sigma_bearing^2 / 2`. At the initial range of about 54 m this bias equals the range noise
   (0.5 m) at `sigma_bearing = sqrt(2 * 0.5 / 54)`, about 0.14 rad. The grid runs from a regime
   where the bias is negligible (0.005 rad: about 0.7 mm) through this crossover to a strongly
   nonlinear regime (0.5 rad, about 29 degrees: cross-range standard deviation of about 27 m at
   54 m). The nominal value 0.05 rad is included.
2. **Initial position uncertainty** `sigma_p0` [m] (standard deviation of each position
   component in P0; velocity variance kept at 1 (m/s)^2): **0.5, 1, 2, 5, 10, 20, 30, 40**.
   Reasoning: the first updates linearise the measurement at the prior mean; the linearisation is
   accurate while the prior position spread is small compared with the range (about 54 m) and
   fails when the prior spans a large part of the bearing circle. The grid runs from a spread of
   about 1 % of the range (0.5 m) to about 75 % (40 m); larger values would place the sensor
   inside the prior, where the bearing is undefined. Only the position block is scaled so that the
   true trajectories stay comparable (a larger velocity spread would change the geometry over
   60 s). The nominal value 5 m is included.
3. **Closest-approach distance** `d` [m]: **1, 2, 3, 5, 10, 20, 50, 100**.
   Set-up: straight pass along +x at 1 m/s, mean initial state `[-30, d, 1, 0]`, closest approach
   at t = 30 s. To keep the realised pass close to the planned one, this sweep uses a small
   process noise `q = 1e-4 m^2/s^3` (position standard deviation of about 0.95 m after 30 s) and
   `P0 = diag(1, 1, 0.0025, 0.0025)` (velocity standard deviation 0.05 m/s); `sigma_range` and
   `sigma_bearing` are nominal. Reasoning: the bearing Jacobian scales with `1/r` and its rate of
   change with `1/r^2`, so linearisation errors grow sharply when the range becomes comparable to
   the position uncertainty (about 1 to 2 m here). The grid spans passes through or next to that
   uncertainty (1 to 3 m), a few uncertainties away (5 to 10 m) and far passes (20 to 100 m) where
   the bearing changes slowly. Because the realised trajectory deviates from the planned one by
   about 1 to 2 m, the realised minimum range per setting is reported alongside `d`.

All settings are reported, including those where the filters do not differ. Outputs:
`results/sweep_<name>.csv` and `figures/sweep_<name>.png`.

## Statistical closure protocol (`closure.py`)

Fixed before running; not changed after seeing results.

* Settings (as in the sweeps, N = 100, 60 steps, master seed 0): nominal; initial position
  standard deviation 40 m; bearing noise 0.5 rad; R x 100; Q x 0.01; P0 x 100.
* Per filter and setting: position RMSE, mean NEES, mean NIS with 95 % bootstrap intervals over the
  non-diverged runs, and the divergence count.
* Paired differences between every pair of filters (position RMSE and mean NEES), resampling runs
  jointly; runs in which any of the four filters diverged are excluded from the paired analysis
  of that setting (their number is reported).
* Seed-to-seed spread: the nominal setting is repeated with **5 master seeds** (0, 1, 2, 3, 4;
  master seed 0 reproduces the earlier results); the spread is reported as the five point estimates
  and their range.
* Intervals: bootstrap over runs as defined in `experiments/common/README.md` (B = 10 000, 95 %
  percentile, resampling seed 2026).
* Previously reported comparisons re-examined: nominal EKF / IEKF / UKF agreement and the slightly
  higher adaptive-filter RMSE and NEES; IEKF best at 40 m initial uncertainty; UKF best at 0.5 rad
  bearing noise; adaptive filter better at R x 100 and worse at Q x 0.01; UKF worse at P0 x 100.
* Outputs: `results/closure_summary.csv`, `results/closure_paired.csv`, `results/closure_seeds.csv`.

### References

* Residual-based adaptive estimation of R in the INS/GPS context: A. H. Mohamed and K. P. Schwarz,
  "Adaptive Kalman filtering for INS/GPS", *Journal of Geodesy*, 1999. Volume and page numbers:
  OPEN (to be verified). Whether the implemented windowed rule matches that paper's estimator
  equation exactly: OPEN (to be checked against the paper).
* References for the scaled unscented transform and the iterated EKF as a Gauss-Newton method:
  OPEN.

## Adaptive-R diagnosis protocol (`adaptive_r.py`)

Fixed before any of these runs; not changed after seeing results. Purpose: explain the behaviour
of `AdaptiveEKF` when the true measurement covariance is constant and the filter is given the true
Q and R (the nominal closure reports a mean NEES of 4.36 [4.17, 4.56] against an expected 4 on the
range-bearing benchmark).

### Configurations

| Configuration | Model | Steps T | Steady-state segment |
|---------------|-------|---------|----------------------|
| `linear_T400` | `linear_cv_model()` (`dt = 0.1 s`, `q = 1`, `r = 1 m^2`) | 400 | steps 201..400 |
| `rb_T400` | `range_bearing_model()` (nominal) | 400 | steps 201..400 |
| `rb_T60` | `range_bearing_model()` (nominal; the benchmark of the earlier closure) | 60 | not used |

* Monte Carlo **N = 100** runs per configuration, simulation seed **0**, generated with
  `models.simulate`. All variants of a configuration run on the same trajectories and
  measurements.
* True Q and R are given to every variant; the adaptive filter starts from the true R.
* The longer runs (T = 400) are added because a window of W steps leaves the filter non-adaptive
  for the first W updates; with T = 60, windows of 100 and 200 never fill and the adaptive filter
  is then identical to the non-adaptive one. `rb_T60` is kept only to connect with the earlier
  closure result; its windows 100 and 200 are reported but are by construction equal to "off".
  On the 400-step range-bearing runs the noise-free target reaches a range of about 405 m at the
  end (closest approach about 36 m at t = 36 s); the geometry is therefore not the same as in the
  60-step benchmark.

### Variants

* **off**: `AdaptiveEKF(adapt=False)` (identical to the EKF; to the KF on the linear model).
* **on, W**: `AdaptiveEKF(adapt=True, window=W)` for **W = 5, 10, 20, 50, 100, 200**.

### Metrics (per run, then over runs)

* Position RMSE (linear model: the position component; range-bearing: 2D position), mean NEES
  (full state; expected value n = 2 for the linear model, 4 for range-bearing), mean NIS (expected
  m = 1, 2), each computed over **all steps** and, for the T = 400 configurations, also over the
  **steady-state segment**.
* Divergence as defined in the sensitivity protocol (numerical error, non-finite estimate, or final
  position error above 10 times the single-fix error; for the linear model the single-fix error is
  `sqrt(r)`). Runs in which any variant of a configuration diverges are excluded from that
  configuration's paired analysis and their number is reported.
* Intervals: bootstrap over whole runs as in `experiments/common/README.md` (B = 10 000, 95 %
  percentile, resampling seed 2026).

### Analyses

a) **Adaptation off vs on**: paired differences (on minus off) in position RMSE, mean NEES and
   mean NIS for every window, with 95 % intervals, plus the per-variant values with intervals.
b) **Window sweep**: the same quantities as a function of W = 5, 10, 20, 50, 100, 200.
c) **Estimated R over time**: for each window, the R used at each update (`R_history`),
   normalised element-wise by the true diagonal, `R_hat_ii(k) / R_true_ii`; per step the mean,
   standard deviation and 5 % / 95 % quantiles over runs. **Steady-state bias** per diagonal
   element: per run, the mean of `R_hat_ii(k) / R_true_ii - 1` over the steady-state segment; then
   the mean over runs with a bootstrap interval. For range-bearing the steady-state mean of the
   off-diagonal correlation coefficient `R_hat_12 / sqrt(R_hat_11 R_hat_22)` is also reported.
   **Open-loop diagnostic**: the same windowed estimator evaluated on the residuals of the "off"
   filter (no feedback of the estimate into the filter), to separate a bias of the estimator itself
   from an effect of feeding the estimate back.
d) **Estimator documentation and checks**: the exact estimator as implemented (see the
   `AdaptiveEKF` description above), restated with the residual form, the treatment of the
   `H P H^T` term and the presence or absence of a positive-semidefinite projection. Counted over
   all adaptive updates: the number of estimates with a negative eigenvalue (the implementation has
   no projection, so this count is what a projection would act on). As a diagnostic only (not used
   by any filter), the innovation-form estimate `(1/W) sum nu_j nu_j^T - H P- H^T` is evaluated on
   the closed-loop innovations of the same runs, and the fraction of updates at which it has a
   negative eigenvalue is reported.

**Question answered**: whether the over-confidence (mean NEES above its expected value) decreases
with window length. Primary evidence: steady-state mean NEES of `rb_T400` and `linear_T400` versus
W; secondary: all-steps mean NEES of `rb_T60`. "Decreases" is called supported when the paired
difference of mean NEES between successive windows (larger minus smaller W) has an interval
entirely below zero; the result is stated per pair of successive windows.

Outputs: `results/adaptive_r_summary.csv`, `results/adaptive_r_paired.csv`,
`results/adaptive_r_bias.csv`, `results/adaptive_r_window_pairs.csv`,
`results/adaptive_r_psd.csv`, `results/adaptive_r_time.csv`; figures
`figures/adaptive_r_window.png`, `figures/adaptive_r_estimate_<configuration>.png`.

## Time-varying measurement noise benchmark protocol (`time_varying_r.py`)

Fixed before any of these runs; not changed after seeing results.

### Set-up

* Models: `linear_cv_model()` and `range_bearing_model()` (nominal), **T = 400** steps, Monte
  Carlo **N = 100**, simulation seed **0**. Q is true and fixed throughout.
* True measurement covariance `R_true(k) = c(k) R_nom` for k = 1..T, with `R_nom` the model's
  nominal R (for range-bearing both the range and the bearing variance are scaled by `c(k)`).
* Simulation: the same draw order as `models.simulate` (initial state, process noise, measurement
  noise); the standard-normal measurement draws are scaled by `sqrt(c(k))`. The same
  standard-normal draws are used for every pattern (common random numbers), and every filter of a
  pattern runs on the same trajectories and measurements.

### Patterns `c(k)` (fixed in advance)

* **step**: `c(k) = 1` for k <= 200, `c(k) = 10` for k > 200.
* **ramp**: `c(k) = 1 + 9 (k - 1) / (T - 1)`, i.e. linear from 1 at k = 1 to 10 at k = T.
* **burst**: `c(k) = 25` inside bursts, 1 elsewhere. Four bursts of 10 steps per run; the run is
  divided into four blocks of 100 steps and in each block the burst start is drawn uniformly from
  the first 91 steps of the block (so bursts never overlap or leave the block). Start times are
  drawn independently per run from a separate generator seeded with **1**, so they do not change
  the noise draws. The time average of `c` is `1 + 24 * 40 / 400 = 3.4` for every run.

### Filters (EKF; on the linear model this is the KF)

* **fixed nominal**: R = `R_nom`.
* **fixed average (hindsight reference)**: R = `c_bar R_nom`, with `c_bar` the time average of
  `c(k)` over the run (step 5.5, ramp 5.5, burst 3.4). It uses the true mean of `R_true(k)`, which
  is known only in hindsight; it is a reference, not a filter available in practice.
* **true-covariance filter**: R = `R_true(k)` at every update. It is the reference filter that
  knows the measurement covariance; it is not guaranteed to be best on every metric (for example,
  on a nonlinear model the EKF with the true R need not have the lowest RMSE or a NEES equal to
  its expected value).
* **adaptive, W**: `AdaptiveEKF` with W = 5, 10, 20, 50, 100, 200, starting from `R_nom`.

### Metrics (over all steps of a run, then over runs)

* Position RMSE, mean NEES, mean NIS, as in the adaptive-R protocol.
* Gaussian innovation negative log-likelihood per step,
  `0.5 (nu^T S^-1 nu + ln det S + m ln 2 pi)`, averaged over steps.
* **R tracking error**: per step and diagonal element, `ln(R_filter_ii(k) / R_true_ii(k))`, where
  `R_filter(k)` is the R used at update k. Reported as the mean absolute log-ratio (averaged over
  diagonal elements and steps; scale-free, zero for the true-covariance filter) and the mean signed log-ratio
  (bias; negative means the filter assumes less noise than present).
* Divergence as in the adaptive-R protocol (single-fix error evaluated with `R_true(T)`).
* Paired differences **filter minus true-covariance filter** for position RMSE, mean NEES, mean NIS and innovation
  NLL, resampling runs jointly; runs in which any filter of the pattern diverged are excluded from
  the paired analysis of that pattern and their number is reported. Intervals as above.

Outputs: `results/tv_r_summary.csv`, `results/tv_r_paired.csv`, `results/tv_r_time.csv`;
figures `figures/tv_r_<model>_<pattern>.png`.

### Amendments (reviewer, after the first results)

These amendments were requested by a reviewer after the first results of this benchmark had been
seen. They do not change the data, the patterns, the filters or the metrics above; the filter
names below replace the earlier names (`oracle` became `true_covariance`, `fixed_average` became
`fixed_average_hindsight`) in code, CSV files and figures.

* **A1.** The filter given `R_true(k)` is called the **true-covariance filter**; it need not win
  on every metric.
* **A2.** The fixed-average filter is labelled a **hindsight reference** that uses the true mean
  of `R_true(k)` and is not available in practice.
* **A3. Window selection on validation runs.** For each model and pattern, the adaptive filter is
  run with every window W on separate **validation runs** (N = 100, simulation seed **1000**,
  burst start seed **1001**; same patterns and filter settings). The window with the lowest mean
  innovation NLL (mean over runs of the per-run mean over steps, non-diverged runs) is selected
  and reported on the evaluation runs (seed 0) as `adaptive_selected`, alongside the full sweep.
  Output: `results/tv_r_window_selection.csv`.
* **A4.** The innovation NLL is `0.5 (nu^T S^-1 nu + ln det S + m ln 2 pi)` with `S` the filter's
  innovation covariance `H P- H^T + R` at that update (checked by a test against the Gaussian
  log-density).
* **A5. Identifiability of the R estimate** (Q true and fixed). For the R used at update k >= W,
  `R_hat(k) = A(k) + B(k)` with the residual part `A(k) = (1/W) sum e e^T` and `B(k) = H P+ H^T`
  of update k - 1. Per run and diagonal element, over the steps that use an estimate:
  the share of the temporal variance of `R_hat` carried by B, `Cov(R_hat, B) / Var(R_hat)` (the
  shares of A and B add to one); the correlation of `log R_hat` with `log R_true`; the correlation
  of `log R_hat` with `log B`; the mean of `B / R_hat`. Means over runs with bootstrap intervals.
  Rule: **follows R_true** if the interval of the B share lies below 0.5 and the interval of the
  correlation with `log R_true` lies above 0; **follows H P H^T** if the interval of the B share
  lies above 0.5; otherwise **not separated**. As a control the same analysis is run with
  constant R (`c(k) = 1`, evaluation seed 0), where any variation of `R_hat` is unrelated to a
  change of `R_true`. Output: `results/tv_r_identifiability.csv`.

## Files

| File | Purpose |
|------|---------|
| `models.py` | Linear and range-bearing models, Jacobians, residual and measurement mean, simulator. |
| `filters.py` | KF, EKF, IEKF, UKF, AdaptiveEKF and `run_filter`. |
| `run.py` | Nominal Monte Carlo comparison on the range-bearing benchmark; writes `results/`. |
| `sweep.py` | One-parameter-at-a-time sensitivity study (protocol above); writes `results/sweep_<name>.csv`. |
| `closure.py` | Statistical closure: bootstrap intervals, paired filter differences, seed spread. |
| `adaptive_r.py` | Adaptive-R diagnosis under constant true R (protocol above). |
| `time_varying_r.py` | Time-varying measurement noise benchmark (protocol above). |
| `plots.py` | Writes figures to `figures/`. |
| `tests/` | Equivalence, adaptation, Jacobian and wrap-around tests. |

## Running (from the repository root)

```bash
python -m pytest -v -s experiments/filter_family
python -m experiments.filter_family.run
python -m experiments.filter_family.sweep
python -m experiments.filter_family.closure
python -m experiments.filter_family.plots
python -m experiments.filter_family.adaptive_r
python -m experiments.filter_family.time_varying_r
```

Runtime per step is wall-clock time of `predict` + `update` in pure Python/numpy on the machine
used and is only meaningful as a relative comparison.
