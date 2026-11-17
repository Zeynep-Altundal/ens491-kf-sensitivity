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

## Files

| File | Purpose |
|------|---------|
| `models.py` | Linear and range-bearing models, Jacobians, residual and measurement mean, simulator. |
| `filters.py` | KF, EKF, IEKF, UKF, AdaptiveEKF and `run_filter`. |
| `run.py` | Nominal Monte Carlo comparison on the range-bearing benchmark; writes `results/`. |
| `sweep.py` | One-parameter-at-a-time sensitivity study (protocol above); writes `results/sweep_<name>.csv`. |
| `closure.py` | Statistical closure: bootstrap intervals, paired filter differences, seed spread. |
| `plots.py` | Writes figures to `figures/`. |
| `tests/` | Equivalence, adaptation, Jacobian and wrap-around tests. |

## Running (from the repository root)

```bash
python -m pytest -v -s experiments/filter_family
python -m experiments.filter_family.run
python -m experiments.filter_family.sweep
python -m experiments.filter_family.closure
python -m experiments.filter_family.plots
```

Runtime per step is wall-clock time of `predict` + `update` in pure Python/numpy on the machine
used and is only meaningful as a relative comparison.
