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
| `plots.py` | Writes figures to `figures/`. |
| `tests/` | Equivalence, adaptation, Jacobian and wrap-around tests. |

## Running (from the repository root)

```bash
python -m pytest -v -s experiments/filter_family
python -m experiments.filter_family.run
python -m experiments.filter_family.plots
```

Runtime per step is wall-clock time of `predict` + `update` in pure Python/numpy on the machine
used and is only meaningful as a relative comparison.
