# experiments/linear_kf

Layer A of the study: behaviour of a standard linear Kalman filter when exactly one tuning quantity
is mis-specified, on a linear Gaussian state-space model in the style of the linear synthetic
experiments used in the KalmanNet literature.

## Model

    x[k+1] = F x[k] + w[k],   w[k] ~ N(0, Q_true)
    y[k]   = H x[k] + v[k],   v[k] ~ N(0, R_true)

Default: two-state constant-velocity model, state `x = [position, velocity]`, `dt = 0.1 s`,
position measured, `T = 200` steps.

| Quantity | Default value |
|----------|---------------|
| `F` | `[[1, dt], [0, 1]]` |
| `H` | `[[1, 0]]` |
| `Q_true` | `q * [[dt^3/3, dt^2/2], [dt^2/2, dt]]`, `q = 1` (white-noise acceleration) |
| `R_true` | `r = 1 m^2` |
| `x0_true` | `[0 m, 1 m/s]` |
| `P0_true` | `diag(1, 1)` |

These defaults are chosen for illustration and are not derived from a specific sensor.

The true initial state is drawn from `N(x0_true, P0_true)`; the filter is initialised at `x0_true`.
Measurements are available for `k = 1, ..., T`. All randomness comes from a seeded
`numpy.random.Generator`.

## Sweeps

One quantity is scaled by `s` in `logspace(-3, 3, 13)` while everything else stays at truth:
`Q_filter = s Q_true`, `R_filter = s R_true`, `P0 = s P0_true`. Each setting is evaluated on the
same `N = 200` Monte Carlo trajectories. Recorded metrics, averaged over all trajectories and steps:
position RMSE, velocity RMSE, mean NEES (expected value `n = 2` for a consistent filter) and mean
NIS (expected value `m = 1`).

## Files

| File | Purpose |
|------|---------|
| `model.py` | Model definition and trajectory simulation. |
| `kf.py` | Standard Kalman filter (Joseph-form covariance update); step functions `kf_predict` and `kf_update` are reused by `experiments/filter_family`. |
| `sweep.py` | Monte Carlo sweeps; writes `results/sweep_{q,r,p0}.csv`. |
| `ratio_study.py` | Ratio versus absolute-scale experiments; writes `results/open_point.csv` and `results/grid_*.csv`. |
| `plots.py` | Writes figures to `figures/`. |
| `tests/` | Consistency and reproducibility tests. |

## Ratio versus absolute scale

Scaling all filter covariances `(Q_f, R_f, P0_f)` by a common factor `c` leaves the Kalman gains,
the estimates and the RMSE unchanged and multiplies NEES and NIS by `1/c`. The estimates therefore
depend only on the ratios `Q_f/R_f` and `P0_f/R_f`. This identity is checked by
`test_common_scale_invariance`. `ratio_study.py` adds:

* the configurations `Q x 0.001`, `R x 1000`, `R x 1000, P0 x 1000` and their mirrored
  counterparts, evaluated on the full `T = 200` run and on the steady-state window `k >= 1000` of a
  `T = 2000` run, together with the step at which the gain stays within 1 % of its steady-state
  value;
* a 13 x 13 grid over the Q and R scale factors (P0 at truth) for both windows, and the spread of
  each metric along lines of constant `Q_f/R_f`.

## Running (from the repository root)

```bash
python -m pytest -v -s experiments/linear_kf/tests
python -m experiments.linear_kf.sweep
python -m experiments.linear_kf.ratio_study
python -m experiments.linear_kf.plots
```
