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
| `kf.py` | Standard Kalman filter (Joseph-form covariance update). |
| `sweep.py` | Monte Carlo sweeps; writes `results/sweep_{q,r,p0}.csv`. |
| `plots.py` | Writes figures to `figures/`. |
| `tests/` | Consistency and reproducibility tests. |

## Running (from the repository root)

```bash
python -m pytest -v -s experiments/linear_kf/tests
python -m experiments.linear_kf.sweep
python -m experiments.linear_kf.plots
```
