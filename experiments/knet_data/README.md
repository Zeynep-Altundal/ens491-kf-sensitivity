# KalmanNet benchmark datasets

Datasets of the KalmanNet paper (Revach et al., IEEE TSP 2022, arXiv:2107.10043), generated with our
own code in one common format, so that KF, EKF, ESKF and KalmanNet run on exactly the same
trajectories.

| Script | Benchmark | Status |
|---|---|---|
| `linear.py` | 2x2 linear Gaussian model, paper Sec. IV-B | done |
| `sinusoidal.py` | toy non-linear model, paper eq. (17), Tables II–IV | done |
| Lorenz | uses `experiments/filter_family/lorenz.py` | to be exported to this format |
| NCLT | real Segway (UGV) data, session 2012-01-22 | to do |

## Common format (`common.py`)

One `.npz` file per scenario in `~/data/knet_benchmarks/` (override with `KNET_DATA_DIR`).
Datasets are not committed.

- `x_train, y_train, x_val, y_val, x_test, y_test`: states `(N, T, n)` and measurements `(N, T, m)`;
  N = 1000 / 100 / 200 as in the reference code.
- `x0`: initial state; `F_true, H_true` generate the data; `F_model, H_model` are given to the filters.
- `Q, R`: noise covariances; `meta`: JSON with scenario, seed, T, nu, 1/r^2 and source.
- Each split has its own random stream (`SeedSequence(seed).spawn(3)`).

## Linear benchmark (`linear.py`)

x_t = F x_{t-1} + w_t, y_t = H x_t + v_t, Q = q^2 I, R = r^2 I, nu = q^2 / r^2, r^2 = 1, x0 = 0.
F0 = [[1, 1], [0, 1]] (controllable canonical form), H0 = I.

| Scenario | T | nu [dB] | Data generated with | Filters get | Paper |
|---|---|---|---|---|---|
| `full` | 100 | -20 | F0, H0 | F0, H0 | Fig. 5a setting (reference-code default) |
| `rotF10`, `rotF20` | 20 | 0 | Rot(alpha) F0, H0 | F0, H0 | Fig. 6a, eq. (16) |
| `rotH10` | 100 | -20 | F0, Rot(10 deg) H0 | F0, H0 | Fig. 6b |

The reference code multiplies the rotation on the right and, in `architecture-#1`, gives the rotated
matrix to the filter. We follow the paper (data from the rotated matrix, filters get F0 / H0).

Run from the repository root:

```
python -m experiments.knet_data.linear
python -m pytest -v experiments/knet_data
```

`results/linear_check.csv` compares, on the test split, the KF with the true model (optimal; its
empirical MSE must match the MSE predicted by its covariance), the KF with the given model
(baseline under mismatch) and the measurement-only estimate (noise floor).

## Sinusoidal benchmark (`sinusoidal.py`)

f(x) = alpha sin(beta x + phi) + delta, h(x) = a (b x + c)^2, per component, x in R^2, T = 100,
nu = -20 dB, 1/r^2 in {-12.04, -6.02, 0, 20, 40} dB (one file per level). Data always come from the
"full" parameters (0.9, 1.1, 0.1 pi, 0.01, 1, 1, 0); the "partial" model (1, 1, 0, 0, 1, 1, 0) is
given to the filters in the mismatch case. x0 = [0.1, 0.1], P0 = 0 from the reference code. The
reference code uses delta = 1 and nu = 0 dB; we follow the paper.

`results/sinusoidal_check.csv` compares our EKF with the paper:

- full model (Table III): within 0.11 dB at every noise level, and the spread over trajectories
  matches the paper's sigma-hat. This validates the generator.
- partial model with the true Q: far worse than Table IV at 20 and 40 dB (about -7.3 dB). With a
  wrong f the filter trusts its model and ignores the precise measurements.
- partial model with Q tuned on the validation split (grid 1e-2 ... 1e8 times the true Q): within
  1.3 dB of Table IV everywhere (0.4 dB at 20 and 40 dB). The paper does not describe its EKF set-up
  for Table IV; a tuned Q is the only way we found to reach its numbers.
  
### Open point

With r^2 = 1 the mismatched KF in `rotF10` / `rotF20` is about 33 / 51 dB worse than the KF with the
true model: Rot(alpha) F0 has an eigenvalue of 1.46 / 1.59, so the state grows quickly over T = 20
steps and a wrong F gives large prediction errors. The paper reports only "a 3 dB gain" of
KalmanNet over the model-based KF in Fig. 6a and gives no numbers; this has to be checked against
the figure before comparing.
