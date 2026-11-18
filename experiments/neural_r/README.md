# experiments/neural_r

Evaluation protocol for a future learned, time-varying measurement covariance `R_t`. This folder
contains the protocol only; no model is implemented or trained here. The protocol is fixed before
any learned model exists and is not changed after seeing results.

## Task

At each update k the learned model outputs a measurement covariance `R_hat(k)` (symmetric positive
definite, of the size of the measurement), computed only from information available to the filter
before the update at k (past measurements, past innovations and post-fit residuals, past filter
covariances). The EKF (the KF on the linear model) uses `R_hat(k)` in place of its fixed R. Q is
true and fixed throughout, as in the time-varying benchmark of `experiments/filter_family`.

## Models and data generation

* Models: `linear_cv_model()` and `range_bearing_model()` of `experiments/filter_family/models.py`
  (nominal parameters), **T = 400** steps.
* Measurement noise `R_true(k) = c(k) R_nom` and the simulation procedure of the time-varying
  measurement noise benchmark protocol in `experiments/filter_family/README.md`.
* Seed separation: training and validation data use simulation seeds **1000 to 1999**; the test
  sets use seed **0** (identical to the benchmark data, so results are directly comparable with
  the benchmark tables) and, for the held-out patterns, seeds **2000 to 2099**. No test seed is
  used for training, validation, early stopping or model selection.

## Training patterns

Factors and timings in the training distribution:

* **step**: one jump by a factor drawn from {2, 5, 10}, at a step drawn uniformly from 100..300;
  jumps upwards and downwards (factor `f` or `1/f`) with equal probability.
* **ramp**: linear change from 1 to a factor drawn from {2, 5, 10} over the run, increasing or
  decreasing with equal probability.
* **burst**: bursts of factor drawn from {10, 25}, length drawn from {5, 10, 20} steps, two to
  six bursts per run, non-overlapping, start times uniform.
* **constant**: `c(k) = 1` throughout (so that the model is also trained on the case where
  adaptation is unnecessary).

## Test sets

1. **In-distribution**: the three benchmark patterns exactly as defined in the time-varying
   benchmark (step x10 at k = 200, ramp 1 to 10, four bursts x25 of 10 steps), seed 0.
2. **Held-out factors and timings** (not seen in training):
   * step by a factor of **20** at **k = 60** and step down by a factor of 10 at k = 340;
   * ramp from 1 to **30**;
   * bursts with factor **50** and length **3** steps, and bursts with factor 25 and length **40**
     steps.
3. **Held-out pattern type** (not seen in training): **sinusoidal** modulation
   `c(k) = 10^(0.5 + 0.5 sin(2 pi k / 100))` (between 1 and 10 with a period of 100 steps).
4. **Constant R** at the nominal value (to measure the cost of a learned model when no adaptation
   is needed; compare with the adaptive-R diagnosis).

Each held-out test set uses N = 100 Monte Carlo runs.

## Baselines

The baselines of the time-varying benchmark, on the same data:

* fixed-R filter with the nominal R;
* fixed-R filter with the time-averaged R of the pattern (hindsight reference: it uses the true
  mean of `R_true(k)` and is not available in practice);
* true-covariance filter given `R_true(k)` (reference that knows R; it need not win on every
  metric);
* adaptive-R filter (`AdaptiveEKF`) with window W = 5, 10, 20, 50, 100, 200, and the window
  selected on validation runs by innovation NLL (as in amendment A3 of the benchmark protocol).

## Metrics

As in the time-varying benchmark: position RMSE, mean NEES, mean NIS, Gaussian innovation negative
log-likelihood, R tracking error (mean absolute and mean signed log-ratio of the diagonal of the R
used against `R_true(k)`), divergence count. Paired differences to the true-covariance filter, and between the
learned model and the best adaptive baseline, with 95 % bootstrap intervals over whole Monte Carlo
runs (`experiments/common/README.md`: B = 10 000, percentile, resampling seed 2026). Runs in which
any compared filter diverges are excluded from the paired analysis and counted.

## Equal tuning budget

The learned model and the adaptive baseline receive the **same tuning budget**:

* Both are tuned only on the training and validation seeds (1000 to 1999), never on test data.
* The budget is counted as the number of configurations evaluated on validation data (for the
  learned model: hyperparameter configurations, each trained to completion; for the adaptive
  baseline: window lengths and any other estimator settings). Both get the same maximum number of
  configurations; the number used is reported.
* The six window lengths above are reported in every case; the window selected on validation data
  under the budget is the "tuned adaptive" baseline compared with the learned model.
* The selection criterion is fixed in advance: mean innovation negative log-likelihood on the
  validation set.

## Reporting rules

* All test sets and all baselines are reported, including those where the learned model is worse.
* Training cost (wall-clock time, number of parameters) and per-step inference cost are reported.
* Open items: architecture, input features and training loss are not part of this protocol and
  are OPEN; they must be fixed and documented before the first test-set evaluation.
