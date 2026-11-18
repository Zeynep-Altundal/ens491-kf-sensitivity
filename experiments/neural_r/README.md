# experiments/neural_r

Design and evaluation of a learned, time-varying measurement covariance `R_t` for the Kalman
filter, and short diagnostics of the adaptive-R baseline that motivate it. This document was
written before any training or diagnostic run of this folder; its SHA-256 and the time of the
freeze are recorded in `DESIGN_FREEZE.txt`. Any later change is listed under "Post-result
amendments" at the end.

This version refines the protocol of the previous commit (`d3e7cb2`) before any result was
produced: the seeds are reassigned so that training, validation and test sets are disjoint from the
validation runs of the time-varying benchmark, the number of bursts of the held-out burst sets is
fixed, and the model, features, loss, budget and gates are specified.

The validated baseline filters of `experiments/filter_family` are imported unchanged; all code
specific to the learned model lives in this folder.

## Part 0 - Diagnostics of the adaptive-R baseline (`diagnostics.py`)

### 0a. Over-confidence of the adaptive filter with W = 5 under constant R_true

Set-up: `linear_cv_model()` and `range_bearing_model()` (nominal), T = 400, N = 100, simulation
seed 0, constant `R_true = R_nom`, true Q; windows **W = 5, 20, 100**; metrics over the
steady-state segment (steps 201..400): mean NEES, mean NIS, position RMSE. All conditions run on the
same measurement sequences.

Conditions (EKF; the KF on the linear model):

| Condition | R used at update k |
|-----------|--------------------|
| `off` | `R_true` (non-adaptive filter) |
| `closed` | `AdaptiveEKF` with window W (estimate fed back) |
| `open_replay` | the windowed estimate computed from the post-fit residuals and `H P+ H^T` of the `off` filter of the **same run** (frozen, non-adaptive filter trajectory), used one step later as in the adaptive filter; no feedback |
| `indep_open_replay` | the `open_replay` sequence of **another run** (run i uses run (i + 1) mod N): same statistics, independent of the current run's data |
| `indep_closed_replay` | the `closed` R sequence of another run (run (i + 1) mod N) |

Before the window is full every condition uses `R_true`.

Exact decomposition of the excess NEES of the adaptive filter (paired over runs):

    closed - off = (closed - open_replay)                 [feedback]
                 + (open_replay - indep_open_replay)      [correlation of the estimate with the
                                                           current run's own data]
                 + (indep_open_replay - off)              [fluctuation of R alone]

Additionally, the spread of the estimate: per run the standard deviation over steady-state steps of
`ln R_hat_ii`, averaged over runs, for `closed` and `open_replay`.

Hypotheses and labelling rules (each label refers to W = 5; W = 20 and 100 are reported for
comparison):

1. **High variance of the covariance estimate from the short window.** The spread is measured
   directly; the label is **Observed** if the spread at W = 5 exceeds that at W = 20 and W = 100
   (descriptive; its effect on NEES is tested under 3).
2. **Feedback through the post-fit residual.** **Verified** if the 95 % interval of
   `closed - open_replay` (mean NEES) lies above zero; otherwise **Hypothesized, not supported by
   the control**.
3. **Fluctuations of the estimate degrading NEES through the Kalman gain.** **Verified** if the
   95 % interval of `indep_open_replay - off` (mean NEES) lies above zero: an R sequence with the
   statistics of the estimate but independent of the current data and without feedback still
   raises NEES. Otherwise **Hypothesized, not supported by the control**.

Outputs: `results/diag_overconfidence.csv` (per condition and W), `results/diag_decomposition.csv`
(paired differences with intervals), `results/diag_spread.csv`.

### 0b. Identifiability: residual part and `H P H^T` part logged separately

Same measurement sequences as the time-varying benchmark (patterns constant, step, ramp, burst;
seed 0, burst seed 1), both models, W = 5, 20, 100. For the R used at update k >= W,
`R_hat = A + B` with `A = (1/W) sum e e^T` and `B = H P+ H^T` (of update k - 1). Both parts are
logged:

* in **closed loop** (`AdaptiveEKF`), and
* on the **frozen trajectory** (`off` filter with the nominal R; the same estimator applied to its
  residuals and covariances, never fed back).

Reported per pattern, W and diagonal element (means over runs with intervals): the correlation of
`ln A` with `ln R_true`, the mean of `B / (A + B)`, the coefficient of variation of B over time
(standard deviation over steps divided by the mean), for closed loop and frozen trajectory, and
the mean ratio `B_closed / B_frozen`. On the linear model with fixed R the covariance P of the
frozen filter does not depend on the measurements, so any variation of B in closed loop beyond the
frozen one is caused by feedback. Output: `results/diag_identifiability.csv`.

## Part 1 - Design of the learned covariance

### Model

    R_t = diag(alpha_t) R_base,    alpha_t = softplus(f_theta(features up to t - 1)) + eps

with `R_base` the nominal R of the model, `eps = 1e-3`, `alpha_t` of the measurement dimension m.
`R_t` is diagonal and positive definite by construction. The output layer bias is initialised to
`ln(e - 1)` so that `alpha` starts at about 1 (that is, at `R_base`).

### Candidates (fixed sizes)

* **GRU**: one GRU layer, hidden size **16**, followed by a linear layer 16 -> m.
* **MLP** (control without recurrent memory): input the features of the last **L = 20** steps
  (zero-padded at the start), flattened (`20 * 3m`), two hidden layers of **32** units with tanh,
  linear output layer 32 -> m.

No other architecture is trained.

### Inputs and inference timing

Per completed update j the feature vector has 3m entries:

* innovation scaled by the base standard deviation, `nu_j,i / sqrt(R_base,ii)`;
* normalised innovation `nu_j,i / sqrt(S_j,ii)`;
* `ln(S_j,ii / R_base,ii)`.

Timing at step t: (1) prediction; (2) `alpha_t` is computed from the features of steps 1..t-1 only
(GRU: hidden state after step t - 1; MLP: features of steps t - 20..t - 1); for t = 1 the input is
zero; (3) innovation `nu_t` and `S_t = H P_t|t-1 H^T + R_t`; (4) update (Joseph form); (5) the
features of step t are formed. `R_true(t)` is never an input and never a training label; the
pattern type is not an input.

### Loss

Innovation negative log-likelihood with the filter's innovation covariance,
`0.5 (nu_t^T S_t^-1 nu_t + ln det S_t + m ln 2 pi)`, averaged over steps and trajectories,
back-propagated through the filter over the full trajectory (T = 400 steps, no truncation). The
differentiable filter is first checked to reproduce `experiments/linear_kf/kf.py` to numerical
precision (test in `tests/`).

### Models covered

The linear model `linear_cv_model()` (`dt = 0.1 s`, `q = 1`, `r = 1 m^2`, m = 1), T = 400, Q true
and fixed. The range-bearing benchmark is repeated only if the linear prototype is complete and
time allows; otherwise this is stated.

### Data and seeds (disjoint)

* **Training**: generated on the fly; iteration `it` (0..399) draws a batch of 32 trajectories from
  `numpy.random.default_rng(1100 + it)` (pattern parameters first, then the simulation as in
  `filter_family.time_varying_r.simulate_tv`). Pattern mix (development focus on bursts):
  **burst 40 %, step 20 %, ramp 20 %, constant 20 %**, with
  * step: factor in {2, 5, 10}, jump at a step uniform in 100..300, up (`f`) or down (`1/f`) with
    equal probability;
  * ramp: linear from 1 to a factor in {2, 5, 10}, increasing or decreasing (from the factor to 1)
    with equal probability;
  * burst: factor in {10, 25}, length in {5, 10, 20}, 2 to 6 bursts, non-overlapping, uniform starts;
  * constant: `c = 1`.
  Training seeds: 1100..1499.
* **Validation**: the validation runs of the time-varying benchmark (amendment A3 there):
  patterns constant, step, ramp, burst of the benchmark definitions, N = 100 each, simulation seed
  1000, burst start seed 1001.
* **Test** (N = 100 each):

| Set | Pattern | Seeds (simulation, burst starts) |
|-----|---------|----------------------------------|
| `constant` | `c = 1` | 0 |
| `step` | benchmark step (x10 at k = 200) | 0 |
| `ramp` | benchmark ramp (1 to 10) | 0 |
| `burst` | benchmark burst (4 x 25, length 10) | 0, 1 |
| `ho_step` | `c = 1` for k <= 60, 20 for 61 <= k <= 340, 2 for k > 340 | 2000 |
| `ho_ramp` | linear from 1 to 30 | 2001 |
| `ho_burst_short` | 4 bursts, factor 50, length 3, one per 100-step block, start uniform | 2002, 2052 |
| `ho_burst_long` | 4 bursts, factor 25, length 40, one per 100-step block, start uniform | 2003, 2053 |
| `ho_sine` | `c(k) = 10^(0.5 + 0.5 sin(2 pi k / 100))` (unseen pattern type) | 2004 |

The held-out sets use factors, burst lengths and timings not in the training distribution, and
one unseen pattern type (sinusoidal).

### Baselines (same data)

* fixed nominal R (the non-adaptive filter);
* fixed average R (hindsight reference: the time average of `R_true(t)` of the run; not available
  in practice);
* true-covariance filter given `R_true(t)` (a reference that knows R; it is not a performance
  bound and need not be best on every metric);
* adaptive filter (`AdaptiveEKF`) with the window selected on validation runs by innovation NLL:
  `adaptive_burst_sel` selected on the burst validation runs (the baseline of gate G1) and
  `adaptive_pooled_sel` selected on all four validation patterns pooled.

### Equal tuning budget

Each of the three methods gets **6 configurations**, selected by validation NLL only:

* adaptive filter: W in {5, 10, 20, 50, 100, 200};
* GRU and MLP each: learning rate in {1e-3, 3e-3, 1e-2} x weight decay in {0, 1e-4} (Adam).

Fixed for all learned configurations: float64, batch 32, 400 iterations, gradient-norm clipping at
1.0, parameter initialisation with `torch.manual_seed(0)`, validation NLL (pooled over the four
validation patterns) evaluated every 25 iterations and at the end; the checkpoint with the lowest
validation NLL is kept. The configuration with the lowest validation NLL is selected per candidate.
Test data are never used for any choice. If training fails, it is reported as is; it is not
repeated with a different budget.

### Metrics

Position RMSE, mean NEES (expected 2), mean NIS (expected 1), innovation NLL, R tracking error
(mean absolute `ln(R_t,ii / R_true,ii(t))`), and **recovery time** after a burst: for each burst
ending at step `k_e`, the smallest `j >= 1` with `|ln(R_11(k_e + j) / R_true,11(k_e + j))| <= ln 2`,
censored at the start of the next burst or the end of the run (censored values count with the
censoring length; the censored fraction is reported); per run the mean over bursts. Paired
differences learned minus baseline, and GRU minus MLP, with 95 % bootstrap intervals over whole
runs (`experiments/common/README.md`).

### Gates (fixed in advance)

Evaluated for the selected GRU and the selected MLP separately.

* **G1** held-out bursts (`ho_burst_short`, `ho_burst_long`): paired NLL difference learned minus
  `adaptive_burst_sel`; **passed** if the 95 % interval lies entirely below zero on both sets,
  **failed** otherwise (an interval entirely above zero is reported as the learned model being
  worse).
* **G2** NEES and NIS reported for every test set and baseline; every case where the learned model
  is farther from the expected value than a baseline is listed. Passed when reported.
* **G3** negative control, `constant` test set: paired differences learned minus fixed nominal R
  (the non-adaptive filter, here with the true R). **Passed** if the interval for RMSE does not lie
  entirely above zero and the intervals for NEES and NIS both contain zero.
* **G4** R tracking error against `R_true(t)` on every test set, compared with the adaptive
  baselines; reported to separate learning the covariance from fitting innovation patterns
  (evaluated when reported; no threshold).
* **G5** recovery time after bursts on `burst`, `ho_burst_short`, `ho_burst_long` (evaluated when
  reported; no threshold).
* **G6** GRU versus MLP: paired NLL difference GRU minus MLP on every test set; temporal memory
  beyond the 20-step window is called **contributing** where the interval lies entirely below zero.

No claim of superiority is made beyond the gate outcomes.

## Files

| File | Purpose |
|------|---------|
| `diagnostics.py` | Part 0 diagnostics of the adaptive-R baseline. |
| `dkf.py` | Differentiable Kalman filter (PyTorch) for the linear model and the learned-R models. |
| `data.py` | Training pattern generator and test-set definitions. |
| `train.py` | Training of the GRU and MLP configurations under the budget above. |
| `evaluate.py` | Evaluation on the test sets against the baselines; gates. |
| `requirements.txt` | Additional dependency of this folder (PyTorch, CPU build). |
| `DESIGN_FREEZE.txt` | SHA-256 and timestamp of this document at the freeze. |
| `results/`, `figures/` | Outputs. |
| `tests/` | Equivalence test of the differentiable filter. |

## Running (from the repository root, with PyTorch installed)

```bash
pip install -r experiments/neural_r/requirements.txt --index-url https://download.pytorch.org/whl/cpu
python -m experiments.neural_r.diagnostics
python -m experiments.neural_r.train
python -m experiments.neural_r.evaluate
```

## Post-result amendments

None.
