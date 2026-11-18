# experiments/neural_r/figures

PNG figures written by `train.py` and `evaluate.py`:

| File | Content |
|------|---------|
| `training_curves.png` | Training NLL per iteration and pooled validation NLL every 25 iterations for the six GRU and six MLP configurations. |
| `eval_<test set>.png` | Run 0: R used by the true-covariance filter (`R_true(k)`), the burst-selected adaptive filter, the GRU and the MLP, relative to `R_base`; position RMSE over runs for all filters. |
| `eval_nll_differences.png` | Paired difference in mean innovation NLL of GRU and MLP against the burst-selected adaptive filter, the fixed nominal R filter and the true-covariance filter, per test set, with 95 % intervals. |
| `saturation_alpha_vs_factor.png` | Amendment 1: alpha of the selected GRU and MLP versus the true factor (settled steps) on the burst test sets, with the line alpha = factor and the architectural bound. |
| `saturation_training_curves.png` | Training and validation NLL of the selected GRU and MLP with the least-squares slope over iterations 351..400. |
