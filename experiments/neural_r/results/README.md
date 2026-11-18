# experiments/neural_r/results

CSV output of `diagnostics.py`, `train.py` and `evaluate.py` (linear model, protocol in `../README.md`).

| File | Content |
|------|---------|
| `diag_overconfidence.csv` | Part 0a: steady-state RMSE, NEES, NIS with intervals per model, window and control condition (`off`, `closed`, `open_replay`, `indep_open_replay`, `indep_closed_replay`). |
| `diag_decomposition.csv` | Part 0a: paired differences of the decomposition of the excess NEES and NIS (feedback, own-data correlation, fluctuation). |
| `diag_spread.csv` | Part 0a: mean over runs of the standard deviation over time of `ln R_hat`, closed loop and open loop. |
| `diag_identifiability.csv` | Part 0b: residual part A and `H P+ H^T` part B, closed loop and frozen trajectory: correlation of `ln A` with `ln R_true`, mean `B / (A + B)`, coefficient of variation of B, ratio `B_closed / B_frozen`. |
| `training_curves.csv` | Training NLL per iteration and validation NLL every 25 iterations (iteration 0 for the curves only), per configuration. |
| `training_summary.csv` | Per configuration: status, best validation NLL and iteration, training time, number of parameters, selected flag. |
| `selected_models.json` | Configuration selected per candidate by validation NLL. |
| `checkpoints/` | Parameters (best validation checkpoint) of every configuration. |
| `neural_r_adaptive_selection.csv` | Validation NLL of the adaptive filter per window (burst runs and pooled) and the selected windows. |
| `neural_r_summary.csv` | Per test set and filter: divergence count, position RMSE, NEES, NIS, NLL, R tracking error, recovery time (burst sets, with censored fraction), with intervals. |
| `neural_r_paired.csv` | Paired differences learned minus every baseline, and GRU minus MLP, with intervals. |
| `neural_r_gates.csv` | Outcome of gates G1 to G6. |
| `neural_r_g2_cases.csv` | Every case where the NEES or NIS point estimate of a learned model is farther from its expected value than that of a baseline. |
