# experiments/eskf_sensitivity/tests

Validation tests for the ESKF: error-state transition `F` against finite differences of the
nominal propagation, noise-free scenarios, a 50-run Monte Carlo consistency check on the
`turning` profile (position RMSE, position NEES, NIS, bias convergence) and seed reproducibility.
The thresholds and their justification are documented in `test_eskf.py`.
