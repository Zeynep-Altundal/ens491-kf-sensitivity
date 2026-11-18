# experiments/filter_family/figures

PNG figures written by `plots.py`:

| File | Content |
|------|---------|
| `nominal_example_trajectory.png` | Example run: truth, measurements, all estimates, sensor, and the point where the true bearing wraps. |
| `nominal_error_over_time.png` | Position RMSE over the Monte Carlo runs versus time for each filter. |
| `sweep_<name>.png` | Position RMSE and mean NEES (expected value 4 as reference) versus the swept parameter, one line per filter; a divergence-rate panel is added when any filter diverges in that sweep. |
| `adaptive_r_window.png` | Adaptive-R diagnosis: position RMSE, mean NEES and mean NIS with 95 % intervals versus window length for `linear_T400` (steady state), `rb_T400` (steady state) and `rb_T60` (all steps); band and line: adaptation off; dashed: expected value. |
| `adaptive_r_estimate_<configuration>.png` | R used at each update divided by the true R (diagonal elements), mean and 5-95 % band over runs, per window. |
| `tv_r_<model>_<pattern>.png` | Time-varying R benchmark (`linear`, `rb`; `step`, `ramp`, `burst`): filter R relative to the nominal value for run 0 and averaged over runs, against the true scale `R_true(k)`, and position RMSE over runs versus time; the fixed-average filter is labelled as a hindsight reference and the filter given `R_true(k)` as the true-covariance filter. |
| `lorenz_mse.png` | Lorenz-attractor benchmark: MSE [dB] versus `1/r^2` for all filters of this study, with the paper's EKF and UKF values as markers; full information and J = 2 mismatch. |
| `lorenz_example.png` | Lorenz-attractor run 0 at `1/r^2` = 20 dB: truth, observations, EKF and UKF estimates for the first 400 steps. |
| `lorenz_knet_training_curve.png` | Exploratory: training and validation MSE of the KalmanNet reference run (non-finite from the second step) with the reference EKF level. |
