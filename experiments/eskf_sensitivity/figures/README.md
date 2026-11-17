# experiments/eskf_sensitivity/figures

PNG figures written by `plots.py` for a single run and for the sweeps:

| File | Content |
|------|---------|
| `<profile>_run<i>_trajectory.png` | Top view: truth, measurements and estimate. |
| `<profile>_run<i>_errors.png` | Position, velocity and attitude errors with filter 3-sigma bounds. |
| `<profile>_run<i>_biases.png` | Gyro and accelerometer bias estimates with 3-sigma bounds versus truth. |
| `<profile>_run<i>_nis.png` | NIS per update with the chi-square(3) 95 % interval. |
| `sweep_<name>.png` | Position RMSE, attitude RMSE and mean position NEES (expected value 3 as reference) versus the swept parameter, one line per variant. |
| `sweep_dropout_error_over_time.png` | RMS position error over time for each dropout duration, with the dropout window shaded, the no-dropout reference and the 3-sigma bound from the filter covariance. |
