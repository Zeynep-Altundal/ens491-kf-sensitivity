# experiments/eskf_sensitivity/results

CSV output of `run.py` and `sweep.py` (protocol in `../README.md`).

| File | Content |
|------|---------|
| `<profile>_run<i>_history.csv` | One row per measurement epoch: true position, measurement, estimate, error state and filter standard deviations (15 each), true and estimated biases, NIS, position NEES, availability and acceptance flags. |
| `<profile>_run<i>_summary.csv` | Position, velocity and attitude RMSE (3D), mean position NEES, mean NIS, number of updates, maximum position and attitude error. |
| `sweep_<name>.csv` | One row per grid value and variant: number of successful and failed runs, position, velocity and attitude RMSE, mean position NEES, mean NIS (accepted updates), gate rejection rate, RMS final gyro and accelerometer bias error, and for `dropout` the mean and maximum over runs of the maximum position error inside the window. Sweeps: `q_imu`, `q_bias`, `r`, `p0`, `bias_estimation`, `dropout`, `outliers`, `imu_spikes`, `profile`, `meas_rate`. |
| `sweep_dropout_timeseries.csv` | Per dropout duration: RMS position error over runs and `3 sqrt(mean tr P_pos)` at each evaluation epoch. |
