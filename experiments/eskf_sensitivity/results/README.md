# experiments/eskf_sensitivity/results

CSV output of `run.py` and `sweep.py` (protocol in `../README.md`).

| File | Content |
|------|---------|
| `<profile>_run<i>_history.csv` | One row per measurement epoch: true position, measurement, estimate, error state and filter standard deviations (15 each), true and estimated biases, NIS, position NEES, availability and acceptance flags. |
| `<profile>_run<i>_summary.csv` | Position, velocity and attitude RMSE (3D), mean position NEES, mean NIS, number of updates, maximum position and attitude error. |
| `sweep_<name>.csv` | One row per grid value and variant: number of successful and failed runs, position, velocity and attitude RMSE, mean position NEES, mean NIS (accepted updates), gate rejection rate, RMS final gyro and accelerometer bias error, and for `dropout` the mean and maximum over runs of the maximum position error inside the window. Sweeps: `q_imu`, `q_bias`, `r`, `p0`, `bias_estimation`, `dropout`, `outliers`, `imu_spikes`, `profile`, `meas_rate`. |
| `sweep_dropout_timeseries.csv` | Per dropout duration: RMS position error over runs and `3 sqrt(mean tr P_pos)` at each evaluation epoch. |
| `closure_summary.csv` | Nominal setting and first / last grid point of every sweep: RMSE, NEES, NIS and final bias errors with 95 % bootstrap intervals. |
| `closure_paired.csv` | Paired differences of position RMSE and mean position NEES to the nominal setting. |
| `closure_seeds.csv` | Nominal metrics for five master seeds. |
| `consistency_summary.csv` | Per profile and reset on / off: mean full-state and sub-block NEES with intervals, 95 % per-epoch bounds and the fractions of epochs inside, above and below. |
| `consistency_reset_paired.csv` | Paired difference of full-state NEES with and without the covariance reset. |
| `consistency_time_<profile>.csv` | Run-averaged full-state and sub-block NEES per evaluation epoch (reset on). |
