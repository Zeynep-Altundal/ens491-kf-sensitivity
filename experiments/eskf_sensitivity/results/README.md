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
| `misalignment_summary.csv` | Per misalignment level: RMSE, NIS, full and sub-block NEES with bootstrap intervals, chi-square intervals and fraction of epochs inside. |
| `misalignment_paired.csv` | Paired differences to 0 degrees (position and attitude RMSE, NIS, full NEES). |
| `misalignment_bias.csv` | Final estimated and true biases per axis, mean bias error and its paired difference to 0 degrees. |
| `misalignment_time.csv` | Run-averaged full and attitude NEES and RMS attitude error per epoch and level. |
| `gnss_summary.csv` | Per GNSS case: final and maximum position error, full NEES inside and outside the outage. |
| `gnss_drift.csv` | RMS position error and 3-sigma at elapsed outage times. |
| `gnss_time.csv` | RMS position error and 3-sigma per epoch and case. |
| `clarify_turning_window.csv` | Turning consistency runs, t = 5..25 s: run-averaged full, attitude and gyro-bias NEES, RMS attitude and gyro-bias errors and mean filter standard deviations. |
| `clarify_F_step.csv` | Relative deviation of the attitude columns of F from finite differences for several step sizes. |
| `clarify_F_magnitude.csv` | Linearisation error of the attitude columns of F for finite perturbation magnitudes. |
| `attitude_reference_summary.csv` | Attitude NEES (mean with interval, chi-square interval, fraction of epochs inside) and RMS attitude error at 0 and 5 degrees misalignment, against the true body attitude and against the true sensor attitude; for the body reference the largest difference to `misalignment_time.csv`. |
| `attitude_reference_time.csv` | Run-averaged attitude NEES and RMS attitude error over time for both references. |
| `scale_factor_summary.csv` | Gyroscope scale-factor study per profile, scenario (S0, S1) and level: position, velocity, attitude RMSE, NIS, full and sub-block NEES with intervals, chi-square intervals and fraction of epochs inside. |
| `scale_factor_paired.csv` | Paired differences to `s = 0` (position, velocity, attitude RMSE, NIS, full NEES). |
| `scale_factor_bias.csv` | Gyro-bias estimate, truth and error per axis at t = 60 s (and t = 20 s for S1), the rate-equivalent `s * mean(w)` over the preceding 10 s, the paired change of the bias error relative to `s = 0` and its ratio to the rate-equivalent. |
| `scale_factor_drift.csv` | S1: RMS position error and `3 sqrt(mean tr P_pos)` at elapsed outage times. |
