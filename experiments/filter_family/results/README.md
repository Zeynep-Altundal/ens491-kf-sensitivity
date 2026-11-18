# experiments/filter_family/results

CSV output of `run.py` (range-bearing benchmark, N = 100 runs, 60 steps, seed 0, true Q and R) and of `sweep.py` (protocol in `../README.md`).

| File | Content |
|------|---------|
| `nominal_comparison.csv` | Per filter: position RMSE, mean NEES (4 states), mean NIS (2 measurements), runtime per step in microseconds, runs with final position error above 10 m, mean IEKF iterations. |
| `nominal_error_over_time.csv` | Position RMSE over the runs at each step, per filter. |
| `nominal_example_run.csv` | Truth, measurements (converted to x, y) and all estimates of the example run (first run whose true bearing crosses +pi / -pi). |
| `sweep_<name>.csv` | Sensitivity sweeps (`mistune_q`, `mistune_r`, `mistune_p0`, `bearing_noise`, `initial_uncertainty`, `closest_approach`): one row per grid value and filter with position RMSE, mean NEES, mean NIS, divergence rate, mean IEKF iterations and the median realised minimum range. RMSE, NEES, NIS and iterations are over non-diverged runs. |
| `closure_summary.csv` | Per setting and filter: divergence count, position RMSE, mean NEES and mean NIS with 95 % bootstrap intervals. |
| `closure_paired.csv` | Paired differences between every pair of filters (position RMSE, mean NEES) with intervals. |
| `closure_seeds.csv` | Nominal metrics per filter for five master seeds. |
| `adaptive_r_summary.csv` | Adaptive-R diagnosis: per configuration (`linear_T400`, `rb_T400`, `rb_T60`), segment (`all`, `steady`) and variant (`off`, `W5` ... `W200`): divergence count, position RMSE, mean NEES, mean NIS with 95 % intervals and the expected NEES and NIS. |
| `adaptive_r_paired.csv` | Paired differences adaptation on minus off per window, metric and segment. |
| `adaptive_r_window_pairs.csv` | Paired differences of mean NEES between successive windows (larger minus smaller W); `primary` marks the segment used to answer the window question. |
| `adaptive_r_bias.csv` | Steady-state bias of the estimated R (`R_ii / R_true_ii - 1`, and the off-diagonal correlation for range-bearing), closed loop and open loop, with intervals. |
| `adaptive_r_psd.csv` | Number of updates using an estimate, negative-eigenvalue counts of the implemented (post-fit) estimate and of the innovation-form diagnostic. |
| `adaptive_r_time.csv` | Per step: mean, standard deviation, 5 % and 95 % quantiles over runs of `R_hat_ii / R_true_ii`. |
| `tv_r_summary.csv` | Time-varying R benchmark: per model, pattern and filter (`fixed_nominal`, `fixed_average_hindsight`, `true_covariance`, `adaptive_W<W>`, `adaptive_selected` with its `selected_window`): divergence count, position RMSE, mean NEES, mean NIS, innovation NLL, R tracking error (absolute and signed log-ratio) with intervals. |
| `tv_r_paired.csv` | Paired differences filter minus true-covariance filter (position RMSE, NEES, NIS, NLL) with intervals. |
| `tv_r_time.csv` | Per step: position RMSE over runs, mean and run-0 filter R (first diagonal element, relative to nominal) and the true scale factor. |
| `tv_r_window_selection.csv` | Validation runs (seed 1000, burst seed 1001): mean innovation NLL per window with intervals; `selected` marks the window reported as `adaptive_selected`. |
| `tv_r_identifiability.csv` | Per model, pattern (including the constant-R control) and window: variance share of the `H P+ H^T` term in the estimated R, correlations of `log R_hat` with `log R_true` and with `log H P+ H^T`, mean `H P+ H^T / R_hat`, with intervals, and the verdict of amendment A5. |
| `lorenz_summary.csv` | Lorenz-attractor benchmark: per setting (`full`, `mismatch_J2`), `1/r^2` and filter: failed runs, MSE in dB with 95 % interval and the paper-style spread, RMSE, mean NEES, mean NIS with intervals. |
| `lorenz_paper_values.csv` | EKF and UKF MSE [dB] transcribed from Tables V and VII of arXiv:2107.10043v3, with table, setting and spread. |
| `lorenz_comparison.csv` | This study's MSE [dB] next to the transcribed paper values, difference and whether the paper value lies inside this study's interval. |
| `lorenz_knet_baseline.csv` | Exploratory: reference-code results (EKF, observation noise floor; KalmanNet absent because training failed) and this repository's EKF, EKF-F, IEKF, UKF on the exported reference test set and on an independent realisation, MSE in dB with interval and spread, NEES, NIS. |
| `lorenz_knet_training_curve.csv` | Exploratory: training and validation MSE [dB] per logged step of the KalmanNet reference run (parsed from its log). |
