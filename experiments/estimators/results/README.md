# experiments/estimators/results

Outputs of `../sanity.py`, `../sweep.py` and `../report.py` (protocol and file formats in
`../README.md`).

| File | Content | Version-controlled |
|---|---|---|
| `sanity.json` | sanity checks 1-3 on the nominal recording, 10 seeds | yes |
| `run_metrics.csv` | one row per run: MSE per phase and overall (position, velocity, attitude), peak error and time, mean NEES and NIS, runtime | yes |
| `recovery.csv` | one row per run and corruption event: end time and recovery time (`inf`: not recovered, `nan`: no end) | yes |
| `summary_rmse.csv`, `summary_peak.csv`, `summary_recovery.csv` | per estimator and scenario: mean over seeds with 95 % bootstrap interval | yes |
| `tables.md` | the summary tables in Markdown (Observed only) | yes |
| `sweep_runtime.txt` | number of runs, wall time, commit | yes |
| `runs/<scenario>/<estimator>_seed<k>.npz` | one file per run (estimate, covariance diagonal, errors, NEES, NIS; float32) | no (1.5 GB) |
