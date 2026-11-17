# experiments/linear_kf/results

CSV output of the experiment scripts.

| File | Written by | Content |
|------|------------|---------|
| `sweep_q.csv`, `sweep_r.csv`, `sweep_p0.csv` | `sweep.py` | One-at-a-time sweeps. Columns: `s, pos_rmse, vel_rmse, mean_nees, mean_nis`. |
| `open_point.csv` | `ratio_study.py` | Metrics of selected scale configurations for the `T200_full` and `T2000_steady` windows, and the gain convergence step (1 % threshold). |
| `grid_T200_full.csv`, `grid_T2000_steady.csv` | `ratio_study.py` | 13 x 13 grid over Q and R scale factors (P0 at truth). |
| `grid_diagonals_T200_full.csv`, `grid_diagonals_T2000_steady.csv` | `ratio_study.py` | Minimum and maximum of each metric along lines of constant `log10(Q scale / R scale)`. |

Window `T200_full`: `T = 200`, all steps. Window `T2000_steady`: `T = 2000`, steps `k >= 1000`.
| `closure_summary.csv` | `closure.py` | Position and velocity RMSE, mean NEES and NIS with 95 % bootstrap intervals for the nominal setting and the sweep endpoints. |
| `closure_paired.csv` | `closure.py` | Paired differences of position RMSE with intervals. |
| `closure_seeds.csv` | `closure.py` | Nominal metrics for five master seeds. |
| `closure_nominal_time.csv` | `closure.py` | Run-averaged NEES and NIS per step with the 95 % chi-square bounds. |
