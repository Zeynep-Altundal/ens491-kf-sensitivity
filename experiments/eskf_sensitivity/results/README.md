# experiments/eskf_sensitivity/results

CSV output of `run.py`.

| File | Content |
|------|---------|
| `<profile>_run<i>_history.csv` | One row per measurement epoch: true position, measurement, estimate, error state and filter standard deviations (15 each), true and estimated biases, NIS, position NEES, availability and acceptance flags. |
| `<profile>_run<i>_summary.csv` | Position, velocity and attitude RMSE (3D), mean position NEES, mean NIS, number of updates, maximum position and attitude error. |
