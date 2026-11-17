# experiments/filter_family/results

CSV output of `run.py` (range-bearing benchmark, N = 100 runs, 60 steps, seed 0, true Q and R).

| File | Content |
|------|---------|
| `nominal_comparison.csv` | Per filter: position RMSE, mean NEES (4 states), mean NIS (2 measurements), runtime per step in microseconds, runs with final position error above 10 m, mean IEKF iterations. |
| `nominal_error_over_time.csv` | Position RMSE over the runs at each step, per filter. |
| `nominal_example_run.csv` | Truth, measurements (converted to x, y) and all estimates of the example run (first run whose true bearing crosses +pi / -pi). |
