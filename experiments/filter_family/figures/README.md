# experiments/filter_family/figures

PNG figures written by `plots.py`:

| File | Content |
|------|---------|
| `nominal_example_trajectory.png` | Example run: truth, measurements, all estimates, sensor, and the point where the true bearing wraps. |
| `nominal_error_over_time.png` | Position RMSE over the Monte Carlo runs versus time for each filter. |
| `sweep_<name>.png` | Position RMSE and mean NEES (expected value 4 as reference) versus the swept parameter, one line per filter; a divergence-rate panel is added when any filter diverges in that sweep. |
