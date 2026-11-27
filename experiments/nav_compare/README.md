# experiments/nav_compare

Seven filters behind one interface on IMU data (drone and ground vehicle), with time-windowed
disturbances, vehicle aiding signals and a peak analysis. Moved here from the earlier `ens491-nkf`
repository (week 3); interface and conventions: `docs/common_interface.md`, notation:
`docs/notation.md`, data design and parameter sources: `docs/synthetic_data_design.md`.

| Path | Content |
|---|---|
| `data/scenarios.py` | UGV (car) and UAV (quadrotor-like) scenario generator: truth first, then IMU errors (noise, bias, bias random walk, scale factor, misalignment, ...) and GPS-like position fixes |
| `data/disturbances.py` | events in a time window (`bias_step`, `noise_burst`, `spike`, `outage`) and the standard event sets; every event is logged |
| `data/odometry.py`, `data/magnetometer.py` | wheel speed and steering angle (UGV), compass heading |
| `data/recording.py` | one flight as one `.npz` (common recording format) |
| `compare/filters.py` | KF-CV, KF-IMU, EKF, ESKF, IESKF, UKF, InEKF; position, body-velocity (NHC / wheel speed), yaw-rate, heading, ZUPT and ZARU updates |
| `compare/runner.py` | `run_filter(rec, name, ...)` writes the run log; `errors`, `summarize` |
| `scripts/` | experiments (disturbance sets, aiding during GNSS outage, initial attitude error) and their analysis |
| `tests/` | unit tests |

Run from the repository root:

```bash
python -m pytest -q experiments/nav_compare
python -m experiments.nav_compare.scripts.compare_filters_events --seeds 5 --logs --out results/compare_events
python -m experiments.nav_compare.scripts.analyze_compare_events results/compare_events
python -m experiments.nav_compare.scripts.compare_aiding_outage --seeds 5 --out results/aiding_outage
python -m experiments.nav_compare.scripts.compare_init_attitude --seeds 5 --out results/init_attitude
```

Results of week 3 (5 seeds, medians) are in `results/` of this folder.
