# Results (week 3, 27 Nov)

| Folder | Made with | Content |
|---|---|---|
| `compare_events/` | `scripts/compare_filters_events --seeds 5` (IMU grade `euroc`) | one row per platform, disturbance set, seed and filter; medians per set; heat map |
| `peaks_events/` | `peaks nav` on the seed-0 logs of the run above | peaks, peaks per set and condition, condition table |
| `drone_sim/` | `scripts/run_drone_sim --seeds 3` (needs `simulation/rviz_drone`) and `peaks curves` | quadrotor flight phases, outages, peaks |
| `wheel_speed_check/` | the script in the folder (10 seeds) | numbers of `docs/synthetic_data_design.md`, section 10 |

The run logs themselves are not committed (size); rerun the commands to get them.
| `euroc/` | `scripts/run_euroc ~/data/euroc/V1_01_easy` | EuRoC V1_01_easy, fixes simulated from the ground truth (10 Hz, 0.1 m), full and 10 s outage |
| `knet_peaks/` | `peaks knet` on the KalmanNet benchmark logs (export_logs.py, KalmanNet seed 0) | peaks per data set and filter |
| `euroc/attitude_diagnostic.txt` | `scripts/diag_euroc_attitude ~/data/euroc/V1_01_easy` | gyro vs ground-truth attitude: 1.7 deg after 5 s, no time offset; the ~5 deg attitude error comes from the data |
| `init_attitude/` | `scripts/compare_init_attitude --seeds 5` (grade `euroc`) | filters started with a 10, 30, 60 deg attitude error |
| `wheel_speed_check/yaw_test.*` | the script in the folder | ESKF vs InEKF with and without wheel speed, 10 seeds |
