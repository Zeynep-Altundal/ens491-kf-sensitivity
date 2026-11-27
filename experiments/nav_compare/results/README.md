# Results (week 3, 27 Nov)

| Folder | Made with | Content |
|---|---|---|
| `compare_events/` | `scripts/compare_filters_events --seeds 5` (IMU grade `euroc`) | one row per platform, disturbance set, seed and filter; medians per set; heat map |
| `peaks_events/` | `peaks nav` on the seed-0 logs of the run above | peaks, peaks per set and condition, condition table |
| `drone_sim/` | `scripts/run_drone_sim --seeds 3` (needs `simulation/rviz_drone`) and `peaks curves` | quadrotor flight phases, outages, peaks |
| `wheel_speed_check/` | the script in the folder (10 seeds) | numbers of `docs/synthetic_data_design.md`, section 10 |

The run logs themselves are not committed (size); rerun the commands to get them.
