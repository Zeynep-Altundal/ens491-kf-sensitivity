# Error peaks of the estimator sweep (Observed only)

Peak rule, attribution and recovery: `../README.md`. 10 seeds per cell. Estimators: `../../estimators/README.md`.

## 1. Peaks per scenario and estimator

Per cell: seeds with a peak / peaks per run (median) / largest peak: height [m], excess [m] (median, range over seeds with a peak) / delay from onset [s] (median, range) / recovery [s] (mean over seeds; "not rec. k/10": seeds not recovered; n/a: no end).

| Scenario | Estimator | Seeds with peak | Peaks/run | Height [m] | Excess [m] | Delay [s] | Recovery [s] |
|---|---|---|---|---|---|---|---|
| `accel_bias_step_small` | KF | 0/10 | 0 | - | - | - | 0.00 |
| `accel_bias_step_small` | EKF | 10/10 | 6 | 0.86 (0.76-1.09) | 0.74 (0.68-0.79) | 29.70 (28.59-31.70) | 14.49 |
| `accel_bias_step_small` | ESKF | 10/10 | 6 | 0.86 (0.76-1.09) | 0.74 (0.68-0.79) | 29.70 (28.59-31.70) | 14.49 |
| `accel_bias_step_large` | KF | 0/10 | 0 | - | - | - | 0.00 |
| `accel_bias_step_large` | EKF | 10/10 | 2 | 4.09 (3.93-4.28) | 3.94 (3.83-4.07) | 31.14 (28.80-31.50) | not rec. 10/10 |
| `accel_bias_step_large` | ESKF | 10/10 | 2 | 4.09 (3.93-4.28) | 3.94 (3.83-4.07) | 31.14 (28.80-31.50) | not rec. 10/10 |
| `gyro_bias_step_small` | KF | 0/10 | 0 | - | - | - | 0.00 |
| `gyro_bias_step_small` | EKF | 10/10 | 2 | 2.20 (2.09-2.39) | 2.10 (1.98-2.18) | 10.54 (9.80-38.20) | not rec. 10/10 |
| `gyro_bias_step_small` | ESKF | 10/10 | 2 | 2.20 (2.09-2.39) | 2.10 (1.98-2.18) | 10.54 (9.80-38.20) | not rec. 10/10 |
| `gyro_bias_step_large` | KF | 0/10 | 0 | - | - | - | 0.00 |
| `gyro_bias_step_large` | EKF | 10/10 | 1 | 10.94 (10.76-11.11) | 10.79 (10.52-10.92) | 10.09 (9.80-10.30) | not rec. 10/10 |
| `gyro_bias_step_large` | ESKF | 10/10 | 1 | 10.94 (10.76-11.10) | 10.79 (10.52-10.91) | 10.09 (9.80-10.30) | not rec. 10/10 |
| `accel_bias_ramp` | KF | 0/10 | 0 | - | - | - | n/a |
| `accel_bias_ramp` | EKF | 10/10 | 4 | 0.71 (0.57-0.92) | 0.59 (0.43-0.65) | 37.14 (35.40-37.70) | n/a |
| `accel_bias_ramp` | ESKF | 10/10 | 4 | 0.71 (0.57-0.92) | 0.59 (0.43-0.65) | 37.14 (35.40-37.70) | n/a |
| `imu_noise_x3` | KF | 0/10 | 0 | - | - | - | n/a |
| `imu_noise_x3` | EKF | 0/10 | 0 | - | - | - | n/a |
| `imu_noise_x3` | ESKF | 0/10 | 0 | - | - | - | n/a |
| `imu_noise_x10` | KF | 0/10 | 0 | - | - | - | n/a |
| `imu_noise_x10` | EKF | 10/10 | 8 | 0.68 (0.52-1.12) | 0.57 (0.38-0.91) | 44.80 (35.49-76.80) | n/a |
| `imu_noise_x10` | ESKF | 10/10 | 8 | 0.68 (0.52-1.12) | 0.57 (0.38-0.91) | 44.80 (35.49-76.80) | n/a |
| `pos_burst_short_calm` | KF | 10/10 | 1 | 6.29 (5.81-6.64) | 5.82 (4.98-6.09) | 0.50 (0.50-0.50) | 1.40 |
| `pos_burst_short_calm` | EKF | 10/10 | 2 | 2.83 (2.43-3.28) | 2.47 (2.08-2.76) | 0.89 (0.50-1.00) | 13.92 |
| `pos_burst_short_calm` | ESKF | 10/10 | 2 | 2.83 (2.43-3.28) | 2.47 (2.08-2.76) | 0.89 (0.50-1.00) | 13.94 |
| `pos_burst_long_calm` | KF | 10/10 | 1 | 5.75 (5.42-6.11) | 5.39 (5.09-5.55) | 1.89 (1.39-1.89) | 1.40 |
| `pos_burst_long_calm` | EKF | 10/10 | 2 | 3.02 (2.65-3.18) | 2.83 (2.27-2.91) | 1.89 (1.89-1.89) | 7.27 |
| `pos_burst_long_calm` | ESKF | 10/10 | 2 | 3.02 (2.65-3.18) | 2.83 (2.27-2.91) | 1.89 (1.89-1.89) | 7.27 |
| `pos_burst_short_aggressive` | KF | 10/10 | 1 | 5.22 (4.76-5.58) | 4.48 (4.20-4.68) | 0.40 (0.30-0.80) | 1.50 |
| `pos_burst_short_aggressive` | EKF | 10/10 | 1 | 0.85 (0.74-1.04) | 0.61 (0.52-0.81) | 0.40 (0.40-1.00) | 3.80 |
| `pos_burst_short_aggressive` | ESKF | 10/10 | 1 | 0.85 (0.74-1.04) | 0.61 (0.52-0.82) | 0.40 (0.40-1.00) | 3.80 |
| `pos_burst_long_aggressive` | KF | 10/10 | 3 | 6.52 (6.22-6.93) | 5.49 (4.77-6.52) | 4.20 (4.20-4.20) | 1.50 |
| `pos_burst_long_aggressive` | EKF | 10/10 | 1 | 2.15 (1.88-2.31) | 1.94 (1.49-2.17) | 3.50 (3.50-3.50) | 14.91 |
| `pos_burst_long_aggressive` | ESKF | 10/10 | 1 | 2.15 (1.88-2.31) | 1.94 (1.49-2.17) | 3.50 (3.50-3.50) | 15.59 |
| `pos_spike_single` | KF | 10/10 | 1 | 4.59 (4.00-5.11) | 4.26 (3.47-4.45) | 0.10 (0.10-0.10) | 1.60 |
| `pos_spike_single` | EKF | 10/10 | 1 | 0.81 (0.52-1.08) | 0.61 (0.37-0.76) | 0.10 (0.10-0.60) | 4.00 |
| `pos_spike_single` | ESKF | 10/10 | 1 | 0.81 (0.52-1.08) | 0.61 (0.37-0.76) | 0.10 (0.10-0.60) | 4.00 |
| `pos_spikes_repeated` | KF | 10/10 | 10 | 4.69 (4.49-5.01) | 4.38 (4.14-4.45) | 0.09 (0.09-0.10) | 8.20 |
| `pos_spikes_repeated` | EKF | 10/10 | 8 | 1.09 (0.95-1.32) | 0.84 (0.70-0.98) | 0.60 (0.09-4.10) | 24.99 |
| `pos_spikes_repeated` | ESKF | 10/10 | 8 | 1.09 (0.95-1.32) | 0.84 (0.70-0.98) | 0.60 (0.09-4.10) | 24.99 |
| `outage_5s_calm` | KF | 10/10 | 1 | 21.81 (17.77-27.29) | 21.45 (17.10-26.87) | 5.00 (5.00-5.00) | 0.91 |
| `outage_5s_calm` | EKF | 10/10 | 1 | 2.03 (1.06-3.72) | 1.84 (0.91-3.46) | 5.00 (4.89-5.00) | 4.06 |
| `outage_5s_calm` | ESKF | 10/10 | 1 | 2.03 (1.06-3.72) | 1.84 (0.91-3.46) | 5.00 (4.89-5.00) | 4.06 |
| `outage_20s_calm` | KF | 10/10 | 1 | 34.81 (24.99-50.54) | 34.64 (24.73-50.35) | 13.40 (11.90-19.80) | 0.93 |
| `outage_20s_calm` | EKF | 10/10 | 2 | 36.72 (21.59-58.71) | 36.47 (21.25-58.47) | 20.00 (20.00-20.00) | 9.80 |
| `outage_20s_calm` | ESKF | 10/10 | 2 | 36.72 (21.59-58.71) | 36.47 (21.25-58.47) | 20.00 (20.00-20.00) | 9.80 |
| `outage_40s_calm` | KF | 10/10 | 1 | 48.90 (24.99-68.37) | 48.66 (24.73-68.09) | 28.40 (11.90-39.90) | 1.01 |
| `outage_40s_calm` | EKF | 10/10 | 2 | 143.93 (82.74-231.43) | 143.63 (82.50-231.17) | 40.00 (40.00-40.00) | 10.23 |
| `outage_40s_calm` | ESKF | 10/10 | 2 | 143.94 (82.75-231.43) | 143.64 (82.51-231.17) | 40.00 (40.00-40.00) | 10.21 |
| `outage_5s_aggressive` | KF | 10/10 | 1 | 14.68 (9.28-16.82) | 14.17 (8.64-16.27) | 4.90 (4.80-4.90) | 1.02 |
| `outage_5s_aggressive` | EKF | 10/10 | 1 | 0.78 (0.43-1.12) | 0.58 (0.25-0.92) | 5.00 (4.50-5.40) | 1.63 |
| `outage_5s_aggressive` | ESKF | 9/10 | 1 | 0.79 (0.43-1.12) | 0.58 (0.26-0.93) | 5.00 (4.50-5.40) | 1.63 |
| `outage_20s_aggressive` | KF | 10/10 | 1 | 25.47 (12.47-36.17) | 25.08 (11.97-35.66) | 20.00 (19.40-20.00) | 0.93 |
| `outage_20s_aggressive` | EKF | 10/10 | 1 | 4.81 (1.68-14.71) | 4.67 (1.55-14.50) | 20.00 (19.80-20.00) | 5.13 |
| `outage_20s_aggressive` | ESKF | 10/10 | 1 | 4.82 (1.68-14.72) | 4.68 (1.55-14.51) | 20.00 (19.80-20.00) | 5.13 |
| `outage_40s_aggressive` | KF | 10/10 | 1 | 153.70 (123.06-190.71) | 153.17 (122.57-189.78) | 40.00 (40.00-40.00) | not rec. 4/10 |
| `outage_40s_aggressive` | EKF | 10/10 | 2 | 14.04 (6.54-54.22) | 13.85 (6.47-54.06) | 40.00 (40.00-40.00) | not rec. 8/10 |
| `outage_40s_aggressive` | ESKF | 10/10 | 2 | 14.01 (6.52-54.24) | 13.83 (6.45-54.08) | 40.00 (40.00-40.00) | not rec. 8/10 |
| `accel_bias_step_small_onset_hover` | KF | 0/10 | 0 | - | - | - | 0.00 |
| `accel_bias_step_small_onset_hover` | EKF | 10/10 | 4 | 0.84 (0.73-0.95) | 0.70 (0.65-0.77) | 13.90 (13.00-15.09) | 32.56 |
| `accel_bias_step_small_onset_hover` | ESKF | 10/10 | 4 | 0.84 (0.73-0.95) | 0.70 (0.65-0.77) | 13.90 (13.00-15.09) | 32.55 |
| `gyro_bias_step_small_onset_hover` | KF | 0/10 | 0 | - | - | - | 0.00 |
| `gyro_bias_step_small_onset_hover` | EKF | 10/10 | 10 | 2.61 (2.38-2.77) | 2.45 (2.24-2.51) | 30.14 (29.70-30.59) | not rec. 9/10 |
| `gyro_bias_step_small_onset_hover` | ESKF | 10/10 | 10 | 2.61 (2.38-2.77) | 2.45 (2.24-2.51) | 30.14 (29.70-30.59) | not rec. 9/10 |
| `imu_noise_burst_x10_onset_hover` | KF | 0/10 | 0 | - | - | - | 0.00 |
| `imu_noise_burst_x10_onset_hover` | EKF | 0/10 | 0 | - | - | - | 0.00 |
| `imu_noise_burst_x10_onset_hover` | ESKF | 0/10 | 0 | - | - | - | 0.00 |
| `accel_bias_step_small_onset_straight` | KF | 0/10 | 0 | - | - | - | 0.00 |
| `accel_bias_step_small_onset_straight` | EKF | 10/10 | 7 | 0.88 (0.70-0.99) | 0.73 (0.51-0.79) | 27.85 (27.49-28.30) | 53.41 |
| `accel_bias_step_small_onset_straight` | ESKF | 10/10 | 7 | 0.88 (0.70-0.99) | 0.73 (0.51-0.79) | 27.85 (27.49-28.30) | 53.39 |
| `gyro_bias_step_small_onset_straight` | KF | 0/10 | 0 | - | - | - | 0.00 |
| `gyro_bias_step_small_onset_straight` | EKF | 10/10 | 8 | 2.50 (2.19-2.69) | 2.38 (2.05-2.44) | 27.25 (26.49-27.30) | not rec. 7/10 |
| `gyro_bias_step_small_onset_straight` | ESKF | 10/10 | 8 | 2.50 (2.19-2.69) | 2.38 (2.05-2.44) | 27.25 (26.49-27.30) | not rec. 6/10 |
| `imu_noise_burst_x10_onset_straight` | KF | 0/10 | 0 | - | - | - | 0.00 |
| `imu_noise_burst_x10_onset_straight` | EKF | 0/10 | 0 | - | - | - | 0.00 |
| `imu_noise_burst_x10_onset_straight` | ESKF | 0/10 | 0 | - | - | - | 0.00 |
| `accel_bias_step_small_onset_figure_eight` | KF | 0/10 | 0 | - | - | - | 0.00 |
| `accel_bias_step_small_onset_figure_eight` | EKF | 10/10 | 4 | 0.83 (0.76-1.06) | 0.68 (0.58-0.74) | 21.10 (6.20-23.70) | 36.03 |
| `accel_bias_step_small_onset_figure_eight` | ESKF | 10/10 | 4 | 0.83 (0.76-1.06) | 0.68 (0.58-0.74) | 21.10 (6.20-23.70) | 36.67 |
| `gyro_bias_step_small_onset_figure_eight` | KF | 0/10 | 0 | - | - | - | 0.00 |
| `gyro_bias_step_small_onset_figure_eight` | EKF | 10/10 | 6 | 1.84 (1.71-2.07) | 1.77 (1.60-1.80) | 14.54 (13.59-14.99) | not rec. 10/10 |
| `gyro_bias_step_small_onset_figure_eight` | ESKF | 10/10 | 6 | 1.84 (1.70-2.07) | 1.77 (1.60-1.80) | 14.54 (13.59-14.99) | not rec. 10/10 |
| `imu_noise_burst_x10_onset_figure_eight` | KF | 0/10 | 0 | - | - | - | 0.00 |
| `imu_noise_burst_x10_onset_figure_eight` | EKF | 0/10 | 0 | - | - | - | 0.00 |
| `imu_noise_burst_x10_onset_figure_eight` | ESKF | 0/10 | 0 | - | - | - | 0.00 |
| `accel_bias_step_small_onset_aggressive` | KF | 0/10 | 0 | - | - | - | 0.00 |
| `accel_bias_step_small_onset_aggressive` | EKF | 10/10 | 2 | 0.99 (0.86-1.12) | 0.82 (0.72-0.86) | 17.20 (16.00-18.09) | 15.10 |
| `accel_bias_step_small_onset_aggressive` | ESKF | 10/10 | 2 | 0.99 (0.86-1.12) | 0.82 (0.72-0.86) | 17.20 (16.00-18.09) | 15.11 |
| `gyro_bias_step_small_onset_aggressive` | KF | 0/10 | 0 | - | - | - | 0.00 |
| `gyro_bias_step_small_onset_aggressive` | EKF | 10/10 | 2 | 2.28 (2.22-2.55) | 2.18 (1.91-2.27) | 11.25 (10.30-12.00) | not rec. 10/10 |
| `gyro_bias_step_small_onset_aggressive` | ESKF | 10/10 | 2 | 2.29 (2.23-2.56) | 2.18 (1.92-2.28) | 11.25 (10.30-11.80) | not rec. 10/10 |
| `imu_noise_burst_x10_onset_aggressive` | KF | 0/10 | 0 | - | - | - | 0.00 |
| `imu_noise_burst_x10_onset_aggressive` | EKF | 1/10 | 0 | 0.42 (0.42-0.42) | 0.31 (0.31-0.31) | 8.80 (8.80-8.80) | 2.12 |
| `imu_noise_burst_x10_onset_aggressive` | ESKF | 1/10 | 0 | 0.42 (0.42-0.42) | 0.31 (0.31-0.31) | 8.80 (8.80-8.80) | 2.12 |

Peaks starting before every onset (unattributed), all runs: 0.

## 2. Disagreement between estimators

Scenarios in which at least one estimator shows a peak (at least 5 of 10 seeds) and at least one other does not.

| Scenario | KF | EKF | ESKF | Peak in | No peak in |
|---|---|---|---|---|---|
| `accel_bias_step_small` | 0/10 | 10/10 | 10/10 | EKF, ESKF | KF |
| `accel_bias_step_large` | 0/10 | 10/10 | 10/10 | EKF, ESKF | KF |
| `gyro_bias_step_small` | 0/10 | 10/10 | 10/10 | EKF, ESKF | KF |
| `gyro_bias_step_large` | 0/10 | 10/10 | 10/10 | EKF, ESKF | KF |
| `accel_bias_ramp` | 0/10 | 10/10 | 10/10 | EKF, ESKF | KF |
| `imu_noise_x10` | 0/10 | 10/10 | 10/10 | EKF, ESKF | KF |
| `accel_bias_step_small_onset_hover` | 0/10 | 10/10 | 10/10 | EKF, ESKF | KF |
| `gyro_bias_step_small_onset_hover` | 0/10 | 10/10 | 10/10 | EKF, ESKF | KF |
| `accel_bias_step_small_onset_straight` | 0/10 | 10/10 | 10/10 | EKF, ESKF | KF |
| `gyro_bias_step_small_onset_straight` | 0/10 | 10/10 | 10/10 | EKF, ESKF | KF |
| `accel_bias_step_small_onset_figure_eight` | 0/10 | 10/10 | 10/10 | EKF, ESKF | KF |
| `gyro_bias_step_small_onset_figure_eight` | 0/10 | 10/10 | 10/10 | EKF, ESKF | KF |
| `accel_bias_step_small_onset_aggressive` | 0/10 | 10/10 | 10/10 | EKF, ESKF | KF |
| `gyro_bias_step_small_onset_aggressive` | 0/10 | 10/10 | 10/10 | EKF, ESKF | KF |

Scenarios without disagreement: 17 of 31.

## 3. Accelerometer bias step +0.2 m/s^2, 10 s, by onset phase

Per cell: seeds with a peak; largest excess `max D` after the onset [m], mean [95 % bootstrap interval]; delay of the largest peak [s], median; recovery [s].

| Onset phase (onset) | KF | EKF | ESKF |
|---|---|---|---|
| hover (5.5 s) | 0/10; 0.000 [0.000, 0.000]; -; 0.00 | 10/10; 0.711 [0.685, 0.737]; 13.90; 32.56 | 10/10; 0.711 [0.685, 0.737]; 13.90; 32.55 |
| straight (9 s) | 0/10; 0.000 [0.000, 0.000]; -; 0.00 | 10/10; 0.694 [0.640, 0.740]; 27.85; 53.41 | 10/10; 0.694 [0.640, 0.739]; 27.85; 53.39 |
| figure_eight (30 s) | 0/10; 0.000 [0.000, 0.000]; -; 0.00 | 10/10; 0.677 [0.649, 0.700]; 21.10; 36.03 | 10/10; 0.676 [0.649, 0.700]; 21.10; 36.67 |
| aggressive (52 s) | 0/10; 0.000 [0.000, 0.000]; -; 0.00 | 10/10; 0.812 [0.782, 0.837]; 17.20; 15.10 | 10/10; 0.812 [0.781, 0.837]; 17.20; 15.11 |

## 3. Gyro bias step +0.01 rad/s, 10 s, by onset phase

Per cell: seeds with a peak; largest excess `max D` after the onset [m], mean [95 % bootstrap interval]; delay of the largest peak [s], median; recovery [s].

| Onset phase (onset) | KF | EKF | ESKF |
|---|---|---|---|
| hover (5.5 s) | 0/10; 0.000 [0.000, 0.000]; -; 0.00 | 10/10; 2.415 [2.360, 2.461]; 30.14; not rec. 9/10 | 10/10; 2.414 [2.359, 2.460]; 30.14; not rec. 9/10 |
| straight (9 s) | 0/10; 0.000 [0.000, 0.000]; -; 0.00 | 10/10; 2.344 [2.272, 2.393]; 27.25; not rec. 7/10 | 10/10; 2.344 [2.272, 2.392]; 27.25; not rec. 6/10 |
| figure_eight (30 s) | 0/10; 0.000 [0.000, 0.000]; -; 0.00 | 10/10; 1.740 [1.700, 1.774]; 14.54; not rec. 10/10 | 10/10; 1.738 [1.699, 1.772]; 14.54; not rec. 10/10 |
| aggressive (52 s) | 0/10; 0.000 [0.000, 0.000]; -; 0.00 | 10/10; 2.165 [2.093, 2.227]; 11.25; not rec. 10/10 | 10/10; 2.172 [2.101, 2.235]; 11.25; not rec. 10/10 |

## 4. IMU noise burst x10, 2 s, by onset phase and estimator

Per cell: seeds with a peak; largest excess `max D` after the onset [m], mean [95 % bootstrap interval]; delay of the largest peak [s], median; recovery [s].

| Onset phase (onset) | KF | EKF | ESKF |
|---|---|---|---|
| hover (5.5 s) | 0/10; 0.000 [0.000, 0.000]; -; 0.00 | 0/10; 0.072 [0.058, 0.086]; -; 0.00 | 0/10; 0.072 [0.058, 0.086]; -; 0.00 |
| straight (9 s) | 0/10; 0.000 [0.000, 0.000]; -; 0.00 | 0/10; 0.114 [0.074, 0.156]; -; 0.00 | 0/10; 0.114 [0.073, 0.156]; -; 0.00 |
| figure_eight (30 s) | 0/10; 0.000 [0.000, 0.000]; -; 0.00 | 0/10; 0.092 [0.061, 0.129]; -; 0.00 | 0/10; 0.092 [0.061, 0.129]; -; 0.00 |
| aggressive (52 s) | 0/10; 0.000 [0.000, 0.000]; -; 0.00 | 1/10; 0.097 [0.055, 0.154]; 8.80; 2.12 | 1/10; 0.097 [0.055, 0.154]; 8.80; 2.12 |
