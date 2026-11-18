# analysis/dataset_char

Characterisation of the real IMU data of one KITTI raw sequence and one EuRoC sequence from a
filtering point of view (timing, signal content, statistics, noise, outliers, motion dynamics,
ground truth), and comparison with the synthetic generator in `synthetic/`. Script:
`characterise.py`; outputs in `results/` and `figures/`.

The raw data are read from their local locations and are never copied into the repository or
modified:

| Dataset | Location (read-only) |
|---------|----------------------|
| KITTI raw | `~/data/kitti_raw/2011_09_30` |
| EuRoC MAV | `~/data/euroc/V1_01_easy` |

Every property reported is measured from these files or quoted from the documentation files
shipped with them (file named). Anything that cannot be confirmed from them is marked **OPEN**.

## Protocol

This protocol, including all thresholds, was written before the analysis was run and is not
changed after seeing results.

### 0. Inventory

List the sequences and files present under each location (no other location is searched).

### 1. Timing

* Measured IMU rate `1 / median(dt)`; nominal rate where a shipped file states it.
* Timestamp jitter: histogram of `dt - median(dt)`; standard deviation, minimum and maximum of
  `dt`; largest gap.
* Dropped samples: intervals with `dt > 1.5 median(dt)`, with the estimated number of missing
  samples `round(dt / median(dt)) - 1`. Duplicated or non-monotonic samples: `dt <= 0`; exact
  duplicate data rows.
* Relation of the IMU and ground-truth clocks: whether both use the same time base; the offset of
  each ground-truth timestamp to the nearest IMU timestamp (fraction of exact coincidences); start
  and end of each stream.

### 2. What each signal is

From the shipped files only: whether the inertial data are raw sensor outputs or outputs of an
integrated navigation unit, and from what the ground truth is derived. Measured indicators reported
next to the statement: value quantisation (smallest non-zero difference between distinct values per
axis), and, where an attitude is given in the same records, the agreement of the angular rate with
the finite-difference attitude rate.

### 3. Signal statistics (per axis, gyroscope and accelerometer)

Over the whole sequence and over the longest static segment (Section 4): mean, standard deviation,
minimum, maximum, excess kurtosis, histogram. Gravity check: mean and standard deviation of the
specific-force norm `|f|` over the static segment (if any) and the median over the whole sequence,
compared with 9.81 m/s^2 (the generator's value; the local gravity is not given in the files).

### 4. Noise

* **Static segment detection**: for each sample, a centred 1 s window; the sample is static if the
  standard deviation of `|w|` in the window is below **0.01 rad/s** and that of `|f|` below
  **0.05 m/s^2** and, where the ground truth gives a velocity, the ground-truth speed is below
  **0.05 m/s** at that sample. The longest contiguous run of static samples is the static segment.
* **Allan deviation** (overlapping, per axis) on the static segment if it lasts at least **10 s**;
  averaging times from one sample to `T / 9` (T the segment length), logarithmically spaced.
* **White-noise density** N: the median of `sigma(tau) sqrt(tau)` over `tau` from 5 samples to
  `min(1 s, T / 9)`, reported only if the fitted log-log slope over that range lies in
  `[-0.6, -0.4]`; otherwise "not identifiable".
* **Bias instability** B = `min sigma(tau) / 0.664`, reported only if the minimum lies strictly
  inside the valid range (the curve rises again at larger `tau`); otherwise "not identifiable".
* **Bias random walk**: identifiable only if a region of slope `+1/2` is present within the valid
  range; otherwise "not identifiable".
* **Motion-independent upper bound** (whole sequence, always computed): `sigma_d = std(diff(x)) /
  sqrt(2)` per axis and the density bound `sigma_d sqrt(dt)`. Valid as an upper bound on the white
  noise if the motion changes little between samples; it also contains vibration and motion.

### 5. Outliers and spikes

Per axis, residual `r = x - median_11(x)` (centred running median over 11 samples), robust scale
`s = 1.4826 MAD(r)` over the sequence; a sample is a spike if `|r| > 6 s`. Reported: count,
fraction, largest `|r|` in units of the signal and of `s`.

### 6. Motion dynamics

Speed from the ground-truth velocity; angular-rate norm `|w|` from the gyroscope; kinematic
acceleration norm from the central finite difference of the ground-truth velocity; ranges (minimum,
median, 95 % quantile, maximum); fraction of time turning, defined as `|w| > 0.1 rad/s`. The same
quantities for the synthetic profiles `straight`, `turning`, `aggressive` (60 s at 200 Hz, from
`synthetic/trajectories.py`, true body rate, velocity and acceleration).

### 7. Ground truth

Rate (median dt), coverage relative to the IMU stream, gaps (`dt > 1.5 median(dt)`), smoothness:
RMS of the third finite difference of position divided by `dt^3` (jerk), and RMS difference between
the finite-difference velocity of the position and the velocity given in the ground truth (where
both exist). For KITTI the position is converted from latitude, longitude and altitude to a local
east-north-up frame at the first sample with a spherical Earth of radius 6 378 137 m (an
approximation stated as such). Also the status fields recorded with the ground truth, where present.

### Outputs

`results/summary.csv` (one row per dataset, quantity and axis), `results/allan_<dataset>.csv`,
`results/synthetic_profiles.csv`; figures `figures/<dataset>_timing.png`,
`figures/<dataset>_histograms.png`, `figures/<dataset>_allan.png`, `figures/<dataset>_motion.png`,
`figures/motion_comparison.png`.

## Results

All values are from `results/summary.csv` (run of `characterise.py`); the protocol above was not
changed after the run. Additional descriptive quantity, not part of the protocol: the smallest
1 s window standard deviation of `|w|` and `|f|`, reported to show why no static segment exists.

### Inventory

* **KITTI** (`~/data/kitti_raw/2011_09_30`): one sequence, `2011_09_30_drive_0016_extract`, with
  the `oxts` folder only (2967 data files, `timestamps.txt` with 2967 lines, `dataformat.txt`), and
  `calib_cam_to_cam.txt`, `calib_imu_to_velo.txt`, `calib_velo_to_cam.txt`. No camera or lidar
  data are present. `dataformat.txt` labels the third acceleration field `ay` although its
  description reads "acceleration in z"; the description of `pos_accuracy` reads "velocity
  accuracy (north/east in m)". The fields are used by position (field 14 is the z acceleration).
* **EuRoC** (`~/data/euroc/V1_01_easy/mav0`): `body.yaml`, `imu0/` (data, sensor.yaml),
  `vicon0/` (data, sensor.yaml), `state_groundtruth_estimate0/` (data, sensor.yaml). No camera
  folders are present.

### Per-dataset summary

| Quantity | KITTI drive 0016 | EuRoC V1_01_easy |
|----------|------------------|------------------|
| Duration, samples | 29.66 s, 2967 | 145.60 s, 29 120 |
| Nominal IMU rate | OPEN (not stated in the shipped files) | 200 Hz (`imu0/sensor.yaml`) |
| Measured rate (1 / median dt) | 100.007 Hz | 200.003 Hz |
| dt std, min, max | 81 us, 9.658 ms, 10.342 ms | 0.11 us, 4.99994 ms, 5.00019 ms |
| Gaps > 1.5 median dt, missing, duplicates, non-monotonic | 0, 0, 0, 0 | 0, 0, 0, 0 |
| IMU and ground-truth clock | same OXTS records | same time base; 77.0 % of ground-truth stamps equal an IMU stamp, all within 256 ns; ground truth starts 1.04 s after and ends 1.00 s before the IMU; Vicon starts 1.56 s before the IMU, not aligned to IMU samples |
| Value quantisation | none visible (2967 distinct values per axis, steps of 1e-10 to 1e-7) | gyro 6.98e-4 rad/s, accel 8.17e-3 m/s^2 |
| Gyro excess kurtosis x, y, z | 3.19, 1.85, 0.39 | 0.39, 1.88, 0.89 |
| Accel excess kurtosis x, y, z | -0.78, 2.03, 2.01 | 0.62, 0.82, 0.25 |
| `|f|` median (whole sequence; no static segment) | 9.818 m/s^2 | 9.808 m/s^2 |
| Static fraction, longest static segment | 0, none | 0, none |
| Smallest 1 s std of `|w|`, `|f|` | 2.6e-3 rad/s, 0.071 m/s^2 | 5.0e-3 rad/s, 0.108 m/s^2 |
| Allan deviation | not computed (no static segment of 10 s) | not computed (no static segment of 10 s) |
| White-noise density, bias instability, bias random walk | not identifiable | not identifiable |
| Gyro density upper bound from differences x, y, z [rad/s/sqrt(Hz)] | 1.6e-4, 2.6e-4, 5.1e-5 | 1.1e-3, 3.9e-3, 2.0e-3 |
| Accel density upper bound x, y, z [m/s^2/sqrt(Hz)] | 3.3e-3, 5.5e-3, 5.8e-3 | 0.099, 0.040, 0.079 |
| Spikes (6 robust scales from an 11-sample median), gyro x, y, z | 135, 30, 5 | 48, 2, 4 |
| Spikes, accel x, y, z | 28, 23, 37 | 7, 8, 4 |
| Speed min / median / max | 12.4 / 13.7 / 15.7 m/s | 0.0003 / 0.40 / 1.06 m/s |
| `|w|` median / 95 % / max | 0.015 / 0.043 / 0.090 rad/s | 0.23 / 0.64 / 0.94 rad/s (raw gyro, including bias) |
| `|dv/dt|` median / 95 % / max | 0.30 / 0.60 / 1.32 m/s^2 | 0.64 / 1.39 / 7.20 m/s^2 |
| Fraction of time with `|w| > 0.1 rad/s` | 0 | 0.84 |
| Ground truth | OXTS navigation solution in the same records, 100 Hz, full coverage, no gaps; `navstat` 4 in all records; `posmode` 4 (400), 5 (2417), 6 (150); `pos_accuracy` 0.10 to 0.15 | batch estimate, 200 Hz, 98.6 % of the IMU interval, no gaps; Vicon 100.04 Hz, 32 intervals > 1.5 median dt, largest 18.4 ms |
| Ground-truth smoothness: RMS jerk (third difference) | 5665 m/s^3 | estimate 645 m/s^3; Vicon 2887 m/s^3 |
| RMS(finite-difference velocity - given velocity) | 0.179 m/s | 0.0047 m/s |

Synthetic profiles (60 s, 200 Hz, `results/synthetic_profiles.csv`):

| Profile | Speed min / median / max [m/s] | `|w|` median / max [rad/s] | `|a|` median / max [m/s^2] | Fraction turning |
|---------|------------------|------------------|-----------------|------------------|
| straight | 0 / 2.0 / 4.0 | 0 / 0 | 0.36 / 0.50 | 0 |
| turning | 2.77 / 3.92 / 5.92 | 0.27 / 0.66 | 1.40 / 1.86 | 0.82 |
| aggressive | 1.30 / 3.47 / 4.93 | 2.61 / 4.10 | 1.46 / 1.95 | 1.00 |

### What each signal is

* **KITTI**: the inertial fields (`ax..az`, `wx..wz`) are fields of the records of an "oxts-unit"
  (`dataformat.txt`) that also contain latitude, longitude, altitude, roll, pitch, yaw, velocities,
  navigation status, number of satellites and GPS position, velocity and orientation modes. The
  inertial data are therefore delivered by an integrated GPS/inertial navigation unit. Whether the
  acceleration and rate fields are raw sensor outputs or bias-compensated, filtered outputs of that
  unit is not stated in the shipped files: **OPEN**. Measured indicators: no quantisation is visible,
  and `wz` agrees with the finite-difference yaw rate with correlation 0.886 (RMS difference
  3.1e-3 rad/s for an RMS `wz` of 6.1e-3 rad/s); neither indicator settles the question. The ground
  truth is the navigation solution of the same unit (no independent reference is shipped). The
  model of the unit is not named in the shipped files: **OPEN**.
* **EuRoC**: `imu0/sensor.yaml` describes the IMU as "VI-Sensor IMU (ADIS16448)", i.e. a sensor
  output, not a navigation solution; the measured quantisation steps are consistent with digital
  sensor outputs. Whether factory calibration or filtering inside the sensor is applied is not
  stated: **OPEN**. The ground truth (`state_groundtruth_estimate0/sensor.yaml`) is "the nonlinear
  least-squares batch solution over the Vicon pose and IMU measurements including time offset
  estimation"; it therefore depends on the same IMU data and contains estimated gyro biases (mean
  -2.2e-3, 2.08e-2, 7.63e-2 rad/s) and accelerometer biases (mean -0.013, 0.550, 0.068 m/s^2), each
  varying by less than 2e-4 rad/s and 5e-3 m/s^2 over the sequence. `vicon0` is the Vicon pose
  measurement (`vicon0/sensor.yaml`).

## Implications for the classical ESKF

### KITTI drive 0016

* **Observed in the data**: IMU at about 100 Hz with small jitter (81 us) and no gaps; specific
  force including gravity (`|f|` median 9.818 m/s^2); constant-speed driving (12.4 to 15.7 m/s)
  with small angular rates (`|w|` at most 0.090 rad/s, never above the turning threshold);
  heavy-tailed rate and acceleration residuals (spikes under the stated criterion on every axis,
  largest 28.7 robust scales on gyro x); navigation solution at 100 Hz with reported position
  accuracy 0.10 to 0.15 and changing GPS position modes.
* **Not present in the data (would have to be injected synthetically)**: static periods (needed for
  bias and noise identification); turns or aggressive rotation; measurement dropouts (the
  navigation solution is continuous); an independent ground truth.
* **Could not be determined**: white-noise densities, bias instability and bias random walk (no
  static segment); whether the inertial fields are raw or processed; nominal IMU rate; local
  gravity; the cause of the 0.18 m/s RMS mismatch between the finite-difference position and the
  reported velocity (the spherical-Earth conversion is an approximation).

### EuRoC V1_01_easy

* **Observed in the data**: IMU at 200 Hz with sub-microsecond jitter and no gaps; quantised
  outputs (6.98e-4 rad/s, 8.17e-3 m/s^2); large high-frequency content in the accelerometer
  (difference-based bound up to 0.099 m/s^2/sqrt(Hz)) and gyroscope; near-zero speed at the start
  and end while `|w|` stays near 0.08 rad/s (consistent with the gyro bias in the ground-truth file);
  rotation above 0.1 rad/s for 84 % of the time (raw gyro); ground truth on the IMU time base at
  200 Hz, Vicon at 100 Hz with 32 small gaps.
* **Not present in the data (would have to be injected synthetically)**: static periods with low
  vibration (no 1 s window with `|f|` std below 0.05 m/s^2); IMU sample dropouts or timestamp
  errors; long position outages (Vicon gaps are at most 18.4 ms).
* **Could not be determined**: white-noise densities, bias instability and bias random walk from
  this sequence (no static segment); whether the vibration is caused by running rotors (not stated
  in the shipped files); internal processing of the sensor.

### Generator parameters compared with the measured values (generator unchanged)

| Parameter | Generator (`ScenarioConfig`) | KITTI (measured) | EuRoC (measured) |
|-----------|------------------------------|------------------|------------------|
| IMU rate | 200 Hz | 100.007 Hz (0.50x) | 200.003 Hz (1.00x) |
| Position measurement rate | 10 Hz | navigation solution 100 Hz (10x) | estimate 200 Hz (20x), Vicon 100 Hz (10x) |
| Gyro noise density | 2.0e-4 rad/s/sqrt(Hz) | not identifiable; difference bound 5.1e-5 to 2.6e-4 (0.25x to 1.3x) | not identifiable; difference bound 1.1e-3 to 3.9e-3 (5.7x to 19.5x) |
| Accel noise density | 2.0e-3 m/s^2/sqrt(Hz) | not identifiable; bound 3.3e-3 to 5.8e-3 (1.6x to 2.9x) | not identifiable; bound 0.040 to 0.099 (20x to 49x) |
| Gyro bias random walk | 2.0e-5 rad/s^2/sqrt(Hz) | not identifiable | not identifiable |
| Accel bias random walk | 3.0e-3 m/s^3/sqrt(Hz) | not identifiable | not identifiable |
| Turn-on gyro bias (std used in runs) | 2.0e-3 rad/s | not determined | ground-truth estimate up to 7.63e-2 rad/s (38x) |
| Turn-on accel bias (std used in runs) | 0.05 m/s^2 | not determined | ground-truth estimate up to 0.550 m/s^2 (11x) |
| Position noise std | 0.5 m per axis | reported `pos_accuracy` 0.10 to 0.15 (unit as in `dataformat.txt`) | not stated |
| Gravity | 9.81 m/s^2 | `|f|` median 9.818 (moving) | `|f|` median 9.808 (moving) |

The difference-based bounds include motion and vibration and are upper bounds, not noise
densities. For EuRoC, `imu0/sensor.yaml` states noise and random-walk parameters (gyro
1.6968e-4 rad/s/sqrt(Hz), 1.9393e-5 rad/s^2/sqrt(Hz); accelerometer 2.0e-3 m/s^2/sqrt(Hz),
3.0e-3 m/s^3/sqrt(Hz)); these are shipped values, not measured here.

## Files

| File | Purpose |
|------|---------|
| `characterise.py` | Implementation of the protocol above. |
| `__init__.py` | Package marker for `python -m`. |
| `results/` | CSV output. |
| `figures/` | PNG output. |

## Running (from the repository root)

```bash
python -m analysis.dataset_char.characterise
```
