# synthetic

Synthetic scenario generator producing IMU and position-measurement data with fully controlled
sensor noise, bias drift, outliers, measurement dropout and motion profiles.

## Conventions

| Item | Convention |
|------|------------|
| World frame | ENU (x east, y north, z up). |
| Gravity | `g_w = [0, 0, -9.81] m/s^2`. |
| Attitude | Hamilton unit quaternion, scalar first `[w, x, y, z]`, representing the body-to-world rotation `R_wb` (`v_w = R_wb v_b`). |
| Euler angles (profile definition only) | Intrinsic Z-Y-X: `R_wb = Rz(yaw) Ry(pitch) Rx(roll)`. |
| Gyroscope | Measures the body angular rate `w_b`, defined by `[w_b]_x = R_wb^T dR_wb/dt`. |
| Accelerometer | Measures the specific force `f_b = R_wb^T (a_w - g_w)`. |
| Time | IMU samples at `t_k = k / imu_rate`, `k = 0, ..., duration * imu_rate`; position measurements at every `imu_rate / meas_rate`-th IMU sample, starting at `t = 0`. |

## Motion profiles (`trajectories.py`)

All quantities are evaluated in closed form (no numerical differentiation).

| Profile | Description |
|---------|-------------|
| `straight` | Horizontal straight line at a 30 deg heading. Repeating 30 s pattern of constant-acceleration segments: rest 7 s, +0.5 m/s^2 for 8 s, cruise 7 s, -0.5 m/s^2 for 8 s. Segments are joined by 1 s linear acceleration ramps, so the acceleration is continuous. Constant attitude. |
| `turning` | Planar figure-eight `x = 20 sin(w t)`, `y = 10 sin(2 w t)` m with a 30 s period, yaw along the velocity, zero roll and pitch. |
| `aggressive` | 3D Lissajous position (amplitudes 10, 8, 2 m; periods 20, 14, 10 s) with sinusoidal roll (0.6 rad, 0.8 Hz), pitch (0.4 rad, 0.6 Hz) and yaw (1.0 rad, 0.25 Hz, plus 0.3 rad/s drift). |

## Error models

IMU (`imu_model.py`), per sensor: `y[k] = clean[k] + b[k] + n[k] + o[k]`.

| Error | Model | Switch |
|-------|-------|--------|
| White noise | `n[k] ~ N(0, sigma_d^2)` per axis, `sigma_d = density / sqrt(dt)` | `enable_white_noise` |
| Turn-on bias | constant vector added to `b[k]` | `enable_turn_on_bias` |
| Bias random walk | `r[0] = 0`, `r[k+1] = r[k] + sigma_bw sqrt(dt) n`, `n ~ N(0, 1)`, added to `b[k]` | `enable_bias_rw` |
| Outlier spikes | per sample and axis with probability `outlier_prob`: `+/- outlier_magnitude` (random sign) | `enable_imu_outliers` |

Position measurements (`measurements.py`): `z[j] = p_true(t_j) + v[j] + o[j]`.

| Error | Model | Switch |
|-------|-------|--------|
| Noise | `v[j] ~ N(0, diag(meas_sigma^2))` | `enable_meas_noise` |
| Dropout | epochs with `start <= t_j < start + duration` for any window are unavailable (NaN) | `enable_meas_dropout` |
| Outliers | per epoch with probability `meas_outlier_prob`: offset of norm `meas_outlier_magnitude` in a uniformly random direction | `enable_meas_outliers` |

IMU switches apply to both gyroscope and accelerometer; a single sensor's error is disabled by
setting its parameter to zero. Units: gyro noise density rad/s/sqrt(Hz), gyro bias random walk
rad/s^2/sqrt(Hz), accelerometer noise density m/s^2/sqrt(Hz), accelerometer bias random walk
m/s^3/sqrt(Hz). The default values in `ScenarioConfig` are illustrative, of the order of a
MEMS-grade IMU, and are not taken from a specific sensor datasheet.

## Reproducibility

`ScenarioConfig.seed` seeds a `numpy.random.SeedSequence` that is split into one stream per error
source, each driving its own `numpy.random.Generator`. Identical configurations give identical
output, and switching one error source on or off leaves the realisation of the others unchanged.

## Sensor-to-body misalignment

`ScenarioConfig.imu_misalignment_x_deg` sets a constant rotation `R_BS = Exp(dphi e_x)` from the
sensor frame to the body frame about the body x axis (default 0). The ideal sensor then measures
`w_S = R_BS^T w_B` and `f_S = R_BS^T f_B`; white noise, turn-on bias, bias random walk and spikes
are generated in the sensor frame and added to these signals, because they are properties of the
sensor axes. Since the per-axis white noise is isotropic, its covariance is unchanged by the
rotation (`R^T sigma^2 I R = sigma^2 I`). `gyro_clean`, `accel_clean`, the measured signals and the
true biases are all expressed in the sensor frame; `R_bs` is stored with the scenario. With an
angle of 0 no rotation is applied, so existing scenarios are reproduced bit for bit.

## Gyroscope scale factor

`ScenarioConfig.gyro_scale_factor = s` (default 0) multiplies the clean gyroscope signal by
`(1 + s)` on all three axes before the sensor errors are added, `w_m = (1 + s) w_S + b_g + n_g`:
the scale-factor error acts on the sensed rate, while the noise density, the biases and the spikes
keep their configured values. `gyro_clean` is the ideal rate before scaling. With `s = 0` no
operation is applied, so existing scenarios are reproduced bit for bit.

## Known discretization error

The ground truth is exact: the analytic body rate, velocity and acceleration agree with central
finite differences of the attitude, position and velocity to within 2.4e-9 rad/s, 1.8e-9 m/s and
7.2e-10 m/s^2 respectively (`aggressive` profile). Integrating the noise-free IMU with the
second-order strapdown scheme used in the tests (rotation vector with coning term
`dt^2/12 (w_k x w_k+1)`, trapezoidal velocity) nevertheless leaves a discretization error that
depends on the IMU rate and the rotational dynamics. For the `aggressive` profile the final
position error after 60 s is **0.45 m at 200 Hz**:

| IMU rate | Final position error after 60 s (`aggressive`) | Ratio to next rate | Observed order |
|----------|-----------------------------------------------|--------------------|----------------|
| 100 Hz | 1.7977 m | 3.9992 | 1.9997 |
| 200 Hz | 0.44952 m | 3.9998 | 1.9999 |
| 400 Hz | 0.11239 m | 4.0000 | 2.0000 |
| 800 Hz | 0.028097 m | - | - |

The error falls by a factor of 4 each time `dt` is halved, as expected for a second-order
integrator; it is a property of the discrete-time integration, not of the generated signals. The
tests therefore check the `aggressive` profile by its convergence order (`2 +/- 0.1`), while
`straight` (9.1e-12 m) and `turning` (6.9e-05 m) are held to an absolute bound of 1 cm at 200 Hz.
Any filter that propagates this IMU data at 200 Hz inherits a comparable integration error on the
`aggressive` profile, which should be kept in mind when interpreting its position error.

## Files

| File | Purpose |
|------|---------|
| `rotation.py` | Quaternion and Euler-angle utilities. |
| `trajectories.py` | Analytic ground-truth profiles. |
| `imu_model.py` | Ideal IMU and IMU error model. |
| `measurements.py` | Position measurements with noise, dropout and outliers. |
| `scenario.py` | `ScenarioConfig`, `generate(config)` and `.npz` input/output. |
| `plots.py` | Writes per-profile figures to `figures/`. |
| `tests/` | Pytest tests. |

`generate(config)` returns a dictionary with ground truth (`t, p_true, v_true, a_true, q_true,
w_true`), clean IMU (`gyro_clean, accel_clean`), corrupted IMU (`gyro_meas, accel_meas`), true
biases (`gyro_bias, accel_bias`), IMU outlier masks, measurements (`t_meas, meas_index,
p_meas_true, z_meas, meas_available, meas_outlier_mask`) and the configuration (`config_json`).
`meas_outlier_mask` is false for unavailable epochs.

## Running (from the repository root)

```bash
python -m pytest -v -s synthetic
python -m synthetic.scenario --profile turning --out scenario.npz
python -m synthetic.plots
```
