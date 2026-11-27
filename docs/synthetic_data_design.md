# How the synthetic data is built, and where the numbers come from

This document explains the scenario generator (`experiments/nav_compare/data/scenarios.py`), the disturbance sets
(`experiments/nav_compare/data/disturbances.py`) and the simulated vehicle signals (`experiments/nav_compare/data/odometry.py`,
`experiments/nav_compare/data/magnetometer.py`). For every number it says whether it comes from a source or was chosen
by us, and why. Notation: `docs/notation.md`.

**Honest summary.** The *structure* of the models (white noise + bias random walk, measurement =
truth + errors, quadrotor attitude from thrust direction) is standard and sourced. Most *values*
are order-of-magnitude choices by us, checked against the sources below where possible; they are
not fitted to one real sensor. Section 8 lists what should be aligned with a real datasheet.

## 1. Truth first, then sensors

1. Write the true motion as smooth formulas: position, velocity, attitude at every IMU time stamp.
2. Compute the ideal sensor outputs from the truth (no errors): angular rate in the body frame and
   specific force `f_b = R^T (a_w - g)` (the accelerometer feels acceleration minus gravity).
3. Add errors one by one (each with its own random stream, so one error can be switched on alone).
4. Keep the truth and the clean signals for evaluation; the filter never sees them.

This is the usual way to build IMU simulators; it makes the true state, the true bias and the exact
time of every disturbance known, which is what NEES and the peak analysis need.

## 2. Motion models

| Platform | Model | Default values | Source / reason |
| --- | --- | --- | --- |
| UGV | Car moving forward along its body x axis; yaw rate = A sin(2πt/T); full stops; optional side slip and road grade | 8 m/s, A = 0.15 rad/s, T = 30 s, a stop at 40–50 s, slip = grade = 0 | Chosen: city-like speed with turns and a stop (ZUPT chance). Slip and grade exist to break the NHC assumption on purpose |
| UAV | Smooth 3-D Lissajous path; attitude from the thrust direction (body z along `a - g`) plus a slow yaw | ±2 m / ±1.5 m / ±0.3 m, period 20 s, yaw ±0.3 rad | The thrust-direction attitude is the differential-flatness property of quadrotors (Mellinger & Kumar, ICRA 2011, from memory); amplitudes chosen for an indoor-like flight similar to EuRoC |
| Both | Start ramp with a 5th-order smooth step | – | Avoids the acceleration jump of a 3rd-order ramp (found in the older generator) |

## 3. IMU error model

Per axis, gyroscope and accelerometer:

```
omega_m = (I + S_g)(I + M_g) omega + b_g(t) + n_g ,   d b_g / dt = n_bg
f_m     = (I + S_a)(I + M_a) f     + b_a(t) + n_a ,   d b_a / dt = n_ba
```

`n_g, n_a` white noise with density `sigma_g, sigma_a`; `n_bg, n_ba` white noise driving the bias
random walk with density `sigma_bg, sigma_ba`; `b(0)` a random turn-on bias; `S` scale factor
(diagonal), `M` misalignment (small rotation). Optional: quantization, saturation, a vibration
sinusoid, a gravity-magnitude error, time-stamp jitter and dropped samples.

Discrete time (step `dt`): white noise std per sample `sigma / sqrt(dt)`, bias random-walk step
std `sigma_b * sqrt(dt)`. This is the continuous-time model used by Kalibr
([IMU noise model](https://github-wiki-see.page/m/stereolabs/kalibr/wiki/IMU-Noise-Model): white
noise + bias random walk, parameters `*_noise_density` and `*_random_walk`; the page also warns that
the model is optimistic and that values for low-cost sensors may need to be 10x larger).

### Values (grade `euroc` is the default since 27 Nov; `medium` was the default before)

| Parameter | low | medium | high | **euroc** | EuRoC ADIS16448 (Kalibr yaml) |
| --- | --- | --- | --- | --- | --- |
| Gyro noise density [rad/s/√Hz] | 1e-5 | 1e-4 (0.34 °/√h) | 5e-4 | **2e-4** | 1.70e-4 |
| Accel noise density [m/s²/√Hz] | 1e-4 | 1e-3 | 5e-3 | **2e-3** | 2.0e-3 |
| Gyro bias random walk [rad/s²/√Hz] | 1e-7 | 1e-6 | 1e-5 | **2e-5** | 1.94e-5 |
| Accel bias random walk [m/s³/√Hz] | 1e-6 | 1e-5 | 1e-4 | **3e-3** | 3.0e-3 |
| Gyro turn-on bias std [rad/s] | 5e-5 | 5e-4 (≈100 °/h) | 3e-3 | **1.5e-3** | – |
| Accel turn-on bias std [m/s²] | 2e-3 | 1e-2 (≈1 mg) | 5e-2 | **4e-2** | – |

EuRoC values: `imu0/sensor.yaml` of the EuRoC MAV dataset
([copy](https://docs.rs/crate/euroc/latest/source/test_data/imu0/sensor.yaml), comment "VI-Sensor
IMU (ADIS16448)", 200 Hz). The `medium` random-walk values were much smaller than EuRoC (gyro about 20x, accel about 300x);
therefore the default is now `euroc`, which uses the same densities as `synthetic/ScenarioConfig`
of this repository (so both generators have the same IMU noise level). The EuRoC yaml values are the ones
used for VIO tuning and are probably inflated as Kalibr suggests; we have not checked the ADIS16448
datasheet. Grade names: "low" = low noise (tactical-like), "high" = high noise (consumer-like);
[SBG](https://www.sbg-systems.com/glossary/imu-grades/) only states that tactical gyros are often
below 1 °/h bias stability, so the grade labels are approximate.

## 4. Position measurement (GPS-like)

`z = p + eta`, `eta ~ N(0, 0.5² I)` m, 1 Hz; the filter is told the same R unless a scenario changes
it on purpose. **0.5 m is optimistic** for standalone GNSS: in Hong Kong urban canyons a u-blox
receiver had mean 2-D errors of 9–17 m and maxima of 24–112 m
([Wen et al., GNC factor graph](https://arxiv.org/pdf/2109.00667)). We keep 0.5 m as the clean
baseline and model the bad parts with the disturbance sets (section 6).

## 5. Vehicle signals (UGV)

| Signal | Model | Values | Source / reason |
| --- | --- | --- | --- |
| Wheel speed | forward body speed × (1 + scale error) + white noise, 50 Hz | std 0.05 m/s, scale error 0 | Chosen. A real odometer counts pulses; [Ouyang et al.](https://arxiv.org/pdf/2007.10543.pdf) model the pulse round-off (std ≈ 0.3 pulse) and estimate the scale online. A scale error of about 1 % would be a realistic next test (not from a source) |
| Steering angle | bicycle model `delta = atan(L ω_z / v)` + white noise; filter derives `ω_z = v tan(delta) / L` | L = 2.7 m, std 0.003 rad; not used below 1 m/s | Bicycle model: standard. Noise chosen; it makes the steering yaw rate about 10x noisier than the medium gyro |
| Magnetometer heading | true heading + offset + white noise, 10 Hz | std 2°, offset 0 | Calibrated vehicle magnetometers reach about 0.6–1.9° RMSE, uncalibrated 8.8–24° ([JHU study](https://par.nsf.gov/servlets/purl/10184554)); 2° is the "calibrated" case. Hard/soft-iron offsets are a realistic next test |

## 6. Disturbance sets (`standard_event_sets`)

Each set is applied to the same flight; windows avoid the first 10 s. Sizes are chosen to be
**clearly visible** (many times the normal noise), so that the reaction of each filter can be seen;
they are stress tests, not typical values.

| Set | Size | Compared with normal | Real-world reason |
| --- | --- | --- | --- |
| gyro_noise_burst (3 s) | 0.05 rad/s extra std | ~50x the per-sample gyro noise (1e-3 rad/s at 100 Hz) | vibration, rough road, rotor resonance |
| accel_noise_burst (3 s) | 1.0 m/s² extra std | ~100x the per-sample accel noise | bumps, vibration |
| gyro_spikes (20 samples) | ±1 rad/s | single-sample shocks | impacts, electrical glitches |
| accel_spikes (20 samples) | ±20 m/s² (~2 g) | single-sample shocks | potholes, landing impacts |
| gyro_bias_step x/y/z | 0.01 rad/s from 40 % to 70 % of the run | 20x the medium turn-on bias | temperature jump or sensor fault (large on purpose) |
| accel_bias_step x/y/z | 0.2 m/s² | 20x the medium turn-on bias | same |
| pos_noise_burst (3 s) | 5 m extra std | 10x the 0.5 m GPS noise | urban multipath (errors of tens of metres reported in the source of section 4) |
| pos_spikes (5 samples) | 20 m | 40x | NLOS jumps; pseudorange errors of 62–96 m reported in the same source |
| pos_bias_step | 3 m in x and y for 30 % of the run | 6x | multipath bias that does not average out |
| pos_outage_10s / 30s, gnss_denied | no position | – | tunnels, indoor, jamming; gnss_denied = the setting of the project title |
| imu_and_pos_burst | the three bursts together | – | rough terrain in a city |

The filters are never told about a disturbance (R and Q stay as for the clean case).

## 7. Where each piece lives

| File | Content |
| --- | --- |
| `experiments/nav_compare/data/scenarios.py` | motion models, IMU error model, GPS-like updates, scenario grid |
| `experiments/nav_compare/data/disturbances.py` | time-windowed events and the standard sets, with an event log |
| `experiments/nav_compare/data/odometry.py`, `experiments/nav_compare/data/magnetometer.py` | wheel speed, steering, heading |
| `experiments/nav_compare/data/recording.py` | one run as a `.npz` (common recording format) |

## 8. Open points (to align with reality)

1. Fit the IMU values to the datasheet or an Allan-variance analysis of the IMU we will actually use
   (EuRoC ADIS16448 for the drone; for the UGV the IMU of the chosen data set).
2. GNSS: use a realistic receiver error (metres, time-correlated) instead of 0.5 m white noise for
   the "realistic" runs; keep 0.5 m as the clean baseline.
3. Wheel speed scale error (~1 %) and magnetometer hard-iron offset as extra scenarios.
4. Check the disturbance sizes against a real recording (e.g. the vibration level of a real drone).

## 9. Relation to the generator in `synthetic/`

Both generators build the truth first and use the same conventions and the same IMU noise model.
They are used for different questions:

| | `synthetic/` | `experiments/nav_compare/data/` |
| --- | --- | --- |
| Motion | `straight`, `turning` (figure-eight, yaw along velocity), `aggressive` (3-D, large roll/pitch); the quadrotor of `simulation/rviz_drone` uses it for its sensors | car with stops (UGV), smooth Lissajous drone path (UAV) |
| IMU noise | EuRoC level | `euroc` grade by default (same densities) |
| Turn-on bias | fixed vector | random per seed (std above) |
| Position | 10 Hz, 0.5 m, random outliers and dropout windows | 1 Hz, 0.5 m (GNSS-like); events in a time window |
| Disturbances | random spikes with a probability | logged events: bias step, noise burst, spike, outage |
| Extra signals | - | wheel speed, steering, magnetometer heading |

The 7 filters can be run on recordings of either generator (common recording format), so a
separate experiment on `synthetic/` data is only needed for its aggressive profile (large roll and
pitch), which our UAV path does not have.

## 10. Wheel-speed check (27 Nov)

The earlier result "wheel speed sometimes makes the ESKF worse" was checked:

- The wheel-speed innovations are consistent: NIS mean 0.98-1.00 in every part of the drive
  (expected 1), innovation mean below 0.003 m/s. Time alignment and the given std are correct.
- The final position error is dominated by the heading (yaw) error in all runs.
- 10 seeds, UGV, GPS only for the first 10 s, ESKF, final position error (median) and yaw error:

  | Aiding | `medium` grade | `euroc` grade |
  | --- | --- | --- |
  | NHC | 6.6 m, 1.1 deg | 23.2 m, 3.6 deg |
  | NHC + wheel speed | 12.7 m, 1.7 deg | 42.5 m, 6.3 deg |
  | NHC + magnetometer | 0.71 m, 0.08 deg | 1.56 m, 0.11 deg |
  | NHC + wheel speed + magnetometer | 0.77 m, 0.08 deg | 0.69 m, 0.11 deg |

  Wheel speed was better than NHC alone in 3/10 and 2/10 seeds; with the magnetometer it helped
  in 4/10 and 8/10 seeds.
- Interpretation (hypothesis): without a heading measurement the yaw angle is unobservable, but the
  ESKF, linearised at its own estimate, treats it as partly observable and corrects it from the
  wheel-speed updates in the wrong direction (the known EKF inconsistency of unobservable
  directions). Support: the right-invariant EKF, designed to avoid this, had 1.7 m instead of
  16.3 m in the same setting (`compare_aiding_outage`, `ugv_static_short`, median of 5 seeds).
  Check (27 Nov, 10 seeds, `results/wheel_speed_check/yaw_test.py`): with wheel speed the ESKF's
  yaw error grows from 3.6 to 6.3 deg and its attitude NEES at the end from 1.1 to 5.0 (it becomes
  overconfident about the attitude); the InEKF's yaw error grows only from 3.9 to 4.6 deg and its
  NEES stays at 1.7. This supports the hypothesis (the extra error comes from the linearisation of the
  ESKF, not from the wheel-speed data); a proof would need first-estimate Jacobians.
