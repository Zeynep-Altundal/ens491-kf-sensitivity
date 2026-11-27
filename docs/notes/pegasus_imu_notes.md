# Notes for the Pegasus / Isaac Sim recordings (read before writing the recorder)

Source checked: `PegasusSimulator/PegasusSimulator`, commit `e13dc65` (24 Jul 2026),
`extensions/pegasus.simulator/pegasus/simulator/logic/sensors/imu.py` and `gps.py`.
Please re-check if a newer Pegasus version is used.

## Why this matters

The bias experiments compare the estimated IMU bias with the **true** bias. The common recording
format (`docs/common_interface.md`) therefore needs the IMU without simulator noise, and all errors
are added afterwards with `experiments/nav_compare/data/disturbances.py`, which logs them.

## IMU sensor (`imu.py`)

- The IMU adds white noise and a gyroscope bias (first-order Gauss–Markov) internally. The bias is
  kept in the private attribute `_gyroscope_bias` and is **not** part of the sensor output.
- **Recommended:** set the noise to zero in the IMU config, then add errors ourselves:
  `{"gyroscope": {"noise_density": 0.0, "random_walk": 0.0},
    "accelerometer": {"noise_density": 0.0, "random_walk": 0.0}}`.
- Things that look wrong in this version (another reason not to rely on the built-in model):
  - the accelerometer bias is computed but not added (`#+ self._accelerometer_bias[i]`
    is commented out), and its random walk uses `np.random.rand()` (uniform 0..1) instead of
    `randn()`;
  - the gyroscope bias uses the **accelerometer** correlation time (`tau_g = ...accelerometer...`);
  - `turn_on_bias_sigma` is read but never used.
- The specific force is computed by differencing the simulated velocity at the IMU update rate
  (default 250 Hz), so it carries a small numerical error; log `dt` as measured.
- **Frames:** the output follows PX4: body **FRD**, world **NED**. The repository uses body FLU /
  world ENU (`docs/interfaces.md`). Convert when writing the recording, never inside a filter:
  FLU = rotate FRD by 180 deg about x (y and z change sign); ENU from NED: x and y swap, z changes
  sign. The `orientation` field is the true attitude (no noise); it is truth, not a sensor.
- The random numbers come from the global `np.random` without a seed; set a seed if a recording
  must be repeatable.

## GPS sensor (`gps.py`)

- Output is latitude / longitude / altitude with a correlated random-walk error
  (`gps_xy_random_walk`, `gps_correlation_time`, ...). Easiest for us: record the true ENU position
  from the simulator state and add position noise afterwards with `disturbances.py`.

## What the recorder should write

The keys of `docs/common_interface.md`, section 2: `imu_t`, `imu_gyro`, `imu_accel` (FLU,
specific force), `truth_*` (ENU), `pos_t`, `pos_z`, `pos_R`, `pos_valid`, `meta`, `events`.
