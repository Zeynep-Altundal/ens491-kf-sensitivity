# rviz_drone

ROS 2 Jazzy Python package: a 6-DoF quadrotor simulation with a cascaded position/attitude
controller that follows a fixed waypoint trajectory, and IMU and position measurements generated
with the error models of `synthetic/`. The drone, its truth trace and the measurements are shown in
RViz. It replaces the Isaac Sim attempt in `../isaac_drone/` (abandoned, see its README).

The package is built in its own workspace `../rviz_drone_ws/` inside this repository.

## Conventions (identical to `synthetic/`)

| Item | Convention |
|---|---|
| World frame `world` | ENU: x east, y north, z up. Origin on the ground at the take-off point. |
| Body frame `base_link` | FLU: x forward, y left, z up, origin at the centre of mass. The IMU frame equals the body frame (no misalignment unless configured). |
| Attitude | Hamilton unit quaternion, scalar first `[w, x, y, z]` in the code, body-to-world rotation `R_wb` (`v_w = R_wb v_b`). ROS messages carry the same quaternion in their `(x, y, z, w)` fields. |
| Gravity | `g_w = [0, 0, -9.81] m/s^2` (`synthetic.imu_model.GRAVITY_W`). |
| Gyroscope | Body angular rate `w_b`. |
| Accelerometer | Specific force `f_b = R_wb^T (a_w - g_w)`; at rest or in hover it reads `+9.81 m/s^2` on body z. |
| Units | SI: s, m, m/s, m/s^2, rad, rad/s, N, N m. |

## Model

* **Rigid body**: `p' = v`, `m v' = m g_w + R_wb [0, 0, T] - c_d v`,
  `q' = 1/2 q (x) [0, w_b]`, `J w_b' = tau - w_b x J w_b`. Integrated with fourth-order
  Runge-Kutta at 1 kHz; the quaternion is renormalised after every step.
* **Rotors**: four rotors in X configuration; per-rotor thrust limited to `[0, f_max]`; yaw torque
  `k_m f_i` with alternating sign. Commanded thrust and torques are mapped to rotor thrusts, clipped
  and mapped back, so saturation acts on the body.
* **Ground**: the plane `z = 0` supports the vehicle; while the net vertical force does not lift it,
  position, velocity and angular rate stay at their ground values (acceleration 0).
* **Controller** (1 kHz, zero-order hold): position PD with acceleration feed-forward and drag
  compensation gives the desired force; its direction and the yaw reference of the trajectory give
  the desired attitude (tilt limited to 45 deg); geometric attitude controller on SO(3) with
  angular-rate damping (with a feed-forward of the reference yaw rate) gives the torques, with lower
  gains on yaw. When the rotor thrusts would saturate, the yaw torque is
  reduced first, because the rotors have little yaw authority (at most 0.32 N m).
* **Parameters**: mass, inertia and arm are those of the vehicle rendered in Isaac Sim
  (`../isaac_render/`, NTNU ARL-Robot-1 asset), so that the visual and the dynamics describe the same
  vehicle: base link 1.2 kg with `diag(0.013, 0.014, 0.013) kg m^2` and four 0.01 kg propellers at
  `(+-0.1, +-0.1, 0) m` give `m = 1.24 kg`, `J = diag(0.013404, 0.014404, 0.013804) kg m^2`
  (parallel-axis theorem) and arm `0.1 sqrt(2) = 0.141421 m`. The rotor and drag parameters are
  illustrative, not taken from the asset: `f_max = 10 N` per rotor, `k_m = 0.016 m`,
  `c_d = 0.1 N s/m`. (Until 2026-11-27 the illustrative values `m = 1.5 kg`,
  `J = diag(0.029, 0.029, 0.055) kg m^2` and arm 0.25 m were used.)

## Trajectories (`reference.py`, closed form, analytic velocity and acceleration)

Two trajectories, selected by name (`flight.run(trajectory=...)`, ROS parameter / launch argument
`trajectory`); the default is `mission`.

### `mission` (default, 83 s, area about 42 m x 28 m, cruise height 3 m)

Every moving segment starts and ends at rest: its path is traversed with minimum-jerk timing, so
that position, velocity and acceleration are continuous at the joints. The heading reference is the
direction of the path tangent (direction of travel); heading changes between segments are made while
hovering (minimum-jerk yaw). In the slalom the heading is the direction of the velocity with the
westward speed taken as at least 2 m/s, so that it is defined at the slow ends. Positions in metres
(east, north).

| Segment | Time [s] | Content |
|---|---|---|
| ground | 0-1 | on the ground at home (0, 0), heading east |
| take-off | 1-5 | minimum-jerk climb to 3 m |
| hover | 5-7 | hover at (0, 0, 3) |
| straight | 7-16 | straight leg east to (30, 0); peak speed 6.25 m/s |
| turn | 16-22 | coordinated 90 deg left turn, radius 8 m, to (38, 8), heading north; peak speed 3.9 m/s |
| hover_turn | 22-25 | hover, heading north to west |
| figure_eight | 25-49 | one figure-eight from its south tip (38, 8): `x = 38 - 3.5 sin 2u`, `y = 8 + 10 (1 - cos u)`, `u` from 0 to 2 pi |
| aggressive | 49-63 | slalom west to (18, 8): lateral 1.5 m at period 2.8 s and vertical 0.5 m at period 3.5 s, envelope 2 s; peak acceleration 8.3 m/s^2 |
| hover_turn | 63-66 | hover, heading west to the direction of home (-156 deg) |
| return | 66-74 | straight back to home |
| hover | 74-77 | hover at (0, 0, 3) |
| landing | 77-82 | minimum-jerk descent to the ground |
| ground | 82-83 | on the ground |

### `figure_eights` (50 s, the default until 2026-11-27, unchanged)

| Segment | Time [s] | Content |
|---|---|---|
| ground | 0-1 | on the ground |
| take-off | 1-5 | minimum-jerk climb to 2 m |
| hover | 5-7 | hover at (0, 0, 2) |
| figure-eight | 7-27 | `x = 3 e sin(w t)`, `y = 1.5 e sin(2 w t)`, period 10 s, envelope `e` ramps in and out over 5 s |
| aggressive | 27-39 | figure-eight with period 4 s, amplitudes 2 m and 0.6 m, plus 0.5 m vertical oscillation; envelope 4 s |
| hover | 39-44 | hover at (0, 0, 2) |
| landing | 44-49 | minimum-jerk descent to the ground |
| ground | 49-50 | on the ground |

Yaw reference 0 throughout.

### Tracking

Closed-loop tracking with the default parameters (seed-independent; position error `|p - p_ref|`,
tilt and yaw error from the simulated attitude; produced by `python -m rviz_drone.tracking
[mission|figure_eights]`).

`mission`:

| Segment | max `|a_ref|` [m/s^2] | max error [m] | RMS error [m] | max tilt [deg] | max yaw error [deg] |
|---|---|---|---|---|---|
| take-off | 1.08 | < 0.001 | < 0.001 | 0.0 | 0.0 |
| hover (all) | 0.00 | 0.037 | 0.012 | 1.8 | 0.0 |
| straight leg | 2.14 | 0.042 | 0.024 | 13.7 | 0.0 |
| coordinated turn | 2.06 | 0.064 | 0.038 | 12.8 | 1.5 |
| hover with turn | 0.00 | 0.208 | 0.054 | 18.6 | 6.0 |
| figure-eight | 3.57 | 0.069 | 0.033 | 20.6 | 2.7 |
| aggressive | 8.34 | 1.309 | 0.618 | 45.5 (tilt limit) | 15.8 |
| return | 1.78 | 0.038 | 0.022 | 11.3 | 0.4 |
| landing | 0.69 | < 0.001 | < 0.001 | 0.0 | 0.0 |

The "hover with turn" maximum (0.208 m, 18.6 deg) is the settling after the aggressive segment, at
the start of the hover that follows it. In the aggressive segment the tilt limit is reached and the
heading lags the commanded swing of +-59 deg (low yaw authority); this segment is meant to excite
the filters, not to be tracked exactly.

`figure_eights` (unchanged by the yaw-rate feed-forward, which is 0 for this trajectory):

| Segment | max `|a_ref|` [m/s^2] | max error [m] | RMS error [m] | max tilt [deg] | max yaw error [deg] |
|---|---|---|---|---|---|
| take-off | 0.72 | < 0.001 | < 0.001 | 0.0 | 0.0 |
| hover (all) | 0.00 | 0.036 | 0.008 | 4.1 | 0.4 |
| figure-eight | 3.10 | 0.089 | 0.052 | 19.2 | 2.8 |
| aggressive | 7.53 | 0.611 | 0.284 | 45.5 (tilt limit) | 21.6 |
| landing | 0.46 | < 0.001 | < 0.001 | 0.0 | 0.0 |

With the earlier illustrative vehicle parameters the `figure_eights` figure-eight yaw was 2.9 deg
and the aggressive errors 0.619 m (max) and 0.287 m (RMS); all other entries were equal.

## Sensors (from `synthetic/`)

The whole flight is simulated first. The ideal IMU is computed from the simulated state with
`synthetic.imu_model.ideal_imu` and corrupted with `synthetic.imu_model.corrupt_sensor`; position
measurements come from `synthetic.measurements.position_measurements`. Error parameters and seeds
are those of `synthetic.scenario.ScenarioConfig` (same random-stream split, `RNG_STREAMS`), so the
same seed always gives the same flight and the same measurements.

## Flight recorder (`recorder.py`, common recording format v1)

One flight is written to one `.npz` file in the common recording format of the filter comparison
(`docs/common_interface.md` on branch `feat/W03-synthetic-scenarios` of the companion repository
`ens491-nkf`, frozen 2026-11-25, v1; read-only reference, not copied here). The file is readable
with `numpy.load(path, allow_pickle=False)` and with the reader `load_recording` of that repository;
no ROS or Isaac Sim is needed.

### Signals, rates, units and frames

| Key | Shape | Unit | Rate | Frame | Content |
|---|---|---|---|---|---|
| `imu_t` | [N] float64 | s | 200 Hz | - | IMU time stamps, `k / 200`, from 0 to the end of the flight |
| `imu_gyro` | [N,3] | rad/s | 200 Hz | body (FLU) | measured angular rate (white noise, turn-on bias, bias random walk) |
| `imu_accel` | [N,3] | m/s^2 | 200 Hz | body (FLU) | measured specific force `f_b = R^T (a_w - g_w)`, gravity not removed |
| `imu_gyro_clean`, `imu_accel_clean` | [N,3] | as above | 200 Hz | body | the same without any error |
| `truth_t` | [M] | s | 200 Hz | - | truth time stamps (equal to `imu_t`) |
| `truth_p`, `truth_v` | [M,3] | m, m/s | 200 Hz | world (ENU) | true position and velocity of the centre of mass |
| `truth_q` | [M,4] | - | 200 Hz | body to world | true attitude, Hamilton `[w, x, y, z]`, unit norm |
| `truth_bg`, `truth_ba` | [M,3] | rad/s, m/s^2 | 200 Hz | body | true gyro and accelerometer bias (turn-on bias plus random walk) |
| `pos_t` | [K] | s | 10 Hz | - | position measurement time stamps (every 20th IMU sample) |
| `pos_z` | [K,3] | m | 10 Hz | world (ENU) | measured position; NaN where `pos_valid` is false |
| `pos_R` | [K,3,3] | m^2 | 10 Hz | world | covariance told to the filter: `diag(meas_sigma^2)` (equal to the true noise here) |
| `pos_valid` | [K] bool | - | 10 Hz | - | false inside a dropout (none in the default configuration) |
| `events` | JSON string | - | - | - | injected disturbances; `[]` (disturbances are added afterwards in `ens491-nkf`) |
| `meta` | JSON string | - | - | - | metadata, below |
| `sim_p_ref`, `sim_w_b`, `sim_a_w`, `sim_rotor_thrust` | [M,3], [M,3], [M,3], [M,4] | m, rad/s, m/s^2, N | 200 Hz | world, body, world, - | extra keys of this simulator (reference position, true body rate, true acceleration, rotor thrusts); not part of the common format, ignored by its reader |

`N = M = 200 T + 1` and `K = 10 T + 1` for a flight of `T` seconds (mission: 16 601 and 831).

### Metadata (`meta`)

`version` (`"v1"`), `id` (`<trajectory>_seed<seed>`), `platform` (`"uav"`), `source`
(`"ens491_quadrotor_sim"`), `imu_rate_hz` (200), `pos_rate_hz` (10), `seed`, `trajectory`,
`duration_s`, `true_noise` (noise densities, turn-on biases, bias random walks, position sigma),
`frames` (world ENU, body FLU = IMU frame, quaternion Hamilton `[w, x, y, z]` body to world),
`gravity_w`, `git_commit` and `git_dirty` (state of this repository when the file was written),
`parameters` (vehicle `QuadParams`, `ControllerGains`, `sim_rate_hz` 1000, the full
`ScenarioConfig` of `synthetic/`), `generator` (`rviz_drone.recorder`).

### File layout

`simulation/recordings/<trajectory>_seed<seed>.npz` (folder version-controlled with its README,
the `.npz` files are not). The archive is written member by member in sorted key order with a fixed
time stamp (1980-01-01) and fixed permissions, so that the same code (`git_commit`), trajectory and
seed give a bit-identical file; `numpy.savez_compressed` stores the current time and does not.

### Frames used by this simulator (answer to the OPEN item of `common_interface.md`, section 1)

The recorder of this repository does not take its frames from Isaac Sim or Pegasus: the flight is
simulated by `rviz_drone` and Isaac Sim 6.0.1 only renders it (kinematic replay, `../isaac_render/`);
Pegasus is not used (the Isaac Sim 5.1.0 / Pegasus attempt in `../isaac_drone/` was abandoned).
The simulator works natively in the frames of the common interface: world ENU (origin on the
ground at the take-off point, which is also the first truth sample), body FLU equal to the IMU
frame, Hamilton `[w, x, y, z]` body to world, specific force with gravity not removed. No
conversion is applied when the recording is written. Isaac Sim's stage is set up as ENU with z up
and metres (`../isaac_render/usd_pose.py`), so the rendered and recorded poses are the same numbers.

Mismatches with the common interface (none in the frames, three in the bookkeeping):

1. `meta.source`: the list of `common_interface.md` has no value for this simulator; the file uses
   `"ens491_quadrotor_sim"` (OPEN for the owner of the common interface: add it or name another
   value).
2. `meta.version`: the document says v1, while its writer `nkf/data/recording.py` writes
   `"draft-2026-11-25"`; this recorder writes `"v1"`.
3. `meta.pos_rate_hz` is listed in the document but not written by that writer; this recorder
   writes it.

### Done criteria

1. Round trip: every array loaded from the file equals the source array exactly (`np.array_equal`,
   NaN in the same places) and the JSON fields parse back to the same values.
2. Time stamps: strictly increasing, steps equal to 1/200 s (IMU, truth) and 1/10 s (position)
   within 1e-9 s; `truth_t` equals `imu_t`; `pos_t` is every 20th IMU time stamp.
3. Consistency: strapdown integration of the recorded clean IMU (the second-order scheme of
   `synthetic/tests/test_synthetic.py`), restarted from the recorded truth every 1 s, stays within
   5 mm of the recorded truth position (measured 2.0 mm, mission, worst window at t = 62 s in the
   aggressive segment); `imu_gyro - imu_gyro_clean` minus `truth_bg` has the white-noise standard
   deviation of the configuration within 5 % (same for the accelerometer).
4. Reproducibility: two recordings with the same seed are byte-identical; a different seed gives
   different measurements and the same truth.
5. The full test suite passes; the mission recording is written and its size and summary are
   recorded below.

### Usage

```bash
.venv/bin/python -m rviz_drone.recorder --trajectory mission --seed 0   # from simulation/rviz_drone/
```

Mission recording of 2026-11-27 (`simulation/recordings/mission_seed0.npz`, not version-controlled):
4 788 647 bytes, written in 20.8 s; two runs in separate processes gave the same SHA-256
(`e6138b37...05a079cd`). Written at commit `f324994` with uncommitted changes (`git_dirty: true`).
It also loads with `load_recording` of `ens491-nkf` (`nkf/data/recording.py` on
`feat/W03-synthetic-scenarios`, run from a temporary copy, that repository unchanged).

| Item | Value |
|---|---|
| trajectory, seed | mission, 0 |
| duration | 83.000 s |
| IMU samples (rate) | 16601 (200 Hz) |
| position samples (rate), valid | 831 (10 Hz), 831 |
| max speed | 6.26 m/s |
| max gyro norm | 4.78 rad/s |
| accel norm range | 7.86 to 17.10 m/s^2 |
| position error RMS (per axis) | 0.505, 0.478, 0.484 m |
| gyro bias at end | 2.00e-03, -1.02e-03, 1.26e-03 rad/s |
| accel bias at end | 0.038, -0.041, 0.012 m/s^2 |

## Topics and rates

| Topic / TF | Type | Frame | Rate |
|---|---|---|---|
| `/drone/truth/pose` | `geometry_msgs/PoseStamped` | `world` | 200 Hz |
| TF `world` -> `base_link` | `tf2_msgs/TFMessage` on `/tf` | - | 200 Hz |
| `/drone/imu` | `sensor_msgs/Imu` (orientation not provided: `orientation_covariance[0] = -1`) | `base_link` | 200 Hz |
| `/drone/position` | `geometry_msgs/PointStamped` (not published during dropout) | `world` | 10 Hz |
| `/drone/truth/path` | `nav_msgs/Path` (truth, one pose per 0.1 s) | `world` | 2 Hz |
| `/drone/marker` | `visualization_msgs/Marker`, ids 0-2 (body, arms, rotors), frame-locked | `base_link` | 1 Hz per id (3 messages/s) |

Time stamps are `t0 + t_k`, with `t0` the ROS time at the start of publication. The node simulates the whole flight at start-up (about 21 s for the 83 s mission) before it starts
publishing. The simulation itself runs at 1 kHz; the published streams are sub-sampled from it. After the end of the flight the
node stops publishing and logs a message.

## Isolation from other ROS 2 work on this machine

`env.sh` sets `ROS_DOMAIN_ID=101` and `ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST` for the current
shell, so that the nodes of this package never discover nodes on the default domain, and
`ROS_LOG_DIR` to `../rviz_drone_ws/log/ros`, so that ROS 2 logs are not written to `~/.ros/log`.

## Usage

```bash
source simulation/rviz_drone/env.sh          # ROS 2 Jazzy, private domain, workspace overlay
simulation/rviz_drone/build.sh               # colcon build in simulation/rviz_drone_ws
source simulation/rviz_drone/env.sh          # again, to pick up the freshly built overlay
ros2 launch rviz_drone rviz_drone.launch.py              # with RViz
ros2 launch rviz_drone rviz_drone.launch.py rviz:=false  # headless
ros2 launch rviz_drone rviz_drone.launch.py trajectory:=figure_eights   # the earlier trajectory
```

Tests (pure NumPy, no ROS needed): `.venv/bin/python -m pytest -q simulation/rviz_drone/test`.

OPEN: the node and the launch file were changed for the trajectory parameter (2026-11-27) but not
run again with ROS 2; the RViz view (`Distance: 14` in `rviz/rviz_drone.rviz`) was set for the
area of `figure_eights` and has to be zoomed out for the mission.

## Done criteria

1. The unit tests pass: hover equilibrium and energy consistency of the dynamics, accelerometer
   `+g` on body z at hover, reproducibility from a fixed seed, and trajectory tracking.
2. `colcon build` succeeds in `simulation/rviz_drone_ws`.
3. A 20 s headless run publishes `/drone/imu` and `/drone/truth/pose` at 200 Hz and
   `/drone/position` at 10 Hz within 5 % (measured with `ros2 topic hz`).
4. With RViz, the drone body, the TF frames, the truth path and the position measurements are
   visible (manual check; not part of the automated tests).

## Files

| File | Purpose |
|---|---|
| `rviz_drone/quadrotor.py` | Parameters, rigid-body dynamics with rotor allocation and ground contact, controller. |
| `rviz_drone/reference.py` | Reference trajectories (`mission`, `figure_eights`) with analytic derivatives and yaw reference. |
| `rviz_drone/tracking.py` | Tracking table of the closed-loop flight. |
| `rviz_drone/flight.py` | Simulates the flight and generates the sensor streams with `synthetic/`. |
| `rviz_drone/repo_path.py` | Makes the repository root importable (for `synthetic/`). |
| `rviz_drone/recorder.py` | Writes one flight in the common recording format (`.npz`). |
| `rviz_drone/sim_node.py` | ROS 2 node that publishes the streams in real time. |
| `launch/rviz_drone.launch.py` | Starts the node and, optionally, RViz. |
| `rviz/rviz_drone.rviz` | RViz configuration. |
| `test/` | Unit tests. |
| `env.sh`, `build.sh` | Environment and build helpers. |
