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

## ROS 2 graph (simulator, Isaac Sim, RViz)

All names, types, rates, QoS and colours below are defined once in `rviz_drone/topics.py`; the
node, the launch file and the Isaac Sim subscriber import them from there.

### Node graph

```
                       /drone/truth/pose (200 Hz) ─────────────────┐
                       /drone/sim/info (latched, once) ────────────┤
 rviz_drone_sim ──────┤                                            ├──> isaac_drone_render (Isaac Sim 6.0.1,
 (system Jazzy rclpy)  │ /tf, /drone/imu, /drone/position,          │     bundled Jazzy rclpy; moves the
                       │ /drone/truth/path, /drone/marker ─────────>│     drone, renders, no publications)
                       │                                            └──> rviz2 (optional)
 (reserved, not implemented) estimators kf | ekf | eskf | spare ──> /drone/estimate/<name>/* ──> rviz2
 (reserved, not implemented) analysis ─────────────────────────────> /drone/analysis/* ─────────> rviz2
```

`rviz_drone_sim` simulates the whole flight at start-up (about 21 s for the mission), then, if the
parameter `wait_for_node` is set (the launch file sets it to `isaac_drone_render` when Isaac Sim is
started), waits until that node is in the graph (at most `wait_timeout_s`, 300 s) and publishes the
flight in real time. Isaac Sim follows `/drone/truth/pose` only; the flight time for the overlay and
the upcoming route comes from the header stamp minus `t0_ns` of `/drone/sim/info`.

### Topics

| Topic | Type | Frame id | Rate | QoS | Publisher |
|---|---|---|---|---|---|
| `/drone/truth/pose` | `geometry_msgs/PoseStamped` | `world` | 200 Hz | reliable, volatile, keep last 10 | `rviz_drone_sim` |
| `/tf` (`world` -> `base_link`) | `tf2_msgs/TFMessage` | `world` | 200 Hz | tf2 default (reliable, volatile, keep last 100) | `rviz_drone_sim` |
| `/drone/imu` | `sensor_msgs/Imu` (orientation not provided: `orientation_covariance[0] = -1`) | `base_link` | 200 Hz | reliable, volatile, keep last 50 | `rviz_drone_sim` |
| `/drone/position` | `geometry_msgs/PointStamped` (not published during dropout) | `world` | 10 Hz | reliable, volatile, keep last 10 | `rviz_drone_sim` |
| `/drone/truth/path` | `nav_msgs/Path` (truth, one pose per 0.1 s) | `world` | 2 Hz | reliable, volatile, keep last 1 | `rviz_drone_sim` |
| `/drone/marker` | `visualization_msgs/Marker`, ids 0-2 (body, arms, rotors), frame-locked | `base_link` | 3 messages/s | reliable, volatile, keep last 10 | `rviz_drone_sim` |
| `/drone/sim/info` | `std_msgs/String`, JSON: `trajectory`, `seed`, `duration_s`, `t0_ns`, `imu_rate_hz`, `pos_rate_hz` | - | once, before the first pose | reliable, transient local, keep last 1 | `rviz_drone_sim` |

Reserved for later steps (names, types and colours fixed now; nothing publishes them yet), with
`<name>` one of `kf`, `ekf`, `eskf`, `spare`:

| Topic | Type | Frame id | Rate | QoS | Content |
|---|---|---|---|---|---|
| `/drone/estimate/<name>/pose` | `geometry_msgs/PoseWithCovarianceStamped` | `world` | 200 Hz (every prediction) | reliable, volatile, keep last 50 | estimated pose and its 6x6 covariance |
| `/drone/estimate/<name>/path` | `nav_msgs/Path` | `world` | 2 Hz | reliable, volatile, keep last 1 | estimate trace |
| `/drone/estimate/<name>/covariance` | `visualization_msgs/Marker` (sphere scaled to the 3-sigma position ellipsoid) | `world` | 10 Hz | reliable, volatile, keep last 10 | position covariance |
| `/drone/analysis/peaks` | `visualization_msgs/MarkerArray` (one marker per detected error peak) | `world` | on detection | reliable, transient local, keep last 1 | peak markers |
| `/drone/analysis/outage` | `std_msgs/Bool` (true while no position measurement is valid) | - | 10 Hz | reliable, volatile, keep last 10 | outage indicator |
| `/drone/analysis/outage_marker` | `visualization_msgs/Marker` (text above the drone) | `base_link` | 10 Hz | reliable, volatile, keep last 10 | outage indicator in RViz |

### Colours (RGB 0-255, RViz and Isaac Sim)

| Item | Colour |
|---|---|
| truth path | green `25, 255, 0` |
| position measurement | magenta `204, 41, 204` |
| KF | blue `31, 119, 180` |
| EKF | orange `255, 127, 14` |
| ESKF | cyan `23, 190, 207` |
| spare | yellow `255, 215, 0` |
| covariance of a filter | the filter colour, alpha 0.25 |
| error peak markers | red `214, 39, 40` |
| outage indicator | grey `150, 150, 150` |

The RViz configuration contains displays for the reserved estimate paths with these colours.

### TF tree

`world` (ENU, fixed) -> `base_link` (FLU body = IMU frame, dynamic, 200 Hz, `rviz_drone_sim`).
No static transforms. The Isaac Sim stage uses the same ENU axes in metres (z up), so the pose of
`/drone/truth/pose` is applied to the drone prim without any conversion (`../isaac_render/usd_pose.py`).
Reserved child frames of `world` for the estimators: `base_link_kf`, `base_link_ekf`,
`base_link_eskf`, `base_link_spare` (not published yet).

### Time stamps

`t0 + t_k`, with `t0` the ROS time at the start of publication (also in `/drone/sim/info`). The
node simulates the whole flight at start-up before it starts publishing. The simulation itself
runs at 1 kHz; the published streams are sub-sampled from it. After the end of the flight the node
stops publishing and logs a message.

## Isolation from other ROS 2 work on this machine

`env.sh` sets, for the current shell: `ROS_DOMAIN_ID=101` and
`ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST`, so that the nodes of this package never discover nodes on
the default domain; `ROS_LOG_DIR` to `../rviz_drone_ws/log/ros` and `ROS_HOME` to
`../rviz_drone_ws/ros_home`, so that nothing is written to `~/.ros`; `RMW_IMPLEMENTATION=rmw_fastrtps_cpp`
and `FASTRTPS_DEFAULT_PROFILES_FILE=../rviz_drone_ws/fastdds_udp_localhost.xml` (UDPv4 on 127.0.0.1
only, built-in transports disabled). `../isaac_render/isaac_python.sh` sets the same for Isaac Sim
in its ROS 2 mode. The launch file starts RViz with `HOME` set to `../rviz_drone_ws/ros_home`, so
that RViz does not write `~/.rviz2`. Use the command-line tools with `--no-daemon` where they offer
it, and stop the daemon (domain 101) with `ros2 daemon stop` in the same environment at the end.

Rule: the ROS 2 processes of this repository are never started while any other ROS 2 process
(ROS 2 node or tool, RViz, Gazebo, Isaac Sim) is running on this machine; check first
(`ps -eo args | grep -E '/opt/ros/|ros2 |rviz2|gz sim|gazebo|isaacsim'`), and if one is running,
stop and report.

Accepted side effect (decision of the project owner, 2026-11-27): Fast DDS keeps lock and port files
in `/dev/shm` (`fastrtps_port*`, `fastrtps_*_el`, `sem.fastrtps_*`). They are transient run-time
state of DDS, not project files, and `/dev/shm` is shared with other ROS 2 work on this machine.
Measurements (listing of `/dev/shm` with names, sizes, mtimes and owners before and after each run):

| Run | Configuration | Changes in `/dev/shm` |
|---|---|---|
| 2026-11-27 16:11, node alone | UDP requested with `FASTDDS_BUILTIN_TRANSPORTS=UDPv4` | none |
| 2026-11-27 16:17, `ros2 topic hz` (6), `ros2 topic echo` (started a `ros2-daemon`) | same | 3 existing port files of earlier processes rewritten (`fastrtps_port7001`, `7005`, `7007`; same size, new mtime) |
| 2026-11-27 16:53, node and the same tools | XML profile, UDPv4 on 127.0.0.1, built-in transports off | 1 existing port file rewritten (`fastrtps_port7008`, same size, new mtime, at the start of the tools) |
| 2026-11-27 16:56, full launch (node, RViz, Isaac Sim; Isaac Sim stopped early by an error of the live mode, since fixed) | XML profile | new: `fastrtps_port32661`, `fastrtps_port32663`, `fastrtps_port7000` (52 400 bytes) and their `sem.*_mutex` files, created at the start of the launch; removed: the stale `fastrtps_port7002` and its mutex |
| 2026-11-27 17:00, full launch (node, RViz, Isaac Sim), 83 s flight | XML profile | new and left after the end: `fastrtps_port32665`, `fastrtps_port7002` (52 400 bytes each), a 549 408-byte segment `fastrtps_05a43ae57994a416`, three `*_el` lock files and two `sem.*_mutex` files, all created at 17:00:43 when the Isaac Sim node was created |

The UDP-only profile did not prevent the use of `/dev/shm`: the command-line tools, RViz and the
Isaac Sim bridge (its bundled Fast DDS) still create or reuse shared-memory port files and a
data-sharing segment, and files of processes that are stopped by a signal stay there. Which setting
would remove them completely was not determined (OPEN, not pursued after the decision; `fastdds shm
clean` would also remove the files of other work and is not used). All files are owned by this user
and are Fast DDS run-time state. The shared directories of the isolation
check, `~/.ros` and `/opt/ros/jazzy` had 0 differences in all runs.

## Usage

```bash
source simulation/rviz_drone/env.sh          # ROS 2 Jazzy, private domain, workspace overlay
simulation/rviz_drone/build.sh               # colcon build in simulation/rviz_drone_ws
source simulation/rviz_drone/env.sh          # again, to pick up the freshly built overlay
ros2 launch rviz_drone rviz_drone.launch.py              # simulator, Isaac Sim and RViz
ros2 launch rviz_drone rviz_drone.launch.py isaac:=false             # simulator and RViz
ros2 launch rviz_drone rviz_drone.launch.py isaac:=false rviz:=false # headless
ros2 launch rviz_drone rviz_drone.launch.py trajectory:=figure_eights   # the earlier trajectory
```

Tests (pure NumPy, no ROS needed): `.venv/bin/python -m pytest -q simulation/rviz_drone/test`.

Launch arguments: `isaac` (default `true`: Isaac Sim 6.0.1 window through
`../isaac_render/isaac_python.sh` with `ISAACSIM_ROS2=1`, `render_flight.py --source ros2 --gui`;
the node waits for the Isaac Sim node), `isaac_camera` (`chase`, `overview`, `topdown`), `rviz`
(default `true`), `trajectory`, `seed`.

Run of 2026-11-27 16:16 (`isaac:=false rviz:=false trajectory:=mission`, rates with `ros2 topic hz`
over 30 s, window 10 000 messages): the node simulated the 83 s mission in 20.5 s and published it
in 83.01 s of wall-clock time.

| Topic | Expected | Measured | Min / max period |
|---|---|---|---|
| `/drone/truth/pose` | 200 Hz | 200.001 Hz (5816 messages) | 3 / 7 ms |
| `/tf` | 200 Hz | 199.998 Hz | 3 / 7 ms |
| `/drone/imu` | 200 Hz | 200.000 Hz | 3 / 7 ms |
| `/drone/position` | 10 Hz | 10.000 Hz | 99 / 102 ms |
| `/drone/truth/path` | 2 Hz | 2.000 Hz | 499 / 501 ms |
| `/drone/marker` | 3 messages/s | 3.000 /s | 0 / 1001 ms (three ids at once) |

`/drone/sim/info` (latched) carried `{"duration_s": 83.0, "imu_rate_hz": 200.0, "pos_rate_hz": 10.0,
"seed": 0, "t0_ns": ..., "trajectory": "mission"}`; `ros2 node list` showed `/rviz_drone_sim` only.

Second run of 2026-11-27 16:53 with the UDP-only profile, same measurement: 200.002 Hz
(`/drone/truth/pose`), 200.006 Hz (`/tf`), 200.001 Hz (`/drone/imu`), 10.000 Hz, 2.000 Hz, 3.000 /s.

Full launch of 2026-11-27 17:00 (`ros2 launch rviz_drone rviz_drone.launch.py trajectory:=mission`:
node, Isaac Sim live mode, RViz): Isaac Sim was up and its node in the graph 34 s after the launch;
the node then published the 83 s flight (from 17:00:44 to 17:02:07). Rates with `ros2 topic hz`
over 30 s, started 3 s after the first message, while Isaac Sim and RViz were running:

| Topic | Expected | Measured | Min / max period |
|---|---|---|---|
| `/drone/truth/pose` | 200 Hz | 200.022 Hz | 2 / 7 ms |
| `/tf` | 200 Hz | 199.038 Hz | 1 / 153 ms |
| `/drone/imu` | 200 Hz | 200.006 Hz | 3 / 7 ms |
| `/drone/position` | 10 Hz | 10.000 Hz | 98 / 102 ms |
| `/drone/truth/path` | 2 Hz | 1.990 Hz | 499 / 640 ms |
| `/drone/marker` | 3 messages/s | 3.000 /s | 0 / 1001 ms |

Received by Isaac Sim (`/drone/truth/pose`, its own log, 10 s windows): 199.9 to 200.0 poses/s in
every full window of the flight. RViz: started, in the graph (`/rviz2` and its TF listener), ran
until the end without warnings or errors in its log; that the drone, TF, path and measurements are
visible in the widened view was not checked by looking at the window (manual check, OPEN). After
the end, `SIGINT` to the launch stopped all three processes cleanly and the `ros2` daemon of
domain 101 was stopped.

## Done criteria

1. The unit tests pass: hover equilibrium and energy consistency of the dynamics, accelerometer
   `+g` on body z at hover, reproducibility from a fixed seed, and trajectory tracking.
2. `colcon build` succeeds in `simulation/rviz_drone_ws`.
3. A 20 s headless run publishes `/drone/imu` and `/drone/truth/pose` at 200 Hz and
   `/drone/position` at 10 Hz within 5 % (measured with `ros2 topic hz`).
4. With RViz, the drone body, the TF frames, the truth path and the position measurements are
   visible (manual check; not part of the automated tests).
5. With `isaac:=true`, the drone in the Isaac Sim window follows `/drone/truth/pose`; Isaac Sim
   logs the ROS 2 libraries it loaded (bundled or `/opt/ros`), the received pose rate (200 Hz
   within 5 %) and the latency.

Status 2026-11-27: 1 to 3 met (rates in "Usage"); 5 met (Isaac Sim followed the pose at 200 Hz,
bundled ROS 2 libraries, see `../isaac_render/README.md`); 4 partly: RViz ran without errors, the
visual check in its window is OPEN.

## Files

| File | Purpose |
|---|---|
| `rviz_drone/quadrotor.py` | Parameters, rigid-body dynamics with rotor allocation and ground contact, controller. |
| `rviz_drone/reference.py` | Reference trajectories (`mission`, `figure_eights`) with analytic derivatives and yaw reference. |
| `rviz_drone/tracking.py` | Tracking table of the closed-loop flight. |
| `rviz_drone/flight.py` | Simulates the flight and generates the sensor streams with `synthetic/`. |
| `rviz_drone/repo_path.py` | Makes the repository root importable (for `synthetic/`). |
| `rviz_drone/recorder.py` | Writes one flight in the common recording format (`.npz`). |
| `rviz_drone/topics.py` | Topic names, types, rates, QoS, frames and colours (single source). |
| `rviz_drone/sim_node.py` | ROS 2 node that publishes the streams in real time. |
| `launch/rviz_drone.launch.py` | Starts the node and, optionally, Isaac Sim and RViz. |
| `rviz/rviz_drone.rviz` | RViz configuration. |
| `test/` | Unit tests. |
| `env.sh`, `build.sh` | Environment and build helpers. |
