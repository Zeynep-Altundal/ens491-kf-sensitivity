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
  compensation gives the desired force; its direction and a fixed yaw of 0 give the desired attitude
  (tilt limited to 45 deg); geometric attitude controller on SO(3) with angular-rate damping gives
  the torques, with lower gains on yaw. When the rotor thrusts would saturate, the yaw torque is
  reduced first, because the rotors have little yaw authority (at most 0.32 N m).
* **Parameters**: mass, inertia and arm are those of the vehicle rendered in Isaac Sim
  (`../isaac_render/`, NTNU ARL-Robot-1 asset), so that the visual and the dynamics describe the same
  vehicle: base link 1.2 kg with `diag(0.013, 0.014, 0.013) kg m^2` and four 0.01 kg propellers at
  `(+-0.1, +-0.1, 0) m` give `m = 1.24 kg`, `J = diag(0.013404, 0.014404, 0.013804) kg m^2`
  (parallel-axis theorem) and arm `0.1 sqrt(2) = 0.141421 m`. The rotor and drag parameters are
  illustrative, not taken from the asset: `f_max = 10 N` per rotor, `k_m = 0.016 m`,
  `c_d = 0.1 N s/m`. (Until 2026-11-27 the illustrative values `m = 1.5 kg`,
  `J = diag(0.029, 0.029, 0.055) kg m^2` and arm 0.25 m were used.)

## Trajectory (`reference.py`, closed form, analytic velocity and acceleration)

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

Closed-loop tracking of this trajectory with the default parameters (seed-independent; position
error `|p - p_ref|`, tilt and yaw from the simulated attitude; produced by
`python -m rviz_drone.tracking`). With the earlier illustrative parameters the figure-eight yaw was
2.9 deg and the aggressive errors 0.619 m (max) and 0.287 m (RMS); all other entries were equal.

| Segment | max `|a_ref|` [m/s^2] | max error [m] | RMS error [m] | max tilt [deg] | max yaw [deg] |
|---|---|---|---|---|---|
| take-off | 0.72 | < 0.001 | < 0.001 | 0.0 | 0.0 |
| hover (both) | 0 | 0.036 | 0.008 | 4.1 | 0.4 |
| figure-eight | 3.10 | 0.089 | 0.052 | 19.2 | 2.8 |
| aggressive | 7.53 | 0.611 | 0.284 | 45.5 (tilt limit) | 21.6 |
| landing | 0.46 | < 0.001 | < 0.001 | 0.0 | 0.0 |

In the aggressive segment the tilt limit is reached and the yaw is not held at 0 (low yaw
authority); this segment is meant to excite the filters, not to be tracked exactly.

## Sensors (from `synthetic/`)

The whole flight is simulated first. The ideal IMU is computed from the simulated state with
`synthetic.imu_model.ideal_imu` and corrupted with `synthetic.imu_model.corrupt_sensor`; position
measurements come from `synthetic.measurements.position_measurements`. Error parameters and seeds
are those of `synthetic.scenario.ScenarioConfig` (same random-stream split, `RNG_STREAMS`), so the
same seed always gives the same flight and the same measurements.

## Topics and rates

| Topic / TF | Type | Frame | Rate |
|---|---|---|---|
| `/drone/truth/pose` | `geometry_msgs/PoseStamped` | `world` | 200 Hz |
| TF `world` -> `base_link` | `tf2_msgs/TFMessage` on `/tf` | - | 200 Hz |
| `/drone/imu` | `sensor_msgs/Imu` (orientation not provided: `orientation_covariance[0] = -1`) | `base_link` | 200 Hz |
| `/drone/position` | `geometry_msgs/PointStamped` (not published during dropout) | `world` | 10 Hz |
| `/drone/truth/path` | `nav_msgs/Path` (truth, one pose per 0.1 s) | `world` | 2 Hz |
| `/drone/marker` | `visualization_msgs/Marker`, ids 0-2 (body, arms, rotors), frame-locked | `base_link` | 1 Hz per id (3 messages/s) |

Time stamps are `t0 + t_k`, with `t0` the ROS time at the start of publication. The node simulates the whole flight at start-up (about 11 s for 50 s of flight) before it starts
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
```

Tests (pure NumPy, no ROS needed): `.venv/bin/python -m pytest -q simulation/rviz_drone/test`.

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
| `rviz_drone/reference.py` | Waypoint trajectory with analytic derivatives. |
| `rviz_drone/flight.py` | Simulates the flight and generates the sensor streams with `synthetic/`. |
| `rviz_drone/repo_path.py` | Makes the repository root importable (for `synthetic/`). |
| `rviz_drone/sim_node.py` | ROS 2 node that publishes the streams in real time. |
| `launch/rviz_drone.launch.py` | Starts the node and, optionally, RViz. |
| `rviz/rviz_drone.rviz` | RViz configuration. |
| `test/` | Unit tests. |
| `env.sh`, `build.sh` | Environment and build helpers. |
