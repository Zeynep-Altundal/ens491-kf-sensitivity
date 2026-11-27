# Isaac Sim 6.0.1 as renderer for the quadrotor simulation

Goal: render the flight of `../rviz_drone/` (own dynamics, no Pegasus) with Isaac Sim 6.0.1, using
a separate pip installation in `~/isaacsim-6.0.1-ens491` that shares no per-user directory with the
existing Isaac Sim 6.0.1 installation in `~/isaacsim-venv`.

Status: **working** headless (PNG sequence, MP4, stills, heading check, inspection views) and with a
window (GUI replay). Isaac Sim only renders; the flight comes from the simulator of `../rviz_drone/`.

## Files

| File | Purpose |
|---|---|
| `env.sh` | Sourced; exports `ISAACSIM601_ROOT`, `ISAACSIM_PYTHON` and `RVIZ_DRONE_REPO_ROOT` for the current shell only. |
| `isaac_python.sh` | The isolation wrapper of `../isaac_drone/isaac_python.sh` (same environment variables and Kit settings), running the venv Python. |
| `fetch_assets.sh` | Downloads the assets below into `$ISAACSIM601_ROOT/assets` (outside the repository) and checks their sha256. |
| `smoke_headless.py` | Minimal headless start-up check (new stage, 120 updates, shut down); uses `omni.usd` and `pxr` only. |
| `usd_pose.py` | ENU/FLU/Hamilton to USD pose conversion of the body prim (numpy only). |
| `asset_alignment.py` | Fixed rotation of the drone asset relative to the body frame (identity) and the measured propeller positions it is checked against (numpy only). |
| `replay.py` | Camera paths, propeller angles, trail, flight phase and overlay text (numpy only). |
| `render_flight.py` | Scene, kinematic replay, PNG/MP4/stills, heading check, inspection views (headless) or real-time replay (`--gui`). |
| `tests/` | Tests of `usd_pose.py`, `asset_alignment.py` and `replay.py` (part of the repository test suite). |
| `frames/` | Output of `render_flight.py`; git-ignored. |

## Usage

```bash
source simulation/isaac_render/env.sh
bash simulation/isaac_render/fetch_assets.sh            # once
R=simulation/isaac_render/render_flight.py
# 15 s clip (PNG sequence + MP4) of the chase camera: straight leg, turn, start of the figure-eight
OMNI_KIT_ACCEPT_EULA=YES "$ISAACSIM_PYTHON" $R --start 13 --end 28 --camera chase --mp4
# stills TIME:CAMERA
OMNI_KIT_ACCEPT_EULA=YES "$ISAACSIM_PYTHON" $R --stills 3:chase,29.5:chase,20:overview,36:chase --rt-subframes 16
# heading check (pure +x flight and pure yaw, seen from above)
OMNI_KIT_ACCEPT_EULA=YES "$ISAACSIM_PYTHON" $R --verify-heading
# inspection views: hover, +x translation, +90 deg yaw and t = 36 s, with the expected geometry drawn
# (the results below were recorded with the figure_eights trajectory)
OMNI_KIT_ACCEPT_EULA=YES "$ISAACSIM_PYTHON" $R --inspect --trajectory figure_eights
# window: real-time replay in a loop; close the window to exit
OMNI_KIT_ACCEPT_EULA=YES "$ISAACSIM_PYTHON" $R --gui --camera chase
# estimator replay of one results run (section "Estimator replay"): stills (camera "auto" follows the
# camera rule), a 10 s clip, the window
OMNI_KIT_ACCEPT_EULA=YES "$ISAACSIM_PYTHON" $R --results outage_40s_calm --seed 0 --renderer realtime --stills 7:auto,11:auto
OMNI_KIT_ACCEPT_EULA=YES "$ISAACSIM_PYTHON" $R --results outage_40s_calm --seed 0 --renderer realtime --start 5 --end 15 --mp4
OMNI_KIT_ACCEPT_EULA=YES "$ISAACSIM_PYTHON" $R --results outage_40s_calm --seed 0 --gui
```

| Option | Values (default first) |
|---|---|
| `--trajectory` | `mission` (default), `figure_eights` (see `../rviz_drone/README.md`) |
| `--camera` | `chase` (follows position and yaw), `overview` (fixed), `topdown` (fixed, north up); chosen once per run; with `--results` and `chase`, the camera rule switches to `overview` and back |
| `--results` | scenario name or `nominal` (estimator replay; output in `frames/results_<scenario>_seed<k>/`) |
| `--env` | `field` (Isaac flat terrain with grass), `terrain` (Isaac rough terrain with grass), `grid` (plain checkerboard, fallback) |
| `--sky` | `hdri` (sky image), `plain` (uniform dome) |
| `--renderer` (headless) | `pathtraced` (`--spp 64`), `realtime` (RTX real-time, FXAA, `--supersample 2`) |
| other | `--start`, `--end`, `--fps 30`, `--mp4`, `--width 1280`, `--height 720`, `--rt-subframes 8`, `--sun 3000`, `--sky-intensity 400`, `--sun-elevation 47.9`, `--sun-azimuth 146.5`, `--no-overlay`, `--seed 0`, `--out` |

`OMNI_KIT_ACCEPT_EULA=YES` states acceptance of the NVIDIA licence of Isaac Sim and is given per
command only.

## External assets

All external assets in use, from the Isaac Sim 6.0 assets
(`https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/` + source path),
downloaded by `fetch_assets.sh` into `$ISAACSIM601_ROOT/assets` (outside the repository; not
redistributed here):

| Use | Source path | Files | Licence |
|---|---|---|---|
| Drone | `Isaac/Robots/NTNU/ARL-Robot-1/` | `arl_robot_1.usd`, `configuration/arl_robot_1_{base,physics,robot,sensor}.usd`, `LICENSE` (22 MB) | BSD 3-Clause, copyright 2025 NTNU Autonomous Robots Lab (`LICENSE` in the folder) |
| Sky | `NVIDIA/Assets/Skies/Cloudy/` | `kloofendal_48d_partly_cloudy_4k.hdr` (25 MB) | CC0 (Poly Haven). The file is byte-identical to the 4k HDR "Kloofendal 48d Partly Cloudy" of Poly Haven (`https://polyhaven.com/a/kloofendal_48d_partly_cloudy`): same size, 25 820 705 bytes, and same MD5, `afb5e5f0c035d24567e8d778dd35c752`, as listed by `https://api.polyhaven.com/files/kloofendal_48d_partly_cloudy` (checked 2026-11-27). Poly Haven licenses all its assets as CC0 (`https://polyhaven.com/license`); no attribution required. |
| Terrain, `--env field` (default) | `Isaac/Environments/Terrains/` | `flat_plane.usd`, prim `/World/GroundPlane` (12 kB) | no licence file in the folder: NVIDIA Omniverse License Agreement |
| Terrain, `--env terrain` | `Isaac/Environments/Terrains/` | `rough_plane.usd`, prim `/World/mesh` (171 MB) | no licence file in the folder: NVIDIA Omniverse License Agreement |
| Ground material (both terrains) | `NVIDIA/Materials/Base/Natural/` | `Grass_Countryside.mdl`, `Grass_Countryside/Grass_Countryside_{BaseColor,H,Normal,ORM}.png` (15 MB) | no licence file in the folder: NVIDIA Omniverse License Agreement |

`--env grid` uses no external asset. The NVIDIA Omniverse License Agreement is the licence shipped
with the installation (`isaacsim/LICENSE.txt`): "Content" (2D/3D assets) may be used and modified,
and user-generated content such as video, stills and screen captures may be distributed; Content
may not be distributed on a stand-alone basis.

Drone candidates that were considered and are not used: `Isaac/Robots/Bitcraze/Crazyflie/cf2x.usd`
(MIT, copyright 2022 Bitcraze; a 28 g vehicle with rotors at +-0.031 m) and
`Isaac/Robots/IsaacSim/Quadcopter/quadcopter.usd` (CC BY 4.0; primitive shapes). ARL-Robot-1 is a
detailed model (frame, protective cage, sensors, four propeller prims) of a 1.24 kg vehicle, close
to the simulated one. Not considered: `Isaac/Environments/Outdoor/Rivermark` (26 925 files,
11.7 GB; loading it over the network would fill a Kit cache whose location under the isolation
settings was not checked).

## Vehicle: visual and dynamics describe the same vehicle

The simulator parameters were set to the asset (not the asset scaled to the model), see
`../rviz_drone/README.md`: mass 1.24 kg (base link 1.2 kg + four 0.01 kg propellers), inertia
`diag(0.013404, 0.014404, 0.013804) kg m^2` (asset inertias with the parallel-axis theorem), arm
`0.141421 m` (rotors at `(+-0.1, +-0.1, 0) m`). At start-up `render_flight.py` prints the rotor
centres of the asset next to `QuadParams.rotor_positions()` and stops if they differ by more than
1 mm (measured: 2.5e-07 m). The asset is rendered at its real scale (metres, no scaling).

## Asset alignment and forward axis

Prim structure: `/World/Quadrotor` (body pose from `usd_pose.py`) / `alignment` (fixed
`xformOp:orient` = `asset_alignment.ASSET_TO_BODY`) / `asset` (reference to the asset). The alignment
is a separate constant, not part of the pose conversion; it is the identity, because

* the propeller prims of the asset are named and placed as the FLU body frame expects:
  `front_left` at (0.1, 0.1, 0), `front_right` at (0.1, -0.1, 0), `back_left` at (-0.1, 0.1, 0),
  `back_right` at (-0.1, -0.1, 0) m, i.e. front = +x, left = +y, all in the plane z = 0;
* the propellers spin about their z axis, which is +z of the asset (thrust up);
* the asset stage has `upAxis = Z`, `metersPerUnit = 1`.

Tests (`tests/test_isaac_render_alignment.py`): the constant is a unit quaternion and a proper
rotation; the aligned measured propeller positions equal `QuadParams.rotor_positions()`; the
propeller names match their positions; the spin axis maps to body +z; the pose conversion has no
asset parameter. At every start `render_flight.py` repeats the check on the composed stage
(propeller centres and spin axes in the body frame; maximum deviation 2.5e-07).

Two changes to the asset are made in the render stage only (the asset file is unchanged):

* Instancing is turned off on the six instanceable visual prims. With instancing the rendered
  asset did not follow a change of the body pose until a later change inside the asset: the
  inspection views showed the pose of the previous view, while the nose marker (not part of the
  asset) was at the right place. This affected the renders made before 2026-11-27 11:46 (the
  earlier clip, stills and GUI replay, where it showed as a drone at a wrong attitude); they were
  rendered again.
* The physics description (12 applied Physics/Physx schemas, 4 joints, collision prims) is removed,
  because the replay is kinematic. Removing it alone did not change the delayed visuals.

Inspection (`--inspect`, trajectory `figure_eights`, 2026-11-27 11:46; camera 24 mm; the expected rotor quadrilateral (cyan) and
body z axis (yellow) from the simulator pose are drawn over each image; "image angle" is the major
axis of the drone silhouette in views where the rotor plane is seen edge-on):

| View | Simulator | USD (propeller prims, nose prim) | Image |
|---|---|---|---|
| hover, side | tilt 0.0 deg | rotor-axis tilt 0.0 deg, rotors up | rotor plane at -1.5 deg (cage not symmetric), level |
| +x translation x = -0.8, 0, +0.8 m, side (camera at -y) | tilt 0.0 deg | tilt 0.0 deg, rotors up, rotor centres within 2.5e-07 m | drone and nose arrow move to the right (east); rotor plane -1.5, -1.5, -3.3 deg |
| +90 deg yaw, oblique and top-down | yaw 90.0 deg | nose yaw 90.0 deg | nose arrow points north (along the green axis) |
| t = 36 s, side (along the tilt axis) | tilt 43.2 deg, yaw -21.5 deg | rotor-axis tilt 43.2 deg, nose yaw -21.5 deg | rotor plane at 42.6 deg |
| t = 36 s, chase camera | tilt 43.2 deg | 43.2 deg | the asset coincides with the drawn rotor quadrilateral |

Simulator tilt at t = 36 s: 43.2 deg (body z = (-0.618, -0.294, 0.729)); maximum in the aggressive
segment 45.5 deg at t = 33.66 s. In the chase view at t = 36 s the rotor plane is turned towards
the camera and the vertical hoop of the cage faces it, so the drone looks steeper than 43 deg; the
side view shows the actual tilt.

Heading check (`--verify-heading`, after the fix): maximum heading error 0.53 deg over 19 poses;
the +x flight moves to the right (east) in the top-down image.

## Scene

* Stage `upAxis = Z`, `metersPerUnit = 1`. Ground: the terrain prim of the asset, its material
  replaced by the grass material, projected in world space (`project_uvw`, `world_or_object`,
  `texture_scale 0.25`), because the meshes have no texture coordinates.
* Sky: dome light with the HDRI (intensity 400). Sun: distant light (intensity 3000, apparent size
  2.5 deg for soft shadow edges) turned to the sun of the sky image: elevation 47.9 deg from the
  brightest pixel of the image, azimuth 146.5 deg (north-west) matched to the shadow of the sky
  light, so that both lights give one shadow. (`shadow:enable = false` on the dome light had no
  visible effect; before the alignment the sky and the distant light gave two shadows.)
* Fill light: a distant light without shadow (`--fill`, intensity 800, apparent size 10 deg),
  turned every frame to shine along the horizontal view direction of the active camera, 30 deg
  downwards (`replay.fill_direction`), so that the side of the drone seen by the camera is lit when
  the drone is between the camera and the sun (before, it was rendered as a dark silhouette).
* No motion blur (`/rtx/post/motionblur/enabled = false`), fixed exposure (auto exposure off).
* Propellers: each propeller prim is turned about its z axis in the direction opposite to its
  reaction torque in the allocation matrix (front-right and rear-left counter-clockwise seen from
  above, the other two clockwise); displayed speed `4 sqrt(f / f_hover)` rev/s, at most 7 rev/s, so
  that it stays below half the frame rate at 30 frames/s (not the physical speed).
* Front marker: red arrow along body +x (`/World/Quadrotor/nose`).
* Path display: only the trail behind the drone and the upcoming route. Trail: the last 2 s of the
  flown path with a self-lit orange material (appears yellow-orange), width from 14 mm at the drone
  to 2 mm at its end. Upcoming route: the reference route of the next 8 s as a 5 mm pale line
  (slightly self-lit, opacity 0.6), visible in the chase view but thinner and paler than the trail. For the fixed cameras, which see the whole area from tens of metres, both
  widths are multiplied by 6.
* Cameras (one per run, `--camera`; no automatic switching):
  * `chase` follows the position and the yaw of the drone only, never its roll or pitch: it is
    2.6 m behind and 0.45 m above the low-pass filtered position (0.35 s), along the low-pass
    filtered yaw (0.8 s, yaw from the attitude quaternion); target 0.25 m above the low-pass
    filtered position (0.12 s), so that the line of sight is nearly level and the horizon and sky
    are in the upper part of the frame; up vector world z; pushed back if the drone comes closer
    than 2 m. Mission: 2.60 m to 4.77 m from the drone (median 3.03 m); line of sight 3.9 deg below
    level (median; 5th to 95th percentile 6.8 to 0.9 deg below), horizon in the frame (half
    vertical field of view 13.8 deg) for 97.8 % of the flight.
  * `overview` and `topdown` are computed from the extent of the flight: overview from the side of
    the sun, 35 deg above the horizon; top-down above the centre, north up, east to the right.
* Overlay: time, flight phase, speed and camera, drawn on each frame (headless) or as labels on a
  dark box in the viewport, 64 px below its top edge so that the viewport toolbar does not cover it
  (GUI).

## ROS 2 live mode (`--source ros2`)

The offline mode (default, `--source replay`) renders the flight generated in memory. In the live
mode (`--source ros2 --gui`, started by `../rviz_drone/launch/rviz_drone.launch.py`), the drone
follows `/drone/truth/pose` of the simulator node `rviz_drone_sim`: the Isaac Sim node
`isaac_drone_render` (rclpy in a background thread) receives the poses; each frame applies the
latest pose to the body prim (same conversion as offline), steps the online chase camera
(`replay.ChaseFilter`, equal to `replay.chase_path` within 1e-9 m, tested), draws the trail from the
received poses and the upcoming route from the reference trajectory at the flight time
`stamp - t0_ns` of the latched `/drone/sim/info`. The propellers turn at the hover display speed
(the pose carries no rotor thrust). Every 10 s it logs the frame rate, the received pose rate and
the latency (receive time minus header stamp).

ROS 2 libraries: `isaac_python.sh` with `ISAACSIM_ROS2=1` sets `ROS_DISTRO=jazzy` and
`LD_LIBRARY_PATH` to the Jazzy libraries bundled with the extension `isaacsim.ros2.core` 1.9.4
(`isaacsim.ros2.bridge` 5.1.2; bundled: Fast DDS 2.14.6, `rmw_fastrtps_cpp`, `rmw_cyclonedds_cpp`,
rclpy for Python 3.12), removes `/opt/ros` from `PATH` and does not source any system ROS 2 setup
file; the system installation is not changed. With `ROS_DISTRO` set and no ROS 2 sourced, the
bridge loads its rclpy as `jazzy.rclpy` (code of `isaacsim.ros2.core`, `impl/extension.py`). At
start-up the live mode prints every ROS 2 / DDS library mapped into the process (from
`/proc/self/maps`) and how many come from `/opt/ros`.

Run of 2026-11-27 17:00 (full launch, mission, chase camera, window): the libraries mapped into
Isaac Sim were all bundled ones, none from `/opt/ros` (log: "libraries from /opt/ros: 0; bundled:
6"): `librcl.so`, `librcutils.so`, `librmw_implementation.so`, `librmw_fastrtps_cpp.so`,
`libfastrtps.so.2.14`, `libfastcdr.so.2`, all from
`isaacsim.ros2.core/jazzy/lib`; rclpy from `isaacsim.ros2.core/jazzy/rclpy`. 10 s windows of the
flight:

| Flight time [s] | Frames/s | Frame time median / max [ms] | Poses received /s | Receive time minus stamp, median / max [ms] |
|---|---|---|---|---|
| 0-9.3 (with the start) | 102.3 | 9.7 / 80.1 | 185.6 | 5.9 / 153.5 |
| 9.3-19.3 | 97.2 | 10.0 / 69.6 | 200.0 | 145.9 / 152.8 |
| 19.3-29.3 | 97.2 | 10.1 / 15.1 | 199.9 | 145.9 / 149.3 |
| 29.3-39.3 | 95.6 | 10.2 / 43.1 | 200.0 | 145.9 / 149.4 |
| 39.3-49.3 | 95.8 | 10.3 / 16.5 | 200.0 | 145.9 / 149.2 |
| 49.3-59.3 | 109.2 | 8.9 / 34.6 | 200.0 | 145.8 / 148.6 |
| 59.3-69.3 | 111.5 | 8.8 / 14.6 | 200.0 | 145.8 / 149.4 |
| 69.3-79.3 | 110.9 | 8.9 / 13.5 | 199.9 | 145.8 / 148.2 |

The receive time minus the header stamp is not the transport latency: the stamp is `t0 + t_k`
(flight time), and the simulator node fell about 0.15 s behind its own time base early in the run
(the flight of 83 s took 83.15 s from `t0`; ROS 2 timers do not catch up missed periods) and then
published at 200 Hz; the constant 146 ms is this offset. The transport latency itself was not
measured separately (OPEN). At shutdown the spin thread raised `ExternalShutdownException` (the
bridge closes the rclpy context first); it is caught since then (not yet run again).

## Estimator replay (`--results SCENARIO`; protocol)

Step 6 of 8, visual half. Written on 2026-11-27 before the mode was implemented; changes after a
run are listed at the end of this section. The replay is offline: it reads one results run
(scenario, measurement seed `--seed`) of `experiments/estimators/` and shows the drone at the true
pose. Isaac Sim only renders. The same elements are published on the reserved ROS 2 topics by
`../rviz_drone/rviz_drone/results_node.py`, so RViz shows them too. The visual model (trails,
ellipsoids, peaks, corruption windows, camera rule, overlay text) is one numpy module used by both,
`../rviz_drone/rviz_drone/estimate_view.py`; colours come from `../rviz_drone/rviz_drone/topics.py`.

| Element | Encodes | Colour (`topics.COLOURS`) | Source file |
|---|---|---|---|
| Drone (asset) | true pose | asset | `simulation/recordings/mission_seed0.npz`: `truth_p`, `truth_q`, `sim_rotor_thrust` (propellers) |
| Truth trail | true position over the last 5 s | `truth` | same |
| Estimator trails | `p_hat` of `KF`, `EKF`, `ESKF` over the last 5 s | `kf`, `ekf`, `eskf`; slot `spare` empty | `experiments/estimators/results/runs/<scenario>/<EST>_seed<k>.npz`: `p_hat` |
| Estimate marker | small sphere at the current `p_hat` | same as the trail | same |
| Covariance ellipsoid | 3-sigma ellipsoid of the position covariance at the current estimate (semi-axes `3 sqrt(eig(P_pos))`), semi-transparent | estimator colour, alpha `COVARIANCE_ALPHA` (0.25) in RViz; tinted glass in Isaac Sim (change 2) | same file: `P_pos` (position block, added to the run files for this mode) |
| Peak markers | one pin per peak of the peak rule, at the estimate position at the time the rule fired (peak start), shown from that time on | pin `peak`, head in the estimator colour | `experiments/peak_analysis/results/peaks.csv` (`t_start`, `t_peak`) |
| Corruption indicator | banner naming each active corruption (position outage, bias step or ramp, IMU noise, position noise burst, position spike; a spike is shown for 0.5 s) and a tint of the image while a window is active | tint `outage` for outages, `corruption` for the others | `simulation/recordings/mission_seed0__<scenario>.npz`: `events` |
| Overlay | time, mission phase, position error `||p_hat - p_true||` of each estimator [m], camera, legend | text white; legend swatches in the element colours | run files and the recording |
| Upcoming route | reference route of the next 8 s | pale (unchanged) | recording: `sim_p_ref` |

Camera: chase camera by default. It sees about +-1.5 m around the drone at its 2.6 m distance;
when an estimate is more than 5 m from the true position, the view switches to the overview camera
and the overlay says so ("camera: overview (EKF estimate 12.3 m from the drone > 5 m)"); it switches
back when every estimate has been within 4 m for 2 s. In this mode the overview camera frames the
truth and all estimates of the run, so a drifting estimate stays in view.

Scenarios for the clips (measurement seed 0). The request named four categories; the outage
reversal is a pair, so there are five scenarios:

| Scenario | Reason | Onset [s] | 10 s clip [s] | Stills [s] |
|---|---|---|---|---|
| `nominal` | baseline: all estimates near the truth, KF visibly noisier | - | 47-57 (start of the aggressive segment) | 10, 30, 52, 70 |
| `outage_40s_calm` | outage reversal, calm side: the EKF/ESKF drift beyond the KF | 8.0 | 5-15 | onset - 1, onset + 3, EKF peak time, outage end + 3 |
| `outage_40s_aggressive` | outage reversal, aggressive side: the KF drifts beyond the EKF/ESKF | 42.0 | 39-49 | same rule |
| `gyro_bias_step_small_onset_aggressive` | bias case: peak in EKF/ESKF only, not recovered, KF unaffected | 52.0 | 49-59 | same rule |
| `pos_spike_single` | spike case: the KF peak is higher and shorter than the EKF/ESKF one | 12.0 | 9-19 | same rule |

"EKF peak time": `t_peak` of the largest EKF peak of the run in `peaks.csv`.

Changes after the protocol (before any render): (1) 2026-11-27, the still times computed by the
rule coincide in two scenarios (`outage_40s_aggressive`: EKF peak 82.0 s and outage end + 3 s past
the flight end; `pos_spike_single`: onset + 3 s = 15.0 s and spike end + 3 s = 15.1 s). The stills
used: `outage_40s_calm` 7, 11, 48.0 (EKF peak), 51 s; `outage_40s_aggressive` 41, 45, 62 (middle of
the outage), 82.0 s (EKF peak, outage end); `gyro_bias_step_small_onset_aggressive` 51, 55, 62.6
(EKF peak), 65 s; `pos_spike_single` 11, 12.1 (spike, EKF peak), 13 (onset + 1 s), 15 s. (2) 2026-11-27,
after test stills of `pos_spike_single`: the real-time renderer drew `UsdPreviewSurface` with
opacity 0.25 as opaque (also with `/rtx/raytracing/fractionalCutoutOpacity`), so the KF shell hid the
drone; the covariance shells use a thin-walled glass (`OmniGlass`, index 1.0, no refraction) tinted
with the estimator colour (weight 0.15; 0.25 darkened the view), translucent in both renderers.
(3) The estimator trails passed next to the chase camera (it flies along the path just flown) and
filled the image; in the chase view every trail fades out within 1.5 m of the camera (width 0
within 0.8 m). (4) EKF and ESKF estimates differ by millimetres, so the EKF line was hidden under
the ESKF line; the EKF trail (18 mm) and marker (70 mm) are drawn wider than the ESKF ones (8 mm,
45 mm), so the EKF colour shows as a border. (5) Clips and stills use the real-time renderer (as
the window), not path tracing (2.8 s per frame in earlier clips). (6) During outages the KF shell grows to
hundreds of metres (3-sigma radius 270 m at 62 s of `outage_40s_aggressive`); casting a shadow, it
darkened the whole scene (still at 62 s: mean brightness 10 of 255). The shells cast no shadow and
are invisible to reflections (`primvars:doNotCastShadows`, `primvars:invisibleToSecondaryRays`).
(7) A drifting estimate can go below the ground (KF at 62 s: z = -9.4 m), where the ground hides
it; the overlay then adds "below ground, z = ... m" to that estimator's line (the estimate itself is
drawn where it is). Renders: four stills
and one 10 s clip (30 frames/s) per scenario; no full-length render yet. The GUI is started once
on `outage_40s_calm` (all elements appear: long corruption window, growing ellipsoids, peaks of all
three estimators, the camera switch).

### Estimator replay: runs of 2026-11-27

All through `isaac_python.sh`, real-time renderer, 1280x720, seed 0; outputs in
`frames/results_<scenario>_seed0/` (git-ignored): `stills/` (four PNGs) and `clip_<start>_<end>/`
(301 PNGs, 30 frames/s) with `clip_<start>_<end>.mp4` (55-62 MB each). Every run exited with 0, and
the comparison after each run (five shared directories, screenshots metadata, `~/isaacsim-venv`,
`~/.ros`, `/opt/ros/jazzy`) gave 0 differences.

| Scenario | Stills [s] (camera) | Clip [s] | Clip run, start to exit | Camera switches in the clip |
|---|---|---|---|---|
| `nominal` | 10, 30, 52, 70 (chase) | 47-57 | 8 min 22 s | 0 |
| `outage_40s_calm` | 7 (chase), 11, 48 (overview), 51 (chase) | 5-15 | 10 min 14 s | 1 |
| `outage_40s_aggressive` | 41 (chase), 45, 62, 82 (overview) | 39-49 | 9 min 54 s | 1 |
| `gyro_bias_step_small_onset_aggressive` | 51, 55, 62.6, 65 (chase) | 49-59 | 8 min 15 s | 0 |
| `pos_spike_single` | 11 (chase), 12.1, 13 (overview), 15 (chase) | 9-19 | 8 min 7 s | 2 |

Overview share of the flight under the camera rule (whole run): `outage_40s_calm` 0.484,
`outage_40s_aggressive` 0.468, `pos_spike_single` 0.024, `nominal` 0.

Observed in the renders: the EKF colour (orange in `topics.py`) is rendered close to yellow (the
self-lit material with lighting), near the spare colour; not corrected (OPEN). Real-time images
show grain on the ground and the sky (no denoiser at 8 subframes).

ROS 2 check of `results_node` (2026-11-27 19:20, `results:=outage_40s_calm`, no RViz, no Isaac
Sim, 20 s of `ros2 topic hz`): truth pose 196.1 Hz, `/tf` 199.8 Hz, estimate poses 196.1-199.7 Hz,
EKF covariance marker 9.8 Hz, ESKF path 1.96 Hz, outage flag and corruption banner 9.8 Hz; samples:
outage flag `true`, banner "CORRUPTION: position outage (8.0-48.0 s)", the peaks array with the pins
of the three estimators. No process was left after the shutdown, the ROS 2 daemon was stopped; 0
differences in the compared directories; `/dev/shm`: Fast DDS port and lock files of this user
replaced (accepted transient state). RViz was not started; its view of these topics is OPEN.

GUI (2026-11-27 20:57, `--results outage_40s_calm --gui`, left open): 59-77 frames/s, median frame
time 12.9-15.7 ms, longest frame 509 ms at start-up; one camera switch in the first 40 s. Comparison with
the window open: 0 differences; `/dev/shm` holds two `carb-RStringInternals-<pid>` entries of the
running Kit process.

## GUI replay timing

The replay time is the wall-clock time since the start, modulo the flight duration; each frame
shows the truth sample nearest to it (200 Hz). The script logs every 10 s the frame rate, the frame
times and the largest difference between the replay step and the wall-clock step. Measured on
2026-11-27 (mission, chase camera, 1280x720 window), first six 10 s windows: 92.9 to 102.6 frames/s,
median frame time 9.1-10.1 ms. Occasional long frames: 506 ms in the first window (start-up), then
181, 18.4, 223, 73 and 71 ms as the longest frame of the following windows. Short check
(2026-11-27): from the sixth to the eleventh window the longest frame was 71.2 to 74.7 ms in every
window, which suggests a periodic task, but the log has no per-frame time stamps to match against
its 10 warning and error lines; the cause is still OPEN. Replay step minus wall-clock step: 0.000 ms in all windows (no drift and no
jumps in time; after a long frame the drone moves on by the elapsed time, which shows as a jump in
position).

## Pose conversion (ENU/FLU/Hamilton to USD)

* World: ENU on a Z-up, metre stage is the stage frame itself: `p_usd = p_enu`.
* Attitude: the simulator quaternion `[w, x, y, z]` (Hamilton, body to world, FLU body) is written
  unchanged (normalised) to `xformOp:orient` (`GfQuatd`, Hamilton, real part first) of the body
  prim, after `xformOp:translate`. USD uses row vectors: `[p_w, 1] = [p_b, 1] M`,
  `M = [[R(q)^T, 0], [p, 1]]`. The asset alignment is separate (see above).
* Run-time check: the local-to-world matrix computed by USD is compared with the expected matrix for
  200 samples of the flight; tolerance 1e-7. Measured maximum difference: 1.392e-08 with the earlier
  vehicle parameters, 1.311e-08 with the current ones (double-precision quaternion-to-matrix
  conversion of Gf).
* Heading check (`--verify-heading`, grid ground, plain sky, camera 6 m above, north up): each pose
  is rendered with and without the nose arrow; the arrow pixels (red and changed) give the rendered
  heading from the projected body origin. Pure +x flight from x = -1.5 m to +1.5 m at zero attitude
  and pure yaw 0 to 330 deg in 30 deg steps: maximum heading error 0.57 deg before and 0.53 deg after
  the instancing fix (19 poses); the drone moves to the right (east). This check measures the nose
  marker, which is not part of the asset; the asset itself is checked by `--inspect`.
* Tests: `tests/test_isaac_render_pose.py` (6), `tests/test_isaac_render_alignment.py` (5) and
  `tests/test_isaac_render_replay.py` (10).

## Rendering notes (found while building the scene)

* Instanced asset visuals did not follow a change of the body pose (see "Asset alignment"); instancing
  is turned off in the render stage.
* A render product created after another one was destroyed returned images of an earlier state;
  one render product per camera is created and kept, with 5 warm-up steps.
* Temporal anti-aliasing showed a ghost of the drone at an earlier position; the headless renders
  use path tracing (64 samples per pixel per frame, accumulation restarts at every change) or, with
  `--renderer realtime`, FXAA on a 2x supersampled image.
* The first path-traced image of a new render product stayed black; each camera is primed with one
  discarded image.
* Every launch logs `No space left on device for creating change watches ...
  fs.inotify.max_user_watches`; the system setting is not changed, and the message had no visible
  effect.

## Installation

| Item | Value |
|---|---|
| venv | `~/isaacsim-6.0.1-ens491/venv`, created with `/usr/bin/python3.12 -m venv` (Python 3.12.3) |
| Packages | `isaacsim[all,extscache]==6.0.1` from `https://pypi.nvidia.com` (extra index), installed in two steps: `isaacsim[all]` (exit 0, 7 min 49 s), then `isaacsim[all,extscache]` through `isaac_python.sh` (inherited `PYTHONPATH` removed; exit 0, about 14 min; adds `isaacsim-extscache-kit`, `-kit-sdk`, `-physics` 6.0.1.0) |
| Size | venv 24 GB; assets 225 MB |
| Side effects | none: 0 differences after each step in the five shared directories, the metadata of `~/Documents/Kit/shared/screenshots`, the listing of `~/isaacsim-venv`, `~/.ros` and `/opt/ros/jazzy` |

`isaacsim[all]` alone does not start: `isaacsim.exp.base` depends on `isaacsim.anim.robot.schema`,
which is only in the extension-cache packages (first launch, exit code 55).
`isaacsim.core.api` is in `extsDeprecated` in 6.0.1, so the scripts here do not use it.

## Launches

All launches go through `isaac_python.sh`; after every launch (and with the window open) the
comparison of the five shared directories, the screenshots metadata, `~/isaacsim-venv`, `~/.ros`
and `/opt/ros/jazzy` gave 0 differences.

| Date | Launch | Result |
|---|---|---|
| 2026-11-27 02:57 | `smoke_headless.py`, `isaacsim[all]` only | exit 55 after 10.6 s (missing `isaacsim.anim.robot.schema`) |
| 2026-11-27 09:24 | `smoke_headless.py` | exit 0; Kit 110.1.2, app 6.0.1; every path token inside `_sandbox` |
| 2026-11-27 09:25-09:30 | first scene (primitive quadcopter, grid) | 501 frames; GUI replay |
| 2026-11-27 09:58-10:31 | scene development (stills), heading check | exit 0 each; results above |
| 2026-11-27 10:31-10:48 | four stills, 10 s clip, GUI replay | exit 0; affected by the instancing delay, replaced by the renders below |
| 2026-11-27 11:41-11:47 | `--inspect` (four runs while finding the delay), `--verify-heading` | exit 0; results in "Asset alignment and forward axis" |
| 2026-11-27 11:50 | four stills (t = 3, 29.5, 36 s chase; t = 20 s overview), path traced, `--rt-subframes 16` | exit 0; `frames/stills/` |
| 2026-11-27 11:50-12:05 | 10 s clip t = 22-32 s, chase, 30 frames/s, path traced | exit 0; 301 frames in 836 s (2.8 s per frame), `frames/seq_chase_22.00_32.00.mp4` (H.264, 1280x720, 13.4 MB) |
| 2026-11-27 12:05 | `--gui --camera chase` | window open, overlay below the viewport toolbar, real-time replay (RTX real-time renderer of the app) |
| 2026-11-27 12:19-12:22 | mission stills (development) | exit 0 |
| 2026-11-27 12:25-12:47 | 15 s clip of the mission, t = 13-28 s (straight leg, turn, hover with turn, start of the figure-eight), chase, 30 frames/s, path traced | exit 0; 451 frames in 1257 s, `frames/seq_chase_13.00_28.00.mp4` (H.264, 1280x720, 8.6 MB) |
| 2026-11-27 12:47 | `--gui --camera chase`, mission | window open; frame rate below |
| 2026-11-27 15:57 | seven chase stills (t = 3, 12, 20, 30, 45, 55, 70 s) and one overview still (t = 30 s) after the camera pitch, fill light and route line changes | exit 0; `frames/stills/`; horizon and sky in all chase stills, drone lit in all |
| 2026-11-27 16:25 | stills t = 30 s chase, t = 55 s top-down, after the live-mode changes (offline mode unchanged) | exit 0 |
| 2026-11-27 16:56 | full launch, live mode | Isaac Sim started; the live mode stopped with a Python error at the first pose (trail of a one-sample history), fixed (`replay.trail_from_history`) |
| 2026-11-27 17:00 | full launch, live mode, 83 s flight | exit 0 ("finished cleanly"); results in "ROS 2 live mode" |
