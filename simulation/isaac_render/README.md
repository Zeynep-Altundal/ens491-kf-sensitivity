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
```

| Option | Values (default first) |
|---|---|
| `--trajectory` | `mission` (default), `figure_eights` (see `../rviz_drone/README.md`) |
| `--camera` | `chase` (follows position and yaw), `overview` (fixed), `topdown` (fixed, north up); chosen once per run, never switched automatically |
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
* No motion blur (`/rtx/post/motionblur/enabled = false`), fixed exposure (auto exposure off).
* Propellers: each propeller prim is turned about its z axis in the direction opposite to its
  reaction torque in the allocation matrix (front-right and rear-left counter-clockwise seen from
  above, the other two clockwise); displayed speed `4 sqrt(f / f_hover)` rev/s, at most 7 rev/s, so
  that it stays below half the frame rate at 30 frames/s (not the physical speed).
* Front marker: red arrow along body +x (`/World/Quadrotor/nose`).
* Path display: only the trail behind the drone and the upcoming route. Trail: the last 2 s of the
  flown path with a self-lit orange material (appears yellow-orange), width from 14 mm at the drone
  to 2 mm at its end. Upcoming route: the reference route of the next 8 s as a faint 1.2 mm line
  (pale, opacity 0.4). For the fixed cameras, which see the whole area from tens of metres, both
  widths are multiplied by 6.
* Cameras (one per run, `--camera`; no automatic switching):
  * `chase` follows the position and the yaw of the drone only, never its roll or pitch: it is
    2.6 m behind and 0.9 m above the low-pass filtered position (0.35 s), along the low-pass
    filtered yaw (0.8 s, yaw from the attitude quaternion); target the low-pass filtered position
    (0.12 s); up vector world z; pushed back if the drone comes closer than 2 m. Mission: 2.64 m to
    4.83 m from the drone (median 3.13 m).
  * `overview` and `topdown` are computed from the extent of the flight: overview from the side of
    the sun, 35 deg above the horizon; top-down above the centre, north up, east to the right.
* Overlay: time, flight phase, speed and camera, drawn on each frame (headless) or as labels on a
  dark box in the viewport, 64 px below its top edge so that the viewport toolbar does not cover it
  (GUI).

## GUI replay timing

The replay time is the wall-clock time since the start, modulo the flight duration; each frame
shows the truth sample nearest to it (200 Hz). The script logs every 10 s the frame rate, the frame
times and the largest difference between the replay step and the wall-clock step. Measured on
2026-11-27 (mission, chase camera, 1280x720 window), first six 10 s windows: 92.9 to 102.6 frames/s,
median frame time 9.1-10.1 ms. Occasional long frames: 506 ms in the first window (start-up), then
181, 18.4, 223, 73 and 71 ms as the longest frame of the following windows; their cause was not
investigated (OPEN). Replay step minus wall-clock step: 0.000 ms in all windows (no drift and no
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
