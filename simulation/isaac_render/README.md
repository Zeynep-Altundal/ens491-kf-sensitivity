# Isaac Sim 6.0.1 as renderer for the quadrotor simulation

Goal: render the flight of `../rviz_drone/` (own dynamics, no Pegasus) with Isaac Sim 6.0.1, using
a separate pip installation in `~/isaacsim-6.0.1-ens491` that shares no per-user directory with the
existing Isaac Sim 6.0.1 installation in `~/isaacsim-venv`.

Status: **working** headless (PNG sequence, MP4, stills, heading check) and with a window (GUI
replay). Isaac Sim only renders; the flight comes from the simulator of `../rviz_drone/`.

## Files

| File | Purpose |
|---|---|
| `env.sh` | Sourced; exports `ISAACSIM601_ROOT`, `ISAACSIM_PYTHON` and `RVIZ_DRONE_REPO_ROOT` for the current shell only. |
| `isaac_python.sh` | The isolation wrapper of `../isaac_drone/isaac_python.sh` (same environment variables and Kit settings), running the venv Python. |
| `fetch_assets.sh` | Downloads the assets below into `$ISAACSIM601_ROOT/assets` (outside the repository) and checks their sha256. |
| `smoke_headless.py` | Minimal headless start-up check (new stage, 120 updates, shut down); uses `omni.usd` and `pxr` only. |
| `usd_pose.py` | ENU/FLU/Hamilton to USD pose conversion (numpy only). |
| `replay.py` | Camera paths, propeller angles, trail, flight phase and overlay text (numpy only). |
| `render_flight.py` | Scene, kinematic replay, PNG/MP4/stills, heading check (headless) or real-time replay (`--gui`). |
| `tests/` | Tests of `usd_pose.py` and `replay.py` (part of the repository test suite). |
| `frames/` | Output of `render_flight.py`; git-ignored. |

## Usage

```bash
source simulation/isaac_render/env.sh
bash simulation/isaac_render/fetch_assets.sh            # once
R=simulation/isaac_render/render_flight.py
# 10 s clip (PNG sequence + MP4) of the chase camera
OMNI_KIT_ACCEPT_EULA=YES "$ISAACSIM_PYTHON" $R --start 22 --end 32 --camera chase --mp4
# stills TIME:CAMERA
OMNI_KIT_ACCEPT_EULA=YES "$ISAACSIM_PYTHON" $R --stills 3:chase,29.5:chase,20:overview,33:topdown --rt-subframes 16
# heading check (pure +x flight and pure yaw, seen from above)
OMNI_KIT_ACCEPT_EULA=YES "$ISAACSIM_PYTHON" $R --verify-heading
# window: real-time replay in a loop; close the window to exit
OMNI_KIT_ACCEPT_EULA=YES "$ISAACSIM_PYTHON" $R --gui --camera chase
```

| Option | Values (default first) |
|---|---|
| `--camera` | `chase` (smoothed follow camera), `overview` (fixed), `topdown` (fixed, north up) |
| `--env` | `field` (Isaac flat terrain with grass), `terrain` (Isaac rough terrain with grass), `grid` (plain checkerboard, fallback) |
| `--sky` | `hdri` (sky image), `plain` (uniform dome) |
| `--renderer` (headless) | `pathtraced` (`--spp 64`), `realtime` (RTX real-time, FXAA, `--supersample 2`) |
| other | `--start`, `--end`, `--fps 30`, `--mp4`, `--width 1280`, `--height 720`, `--rt-subframes 8`, `--sun 3000`, `--sky-intensity 400`, `--sun-elevation 47.9`, `--sun-azimuth 146.5`, `--no-overlay`, `--seed 0`, `--out` |

`OMNI_KIT_ACCEPT_EULA=YES` states acceptance of the NVIDIA licence of Isaac Sim and is given per
command only.

## Assets (Isaac Sim 6.0 assets, `https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/`)

| Use | Asset | Licence note |
|---|---|---|
| Drone (chosen) | `Isaac/Robots/NTNU/ARL-Robot-1/arl_robot_1.usd` with its four `configuration/` layers (22 MB) | `LICENSE` file in the folder: BSD 3-Clause, copyright 2025 NTNU Autonomous Robots Lab |
| Drone (candidate) | `Isaac/Robots/Bitcraze/Crazyflie/cf2x.usd` (218 kB) | `LICENSE` file: MIT, copyright 2022 Bitcraze |
| Drone (candidate, used before) | `Isaac/Robots/IsaacSim/Quadcopter/quadcopter.usd` (12 kB) | `LICENSE` file: Creative Commons Attribution 4.0 International |
| Ground (`field`) | `Isaac/Environments/Terrains/flat_plane.usd` (prim `/World/GroundPlane`, 1250 m x 1250 m) | no licence file in the folder; see below |
| Ground (`terrain`) | `Isaac/Environments/Terrains/rough_plane.usd` (prim `/World/mesh`, 96 m x 96 m, +-0.05 m relief; 171 MB) | no licence file in the folder; see below |
| Ground material | `NVIDIA/Materials/Base/Natural/Grass_Countryside.mdl` with its four textures (15 MB) | no licence file in the folder; see below |
| Sky | `NVIDIA/Assets/Skies/Cloudy/kloofendal_48d_partly_cloudy_4k.hdr` (25 MB) | no licence file in the folder; see below |

Assets without a licence file are covered by the NVIDIA Omniverse License Agreement shipped with
the installation (`isaacsim/LICENSE.txt`): "Content" (2D/3D assets) may be used and modified, and
user-generated content such as video, stills and screen captures may be distributed; Content may
not be distributed on a stand-alone basis. The assets are therefore kept outside the repository.
Whether the sky image has an upstream licence of its own is OPEN (no notice in the asset folder).

Drone choice: the Crazyflie is a 28 g vehicle with 0.044 m arms (rotors at +-0.031 m), the
IsaacSim Quadcopter is built from primitive shapes; ARL-Robot-1 is a detailed model (frame,
protective cage, sensors, four propeller prims) of a 1.24 kg vehicle, close to the simulated one,
and its propellers are named `front_left`, `front_right`, `back_left`, `back_right`, which fixes the
forward axis. Not considered: `Isaac/Environments/Outdoor/Rivermark` (26 925 files, 11.7 GB;
loading it over the network would fill a Kit cache whose location under the isolation settings was
not checked).

## Vehicle: visual and dynamics describe the same vehicle

The simulator parameters were set to the asset (not the asset scaled to the model), see
`../rviz_drone/README.md`: mass 1.24 kg (base link 1.2 kg + four 0.01 kg propellers), inertia
`diag(0.013404, 0.014404, 0.013804) kg m^2` (asset inertias with the parallel-axis theorem), arm
`0.141421 m` (rotors at `(+-0.1, +-0.1, 0) m`). At start-up `render_flight.py` prints the rotor
centres of the asset next to `QuadParams.rotor_positions()`; they agree to 0.1 mm. The asset is
rendered at its real scale (metres, no scaling).

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
* Path: whole truth path as a 3 mm line; trail: the last 1.5 s, width 9.5 mm to 1.5 mm and colour
  from orange to the colour of the path line with age.
* Cameras: `chase` follows with a world-fixed offset `(-1.7, 1.1, 0.65) m` (sun side), eye and target
  low-pass filtered (0.6 s and 0.12 s), pushed back to keep at least about 2 m (1.68 m to 3.23 m over
  the flight, median 2.17 m); `overview` at `(-5, 7, 3.6) m` looking at `(0, 0, 1.2) m`; `topdown` at
  11 m, north up, east to the right.
* Overlay: time, flight phase, speed and camera, drawn on each frame (headless) or as labels in the
  viewport (GUI).

## Pose conversion (ENU/FLU/Hamilton to USD)

* World: ENU on a Z-up, metre stage is the stage frame itself: `p_usd = p_enu`.
* Attitude: the simulator quaternion `[w, x, y, z]` (Hamilton, body to world, FLU body) is written
  unchanged (normalised) to `xformOp:orient` (`GfQuatd`, Hamilton, real part first) of the body
  prim, after `xformOp:translate`. USD uses row vectors: `[p_w, 1] = [p_b, 1] M`,
  `M = [[R(q)^T, 0], [p, 1]]`. The asset is aligned with FLU (`ASSET_TO_BODY` identity).
* Run-time check: the local-to-world matrix computed by USD is compared with the expected matrix for
  200 samples of the flight; tolerance 1e-7. Measured maximum difference: 1.392e-08 with the earlier
  vehicle parameters, 1.311e-08 with the current ones (double-precision quaternion-to-matrix
  conversion of Gf).
* Heading check (`--verify-heading`, grid ground, plain sky, camera 6 m above, north up): each pose
  is rendered with and without the nose arrow; the arrow pixels (red and changed) give the rendered
  heading from the projected body origin. Result: pure +x flight from x = -1.5 m to +1.5 m at zero
  attitude: heading 0.3 to 0.4 deg and the drone moves to the right (east) in the image; pure yaw
  0 to 330 deg in 30 deg steps: rendered heading within 0.57 deg of the commanded yaw (19 poses).
* Tests: `tests/test_isaac_render_pose.py` (7) and `tests/test_isaac_render_replay.py` (8).

## Rendering notes (found while building the scene)

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
| 2026-11-27 10:31 | four stills, path traced, `--rt-subframes 16` | exit 0; `frames/stills/` |
| 2026-11-27 10:33-10:48 | 10 s clip t = 22-32 s, chase, 30 frames/s, path traced | exit 0; 301 frames in 840 s (2.8 s per frame), `frames/seq_chase_22.00_32.00.mp4` (H.264, 1280x720, 13.7 MB) |
| 2026-11-27 10:48 | `--gui --camera chase` | window open, overlay in the viewport, real-time replay (RTX real-time renderer of the app) |
