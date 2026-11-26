# Drone simulation with Isaac Sim 5.1.0 and Pegasus Simulator (abandoned)

Attempt to produce multirotor recordings for the filters of this repository with NVIDIA Isaac Sim
5.1.0 and Pegasus Simulator v5.1.0.

Status: **abandoned for now.** Isolation from the existing Isaac Sim 6.0.1 installation was
verified, but Isaac Sim 5.1.0 crashes during start-up on this machine. The simulation work falls
back to the RViz quadrotor model in `../rviz_drone/`.

## Versions and environment

| Item | Value |
|---|---|
| Isaac Sim | 5.1.0 standalone, `VERSION` `5.1.0-rc.19+release.26219.9c81211b.gl`, in `~/isaacsim-5.1.0` |
| Pegasus Simulator | v5.1.0 (commit `644da37`), cloned to `~/PegasusSimulator`, not installed |
| PX4 | not installed |
| GPU | NVIDIA GeForce RTX 5090 Laptop GPU, 24463 MiB |
| Driver | 595.99.02 (Vulkan) |
| OS | Ubuntu 24.04.5 LTS |
| Kit driver requirements (`kit/driver-requirements.json`) | RTX blocked below 535.129.3 on Linux; recommended 535.161.07 |

## Files

| File | Purpose |
|---|---|
| `env.sh` | Sourced; exports `ISAACSIM_PATH`, `ISAACSIM_PYTHON` and `PEGASUS_PATH` for the current shell only. |
| `isaac_python.sh` | Runs the Isaac Sim Python with every per-user location redirected into `$ISAACSIM_PATH/_sandbox` (environment variables and Kit settings, see below) and without the ROS 2 environment of the calling shell. |
| `smoke_headless.py` | Headless start-up check (SimulationApp, empty world, 250 physics steps, resolved Kit path tokens). Run once; crashed inside `SimulationApp` before its first output line. |
| `pegasus_python_smoke.py` | Headless Pegasus flight with the example nonlinear controller and a recording backend. Never run. |

## Installation steps performed

1. Downloaded `isaac-sim-standalone-5.1.0-linux-x86_64.zip` (8 768 419 777 bytes, as stated by the
   HTTP headers) into `~/isaacsim-5.1.0/_download/`; `unzip -t` reported no errors; extracted into
   `~/isaacsim-5.1.0` (17.6 GB, 89 816 files).
2. `post_install.sh` was read and not executed. Its first step (symbolic link `extension_examples`
   inside the installation directory) was performed by hand; its second step was skipped because it
   writes `~/.local/share/applications/IsaacSim.desktop`. `isaac-sim.selector.sh` (GUI) and
   `warmup.sh` were not run.

## Isolation (verified)

Requirement: nothing may be written to directories shared with the Isaac Sim 6.0.1 installation.

Settings used by `isaac_python.sh` for a script run:

* Environment: `HOME`, `XDG_CACHE_HOME`, `XDG_DATA_HOME`, `XDG_CONFIG_HOME`, `TMPDIR`,
  `__GL_SHADER_DISK_CACHE_PATH`, `CUDA_CACHE_PATH`, `PIP_CACHE_DIR` below the sandbox; ROS 2
  variables removed, `ROS_LOCALHOST_ONLY=1`, `ROS_DOMAIN_ID=101`.
* Kit arguments: `--portable-root`, `/structuredLog/enable=false`, `/structuredLog/logDirectory`,
  `/structuredLog/privacySettingsFile` (a sandbox `privacy.toml` with all consents off),
  `/telemetry/enableAnonymousData=false`, `/telemetry/log/file`, and `/app/tokens/` overrides for
  `omni_global_{config,data,logs,cache}`, `omni_{config,data,logs,cache}`, `omni_documents`,
  `shared_documents`, `app_documents`, `documents`.

Procedure: the five shared directories (`~/.nvidia-omniverse`, `~/.cache/ov`, `~/.local/share/ov`,
`~/Documents/Kit`, `~/.nv/ComputeCache`) were backed up with `rsync -a` and listed with type, size,
mtime, mode and SHA-256 per entry before and after one headless launch. The directory
`~/Documents/Kit/shared/screenshots` has mode 000 and could not be read; its mode, owner and mtime
were compared instead.

Result: **0 differences** in all five directories and in the metadata of `screenshots`. Kit wrote
its log (`_sandbox/portable/logs/Kit/Isaac-Sim Python/5.1/`), caches and crash dump only below the
sandbox (9.6 MB). Kit also created `shared/screenshots` with mode 000 inside the sandbox, which
indicates that the mode 000 of the shared directory originates from Kit itself.

## Start-up crash

* One launch, 2026-11-27 02:09: `Segmentation fault (core dumped)` after 15 s, exit code 1.
* Python stack: `smoke_headless.py:17` -> `SimulationApp.__init__` (`simulation_app.py:326`) ->
  `_wait_for_viewport` (`simulation_app.py:636`).
* Crashing thread: `librtx.scenedb.plugin.so` called from `libcarb.scenerenderer-rtx.plugin.so` and
  `libomni.hydra.rtx.plugin.so`.
* Cause: consistent with a reported incompatibility between the 595.x driver branch and the
  Isaac Sim 5.1.0 RTX renderer (`librtx.scenedb.plugin.so`) on the NVIDIA forum; **not verified
  locally.**

## Decision

Isaac Sim 5.1.0 is not used further; it is not launched again and the GPU driver is not changed.
The drone recordings are produced with the RViz quadrotor simulation in `../rviz_drone/`.
