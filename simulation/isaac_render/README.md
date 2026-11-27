# Isaac Sim 6.0.1 as renderer for the quadrotor simulation

Goal: render the flight of `../rviz_drone/` (own dynamics, no Pegasus) with Isaac Sim 6.0.1, using
a separate pip installation in `~/isaacsim-6.0.1-ens491` that shares no per-user directory with the
existing Isaac Sim 6.0.1 installation in `~/isaacsim-venv`.

Status: **BLOCKED at the first start-up** (extension dependency failure, see below).

## Installation

| Item | Value |
|---|---|
| venv | `~/isaacsim-6.0.1-ens491/venv`, created with `/usr/bin/python3.12 -m venv` (Python 3.12.3) |
| Command | `pip install "isaacsim[all]==6.0.1" --extra-index-url https://pypi.nvidia.com` with `HOME`, `XDG_*`, `PIP_CACHE_DIR` and `TMPDIR` below `~/isaacsim-6.0.1-ens491/_sandbox` |
| Result | exit 0 after 7 min 49 s; 22 `isaacsim*` packages, `isaacsim 6.0.1.0`, `torch 2.11.0`; no `isaacsim-extscache-*` package |
| Size | venv 8.0 GB, pip cache about 1 GB |
| Side effects | none in the five shared directories and in `~/isaacsim-venv` (0 differences); no file below `/opt/ros/jazzy` changed. pip reported a dependency note for a ROS 2 package seen through the inherited `PYTHONPATH` but installed nothing outside the venv. |

## Files

| File | Purpose |
|---|---|
| `env.sh` | Sourced; exports `ISAACSIM601_ROOT`, `ISAACSIM_PYTHON` and `RVIZ_DRONE_REPO_ROOT` for the current shell only. |
| `isaac_python.sh` | The isolation wrapper of `../isaac_drone/isaac_python.sh` (same environment variables and Kit settings), running the venv Python. |
| `smoke_headless.py` | Minimal headless SimulationApp check (same as `../isaac_drone/smoke_headless.py`). |

## First launch (2026-11-27 02:57, headless, `OMNI_KIT_ACCEPT_EULA=YES`)

* Exit code 55 after 10.6 s, before the first output line of the script:
  `Failed to resolve extension dependencies ... No versions of isaacsim.anim.robot.schema that
  satisfies: isaacsim.exp.base-6.0.1 depends on isaacsim.anim.robot.schema` (not found locally nor
  in the synced registries `kit/prod/default`, 664 packages, and `kit/prod/sdk`, 376 packages).
* The extension is part of the extension-cache packages: `isaacsim.anim.robot.schema-0.1.1+110.1.2`
  exists in the `extscache` folder of `~/isaacsim-venv`, which has `isaacsim-extscache-kit`,
  `isaacsim-extscache-kit-sdk` and `isaacsim-extscache-physics` installed; the new venv has no
  `extscache` folder.
* Isolation: 0 differences in the five shared directories, in the metadata of
  `~/Documents/Kit/shared/screenshots` and in the listing of `~/isaacsim-venv`. The registry index
  was written to `_sandbox/global/data/exts/v2`, i.e. the `omni_global_data` token override is
  effective in 6.0.1 as well.

## Open

Installing the extension-cache packages (`isaacsim[all,extscache]==6.0.1`, the extras present in
the existing installation) into the same venv is expected to resolve the missing extension; not
done, because the install command and a second launch need approval.
