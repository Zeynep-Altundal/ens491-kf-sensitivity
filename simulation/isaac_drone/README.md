# Drone simulation with Isaac Sim 5.1.0 and Pegasus Simulator

Multirotor simulation intended to produce recordings for the filters of this repository, following
the common interface of the drone experiments (`docs/common_interface.md` in `ens491-nkf`).

Status: **BLOCKED before the first start of Isaac Sim** (see "Isolation" below). None of the
scripts in this folder has been run yet.

## Versions

| Component | Version | Location (outside the repository) |
|---|---|---|
| Isaac Sim | 5.1.0 standalone (`VERSION`: `5.1.0-rc.19+release.26219.9c81211b.gl`) | `~/isaacsim-5.1.0` |
| Pegasus Simulator | v5.1.0 (commit `644da37`), documented for Isaac Sim 5.1.0 | `~/PegasusSimulator` |
| PX4 | not installed; the Python control backend of Pegasus is used | - |

## Files

| File | Purpose |
|---|---|
| `env.sh` | Sourced; exports `ISAACSIM_PATH`, `ISAACSIM_PYTHON` and `PEGASUS_PATH` for the current shell only. |
| `isaac_python.sh` | Runs the Isaac Sim Python with per-user paths redirected into `$ISAACSIM_PATH/_sandbox` and without the ROS 2 environment of the calling shell. |
| `smoke_headless.py` | Headless start-up check: SimulationApp, empty world, 250 physics steps, resolved Kit path tokens. Not run. |
| `pegasus_python_smoke.py` | Headless flight of the Pegasus Iris with the example nonlinear controller for a fixed simulated time; records ground truth and all sensor samples to `.npz`. Not run. |

## Installation steps performed

1. Downloaded `isaac-sim-standalone-5.1.0-linux-x86_64.zip` (8 768 419 777 bytes) into
   `~/isaacsim-5.1.0/_download/`; `unzip -t` reported no errors; extracted into `~/isaacsim-5.1.0`.
2. `post_install.sh` was read and not executed. Its first step (symbolic link `extension_examples`
   inside the installation directory) was performed by hand. Its second step was skipped because it
   writes `~/.local/share/applications/IsaacSim.desktop` and calls `gio`. `isaac-sim.selector.sh`
   (opens a GUI) and `warmup.sh` were not run.
3. Pegasus cloned at tag v5.1.0; not installed into the Isaac Sim Python yet.

## Isolation (OPEN)

Requirement: Isaac Sim 5.1.0 must write nothing to directories shared with the Isaac Sim 6.0.1
installation (`~/.cache/ov`, `~/.local/share/ov`, `~/.nvidia-omniverse`, `~/Documents/Kit`).

Found by reading the installation, without starting it:

* `SimulationApp` starts Kit with `--portable` unless `--portable-root` is given
  (`exts/isaacsim.simulation_app/isaacsim/simulation_app/simulation_app.py`, lines 442-445);
  according to the Kit help text, portable mode places the data, cache and logs folders below the
  portable root (default `${kit}`). `extra_args` can pass `--portable-root`.
* The Kit application plugin (`kit/kernel/plugins/libomni.kit.app.plugin.so`) contains
  `getpwuid_r`, `.nvidia-omniverse`, `XDG_DATA_HOME`, `XDG_CACHE_HOME`, `XDG_CONFIG_HOME` but no
  `HOME` string: the home directory is probably taken from the password database, so overriding
  `$HOME` would not redirect paths derived from it.
* The experience files load `omni.kit.telemetry`; its transmitter writes
  `${omni_logs}/omni.telemetry.transmitter.log` and, according to its own tests, a single
  transmitter instance is shared between Kit processes. Whether `${omni_logs}` follows the portable
  root is not documented in the installation.
* The available confinement tools could not be used: `bwrap` fails with
  `setting up uid map: Permission denied` (unprivileged user namespaces are restricted by
  AppArmor), and `strace -P` path filtering matches only the exact path, not files below it.

To resolve: either accept a possible write below `~/.nvidia-omniverse/logs`, or find a documented
setting that disables the telemetry transmitter and relocates `${omni_logs}`, then verify with a
before/after comparison of the four shared directories.
