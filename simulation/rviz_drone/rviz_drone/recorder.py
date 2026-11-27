"""Write one simulated flight in the common recording format (v1) of the filter comparison.

    python -m rviz_drone.recorder [--trajectory mission] [--seed 0] [--out PATH]   (from simulation/rviz_drone/)

Format, rates, frames and metadata: ``../README.md``, section "Flight recorder". The output is
bit-identical for the same code, trajectory and seed: the ``.npz`` archive is written member by
member in sorted key order with a fixed time stamp, instead of with ``numpy.savez_compressed``,
which stores the current time.
"""

from __future__ import annotations

import argparse
import dataclasses
import io
import json
import os
import subprocess
import zipfile
from pathlib import Path

import numpy as np

from . import repo_path  # noqa: F401
from . import reference as references
from .flight import default_config, run
from .quadrotor import ControllerGains, QuadParams
from synthetic.imu_model import GRAVITY_W
from synthetic.scenario import ScenarioConfig

FORMAT_VERSION = "v1"
SOURCE = "ens491_quadrotor_sim"
SIM_RATE_HZ = 1000.0
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DIR = REPO_ROOT / "simulation" / "recordings"
_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


def git_state(root: Path = REPO_ROOT) -> tuple[str, bool]:
    """Commit hash of HEAD and whether tracked files differ from it ("unknown" without git)."""
    try:
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
                                capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                               check=True, capture_output=True, text=True).stdout.strip() != ""
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


def _plain(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, tuple):
        return list(x)
    raise TypeError(f"not JSON serialisable: {type(x)}")


def _json(obj) -> str:
    return json.dumps(obj, sort_keys=True, default=_plain)


def build_recording(trajectory: str | None = None, seed: int = 0, config: ScenarioConfig | None = None,
                    flight: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    """All keys of the file as arrays (``meta`` and ``events`` as 0-d unicode arrays of JSON)."""
    name = trajectory or references.DEFAULT_TRAJECTORY
    duration = references.get(name).duration
    config = config or default_config(seed=seed, duration=duration)
    if config.seed != seed:
        config = dataclasses.replace(config, seed=seed)
    f = flight if flight is not None else run(seed=seed, config=config, trajectory=name)
    t = f["t"]
    sigma = np.asarray(config.meas_sigma, dtype=float)
    k = len(f["t_meas"])
    commit, dirty = git_state()
    meta = {
        "version": FORMAT_VERSION,
        "id": f"{name}_seed{seed}",
        "platform": "uav",
        "source": SOURCE,
        "generator": "rviz_drone.recorder",
        "imu_rate_hz": float(config.imu_rate),
        "pos_rate_hz": float(config.meas_rate),
        "seed": int(seed),
        "trajectory": name,
        "duration_s": float(duration),
        "true_noise": {
            "gyro_noise_density": config.gyro_noise_density, "gyro_turn_on_bias": config.gyro_turn_on_bias,
            "gyro_bias_rw": config.gyro_bias_rw, "accel_noise_density": config.accel_noise_density,
            "accel_turn_on_bias": config.accel_turn_on_bias, "accel_bias_rw": config.accel_bias_rw,
            "pos_sigma": config.meas_sigma,
            "units": {"noise_density": "rad/s/sqrt(Hz), m/s^2/sqrt(Hz)", "turn_on_bias": "rad/s, m/s^2",
                      "bias_rw": "rad/s^2/sqrt(Hz), m/s^3/sqrt(Hz)", "pos_sigma": "m"},
        },
        "frames": {"world": "ENU, origin on the ground at the take-off point (first truth sample)",
                   "body": "FLU, equal to the IMU frame", "quaternion": "Hamilton [w,x,y,z], body to world"},
        "gravity_w": GRAVITY_W,
        "git_commit": commit,
        "git_dirty": dirty,
        "parameters": {"vehicle": dataclasses.asdict(QuadParams()), "controller": dataclasses.asdict(ControllerGains()),
                       "sim_rate_hz": SIM_RATE_HZ, "scenario_config": json.loads(config.to_json())},
    }
    return {
        "meta": np.array(_json(meta)),
        "events": np.array(_json([])),
        "imu_t": t.copy(),
        "imu_gyro": f["gyro_meas"], "imu_accel": f["accel_meas"],
        "imu_gyro_clean": f["gyro_clean"], "imu_accel_clean": f["accel_clean"],
        "truth_t": t.copy(),
        "truth_p": f["p"], "truth_v": f["v"], "truth_q": f["q"],
        "truth_bg": f["gyro_bias"], "truth_ba": f["accel_bias"],
        "pos_t": f["t_meas"], "pos_z": f["z_meas"],
        "pos_R": np.broadcast_to(np.diag(sigma ** 2), (k, 3, 3)).copy(),
        "pos_valid": np.asarray(f["meas_available"], dtype=bool),
        "sim_p_ref": f["p_ref"], "sim_w_b": f["w_b"], "sim_a_w": f["a"], "sim_rotor_thrust": f["rotor"],
    }


def write_npz(path: Path, arrays: dict[str, np.ndarray]) -> Path:
    """Deterministic ``.npz``: members ``<key>.npy`` in sorted order, deflate, fixed time stamp."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for key in sorted(arrays):
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.asarray(arrays[key]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{key}.npy", date_time=_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, buf.getvalue(), compresslevel=6)
    os.replace(tmp, path)
    return path


def load(path: Path) -> dict:
    """Arrays of the file, with ``meta`` and ``events`` parsed from JSON."""
    with np.load(path, allow_pickle=False) as d:
        out = {k: d[k] for k in d.files}
    out["meta"] = json.loads(str(out["meta"]))
    out["events"] = json.loads(str(out["events"]))
    return out


def record(trajectory: str | None = None, seed: int = 0, out: Path | None = None) -> Path:
    name = trajectory or references.DEFAULT_TRAJECTORY
    out = Path(out) if out is not None else DEFAULT_DIR / f"{name}_seed{seed}.npz"
    return write_npz(out, build_recording(name, seed))


def summary(rec: dict) -> str:
    """Markdown summary table of a loaded recording."""
    m = rec["meta"]
    rows = [
        ("trajectory, seed", f"{m['trajectory']}, {m['seed']}"),
        ("duration", f"{rec['imu_t'][-1]:.3f} s"),
        ("IMU samples (rate)", f"{len(rec['imu_t'])} ({m['imu_rate_hz']:.0f} Hz)"),
        ("position samples (rate), valid", f"{len(rec['pos_t'])} ({m['pos_rate_hz']:.0f} Hz), {int(rec['pos_valid'].sum())}"),
        ("max speed", f"{np.linalg.norm(rec['truth_v'], axis=1).max():.2f} m/s"),
        ("max gyro norm", f"{np.linalg.norm(rec['imu_gyro'], axis=1).max():.2f} rad/s"),
        ("accel norm range", f"{np.linalg.norm(rec['imu_accel'], axis=1).min():.2f} to "
                          f"{np.linalg.norm(rec['imu_accel'], axis=1).max():.2f} m/s^2"),
        ("position error RMS (per axis)", ", ".join(f"{x:.3f}" for x in np.sqrt(np.nanmean(
            (rec['pos_z'] - rec['truth_p'][::int(round(m['imu_rate_hz'] / m['pos_rate_hz']))]) ** 2, axis=0))) + " m"),
        ("gyro bias at end", ", ".join(f"{x:.2e}" for x in rec["truth_bg"][-1]) + " rad/s"),
        ("accel bias at end", ", ".join(f"{x:.3f}" for x in rec["truth_ba"][-1]) + " m/s^2"),
        ("git commit (dirty)", f"{m['git_commit'][:12]} ({m['git_dirty']})"),
    ]
    return "\n".join(["| Item | Value |", "|---|---|"] + [f"| {a} | {b} |" for a, b in rows])


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trajectory", choices=sorted(references.TRAJECTORIES), default=references.DEFAULT_TRAJECTORY)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    path = record(args.trajectory, args.seed, args.out)
    print(f"{path} ({path.stat().st_size} bytes)")
    print(summary(load(path)))


if __name__ == "__main__":
    main()
