"""Corruption scenarios added afterwards to a recording in the common format (v1).

    python -m rviz_drone.corruption [--nominal PATH] [--out-dir DIR]   (from simulation/rviz_drone/)

writes one file per scenario of ``SCENARIOS`` next to the nominal recording. The layer works on
recordings only (arrays in, arrays out); it knows nothing about estimators or the simulator.
Scenario table, application rules and the stored corruption truth: ``../README.md``, section
"Corruption scenarios".
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import recorder

IMU_SENSORS = ("gyro", "accel")
UNITS = {"gyro": "rad/s", "accel": "m/s^2", "pos": "m"}


@dataclass(frozen=True)
class Corruption:
    """One corruption. ``type``: bias_step, bias_ramp, noise_scale, noise_burst, spike, outage."""
    type: str
    sensor: str  # gyro, accel or pos
    t_start: float = 0.0
    t_end: float = float("inf")
    magnitude: float = 0.0
    axes: tuple[int, ...] = (0, 1, 2)
    count: int = 1  # spikes
    min_spacing_s: float = 0.0  # spikes
    hold_until_end: bool = False  # bias_ramp: keep the final value after t_end


@dataclass(frozen=True)
class Scenario:
    name: str
    seed: int
    corruptions: tuple[Corruption, ...] = field(default_factory=tuple)
    reveals: str = ""


def _scenarios() -> dict[str, Scenario]:
    s = [
        Scenario("accel_bias_step_small", 1001, (Corruption("bias_step", "accel", 25.0, 49.0, 0.2, (0,)),),
                 "moderate accelerometer bias change and recovery"),
        Scenario("accel_bias_step_large", 1002, (Corruption("bias_step", "accel", 25.0, 49.0, 1.0, (0,)),),
                 "large accelerometer bias change and recovery"),
        Scenario("gyro_bias_step_small", 1003, (Corruption("bias_step", "gyro", 25.0, 49.0, 0.01, (0,)),),
                 "roll error from a gyro bias"),
        Scenario("gyro_bias_step_large", 1004, (Corruption("bias_step", "gyro", 25.0, 49.0, 0.05, (0,)),),
                 "roll error from a large gyro bias"),
        Scenario("accel_bias_ramp", 1005,
                 (Corruption("bias_ramp", "accel", 7.0, 74.0, 0.5, (0,), hold_until_end=True),),
                 "slow accelerometer bias drift"),
        Scenario("imu_noise_x3", 1006, (Corruption("noise_scale", "gyro", magnitude=3.0),
                                        Corruption("noise_scale", "accel", magnitude=3.0)),
                 "IMU noise three times the nominal level"),
        Scenario("imu_noise_x10", 1007, (Corruption("noise_scale", "gyro", magnitude=10.0),
                                         Corruption("noise_scale", "accel", magnitude=10.0)),
                 "IMU noise ten times the nominal level"),
        Scenario("pos_burst_short_calm", 1008, (Corruption("noise_burst", "pos", 10.0, 11.0, 3.0),),
                 "short position noise burst, straight leg"),
        Scenario("pos_burst_long_calm", 1009, (Corruption("noise_burst", "pos", 10.0, 15.0, 3.0),),
                 "long position noise burst, straight leg"),
        Scenario("pos_burst_short_aggressive", 1010, (Corruption("noise_burst", "pos", 55.0, 56.0, 3.0),),
                 "short position noise burst, aggressive segment"),
        Scenario("pos_burst_long_aggressive", 1011, (Corruption("noise_burst", "pos", 52.0, 57.0, 3.0),),
                 "long position noise burst, aggressive segment"),
        Scenario("pos_spike_single", 1012, (Corruption("spike", "pos", 12.0, 12.1, 10.0, count=1),),
                 "one position outlier"),
        Scenario("pos_spikes_repeated", 1013,
                 (Corruption("spike", "pos", 25.0, 63.0, 10.0, count=10, min_spacing_s=1.0),),
                 "repeated position outliers"),
        Scenario("outage_5s_calm", 1014, (Corruption("outage", "pos", 8.0, 13.0),), "5 s outage, straight leg"),
        Scenario("outage_20s_calm", 1015, (Corruption("outage", "pos", 8.0, 28.0),), "20 s outage, calm phases"),
        Scenario("outage_40s_calm", 1016, (Corruption("outage", "pos", 8.0, 48.0),), "40 s outage, calm phases"),
        Scenario("outage_5s_aggressive", 1017, (Corruption("outage", "pos", 52.0, 57.0),),
                 "5 s outage, aggressive segment"),
        Scenario("outage_20s_aggressive", 1018, (Corruption("outage", "pos", 49.0, 69.0),),
                 "20 s outage from the start of the aggressive segment"),
        Scenario("outage_40s_aggressive", 1019, (Corruption("outage", "pos", 42.0, 82.0),),
                 "40 s outage over the aggressive segment to the end of the landing"),
    ]
    return {x.name: x for x in s}


SCENARIOS = _scenarios()


def _window(t: np.ndarray, c: Corruption) -> np.ndarray:
    return (t >= c.t_start) & (t < c.t_end)


def _event(c: Corruption, **extra) -> dict:
    e = {"type": c.type, "sensor": c.sensor, "axes": list(c.axes), "t_start": c.t_start,
         "t_end": c.t_end if np.isfinite(c.t_end) else None, "magnitude": c.magnitude,
         "unit": "factor on the noise std" if c.type == "noise_scale" else
                 (f"{UNITS[c.sensor]} (std)" if c.type == "noise_burst" else UNITS[c.sensor])}
    e.update(extra)
    return e


def apply(rec: dict, scenario: Scenario, nominal_sha256: str | None = None) -> dict:
    """Corrupted copy of a recording (``meta``/``events`` as dict/list, as from ``recorder.load``).

    Without corruptions the recording is returned unchanged (no extra keys)."""
    out = {k: (v.copy() if isinstance(v, np.ndarray) else json.loads(json.dumps(v))) for k, v in rec.items()}
    if not scenario.corruptions:
        return out
    t, tp = rec["imu_t"], rec["pos_t"]
    n, k = len(t), len(tp)
    corr = {"corr_gyro_bias": np.zeros((n, 3)), "corr_accel_bias": np.zeros((n, 3)),
            "corr_gyro_noise_scale": np.ones(n), "corr_accel_noise_scale": np.ones(n),
            "corr_pos_noise": np.zeros((k, 3)), "corr_pos_noise_std": np.zeros(k),
            "corr_pos_spike": np.zeros((k, 3)), "corr_pos_outage": np.zeros(k, dtype=bool)}
    rng = np.random.default_rng(scenario.seed)
    events = list(out["events"])
    order = {"noise_scale": 0, "bias_step": 1, "bias_ramp": 1, "noise_burst": 2, "spike": 3, "outage": 4}
    for c in scenario.corruptions:
        if c.type not in order:
            raise ValueError(f"unknown corruption type {c.type!r}")
    for c in sorted(scenario.corruptions, key=lambda c: order[c.type]):
        extra = {}
        if c.type == "noise_scale":
            bias = {"gyro": "truth_bg", "accel": "truth_ba"}[c.sensor]
            white = rec[f"imu_{c.sensor}"] - rec[f"imu_{c.sensor}_clean"] - rec[bias]
            scale = np.where(_window(t, c), c.magnitude, 1.0)
            out[f"imu_{c.sensor}"] = out[f"imu_{c.sensor}"] + (scale - 1.0)[:, None] * white
            corr[f"corr_{c.sensor}_noise_scale"] = corr[f"corr_{c.sensor}_noise_scale"] * scale
        elif c.type in ("bias_step", "bias_ramp"):
            if c.type == "bias_step":
                value = np.where(_window(t, c), c.magnitude, 0.0)
            else:
                frac = np.clip((t - c.t_start) / (c.t_end - c.t_start), 0.0, 1.0)
                frac[t >= c.t_end] = 1.0 if c.hold_until_end else 0.0
                value = c.magnitude * frac
            b = np.zeros((n, 3))
            b[:, list(c.axes)] = value[:, None]
            out[f"imu_{c.sensor}"] = out[f"imu_{c.sensor}"] + b
            truth = {"gyro": "truth_bg", "accel": "truth_ba"}[c.sensor]
            out[truth] = out[truth] + b
            corr[f"corr_{c.sensor}_bias"] = corr[f"corr_{c.sensor}_bias"] + b
            if c.type == "bias_ramp":
                extra["hold_until_end"] = c.hold_until_end
        elif c.type == "noise_burst":
            m = _window(tp, c)
            noise = np.zeros((k, 3))
            noise[np.ix_(m, list(c.axes))] = rng.normal(0.0, c.magnitude, size=(int(m.sum()), len(c.axes)))
            out["pos_z"] = out["pos_z"] + noise
            corr["corr_pos_noise"] = corr["corr_pos_noise"] + noise
            corr["corr_pos_noise_std"] = np.where(m, np.hypot(corr["corr_pos_noise_std"], c.magnitude),
                                                  corr["corr_pos_noise_std"])
        elif c.type == "spike":
            candidates = np.flatnonzero(_window(tp, c) & rec["pos_valid"])
            chosen: list[int] = []
            if c.count == 1:
                chosen = [int(candidates[0])]
            else:
                for idx in rng.permutation(candidates):
                    if all(abs(tp[idx] - tp[j]) >= c.min_spacing_s for j in chosen):
                        chosen.append(int(idx))
                    if len(chosen) == c.count:
                        break
                if len(chosen) < c.count:
                    raise ValueError(f"cannot place {c.count} spikes in [{c.t_start}, {c.t_end})")
            chosen.sort()
            for idx in chosen:
                u = np.zeros(3)
                u[list(c.axes)] = rng.normal(size=len(c.axes))
                spike = c.magnitude * u / np.linalg.norm(u)
                out["pos_z"][idx] += spike
                corr["corr_pos_spike"][idx] += spike
            extra = {"count": len(chosen), "times": [float(tp[i]) for i in chosen]}
        elif c.type == "outage":
            m = _window(tp, c)
            out["pos_valid"] = out["pos_valid"] & ~m
            out["pos_z"][m] = np.nan
            corr["corr_pos_outage"] = corr["corr_pos_outage"] | m
        else:
            raise ValueError(f"unknown corruption type {c.type!r}")
        events.append(_event(c, **extra))
    out.update(corr)
    out["events"] = events
    meta = dict(out["meta"])
    meta["id"] = f"{meta.get('id', 'recording')}__{scenario.name}"
    meta["scenario"] = {"name": scenario.name, "seed": scenario.seed, "reveals": scenario.reveals,
                        "nominal_sha256": nominal_sha256,
                        "corruptions": [json.loads(json.dumps(c.__dict__, default=list)) for c in scenario.corruptions]}
    for c in meta["scenario"]["corruptions"]:
        if not np.isfinite(c["t_end"]):
            c["t_end"] = None
    out["meta"] = meta
    return out


def to_arrays(rec: dict) -> dict[str, np.ndarray]:
    """Arrays for ``recorder.write_npz`` (``meta`` and ``events`` back to JSON strings)."""
    arrays = {k: v for k, v in rec.items() if k not in ("meta", "events")}
    arrays["meta"] = np.array(json.dumps(rec["meta"], sort_keys=True))
    arrays["events"] = np.array(json.dumps(rec["events"], sort_keys=True))
    return arrays


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_all(nominal: Path, out_dir: Path | None = None, names=None) -> list[Path]:
    nominal = Path(nominal)
    out_dir = Path(out_dir) if out_dir is not None else nominal.parent
    digest = sha256(nominal)
    rec = recorder.load(nominal)
    paths = []
    for name in names or SCENARIOS:
        path = out_dir / f"{nominal.stem}__{name}.npz"
        recorder.write_npz(path, to_arrays(apply(rec, SCENARIOS[name], digest)))
        paths.append(path)
    if sha256(nominal) != digest:
        raise RuntimeError("the nominal recording changed")
    return paths


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--nominal", type=Path, default=recorder.DEFAULT_DIR / "mission_seed0.npz")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    before = sha256(args.nominal)
    for p in write_all(args.nominal, args.out_dir):
        print(f"{p} {p.stat().st_size}")
    print(f"nominal {args.nominal} sha256 {before} (unchanged: {sha256(args.nominal) == before})")


if __name__ == "__main__":
    main()
