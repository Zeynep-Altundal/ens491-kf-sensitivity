"""Recordings for the estimators: the nominal mission recording per measurement seed, the scenarios,
the fixed tuning and the mission phases (protocol: ``README.md``)."""

from __future__ import annotations

import functools
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
_RVIZ = REPO / "simulation" / "rviz_drone"
if str(_RVIZ) not in sys.path:
    sys.path.insert(0, str(_RVIZ))

from rviz_drone import corruption, recorder  # noqa: E402
from rviz_drone import reference as references  # noqa: E402
from rviz_drone.flight import default_config, sensor_streams, simulate_truth  # noqa: E402

from experiments.estimators.filters import Initial, Tuning  # noqa: E402

TRAJECTORY = "mission"
SEEDS = tuple(range(10))
NOMINAL_FILE = REPO / "simulation" / "recordings" / "mission_seed0.npz"
DATA_KEYS = ("imu_t", "imu_gyro", "imu_accel", "truth_t", "truth_p", "truth_v", "truth_q", "truth_bg",
             "truth_ba", "pos_t", "pos_z", "pos_R", "pos_valid")


@functools.lru_cache(maxsize=1)
def truth_flight() -> dict:
    return simulate_truth(trajectory=TRAJECTORY)


def nominal_recording(seed: int) -> dict:
    """Recording of the mission with sensor seed ``seed`` (same truth for every seed), as loaded."""
    truth = truth_flight()
    config = default_config(seed=seed, duration=references.get(TRAJECTORY).duration)
    flight = dict(truth)
    flight.update(sensor_streams(truth, config))
    arrays = recorder.build_recording(TRAJECTORY, seed, config=config, flight=flight)
    rec = {k: v for k, v in arrays.items() if k not in ("meta", "events")}
    import json
    rec["meta"] = json.loads(str(arrays["meta"]))
    rec["events"] = json.loads(str(arrays["events"]))
    return rec


def scenario_recording(nominal: dict, scenario: str) -> dict:
    return nominal if scenario == "nominal" else corruption.apply(nominal, corruption.SCENARIOS[scenario])


def scenario_names() -> list[str]:
    return ["nominal"] + list(corruption.SCENARIOS)


def tuning_from(nominal: dict) -> Tuning:
    """Fixed tuning from the nominal recording (generator noise and nominal truth)."""
    n = nominal["meta"]["true_noise"]
    a_rms = float(np.sqrt(np.mean(nominal["sim_a_w"] ** 2, axis=0)).mean())
    return Tuning(
        gyro_noise_density=n["gyro_noise_density"], accel_noise_density=n["accel_noise_density"],
        gyro_bias_rw=n["gyro_bias_rw"], accel_bias_rw=n["accel_bias_rw"],
        std_p=float(np.max(n["pos_sigma"])), std_v=0.1, std_att=float(np.radians(1.0)),
        std_bg=float(np.max(np.abs(n["gyro_turn_on_bias"]))),
        std_ba=float(np.max(np.abs(n["accel_turn_on_bias"]))),
        q_cv=a_rms ** 2 * 1.0)


def initial(rec: dict) -> Initial:
    return Initial(p=rec["truth_p"][0], v=rec["truth_v"][0], q=rec["truth_q"][0],
                   gyro=rec["imu_gyro"][0], accel=rec["imu_accel"][0])


@functools.lru_cache(maxsize=1)
def phase_labels() -> np.ndarray:
    """Mission segment name at each IMU epoch."""
    traj = references.get(TRAJECTORY)
    t = truth_flight()["t"]
    return np.array([traj.segment_at(float(x)).name for x in t])
