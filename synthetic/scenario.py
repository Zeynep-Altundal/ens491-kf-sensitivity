"""Scenario configuration and generation.

A :class:`ScenarioConfig` fully determines a scenario. :func:`generate` returns
a dictionary of numpy arrays (ground truth, clean IMU, corrupted IMU, true
biases, position measurements) plus the configuration as a JSON string; the
dictionary can be written to and read from ``.npz`` files with :func:`save_npz`
and :func:`load_npz`.

Randomness: a ``numpy.random.SeedSequence`` built from ``config.seed`` is split
into one child stream per error source, and each stream drives its own
``numpy.random.Generator``. Identical configurations therefore give identical
output, and switching one error source on or off does not alter the
realisation of the others.

Usage (from the repository root)::

    python -m synthetic.scenario --profile turning --out scenario.npz
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

from .imu_model import SensorErrorModel, corrupt_sensor, ideal_imu
from .measurements import PositionMeasurementModel, position_measurements
from .trajectories import PROFILES, get_trajectory

# Order of the random streams; appending new sources keeps existing streams unchanged.
RNG_STREAMS = (
    "gyro_noise",
    "gyro_bias_rw",
    "gyro_outliers",
    "accel_noise",
    "accel_bias_rw",
    "accel_outliers",
    "meas_noise",
    "meas_outliers",
)


@dataclass(frozen=True)
class ScenarioConfig:
    """All parameters of a synthetic scenario.

    Default error values are illustrative, of the order of a MEMS-grade IMU, and
    are not taken from a specific sensor datasheet.
    """

    profile: str = "turning"
    duration: float = 60.0  # [s]
    imu_rate: float = 200.0  # [Hz]
    meas_rate: float = 10.0  # [Hz]
    seed: int = 0

    # Gyroscope.
    gyro_noise_density: float = 2.0e-4  # [rad/s/sqrt(Hz)]
    gyro_turn_on_bias: tuple[float, float, float] = (2.0e-3, -1.0e-3, 1.5e-3)  # [rad/s]
    gyro_bias_rw: float = 2.0e-5  # [rad/s^2/sqrt(Hz)]
    gyro_outlier_prob: float = 0.0  # per sample and axis
    gyro_outlier_magnitude: float = 0.5  # [rad/s]

    # Accelerometer.
    accel_noise_density: float = 2.0e-3  # [m/s^2/sqrt(Hz)]
    accel_turn_on_bias: tuple[float, float, float] = (0.05, -0.03, 0.04)  # [m/s^2]
    accel_bias_rw: float = 3.0e-3  # [m/s^3/sqrt(Hz)]
    accel_outlier_prob: float = 0.0  # per sample and axis
    accel_outlier_magnitude: float = 5.0  # [m/s^2]

    # Fixed sensor-to-body misalignment: rotation R_BS about the body x axis [deg].
    imu_misalignment_x_deg: float = 0.0
    # Gyroscope scale-factor error s: the clean gyro signal is multiplied by (1 + s) on all axes
    # before the sensor errors are added. 0 leaves the generator unchanged.
    gyro_scale_factor: float = 0.0

    # IMU error switches (apply to both gyro and accelerometer).
    enable_white_noise: bool = True
    enable_turn_on_bias: bool = True
    enable_bias_rw: bool = True
    enable_imu_outliers: bool = True

    # Position measurements.
    meas_sigma: tuple[float, float, float] = (0.5, 0.5, 0.5)  # [m]
    meas_dropout_windows: tuple[tuple[float, float], ...] = ()  # (start [s], duration [s])
    meas_outlier_prob: float = 0.0  # per epoch
    meas_outlier_magnitude: float = 5.0  # [m]
    enable_meas_noise: bool = True
    enable_meas_dropout: bool = True
    enable_meas_outliers: bool = True

    def __post_init__(self) -> None:
        if self.profile not in PROFILES:
            raise ValueError(f"unknown profile {self.profile!r}; expected one of {tuple(PROFILES)}")
        if self.gyro_scale_factor <= -1.0:
            raise ValueError("gyro_scale_factor must be greater than -1")
        if self.duration <= 0 or self.imu_rate <= 0 or self.meas_rate <= 0:
            raise ValueError("duration and rates must be positive")
        ratio = self.imu_rate / self.meas_rate
        if abs(ratio - round(ratio)) > 1e-9:
            raise ValueError("imu_rate must be an integer multiple of meas_rate")
        n = self.duration * self.imu_rate
        if abs(n - round(n)) > 1e-6:
            raise ValueError("duration * imu_rate must be an integer")
        for name in ("gyro_outlier_prob", "accel_outlier_prob", "meas_outlier_prob"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        for name in ("gyro_turn_on_bias", "accel_turn_on_bias", "meas_sigma"):
            if len(getattr(self, name)) != 3:
                raise ValueError(f"{name} must have three components")

    def gyro_model(self) -> SensorErrorModel:
        return self._sensor_model("gyro")

    def accel_model(self) -> SensorErrorModel:
        return self._sensor_model("accel")

    def _sensor_model(self, prefix: str) -> SensorErrorModel:
        return SensorErrorModel(
            noise_density=getattr(self, f"{prefix}_noise_density"),
            turn_on_bias=getattr(self, f"{prefix}_turn_on_bias"),
            bias_rw=getattr(self, f"{prefix}_bias_rw"),
            outlier_prob=getattr(self, f"{prefix}_outlier_prob"),
            outlier_magnitude=getattr(self, f"{prefix}_outlier_magnitude"),
            enable_white_noise=self.enable_white_noise,
            enable_turn_on_bias=self.enable_turn_on_bias,
            enable_bias_rw=self.enable_bias_rw,
            enable_outliers=self.enable_imu_outliers,
        )

    def measurement_model(self) -> PositionMeasurementModel:
        return PositionMeasurementModel(
            sigma=self.meas_sigma,
            dropout_windows=self.meas_dropout_windows,
            outlier_prob=self.meas_outlier_prob,
            outlier_magnitude=self.meas_outlier_magnitude,
            enable_noise=self.enable_meas_noise,
            enable_dropout=self.enable_meas_dropout,
            enable_outliers=self.enable_meas_outliers,
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "ScenarioConfig":
        data = json.loads(text)
        tuple_fields = {f.name for f in fields(cls) if str(f.type).startswith("tuple")}
        for name in tuple_fields:
            value = data[name]
            data[name] = tuple(tuple(v) if isinstance(v, list) else v for v in value)
        return cls(**data)


def sensor_to_body_rotation(angle_deg: float) -> np.ndarray:
    """``R_BS = Exp(dphi e_x)``: rotation from the sensor frame to the body frame."""
    a = np.deg2rad(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def generate(config: ScenarioConfig, path: str | Path | None = None) -> dict[str, np.ndarray]:
    """Generate a scenario; optionally also write it to ``path`` as ``.npz``.

    Returned keys
    -------------
    t, p_true, v_true, a_true, q_true, w_true
        IMU time stamps and ground truth at the IMU rate.
    gyro_clean, accel_clean, gyro_meas, accel_meas
        Ideal and corrupted IMU signals in the sensor frame (equal to the body
        frame when ``imu_misalignment_x_deg`` is 0). ``gyro_clean`` is the ideal
        rate before the scale factor ``gyro_scale_factor`` is applied.
    gyro_bias, accel_bias
        True time-varying biases contained in the corrupted signals (sensor frame).
    gyro_outlier_mask, accel_outlier_mask
        Samples and axes that received an outlier spike.
    t_meas, meas_index, p_meas_true, z_meas, meas_available, meas_outlier_mask
        Measurement epochs, their IMU sample indices, true positions,
        measurements (NaN when unavailable), availability and outlier flags.
    R_bs
        Sensor-to-body rotation (identity without misalignment).
    config_json
        The full configuration as a JSON string.
    """
    n = int(round(config.duration * config.imu_rate)) + 1
    t = np.arange(n) / config.imu_rate
    dt = 1.0 / config.imu_rate

    state = get_trajectory(config.profile).evaluate(t)
    gyro_clean, accel_clean = ideal_imu(state)
    r_bs = sensor_to_body_rotation(config.imu_misalignment_x_deg)
    if config.imu_misalignment_x_deg != 0.0:
        # Sensor-frame signals w_S = R_BS^T w_B and f_S = R_BS^T f_B (row vectors: x @ R_BS).
        gyro_clean = gyro_clean @ r_bs
        accel_clean = accel_clean @ r_bs
    gyro_sensed = gyro_clean
    if config.gyro_scale_factor != 0.0:
        # Scale-factor error on the sensed rate: w_m = (1 + s) w_S + b_g + n_g.
        gyro_sensed = (1.0 + config.gyro_scale_factor) * gyro_clean

    children = np.random.SeedSequence(config.seed).spawn(len(RNG_STREAMS))
    rng = {name: np.random.default_rng(ss) for name, ss in zip(RNG_STREAMS, children)}

    gyro = corrupt_sensor(gyro_sensed, dt, config.gyro_model(),
                          rng["gyro_noise"], rng["gyro_bias_rw"], rng["gyro_outliers"])
    accel = corrupt_sensor(accel_clean, dt, config.accel_model(),
                           rng["accel_noise"], rng["accel_bias_rw"], rng["accel_outliers"])

    step = int(round(config.imu_rate / config.meas_rate))
    meas_index = np.arange(0, n, step)
    meas = position_measurements(t[meas_index], state.p[meas_index], config.measurement_model(),
                                 rng["meas_noise"], rng["meas_outliers"])

    result = {
        "t": t,
        "p_true": state.p,
        "v_true": state.v,
        "a_true": state.a,
        "q_true": state.q,
        "w_true": state.w_b,
        "gyro_clean": gyro_clean,
        "accel_clean": accel_clean,
        "gyro_meas": gyro.measured,
        "accel_meas": accel.measured,
        "gyro_bias": gyro.bias,
        "accel_bias": accel.bias,
        "gyro_outlier_mask": gyro.outlier_mask,
        "accel_outlier_mask": accel.outlier_mask,
        "t_meas": t[meas_index],
        "meas_index": meas_index,
        "p_meas_true": state.p[meas_index],
        "z_meas": meas["z"],
        "meas_available": meas["available"],
        "meas_outlier_mask": meas["outlier_mask"],
        "R_bs": r_bs,
        "config_json": np.array(config.to_json()),
    }
    if path is not None:
        save_npz(result, path)
    return result


def save_npz(result: dict[str, np.ndarray], path: str | Path) -> None:
    np.savez_compressed(Path(path), **result)


def load_npz(path: str | Path) -> tuple[dict[str, np.ndarray], ScenarioConfig]:
    """Read a scenario written by :func:`save_npz`; returns (arrays, config)."""
    with np.load(Path(path), allow_pickle=False) as data:
        result = {key: data[key] for key in data.files}
    return result, ScenarioConfig.from_json(str(result["config_json"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic scenario (.npz).")
    parser.add_argument("--profile", choices=tuple(PROFILES), default="turning")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    generate(ScenarioConfig(profile=args.profile, duration=args.duration, seed=args.seed), args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
