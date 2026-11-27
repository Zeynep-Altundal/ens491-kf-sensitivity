"""Time-windowed disturbances added to a sequence after it was generated.

The same flight can be replayed with different errors, so filters are compared on identical
motion. Every disturbance is an `Event` with a sensor, a time window and a size; IMU events and
position-measurement events are separate, so their effect can be studied one by one. Every
applied event is written to `truth["events"]` with the samples it changed, and IMU bias events
are also added to the true bias series, so the truth stays exact.

Event types (sensor in parentheses):
  bias_step   constant offset inside [t_start, t_end)                 (gyro, accel, pos)
  noise_burst extra white noise of std `magnitude` inside the window  (gyro, accel, pos)
  spike       `count` single samples with an offset of size `magnitude`, random sign
              (IMU, per axis) or random direction (pos)                (gyro, accel, pos)
  outage      no position measurement inside the window               (pos)

Units: gyro rad/s, accel m/s^2, pos m.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from experiments.nav_compare.data.sequence import ImuSequence

SENSORS = ("gyro", "accel", "pos")
TYPES = ("bias_step", "noise_burst", "spike", "outage")


@dataclasses.dataclass(frozen=True)
class Event:
    type: str
    sensor: str
    t_start: float
    t_end: float
    magnitude: float = 0.0
    axes: tuple = (0, 1, 2)
    count: int = 1

    def __post_init__(self):
        if self.type not in TYPES:
            raise ValueError(f"unknown event type {self.type}")
        if self.sensor not in SENSORS:
            raise ValueError(f"unknown sensor {self.sensor}")
        if self.type == "outage" and self.sensor != "pos":
            raise ValueError("outage applies to the position measurement only")
        if not self.t_end > self.t_start:
            raise ValueError("t_end must be larger than t_start")
        if self.magnitude < 0 or self.count < 1:
            raise ValueError("magnitude must be >= 0 and count >= 1")
        if not set(self.axes) <= {0, 1, 2}:
            raise ValueError("axes must be a subset of (0, 1, 2)")


def _rng(seed: int, index: int) -> np.random.Generator:
    """One stream per event, so adding an event does not change the others."""
    return np.random.default_rng(np.random.SeedSequence([seed, 7919, index]))


def apply_events(seq, events, seed: int = 0):
    """Return a copy of `seq` with `events` applied; the input sequence is not changed."""
    t = seq.imu.t_s
    gyro = seq.imu.gyro_rad_s.copy()
    accel = seq.imu.accel_m_s2.copy()
    truth = dict(seq.truth)
    bg = np.array(
        truth.get("gyro_bias_series", np.zeros_like(gyro)), dtype=float, copy=True
    )
    ba = np.array(
        truth.get("accel_bias_series", np.zeros_like(accel)), dtype=float, copy=True
    )
    obs = list(seq.observations)
    hidden = seq.hidden_observations
    log = list(truth.get("events", []))

    for i, ev in enumerate(events):
        rng = _rng(seed, i)
        ax = list(ev.axes)
        entry = dataclasses.asdict(ev)
        if ev.sensor in ("gyro", "accel"):
            meas = gyro if ev.sensor == "gyro" else accel
            bias = bg if ev.sensor == "gyro" else ba
            idx = np.flatnonzero((t >= ev.t_start) & (t < ev.t_end))
            if ev.type == "bias_step":
                meas[np.ix_(idx, ax)] += ev.magnitude
                bias[np.ix_(idx, ax)] += ev.magnitude
            elif ev.type == "noise_burst":
                meas[np.ix_(idx, ax)] += rng.normal(
                    0.0, ev.magnitude, (len(idx), len(ax))
                )
            elif ev.type == "spike":
                pick = np.sort(
                    rng.choice(idx, size=min(ev.count, len(idx)), replace=False)
                )
                signs = rng.choice([-1.0, 1.0], size=(len(pick), len(ax)))
                meas[np.ix_(pick, ax)] += signs * ev.magnitude
                entry["spike_t"] = t[pick].tolist()
            entry["n_samples"] = len(idx)
        else:
            in_win = [j for j, o in enumerate(obs) if ev.t_start <= o.t_s < ev.t_end]
            if ev.type == "outage":
                keep = set(range(len(obs))) - set(in_win)
                hidden += len(in_win)
                obs = [o for j, o in enumerate(obs) if j in keep]
            else:
                pick = in_win
                if ev.type == "spike":
                    n = min(ev.count, len(in_win))
                    pick = (
                        sorted(rng.choice(in_win, size=n, replace=False).tolist())
                        if n
                        else []
                    )
                    entry["spike_t"] = [float(obs[j].t_s) for j in pick]
                for j in pick:
                    o = obs[j]
                    z = np.array(o.z, dtype=float, copy=True)
                    if ev.type == "bias_step":
                        z[ax] += ev.magnitude
                    elif ev.type == "noise_burst":
                        z[ax] += rng.normal(0.0, ev.magnitude, len(ax))
                    else:  # spike: random direction in the chosen axes
                        d = rng.normal(size=len(ax))
                        z[ax] += ev.magnitude * d / np.linalg.norm(d)
                    prov = {**o.provenance, ev.type: True}
                    obs[j] = dataclasses.replace(o, z=z, provenance=prov)
            entry["n_samples"] = len(in_win)
        log.append(entry)

    truth["gyro_bias_series"] = bg
    truth["accel_bias_series"] = ba
    truth["events"] = log
    return dataclasses.replace(
        seq,
        id=f"{seq.id}+ev{len(events)}" if events else seq.id,
        imu=ImuSequence(t, gyro, accel),
        observations=obs,
        hidden_observations=hidden,
        truth=truth,
    )


def standard_event_sets(platform: str, duration_s: float) -> dict:
    """Named disturbance sets for the filter comparison, IMU and position separately.

    Sizes are about 10 to 50 times the normal noise of a medium-grade IMU and of a 0.5 m GPS, so
    each window is clearly visible; the windows avoid the first 10 s (filter start-up).
    """
    a, b = (
        0.40 * duration_s,
        0.40 * duration_s + 3.0,
    )  # a 3 s window in the middle of the flight
    turn = 0.70 * duration_s  # a later window, for the bias on/off test
    sets = {
        "clean": [],
        # --- IMU
        "gyro_noise_burst": [Event("noise_burst", "gyro", a, b, 0.05)],
        "accel_noise_burst": [Event("noise_burst", "accel", a, b, 1.0)],
        "gyro_spikes": [Event("spike", "gyro", 10.0, duration_s, 1.0, count=20)],
        "accel_spikes": [Event("spike", "accel", 10.0, duration_s, 20.0, count=20)],
        "accel_bias_step_x": [Event("bias_step", "accel", a, turn, 0.2, axes=(0,))],
        "accel_bias_step_y": [Event("bias_step", "accel", a, turn, 0.2, axes=(1,))],
        "accel_bias_step_z": [Event("bias_step", "accel", a, turn, 0.2, axes=(2,))],
        "gyro_bias_step_x": [Event("bias_step", "gyro", a, turn, 0.01, axes=(0,))],
        "gyro_bias_step_y": [Event("bias_step", "gyro", a, turn, 0.01, axes=(1,))],
        "gyro_bias_step_z": [Event("bias_step", "gyro", a, turn, 0.01, axes=(2,))],
        # --- position measurement
        "pos_noise_burst": [Event("noise_burst", "pos", a, b, 5.0)],
        "pos_spikes": [Event("spike", "pos", 10.0, duration_s, 20.0, count=5)],
        "pos_bias_step": [Event("bias_step", "pos", a, turn, 3.0, axes=(0, 1))],
        "pos_outage_10s": [Event("outage", "pos", a, a + 10.0)],
        "pos_outage_30s": [Event("outage", "pos", a, a + 30.0)],
        # GNSS-denied: no position at all after 10 s (the setting of the project title)
        "gnss_denied": [Event("outage", "pos", 10.0, duration_s + 1.0)],
        # --- both at once
        "imu_and_pos_burst": [
            Event("noise_burst", "accel", a, b, 1.0),
            Event("noise_burst", "gyro", a, b, 0.05),
            Event("noise_burst", "pos", a, b, 5.0),
        ],
    }
    if platform not in ("ugv", "uav"):
        raise ValueError(platform)
    return sets
