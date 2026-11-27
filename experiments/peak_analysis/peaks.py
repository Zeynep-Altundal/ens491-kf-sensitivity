"""Peak rule of the protocol (``README.md``, "Peak rule"): detection and attribution per run."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

THRESHOLD_MIN_M = 0.1
MIN_DURATION_S = 0.2
MERGE_GAP_S = 0.2
SHOW_MIN_SEEDS = 5  # an estimator "shows a peak" in a scenario if at least this many seeds have one


@dataclass(frozen=True)
class Peak:
    t_start: float
    t_end: float
    t_peak: float
    height: float  # eps_s at t_peak [m]
    excess: float  # D at t_peak [m]
    event: str  # attributed event, "" if unattributed
    onset: float  # onset of the attributed event, nan if unattributed

    @property
    def delay(self) -> float:
        return self.t_peak - self.onset


def threshold(eps_n: np.ndarray) -> float:
    return max(THRESHOLD_MIN_M, float(np.sqrt(np.mean(eps_n ** 2))))


def onsets(events: list[dict]) -> list[tuple[str, float]]:
    """(label, onset) of every corruption event, sorted by onset. Gyro and accelerometer parts of
    one IMU noise corruption (same type and window) are one event; every spike is one event."""
    out, seen = [], set()
    for e in events:
        if e["type"] == "spike":
            out += [(f"spike@{x:.1f}", float(x)) for x in e["times"]]
            continue
        label = "noise_scale" if e["type"] == "noise_scale" else f"{e['type']}:{e['sensor']}"
        key = (label, float(e["t_start"]), e["t_end"])
        if key not in seen:
            seen.add(key)
            out.append((label, float(e["t_start"])))
    return sorted(out, key=lambda x: x[1])


def segments(mask: np.ndarray, dt: float) -> list[tuple[int, int]]:
    """Inclusive index ranges of the ``True`` runs of ``mask`` after merging gaps shorter than
    ``MERGE_GAP_S`` and dropping runs shorter than ``MIN_DURATION_S``."""
    if not mask.any():
        return []
    m = np.r_[False, mask, False].astype(np.int8)
    d = np.diff(m)
    starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0] - 1
    merge_gap = int(round(MERGE_GAP_S / dt))
    min_len = int(round(MIN_DURATION_S / dt))
    merged = [[starts[0], ends[0]]]
    for s, e in zip(starts[1:], ends[1:]):
        if s - merged[-1][1] - 1 < merge_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(int(s), int(e)) for s, e in merged if e - s + 1 >= min_len]


def detect(t: np.ndarray, eps_s: np.ndarray, eps_n: np.ndarray,
           events: list[tuple[str, float]]) -> tuple[list[Peak], float, np.ndarray]:
    """Peaks of one run, the threshold and the excess error ``D``."""
    theta = threshold(eps_n)
    D = eps_s - eps_n
    dt = float(t[1] - t[0])
    peaks = []
    for s, e in segments(D > theta, dt):
        k = s + int(np.argmax(D[s:e + 1]))
        before = [(lab, t0) for lab, t0 in events if t0 <= t[s] + 1e-9]
        lab, t0 = before[-1] if before else ("", float("nan"))
        peaks.append(Peak(float(t[s]), float(t[e]), float(t[k]), float(eps_s[k]), float(D[k]), lab, t0))
    return peaks, theta, D
