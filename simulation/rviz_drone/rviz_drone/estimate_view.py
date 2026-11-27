"""Visual model of one estimator results run (pure numpy, no ROS and no Isaac Sim import).

Used by the Isaac Sim replay (``../../isaac_render/render_flight.py --results``) and by the ROS 2
node ``results_node.py``, so that both show the same elements. Protocol (what each element encodes,
its colour from ``topics.py`` and the results file that feeds it): ``../../isaac_render/README.md``,
section "Estimator replay".
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import reference as references
from . import topics

# Estimators shown, in topic names (topics.ESTIMATORS) and in the names of the results files.
SHOWN = ("kf", "ekf", "eskf")  # the spare slot stays empty
RESULT_NAME = {"kf": "KF", "ekf": "EKF", "eskf": "ESKF"}
TRAJECTORY = "mission"

SIGMA = 3.0  # covariance ellipsoid: 3 sigma
TRAIL_S = 5.0  # [s] trail length of the truth and of every estimate
SPIKE_SHOW_S = 0.5  # [s] a spike (one sample) is announced for this long
# Camera rule: the chase camera sees about +-1.5 m around the drone at its 2.6 m distance. The view
# switches to the overview camera when an estimate is more than SWITCH_M from the true position, and
# back to the chase camera when every estimate has been within BACK_M for BACK_HOLD_S.
SWITCH_M, BACK_M, BACK_HOLD_S = 5.0, 4.0, 2.0

CORRUPTION_NAMES = {
    ("outage", "pos"): "position outage",
    ("bias_step", "accel"): "accelerometer bias step",
    ("bias_step", "gyro"): "gyro bias step",
    ("bias_ramp", "accel"): "accelerometer bias ramp",
    ("bias_ramp", "gyro"): "gyro bias ramp",
    ("noise_scale", "imu"): "IMU noise",
    ("noise_burst", "pos"): "position noise burst",
    ("spike", "pos"): "position spike",
}


def repo_paths(repo: Path, scenario: str, seed: int) -> dict[str, Path]:
    """The results files that feed the replay."""
    rec = repo / "simulation" / "recordings"
    runs = repo / "experiments" / "estimators" / "results" / "runs" / scenario
    return {"truth": rec / "mission_seed0.npz",
            "events": rec / f"mission_seed0__{scenario}.npz",
            "peaks": repo / "experiments" / "peak_analysis" / "results" / "peaks.csv",
            **{name: runs / f"{RESULT_NAME[name]}_seed{seed}.npz" for name in SHOWN}}


@dataclass(frozen=True)
class Window:
    kind: str  # outage, bias_step, bias_ramp, noise_scale, noise_burst, spike
    name: str  # display name
    t0: float
    t1: float  # end of the display window (end of the flight if the corruption has none)

    @property
    def colour(self) -> str:
        return "outage" if self.kind == "outage" else "corruption"


@dataclass
class RunView:
    scenario: str
    seed: int
    t: np.ndarray
    truth_p: np.ndarray
    truth_q: np.ndarray
    truth_v: np.ndarray
    rotor: np.ndarray
    p_ref: np.ndarray
    est_p: dict[str, np.ndarray]
    est_P: dict[str, np.ndarray]  # [N, 3, 3] position covariance
    peaks: dict[str, list[tuple[float, float, np.ndarray]]]  # (t_start, t_peak, position at t_start)
    windows: list[Window]
    sources: dict[str, str] = field(default_factory=dict)

    def err(self, name: str) -> np.ndarray:
        return np.linalg.norm(self.est_p[name] - self.truth_p, axis=1)


def windows_from_events(events: list[dict], t_end: float) -> list[Window]:
    """Display windows of the corruption events; gyro and accelerometer parts of one IMU noise
    corruption are one window, every spike is one short window."""
    out, seen = [], set()
    for e in events:
        if e["type"] == "spike":
            out += [Window("spike", "position spike", float(x), float(x) + SPIKE_SHOW_S) for x in e["times"]]
            continue
        sensor = "imu" if e["type"] == "noise_scale" else e["sensor"]
        t1 = t_end if e["t_end"] is None or e.get("hold_until_end") else min(float(e["t_end"]), t_end)
        key = (e["type"], sensor, float(e["t_start"]), t1)
        if key in seen:
            continue
        seen.add(key)
        out.append(Window(e["type"], CORRUPTION_NAMES[(e["type"], sensor)], float(e["t_start"]), t1))
    return sorted(out, key=lambda w: w.t0)


def active(windows: list[Window], time: float) -> list[Window]:
    return [w for w in windows if w.t0 <= time < w.t1]


def unpack_cov(p6: np.ndarray) -> np.ndarray:
    """[N, 6] upper triangle (xx, xy, xz, yy, yz, zz) to [N, 3, 3]."""
    p6 = np.asarray(p6, dtype=float)
    P = np.empty((len(p6), 3, 3))
    iu = np.triu_indices(3)
    P[:, iu[0], iu[1]] = p6
    P[:, iu[1], iu[0]] = p6
    return P


def ellipsoid(P: np.ndarray, sigma: float = SIGMA) -> tuple[np.ndarray, np.ndarray]:
    """Semi-axes [3] and rotation [3, 3] (columns: axes, right-handed) of the ``sigma`` ellipsoid
    of a 3x3 covariance; the ellipsoid is ``R diag(semi) `` applied to the unit sphere."""
    w, v = np.linalg.eigh(0.5 * (P + P.T))
    if np.linalg.det(v) < 0:
        v[:, 0] = -v[:, 0]
    return sigma * np.sqrt(np.clip(w, 0.0, None)), v


def rot_to_quat(r: np.ndarray) -> np.ndarray:
    """Hamilton quaternion [w, x, y, z] of a rotation matrix (Shepperd's method), w >= 0."""
    r = np.asarray(r, dtype=float)
    tr = np.trace(r)
    if tr > 0:
        s = 2.0 * np.sqrt(1.0 + tr)
        q = np.array([0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + r[i, i] - r[j, j] - r[k, k])
        q = np.empty(4)
        q[0] = (r[k, j] - r[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (r[j, i] + r[i, j]) / s
        q[1 + k] = (r[k, i] + r[i, k]) / s
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def camera_modes(t: np.ndarray, truth_p: np.ndarray, est_p: dict[str, np.ndarray]
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per sample: overview flag, the estimator farthest from the truth and its distance [m]."""
    names = list(est_p)
    d = np.stack([np.linalg.norm(est_p[n] - truth_p, axis=1) for n in names])
    far = np.argmax(d, axis=0)
    dmax = d.max(axis=0)
    hold = int(round(BACK_HOLD_S / (t[1] - t[0])))
    over = np.zeros(len(t), bool)
    state, ok_run = False, 0
    for k in range(len(t)):
        if not state and dmax[k] > SWITCH_M:
            state, ok_run = True, 0
        elif state:
            ok_run = ok_run + 1 if dmax[k] <= BACK_M else 0
            if ok_run >= hold:
                state = False
        over[k] = state
    return over, np.array(names)[far], dmax


def camera_reason(far: str, distance: float) -> str:
    """On-screen reason while the overview camera is active."""
    label = RESULT_NAME.get(far, far)
    if distance > BACK_M:
        return f"{label} estimate {distance:.1f} m from the drone; switch at > {SWITCH_M:g} m"
    return f"all estimates within {BACK_M:g} m; back to chase after {BACK_HOLD_S:g} s"


def trail(t: np.ndarray, p: np.ndarray, k: int, length_s: float = TRAIL_S, n: int = 61) -> np.ndarray:
    """Points of a trail ending at sample k (oldest first), ``n`` samples over ``length_s``."""
    dt = (t[-1] - t[0]) / (len(t) - 1)
    times = t[k] - np.linspace(length_s, 0.0, n)
    idx = np.clip(np.round((times - t[0]) / dt).astype(int), 0, k)
    return p[idx]


def visible_peaks(view: RunView, name: str, time: float) -> list[tuple[float, float, np.ndarray]]:
    return [pk for pk in view.peaks[name] if pk[0] <= time]


def phase_name(time: float) -> str:
    return references.get(TRAJECTORY).segment_at(min(max(time, 0.0), references.get(TRAJECTORY).duration)).name


def overlay_lines(view: RunView, k: int, camera: str, reason: str = "") -> list[str]:
    """Overlay text: time, phase, per-estimator position error, camera (legend drawn by the caller)."""
    time = float(view.t[k])
    lines = [f"{view.scenario}, seed {view.seed}", f"t = {time:6.2f} s   phase: {phase_name(time)}"]
    for n in SHOWN:
        z = float(view.est_p[n][k, 2])
        # An estimate below the ground (z < 0, e.g. a drifting KF) is hidden by the ground in the scene.
        below = f"  (below ground, z = {z:.1f} m)" if z < 0.0 else ""
        lines.append(f"{RESULT_NAME[n]:4s} error {view.err(n)[k]:7.2f} m{below}")
    lines.append(f"camera: {camera}" + (f" ({reason})" if reason else ""))
    return lines


def banner(windows: list[Window]) -> str:
    if not windows:
        return ""
    return "CORRUPTION: " + "; ".join(f"{w.name} ({w.t0:.1f}-{w.t1:.1f} s)" for w in windows)


def legend() -> list[tuple[str, tuple[int, int, int]]]:
    items = [("truth", topics.COLOURS["truth"])]
    items += [(RESULT_NAME[n], topics.COLOURS[n]) for n in SHOWN]
    items += [("peak", topics.COLOURS["peak"]), ("outage", topics.COLOURS["outage"]),
              ("other corruption", topics.COLOURS["corruption"])]
    return items


def load(repo: Path, scenario: str, seed: int) -> RunView:
    paths = repo_paths(Path(repo), scenario, seed)
    with np.load(paths["truth"]) as z:
        t = z["truth_t"].astype(float)
        truth = {k: z[k].astype(float) for k in ("truth_p", "truth_q", "truth_v", "sim_rotor_thrust", "sim_p_ref")}
    events: list[dict] = []
    if scenario != "nominal":
        with np.load(paths["events"]) as z:
            events = json.loads(str(z["events"]))
    est_p, est_P = {}, {}
    for name in SHOWN:
        with np.load(paths[name]) as z:
            if len(z["t"]) != len(t) or np.abs(z["t"] - t).max() > 1e-9:
                raise ValueError(f"time stamps of {paths[name]} differ from the recording")
            if "P_pos" not in z.files:
                raise ValueError(f"{paths[name]} has no P_pos; rerun the estimator sweep")
            est_p[name] = z["p_hat"].astype(float)
            est_P[name] = unpack_cov(z["P_pos"])
    peaks: dict[str, list] = {n: [] for n in SHOWN}
    by_label = {RESULT_NAME[n]: n for n in SHOWN}
    with paths["peaks"].open() as fh:
        for r in csv.DictReader(fh):
            if r["scenario"] == scenario and int(r["seed"]) == seed and r["estimator"] in by_label:
                n = by_label[r["estimator"]]
                ts = float(r["t_start"])
                k = int(np.clip(np.searchsorted(t, ts - 1e-9), 0, len(t) - 1))
                peaks[n].append((ts, float(r["t_peak"]), est_p[n][k].copy()))
    return RunView(scenario, seed, t, truth["truth_p"], truth["truth_q"], truth["truth_v"],
                   truth["sim_rotor_thrust"], truth["sim_p_ref"], est_p, est_P, peaks,
                   windows_from_events(events, float(t[-1])), {k: str(v) for k, v in paths.items()})
