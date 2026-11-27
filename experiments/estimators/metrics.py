"""Per-run metrics (protocol: ``README.md``, "Metrics")."""

from __future__ import annotations

import numpy as np

RECOVERY_TOL_M = 0.1
RECOVERY_HOLD_S = 2.0
PHASES = ("takeoff", "hover", "straight", "turn", "hover_turn", "figure_eight", "aggressive", "return",
          "landing")


def mse_by_phase(err: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Mean squared norm of ``err`` [N,3] per phase and overall (``nan`` rows ignored)."""
    sq = np.sum(err ** 2, axis=1)
    out = {"overall": float(np.mean(sq))}
    for ph in PHASES:
        out[ph] = float(np.mean(sq[labels == ph]))
    return out


def events(scenario_rec: dict) -> list[tuple[str, float | None]]:
    """(label, t_end) of every corruption event; ``t_end`` None when it has no end in the flight."""
    t_last = float(scenario_rec["imu_t"][-1])
    out = []
    for e in scenario_rec.get("events", []):
        if e["type"] == "spike":
            out += [(f"spike@{x:.1f}", float(x)) for x in e["times"]]
        elif e["type"] == "bias_ramp" and e.get("hold_until_end"):
            out.append(("bias_ramp", None))
        else:
            end = e["t_end"]
            out.append((f"{e['type']}:{e['sensor']}", None if end is None or end > t_last else float(end)))
    # IMU noise on gyro and accelerometer with the same window is one event.
    dedup, seen = [], set()
    for label, end in out:
        key = (label.split(":")[0], end)
        if key not in seen:
            seen.add(key)
            dedup.append((label.split(":")[0] if label.startswith("noise_scale") else label, end))
    return dedup


def recovery_time(t: np.ndarray, p_hat: np.ndarray, p_hat_nominal: np.ndarray, t_end: float) -> float:
    """Seconds after ``t_end`` until the estimate stays within the tolerance of the nominal run's
    estimate for ``RECOVERY_HOLD_S`` (or to the end of the flight); ``inf`` if never."""
    d = np.linalg.norm(p_hat - p_hat_nominal, axis=1)
    ok = d <= RECOVERY_TOL_M
    rate = 1.0 / (t[1] - t[0])
    hold = int(round(RECOVERY_HOLD_S * rate))
    k_end = int(np.searchsorted(t, t_end - 1e-9))
    # bad_after[k]: number of tolerance violations from k to the end.
    viol = np.cumsum((~ok)[::-1])[::-1]
    for k in range(k_end, len(t)):
        if not ok[k]:
            continue
        stop = min(k + hold, len(t))
        if viol[k] - (viol[stop] if stop < len(t) else 0) == 0:
            return float(t[k] - t_end)
    return float("inf")


def run_metrics(out: dict, labels: np.ndarray) -> dict:
    ep = np.linalg.norm(out["err_p"], axis=1)
    k = int(np.argmax(ep))
    m = {"peak_p": float(ep[k]), "peak_t": float(out["t"][k])}
    for key, err in (("p", out["err_p"]), ("v", out["err_v"]), ("att_deg", np.degrees(out["err_att"]))):
        for ph, val in mse_by_phase(err, labels).items():
            m[f"mse_{key}_{ph}"] = val
    m["mean_nees"] = float(np.mean(out["nees"]))
    m["mean_nis"] = float(np.mean(out["nis"])) if len(out["nis"]) else float("nan")
    return m
