"""Pre-declared diagnostic of the outage reversal (protocol: ``README.md``, "Outage reversal diagnostic").

    .venv/bin/python -m experiments.peak_analysis.outage_diagnostic      (from the repository root)

Writes ``results/outage_diagnostic.json``. The decision rule is fixed in the README; this script
only evaluates it.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from experiments.estimators import data
from experiments.estimators.filters import ekf_step
from experiments.estimators.run import position_epochs, run
from synthetic.rotation import quat_conjugate, quat_log, quat_multiply

RESULTS = Path(__file__).resolve().parent / "results"
SCENARIOS = ("outage_40s_calm", "outage_40s_aggressive")
REPRO_TOL_M = 1e-6
# Counterfactual onset states of the EKF: which parts of x0 are set to the truth.
VARIANTS = {"C0": (), "C1": ("q", "bg", "ba"), "C1a": ("bg", "ba"), "C1b": ("q",), "C2": ("p", "v")}
SLICES = {"p": slice(0, 3), "v": slice(3, 6), "q": slice(6, 10), "bg": slice(10, 13), "ba": slice(13, 16)}


def _rms(e: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(e ** 2, axis=1))))


def window(rec: dict) -> tuple[int, int]:
    """IMU indices ``[k0, k1]`` of the outage window: from the first invalid position epoch to the
    epoch before the next valid position update (or the end of the flight)."""
    k = position_epochs(rec["imu_t"], rec["pos_t"])
    valid = rec["pos_valid"]
    j0 = int(np.argmax(~valid))
    after = np.nonzero(valid[j0:])[0]
    k1 = int(k[j0 + after[0]]) - 1 if len(after) else len(rec["imu_t"]) - 1
    return int(k[j0]), k1


def propagate(x0: np.ndarray, rec: dict, k0: int, k1: int) -> np.ndarray:
    """Open-loop nominal propagation from the state at ``k0 - 1`` over ``k0..k1``; positions."""
    t, g, a = rec["imu_t"], rec["imu_gyro"], rec["imu_accel"]
    x = x0.copy()
    out = np.empty((k1 - k0 + 1, 3))
    for i, k in enumerate(range(k0, k1 + 1)):
        x = ekf_step(x, g[k - 1], a[k - 1], g[k], a[k], t[k] - t[k - 1])
        out[i] = x[0:3]
    return out


def _seed(args) -> dict:
    scen, seed = args
    rec = data.scenario_recording(data.nominal_recording(seed), scen)
    tuning = data.tuning_from(data.nominal_recording(0))
    k0, k1 = window(rec)
    tt = rec["imu_t"]
    w = slice(k0, k1 + 1)
    tp, tv = rec["truth_p"], rec["truth_v"]
    # KF: exact split e = T + (dp0 + dv0 tau).
    kf = run("KF", rec, tuning)
    e_kf = tp[w] - kf["p_hat"][w]
    tau = (tt[w] - tt[k0 - 1])[:, None]
    T = tp[w] - tp[k0 - 1] - tv[k0 - 1] * tau
    onset_kf = (tp[k0 - 1] - kf["p_hat"][k0 - 1]) + (tv[k0 - 1] - kf["v_hat"][k0 - 1]) * tau
    split_residual = float(np.abs(e_kf - T - onset_kf).max())
    # EKF: counterfactual open-loop propagation from the onset state.
    ekf = run("own_EKF", rec, tuning)
    x_hat = np.r_[ekf["p_hat"][k0 - 1], ekf["v_hat"][k0 - 1], ekf["q_hat"][k0 - 1], ekf["bg_hat"][k0 - 1],
                  ekf["ba_hat"][k0 - 1]]
    x_true = np.r_[tp[k0 - 1], tv[k0 - 1], rec["truth_q"][k0 - 1], rec["truth_bg"][k0 - 1], rec["truth_ba"][k0 - 1]]
    rms = {}
    repro = None
    for name, parts in VARIANTS.items():
        x0 = x_hat.copy()
        for p in parts:
            x0[SLICES[p]] = x_true[SLICES[p]]
        p_cf = propagate(x0, rec, k0, k1)
        if name == "C0":
            repro = float(np.abs(p_cf - ekf["p_hat"][w]).max())
        rms[name] = _rms(tp[w] - p_cf)
    dth = quat_log(quat_multiply(quat_conjugate(x_hat[6:10][None]), x_true[6:10][None]))[0]
    return {
        "scenario": scen, "seed": seed, "window_s": [float(tt[k0]), float(tt[k1])],
        "kf_rms": _rms(e_kf), "T_rms": _rms(T), "kf_onset_term_rms": _rms(onset_kf), "kf_split_residual_m": split_residual,
        "ekf_run_rms": _rms(tp[w] - ekf["p_hat"][w]), "ekf_c0_repro_m": repro, "ekf_rms": rms,
        "onset_errors": {"tilt_deg": float(np.degrees(np.linalg.norm(dth[:2]))), "yaw_deg": float(np.degrees(abs(dth[2]))),
                         "bg_err": float(np.linalg.norm(x_true[10:13] - x_hat[10:13])),
                         "ba_err": float(np.linalg.norm(x_true[13:16] - x_hat[13:16])),
                         "p_err": float(np.linalg.norm(x_true[0:3] - x_hat[0:3])),
                         "v_err": float(np.linalg.norm(x_true[3:6] - x_hat[3:6]))},
    }


def main() -> None:
    t0 = time.perf_counter()
    data.truth_flight()
    tasks = [(s, k) for s in SCENARIOS for k in data.SEEDS]
    with mp.get_context("fork").Pool(len(tasks)) as pool:
        res = pool.map(_seed, tasks, chunksize=1)
    out = {"per_seed": res, "mean": {}}
    for s in SCENARIOS:
        rr = [r for r in res if r["scenario"] == s]
        m = {k: float(np.mean([r[k] for r in rr])) for k in ("kf_rms", "T_rms", "kf_onset_term_rms", "ekf_run_rms")}
        m["ekf_rms"] = {v: float(np.mean([r["ekf_rms"][v] for r in rr])) for v in VARIANTS}
        m["onset_errors"] = {k: float(np.mean([r["onset_errors"][k] for r in rr])) for k in rr[0]["onset_errors"]}
        m["max_ekf_c0_repro_m"] = max(r["ekf_c0_repro_m"] for r in rr)
        m["max_kf_split_residual_m"] = max(r["kf_split_residual_m"] for r in rr)
        m["window_s"] = rr[0]["window_s"]
        out["mean"][s] = m
    c, a = out["mean"]["outage_40s_calm"], out["mean"]["outage_40s_aggressive"]
    valid = max(c["max_ekf_c0_repro_m"], a["max_ekf_c0_repro_m"]) <= REPRO_TOL_M
    rules = {
        "1_kf_T_ratio_calm": c["T_rms"] / c["kf_rms"], "1_kf_T_ratio_aggressive": a["T_rms"] / a["kf_rms"],
        "2_T_aggressive_minus_calm": a["T_rms"] - c["T_rms"],
        "3_ekf_C1_over_C0_calm": c["ekf_rms"]["C1"] / c["ekf_rms"]["C0"],
        "4_ekf_att_bias_part_calm": c["ekf_rms"]["C0"] - c["ekf_rms"]["C1"],
        "4_ekf_att_bias_part_aggressive": a["ekf_rms"]["C0"] - a["ekf_rms"]["C1"],
    }
    passed = {
        "1": all(0.75 <= rules[k] <= 1.25 for k in ("1_kf_T_ratio_calm", "1_kf_T_ratio_aggressive")),
        "2": rules["2_T_aggressive_minus_calm"] > 0,
        "3": rules["3_ekf_C1_over_C0_calm"] <= 0.25,
        "4": rules["4_ekf_att_bias_part_calm"] > rules["4_ekf_att_bias_part_aggressive"],
    }
    out.update(valid=valid, rules=rules, passed=passed,
               outcome=("invalid" if not valid else "Verified" if all(passed.values()) else
                        "Rejected (failing: " + ", ".join(k for k, v in passed.items() if not v) + ")"),
               wall_s=time.perf_counter() - t0)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "outage_diagnostic.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("valid", "rules", "passed", "outcome", "wall_s")}, indent=2))
    print(json.dumps(out["mean"], indent=2))


if __name__ == "__main__":
    main()
