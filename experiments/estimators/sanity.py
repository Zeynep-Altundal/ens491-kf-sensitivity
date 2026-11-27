"""Sanity checks on the nominal recording before the sweep (protocol: ``README.md``).

    .venv/bin/python -m experiments.estimators.sanity      (from the repository root)
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
from scipy.stats import chi2

from experiments.estimators import data
from experiments.estimators.filters import KF
from experiments.estimators.run import position_epochs, run
from experiments.linear_kf.kf import kalman_filter
from synthetic.rotation import quat_conjugate, quat_log, quat_multiply

RESULTS = Path(__file__).resolve().parent / "results"
TOL_P, TOL_V, TOL_ATT_DEG = 0.05, 0.05, 0.2
FRACTION = 0.90


ALL = ("KF", "EKF", "ESKF", "own_KF", "own_EKF", "own_ESKF")


def _att_deg(qa, qb):
    return np.degrees(np.linalg.norm(quat_log(quat_multiply(quat_conjugate(qa), qb)), axis=1))


def _diff(a, b) -> dict:
    d = {"dp": float(np.abs(a["p_hat"] - b["p_hat"]).max()), "dv": float(np.abs(a["v_hat"] - b["v_hat"]).max()),
         "mean_nees": [float(a["nees"].mean()), float(b["nees"].mean())],
         "max_abs_dnees": float(np.abs(a["nees"] - b["nees"]).max())}
    if a["nees_dim"] == 15:
        d["datt_deg"] = float(_att_deg(a["q_hat"], b["q_hat"]).max())
    return d


def _seed_runs(seed: int) -> dict:
    rec = data.nominal_recording(seed)
    tuning = data.tuning_from(data.nominal_recording(0)) if seed else data.tuning_from(rec)
    outs = {name: run(name, rec, tuning) for name in ALL}
    k = position_epochs(rec["imu_t"], rec["pos_t"])
    F, Q = KF.transition(float(rec["pos_t"][1] - rec["pos_t"][0]), tuning.q_cv)
    H = np.hstack([np.eye(3), np.zeros((3, 3))])
    f0 = KF(data.initial(rec), tuning)
    f0.update_position(rec["pos_z"][0], rec["pos_R"][0])
    ref = kalman_filter(rec["pos_z"][1:], F, H, Q, rec["pos_R"][0], f0.x, f0.P)
    kf_ref = {n: {"dp": float(np.abs(outs[n]["p_hat"][k[1:]] - ref.estimates[:, :3]).max()),
                  "dv": float(np.abs(outs[n]["v_hat"][k[1:]] - ref.estimates[:, 3:]).max())} for n in ("KF", "own_KF")}
    return {
        "seed": seed, "all_valid": bool(rec["pos_valid"].all()),
        "check1": {"own": _diff(outs["own_EKF"], outs["own_ESKF"]), "companion": _diff(outs["EKF"], outs["ESKF"])},
        "own_vs_companion": {n: _diff(outs[f"own_{n}"], outs[n]) for n in ("KF", "EKF", "ESKF")},
        "kf_ref": kf_ref,
        "nees": {n: o["nees"] for n, o in outs.items()}, "nis": {n: o["nis"] for n, o in outs.items()},
        "dims": {n: o["nees_dim"] for n, o in outs.items()},
        "runtime_s": {n: o["runtime_s"] for n, o in outs.items()},
    }


def _max(res, path):
    vals = []
    for r in res:
        x = r
        for p in path:
            x = x[p]
        vals.append(x)
    return vals


def main() -> None:
    t0 = time.perf_counter()
    data.truth_flight()  # computed once before the workers fork
    with mp.get_context("fork").Pool(len(data.SEEDS)) as pool:
        res = pool.map(_seed_runs, data.SEEDS)
    n = len(res)
    report = {"seeds": n, "check1": {}, "check2": {}, "check3": {}, "own_vs_companion": {}}
    for pair in ("own", "companion"):
        c = {k: max(_max(res, ("check1", pair, k))) for k in ("dp", "dv", "datt_deg")}
        c["pass"] = c["dp"] <= TOL_P and c["dv"] <= TOL_V and c["datt_deg"] <= TOL_ATT_DEG
        report["check1"][pair] = c
    for name in ("KF", "own_KF"):
        c = {k: max(_max(res, ("kf_ref", name, k))) for k in ("dp", "dv")}
        c["all_valid"] = all(r["all_valid"] for r in res)
        c["pass"] = c["all_valid"] and c["dp"] <= 1e-9 and c["dv"] <= 1e-9
        report["check3"][name] = c
    for name in ("KF", "EKF", "ESKF"):
        report["own_vs_companion"][name] = {
            "max_dp": max(_max(res, ("own_vs_companion", name, "dp"))),
            "max_dv": max(_max(res, ("own_vs_companion", name, "dv"))),
            "max_datt_deg": (max(_max(res, ("own_vs_companion", name, "datt_deg"))) if name != "KF" else None),
            "mean_nees_own": float(np.mean([r["own_vs_companion"][name]["mean_nees"][0] for r in res])),
            "mean_nees_companion": float(np.mean([r["own_vs_companion"][name]["mean_nees"][1] for r in res])),
            "max_abs_dnees": max(_max(res, ("own_vs_companion", name, "max_abs_dnees")))}
    for name in ALL:
        d = res[0]["dims"][name]
        nees = np.mean([r["nees"][name] for r in res], axis=0)
        nis = np.mean([r["nis"][name] for r in res], axis=0)
        lo, hi = chi2.ppf([0.025, 0.975], n * d) / n
        lo3, hi3 = chi2.ppf([0.025, 0.975], n * 3) / n
        f_nees = float(np.mean((nees >= lo) & (nees <= hi)))
        f_nis = float(np.mean((nis >= lo3) & (nis <= hi3)))
        report["check2"][name] = {
            "dim": d, "nees_bounds": [float(lo), float(hi)], "nees_fraction_inside": f_nees,
            "nees_fraction_above": float(np.mean(nees > hi)), "nees_fraction_below": float(np.mean(nees < lo)),
            "mean_nees": float(nees.mean()), "nis_fraction_inside": f_nis, "mean_nis": float(nis.mean()),
            "pass": f_nees >= FRACTION and f_nis >= FRACTION}
    report["runtime_per_run_s"] = {name: float(np.mean([r["runtime_s"][name] for r in res])) for name in ALL}
    report["wall_s"] = time.perf_counter() - t0
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "sanity.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
