"""Statistical closure for the ESKF sensitivity study.

Implements the "Statistical closure protocol" in
``experiments/eskf_sensitivity/README.md``.

Usage (from the repository root)::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.eskf_sensitivity.closure
"""

from __future__ import annotations

import csv
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

from experiments.common.bootstrap import (
    bootstrap,
    mean_stat,
    paired_difference,
    rms_stat,
    supported_difference,
)

from .run import RESULTS_DIR
from .sweep import N_RUNS, SWEEPS, Spec, run_one

MASTER_SEEDS = (0, 1, 2, 3, 4)


def settings() -> dict[str, Spec]:
    out = {"nominal": Spec()}
    for sw in SWEEPS:
        for value in (sw.grid[0], sw.grid[-1]):
            for variant in sw.variants:
                v = value if isinstance(value, str) else f"{value:g}"
                label = f"{sw.name} {v}" + (f" ({variant})" if variant else "")
                out[label] = sw.specs[(value, variant)]
    return out


def _task(args):
    label, spec, run_index = args
    return label, run_index, run_one(spec, run_index)


def per_run_arrays(results: dict[int, dict], indices) -> dict[str, np.ndarray]:
    keys = ("pos_mse", "vel_mse", "att_mse", "nees", "nis", "bg_final", "ba_final")
    arr = {k: np.full(len(indices), np.nan) for k in keys}
    for j, i in enumerate(indices):
        r = results[i]
        if not r["ok"]:
            continue
        arr["pos_mse"][j] = r["pos2"] / r["n"]
        arr["vel_mse"][j] = r["vel2"] / r["n"]
        arr["att_mse"][j] = r["att2"] / r["n"]
        arr["nees"][j] = r["nees"] / r["n"]
        arr["nis"][j] = r["nis"] / r["n_acc"] if r["n_acc"] else np.nan
        arr["bg_final"][j] = r["bg_final"]
        arr["ba_final"][j] = r["ba_final"]
    return arr


def write_rows(rows: list[dict], path: Path) -> None:
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.10g}" if isinstance(v, float) else v) for k, v in r.items()})


def main() -> None:
    specs = settings()
    base = list(range(N_RUNS))
    tasks = [(label, spec, i) for label, spec in specs.items() for i in base]
    for m in MASTER_SEEDS[1:]:
        tasks += [(f"nominal seed {m}", Spec(), 1000 * m + i) for i in base]
    workers = max(1, (os.cpu_count() or 2) - 2)
    print(f"{len(tasks)} runs on {workers} processes")
    raw: dict[str, dict[int, dict]] = {}
    with mp.Pool(workers) as pool:
        for label, i, res in pool.imap_unordered(_task, tasks, chunksize=4):
            raw.setdefault(label, {})[i] = res

    data = {label: per_run_arrays(raw[label], base) for label in specs}
    summary, paired = [], []
    nom = data["nominal"]
    for label, d in data.items():
        ok = np.isfinite(d["pos_mse"])
        row = {"setting": label, "n_failed": int((~ok).sum())}
        for metric, stat, key, scale in (("pos_rmse", rms_stat, "pos_mse", 1.0),
                                         ("vel_rmse", rms_stat, "vel_mse", 1.0),
                                         ("att_rmse_deg", rms_stat, "att_mse", np.rad2deg(1.0)),
                                         ("mean_nees_pos", mean_stat, "nees", 1.0),
                                         ("mean_nis", mean_stat, "nis", 1.0),
                                         ("final_bg_err_rms", rms_stat, "bg_final", 1.0),
                                         ("final_ba_err_rms", rms_stat, "ba_final", 1.0)):
            x = d[key][ok] ** 2 if key in ("bg_final", "ba_final") else d[key][ok]
            est, lo, hi = bootstrap(stat, x)
            row.update({metric: scale * est, f"{metric}_lo": scale * lo, f"{metric}_hi": scale * hi})
        summary.append(row)
        if label == "nominal":
            continue
        keep = ok & np.isfinite(nom["pos_mse"])
        for metric, stat, key in (("pos_rmse", rms_stat, "pos_mse"), ("mean_nees_pos", mean_stat, "nees")):
            est, lo, hi = paired_difference(stat, d[key][keep], nom[key][keep])
            paired.append({"setting": label, "metric": metric, "minus_nominal": est,
                           "ci_low": lo, "ci_high": hi, "supported": supported_difference(lo, hi),
                           "runs_used": int(keep.sum())})
    write_rows(summary, RESULTS_DIR / "closure_summary.csv")
    write_rows(paired, RESULTS_DIR / "closure_paired.csv")

    seeds = []
    for m in MASTER_SEEDS:
        label = "nominal" if m == 0 else f"nominal seed {m}"
        idx = [1000 * m + i for i in base]
        d = per_run_arrays(raw[label], idx)
        seeds.append({"master_seed": m, "pos_rmse": rms_stat(d["pos_mse"]),
                      "att_rmse_deg": float(np.rad2deg(rms_stat(d["att_mse"]))),
                      "mean_nees_pos": mean_stat(d["nees"]), "mean_nis": mean_stat(d["nis"])})
    write_rows(seeds, RESULTS_DIR / "closure_seeds.csv")

    print("== summary: estimate [95 % CI] ==")
    for r in summary:
        print(f"{r['setting']:34} fail {r['n_failed']}  pos {r['pos_rmse']:7.3f} "
              f"[{r['pos_rmse_lo']:.3f}, {r['pos_rmse_hi']:.3f}]  att {r['att_rmse_deg']:6.2f} "
              f"[{r['att_rmse_deg_lo']:.2f}, {r['att_rmse_deg_hi']:.2f}]  NEES {r['mean_nees_pos']:10.3f} "
              f"[{r['mean_nees_pos_lo']:.3f}, {r['mean_nees_pos_hi']:.3f}]  NIS {r['mean_nis']:8.3f} "
              f"[{r['mean_nis_lo']:.3f}, {r['mean_nis_hi']:.3f}]")
    print("== paired differences to nominal (setting - nominal) ==")
    for r in paired:
        print(f"{r['setting']:34} {r['metric']:13} {r['minus_nominal']:+12.4f} "
              f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] supported: {r['supported']}")
    print("== master seeds (nominal) ==")
    for r in seeds:
        print(f"seed {r['master_seed']}: pos RMSE {r['pos_rmse']:.4f}, att RMSE {r['att_rmse_deg']:.3f} deg, "
              f"NEES {r['mean_nees_pos']:.4f}, NIS {r['mean_nis']:.4f}")


if __name__ == "__main__":
    main()
