"""Statistical closure for the filter-family study.

Implements the "Statistical closure protocol" in ``experiments/filter_family/README.md``.

Usage (from the repository root)::

    python -m experiments.filter_family.closure
"""

from __future__ import annotations

import csv
from itertools import combinations
from pathlib import Path

import numpy as np

from experiments.common.bootstrap import (
    bootstrap,
    mean_stat,
    paired_difference,
    rms_stat,
    supported_difference,
)

from .models import range_bearing_model, simulate
from .sweep import (
    DIVERGENCE_FACTOR,
    FILTER_NAMES,
    N_RUNS,
    RESULTS_DIR,
    STEPS,
    Setting,
    _make_filter,
    _mistuning,
    _true,
)
from .filters import run_filter

MASTER_SEEDS = (0, 1, 2, 3, 4)
SETTINGS = {
    "nominal": lambda: _true(range_bearing_model()),
    "initial std 40 m": lambda: _true(range_bearing_model(p0_diag=(1600.0, 1600.0, 1.0, 1.0))),
    "bearing noise 0.5 rad": lambda: _true(range_bearing_model(sigma_bearing=0.5)),
    "R x 100": lambda: _mistuning("R")(100.0),
    "Q x 0.01": lambda: _mistuning("Q")(0.01),
    "P0 x 100": lambda: _mistuning("P0")(100.0),
}


def per_run(st: Setting, seed: int) -> dict[str, dict[str, np.ndarray]]:
    """Per filter: per-run position MSE, mean NEES, mean NIS and a divergence flag."""
    m = st.true_model
    x_true, y = simulate(m, STEPS, N_RUNS, np.random.default_rng(seed))
    sig_r, sig_b = np.sqrt(m.R[0, 0]), np.sqrt(m.R[1, 1])
    r_T = np.hypot(x_true[:, -1, 0], x_true[:, -1, 1])
    e_fix = np.sqrt(sig_r**2 + (r_T * sig_b) ** 2)
    out = {}
    for name in FILTER_NAMES:
        mse = np.full(N_RUNS, np.nan)
        nees = np.full(N_RUNS, np.nan)
        nis = np.full(N_RUNS, np.nan)
        div = np.zeros(N_RUNS, dtype=bool)
        for i in range(N_RUNS):
            try:
                with np.errstate(all="ignore"):
                    o = run_filter(_make_filter(name, m, st), y[i])
                    e = x_true[i, 1:] - o["x"]
                    finite = np.all(np.isfinite(o["x"])) and np.all(np.isfinite(o["P"]))
                    if not finite or np.linalg.norm(e[-1, :2]) > DIVERGENCE_FACTOR * e_fix[i]:
                        raise FloatingPointError
                    nees[i] = np.mean(np.einsum("ti,tij,tj->t", e, np.linalg.inv(o["P"]), e))
                    nis[i] = np.mean(np.einsum("ti,tij,tj->t", o["nu"], np.linalg.inv(o["S"]), o["nu"]))
                    mse[i] = np.mean(np.sum(e[:, :2] ** 2, axis=1))
            except (np.linalg.LinAlgError, FloatingPointError):
                div[i] = True
        out[name] = {"pos_mse": mse, "nees": nees, "nis": nis, "diverged": div}
    return out


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
    summary, paired = [], []
    for sname, make in SETTINGS.items():
        d = per_run(make(), MASTER_SEEDS[0])
        for f in FILTER_NAMES:
            ok = ~d[f]["diverged"]
            row = {"setting": sname, "filter": f, "n_diverged": int((~ok).sum())}
            for metric, stat, key in (("pos_rmse", rms_stat, "pos_mse"),
                                      ("mean_nees", mean_stat, "nees"),
                                      ("mean_nis", mean_stat, "nis")):
                est, lo, hi = bootstrap(stat, d[f][key][ok])
                row.update({metric: est, f"{metric}_lo": lo, f"{metric}_hi": hi})
            summary.append(row)
        any_div = np.any([d[f]["diverged"] for f in FILTER_NAMES], axis=0)
        keep = ~any_div
        for a, b in combinations(FILTER_NAMES, 2):
            for metric, stat, key in (("pos_rmse", rms_stat, "pos_mse"), ("mean_nees", mean_stat, "nees")):
                est, lo, hi = paired_difference(stat, d[a][key][keep], d[b][key][keep])
                paired.append({"setting": sname, "a": a, "b": b, "metric": metric,
                               "a_minus_b": est, "ci_low": lo, "ci_high": hi,
                               "supported": supported_difference(lo, hi),
                               "runs_used": int(keep.sum())})
    write_rows(summary, RESULTS_DIR / "closure_summary.csv")
    write_rows(paired, RESULTS_DIR / "closure_paired.csv")

    seeds = []
    for seed in MASTER_SEEDS:
        d = per_run(SETTINGS["nominal"](), seed)
        for f in FILTER_NAMES:
            ok = ~d[f]["diverged"]
            seeds.append({"master_seed": seed, "filter": f, "n_diverged": int((~ok).sum()),
                          "pos_rmse": rms_stat(d[f]["pos_mse"][ok]),
                          "mean_nees": mean_stat(d[f]["nees"][ok]),
                          "mean_nis": mean_stat(d[f]["nis"][ok])})
    write_rows(seeds, RESULTS_DIR / "closure_seeds.csv")

    print("== summary: estimate [95 % CI] ==")
    for r in summary:
        print(f"{r['setting']:22} {r['filter']:12} div {r['n_diverged']:2d}  "
              f"RMSE {r['pos_rmse']:8.4f} [{r['pos_rmse_lo']:.4f}, {r['pos_rmse_hi']:.4f}]  "
              f"NEES {r['mean_nees']:9.3f} [{r['mean_nees_lo']:.3f}, {r['mean_nees_hi']:.3f}]  "
              f"NIS {r['mean_nis']:7.3f} [{r['mean_nis_lo']:.3f}, {r['mean_nis_hi']:.3f}]")
    print("== paired differences (A - B) ==")
    for r in paired:
        print(f"{r['setting']:22} {r['a']:11} - {r['b']:11} {r['metric']:9} {r['a_minus_b']:+10.4f} "
              f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] supported: {r['supported']} (runs {r['runs_used']})")
    print("== master seeds (nominal) ==")
    for f in FILTER_NAMES:
        rows = [r for r in seeds if r["filter"] == f]
        print(f"{f:12} RMSE " + ", ".join(f"{r['pos_rmse']:.4f}" for r in rows)
              + " | NEES " + ", ".join(f"{r['mean_nees']:.3f}" for r in rows)
              + " | NIS " + ", ".join(f"{r['mean_nis']:.3f}" for r in rows)
              + " | diverged " + ", ".join(str(r["n_diverged"]) for r in rows))


if __name__ == "__main__":
    main()
