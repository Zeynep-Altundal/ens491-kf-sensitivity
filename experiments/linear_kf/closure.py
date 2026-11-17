"""Statistical closure for the linear Kalman filter study.

Implements the "Statistical closure protocol" in ``experiments/linear_kf/README.md``.

Usage (from the repository root)::

    python -m experiments.linear_kf.closure
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.stats import chi2

from experiments.common.bootstrap import (
    bootstrap,
    mean_stat,
    paired_difference,
    rms_stat,
    supported_difference,
)

from .kf import kalman_filter
from .model import DEFAULT_STEPS, constant_velocity_model, simulate
from .sweep import N_TRAJ, RESULTS_DIR, filter_matrices

MASTER_SEEDS = (0, 1, 2, 3, 4)
SETTINGS = {
    "nominal": ("Q", 1.0),
    "Q x 0.001": ("Q", 1e-3), "Q x 1000": ("Q", 1e3),
    "R x 0.001": ("R", 1e-3), "R x 1000": ("R", 1e3),
    "P0 x 0.001": ("P0", 1e-3), "P0 x 1000": ("P0", 1e3),
}
PAIRS = (("Q x 1000", "R x 0.001"), ("Q x 0.001", "R x 1000"),
         ("P0 x 0.001", "nominal"), ("P0 x 1000", "nominal"))


def per_run(model, x_true, y, sweep: str, s: float) -> dict[str, np.ndarray]:
    """Per-run time-averaged squared errors, NEES and NIS, plus per-step NEES / NIS."""
    res = kalman_filter(y, **filter_matrices(model, sweep, s))
    err = res.estimates - x_true[:, 1:, :]
    nees = np.einsum("rti,tij,rtj->rt", err, np.linalg.inv(res.covariances), err)
    nis = np.einsum("rti,tij,rtj->rt", res.innovations,
                    np.linalg.inv(res.innovation_covariances), res.innovations)
    return {
        "pos_mse": np.mean(err[..., 0] ** 2, axis=1),
        "vel_mse": np.mean(err[..., 1] ** 2, axis=1),
        "nees": nees.mean(axis=1),
        "nis": nis.mean(axis=1),
        "nees_t": nees,
        "nis_t": nis,
    }


def _ci_row(name, metric, triple, expected=None):
    est, lo, hi = triple
    row = {"setting": name, "metric": metric, "estimate": est, "ci_low": lo, "ci_high": hi}
    if expected is not None:
        row["expected"] = expected
        row["expected_inside_ci"] = lo <= expected <= hi
    return row


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
    model = constant_velocity_model()
    x_true, y = simulate(model, DEFAULT_STEPS, N_TRAJ, np.random.default_rng(MASTER_SEEDS[0]))
    data = {name: per_run(model, x_true, y, sw, s) for name, (sw, s) in SETTINGS.items()}

    summary = []
    for name, d in data.items():
        nominal = name == "nominal"
        summary.append(_ci_row(name, "pos_rmse", bootstrap(rms_stat, d["pos_mse"])))
        summary.append(_ci_row(name, "vel_rmse", bootstrap(rms_stat, d["vel_mse"])))
        summary.append(_ci_row(name, "mean_nees", bootstrap(mean_stat, d["nees"]),
                               model.n if nominal else None))
        summary.append(_ci_row(name, "mean_nis", bootstrap(mean_stat, d["nis"]),
                               model.m if nominal else None))
    write_rows(summary, RESULTS_DIR / "closure_summary.csv")

    paired = []
    for a, b in PAIRS:
        est, lo, hi = paired_difference(rms_stat, data[a]["pos_mse"], data[b]["pos_mse"])
        paired.append({"a": a, "b": b, "metric": "pos_rmse", "a_minus_b": est, "ci_low": lo,
                       "ci_high": hi, "supported": supported_difference(lo, hi)})
    write_rows(paired, RESULTS_DIR / "closure_paired.csv")

    seeds = []
    for seed in MASTER_SEEDS:
        xs, ys = simulate(model, DEFAULT_STEPS, N_TRAJ, np.random.default_rng(seed))
        d = per_run(model, xs, ys, "Q", 1.0)
        seeds.append({"master_seed": seed, "pos_rmse": rms_stat(d["pos_mse"]),
                      "vel_rmse": rms_stat(d["vel_mse"]), "mean_nees": mean_stat(d["nees"]),
                      "mean_nis": mean_stat(d["nis"])})
    write_rows(seeds, RESULTS_DIR / "closure_seeds.csv")

    # Per-step chi-square test at the nominal setting.
    d = data["nominal"]
    lo_x, hi_x = chi2.ppf([0.025, 0.975], model.n * N_TRAJ) / N_TRAJ
    lo_z, hi_z = chi2.ppf([0.025, 0.975], model.m * N_TRAJ) / N_TRAJ
    nees_k, nis_k = d["nees_t"].mean(axis=0), d["nis_t"].mean(axis=0)
    t = model.F[0, 1] * np.arange(1, DEFAULT_STEPS + 1)
    np.savetxt(RESULTS_DIR / "closure_nominal_time.csv",
               np.column_stack([t, nees_k, np.full_like(t, lo_x), np.full_like(t, hi_x),
                                nis_k, np.full_like(t, lo_z), np.full_like(t, hi_z)]),
               delimiter=",", comments="", fmt="%.10g",
               header="t,nees_avg,nees_lo,nees_hi,nis_avg,nis_lo,nis_hi")
    frac_x = float(np.mean((nees_k >= lo_x) & (nees_k <= hi_x)))
    frac_z = float(np.mean((nis_k >= lo_z) & (nis_k <= hi_z)))

    print("== summary (estimate [95 % CI]) ==")
    for r in summary:
        extra = f"  expected {r['expected']} inside CI: {r['expected_inside_ci']}" if "expected" in r else ""
        print(f"{r['setting']:12} {r['metric']:9} {r['estimate']:10.4f} "
              f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]{extra}")
    print("== paired differences (A - B) ==")
    for r in paired:
        print(f"{r['a']:11} - {r['b']:11} {r['metric']}: {r['a_minus_b']:+.4f} "
              f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] supported: {r['supported']}")
    print("== master seeds (nominal) ==")
    for r in seeds:
        print(f"seed {r['master_seed']}: pos RMSE {r['pos_rmse']:.4f}, vel RMSE {r['vel_rmse']:.4f}, "
              f"NEES {r['mean_nees']:.4f}, NIS {r['mean_nis']:.4f}")
    for key in ("pos_rmse", "mean_nees", "mean_nis"):
        vals = [r[key] for r in seeds]
        print(f"  {key}: range [{min(vals):.4f}, {max(vals):.4f}]")
    print("== per-step chi-square test (nominal) ==")
    print(f"NEES bounds [{lo_x:.4f}, {hi_x:.4f}], fraction of steps inside = {frac_x:.3f}")
    print(f"NIS  bounds [{lo_z:.4f}, {hi_z:.4f}], fraction of steps inside = {frac_z:.3f}")


if __name__ == "__main__":
    main()
