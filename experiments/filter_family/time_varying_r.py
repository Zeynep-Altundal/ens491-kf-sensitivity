"""Time-varying measurement noise benchmark.

Implements the "Time-varying measurement noise benchmark protocol" in
``experiments/filter_family/README.md``: fixed-R filters (nominal R, and the time-averaged R as a
hindsight reference), the true-covariance filter given R_true(k), and the adaptive-R filter for
several window lengths, on the same data. The window of the adaptive filter is also selected on
separate validation runs by innovation NLL (amendment A3), and the identifiability of the R
estimate is analysed (amendment A5).

Usage (from the repository root)::

    python -m experiments.filter_family.time_varying_r
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.common.bootstrap import bootstrap, mean_stat, paired_difference, rms_stat, \
    supported_difference

from .adaptive_r import FIGURES_DIR, RESULTS_DIR, WINDOWS, safe_run, write_rows
from .filters import EKF, AdaptiveEKF
from .models import StateSpaceModel, linear_cv_model, range_bearing_model

N_RUNS = 100
STEPS = 400
SEED = 0
BURST_SEED = 1
BURST_FACTOR = 25.0
BURST_LEN = 10
N_BLOCKS = 4
MODELS = {"linear": linear_cv_model, "rb": range_bearing_model}
PATTERNS = ("step", "ramp", "burst")
VAL_SEED = 1000
VAL_BURST_SEED = 1001
TRUE_COV = "true_covariance"
SELECTED = "adaptive_selected"
LABELS = {"fixed_nominal": "fixed nominal R",
          "fixed_average_hindsight": "fixed average R (hindsight reference)",
          TRUE_COV: "true-covariance filter"}


def noise_scale(pattern: str, steps: int = STEPS, n_runs: int = N_RUNS,
                seed: int = BURST_SEED) -> np.ndarray:
    """Scale factors c (n_runs, steps) of R_true(k) = c(k) R_nom for k = 1..steps."""
    k = np.arange(1, steps + 1)
    if pattern == "constant":
        return np.ones((n_runs, steps))
    if pattern == "step":
        return np.tile(np.where(k <= steps // 2, 1.0, 10.0), (n_runs, 1))
    if pattern == "ramp":
        return np.tile(1.0 + 9.0 * (k - 1) / (steps - 1), (n_runs, 1))
    if pattern == "burst":
        rng = np.random.default_rng(seed)
        block = steps // N_BLOCKS
        c = np.ones((n_runs, steps))
        for i in range(n_runs):
            for b in range(N_BLOCKS):
                s = b * block + int(rng.integers(0, block - BURST_LEN + 1))
                c[i, s:s + BURST_LEN] = BURST_FACTOR
        return c
    raise ValueError(pattern)


def simulate_tv(model: StateSpaceModel, c: np.ndarray, rng: np.random.Generator):
    """As ``models.simulate`` (same draw order) with measurement noise scaled by sqrt(c)."""
    n_traj, steps = c.shape
    n, m = model.n, model.m
    x0 = model.x0 + rng.standard_normal((n_traj, n)) @ np.linalg.cholesky(model.P0).T
    w = rng.standard_normal((n_traj, steps, n)) @ np.linalg.cholesky(model.Q).T
    v = rng.standard_normal((n_traj, steps, m)) @ np.linalg.cholesky(model.R).T
    v = v * np.sqrt(c)[..., None]
    x = np.empty((n_traj, steps + 1, n))
    x[:, 0] = x0
    for k in range(steps):
        x[:, k + 1] = model.f(x[:, k]) + w[:, k]
    y = model.h(x[:, 1:]) + v
    y = model.residual(y, np.zeros_like(y))
    return x, y


def filter_specs(model: StateSpaceModel, c_bar: float):
    """name -> (factory, uses the R_true(k) sequence)."""
    specs = {
        "fixed_nominal": (lambda: EKF(model, model.x0, model.P0), False),
        "fixed_average_hindsight": (lambda: EKF(model, model.x0, model.P0, R=c_bar * model.R), False),
        "true_covariance": (lambda: EKF(model, model.x0, model.P0), True),
    }
    for W in WINDOWS:
        specs[f"adaptive_W{W}"] = (lambda W=W: AdaptiveEKF(model, model.x0, model.P0, window=W),
                                   False)
    return specs


def run_pattern(mname: str, pattern: str, seed: int = SEED, burst_seed: int = BURST_SEED,
                adaptive_only: bool = False):
    model = MODELS[mname]()
    c = noise_scale(pattern, seed=burst_seed)
    c_bar = float(np.mean(c))
    assert np.allclose(c.mean(axis=1), c_bar)
    x, y = simulate_tv(model, c, np.random.default_rng(seed))
    x_true = x[:, 1:]
    R_true = c[..., None, None] * model.R  # (N, T, m, m)
    specs = filter_specs(model, c_bar)
    if adaptive_only:
        specs = {f: v for f, v in specs.items() if f.startswith("adaptive_W")}
    runs = {f: [] for f in specs}
    div = {f: np.zeros(N_RUNS, dtype=bool) for f in specs}
    for f, (make, use_true_r) in specs.items():
        for i in range(N_RUNS):
            o, d = safe_run(make, model, x_true[i], y[i], R_true[i, -1],
                            R_seq=R_true[i] if use_true_r else None)
            if o is not None:
                lr = np.log(np.einsum("tii->ti", o["R_used"]) / np.einsum("tii->ti", R_true[i]))
                o["abs_logratio"] = np.mean(np.abs(lr), axis=1)
                o["signed_logratio"] = np.mean(lr, axis=1)
            runs[f].append(o)
            div[f][i] = d
    return model, c, c_bar, runs, div


METRICS = (("pos_rmse", rms_stat, "pos_se"), ("mean_nees", mean_stat, "nees"),
           ("mean_nis", mean_stat, "nis"), ("mean_nll", mean_stat, "nll"),
           ("r_abs_logratio", mean_stat, "abs_logratio"),
           ("r_signed_logratio", mean_stat, "signed_logratio"))
PAIRED = ("pos_rmse", "mean_nees", "mean_nis", "mean_nll")


def analyse(mname, pattern, model, c, c_bar, runs, div, tables, selected_window=None):
    filters = list(runs)
    keep = ~np.any([div[f] for f in filters], axis=0)
    per = {f: {key: np.array([np.mean(o[key]) if o is not None else np.nan for o in runs[f]])
               for _, _, key in METRICS} for f in filters}
    for f in filters:
        ok = ~div[f]
        row = {"model": mname, "pattern": pattern, "filter": f,
               "selected_window": selected_window if f == SELECTED else "", "c_bar": c_bar,
               "n_diverged": int(div[f].sum()), "expected_nees": model.n, "expected_nis": model.m}
        for metric, stat, key in METRICS:
            est, lo, hi = bootstrap(stat, per[f][key][ok])
            row.update({metric: est, f"{metric}_lo": lo, f"{metric}_hi": hi})
        tables["summary"].append(row)
    for f in filters:
        if f == TRUE_COV:
            continue
        for metric, stat, key in METRICS:
            if metric not in PAIRED:
                continue
            est, lo, hi = paired_difference(stat, per[f][key][keep], per[TRUE_COV][key][keep])
            tables["paired"].append({"model": mname, "pattern": pattern, "filter": f,
                                     "selected_window": selected_window if f == SELECTED else "",
                                     "metric": metric, "minus_true_covariance": est, "ci_low": lo,
                                     "ci_high": hi, "supported": supported_difference(lo, hi),
                                     "runs_used": int(keep.sum())})
    ok_idx = np.flatnonzero(keep)
    for f in filters:
        if f == SELECTED:
            continue
        se = np.array([runs[f][i]["pos_se"] for i in ok_idx])
        r11 = np.array([runs[f][i]["R_used"][:, 0, 0] for i in ok_idx]) / model.R[0, 0]
        run0 = runs[f][0]["R_used"][:, 0, 0] / model.R[0, 0] if runs[f][0] is not None \
            else np.full(STEPS, np.nan)
        for k in range(STEPS):
            tables["time"].append({"model": mname, "pattern": pattern, "filter": f, "step": k + 1,
                                   "pos_rmse": float(np.sqrt(se[:, k].mean())),
                                   "mean_R11_over_Rnom": float(r11[:, k].mean()),
                                   "mean_c_true": float(c[ok_idx, k].mean()),
                                   "run0_R11_over_Rnom": float(run0[k]),
                                   "run0_c_true": float(c[0, k])})


def plot(tables):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    shown = ("fixed_nominal", "fixed_average_hindsight", TRUE_COV, "adaptive_W5", "adaptive_W20",
             "adaptive_W100")
    for mname in MODELS:
        for pattern in PATTERNS:
            rows = [r for r in tables["time"] if r["model"] == mname and r["pattern"] == pattern]
            fig, axes = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True, sharex=True)
            for ci, f in enumerate(shown):
                rr = [r for r in rows if r["filter"] == f]
                k = [r["step"] for r in rr]
                color = "k" if f == TRUE_COV else f"C{ci}"
                if f != TRUE_COV:
                    axes[0].plot(k, [r["run0_R11_over_Rnom"] for r in rr], lw=1, color=color,
                                 label=LABELS.get(f, f))
                    axes[1].plot(k, [r["mean_R11_over_Rnom"] for r in rr], lw=1, color=color,
                                 label=LABELS.get(f, f))
                axes[2].plot(k, [r["pos_rmse"] for r in rr], lw=1, color=color,
                             label=LABELS.get(f, f))
            rr = [r for r in rows if r["filter"] == TRUE_COV]
            k = [r["step"] for r in rr]
            axes[0].plot(k, [r["run0_c_true"] for r in rr], "k--", lw=1.2, label="R_true(k)")
            axes[1].plot(k, [r["mean_c_true"] for r in rr], "k--", lw=1.2, label="R_true(k)")
            axes[0].set_ylabel("R_11 / R_nom_11 (run 0)")
            axes[1].set_ylabel("R_11 / R_nom_11 (mean over runs)")
            axes[2].set_ylabel("position RMSE over runs")
            axes[2].set_xlabel("step k")
            for ax in axes[:2]:
                ax.set_yscale("log")
            axes[0].legend(fontsize=8, ncol=4)
            axes[2].legend(fontsize=8, ncol=3)
            axes[0].set_title(f"{mname}, pattern {pattern}: filter R (first diagonal element) "
                              "and position RMSE")
            fig.savefig(FIGURES_DIR / f"tv_r_{mname}_{pattern}.png", dpi=120)
            plt.close(fig)


def select_window(mname: str, pattern: str, tables: dict) -> int:
    """Amendment A3: window with the lowest mean innovation NLL on the validation runs."""
    _, _, _, runs, div = run_pattern(mname, pattern, seed=VAL_SEED, burst_seed=VAL_BURST_SEED,
                                     adaptive_only=True)
    rows = []
    for W in WINDOWS:
        f = f"adaptive_W{W}"
        nll = np.array([np.mean(o["nll"]) for o, d in zip(runs[f], div[f]) if not d])
        est, lo, hi = bootstrap(mean_stat, nll)
        rows.append({"model": mname, "pattern": pattern, "window": W,
                     "val_mean_nll": est, "val_mean_nll_lo": lo, "val_mean_nll_hi": hi,
                     "val_n_diverged": int(div[f].sum())})
    best = min(rows, key=lambda r: r["val_mean_nll"])["window"]
    for r in rows:
        r["selected"] = r["window"] == best
    tables["selection"].extend(rows)
    return best


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.ptp(a) == 0.0 or np.ptp(b) == 0.0:  # constant series: correlation undefined
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def identifiability(mname, pattern, model, c, runs, div, tables) -> None:
    """Amendment A5: does the estimated R follow R_true(k) or the H P+ H^T term?

    The R used at update k (k >= W) is ``R_hat(k) = A(k) + B(k)`` with the residual part
    ``A(k) = (1/W) sum e e^T`` and ``B(k) = H P+ H^T`` of update k - 1. Per run and diagonal
    element, over the steps that use an estimate, the temporal variance of ``R_hat`` splits
    exactly into ``Cov(R_hat, A) + Cov(R_hat, B)``; the share of B is reported, together with the
    correlations of ``log R_hat`` with ``log R_true`` and with ``log B`` and the mean of ``B / R_hat``.
    """
    for W in WINDOWS:
        f = f"adaptive_W{W}"
        ok = [i for i in range(N_RUNS) if not div[f][i]]
        for j in range(model.m):
            share_b, corr_true, corr_b, frac_b = [], [], [], []
            for i in ok:
                o = runs[f][i]
                Rh = o["R_used"][W:, j, j]
                B = o["HPH_post"][W - 1:-1, j, j]
                A = Rh - B
                rt = c[i, W:] * model.R[j, j]
                var = np.var(Rh)
                share_b.append(np.mean((Rh - Rh.mean()) * (B - B.mean())) / var)
                corr_true.append(_corr(np.log(Rh), np.log(rt)))
                corr_b.append(_corr(np.log(Rh), np.log(B)))
                frac_b.append(np.mean(B / Rh))
            row = {"model": mname, "pattern": pattern, "window": W, "element": j + 1,
                   "runs_used": len(ok)}
            for name, vals in (("var_share_hph", share_b), ("corr_log_Rtrue", corr_true),
                               ("corr_log_hph", corr_b), ("mean_hph_fraction", frac_b)):
                v = np.array(vals)
                if np.all(np.isnan(v)):
                    row.update({name: float("nan"), f"{name}_lo": float("nan"),
                                f"{name}_hi": float("nan")})
                    continue
                est, lo, hi = bootstrap(mean_stat, v[~np.isnan(v)])
                row.update({name: est, f"{name}_lo": lo, f"{name}_hi": hi})
            sh_lo, sh_hi = row["var_share_hph_lo"], row["var_share_hph_hi"]
            ct_lo = row["corr_log_Rtrue_lo"]
            if sh_hi < 0.5 and ct_lo > 0.0:
                verdict = "follows R_true"
            elif sh_lo > 0.5:
                verdict = "follows H P H^T"
            else:
                verdict = "not separated"
            row["verdict"] = verdict
            tables["ident"].append(row)


def main() -> None:
    tables = {"summary": [], "paired": [], "time": [], "selection": [], "ident": []}
    for mname in MODELS:
        for pattern in PATTERNS:
            print(f"validation {mname} {pattern}", flush=True)
            W_sel = select_window(mname, pattern, tables)
            print(f"running {mname} {pattern} (selected W = {W_sel})", flush=True)
            model, c, c_bar, runs, div = run_pattern(mname, pattern)
            identifiability(mname, pattern, model, c, runs, div, tables)
            runs[SELECTED], div[SELECTED] = runs[f"adaptive_W{W_sel}"], div[f"adaptive_W{W_sel}"]
            analyse(mname, pattern, model, c, c_bar, runs, div, tables, selected_window=W_sel)
        print(f"control {mname} constant", flush=True)
        model, c, _, runs, div = run_pattern(mname, "constant", adaptive_only=True)
        identifiability(mname, "constant", model, c, runs, div, tables)
    write_rows(tables["summary"], RESULTS_DIR / "tv_r_summary.csv")
    write_rows(tables["paired"], RESULTS_DIR / "tv_r_paired.csv")
    write_rows(tables["time"], RESULTS_DIR / "tv_r_time.csv")
    write_rows(tables["selection"], RESULTS_DIR / "tv_r_window_selection.csv")
    write_rows(tables["ident"], RESULTS_DIR / "tv_r_identifiability.csv")
    plot(tables)
    print("done")


if __name__ == "__main__":
    main()
