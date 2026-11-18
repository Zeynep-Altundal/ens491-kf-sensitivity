"""Diagnosis of the adaptive-R filter under a constant true measurement covariance.

Implements the "Adaptive-R diagnosis protocol" in ``experiments/filter_family/README.md``.
The helpers ``run_recorded``, ``per_step_metrics`` and ``diverged`` are shared with
``time_varying_r.py``.

Usage (from the repository root)::

    python -m experiments.filter_family.adaptive_r
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.common.bootstrap import bootstrap, mean_stat, paired_difference, rms_stat, \
    supported_difference

from .filters import AdaptiveEKF, Filter
from .models import StateSpaceModel, linear_cv_model, range_bearing_model, simulate

N_RUNS = 100
SEED = 0
WINDOWS = (5, 10, 20, 50, 100, 200)
DIVERGENCE_FACTOR = 10.0
HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
FIGURES_DIR = HERE / "figures"

# name: (model factory, steps, first steady-state step index (0-based) or None)
CONFIGS = {
    "linear_T400": (linear_cv_model, 400, 200),
    "rb_T400": (range_bearing_model, 400, 200),
    "rb_T60": (range_bearing_model, 60, None),
}

POS_IDX = {"linear_cv": [0], "range_bearing": [0, 1]}


def run_recorded(filt: Filter, y: np.ndarray, R_seq: np.ndarray | None = None) -> dict:
    """Run predict/update over one trajectory and record the quantities used in the diagnosis.

    If ``R_seq`` (T, m, m) is given, the filter's R is set to ``R_seq[k]`` before update k
    (true-covariance filter). Recorded per step: posterior mean and covariance, innovation and its
    covariance, the R used at the update, the post-fit residual ``e = residual(z, h(x+))`` and
    ``H(x+) P+ H(x+)^T``.
    """
    model = filt.model
    T, m, n = y.shape[0], model.m, model.n
    out = {k: np.empty(s) for k, s in (
        ("x", (T, n)), ("P", (T, n, n)), ("nu", (T, m)), ("S", (T, m, m)),
        ("R_used", (T, m, m)), ("e", (T, m)), ("HPH_post", (T, m, m)))}
    for k in range(T):
        filt.predict()
        if R_seq is not None:
            filt.R = R_seq[k].copy()
        out["R_used"][k] = filt.R
        res = filt.update(y[k])
        out["x"][k], out["P"][k] = res.x, res.P
        out["nu"][k], out["S"][k] = res.innovation, res.innovation_cov
        H = model.H_jacobian(res.x)
        out["e"][k] = model.residual(y[k], model.h(res.x))
        out["HPH_post"][k] = H @ res.P @ H.T
    return out


def per_step_metrics(model: StateSpaceModel, x_true: np.ndarray, o: dict) -> dict:
    """Per-step squared position error, NEES, NIS and Gaussian innovation NLL of one run."""
    e = x_true - o["x"]
    pos = POS_IDX[model.name]
    nees = np.einsum("ti,tij,tj->t", e, np.linalg.inv(o["P"]), e)
    nis = np.einsum("ti,tij,tj->t", o["nu"], np.linalg.inv(o["S"]), o["nu"])
    _, logdet = np.linalg.slogdet(o["S"])
    nll = 0.5 * (nis + logdet + model.m * np.log(2.0 * np.pi))
    return {"pos_se": np.sum(e[:, pos] ** 2, axis=1), "nees": nees, "nis": nis, "nll": nll}


def single_fix_error(model: StateSpaceModel, x_final: np.ndarray, R_final: np.ndarray) -> float:
    """Error of a single measurement fix at the final step (divergence reference)."""
    if model.name == "linear_cv":
        return float(np.sqrt(R_final[0, 0]))
    r_T = np.hypot(x_final[0], x_final[1])
    return float(np.sqrt(R_final[0, 0] + r_T**2 * R_final[1, 1]))


def diverged(model: StateSpaceModel, x_true: np.ndarray, o: dict, R_final: np.ndarray) -> bool:
    if not (np.all(np.isfinite(o["x"])) and np.all(np.isfinite(o["P"]))):
        return True
    pos = POS_IDX[model.name]
    err = np.linalg.norm(x_true[-1, pos] - o["x"][-1, pos])
    return bool(err > DIVERGENCE_FACTOR * single_fix_error(model, x_true[-1], R_final))


def safe_run(make_filter, model, x_true, y, R_final, R_seq=None):
    """Run one filter on one trajectory; returns (outputs or None, diverged flag)."""
    try:
        with np.errstate(all="ignore"):
            o = run_recorded(make_filter(), y, R_seq)
            if diverged(model, x_true, o, R_final):
                return None, True
            o.update(per_step_metrics(model, x_true, o))
            return o, False
    except (np.linalg.LinAlgError, FloatingPointError):
        return None, True


def windowed_estimate(e: np.ndarray, hph: np.ndarray, W: int, sign: float) -> np.ndarray:
    """Windowed estimate after each step k >= W-1: (1/W) sum e e^T + sign * hph[k]; NaN before."""
    T, m = e.shape
    out = np.full((T, m, m), np.nan)
    outer = np.einsum("ti,tj->tij", e, e)
    c = np.cumsum(outer, axis=0)
    for k in range(W - 1, T):
        s = c[k] - (c[k - W] if k >= W else 0.0)
        R = s / W + sign * hph[k]
        out[k] = 0.5 * (R + R.T)
    return out


def write_rows(rows: list[dict], path: Path) -> None:
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.10g}" if isinstance(v, float) else v) for k, v in r.items()})


def _segments(T: int, ss: int | None) -> dict[str, slice]:
    seg = {"all": slice(0, T)}
    if ss is not None:
        seg["steady"] = slice(ss, T)
    return seg


def run_config(name: str) -> dict:
    factory, T, ss = CONFIGS[name]
    model = factory()
    x, y = simulate(model, T, N_RUNS, np.random.default_rng(SEED))
    x_true = x[:, 1:]
    variants = {"off": lambda: AdaptiveEKF(model, model.x0, model.P0, adapt=False)}
    for W in WINDOWS:
        variants[f"W{W}"] = (lambda W=W: AdaptiveEKF(model, model.x0, model.P0, window=W))
    runs = {v: [] for v in variants}
    div = {v: np.zeros(N_RUNS, dtype=bool) for v in variants}
    for v, make in variants.items():
        for i in range(N_RUNS):
            o, d = safe_run(make, model, x_true[i], y[i], model.R)
            runs[v].append(o)
            div[v][i] = d
    return {"model": model, "T": T, "ss": ss, "runs": runs, "div": div}


def _per_run(runs: list, key: str, seg: slice, reduce) -> np.ndarray:
    return np.array([reduce(o[key][seg]) if o is not None else np.nan for o in runs])


def analyse(name: str, d: dict, tables: dict) -> None:
    model, T, ss, runs, div = d["model"], d["T"], d["ss"], d["runs"], d["div"]
    variants = list(runs)
    any_div = np.any([div[v] for v in variants], axis=0)
    keep = ~any_div
    m = model.m
    for sname, seg in _segments(T, ss).items():
        per = {v: {"pos_mse": _per_run(runs[v], "pos_se", seg, np.mean),
                   "nees": _per_run(runs[v], "nees", seg, np.mean),
                   "nis": _per_run(runs[v], "nis", seg, np.mean)} for v in variants}
        metrics = (("pos_rmse", rms_stat, "pos_mse"), ("mean_nees", mean_stat, "nees"),
                   ("mean_nis", mean_stat, "nis"))
        for v in variants:
            ok = ~div[v]
            row = {"config": name, "segment": sname, "variant": v, "n_diverged": int(div[v].sum()),
                   "expected_nees": model.n, "expected_nis": m}
            for metric, stat, key in metrics:
                est, lo, hi = bootstrap(stat, per[v][key][ok])
                row.update({metric: est, f"{metric}_lo": lo, f"{metric}_hi": hi})
            tables["summary"].append(row)
        for v in variants[1:]:
            for metric, stat, key in metrics:
                est, lo, hi = paired_difference(stat, per[v][key][keep], per["off"][key][keep])
                tables["paired"].append({"config": name, "segment": sname, "variant": v,
                                         "metric": metric, "on_minus_off": est, "ci_low": lo,
                                         "ci_high": hi, "supported": supported_difference(lo, hi),
                                         "runs_used": int(keep.sum())})
        primary = (sname == "steady") or (ss is None)
        for a, b in zip(variants[1:-1], variants[2:]):
            est, lo, hi = paired_difference(mean_stat, per[b]["nees"][keep], per[a]["nees"][keep])
            tables["pairs"].append({"config": name, "segment": sname, "primary": primary,
                                    "larger_W": b, "smaller_W": a,
                                    "nees_larger_minus_smaller": est, "ci_low": lo, "ci_high": hi,
                                    "decrease_supported": hi < 0.0,
                                    "runs_used": int(keep.sum())})

    # (c) estimated R over time and steady-state bias; (d) PSD checks.
    r_true = np.diag(model.R)
    off_runs = runs["off"]
    for v in variants[1:]:
        W = int(v[1:])
        ok_idx = [i for i in range(N_RUNS) if keep[i]]
        Rh = np.array([runs[v][i]["R_used"] for i in ok_idx])  # (N, T, m, m)
        ratio = np.einsum("ntii->nti", Rh) / r_true
        for k in range(T):
            for j in range(m):
                col = ratio[:, k, j]
                tables["time"].append({"config": name, "variant": v, "step": k + 1,
                                       "element": j, "window_full": k >= W,
                                       "mean_ratio": float(col.mean()),
                                       "std_ratio": float(col.std(ddof=1)),
                                       "q05_ratio": float(np.quantile(col, 0.05)),
                                       "q95_ratio": float(np.quantile(col, 0.95))})
        # Number of updates that used an estimate (window full) and negative-eigenvalue counts.
        est_used = Rh[:, W:] if W < T else Rh[:, :0]
        n_est = int(est_used.shape[0] * est_used.shape[1])
        neg_closed = int(np.sum(np.linalg.eigvalsh(est_used)[..., 0] < 0.0)) if n_est else 0
        neg_innov, n_innov = 0, 0
        for i in ok_idx:
            o = runs[v][i]
            hph_prior = o["S"] - o["R_used"]
            C = windowed_estimate(o["nu"], hph_prior, W, -1.0)[W - 1:] if W <= T else np.empty((0, m, m))
            if len(C):
                neg_innov += int(np.sum(np.linalg.eigvalsh(C)[:, 0] < 0.0))
                n_innov += len(C)
        tables["psd"].append({"config": name, "variant": v, "updates_with_estimate": n_est,
                              "postfit_negative_eig": neg_closed,
                              "innovation_form_evaluations": n_innov,
                              "innovation_form_negative_eig": neg_innov,
                              "innovation_form_negative_fraction":
                                  (neg_innov / n_innov) if n_innov else float("nan")})
        if ss is None or W > ss:
            continue
        seg = slice(ss, T)
        # Open loop: estimator on residuals of the "off" filter; "used at k" = estimate after k-1.
        ol = []
        for i in ok_idx:
            o = off_runs[i]
            est = windowed_estimate(o["e"], o["HPH_post"], W, +1.0)
            used = np.concatenate([np.full((1, m, m), np.nan), est[:-1]])
            ol.append(used)
        ol = np.array(ol)
        for mode, arr in (("closed_loop", Rh), ("open_loop", ol)):
            for j in range(m):
                b = np.mean(arr[:, seg, j, j] / r_true[j] - 1.0, axis=1)
                est, lo, hi = bootstrap(mean_stat, b)
                tables["bias"].append({"config": name, "variant": v, "mode": mode,
                                       "quantity": f"R{j + 1}{j + 1}/Rtrue - 1",
                                       "mean": est, "ci_low": lo, "ci_high": hi,
                                       "runs_used": len(b)})
            if m == 2:
                rho = np.mean(arr[:, seg, 0, 1] / np.sqrt(arr[:, seg, 0, 0] * arr[:, seg, 1, 1]),
                              axis=1)
                est, lo, hi = bootstrap(mean_stat, rho)
                tables["bias"].append({"config": name, "variant": v, "mode": mode,
                                       "quantity": "correlation R12", "mean": est,
                                       "ci_low": lo, "ci_high": hi, "runs_used": len(rho)})


def plot(tables: dict) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    summ = tables["summary"]
    panels = (("linear_T400", "steady"), ("rb_T400", "steady"), ("rb_T60", "all"))
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), constrained_layout=True)
    for c, (cfg, seg) in enumerate(panels):
        rows = [r for r in summ if r["config"] == cfg and r["segment"] == seg]
        off = next(r for r in rows if r["variant"] == "off")
        on = [r for r in rows if r["variant"] != "off"]
        Ws = [int(r["variant"][1:]) for r in on]
        for rr, (metric, label, expected) in enumerate((
                ("pos_rmse", "position RMSE", None), ("mean_nees", "mean NEES", "expected_nees"),
                ("mean_nis", "mean NIS", "expected_nis"))):
            ax = axes[rr, c]
            est = np.array([r[metric] for r in on])
            lo = np.array([r[f"{metric}_lo"] for r in on])
            hi = np.array([r[f"{metric}_hi"] for r in on])
            ax.errorbar(Ws, est, yerr=[est - lo, hi - est], fmt="o-", capsize=3, label="adaptive")
            ax.axhspan(off[f"{metric}_lo"], off[f"{metric}_hi"], color="0.8", alpha=0.6)
            ax.axhline(off[metric], color="0.4", lw=1, label="adaptation off")
            if expected:
                ax.axhline(off[expected], color="k", ls="--", lw=1, label="expected")
            ax.set_xscale("log")
            ax.set_xlabel("window W")
            ax.set_ylabel(label)
            if rr == 0:
                ax.set_title(f"{cfg} ({seg} steps)")
            if rr == 1 and c == 0:
                ax.legend(fontsize=8)
    fig.savefig(FIGURES_DIR / "adaptive_r_window.png", dpi=120)
    plt.close(fig)

    for cfg in CONFIGS:
        rows = [r for r in tables["time"] if r["config"] == cfg]
        elements = sorted({r["element"] for r in rows})
        fig, axes = plt.subplots(len(elements), 1, figsize=(10, 3.5 * len(elements)),
                                 constrained_layout=True, squeeze=False)
        for j in elements:
            ax = axes[j, 0]
            for v in [f"W{W}" for W in WINDOWS]:
                rr = [r for r in rows if r["variant"] == v and r["element"] == j]
                k = np.array([r["step"] for r in rr])
                mu = np.array([r["mean_ratio"] for r in rr])
                line, = ax.plot(k, mu, lw=1.2, label=v)
                ax.fill_between(k, [r["q05_ratio"] for r in rr], [r["q95_ratio"] for r in rr],
                                color=line.get_color(), alpha=0.12)
            ax.axhline(1.0, color="k", ls="--", lw=1)
            ax.set_xlabel("step k")
            ax.set_ylabel(f"R_hat_{j + 1}{j + 1} / R_true_{j + 1}{j + 1}")
            ax.set_title(f"{cfg}: R used at each update, mean and 5-95 % band over runs")
            ax.legend(fontsize=8, ncol=6)
        fig.savefig(FIGURES_DIR / f"adaptive_r_estimate_{cfg}.png", dpi=120)
        plt.close(fig)


def main() -> None:
    tables = {k: [] for k in ("summary", "paired", "pairs", "bias", "psd", "time")}
    for name in CONFIGS:
        print(f"running {name}", flush=True)
        analyse(name, run_config(name), tables)
    for key, fname in (("summary", "adaptive_r_summary"), ("paired", "adaptive_r_paired"),
                       ("pairs", "adaptive_r_window_pairs"), ("bias", "adaptive_r_bias"),
                       ("psd", "adaptive_r_psd"), ("time", "adaptive_r_time")):
        write_rows(tables[key], RESULTS_DIR / f"{fname}.csv")
    plot(tables)
    print("done")


if __name__ == "__main__":
    main()
