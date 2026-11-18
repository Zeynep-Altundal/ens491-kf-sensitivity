"""Evaluation of the selected learned-R models against the baselines; gates G1 to G6.

Implements the evaluation of ``experiments/neural_r/README.md`` on the linear model. Baseline
filters are the validated ones of ``experiments/filter_family`` (imported unchanged).

Usage (from the repository root)::

    python -m experiments.neural_r.evaluate
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.common.bootstrap import bootstrap, mean_stat, paired_difference, rms_stat, \
    supported_difference
from experiments.filter_family.adaptive_r import diverged, per_step_metrics, safe_run, write_rows
from experiments.filter_family.filters import EKF, AdaptiveEKF

from . import data
from .dkf import MODELS, as_tensor, run_kf

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
FIGURES_DIR = HERE / "figures"
WINDOWS = (5, 10, 20, 50, 100, 200)
LEARNED = ("gru", "mlp")
METRICS = (("pos_rmse", rms_stat, "pos_se"), ("mean_nees", mean_stat, "nees"),
           ("mean_nis", mean_stat, "nis"), ("mean_nll", mean_stat, "nll"),
           ("r_abs_logratio", mean_stat, "abs_logratio"),
           ("recovery_time", mean_stat, "recovery"))
LN2 = float(np.log(2.0))


def recovery_time(c: np.ndarray, R_used: np.ndarray, R_true: np.ndarray):
    """Mean recovery time over the bursts of one run and the number of censored bursts.

    For a burst ending at index k_e (c[k_e] > 1, c[k_e + 1] = 1), the smallest j >= 1 with
    |ln(R_11(k_e + j) / R_true_11(k_e + j))| <= ln 2, censored at the next burst start or the end.
    """
    T = len(c)
    hi = c > 1.0
    ends = [k for k in range(T - 1) if hi[k] and not hi[k + 1]]
    taus, censored = [], 0
    for ke in ends:
        nxt = next((k for k in range(ke + 1, T) if hi[k]), T)
        avail = nxt - ke - 1
        if avail <= 0:
            continue
        lr = np.abs(np.log(R_used[ke + 1:nxt, 0, 0] / R_true[ke + 1:nxt, 0, 0]))
        hit = np.flatnonzero(lr <= LN2)
        if len(hit):
            taus.append(hit[0] + 1)
        else:
            taus.append(avail)
            censored += 1
    return (float(np.mean(taus)) if taus else float("nan")), censored, len(taus)


def _finish(o, model, xt, c_i, Rt_i, burst: bool):
    o.update(per_step_metrics(model, xt, o))
    lr = np.log(np.einsum("tii->ti", o["R_used"]) / np.einsum("tii->ti", Rt_i))
    o["abs_logratio"] = np.mean(np.abs(lr), axis=1)
    if burst:
        o["recovery"], o["censored"], o["n_bursts"] = recovery_time(c_i, o["R_used"], Rt_i)
    else:
        o["recovery"], o["censored"], o["n_bursts"] = float("nan"), 0, 0
    return o


def load_learned() -> dict:
    selected = json.loads((RESULTS_DIR / "selected_models.json").read_text())
    nets = {}
    for kind in LEARNED:
        if selected.get(kind) is None:
            continue
        torch.manual_seed(0)
        net = MODELS[kind](1)
        net.load_state_dict(torch.load(RESULTS_DIR / "checkpoints" / f"{selected[kind]}.pt"))
        net.eval()
        nets[kind] = net
    return nets, selected


def select_adaptive(tables: dict) -> dict:
    """Six windows evaluated on the validation runs; selection by mean innovation NLL."""
    model = data.model()
    per = {}
    for pattern in data.VAL_PATTERNS:
        c, x, y = data.validation_set(pattern)
        Rt = c[..., None, None] * model.R
        for W in WINDOWS:
            vals = []
            for i in range(data.N_RUNS):
                o, d = safe_run(lambda: AdaptiveEKF(model, model.x0, model.P0, window=W), model,
                                x[i, 1:], y[i], Rt[i, -1])
                vals.append(np.nan if d else float(np.mean(o["nll"])))
            per[(pattern, W)] = np.array(vals)
    for W in WINDOWS:
        pooled = np.concatenate([per[(p, W)] for p in data.VAL_PATTERNS])
        tables["adaptive_selection"].append({
            "window": W, "val_nll_burst": float(np.nanmean(per[("burst", W)])),
            "val_nll_pooled": float(np.nanmean(pooled))})
    rows = tables["adaptive_selection"]
    sel = {"burst": min(rows, key=lambda r: r["val_nll_burst"])["window"],
           "pooled": min(rows, key=lambda r: r["val_nll_pooled"])["window"]}
    for r in rows:
        r["selected_burst"] = r["window"] == sel["burst"]
        r["selected_pooled"] = r["window"] == sel["pooled"]
    return sel


def evaluate_set(name: str, nets: dict, sel: dict, tables: dict, traces: dict) -> None:
    model = data.model()
    c, x, y = data.test_set(name)
    xt = x[:, 1:]
    Rt = c[..., None, None] * model.R
    burst = name in data.BURST_SETS
    N = data.N_RUNS
    specs = {
        "fixed_nominal": (lambda i: EKF(model, model.x0, model.P0), False),
        "fixed_average_hindsight": (lambda i: EKF(model, model.x0, model.P0,
                                                  R=float(np.mean(c[i])) * model.R), False),
        "true_covariance": (lambda i: EKF(model, model.x0, model.P0), True),
        "adaptive_burst_sel": (lambda i: AdaptiveEKF(model, model.x0, model.P0,
                                                     window=sel["burst"]), False),
        "adaptive_pooled_sel": (lambda i: AdaptiveEKF(model, model.x0, model.P0,
                                                      window=sel["pooled"]), False),
    }
    runs = {}
    for f, (make, use_true) in specs.items():
        runs[f] = []
        for i in range(N):
            o, d = safe_run(lambda: make(i), model, xt[i], y[i], Rt[i, -1],
                            R_seq=Rt[i] if use_true else None)
            runs[f].append(None if d else _finish(o, model, xt[i], c[i], Rt[i], burst))
    for kind, net in nets.items():
        with torch.no_grad():
            out = run_kf(model, as_tensor(y), net=net)
        arr = {k: v.numpy() for k, v in out.items()}
        runs[kind] = []
        for i in range(N):
            o = {"x": arr["x"][i], "P": arr["P"][i], "nu": arr["nu"][i], "S": arr["S"][i],
                 "R_used": arr["R"][i]}
            ok = np.all(np.isfinite(o["x"])) and not diverged(model, xt[i], o, Rt[i, -1])
            runs[kind].append(_finish(o, model, xt[i], c[i], Rt[i], burst) if ok else None)
    filters = list(runs)
    div = {f: np.array([o is None for o in runs[f]]) for f in filters}
    keep = ~np.any([div[f] for f in filters], axis=0)
    per = {f: {key: np.array([np.mean(o[key]) if o is not None else np.nan for o in runs[f]])
               for _, _, key in METRICS} for f in filters}
    for f in filters:
        ok = ~div[f]
        row = {"test_set": name, "filter": f, "n_diverged": int(div[f].sum()),
               "window": sel["burst"] if f == "adaptive_burst_sel" else
               (sel["pooled"] if f == "adaptive_pooled_sel" else "")}
        for metric, stat, key in METRICS:
            v = per[f][key][ok]
            if np.all(np.isnan(v)):
                row.update({metric: float("nan"), f"{metric}_lo": float("nan"),
                            f"{metric}_hi": float("nan")})
                continue
            est, lo, hi = bootstrap(stat, v)
            row.update({metric: est, f"{metric}_lo": lo, f"{metric}_hi": hi})
        if burst:
            cens = sum(o["censored"] for o in runs[f] if o is not None)
            tot = sum(o["n_bursts"] for o in runs[f] if o is not None)
            row["recovery_censored_fraction"] = cens / tot if tot else float("nan")
        tables["summary"].append(row)
    pairs = [(a, b) for a in nets for b in filters if b not in LEARNED]
    if "gru" in nets and "mlp" in nets:
        pairs.append(("gru", "mlp"))
    for a, b in pairs:
        for metric, stat, key in METRICS:
            if key == "recovery" and not burst:
                continue
            est, lo, hi = paired_difference(stat, per[a][key][keep], per[b][key][keep])
            tables["paired"].append({"test_set": name, "a": a, "b": b, "metric": metric,
                                     "a_minus_b": est, "ci_low": lo, "ci_high": hi,
                                     "supported": supported_difference(lo, hi),
                                     "runs_used": int(keep.sum())})
    traces[name] = {"c": c[0], "R": {f: runs[f][0]["R_used"][:, 0, 0] / model.R[0, 0]
                                     for f in filters if runs[f][0] is not None},
                    "rmse": {f: np.sqrt(np.mean([runs[f][i]["pos_se"] for i in np.flatnonzero(keep)],
                                                axis=0)) for f in filters}}


def gates(tables: dict, nets: dict) -> None:
    P = {(r["test_set"], r["a"], r["b"], r["metric"]): r for r in tables["paired"]}
    S = {(r["test_set"], r["filter"]): r for r in tables["summary"]}
    for kind in nets:
        g1 = [P[(s, kind, "adaptive_burst_sel", "mean_nll")] for s in data.HELD_OUT_BURSTS]
        passed = all(r["ci_high"] < 0.0 for r in g1)
        tables["gates"].append({"gate": "G1", "model": kind, "outcome": "passed" if passed else "failed",
                                "detail": "; ".join(f"{r['test_set']}: {r['a_minus_b']:+.4f} "
                                                    f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
                                                    for r in g1)})
        worse = []
        for s in data.TEST_SETS:
            me = S[(s, kind)]
            for f in ("fixed_nominal", "fixed_average_hindsight", "true_covariance",
                      "adaptive_burst_sel", "adaptive_pooled_sel"):
                b = S[(s, f)]
                for metric, exp in (("mean_nees", 2.0), ("mean_nis", 1.0)):
                    if abs(me[metric] - exp) > abs(b[metric] - exp):
                        worse.append(f"{s}/{metric} vs {f}")
        tables["gates"].append({"gate": "G2", "model": kind, "outcome": "passed (reported)",
                                "detail": f"{len(worse)} cases farther from the expected value "
                                          "than a baseline (listed in neural_r_g2_cases.csv)"})
        for w in worse:
            s, rest = w.split("/", 1)
            metric, f = rest.split(" vs ")
            tables["g2"].append({"model": kind, "test_set": s, "metric": metric, "baseline": f,
                                 "learned": S[(s, kind)][metric], "baseline_value": S[(s, f)][metric]})
        g3 = {m: P[("constant", kind, "fixed_nominal", m)] for m in ("pos_rmse", "mean_nees", "mean_nis")}
        ok3 = (not g3["pos_rmse"]["ci_low"] > 0.0) and all(
            g3[m]["ci_low"] <= 0.0 <= g3[m]["ci_high"] for m in ("mean_nees", "mean_nis"))
        tables["gates"].append({"gate": "G3", "model": kind, "outcome": "passed" if ok3 else "failed",
                                "detail": "; ".join(f"{m}: {r['a_minus_b']:+.4f} [{r['ci_low']:+.4f}, "
                                                    f"{r['ci_high']:+.4f}]" for m, r in g3.items())})
        tables["gates"].append({"gate": "G4", "model": kind, "outcome": "evaluated",
                                "detail": "R tracking error per test set in neural_r_summary.csv "
                                          "and paired differences in neural_r_paired.csv"})
        tables["gates"].append({"gate": "G5", "model": kind, "outcome": "evaluated",
                                "detail": "; ".join(f"{s}: {S[(s, kind)]['recovery_time']:.2f} steps "
                                                    f"(censored {S[(s, kind)]['recovery_censored_fraction']:.2f})"
                                                    for s in data.BURST_SETS)})
    if "gru" in nets and "mlp" in nets:
        g6 = [P[(s, "gru", "mlp", "mean_nll")] for s in data.TEST_SETS]
        contrib = [r["test_set"] for r in g6 if r["ci_high"] < 0.0]
        tables["gates"].append({"gate": "G6", "model": "gru vs mlp", "outcome": "evaluated",
                                "detail": "memory contributes (GRU - MLP NLL interval below zero) on: "
                                          + (", ".join(contrib) if contrib else "none")})


def plot(traces: dict, tables: dict) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    shown = ("true_covariance", "adaptive_burst_sel", "gru", "mlp")
    colors = {"true_covariance": "k", "adaptive_burst_sel": "C1", "gru": "C0", "mlp": "C2",
              "fixed_nominal": "C7", "fixed_average_hindsight": "C5", "adaptive_pooled_sel": "C3"}
    for name, tr in traces.items():
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True, sharex=True)
        k = np.arange(1, len(tr["c"]) + 1)
        for f in shown:
            if f in tr["R"]:
                axes[0].plot(k, tr["R"][f], lw=1, color=colors[f],
                             ls="--" if f == "true_covariance" else "-",
                             label="R_true(k) (true-covariance filter)" if f == "true_covariance" else f)
        for f, v in tr["rmse"].items():
            axes[1].plot(k, v, lw=1, color=colors[f], label=f)
        axes[0].set_yscale("log")
        axes[0].set_ylabel("R / R_base (run 0)")
        axes[1].set_ylabel("position RMSE over runs")
        axes[1].set_xlabel("step k")
        axes[0].legend(fontsize=8, ncol=2)
        axes[1].legend(fontsize=7, ncol=3)
        axes[0].set_title(f"test set {name}")
        fig.savefig(FIGURES_DIR / f"eval_{name}.png", dpi=120)
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for ax, (b, title) in zip(axes, (("adaptive_burst_sel", "learned - adaptive (burst-selected W)"),
                                     ("fixed_nominal", "learned - fixed nominal R"),
                                     ("true_covariance", "learned - true-covariance filter"))):
        for off, kind in ((-0.15, "gru"), (0.15, "mlp")):
            rr = [r for r in tables["paired"] if r["a"] == kind and r["b"] == b
                  and r["metric"] == "mean_nll"]
            if not rr:
                continue
            xs = np.arange(len(rr)) + off
            est = np.array([r["a_minus_b"] for r in rr])
            ax.errorbar(xs, est, yerr=[est - [r["ci_low"] for r in rr],
                                       [r["ci_high"] for r in rr] - est],
                        fmt="o", capsize=3, color=colors[kind], label=kind)
            ax.set_xticks(np.arange(len(rr)), [r["test_set"] for r in rr], rotation=45, ha="right",
                          fontsize=8)
        ax.axhline(0.0, color="k", lw=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("paired difference in mean innovation NLL")
        ax.legend(fontsize=8)
    fig.savefig(FIGURES_DIR / "eval_nll_differences.png", dpi=120)
    plt.close(fig)


def main() -> None:
    torch.set_num_threads(4)
    nets, selected = load_learned()
    tables = {k: [] for k in ("adaptive_selection", "summary", "paired", "gates", "g2")}
    sel = select_adaptive(tables)
    print("adaptive selection:", sel, "learned:", selected, flush=True)
    traces = {}
    for name in data.TEST_SETS:
        print("evaluating", name, flush=True)
        evaluate_set(name, nets, sel, tables, traces)
    gates(tables, nets)
    for key, fname in (("adaptive_selection", "neural_r_adaptive_selection"),
                       ("summary", "neural_r_summary"), ("paired", "neural_r_paired"),
                       ("gates", "neural_r_gates"), ("g2", "neural_r_g2_cases")):
        if tables[key]:
            write_rows(tables[key], RESULTS_DIR / f"{fname}.csv")
    plot(traces, tables)
    for g in tables["gates"]:
        print(f"{g['gate']} {g['model']:10} {g['outcome']:18} {g['detail']}")


if __name__ == "__main__":
    main()
