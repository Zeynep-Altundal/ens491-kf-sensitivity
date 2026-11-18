"""Inspection of the saturation of the learned R (``README.md``, Amendment 1).

Forward passes of the saved, selected checkpoints only; no training, no selection, no new test
metric. Writes ``results/saturation_*.csv`` and ``figures/saturation_*.png``.

Usage (from the repository root, with PyTorch installed)::

    python -m experiments.neural_r.inspect_saturation
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.filter_family.adaptive_r import write_rows

from . import data
from .dkf import EPS, MODELS, as_tensor, run_kf
from .train import ITERATIONS

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
FIGURES_DIR = HERE / "figures"
QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
BINS = ((1.0, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, 30.0 + 1e-9))
SETTLE = 10


def load_selected() -> dict:
    selected = json.loads((RESULTS_DIR / "selected_models.json").read_text())
    nets = {}
    for kind, name in selected.items():
        torch.manual_seed(0)
        net = MODELS[kind](1)
        net.load_state_dict(torch.load(RESULTS_DIR / "checkpoints" / f"{name}.pt"))
        net.eval()
        nets[kind] = net
    return nets, selected


def output_layer(net):
    return net.head if net.kind == "gru" else net.net[-1]


def architectural_bound(net) -> dict:
    """Upper bound of alpha implied by inputs in (-1, 1) to the linear output layer."""
    layer = output_layer(net)
    w = layer.weight.detach().numpy()[0]
    b = float(layer.bias.detach().numpy()[0])
    logit_max = float(np.sum(np.abs(w)) + b)
    return {"sum_abs_w": float(np.sum(np.abs(w))), "bias": b, "logit_max": logit_max,
            "alpha_max": float(np.log1p(np.exp(logit_max)) + EPS)}


def alpha_and_logit(net, y: np.ndarray):
    """alpha (B, T) and the output-layer pre-activation (B, T) for measurements y (B, T, 1)."""
    logits = []
    layer = output_layer(net)
    hook = layer.register_forward_hook(lambda m, i, o: logits.append(o.detach()[:, 0]))
    try:
        with torch.no_grad():
            out = run_kf(data.model(), as_tensor(y), net=net)
    finally:
        hook.remove()
    alpha = (out["R"][..., 0, 0] / float(data.model().R[0, 0])).numpy()
    return alpha, torch.stack(logits, dim=1).numpy()


def settled_mask(c: np.ndarray) -> np.ndarray:
    """True where c has been at its current value for at least SETTLE steps."""
    age = np.zeros_like(c, dtype=int)
    for k in range(1, c.shape[1]):
        same = np.isclose(c[:, k], c[:, k - 1])
        age[:, k] = np.where(same, age[:, k - 1] + 1, 0)
    return age >= SETTLE


def factor_groups(c: np.ndarray, binned: bool):
    if binned:
        return [(f"[{lo:g}, {min(hi, 30.0):g}]", (c >= lo) & (c < hi)) for lo, hi in BINS]
    return [(f"{v:g}", np.isclose(c, v)) for v in np.unique(np.round(c, 10))]


def training_factors() -> list[dict]:
    counts: dict[float, int] = {}
    total = 0
    for it in range(ITERATIONS):
        rng = np.random.default_rng(data.TRAIN_SEED0 + it)
        c = np.stack([data._train_scale(rng) for _ in range(data.BATCH)])
        for v, n in zip(*np.unique(np.round(c, 6), return_counts=True)):
            counts[float(v)] = counts.get(float(v), 0) + int(n)
        total += c.size
    rows = [{"factor": v, "steps": n, "fraction": n / total} for v, n in sorted(counts.items())]
    return rows


def main() -> None:
    torch.set_num_threads(4)
    nets, selected = load_selected()
    bound_rows, dist_rows, factor_rows = [], [], []
    for kind, net in nets.items():
        bound_rows.append({"model": kind, "config": selected[kind], **architectural_bound(net)})
    sets = [(name, *data.test_set(name)) for name in data.TEST_SETS]
    train_c, train_y = [], []
    for it in range(10):
        c, _, y = data.training_batch(it)
        train_c.append(c)
        train_y.append(y)
    sets.append(("training_batches_0_9", np.concatenate(train_c), None, np.concatenate(train_y)))
    scatter = {}
    for kind, net in nets.items():
        attained = 0.0
        logit_attained = -np.inf
        for name, c, _, y in sets:
            alpha, logit = alpha_and_logit(net, y)
            attained = max(attained, float(alpha.max()))
            logit_attained = max(logit_attained, float(logit.max()))
            q = np.quantile(alpha, QUANTILES)
            dist_rows.append({"model": kind, "set": name, "max_true_factor": float(c.max()),
                              **{f"alpha_q{int(p * 100):02d}": float(v) for p, v in zip(QUANTILES, q)},
                              "alpha_max": float(alpha.max()), "logit_max": float(logit.max())})
            settled = settled_mask(c)
            binned = name in ("ramp", "ho_ramp", "ho_sine")
            for label, m in factor_groups(c, binned):
                for tag, mm in (("all", m), ("settled", m & settled)):
                    if not mm.any():
                        continue
                    a = alpha[mm]
                    factor_rows.append({"model": kind, "set": name, "true_factor": label,
                                        "steps": tag, "n": int(mm.sum()),
                                        "alpha_median": float(np.median(a)),
                                        "alpha_q95": float(np.quantile(a, 0.95)),
                                        "alpha_max": float(a.max())})
            scatter[(kind, name)] = (c, alpha)
        for r in bound_rows:
            if r["model"] == kind:
                r["alpha_attained_max"] = attained
                r["logit_attained_max"] = logit_attained
                r["attained_fraction_of_logit_bound"] = logit_attained / r["logit_max"]
    train_rows = training_factors()
    write_rows(bound_rows, RESULTS_DIR / "saturation_bound.csv")
    write_rows(dist_rows, RESULTS_DIR / "saturation_alpha_distribution.csv")
    write_rows(factor_rows, RESULTS_DIR / "saturation_alpha_vs_factor.csv")
    write_rows(train_rows, RESULTS_DIR / "saturation_training_factors.csv")

    # (c) curves and slopes of the selected configurations
    curves: dict[str, list] = {}
    with (RESULTS_DIR / "training_curves.csv").open() as fh:
        for r in csv.DictReader(fh):
            curves.setdefault(r["config"], []).append(r)
    slope_rows = []
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, (kind, name) in zip(axes, selected.items()):
        rows = curves[name]
        it = np.array([int(r["iteration"]) for r in rows])
        tr = np.array([float(r["train_nll"]) for r in rows])
        va = np.array([float(r["val_nll"]) for r in rows])
        m = (it >= ITERATIONS - 49) & np.isfinite(tr)
        X = np.column_stack([np.ones(m.sum()), it[m]])
        coef, res, *_ = np.linalg.lstsq(X, tr[m], rcond=None)
        dof = m.sum() - 2
        s2 = float(res[0]) / dof
        se = float(np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1]))
        mv = np.isin(it, (ITERATIONS - 50, ITERATIONS - 25, ITERATIONS)) & np.isfinite(va)
        vslope = float(np.polyfit(it[mv], va[mv], 1)[0])
        slope_rows.append({"model": kind, "config": name,
                           "train_slope_per_iteration_351_400": float(coef[1]),
                           "train_slope_se": se,
                           "val_slope_per_iteration_350_400": vslope,
                           "val_nll_350": float(va[it == ITERATIONS - 50][0]),
                           "val_nll_375": float(va[it == ITERATIONS - 25][0]),
                           "val_nll_400": float(va[it == ITERATIONS][0])})
        ax.plot(it[1:], tr[1:], lw=0.7, color="0.6", label="training NLL (per batch)")
        fv = np.isfinite(va)
        ax.plot(it[fv], va[fv], "o-", ms=3, color="C0", label="validation NLL (pooled)")
        ax.plot(it[m], X @ coef, "r-", lw=1.5, label=f"OLS slope 351..400: {coef[1]:+.2e}/it")
        ax.set_title(f"{kind.upper()} {name}")
        ax.set_xlabel("iteration")
        ax.set_ylabel("innovation NLL per step")
        ax.set_ylim(1.8, 2.6)
        ax.legend(fontsize=8)
    fig.savefig(FIGURES_DIR / "saturation_training_curves.png", dpi=120)
    plt.close(fig)
    write_rows(slope_rows, RESULTS_DIR / "saturation_curve_slopes.csv")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True, sharey=True)
    for col, name in enumerate(("burst", "ho_burst_short", "ho_burst_long")):
        for row, kind in enumerate(nets):
            c, alpha = scatter[(kind, name)]
            ax = axes[row, col]
            levels = np.unique(c)
            ax.boxplot([alpha[np.isclose(c, v) & settled_mask(c)] for v in levels],
                       positions=np.log10(levels), widths=0.12, showfliers=False)
            ax.plot(np.log10(levels), levels, "k--", lw=1, label="alpha = true factor")
            b = next(r for r in bound_rows if r["model"] == kind)
            ax.axhline(b["alpha_max"], color="r", lw=1, label="architectural bound")
            ax.set_xticks(np.log10(levels), [f"{v:g}" for v in levels])
            ax.set_yscale("log")
            ax.set_xlabel("true factor c (settled steps)")
            ax.set_ylabel("alpha")
            ax.set_title(f"{kind.upper()}, {name}")
            ax.legend(fontsize=8)
    fig.savefig(FIGURES_DIR / "saturation_alpha_vs_factor.png", dpi=120)
    plt.close(fig)
    print("bound:", bound_rows)
    print("slopes:", slope_rows)


if __name__ == "__main__":
    main()
