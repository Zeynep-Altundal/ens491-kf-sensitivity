"""Training of the learned-R candidates under the fixed budget of ``README.md`` (Part 1).

Six configurations per candidate (GRU, MLP): learning rate {1e-3, 3e-3, 1e-2} x weight decay
{0, 1e-4}; Adam, float64, batch 32, 400 iterations, gradient-norm clipping at 1.0, initialisation
with ``torch.manual_seed(0)``. The pooled validation NLL is evaluated every 25 iterations; the
checkpoint with the lowest value is kept, and the configuration with the lowest value is selected
per candidate. The iteration-0 validation value is recorded for the curves only.

Usage (from the repository root)::

    python -m experiments.neural_r.train
"""

from __future__ import annotations

import csv
import itertools
import json
import time
from multiprocessing import get_context
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from . import data
from .dkf import MODELS, as_tensor, innovation_nll, run_kf

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
FIGURES_DIR = HERE / "figures"
CKPT_DIR = RESULTS_DIR / "checkpoints"
ITERATIONS = 400
VAL_EVERY = 25
CLIP = 1.0
LEARNING_RATES = (1e-3, 3e-3, 1e-2)
WEIGHT_DECAYS = (0.0, 1e-4)
CONFIGS = [{"kind": k, "lr": lr, "wd": wd, "name": f"{k}_lr{lr:g}_wd{wd:g}"}
           for k, lr, wd in itertools.product(("gru", "mlp"), LEARNING_RATES, WEIGHT_DECAYS)]


def validation_tensors():
    ys = [data.validation_set(p)[2] for p in data.VAL_PATTERNS]
    return as_tensor(np.concatenate(ys, axis=0))


def validation_nll(net, y_val) -> float:
    model = data.model()
    with torch.no_grad():
        o = run_kf(model, y_val, net=net)
        return float(innovation_nll(o["nu"], o["S"]).mean())


def train_config(cfg: dict) -> dict:
    torch.set_num_threads(1)
    model = data.model()
    y_val = validation_tensors()
    torch.manual_seed(0)
    net = MODELS[cfg["kind"]](model.m)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    curve = [{"iteration": 0, "train_nll": float("nan"), "val_nll": validation_nll(net, y_val)}]
    best = (float("inf"), None, None)
    t0 = time.perf_counter()
    status = "completed"
    for it in range(ITERATIONS):
        _, _, y = data.training_batch(it)
        o = run_kf(model, as_tensor(y), net=net)
        loss = innovation_nll(o["nu"], o["S"]).mean()
        if not torch.isfinite(loss):
            status = f"non-finite training loss at iteration {it + 1}"
            break
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP)
        opt.step()
        row = {"iteration": it + 1, "train_nll": float(loss), "val_nll": float("nan")}
        if (it + 1) % VAL_EVERY == 0:
            v = validation_nll(net, y_val)
            row["val_nll"] = v
            if v < best[0]:
                best = (v, it + 1, {k: t.clone() for k, t in net.state_dict().items()})
        curve.append(row)
    seconds = time.perf_counter() - t0
    if best[2] is not None:
        torch.save(best[2], CKPT_DIR / f"{cfg['name']}.pt")
    return {**cfg, "status": status, "best_val_nll": best[0], "best_iteration": best[1],
            "train_seconds": seconds, "n_parameters": sum(p.numel() for p in net.parameters()),
            "curve": curve}


def plot_curves(results: list[dict]) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True, sharey="row")
    for col, kind in enumerate(("gru", "mlp")):
        for r in [r for r in results if r["kind"] == kind]:
            it = np.array([c["iteration"] for c in r["curve"]])
            tr = np.array([c["train_nll"] for c in r["curve"]])
            va = np.array([c["val_nll"] for c in r["curve"]])
            line, = axes[0, col].plot(it, tr, lw=0.8, label=f"lr {r['lr']:g}, wd {r['wd']:g}")
            m = np.isfinite(va)
            axes[1, col].plot(it[m], va[m], "o-", ms=3, lw=1, color=line.get_color(),
                              label=f"lr {r['lr']:g}, wd {r['wd']:g}")
        axes[0, col].set_title(f"{kind.upper()}: training NLL (per batch)")
        axes[1, col].set_title(f"{kind.upper()}: validation NLL (pooled, 400 runs)")
        axes[1, col].set_xlabel("iteration")
        for ax in axes[:, col]:
            ax.legend(fontsize=7)
            ax.set_ylabel("innovation NLL per step")
    fig.savefig(FIGURES_DIR / "training_curves.png", dpi=120)
    plt.close(fig)


def main() -> None:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    with get_context("spawn").Pool(len(CONFIGS)) as pool:
        results = pool.map(train_config, CONFIGS)
    with (RESULTS_DIR / "training_curves.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "iteration", "train_nll", "val_nll"])
        for r in results:
            for c in r["curve"]:
                w.writerow([r["name"], c["iteration"], f"{c['train_nll']:.10g}",
                            f"{c['val_nll']:.10g}"])
    selected = {}
    for kind in ("gru", "mlp"):
        ok = [r for r in results if r["kind"] == kind and r["best_iteration"] is not None]
        selected[kind] = min(ok, key=lambda r: r["best_val_nll"])["name"] if ok else None
    with (RESULTS_DIR / "training_summary.csv").open("w", newline="") as fh:
        keys = ["name", "kind", "lr", "wd", "status", "best_val_nll", "best_iteration",
                "train_seconds", "n_parameters", "selected"]
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in results:
            w.writerow({**{k: r[k] for k in keys[:-1]}, "selected": r["name"] in selected.values()})
    (RESULTS_DIR / "selected_models.json").write_text(json.dumps(selected, indent=2) + "\n")
    plot_curves(results)
    for r in results:
        print(f"{r['name']:24} {r['status']:10} best val {r['best_val_nll']:.5f} "
              f"at {r['best_iteration']}  {r['train_seconds']:.0f} s  {r['n_parameters']} params")
    print("selected:", selected)


if __name__ == "__main__":
    main()
