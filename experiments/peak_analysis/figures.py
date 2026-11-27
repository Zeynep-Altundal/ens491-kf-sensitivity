"""Figures of the peak analysis (protocol: ``README.md``, "Figures").

    .venv/bin/python -m experiments.peak_analysis.figures      (after ``analyze``)

Writes PNG files to ``build/`` (git-ignored).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from experiments.estimators import data  # noqa: E402
from experiments.peak_analysis.analyze import ESTIMATORS, RESULTS, label, load  # noqa: E402

BUILD = Path(__file__).resolve().parent / "build"
SCENARIOS = ("accel_bias_step_large", "gyro_bias_step_small_onset_aggressive",
             "imu_noise_burst_x10_onset_figure_eight", "pos_spike_single", "outage_40s_calm",
             "outage_40s_aggressive")
# Categorical slots 1-3 of the reference palette (blue, orange, aqua); text in neutral inks.
COLORS = {"KF": "#257de0", "EKF": "#f47230", "ESKF": "#00b894"}
TEXT, TEXT2, GRID = "#191919", "#666666", "#e4e4e4"
CORRUPTION = "#6a4ce2"
SEQ = ["#ffffff", "#cce3fd", "#9dc8f7", "#6aabf1", "#338ced", "#2370cd", "#1757ac", "#0e3d7e"]


def _style(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(TEXT2)
    ax.tick_params(colors=TEXT2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _phase_bands(ax, t, labels) -> None:
    ch = np.r_[0, np.nonzero(labels[1:] != labels[:-1])[0] + 1, len(t)]
    for i, (a, b) in enumerate(zip(ch[:-1], ch[1:])):
        if i % 2:
            ax.axvspan(t[a], t[b - 1], color="#f4f4f4", zorder=0, linewidth=0)
        name = str(labels[a]).replace("_", " ")
        ax.text((t[a] + t[b - 1]) / 2, 1.01, name, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=7, color=TEXT2, rotation=0 if t[b - 1] - t[a] > 6 else 90)


def _corruption(ax, events) -> None:
    for e in events:
        if e["type"] == "spike":
            for x in e["times"]:
                ax.axvline(x, color=CORRUPTION, linewidth=1.2, alpha=0.6, zorder=1)
            continue
        t1 = e["t_end"] if e["t_end"] is not None else ax.get_xlim()[1]
        ax.axvspan(e["t_start"], t1, color=CORRUPTION, alpha=0.10, zorder=1, linewidth=0)


def error_figure(scen: str, labels: np.ndarray, events: list[dict]) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=150)
    _style(ax)
    t = None
    for e in ESTIMATORS:
        s_runs, n_runs = [], []
        for seed in data.SEEDS:
            t, es = load(scen, e, seed)
            s_runs.append(es)
            n_runs.append(load("nominal", e, seed)[1])
        lw = 1.2 if e == "ESKF" else 2.0
        ax.plot(t, np.mean(n_runs, 0), color=COLORS[e], linewidth=1.0, linestyle=(0, (2, 2)), alpha=0.8, zorder=2)
        ax.plot(t, np.mean(s_runs, 0), color=COLORS[e], linewidth=lw, label=label(e), zorder=3)
    ax.set_xlim(t[0], t[-1])
    _phase_bands(ax, t, labels)
    _corruption(ax, events)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("time [s]", color=TEXT, fontsize=10)
    ax.set_ylabel("position error norm [m]", color=TEXT, fontsize=10)
    ax.set_title(f"{scen}: mean over {len(data.SEEDS)} seeds (solid), nominal run (dashed), corruption shaded",
                 color=TEXT, fontsize=10, loc="left", pad=50)
    leg = ax.legend(frameon=False, fontsize=9, loc="upper left")
    for txt in leg.get_texts():
        txt.set_color(TEXT)
    fig.tight_layout()
    BUILD.mkdir(exist_ok=True)
    out = BUILD / f"error_{scen}.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def heatmap() -> Path:
    cnt = defaultdict(int)
    scen = []
    with (RESULTS / "run_peaks.csv").open() as fh:
        for r in csv.DictReader(fh):
            cnt[(r["scenario"], r["estimator"])] += int(r["has_peak"])
            if r["scenario"] not in scen:
                scen.append(r["scenario"])
    M = np.array([[cnt[(s, e)] for e in ESTIMATORS] for s in scen], float)
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ)
    fig, ax = plt.subplots(figsize=(6.2, 10.5), dpi=150)
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=len(data.SEEDS), aspect="auto")
    ax.set_xticks(range(len(ESTIMATORS)), [label(e) for e in ESTIMATORS], fontsize=9, color=TEXT)
    ax.set_yticks(range(len(scen)), scen, fontsize=8, color=TEXT)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(ESTIMATORS)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(scen)), minor=True)
    ax.grid(which="minor", color="#ffffff", linewidth=2)
    ax.tick_params(which="minor", length=0)
    for i in range(len(scen)):
        for j in range(len(ESTIMATORS)):
            v = int(M[i, j])
            ax.text(j, i, str(v), ha="center", va="center", fontsize=8, color="#ffffff" if v >= 6 else TEXT)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("seeds with a peak (of 10)", color=TEXT, fontsize=9)
    cb.outline.set_visible(False)
    ax.set_title("Seeds with an error peak, per scenario and estimator", color=TEXT, fontsize=10, loc="left")
    fig.tight_layout()
    BUILD.mkdir(exist_ok=True)
    out = BUILD / "disagreement_heatmap.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    labels = data.phase_labels()
    nominal = data.nominal_recording(0)
    for s in SCENARIOS:
        print(error_figure(s, labels, data.scenario_recording(nominal, s)["events"]))
    print(heatmap())


if __name__ == "__main__":
    main()
