"""Part 0 diagnostics of the adaptive-R baseline (see ``experiments/neural_r/README.md``).

0a: control runs that separate the causes of the over-confidence of the adaptive filter with a
short window under constant R_true. 0b: the residual part and the ``H P+ H^T`` part of the
estimated R logged separately in closed loop and on a frozen (non-adaptive) filter trajectory.

The baseline filters of ``experiments/filter_family`` are imported unchanged.

Usage (from the repository root)::

    python -m experiments.neural_r.diagnostics
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.common.bootstrap import bootstrap, mean_stat, paired_difference, rms_stat, \
    supported_difference
from experiments.filter_family.adaptive_r import run_recorded, safe_run, windowed_estimate, \
    write_rows
from experiments.filter_family.filters import EKF, AdaptiveEKF
from experiments.filter_family.models import linear_cv_model, range_bearing_model, simulate
from experiments.filter_family.time_varying_r import noise_scale, simulate_tv

N_RUNS = 100
STEPS = 400
SEED = 0
BURST_SEED = 1
SS = 200  # first steady-state step index (0-based)
WINDOWS = (5, 20, 100)
MODELS = {"linear": linear_cv_model, "rb": range_bearing_model}
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _replay_sequence(est_after: np.ndarray, R0: np.ndarray, W: int) -> np.ndarray:
    """R used at update k: R0 for k < W, else the estimate formed after update k - 1."""
    T = est_after.shape[0]
    seq = np.repeat(R0[None], T, axis=0)
    seq[W:] = est_after[W - 1:T - 1]
    return seq


def overconfidence(tables: dict) -> None:
    for mname, factory in MODELS.items():
        model = factory()
        x, y = simulate(model, STEPS, N_RUNS, np.random.default_rng(SEED))
        xt = x[:, 1:]
        R = model.R
        make_ekf = lambda: EKF(model, model.x0, model.P0)
        off = [safe_run(make_ekf, model, xt[i], y[i], R)[0] for i in range(N_RUNS)]
        for W in WINDOWS:
            closed = [safe_run(lambda: AdaptiveEKF(model, model.x0, model.P0, window=W),
                               model, xt[i], y[i], R)[0] for i in range(N_RUNS)]
            open_seq = [_replay_sequence(windowed_estimate(o["e"], o["HPH_post"], W, +1.0), R, W)
                        for o in off]
            closed_seq = [o["R_used"] for o in closed]
            nxt = lambda i: (i + 1) % N_RUNS
            conds = {
                "off": off,
                "closed": closed,
                "open_replay": [safe_run(make_ekf, model, xt[i], y[i], R, R_seq=open_seq[i])[0]
                                for i in range(N_RUNS)],
                "indep_open_replay": [safe_run(make_ekf, model, xt[i], y[i], R,
                                               R_seq=open_seq[nxt(i)])[0] for i in range(N_RUNS)],
                "indep_closed_replay": [safe_run(make_ekf, model, xt[i], y[i], R,
                                                 R_seq=closed_seq[nxt(i)])[0]
                                        for i in range(N_RUNS)],
            }
            ok = np.array([all(conds[c][i] is not None for c in conds) for i in range(N_RUNS)])
            per = {c: {k: np.array([np.mean(o[k][SS:]) if o is not None else np.nan
                                    for o in runs]) for k in ("nees", "nis", "pos_se")}
                   for c, runs in conds.items()}
            for c in conds:
                row = {"model": mname, "W": W, "condition": c, "runs_used": int(ok.sum()),
                       "expected_nees": model.n, "expected_nis": model.m}
                for metric, stat, key in (("pos_rmse", rms_stat, "pos_se"),
                                          ("mean_nees", mean_stat, "nees"),
                                          ("mean_nis", mean_stat, "nis")):
                    est, lo, hi = bootstrap(stat, per[c][key][ok])
                    row.update({metric: est, f"{metric}_lo": lo, f"{metric}_hi": hi})
                tables["over"].append(row)
            for name, a, b in (("total: closed - off", "closed", "off"),
                               ("feedback: closed - open_replay", "closed", "open_replay"),
                               ("own-data correlation: open_replay - indep_open_replay",
                                "open_replay", "indep_open_replay"),
                               ("fluctuation: indep_open_replay - off", "indep_open_replay", "off"),
                               ("closed-loop statistics: indep_closed_replay - off",
                                "indep_closed_replay", "off")):
                for metric, key in (("mean_nees", "nees"), ("mean_nis", "nis")):
                    est, lo, hi = paired_difference(mean_stat, per[a][key][ok], per[b][key][ok])
                    tables["decomp"].append({"model": mname, "W": W, "term": name,
                                             "metric": metric, "difference": est, "ci_low": lo,
                                             "ci_high": hi,
                                             "supported": supported_difference(lo, hi),
                                             "runs_used": int(ok.sum())})
            for label, seqs in (("closed", closed_seq), ("open_replay", open_seq)):
                for j in range(model.m):
                    sd = np.array([np.std(np.log(s[SS:, j, j])) for s in seqs])
                    est, lo, hi = bootstrap(mean_stat, sd)
                    tables["spread"].append({"model": mname, "W": W, "estimate": label,
                                             "element": j + 1, "mean_sd_log_R": est,
                                             "ci_low": lo, "ci_high": hi})


def _cv(v: np.ndarray) -> float:
    return float(np.std(v) / np.mean(v))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.ptp(a) == 0.0 or np.ptp(b) == 0.0:  # constant series: correlation undefined
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def identifiability(tables: dict) -> None:
    for mname, factory in MODELS.items():
        model = factory()
        for pattern in ("constant", "step", "ramp", "burst"):
            c = noise_scale(pattern, seed=BURST_SEED)
            x, y = simulate_tv(model, c, np.random.default_rng(SEED))
            xt = x[:, 1:]
            Rt = c[..., None, None] * model.R
            frozen = [safe_run(lambda: EKF(model, model.x0, model.P0), model, xt[i], y[i],
                               Rt[i, -1])[0] for i in range(N_RUNS)]
            for W in WINDOWS:
                closed = [safe_run(lambda: AdaptiveEKF(model, model.x0, model.P0, window=W),
                                   model, xt[i], y[i], Rt[i, -1])[0] for i in range(N_RUNS)]
                ok = [i for i in range(N_RUNS) if closed[i] is not None and frozen[i] is not None]
                for j in range(model.m):
                    vals = {k: [] for k in ("corrA_closed", "corrA_frozen", "Bfrac_closed",
                                            "Bfrac_frozen", "cvB_closed", "cvB_frozen",
                                            "B_ratio")}
                    for i in ok:
                        parts = {}
                        for label, o in (("closed", closed[i]), ("frozen", frozen[i])):
                            est = windowed_estimate(o["e"], o["HPH_post"], W, +1.0)
                            # R used at update k (k >= W) uses the estimate after k - 1.
                            tot = est[W - 1:-1, j, j]
                            B = o["HPH_post"][W - 1:-1, j, j]
                            parts[label] = (tot - B, B)
                        rt = Rt[i, W:, j, j]
                        for label in ("closed", "frozen"):
                            A, B = parts[label]
                            vals[f"corrA_{label}"].append(_corr(np.log(A), np.log(rt)))
                            vals[f"Bfrac_{label}"].append(np.mean(B / (A + B)))
                            vals[f"cvB_{label}"].append(_cv(B))
                        vals["B_ratio"].append(np.mean(parts["closed"][1] / parts["frozen"][1]))
                    row = {"model": mname, "pattern": pattern, "W": W, "element": j + 1,
                           "runs_used": len(ok)}
                    for k, v in vals.items():
                        v = np.array(v)
                        v = v[~np.isnan(v)]
                        if len(v) == 0:
                            row.update({k: float("nan"), f"{k}_lo": float("nan"),
                                        f"{k}_hi": float("nan")})
                            continue
                        est, lo, hi = bootstrap(mean_stat, v)
                        row.update({k: est, f"{k}_lo": lo, f"{k}_hi": hi})
                    tables["ident"].append(row)


def main() -> None:
    tables = {"over": [], "decomp": [], "spread": [], "ident": []}
    print("0a over-confidence controls", flush=True)
    overconfidence(tables)
    print("0b identifiability", flush=True)
    identifiability(tables)
    write_rows(tables["over"], RESULTS_DIR / "diag_overconfidence.csv")
    write_rows(tables["decomp"], RESULTS_DIR / "diag_decomposition.csv")
    write_rows(tables["spread"], RESULTS_DIR / "diag_spread.csv")
    write_rows(tables["ident"], RESULTS_DIR / "diag_identifiability.csv")
    print("done")


if __name__ == "__main__":
    main()
