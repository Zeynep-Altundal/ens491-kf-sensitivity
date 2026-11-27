"""Peak analysis: when does which filter make a large error, and why.

Works on two kinds of run logs:
  * nav  - drone / UGV logs of `scripts/compare_filters_events --logs` (one file per filter and
           flight; the recording is regenerated from the file name, as in the analysis script);
  * knet - the KalmanNet benchmark logs of `experiments/knet_data/export_logs.py`
           (`<dataset>_<given>/<FILTER>.npz`, arrays [N, T, ...], N test trajectories).

Peak definition (one error curve e(t) >= 0 of one filter on one trajectory):
  base = median(e), s = 1.4826 * MAD(e), threshold = max(base + K * s, RATIO * base);
  a peak is a maximal run of samples with e > threshold; runs closer than MERGE samples are
  merged. Defaults K = 5, RATIO = 2, MERGE = 3.
A peak is *unique* when no other filter has a peak on the same trajectory that overlaps it
(+/- TOL samples).
Why (classical filters, from their own covariances):
  * "overconfident" - NEES at the peak maximum above the 99 % chi-square bound: wrong and unaware;
  * "surprised"     - not overconfident, but an innovation in the peak (or TOL samples before)
                      had NIS above the 99 % bound: the measurement disagreed with the prediction;
  * "aware"         - neither: the covariance had grown with the error (e.g. during an outage);
  * "no covariance" - KalmanNet.
Condition (nav only, at the peak maximum): outage, event (inside an event window), recovery
(10 s after an event), stopped, turn (|yaw rate| > 0.05 rad/s), straight.

    python -m experiments.nav_compare.peaks nav  <compare_events output dir> --out <dir>
    python -m experiments.nav_compare.peaks knet ~/data/knet_logs           --out <dir>
    python -m experiments.nav_compare.peaks curves <run_euroc out>/curves  --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import chi2

K, RATIO, MERGE, TOL = 5.0, 2.0, 3, 5
RECOVERY_S, TURN_RATE, STOP_SPEED = 10.0, 0.05, 0.1


def detect_peaks(e, k=K, ratio=RATIO, merge=MERGE):
    """Peaks of one error curve: list of (i0, i1, imax), i1 inclusive; also the threshold."""
    e = np.asarray(e, float)
    ok = np.isfinite(e)
    if ok.sum() < 5:
        return [], np.nan
    base = np.median(e[ok])
    s = 1.4826 * np.median(np.abs(e[ok] - base))
    thr = max(base + k * s, ratio * base)
    above = np.where(ok & (e > thr))[0]
    if above.size == 0:
        return [], thr
    runs, start, prev = [], above[0], above[0]
    for i in above[1:]:
        if i - prev > merge:
            runs.append((start, prev))
            start = i
        prev = i
    runs.append((start, prev))
    return [(a, b, a + int(np.argmax(e[a : b + 1]))) for a, b in runs], thr


def overlaps(p, others, tol=TOL):
    return any(not (q[1] < p[0] - tol or q[0] > p[1] + tol) for q in others)


def why(nees_max, nis_win, m, n):
    if nees_max is None or not np.isfinite(nees_max):
        return "no covariance"
    if nees_max > chi2.ppf(0.99, m):
        return "overconfident"
    if nis_win is not None and np.any(np.asarray(nis_win) > chi2.ppf(0.99, n)):
        return "surprised"
    return "aware"


def analyse(curves, m, n, cond=None):
    """curves: {filter: dict(e, nees, nis)} for one trajectory -> list of peak rows."""
    found = {f: detect_peaks(c["e"]) for f, c in curves.items()}
    rows = []
    for f, (pk, thr) in found.items():
        others = [q for g, (pq, _) in found.items() if g != f for q in pq]
        c = curves[f]
        for i0, i1, im in pk:
            nees = c.get("nees")
            nis = c.get("nis")
            nis_win = None if nis is None else nis[max(0, i0 - TOL) : i1 + 1]
            rows.append({
                "filter": f, "i0": int(i0), "i1": int(i1), "imax": int(im),
                "e_max": float(c["e"][im]), "threshold": float(thr),
                "unique": not overlaps((i0, i1), others),
                "why": why(None if nees is None else float(nees[im]), nis_win, m, n),
                "condition": cond[im] if cond is not None else "",
            })  # fmt: skip
    return rows


# ---------------------------------------------------------------- nav (drone / UGV) logs
def nav_conditions(rec, t):
    """Condition label at the times t of a run log."""
    lab = np.full(len(t), "straight", dtype=object)
    tt = rec["truth_t"]
    yaw_rate = np.interp(t, rec["imu_t"], rec["imu_gyro_clean"][:, 2])
    speed = np.linalg.norm(
        np.stack([np.interp(t, tt, rec["truth_v"][:, i]) for i in range(3)], 1), axis=1
    )
    lab[np.abs(yaw_rate) > TURN_RATE] = "turn"
    lab[speed < STOP_SPEED] = "stopped"
    for ev in rec["events"]:
        a, b = ev["t_start"], ev["t_end"]
        rec_mask = (t >= b) & (t < b + RECOVERY_S)
        lab[rec_mask] = "recovery"
        in_ev = (t >= a) & (t < b)
        lab[in_ev] = "outage" if ev["type"] == "outage" else "event"
    return lab


def nav_runs(log_dir):
    from experiments.nav_compare.compare.runner import errors
    from experiments.nav_compare.data.disturbances import (
        apply_events,
        standard_event_sets,
    )
    from experiments.nav_compare.data.recording import to_recording
    from experiments.nav_compare.data.scenarios import generate
    from experiments.nav_compare.scripts.compare_filters_events import base_scenario

    groups = defaultdict(dict)
    for p in sorted(Path(log_dir).glob("*.npz")):
        run, filt = p.stem.rsplit("_", 1)
        groups[run][filt] = p
    for run, files in groups.items():
        plat, rest = run.split("_", 1)
        set_name, seed = rest.rsplit("_s", 1)
        seed = int(seed)
        seq = generate(base_scenario(plat), seed=seed)
        evs = standard_event_sets(plat, seq.imu.t_s[-1])[set_name]
        rec = to_recording(apply_events(seq, evs, seed=seed))
        curves, t = {}, None
        for filt, path in files.items():
            lg = dict(np.load(path))
            E = errors(rec, lg)
            t = E["t"]
            nis = np.full(len(t), np.nan)
            if len(lg["upd_t"]):
                idx = np.searchsorted(t, lg["upd_t"])
                ok = idx < len(t)
                nis[idx[ok]] = lg["nis"][ok]
            curves[filt] = {"e": E["err_p"], "nees": E["nees_p"], "nis": nis}
        yield {"run": run, "platform": plat, "set": set_name, "seed": seed, "traj": 0, "t": t,
               "curves": curves, "m": 3, "n": 3, "cond": nav_conditions(rec, t)}  # fmt: skip


# ---------------------------------------------------------------- KalmanNet benchmark logs
def knet_runs(root, states=None):
    for folder in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        files = {p.stem: p for p in folder.glob("*.npz")}
        if not files:
            continue
        data = {f: dict(np.load(p, allow_pickle=False)) for f, p in files.items()}
        any_ = next(iter(data.values()))
        meta = json.loads(str(any_["meta"]))
        names = meta.get("state_names") or []
        sel = slice(None)
        if states and names:
            sel = [
                i for i, s in enumerate(names) if s.startswith(tuple(states))
            ] or slice(None)
        N, T, m = any_["err"].shape
        n = any_["innov"].shape[2]
        for j in range(N):
            curves = {}
            for f, d in data.items():
                err = d["err"][j][:, sel].astype(float)
                curves[f] = {"e": np.linalg.norm(err, axis=1),
                             "nees": d["nees"][j].astype(float), "nis": d["nis"][j].astype(float)}  # fmt: skip
            yield {"run": folder.name, "platform": meta.get("source", ""), "set": folder.name,
                   "seed": meta.get("seed", 0), "traj": j, "t": any_["t"], "curves": curves,
                   "m": m, "n": n, "cond": None}  # fmt: skip


# ---------------------------------------------------------------- generic error-curve bundles
def save_curves(path, t, curves, cond, platform="", set_name=""):
    """One trajectory, several filters: t, condition labels and e / nees / nis per filter."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"t": t, "cond": np.asarray(cond, dtype=str),
              "info": json.dumps({"platform": platform, "set": set_name, "filters": list(curves)})}  # fmt: skip
    for f, c in curves.items():
        for k in ("e", "nees", "nis"):
            v = c.get(k)
            arrays[f"{f}__{k}"] = (
                np.full(len(t), np.nan) if v is None else np.asarray(v, float)
            )
    np.savez_compressed(path, **arrays)


def curve_runs(root):
    for p in sorted(Path(root).glob("*.npz")):
        d = np.load(p, allow_pickle=False)
        info = json.loads(str(d["info"]))
        curves = {
            f: {k: d[f"{f}__{k}"] for k in ("e", "nees", "nis")}
            for f in info["filters"]
        }
        yield {"run": p.stem, "platform": info["platform"], "set": info["set"], "seed": 0,
               "traj": 0, "t": d["t"], "curves": curves, "m": 3, "n": 3,
               "cond": d["cond"].astype(object)}  # fmt: skip


# ---------------------------------------------------------------- summaries
def condition_table(run_iter_rows):
    """RMSE of the error per condition and filter (nav)."""
    acc = defaultdict(lambda: [0.0, 0])
    for r in run_iter_rows:
        for f, c in r["curves"].items():
            for lab in np.unique(r["cond"]):
                e = c["e"][r["cond"] == lab]
                acc[(r["platform"], lab, f)][0] += float(np.sum(e**2))
                acc[(r["platform"], lab, f)][1] += len(e)
    return [{"platform": k[0], "condition": k[1], "filter": k[2], "rmse": (v[0] / v[1]) ** 0.5,
             "samples": v[1]} for k, v in sorted(acc.items())]  # fmt: skip


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def summary(rows, by=("set", "filter")):
    agg = defaultdict(lambda: defaultdict(int))
    for r in rows:
        key = tuple(r[b] for b in by)
        a = agg[key]
        a["peaks"] += 1
        a["unique"] += r["unique"]
        a[r["why"]] += 1
        a["e_max_sum"] += r["e_max"]
    out = []
    for key, a in sorted(agg.items()):
        row = dict(zip(by, key, strict=True))
        row.update({"peaks": a["peaks"], "unique": a["unique"],
                    "mean_peak": a["e_max_sum"] / a["peaks"]})  # fmt: skip
        for w in ("overconfident", "surprised", "aware", "no covariance"):
            row[w] = a[w]
        out.append(row)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("kind", choices=("nav", "knet", "curves"))
    ap.add_argument(
        "path", help="nav: compare_events output dir (with logs/); knet: knet_logs dir"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--states", nargs="*", help="knet: only states whose name starts with these"
    )
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.kind == "nav":
        runs = nav_runs(Path(a.path) / "logs")
    elif a.kind == "knet":
        runs = knet_runs(Path(a.path).expanduser(), a.states)
    else:
        runs = curve_runs(Path(a.path).expanduser())
    rows, kept = [], []
    for r in runs:
        for p in analyse(r["curves"], r["m"], r["n"], r["cond"]):
            p.update({"run": r["run"], "platform": r["platform"], "set": r["set"],
                      "seed": r["seed"], "traj": r["traj"], "t_max": float(r["t"][p["imax"]]),
                      "t_start": float(r["t"][p["i0"]]), "t_end": float(r["t"][p["i1"]])})  # fmt: skip
            rows.append(p)
        if r["cond"] is not None:
            kept.append(r)
    write_csv(out / "peaks.csv", rows)
    write_csv(out / "peaks_by_set.csv", summary(rows))
    if kept:
        write_csv(
            out / "peaks_by_condition.csv",
            summary(rows, ("platform", "condition", "filter")),
        )
        write_csv(out / "condition_table.csv", condition_table(kept))
    print(f"{len(rows)} peaks -> {out}")


if __name__ == "__main__":
    main()
