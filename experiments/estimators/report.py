"""Summary tables of the sweep with bootstrap intervals over seeds (protocol: ``README.md``).

    .venv/bin/python -m experiments.estimators.report      (after ``sweep``)

Writes ``results/summary_rmse.csv``, ``results/summary_peak.csv``, ``results/summary_recovery.csv``
and ``results/tables.md``. Only Observed numbers; no explanations.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.common.bootstrap import bootstrap, mean_stat, rms_stat
from experiments.estimators.metrics import PHASES
from experiments.estimators.sweep import PROVISIONAL, RESULTS

ESTIMATORS = ("KF", "EKF", "ESKF")


def _read(path: Path) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def _fmt(est, lo, hi, digits=3) -> str:
    return f"{est:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def _label(e: str) -> str:
    return f"{e} (provisional)" if e in PROVISIONAL else e


def main() -> None:
    rows = _read(RESULTS / "run_metrics.csv")
    by = defaultdict(list)
    for r in rows:
        by[(r["estimator"], r["scenario"])].append(r)
    scenarios = list(dict.fromkeys(r["scenario"] for r in rows))
    rmse_rows, peak_rows = [], []
    for scen in scenarios:
        for e in ESTIMATORS:
            rr = sorted(by[(e, scen)], key=lambda r: int(r["seed"]))
            n = len(rr)
            out = {"estimator": e, "scenario": scen, "seeds": n}
            for q in ("p", "v", "att_deg"):
                for ph in ("overall",) + PHASES:
                    if q != "p" and ph != "overall":
                        continue
                    mse = np.array([float(r[f"mse_{q}_{ph}"]) for r in rr])
                    if np.all(np.isnan(mse)):
                        out[f"rmse_{q}_{ph}"] = out[f"rmse_{q}_{ph}_lo"] = out[f"rmse_{q}_{ph}_hi"] = float("nan")
                        continue
                    est, lo, hi = bootstrap(rms_stat, mse)
                    out[f"rmse_{q}_{ph}"], out[f"rmse_{q}_{ph}_lo"], out[f"rmse_{q}_{ph}_hi"] = est, lo, hi
            rmse_rows.append(out)
            peaks = np.array([float(r["peak_p"]) for r in rr])
            times = np.array([float(r["peak_t"]) for r in rr])
            est, lo, hi = bootstrap(mean_stat, peaks)
            peak_rows.append({"estimator": e, "scenario": scen, "seeds": n, "peak_p": est, "peak_p_lo": lo,
                              "peak_p_hi": hi, "peak_t_median": float(np.median(times)),
                              "peak_t_min": float(times.min()), "peak_t_max": float(times.max())})
    rec = _read(RESULTS / "recovery.csv")
    groups = defaultdict(list)
    for r in rec:
        ev = "spike" if r["event"].startswith("spike@") and r["scenario"] == "pos_spikes_repeated" else r["event"]
        t_end = "" if ev == "spike" and r["scenario"] == "pos_spikes_repeated" else r["t_end"]
        groups[(r["estimator"], r["scenario"], ev, t_end)].append((int(r["seed"]), float(r["recovery_s"])))
    rec_rows = []
    for (e, scen, ev, t_end), vals in groups.items():
        if t_end == "":
            if ev != "spike":
                rec_rows.append({"estimator": e, "scenario": scen, "event": ev, "t_end": "", "note": "n/a (no end)"})
                continue
        per_seed = defaultdict(list)
        for seed, v in vals:
            per_seed[seed].append(v)
        if ev == "spike" and scen == "pos_spikes_repeated":
            stat_seed = {s: (float(np.median(v)), float(np.max(v))) for s, v in per_seed.items()}
            med = np.array([stat_seed[s][0] for s in sorted(stat_seed)])
            mx = np.array([stat_seed[s][1] for s in sorted(stat_seed)])
            row = {"estimator": e, "scenario": scen, "event": "10 spikes", "t_end": "25-63"}
            for name, arr in (("median_over_spikes", med), ("max_over_spikes", mx)):
                fin = np.isfinite(arr)
                row[f"{name}_not_recovered"] = int((~fin).sum())
                if fin.all():
                    est, lo, hi = bootstrap(mean_stat, arr)
                    row[f"{name}_mean"], row[f"{name}_lo"], row[f"{name}_hi"] = est, lo, hi
            rec_rows.append(row)
            continue
        arr = np.array([v[0] for s, v in sorted(per_seed.items())])
        fin = np.isfinite(arr)
        row = {"estimator": e, "scenario": scen, "event": ev, "t_end": t_end, "seeds": len(arr),
               "not_recovered": int((~fin).sum())}
        if fin.all():
            est, lo, hi = bootstrap(mean_stat, arr)
            row.update(recovery_mean=est, recovery_lo=lo, recovery_hi=hi)
        elif fin.any():
            row["recovery_mean_of_recovered"] = float(arr[fin].mean())
        rec_rows.append(row)
    for name, rr in (("summary_rmse.csv", rmse_rows), ("summary_peak.csv", peak_rows),
                     ("summary_recovery.csv", rec_rows)):
        keys = list(dict.fromkeys(k for r in rr for k in r))
        with (RESULTS / name).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rr)
    _tables(rmse_rows, peak_rows, rec_rows)


def _tables(rmse_rows, peak_rows, rec_rows) -> None:
    L = ["# Results of the estimator sweep (Observed only)", "",
         "Mean over 10 measurement seeds with the 95 % bootstrap interval over seeds in brackets.",
         "`KF`: companion KF-CV (position only, system-level baseline). `EKF`, `ESKF`: own filters, "
         "provisional (the companion EKF and ESKF were stopped; see `../README.md`).", ""]
    idx = {(r["estimator"], r["scenario"]): r for r in rmse_rows}
    scen = list(dict.fromkeys(r["scenario"] for r in rmse_rows))
    L += ["## Position RMSE over the flight [m]", "", "| Scenario | " + " | ".join(_label(e) for e in ESTIMATORS) + " |",
          "|---|" + "---|" * len(ESTIMATORS)]
    for s in scen:
        L.append(f"| `{s}` | " + " | ".join(_fmt(idx[(e, s)]["rmse_p_overall"], idx[(e, s)]["rmse_p_overall_lo"],
                                                idx[(e, s)]["rmse_p_overall_hi"]) for e in ESTIMATORS) + " |")
    L += ["", "## Velocity RMSE [m/s] and attitude RMSE [deg] over the flight", "",
          "| Scenario | " + " | ".join(f"{_label(e)} v" for e in ESTIMATORS) + " | " +
          " | ".join(f"{_label(e)} att" for e in ("EKF", "ESKF")) + " |", "|---|" + "---|" * 5]
    for s in scen:
        L.append(f"| `{s}` | " + " | ".join(_fmt(idx[(e, s)]["rmse_v_overall"], idx[(e, s)]["rmse_v_overall_lo"],
                                                idx[(e, s)]["rmse_v_overall_hi"]) for e in ESTIMATORS) + " | " +
                 " | ".join(_fmt(idx[(e, s)]["rmse_att_deg_overall"], idx[(e, s)]["rmse_att_deg_overall_lo"],
                                 idx[(e, s)]["rmse_att_deg_overall_hi"], 2) for e in ("EKF", "ESKF")) + " |")
    for e in ESTIMATORS:
        L += ["", f"## Position RMSE per phase [m], {_label(e)}", "",
              "| Scenario | " + " | ".join(PHASES) + " |", "|---|" + "---|" * len(PHASES)]
        for s in scen:
            r = idx[(e, s)]
            L.append(f"| `{s}` | " + " | ".join(_fmt(r[f"rmse_p_{ph}"], r[f"rmse_p_{ph}_lo"], r[f"rmse_p_{ph}_hi"])
                                               for ph in PHASES) + " |")
    pk = {(r["estimator"], r["scenario"]): r for r in peak_rows}
    L += ["", "## Peak position error [m] and its time [s] (median, range over seeds)", "",
          "| Scenario | " + " | ".join(_label(e) for e in ESTIMATORS) + " |", "|---|" + "---|" * len(ESTIMATORS)]
    for s in scen:
        L.append(f"| `{s}` | " + " | ".join(
            f"{_fmt(pk[(e, s)]['peak_p'], pk[(e, s)]['peak_p_lo'], pk[(e, s)]['peak_p_hi'])} at "
            f"{pk[(e, s)]['peak_t_median']:.2f} ({pk[(e, s)]['peak_t_min']:.2f}-{pk[(e, s)]['peak_t_max']:.2f})"
            for e in ESTIMATORS) + " |")
    L += ["", "## Time to recover after each corruption event [s]", "",
          "Recovered: the estimate stays within 0.1 m of the nominal run's estimate (same estimator and seed) "
          "for 2 s. \"n/a\": the corruption has no end in the flight. \"not rec.\": number of seeds (of 10) "
          "without recovery before the end of the flight.", "",
          "| Scenario | Event | End [s] | " + " | ".join(_label(e) for e in ESTIMATORS) + " |",
          "|---|---|---|" + "---|" * len(ESTIMATORS)]
    keyed = defaultdict(dict)
    for r in rec_rows:
        keyed[(r["scenario"], r["event"], str(r["t_end"]))][r["estimator"]] = r

    def cell(r):
        if r is None:
            return "-"
        if "note" in r:
            return r["note"]
        if r["event"] == "10 spikes":
            parts = []
            for name, lab in (("median_over_spikes", "median"), ("max_over_spikes", "max")):
                if f"{name}_mean" in r:
                    parts.append(f"{lab} {_fmt(r[f'{name}_mean'], r[f'{name}_lo'], r[f'{name}_hi'], 2)}")
                else:
                    parts.append(f"{lab}: not rec. {r[f'{name}_not_recovered']}/10")
            return "; ".join(parts)
        if "recovery_mean" in r:
            return _fmt(r["recovery_mean"], r["recovery_lo"], r["recovery_hi"], 2)
        extra = f", mean of recovered {r['recovery_mean_of_recovered']:.2f}" if "recovery_mean_of_recovered" in r else ""
        return f"not rec. {r['not_recovered']}/10{extra}"

    for (s, ev, t_end), d in keyed.items():
        L.append(f"| `{s}` | {ev} | {t_end} | " + " | ".join(cell(d.get(e)) for e in ESTIMATORS) + " |")
    (RESULTS / "tables.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
