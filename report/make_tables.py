"""Generate the LaTeX tables and number macros of the report from the CSV files of the repository.

Every number in the report is produced here from a CSV file; the source file of each table is
written into its caption. Outputs (all in ``report/generated/``):

* ``tab_<name>.tex``: one table per file;
* ``numbers.tex``: ``\\newcommand`` macros for the numbers quoted in the prose;
* ``generator_defaults.csv``: default parameters of ``synthetic.scenario.ScenarioConfig``, written
  from the code so that the generator comparison table also reads from a CSV;
* ``crosscheck.txt``: values quoted in the READMEs compared with the CSV files.

Usage (from the repository root)::

    python report/make_tables.py
"""

from __future__ import annotations

import csv
import dataclasses
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "report" / "generated"

LK = ROOT / "experiments/linear_kf/results"
FF = ROOT / "experiments/filter_family/results"
ES = ROOT / "experiments/eskf_sensitivity/results"
NR = ROOT / "experiments/neural_r/results"
DC = ROOT / "analysis/dataset_char/results"

MACROS: list[tuple[str, str]] = []
CHECKS: list[str] = []


# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------

def read(path: Path) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def esc(s: str) -> str:
    return (str(s).replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("%", "\\%")
            .replace("&", "\\&").replace("#", "\\#").replace("^", "\\^{}"))


def num(x, d: int = 3) -> str:
    """Format a number with d significant digits (fixed or scientific as appropriate)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return esc(x)
    if math.isnan(v):
        return "--"
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 1e5 or a < 1e-3:
        m, e = f"{v:.{max(d - 1, 0)}e}".split("e")
        return f"${m}\\times10^{{{int(e)}}}$"
    digits = max(d - 1 - int(math.floor(math.log10(a))), 0)
    return f"{v:.{digits}f}"


def ci(row: dict, key: str, lo: str | None = None, hi: str | None = None, d: int = 3) -> str:
    lo = lo or f"{key}_lo"
    hi = hi or f"{key}_hi"
    return f"{num(row[key], d)} [{num(row[lo], d)}, {num(row[hi], d)}]"


def macro(name: str, value: str) -> None:
    MACROS.append((name, value))


def table(name: str, caption: str, sources: list[Path], header: list[str], rows: list[list[str]],
          align: str | None = None, size: str = "\\small", long: bool = False) -> None:
    align = align or "l" * len(header)
    src = ", ".join(f"\\texttt{{{esc(rel(s))}}}" for s in sources)
    cap = f"{caption} Source: {src}."
    lines = []
    if long:
        lines += [f"{{{size}", f"\\begin{{longtable}}{{{align}}}",
                  f"\\caption{{{cap}}}\\label{{tab:{name}}}\\\\", "\\toprule",
                  " & ".join(header) + " \\\\", "\\midrule", "\\endfirsthead",
                  "\\toprule", " & ".join(header) + " \\\\", "\\midrule", "\\endhead"]
        lines += [" & ".join(r) + " \\\\" for r in rows]
        lines += ["\\bottomrule", "\\end{longtable}", "}"]
    else:
        lines += ["\\begin{table}[htbp]", "\\centering", size,
                  f"\\caption{{{cap}}}\\label{{tab:{name}}}",
                  "\\resizebox{\\linewidth}{!}{%", f"\\begin{{tabular}}{{{align}}}", "\\toprule",
                  " & ".join(header) + " \\\\", "\\midrule"]
        lines += [" & ".join(r) + " \\\\" for r in rows]
        lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    (OUT / f"tab_{name}.tex").write_text("\n".join(lines) + "\n")


def check(label: str, quoted: float, csv_value: float, tol: float) -> None:
    ok = abs(quoted - csv_value) <= tol
    CHECKS.append(f"{'OK      ' if ok else 'MISMATCH'} {label}: README {quoted} vs CSV {csv_value:.6g}")


# --------------------------------------------------------------------------------------------
# 4.1 Linear KF
# --------------------------------------------------------------------------------------------

def linear_kf() -> None:
    src = LK / "closure_summary.csv"
    rows = read(src)
    settings = list(dict.fromkeys(r["setting"] for r in rows))
    out = []
    for s in settings:
        rr = {r["metric"]: r for r in rows if r["setting"] == s}
        out.append([esc(s)] + [f"{num(rr[m]['estimate'])} [{num(rr[m]['ci_low'])}, {num(rr[m]['ci_high'])}]"
                               for m in ("pos_rmse", "vel_rmse", "mean_nees", "mean_nis")])
    table("lin_closure", "Linear KF, N = 200 runs, T = 200: one filter quantity scaled, 95\\,\\% "
          "bootstrap intervals over runs. Expected NEES 2, NIS 1.", [src],
          ["Setting", "Position RMSE [m]", "Velocity RMSE [m/s]", "Mean NEES", "Mean NIS"], out)
    nom = {r["metric"]: r for r in rows if r["setting"] == "nominal"}
    macro("LinNomNEES", f"{num(nom['mean_nees']['estimate'])} [{num(nom['mean_nees']['ci_low'])}, {num(nom['mean_nees']['ci_high'])}]")
    macro("LinNomNIS", f"{num(nom['mean_nis']['estimate'])} [{num(nom['mean_nis']['ci_low'])}, {num(nom['mean_nis']['ci_high'])}]")

    src = LK / "open_point.csv"
    rows = read(src)
    out = [[esc(r["config"]), num(r["q_scale"]), num(r["r_scale"]), num(r["p0_scale"]),
            num(r["pos_rmse_T200_full"], 4), num(r["mean_nees_T200_full"], 4),
            num(r["mean_nis_T200_full"], 4), r["gain_conv_step"]] for r in rows]
    table("lin_scale", "Common-scale identity on data: configurations that differ by a common factor "
          "of all filter covariances give identical RMSE, and NEES and NIS scaled by the inverse "
          "factor (T = 200, all steps). The gain-convergence step uses a 1\\,\\% threshold.", [src],
          ["Configuration", "$Q$ scale", "$R$ scale", "$P_0$ scale", "Pos. RMSE", "NEES", "NIS",
           "Gain conv. step"], out)
    a = next(r for r in rows if r["config"] == "Q x 0.001")
    b = next(r for r in rows if r["config"] == "R x 1000, P0 x 1000")
    macro("LinScaleRMSEa", num(a["pos_rmse_T200_full"], 6))
    macro("LinScaleRMSEb", num(b["pos_rmse_T200_full"], 6))
    macro("LinScaleNEESa", num(a["mean_nees_T200_full"], 5))
    macro("LinScaleNEESb", num(b["mean_nees_T200_full"], 5))

    src = LK / "closure_paired.csv"
    out = [[esc(r["a"]), esc(r["b"]), esc(r["metric"]),
            f"{num(r['a_minus_b'])} [{num(r['ci_low'])}, {num(r['ci_high'])}]", r["supported"]]
           for r in read(src)]
    table("lin_paired", "Linear KF: paired differences (A minus B) on the same runs.", [src],
          ["A", "B", "Metric", "Difference [95\\,\\% CI]", "Supported"], out)

    src = LK / "closure_nominal_time.csv"
    rows = read(src)
    fin = sum(float(r["nees_lo"]) <= float(r["nees_avg"]) <= float(r["nees_hi"]) for r in rows) / len(rows)
    fis = sum(float(r["nis_lo"]) <= float(r["nis_avg"]) <= float(r["nis_hi"]) for r in rows) / len(rows)
    macro("LinChiNEESinside", num(100 * fin, 3))
    macro("LinChiNISinside", num(100 * fis, 3))
    macro("LinChiSteps", str(len(rows)))


# --------------------------------------------------------------------------------------------
# 4.2 Filter family
# --------------------------------------------------------------------------------------------

def filter_family() -> None:
    src = FF / "closure_summary.csv"
    out = [[esc(r["setting"]), esc(r["filter"]), r["n_diverged"], ci(r, "pos_rmse"), ci(r, "mean_nees"),
            ci(r, "mean_nis")] for r in read(src)]
    table("ff_closure", "Range-bearing benchmark, N = 100 runs, 60 steps: divergence count and metrics "
          "over non-diverged runs with 95\\,\\% intervals. Expected NEES 4, NIS 2.", [src],
          ["Setting", "Filter", "Div.", "Position RMSE [m]", "Mean NEES", "Mean NIS"], out, long=True)
    qd = next(r for r in read(src) if r["setting"] == "Q x 0.01" and r["filter"] == "AdaptiveEKF")
    macro("FFAdaptQdiv", qd["n_diverged"])
    nom = next(r for r in read(src) if r["setting"] == "nominal" and r["filter"] == "AdaptiveEKF")
    macro("FFAdaptNomNEES", ci(nom, "mean_nees"))
    check("filter_family README adaptive nominal NEES 4.36", 4.36, float(nom["mean_nees"]), 0.005)
    check("filter_family README adaptive nominal NEES lower 4.17", 4.17, float(nom["mean_nees_lo"]), 0.005)
    check("filter_family README adaptive nominal NEES upper 4.56", 4.56, float(nom["mean_nees_hi"]), 0.005)

    src = FF / "closure_paired.csv"
    rows = read(src)
    sup = [r for r in rows if r["supported"] == "True"]
    out = [[esc(r["setting"]), esc(r["a"]), esc(r["b"]), esc(r["metric"]),
            f"{num(r['a_minus_b'])} [{num(r['ci_low'])}, {num(r['ci_high'])}]", r["runs_used"]] for r in sup]
    table("ff_paired", f"Filter family: the {len(sup)} supported paired differences (A minus B) out of "
          f"{len(rows)} computed; all others have intervals containing zero.", [src],
          ["Setting", "A", "B", "Metric", "Difference [95\\,\\% CI]", "Runs"], out, long=True)
    macro("FFPairedSupported", str(len(sup)))
    macro("FFPairedTotal", str(len(rows)))

    src = FF / "nominal_comparison.csv"
    out = [[esc(r["filter"]), num(r["pos_rmse"]), num(r["mean_nees"]), num(r["mean_nis"]),
            num(r["runtime_us_per_step"]), num(r["mean_iterations"])] for r in read(src)]
    table("ff_runtime", "Nominal range-bearing comparison (N = 100, 60 steps) with the wall-clock time "
          "per predict and update step in pure Python and NumPy (relative comparison only).", [src],
          ["Filter", "Pos. RMSE [m]", "NEES", "NIS", "Time per step [$\\mu$s]", "Mean IEKF iterations"], out)
    for r in read(src):
        macro(f"FFRuntime{r['filter'].replace('Adaptive', 'Adapt')}", num(r["runtime_us_per_step"]))


# --------------------------------------------------------------------------------------------
# 4.3 and 4.4 ESKF
# --------------------------------------------------------------------------------------------

def eskf() -> None:
    src = ES / "closure_summary.csv"
    # Split in two long tables so that each fits the text width (a longtable cannot be scaled).
    out = [[esc(r["setting"]), r["n_failed"], ci(r, "pos_rmse"), ci(r, "att_rmse_deg")] for r in read(src)]
    table("es_closure", "ESKF sensitivity, profile \\texttt{turning} unless varied, N = 30 runs: "
          "nominal setting and the end points of every sweep, accuracy with 95\\,\\% intervals "
          "(consistency in Table~\\ref{tab:es_closure_cons}).", [src],
          ["Setting", "Failed", "Pos. RMSE [m]", "Att. RMSE [deg]"], out, long=True, size="\\scriptsize")
    out = [[esc(r["setting"]), ci(r, "mean_nees_pos"), ci(r, "mean_nis")] for r in read(src)]
    table("es_closure_cons", "ESKF sensitivity, same settings as Table~\\ref{tab:es_closure}: mean "
          "position NEES and mean NIS with 95\\,\\% intervals (expected 3 each).", [src],
          ["Setting", "Pos. NEES", "NIS"], out, long=True, size="\\scriptsize")

    src = ES / "closure_paired.csv"
    rows = read(src)
    out = [[esc(r["setting"]), esc(r["metric"]),
            f"{num(r['minus_nominal'])} [{num(r['ci_low'])}, {num(r['ci_high'])}]", r["supported"]]
           for r in rows]
    table("es_paired", "ESKF: paired differences to the nominal setting (same run indices).", [src],
          ["Setting", "Metric", "Minus nominal [95\\,\\% CI]", "Supported"], out, long=True)

    src = ES / "consistency_summary.csv"
    rows = [r for r in read(src) if r["reset"] == "True"]
    out = [[esc(r["profile"]), esc(r["block"]), r["dof"],
            f"{num(r['mean_nees'])} [{num(r['ci_low'])}, {num(r['ci_high'])}]",
            f"[{num(r['epoch_bound_low'])}, {num(r['epoch_bound_high'])}]",
            num(r["fraction_epochs_inside"]), num(r["fraction_epochs_above"]), num(r["fraction_epochs_below"])]
           for r in rows]
    table("es_consistency", "ESKF full-state and sub-block NEES, N = 50 runs, covariance reset on: "
          "mean over runs and epochs with 95\\,\\% bootstrap interval, the 95\\,\\% chi-square "
          "interval of the N-run average at one epoch, and the fractions of epochs inside, above "
          "and below it.", [src],
          ["Profile", "Block", "Dof", "Mean NEES [95\\,\\% CI]", "$\\chi^2$ interval", "Inside",
           "Above", "Below"], out, long=True)
    t = next(r for r in rows if r["profile"] == "turning" and r["block"] == "full")
    check("eskf README '4.2 % of epochs' (turning, full, above)", 0.042, float(t["fraction_epochs_above"]), 0.0005)

    src = ES / "misalignment_summary.csv"
    rows = read(src)
    out = [[num(r["misalignment_deg"]), ci(r, "pos_rmse"), ci(r, "att_rmse_deg"), ci(r, "nees_full"),
            ci(r, "nees_attitude"), ci(r, "nees_gyro_bias"), ci(r, "mean_nis")] for r in rows]
    table("es_misalign", "Fixed misalignment about the body $x$ axis, \\texttt{turning}, N = 50 paired "
          "runs; NEES computed against the true body attitude. Chi-square intervals: full "
          f"[{num(rows[0]['nees_full_chi2_lo'])}, {num(rows[0]['nees_full_chi2_hi'])}], 3-dof "
          f"[{num(rows[0]['nees_position_chi2_lo'])}, {num(rows[0]['nees_position_chi2_hi'])}].", [src],
          ["Angle [deg]", "Pos. RMSE [m]", "Att. RMSE [deg]", "Full NEES", "Att. NEES", "Gyro-bias NEES", "NIS"], out)

    src = ES / "misalignment_paired.csv"
    out = [[num(r["misalignment_deg"]), esc(r["metric"]),
            f"{num(r['minus_0deg'])} [{num(r['ci_low'])}, {num(r['ci_high'])}]", r["supported"]] for r in read(src)]
    table("es_misalign_paired", "Misalignment: paired differences to 0 degrees.", [src],
          ["Angle [deg]", "Metric", "Minus 0 deg [95\\,\\% CI]", "Supported"], out)

    src = ES / "attitude_reference_summary.csv"
    out = [[num(r["misalignment_deg"]), esc(r["reference"]), ci(r, "mean_att_nees"),
            f"[{num(r['chi2_lo'])}, {num(r['chi2_hi'])}]", num(r["frac_epochs_inside"]),
            num(r["att_rms_err_deg"])] for r in read(src)]
    table("es_attref", "Attitude NEES against the true body attitude and against the true sensor "
          "attitude (same error definition and tangent frame, same covariance block).", [src],
          ["Angle [deg]", "Reference", "Mean att. NEES [95\\,\\% CI]", "$\\chi^2$ interval",
           "Epochs inside", "Att. RMS error [deg]"], out)
    src2 = ES / "attitude_reference_time.csv"
    rows = read(src2)
    def seg_mean(col, a, b):
        v = [float(r[col]) for r in rows if a <= float(r["t"]) < b]
        return sum(v) / len(v)
    macro("AttRefSensorEarly", num(seg_mean("nees_att_sensor_5deg", 0, 10)))
    macro("AttRefSensorLate", num(seg_mean("nees_att_sensor_5deg", 30, 60.01)))
    macro("AttRefBodyZeroLate", num(seg_mean("nees_att_body_0deg", 30, 60.01)))

    src = ES / "gnss_summary.csv"
    out = []
    for r in read(src):
        ins = ci(r, "nees_full_inside") if r["epochs_inside"] != "0" else "--"
        out.append([esc(r["case"]), ci(r, "final_pos_err_mean", "final_pos_err_lo", "final_pos_err_hi"),
                    num(r["final_pos_err_max"]), ins, ci(r, "nees_full_outside")])
    table("es_gnss", "GNSS availability (\\texttt{turning}, N = 50): S0 position at 10\\,Hz, S1 outage "
          "from 20\\,s, S2 no position measurement; S1 also with misalignment.", [src],
          ["Case", "Final pos. error [m]", "Max over runs [m]", "Full NEES in outage", "Full NEES outside"], out)

    src = ES / "gnss_drift.csv"
    rows = read(src)
    cases = list(dict.fromkeys(r["case"] for r in rows))
    taus = list(dict.fromkeys(r["elapsed_outage_s"] for r in rows))
    out = [[esc(c)] + [f"{num(r['rms_pos_err'])} / {num(r['three_sigma'])}" for tau in taus
                       for r in rows if r["case"] == c and r["elapsed_outage_s"] == tau] for c in cases]
    table("es_drift", "Drift: RMS position error / $3\\sqrt{\\overline{\\mathrm{tr}\\,P_{p}}}$ [m] at "
          "elapsed outage time (S2 measured from t = 0).", [src],
          ["Case"] + [f"{num(t)}\\,s" for t in taus], out, size="\\scriptsize")

    src = ES / "scale_factor_summary.csv"
    out = [[esc(r["profile"]), esc(r["scenario"]), num(100 * float(r["scale_factor"])), ci(r, "pos_rmse"),
            ci(r, "att_rmse_deg"), ci(r, "nees_full"), num(r["nees_full_frac_epochs_inside"]),
            ci(r, "nees_gyro_bias"), ci(r, "mean_nis")] for r in read(src)]
    table("es_sf", "Gyroscope scale-factor error $s$ (filter assumes $s = 0$), N = 50 paired runs. "
          "Full-NEES chi-square interval "
          f"[{num(read(src)[0]['nees_full_chi2_lo'])}, {num(read(src)[0]['nees_full_chi2_hi'])}].", [src],
          ["Profile", "Scen.", "$s$ [\\%]", "Pos. RMSE [m]", "Att. RMSE [deg]", "Full NEES",
           "Inside", "Gyro-bias NEES", "NIS"], out, size="\\scriptsize")

    src = ES / "scale_factor_paired.csv"
    rows = read(src)
    keys = list(dict.fromkeys((r["profile"], r["scenario"], r["scale_factor"]) for r in rows))
    metrics = ["pos_rmse", "att_rmse_deg", "nees_full", "mean_nis"]
    out = []
    for k in keys:
        cells = []
        for m in metrics:
            r = next(x for x in rows if (x["profile"], x["scenario"], x["scale_factor"]) == k and x["metric"] == m)
            mark = "" if r["supported"] == "True" else "$^{\\dagger}$"
            cells.append(f"{num(r['minus_s0'])} [{num(r['ci_low'])}, {num(r['ci_high'])}]{mark}")
        out.append([esc(k[0]), esc(k[1]), num(100 * float(k[2]))] + cells)
    nsup = sum(r["supported"] == "False" for r in rows)
    table("es_sf_paired", f"Scale factor: paired differences to $s = 0$; $^{{\\dagger}}$ marks the {nsup} "
          f"of {len(rows)} differences whose interval contains zero.", [src],
          ["Profile", "Scen.", "$s$ [\\%]", "Pos. RMSE [m]", "Att. RMSE [deg]", "Full NEES", "NIS"],
          out, size="\\scriptsize")
    macro("SFNotSupported", str(nsup))
    macro("SFTotal", str(len(rows)))
    nis = [abs(float(r["minus_s0"])) for r in rows if r["metric"] == "mean_nis"]
    nees = [float(r["minus_s0"]) for r in rows if r["metric"] == "nees_full"]
    macro("SFNISmaxchange", num(max(nis)))
    macro("SFNEESminchange", num(min(nees)))
    macro("SFNEESmaxchange", num(max(nees)))
    mp = read(ES / "misalignment_paired.csv")
    macro("MisNISmaxchange", num(max(abs(float(r["minus_0deg"])) for r in mp if r["metric"] == "mean_nis")))

    src = ES / "scale_factor_bias.csv"
    rows = [r for r in read(src) if r.get("ratio_to_rate_equivalent") not in (None, "", "nan")]
    out = [[esc(r["profile"]), esc(r["scenario"]), num(100 * float(r["scale_factor"])), esc(r["time"]),
            r["axis"], num(r["rate_equivalent_s_mean_w"]),
            f"{num(r['error_minus_s0'])} [{num(r['diff_lo'])}, {num(r['diff_hi'])}]",
            num(r["ratio_to_rate_equivalent"])] for r in rows]
    table("es_sf_bias", "Scale factor: change of the gyro-bias error relative to $s = 0$ and its ratio "
          "to the rate-equivalent $s\\,\\overline{\\omega}$ over the preceding 10\\,s (rows with a "
          "non-zero rate-equivalent only).", [src],
          ["Profile", "Scen.", "$s$ [\\%]", "Time", "Axis", "$s\\,\\overline{\\omega}$ [rad/s]",
           "Bias-error change [rad/s]", "Ratio"], out, long=True, size="\\scriptsize")


# --------------------------------------------------------------------------------------------
# 4.5 Adaptive R
# --------------------------------------------------------------------------------------------

def adaptive() -> None:
    src = FF / "adaptive_r_summary.csv"
    rows = read(src)
    sel = [r for r in rows if (r["config"] != "rb_T60" and r["segment"] == "steady") or
           (r["config"] == "rb_T60" and r["segment"] == "all")]
    out = [[esc(r["config"]), esc(r["segment"]), esc(r["variant"]), ci(r, "pos_rmse"), ci(r, "mean_nees"),
            ci(r, "mean_nis")] for r in sel]
    table("ar_summary", "Adaptive-R filter under constant true R, N = 100: steady-state segment "
          "(T = 400) and all steps (T = 60). Expected NEES 2 (linear) and 4 (range-bearing).", [src],
          ["Config.", "Segment", "Variant", "Pos. RMSE", "Mean NEES", "Mean NIS"], out, long=True)

    src = NR / "diag_decomposition.csv"
    rows = [r for r in read(src) if r["metric"] == "mean_nees"]
    out = [[esc(r["model"]), r["W"], esc(r["term"]),
            f"{num(r['difference'])} [{num(r['ci_low'])}, {num(r['ci_high'])}]", r["supported"]] for r in rows]
    table("ar_decomp", "Control runs for the over-confidence of the adaptive filter (steady state, "
          "constant true R): exact decomposition of the excess mean NEES into feedback, own-data "
          "correlation and fluctuation, and the replay of a closed-loop R sequence of another run.", [src],
          ["Model", "W", "Term", "NEES difference [95\\,\\% CI]", "Supported"], out, long=True, size="\\scriptsize")
    for r in rows:
        if r["model"] == "linear" and r["W"] == "5":
            tag = {"total": "Total", "feedback": "Feedback", "own-data correlation": "OwnData",
                   "fluctuation": "Fluct", "closed-loop statistics": "ClosedStats"}[r["term"].split(":")[0]]
            macro(f"DiagLinFive{tag}", f"{num(r['difference'])} [{num(r['ci_low'])}, {num(r['ci_high'])}]")

    src = NR / "diag_spread.csv"
    out = [[esc(r["model"]), r["W"], esc(r["estimate"]), r["element"],
            f"{num(r['mean_sd_log_R'])} [{num(r['ci_low'])}, {num(r['ci_high'])}]"] for r in read(src)]
    table("ar_spread", "Spread of the R estimate: mean over runs of the standard deviation over "
          "steady-state steps of $\\ln \\hat R_{ii}$, closed loop and open loop.", [src],
          ["Model", "W", "Estimate", "Element", "Mean sd of $\\ln \\hat R$ [95\\,\\% CI]"], out)

    src = FF / "adaptive_r_psd.csv"
    out = [[esc(r["config"]), esc(r["variant"]), r["updates_with_estimate"], r["postfit_negative_eig"],
            r["innovation_form_negative_eig"], num(r["innovation_form_negative_fraction"])] for r in read(src)]
    table("ar_psd", "Positive semidefiniteness: negative-eigenvalue counts of the implemented "
          "post-fit estimate and of the innovation-form estimate evaluated as a diagnostic.", [src],
          ["Config.", "Variant", "Updates with estimate", "Post-fit negative", "Innovation-form negative",
           "Fraction"], out)

    src = FF / "tv_r_summary.csv"
    rows = read(src)
    filters = list(dict.fromkeys(r["filter"] for r in rows))
    out = []
    for f in filters:
        cells = []
        for m in ("linear", "rb"):
            for p in ("step", "ramp", "burst"):
                r = next(x for x in rows if x["model"] == m and x["pattern"] == p and x["filter"] == f)
                cells.append(num(r["mean_nll"]))
        out.append([esc(f)] + cells)
    table("tv_nll", "Time-varying measurement noise: mean innovation NLL (lower is better) per filter "
          "and pattern, N = 100. \\texttt{adaptive\\_selected} uses the window chosen on validation "
          "runs.", [src],
          ["Filter", "lin. step", "lin. ramp", "lin. burst", "rb step", "rb ramp", "rb burst"], out)
    out = []
    for f in filters:
        cells = []
        for m in ("linear", "rb"):
            for p in ("step", "ramp", "burst"):
                r = next(x for x in rows if x["model"] == m and x["pattern"] == p and x["filter"] == f)
                cells.append(f"{num(r['pos_rmse'])} / {num(r['mean_nees'])}")
        out.append([esc(f)] + cells)
    table("tv_rmse", "Time-varying measurement noise: position RMSE / mean NEES per filter and pattern.", [src],
          ["Filter", "lin. step", "lin. ramp", "lin. burst", "rb step", "rb ramp", "rb burst"], out, size="\\scriptsize")

    src = FF / "tv_r_window_selection.csv"
    rows = [r for r in read(src) if r["selected"] == "True"]
    out = [[esc(r["model"]), esc(r["pattern"]), r["window"], ci(r, "val_mean_nll", d=4)] for r in rows]
    table("tv_sel", "Window selected on validation runs (seed 1000, burst seed 1001) by mean "
          "innovation NLL.", [src], ["Model", "Pattern", "Selected W", "Validation NLL [95\\,\\% CI]"], out)

    src = FF / "tv_r_paired.csv"
    rows = read(src)
    worse = [r for r in rows if r["supported"] == "True" and float(r["minus_true_covariance"]) > 0]
    better = [r for r in rows if r["supported"] == "True" and float(r["minus_true_covariance"]) < 0]
    macro("TVPairedTotal", str(len(rows)))
    macro("TVPairedWorse", str(len(worse)))
    macro("TVPairedBetter", str(len(better)))

    src = FF / "tv_r_identifiability.csv"
    rows = read(src)
    out = []
    for m in ("linear", "rb"):
        for p in ("constant", "step", "ramp", "burst"):
            rr = [r for r in rows if r["model"] == m and r["pattern"] == p]
            for v in ("follows R_true", "follows H P H^T", "not separated"):
                pass
            cnt = {v: sum(r["verdict"] == v for r in rr) for v in ("follows R_true", "follows H P H^T", "not separated")}
            ws = ", ".join(f"{r['window']}{'' if m == 'linear' else '/e' + r['element']}" for r in rr
                           if r["verdict"] == "follows R_true")
            out.append([esc(m), esc(p), str(len(rr)), str(cnt["follows R_true"]), str(cnt["follows H P H^T"]),
                        str(cnt["not separated"]), esc(ws) if ws else "--"])
    table("tv_ident", "Identifiability of the adaptive R estimate (rule of amendment A5): number of "
          "(window, element) cases per verdict.", [src],
          ["Model", "Pattern", "Cases", "Follows $R_{\\mathrm{true}}$", "Follows $HPH^\\top$",
           "Not separated", "Windows following $R_{\\mathrm{true}}$"], out, size="\\scriptsize")


# --------------------------------------------------------------------------------------------
# 4.6 Learned R
# --------------------------------------------------------------------------------------------

def learned() -> None:
    src = NR / "training_summary.csv"
    out = [[esc(r["name"]), esc(r["status"]), num(r["best_val_nll"], 5), r["best_iteration"],
            num(r["train_seconds"]), r["n_parameters"], r["selected"]] for r in read(src)]
    table("nr_train", "Learned R: the six configurations per candidate (equal budget), best pooled "
          "validation NLL and iteration, training time and selection.", [src],
          ["Configuration", "Status", "Best val. NLL", "Iteration", "Time [s]", "Parameters", "Selected"], out)

    src = NR / "neural_r_gates.csv"
    out = [[esc(r["gate"]), esc(r["model"]), esc(r["outcome"]), esc(r["detail"])] for r in read(src)]
    table("nr_gates", "Gates fixed before training and their outcomes.", [src],
          ["Gate", "Model", "Outcome", "Detail"], out, align="lllp{0.62\\linewidth}", size="\\scriptsize")

    asel = read(NR / "neural_r_adaptive_selection.csv")
    wb = next(r["window"] for r in asel if r["selected_burst"] == "True")
    wp = next(r["window"] for r in asel if r["selected_pooled"] == "True")
    macro("NRWinBurst", wb)
    macro("NRWinPooled", wp)
    src = NR / "neural_r_summary.csv"
    rows = read(src)
    sets = list(dict.fromkeys(r["test_set"] for r in rows))
    filters = list(dict.fromkeys(r["filter"] for r in rows))
    for metric, name, cap in (("mean_nll", "nr_nll", "mean innovation NLL"),
                              ("pos_rmse", "nr_rmse", "position RMSE"),
                              ("mean_nees", "nr_nees", "mean NEES (expected 2)"),
                              ("r_abs_logratio", "nr_track", "R tracking error, mean $|\\ln(R/R_{\\mathrm{true}})|$")):
        out = []
        for s in sets:
            cells = []
            for f in filters:
                r = next(x for x in rows if x["test_set"] == s and x["filter"] == f)
                cells.append(num(r[metric]))
            out.append([esc(s)] + cells)
        table(name, f"Learned R evaluation (linear model, N = 100 per set): {cap} per test set and "
              "filter. \\texttt{ho\\_} marks held-out sets.", [src],
              ["Test set", "fixed nom.", "fixed avg. (hindsight)", "true cov.", f"adaptive W{wb} (burst sel.)",
               f"adaptive W{wp} (pooled sel.)", "GRU", "MLP"], out, size="\\scriptsize")
    rec = [r for r in rows if r["test_set"] in ("burst", "ho_burst_short", "ho_burst_long")]
    out = [[esc(r["test_set"]), esc(r["filter"]), ci(r, "recovery_time"), num(r["recovery_censored_fraction"])]
           for r in rec]
    table("nr_recovery", "Recovery time after a burst (steps until the R used is within a factor of two "
          "of $R_{\\mathrm{true}}$), censored at the next burst.", [src],
          ["Test set", "Filter", "Recovery time [95\\,\\% CI]", "Censored fraction"], out, long=True, size="\\scriptsize")

    src = NR / "neural_r_paired.csv"
    rows = read(src)
    out = []
    for s in sets:
        cells = []
        for a, b in (("gru", "adaptive_burst_sel"), ("mlp", "adaptive_burst_sel"), ("gru", "mlp")):
            r = next(x for x in rows if x["test_set"] == s and x["a"] == a and x["b"] == b and x["metric"] == "mean_nll")
            cells.append(f"{num(r['a_minus_b'])} [{num(r['ci_low'])}, {num(r['ci_high'])}]")
        out.append([esc(s)] + cells)
    table("nr_paired", "Paired differences in mean innovation NLL (negative favours the first "
          "filter).", [src], ["Test set", f"GRU $-$ adaptive W{wb}", f"MLP $-$ adaptive W{wb}", "GRU $-$ MLP"],
          out, size="\\scriptsize")

    metrics = ("pos_rmse", "mean_nll", "r_abs_logratio", "recovery_time")
    out = []
    for kind in ("gru", "mlp"):
        for b in ("fixed_nominal", "fixed_average_hindsight", "true_covariance", "adaptive_burst_sel",
                  "adaptive_pooled_sel"):
            cnt = [sum(1 for r in rows if r["a"] == kind and r["b"] == b and r["metric"] == m
                       and r["supported"] == "True" and float(r["a_minus_b"]) > 0) for m in metrics]
            tot = [sum(1 for r in rows if r["a"] == kind and r["b"] == b and r["metric"] == m) for m in metrics]
            out.append([kind.upper(), esc(b)] + [f"{c} / {t}" for c, t in zip(cnt, tot)])
    table("nr_worse", "Number of test sets on which the learned model is worse with a supported "
          "paired difference (higher RMSE, NLL, tracking error or recovery time) / number of test "
          "sets evaluated.", [src],
          ["Model", "Baseline", "Pos. RMSE", "NLL", "Tracking", "Recovery"], out)
    g2 = read(NR / "neural_r_g2_cases.csv")
    macro("GTwoGRU", str(sum(r["model"] == "gru" for r in g2)))
    macro("GTwoMLP", str(sum(r["model"] == "mlp" for r in g2)))

    src = NR / "saturation_bound.csv"
    b = read(src)
    out = [[r["model"].upper(), esc(r["config"]), num(r["logit_max"], 4), num(r["alpha_max"], 4),
            num(r["alpha_attained_max"], 4), num(100 * float(r["attained_fraction_of_logit_bound"]), 4)] for r in b]
    table("nr_bound", "Architectural upper bound of $\\alpha_t$ ($\\mathrm{softplus}(\\sum|W| + b) + "
          "\\varepsilon$ with inputs in $(-1, 1)$) and the largest value attained on the inspected data.", [src],
          ["Model", "Configuration", "$\\sum|W| + b$", "Bound on $\\alpha$", "Attained max", "Attained / bound [\\%]"], out)
    for r in b:
        macro(f"Bound{r['model'].upper()}", num(r["alpha_max"], 4))
    src2 = NR / "saturation_alpha_vs_factor.csv"
    rows = [r for r in read(src2) if r["steps"] == "settled" and float(r["true_factor"]) >= 10]
    out = [[r["model"].upper(), esc(r["set"]), num(r["true_factor"]), r["n"], num(r["alpha_median"]),
            num(r["alpha_q95"]), num(r["alpha_max"])] for r in rows]
    table("nr_alpha", "$\\alpha_t$ versus the true factor $c$ on settled steps ($c$ at its level for "
          "at least 10 steps), factors of at least 10.", [src2],
          ["Model", "Set", "True $c$", "Steps", "Median $\\alpha$", "95\\,\\% $\\alpha$", "Max $\\alpha$"], out)
    tf = read(NR / "saturation_training_factors.csv")
    mx = max(tf, key=lambda r: float(r["factor"]))
    macro("TrainMaxFactor", num(mx["factor"]))
    macro("TrainMaxFactorPct", num(100 * float(mx["fraction"])))
    sl = read(NR / "saturation_curve_slopes.csv")
    out = [[r["model"].upper(), esc(r["config"]),
            f"{num(r['train_slope_per_iteration_351_400'])} $\\pm$ {num(r['train_slope_se'])}",
            num(r["val_slope_per_iteration_350_400"]), num(r["val_nll_350"], 5), num(r["val_nll_375"], 5),
            num(r["val_nll_400"], 5)] for r in sl]
    table("nr_slopes", "Training-curve slopes of the selected configurations over the last 50 iterations.",
          [NR / "saturation_curve_slopes.csv"],
          ["Model", "Configuration", "Train slope / iteration", "Val. slope / iteration", "Val. 350",
           "Val. 375", "Val. 400"], out)
    gru25 = next(r for r in rows if r["model"] == "gru" and r["set"] == "ho_burst_long")
    macro("GRUalphaAtTwentyFive", num(gru25["alpha_median"]))
    trn = next(r for r in rows if r["model"] == "gru" and r["set"] == "training_batches_0_9"
               and float(r["true_factor"]) == 25)
    macro("GRUalphaTrainTwentyFive", num(trn["alpha_median"]))
    check("neural_r README 'about 10 times R_base' (GRU, factor 25, settled median)", 10.0,
          float(gru25["alpha_median"]), 1.0)


# --------------------------------------------------------------------------------------------
# 4.7 Datasets
# --------------------------------------------------------------------------------------------

def datasets() -> None:
    src = DC / "summary.csv"
    rows = read(src)

    def val(ds, q, axis=""):
        r = next(x for x in rows if x["dataset"] == ds and x["quantity"] == q and x["axis"] == axis)
        return r["value"]

    items = [("Samples", "samples", ""), ("Duration [s]", "duration", ""),
             ("Nominal IMU rate [Hz]", "nominal_rate", ""),
             ("Measured rate [Hz]", "measured_rate_1_over_median_dt", ""),
             ("dt std [s]", "dt_std", ""), ("dt max [s]", "dt_max", ""),
             ("Missing samples", "estimated_missing_samples", ""),
             ("Duplicate rows", "exact_duplicate_data_rows", ""),
             ("Gyro quantisation step x [rad/s]", "gyro_smallest_value_step", "x"),
             ("Accel quantisation step x [m/s$^2$]", "accel_smallest_value_step", "x"),
             ("Median $|f|$ [m/s$^2$]", "specific_force_norm_median", ""),
             ("Longest static segment [s]", "longest_static_segment", ""),
             ("Smallest 1\\,s std of $|f|$ [m/s$^2$]", "min_1s_window_std_of_specific_force_norm", ""),
             ("Gyro density bound x [rad/s/$\\sqrt{\\mathrm{Hz}}$]", "gyro_density_upper_bound_from_differences", "x"),
             ("Gyro density bound y", "gyro_density_upper_bound_from_differences", "y"),
             ("Gyro density bound z", "gyro_density_upper_bound_from_differences", "z"),
             ("Accel density bound x [m/s$^2$/$\\sqrt{\\mathrm{Hz}}$]", "accel_density_upper_bound_from_differences", "x"),
             ("Accel density bound y", "accel_density_upper_bound_from_differences", "y"),
             ("Accel density bound z", "accel_density_upper_bound_from_differences", "z"),
             ("White-noise density (gyro)", "gyro_white_noise_density", ""),
             ("Bias instability (gyro)", "gyro_bias_instability", ""),
             ("Spikes gyro x", "gyro_count", "x"), ("Spikes accel x", "accel_count", "x"),
             ("Speed median [m/s]", "speed_median", ""), ("Speed max [m/s]", "speed_max", ""),
             ("$|\\omega|$ median [rad/s]", "angular_rate_norm_median", ""),
             ("$|\\omega|$ max [rad/s]", "angular_rate_norm_max", ""),
             ("Fraction turning", "fraction_time_turning", "")]
    out = [[lab, num(val("kitti", q, a)), num(val("euroc", q, a))] for lab, q, a in items]
    table("ds_summary", "Real-data characterisation (one sequence each; protocol in "
          "\\texttt{analysis/dataset\\_char/README.md}). Density bounds come from sample-to-sample "
          "differences over moving data and are upper bounds, not noise densities.", [src],
          ["Quantity", "KITTI 2011\\_09\\_30\\_drive\\_0016", "EuRoC V1\\_01\\_easy"], out)

    # dataset README cross-checks
    for label, quoted, ds, q, a, tol in (
            ("dataset_char measured rate KITTI 100.007", 100.007, "kitti", "measured_rate_1_over_median_dt", "", 0.0005),
            ("dataset_char measured rate EuRoC 200.003", 200.003, "euroc", "measured_rate_1_over_median_dt", "", 0.0005),
            ("dataset_char |f| median KITTI 9.818", 9.818, "kitti", "specific_force_norm_median", "", 0.0005),
            ("dataset_char |f| median EuRoC 9.808", 9.808, "euroc", "specific_force_norm_median", "", 0.0005),
            ("dataset_char fraction turning EuRoC 0.84", 0.84, "euroc", "fraction_time_turning", "", 0.005),
            ("dataset_char speed max KITTI 15.7", 15.7, "kitti", "speed_max", "", 0.05),
            ("dataset_char gyro spikes x KITTI 135", 135, "kitti", "gyro_count", "x", 0),
            ("dataset_char Vicon gaps 32", 32, "euroc", "gaps_above_1.5_median", "", 0)):
        if q == "gaps_above_1.5_median":
            v = next(x for x in rows if x["dataset"] == ds and x["section"] == "gt_vicon" and x["quantity"] == q)["value"]
        else:
            v = val(ds, q, a)
        check(label, quoted, float(v), tol)

    from synthetic.scenario import ScenarioConfig
    cfg = ScenarioConfig()
    gen = [("imu_rate", cfg.imu_rate, "Hz"), ("meas_rate", cfg.meas_rate, "Hz"),
           ("gyro_noise_density", cfg.gyro_noise_density, "rad/s/sqrt(Hz)"),
           ("accel_noise_density", cfg.accel_noise_density, "m/s^2/sqrt(Hz)"),
           ("gyro_bias_rw", cfg.gyro_bias_rw, "rad/s^2/sqrt(Hz)"),
           ("accel_bias_rw", cfg.accel_bias_rw, "m/s^3/sqrt(Hz)")]
    gsrc = OUT / "generator_defaults.csv"
    with gsrc.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["parameter", "value", "unit"])
        w.writerows(gen)
    g = {r["parameter"]: float(r["value"]) for r in read(gsrc)}
    k_rate, e_rate = float(val("kitti", "measured_rate_1_over_median_dt")), float(val("euroc", "measured_rate_1_over_median_dt"))

    def bound_range(ds, sensor):
        v = [float(val(ds, f"{sensor}_density_upper_bound_from_differences", a)) for a in "xyz"]
        return min(v), max(v)

    out = [["IMU rate [Hz]", num(g["imu_rate"]), f"{num(k_rate)} ({num(k_rate / g['imu_rate'])}$\\times$)",
            f"{num(e_rate)} ({num(e_rate / g['imu_rate'])}$\\times$)"]]
    for sensor, key, unit in (("gyro", "gyro_noise_density", "rad/s/$\\sqrt{\\mathrm{Hz}}$"),
                              ("accel", "accel_noise_density", "m/s$^2$/$\\sqrt{\\mathrm{Hz}}$")):
        cells = []
        for ds in ("kitti", "euroc"):
            lo, hi = bound_range(ds, sensor)
            cells.append(f"not identifiable; bound {num(lo, 2)}--{num(hi, 2)} "
                         f"({num(lo / g[key])}--{num(hi / g[key])}$\\times$)")
        out.append([f"{sensor.capitalize()} noise density [{unit}]", num(g[key])] + cells)
    for key, lab in (("gyro_bias_rw", "Gyro bias random walk"), ("accel_bias_rw", "Accel bias random walk")):
        out.append([lab, num(g[key]), "not identifiable", "not identifiable"])
    table("ds_generator", "Generator defaults compared with the measured values (generator unchanged). "
          "Ratios are measured value divided by the generator value.", [gsrc, src],
          ["Parameter", "Generator", "KITTI", "EuRoC"], out, align="lp{0.12\\linewidth}p{0.3\\linewidth}p{0.3\\linewidth}")


# --------------------------------------------------------------------------------------------
# Compact tables for the main text (the full tables are in the appendices)
# --------------------------------------------------------------------------------------------

def main_tables() -> None:
    src = FF / "adaptive_r_summary.csv"
    rows = read(src)
    variants = ("off", "W5", "W20", "W100", "W200")
    out = []
    for cfg, seg in (("linear_T400", "steady"), ("rb_T400", "steady"), ("rb_T60", "all")):
        cells = []
        for v in variants:
            r = next(x for x in rows if x["config"] == cfg and x["segment"] == seg and x["variant"] == v)
            cells.append(ci(r, "mean_nees"))
        exp = next(x for x in rows if x["config"] == cfg)["expected_nees"]
        out.append([esc(cfg), esc(seg), exp] + cells)
    table("main_const", "Constant true $R$: mean NEES of the non-adaptive filter (off) and of the "
          "adaptive filter for selected windows, with 95\\,\\% intervals (all windows in "
          "Table~\\ref{tab:ar_summary}).", [src],
          ["Configuration", "Segment", "Expected", "off", "W5", "W20", "W100", "W200"], out, size="\\scriptsize")

    src = FF / "tv_r_summary.csv"
    rows = read(src)
    keep = ("fixed_nominal", "fixed_average_hindsight", "true_covariance", "adaptive_W5", "adaptive_W20",
            "adaptive_W200", "adaptive_selected")
    out = []
    for f in keep:
        cells = []
        for m in ("linear", "rb"):
            for p in ("step", "ramp", "burst"):
                r = next(x for x in rows if x["model"] == m and x["pattern"] == p and x["filter"] == f)
                cells.append(num(r["mean_nll"]))
        out.append([esc(f)] + cells)
    sel = {(r["model"], r["pattern"]): r["selected_window"] for r in rows if r["filter"] == "adaptive_selected"}
    out.append(["selected window"] + [sel[(m, p)] for m in ("linear", "rb") for p in ("step", "ramp", "burst")])
    table("main_tv", "Time-varying measurement noise: mean innovation NLL for selected filters (all "
          "windows and RMSE/NEES in Tables~\\ref{tab:tv_nll} and \\ref{tab:tv_rmse}). The last row "
          "gives the window selected on validation runs (post-result amendment A3).", [src],
          ["Filter", "lin. step", "lin. ramp", "lin. burst", "rb step", "rb ramp", "rb burst"], out, size="\\scriptsize")

    src = NR / "diag_decomposition.csv"
    rows = [r for r in read(src) if r["metric"] == "mean_nees" and r["W"] == "5"]
    terms = list(dict.fromkeys(r["term"] for r in rows))
    out = []
    for t in terms:
        cells = []
        for m in ("linear", "rb"):
            r = next(x for x in rows if x["model"] == m and x["term"] == t)
            cells.append(f"{num(r['difference'])} [{num(r['ci_low'])}, {num(r['ci_high'])}]"
                         + ("" if r["supported"] == "True" else "$^{\\dagger}$"))
        out.append([esc(t)] + cells)
    table("main_decomp", "Over-confidence of the adaptive filter with $W = 5$ under constant true $R$: "
          "control runs, steady-state mean NEES differences; $^{\\dagger}$ interval contains zero "
          "(W = 20 and 100 in Table~\\ref{tab:ar_decomp}).", [src],
          ["Term", "Linear model", "Range-bearing"], out, size="\\scriptsize")

    s_src, p_src = NR / "neural_r_summary.csv", NR / "neural_r_paired.csv"
    S = read(s_src)
    P = read(p_src)
    out = []
    for ts in ("burst", "ho_burst_short", "ho_burst_long"):
        cells = [num(next(x for x in S if x["test_set"] == ts and x["filter"] == f)["mean_nll"])
                 for f in ("true_covariance", "adaptive_burst_sel", "gru", "mlp")]
        for a in ("gru", "mlp"):
            r = next(x for x in P if x["test_set"] == ts and x["a"] == a and x["b"] == "adaptive_burst_sel"
                     and x["metric"] == "mean_nll")
            cells.append(f"{num(r['a_minus_b'])} [{num(r['ci_low'])}, {num(r['ci_high'])}]")
        out.append([esc(ts)] + cells)
    table("main_heldout", "Bursts (in-distribution and held-out): mean innovation NLL and paired NLL "
          "difference of the learned models to the burst-selected adaptive window (gate G1 uses the "
          "two held-out sets).", [s_src, p_src],
          ["Test set", "true cov.", "adaptive (burst sel.)", "GRU", "MLP", "GRU $-$ adaptive", "MLP $-$ adaptive"],
          out, size="\\scriptsize")

    out = []
    for ts, b, metrics in (("constant", "fixed_nominal", ("pos_rmse", "mean_nees", "mean_nis", "mean_nll")),
                           ("ho_step", "adaptive_burst_sel", ("mean_nll", "r_abs_logratio")),
                           ("ho_ramp", "adaptive_burst_sel", ("mean_nll", "r_abs_logratio")),
                           ("ho_burst_long", "adaptive_burst_sel", ("r_abs_logratio",))):
        for m in metrics:
            cells = []
            for a in ("gru", "mlp"):
                r = next(x for x in P if x["test_set"] == ts and x["a"] == a and x["b"] == b and x["metric"] == m)
                cells.append(f"{num(r['a_minus_b'])} [{num(r['ci_low'])}, {num(r['ci_high'])}]")
            out.append([esc(ts), esc(b), esc(m)] + cells)
    table("main_limits", "Negative results of the learned models: negative control with constant true "
          "$R$ (gate G3, against the non-adaptive filter), held-out step and ramp, and the tracking "
          "error on the long held-out bursts. Paired differences learned minus baseline.", [p_src],
          ["Test set", "Baseline", "Metric", "GRU", "MLP"], out, size="\\scriptsize")

    src = NR / "neural_r_gates.csv"
    out = [[esc(r["gate"]), esc(r["model"]), esc(r["outcome"])] for r in read(src)]
    table("main_gates", "Gates fixed before training (details in Table~\\ref{tab:nr_gates}).", [src],
          ["Gate", "Model", "Outcome"], out, align="lll", size="\\small")

    m_src, f_src = ES / "misalignment_paired.csv", ES / "scale_factor_paired.csv"
    M = read(m_src)
    F = read(f_src)
    out = []
    for lvl in dict.fromkeys(r["misalignment_deg"] for r in M):
        ne = next(r for r in M if r["misalignment_deg"] == lvl and r["metric"] == "nees_full")
        ni = next(r for r in M if r["misalignment_deg"] == lvl and r["metric"] == "mean_nis")
        out.append([f"misalignment {num(lvl)}\\,deg", "turning S0",
                    f"{num(ne['minus_0deg'])} [{num(ne['ci_low'])}, {num(ne['ci_high'])}]",
                    f"{num(ni['minus_0deg'])} [{num(ni['ci_low'])}, {num(ni['ci_high'])}]"])
    for key in dict.fromkeys((r["profile"], r["scenario"], r["scale_factor"]) for r in F):
        ne = next(r for r in F if (r["profile"], r["scenario"], r["scale_factor"]) == key and r["metric"] == "nees_full")
        ni = next(r for r in F if (r["profile"], r["scenario"], r["scale_factor"]) == key and r["metric"] == "mean_nis")
        out.append([f"scale factor {num(100 * float(key[2]))}\\,\\%", f"{esc(key[0])} {esc(key[1])}",
                    f"{num(ne['minus_s0'])} [{num(ne['ci_low'])}, {num(ne['ci_high'])}]",
                    f"{num(ni['minus_s0'])} [{num(ni['ci_low'])}, {num(ni['ci_high'])}]"])
    table("main_mismatch", "Structural IMU mismatch: paired change of the full-state NEES (expected 15) "
          "and of the mean NIS (expected 3) relative to the matched model.", [m_src, f_src],
          ["Mismatch", "Profile, scenario", "$\\Delta$ full NEES", "$\\Delta$ NIS"], out, size="\\scriptsize")

    src = DC / "summary.csv"
    rows = read(src)

    def val(ds, q, axis=""):
        return next(x for x in rows if x["dataset"] == ds and x["quantity"] == q and x["axis"] == axis)["value"]

    items = [("Measured IMU rate [Hz]", "measured_rate_1_over_median_dt"),
             ("Longest static segment [s]", "longest_static_segment"),
             ("Noise density, bias instability", "gyro_white_noise_density"),
             ("Speed median / max [m/s]", ("speed_median", "speed_max")),
             ("$|\\omega|$ median / max [rad/s]", ("angular_rate_norm_median", "angular_rate_norm_max")),
             ("Fraction of time turning", "fraction_time_turning")]
    out = []
    for lab, q in items:
        if isinstance(q, tuple):
            out.append([lab] + [f"{num(val(d, q[0]))} / {num(val(d, q[1]))}" for d in ("kitti", "euroc")])
        else:
            out.append([lab] + [num(val(d, q)) for d in ("kitti", "euroc")])
    s_src = DC / "synthetic_profiles.csv"
    syn = read(s_src)
    for r in syn:
        out.append([f"synthetic \\texttt{{{esc(r['profile'])}}}: speed max, $|\\omega|$ max, turning",
                    f"{num(r['speed_max'])}, {num(r['angular_rate_norm_max'])}, {num(r['fraction_turning'])}", ""])
    table("main_ds", "Real sequences compared with the synthetic profiles (full characterisation in "
          "Table~\\ref{tab:ds_summary}).", [src, s_src], ["Quantity", "KITTI drive 0016", "EuRoC V1\\_01\\_easy"],
          out, size="\\scriptsize")


def commits() -> None:
    """List of the commits of the branch (identifiers from git, not results)."""
    import subprocess
    log = subprocess.run(["git", "log", "--format=%h %s", "main..HEAD"], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout.strip().splitlines()
    if not log:
        log = subprocess.run(["git", "log", "--format=%h %s"], cwd=ROOT, capture_output=True, text=True,
                             check=True).stdout.strip().splitlines()
    items = [f"\\item \\texttt{{{esc(line.split(' ', 1)[0])}}} {esc(line.split(' ', 1)[1])}" for line in log]
    (OUT / "commits.tex").write_text("\\begin{itemize}\\small\n" + "\n".join(items) + "\n\\end{itemize}\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    commits()
    linear_kf()
    filter_family()
    eskf()
    adaptive()
    learned()
    datasets()
    main_tables()
    (OUT / "numbers.tex").write_text("".join(f"\\newcommand{{\\{n}}}{{{v}}}\n" for n, v in MACROS))
    (OUT / "crosscheck.txt").write_text("\n".join(CHECKS) + "\n")
    print(f"{len(list(OUT.glob('tab_*.tex')))} tables, {len(MACROS)} macros")
    print("\n".join(CHECKS))


if __name__ == "__main__":
    main()
