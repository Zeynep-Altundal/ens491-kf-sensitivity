# experiments/peak_analysis/results

Outputs of the peak analysis (protocol: `../README.md`), version-controlled.

| File | Content |
|---|---|
| `run_peaks.csv` | One row per run (estimator, scenario, seed): threshold, number of peaks, largest peak (height, excess, time, delay, event, duration), largest excess after the first onset |
| `peaks.csv` | Every detected peak: start, end, peak time, height, excess, attributed event, onset, delay |
| `tables.md` | Tables 1-4 of the protocol (Observed only) |
| `outage_diagnostic.json` | Outage reversal diagnostic: per seed, means, decision rule and outcome |

Regenerate (from the repository root, after the estimator sweep):
`.venv/bin/python -m experiments.peak_analysis.analyze`,
`.venv/bin/python -m experiments.peak_analysis.outage_diagnostic`.
