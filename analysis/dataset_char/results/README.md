# analysis/dataset_char/results

CSV output of `characterise.py` (protocol in `../README.md`).

| File | Content |
|------|---------|
| `summary.csv` | One row per dataset, section, quantity and axis: value, unit and note (inventory, timing, signal, statistics, noise, spikes, dynamics, ground truth). |
| `synthetic_profiles.csv` | Speed, angular-rate and acceleration ranges and fraction of time turning of the synthetic `straight`, `turning` and `aggressive` profiles. |

No Allan deviation file is written: neither sequence contains a static segment of at least 10 s.
