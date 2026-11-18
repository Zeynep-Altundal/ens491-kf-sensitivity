# report

LaTeX report of the study, built from the CSV files and figures of the repository.

| File | Purpose |
|------|---------|
| `main.tex` | Report source. |
| `refs.bib` | Bibliography; only entries whose authors, title, venue and year are certain. Missing citations are marked with `\todo{citation needed: ...}` in `main.tex`. |
| `make_tables.py` | Reads the result CSV files and writes every table and every number used in the prose to `generated/`; also cross-checks values quoted in the READMEs against the CSV files (`generated/crosscheck.txt`). |
| `generated/` | Output of `make_tables.py` (tables, number macros, generator defaults, commit list, cross-check log). |
| `check_tex.py` | Structural checks without a LaTeX installation (inputs, figures, references, citations, braces, environments) and the list of `\todo` items. |
| `package.py` | Builds `build/overleaf.zip` with the sources and copies of the referenced figures. |

## Building (from the repository root)

```bash
python report/make_tables.py
python report/check_tex.py
python report/package.py          # upload build/overleaf.zip to Overleaf
cd report && latexmk -pdf main.tex   # local build, if a LaTeX installation is present
```

Figures are referenced through `\graphicspath{{../experiments/}{../analysis/}{figures/}}`: the first
two entries serve a local build inside `report/`, the last one the Overleaf archive. Figures are not
regenerated.
