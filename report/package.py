"""Build an Overleaf-ready archive of the report: ``report/build/overleaf.zip``.

The archive contains ``main.tex``, ``refs.bib``, ``generated/`` and every figure referenced by
``\\includegraphics``, copied from the repository into ``figures/<path>`` (``\\graphicspath`` of
``main.tex`` lists ``figures/``). The figures are copied unchanged; nothing is regenerated.

Usage (from the repository root, after ``python report/make_tables.py``)::

    python report/package.py
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

REPORT = Path(__file__).resolve().parent
ROOT = REPORT.parent
SEARCH = (ROOT / "experiments", ROOT / "analysis")


def main() -> None:
    tex = (REPORT / "main.tex").read_text()
    figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
    out = REPORT / "build" / "overleaf.zip"
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(REPORT / "main.tex", "main.tex")
        z.write(REPORT / "refs.bib", "refs.bib")
        for f in sorted((REPORT / "generated").glob("*")):
            z.write(f, f"generated/{f.name}")
        for fig in figs:
            src = next((base / fig for base in SEARCH if (base / fig).is_file()), None)
            if src is None:
                raise FileNotFoundError(fig)
            z.write(src, f"figures/{fig}")
    print(f"wrote {out} with {len(figs)} figures")


if __name__ == "__main__":
    main()
