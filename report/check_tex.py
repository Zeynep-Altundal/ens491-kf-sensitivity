"""Structural checks of the report without a LaTeX installation.

Checks: every ``\\input`` file exists; every ``\\includegraphics`` path resolves; every ``\\ref`` has
a ``\\label``; every ``\\cite`` key is in ``refs.bib``; every macro used from ``numbers.tex`` is
defined; braces and ``\\begin``/``\\end`` environments are balanced in each file; lists the
``\\todo`` items. This is not a substitute for compiling.

Usage (from the repository root)::

    python report/check_tex.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPORT = Path(__file__).resolve().parent
ROOT = REPORT.parent


def strip_comments(s: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", s)


def balanced(name: str, s: str, errors: list) -> None:
    body = re.sub(r"\\begin\{verbatim\}.*?\\end\{verbatim\}", "", s, flags=re.S)
    body = body.replace("\\{", "").replace("\\}", "")
    depth = 0
    for ch in body:
        depth += ch == "{"
        depth -= ch == "}"
        if depth < 0:
            errors.append(f"{name}: unmatched closing brace")
            return
    if depth:
        errors.append(f"{name}: {depth} unclosed brace(s)")
    stack = []
    for kind, env in re.findall(r"\\(begin|end)\{([^}]+)\}", body):
        if kind == "begin":
            stack.append(env)
        elif not stack or stack.pop() != env:
            errors.append(f"{name}: environment mismatch at \\end{{{env}}}")
            return
    if stack:
        errors.append(f"{name}: unclosed environments {stack}")


def main() -> int:
    errors: list[str] = []
    main_tex = strip_comments((REPORT / "main.tex").read_text())
    files = {"main.tex": main_tex}
    for inc in re.findall(r"\\input\{([^}]+)\}", main_tex):
        p = REPORT / (inc if inc.endswith(".tex") else inc + ".tex")
        if not p.is_file():
            errors.append(f"missing \\input file {inc}")
        else:
            files[inc] = strip_comments(p.read_text())
    text = "\n".join(files.values())
    for name, s in files.items():
        balanced(name, s, errors)
    for fig in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", main_tex):
        if not any((base / fig).is_file() for base in (ROOT / "experiments", ROOT / "analysis")):
            errors.append(f"missing figure {fig}")
    labels = set(re.findall(r"\\label\{([^}]+)\}", text))
    for ref in set(re.findall(r"\\ref\{([^}]+)\}", text)):
        if ref not in labels:
            errors.append(f"undefined reference {ref}")
    bib = (REPORT / "refs.bib").read_text()
    keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    cited = set(k.strip() for grp in re.findall(r"\\cite[tp]?\{([^}]+)\}", text) for k in grp.split(","))
    for k in sorted(cited - keys):
        errors.append(f"citation key not in refs.bib: {k}")
    defined = set(re.findall(r"\\newcommand\{\\(\w+)\}", text))
    builtin_like = set(re.findall(r"\\(\w+)", text))
    numbers = set(re.findall(r"\\newcommand\{\\(\w+)\}", files.get("generated/numbers.tex", "")))
    for m in sorted(builtin_like):
        if m[0].isupper() and m not in defined and m not in {"Exp", "Log", "UrlBreaks"} and len(m) > 3 \
                and re.search(r"[a-z][A-Z]", m) and m not in numbers:
            errors.append(f"possibly undefined macro \\{m}")
    todos = re.findall(r"\\todo\{([^}]*)\}", main_tex)
    print(f"files checked: {len(files)}; figures: {len(re.findall(r'includegraphics', main_tex))}; "
          f"citations: {len(cited)}; labels: {len(labels)}")
    print("todo items:")
    for t in todos:
        print(f"  - {t}")
    print("unused bibliography entries:", ", ".join(sorted(keys - cited)) or "none")
    if errors:
        print("ERRORS:")
        for e in errors:
            print(f"  {e}")
        return 1
    print("no structural errors found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
