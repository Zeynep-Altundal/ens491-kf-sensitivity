"""Make the repository root importable, so that ``synthetic`` can be reused without copying it.

The root is taken from ``RVIZ_DRONE_REPO_ROOT`` if set; otherwise it is found from the real path
of this file (the workspace is built with ``--symlink-install``, so the installed module resolves
to the source tree ``<repo>/simulation/rviz_drone/rviz_drone/``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def repo_root() -> Path:
    env = os.environ.get("RVIZ_DRONE_REPO_ROOT")
    if env:
        return Path(env).resolve()
    for parent in Path(os.path.realpath(__file__)).parents:
        if (parent / "synthetic" / "imu_model.py").is_file():
            return parent
    raise RuntimeError("repository root with synthetic/ not found; set RVIZ_DRONE_REPO_ROOT")


_root = str(repo_root())
if _root not in sys.path:
    sys.path.insert(0, _root)
