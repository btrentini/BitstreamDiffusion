#!/usr/bin/env python3
"""Check Python style and the hard 79-column limit for protein code."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    """Run Ruff and check every source line in its configured scope."""
    command = [sys.executable, "-m", "ruff"]
    environment = {**os.environ, "RAYON_NUM_THREADS": "1"}
    listing = subprocess.run(
        command + ["check", "--show-files", "."],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    passed = True
    for filename in listing.stdout.splitlines():
        path = Path(filename)
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if len(line.expandtabs(8)) > 79:
                print(
                    f"{path.relative_to(ROOT)}:{number}: "
                    "line exceeds 79 columns"
                )
                passed = False
    for arguments in (["check", "."], ["format", "--check", "."]):
        result = subprocess.run(
            command + arguments, cwd=ROOT, env=environment, check=False
        )
        passed = passed and result.returncode == 0
    if passed:
        print("Python style checks and the 79-column limit passed.")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
