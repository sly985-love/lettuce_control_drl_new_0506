# -*- coding: utf-8 -*-
"""Shared helpers for the numbered paper experiment entrypoints."""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = ROOT / "experiments"


def build_python_command(
    script_name: str,
    *args: object,
    python_executable: str | None = None,
) -> list[str]:
    """Build a repo-root command that executes one experiment script."""
    cmd = [python_executable or sys.executable, str(EXPERIMENTS_DIR / script_name)]
    for arg in args:
        if arg is None:
            continue
        cmd.append(str(arg))
    return cmd


def run_command(
    cmd: Iterable[object],
    *,
    dry_run: bool = False,
    cwd: Path | None = None,
) -> None:
    """Print and optionally execute a command."""
    resolved = [str(part) for part in list(cmd)]
    print("[RUN]", shlex.join(resolved))
    if dry_run:
        return
    subprocess.run(resolved, cwd=str(cwd or ROOT), check=True)
