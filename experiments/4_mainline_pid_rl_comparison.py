# -*- coding: utf-8 -*-
"""Run the mainline M3/M4 PID-vs-RL comparison post-processing."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiment_suite_common import ROOT, build_python_command, run_command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run M4 PID-vs-RL comparison.")
    parser.add_argument(
        "--pid-csv",
        type=str,
        default=str(ROOT / "results" / "exp02_exact_pid_baseline" / "pid_exact_schedule_results.csv"),
    )
    parser.add_argument(
        "--rl-csv",
        type=str,
        default=str(ROOT / "results" / "exp03_exact_rl_baseline" / "rl_exact_schedule_results.csv"),
    )
    parser.add_argument("--feasible-csv", type=str, default=None)
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT / "results" / "exp04_pid_rl_comparison"),
    )
    parser.add_argument("--python", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.out_dir).resolve().mkdir(parents=True, exist_ok=True)
    cmd = build_python_command(
        "compare_exact_pid_rl_baselines.py",
        "--pid-csv",
        args.pid_csv,
        "--rl-csv",
        args.rl_csv,
        "--out-dir",
        args.out_dir,
        python_executable=args.python,
    )
    if args.feasible_csv:
        cmd.extend(["--feasible-csv", str(args.feasible_csv)])
    run_command(cmd, dry_run=bool(args.dry_run))


if __name__ == "__main__":
    main()
