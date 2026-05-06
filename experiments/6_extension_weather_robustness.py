# -*- coding: utf-8 -*-
"""Run the A1 seasonal weather-robustness experiment wrapper."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiment_suite_common import ROOT, build_python_command, run_command


SEASON_START_DATES = {
    "winter": "2024-01-01",
    "spring": "2024-04-01",
    "summer": "2024-07-01",
    "autumn": "2024-10-01",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run A1 weather robustness experiments.")
    parser.add_argument("--load", type=str, default=None, help="RL run directory for RL robustness runs.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--feasible-csv", type=str, default=None)
    parser.add_argument("--duration", type=float, default=32.0)
    parser.add_argument("--dt", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=["winter", "spring", "summer", "autumn"],
        choices=sorted(SEASON_START_DATES.keys()),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT / "results" / "exp06_weather_robustness"),
    )
    parser.add_argument("--python", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    for season in args.seasons:
        start_date = SEASON_START_DATES[season]
        pid_out = out_root / season / "pid"
        pid_out.mkdir(parents=True, exist_ok=True)
        pid_cmd = build_python_command(
            "exact_pid_schedule_baseline.py",
            "--start-date",
            start_date,
            "--duration",
            args.duration,
            "--dt",
            args.dt,
            "--seed",
            args.seed,
            "--price-model-type",
            "constant",
            "--out-dir",
            pid_out,
            python_executable=args.python,
        )
        if args.feasible_csv:
            pid_cmd.extend(["--feasible-csv", str(args.feasible_csv)])
        else:
            pid_cmd.append("--only-default")
        run_command(pid_cmd, dry_run=bool(args.dry_run))

        if args.load:
            rl_out = out_root / season / "rl"
            rl_out.mkdir(parents=True, exist_ok=True)
            rl_cmd = build_python_command(
                "exact_rl_schedule_baseline.py",
                "--load",
                args.load,
                "--device",
                args.device,
                "--start-date",
                start_date,
                "--duration",
                args.duration,
                "--dt",
                args.dt,
                "--seed",
                args.seed,
                "--price-model-type",
                "constant",
                "--out-dir",
                rl_out,
                python_executable=args.python,
            )
            if args.feasible_csv:
                rl_cmd.extend(["--feasible-csv", str(args.feasible_csv)])
            else:
                rl_cmd.append("--only-default")
            run_command(rl_cmd, dry_run=bool(args.dry_run))


if __name__ == "__main__":
    main()
