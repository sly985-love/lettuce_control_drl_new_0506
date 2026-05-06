# -*- coding: utf-8 -*-
"""Run the B1 economic-sensitivity wrapper."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiment_suite_common import ROOT, build_python_command, run_command


DEFAULT_VARIATIONS = {
    "electricity_price": [0.37, 0.56, 0.74, 0.93, 1.11, 1.30],
    "co2_price": [0.27, 0.41, 0.54, 0.68, 0.81],
    "lettuce_price_fw": [24.0, 32.0, 40.0, 48.0, 56.0],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run B1 economic sensitivity.")
    parser.add_argument(
        "--controllers",
        nargs="+",
        default=["pid"],
        choices=["pid", "rl"],
    )
    parser.add_argument("--load", type=str, default=None, help="RL run directory when controllers include rl.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--schedule",
        type=str,
        default="t1=14,t2=14,N1=20,rho2=36",
    )
    parser.add_argument("--start-date", type=str, default="2024-01-01")
    parser.add_argument("--duration", type=float, default=32.0)
    parser.add_argument("--dt", type=float, default=600.0)
    parser.add_argument(
        "--variable",
        type=str,
        default="electricity_price",
        choices=sorted(DEFAULT_VARIATIONS.keys()),
    )
    parser.add_argument(
        "--values",
        type=float,
        nargs="+",
        default=None,
        help="Override the default variation grid.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT / "results" / "exp09_economic_sensitivity"),
    )
    parser.add_argument("--python", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    values = list(args.values) if args.values else list(DEFAULT_VARIATIONS[args.variable])
    controllers = [str(item).strip().lower() for item in args.controllers]
    if "rl" in controllers and not args.load:
        raise RuntimeError("--load is required when controllers include rl.")

    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    for value in values:
        tag = f"{args.variable}_{str(value).replace('.', 'p')}"
        for controller in controllers:
            cmd = build_python_command(
                "simulate_hangzhou.py",
                "--controller",
                controller,
                "--schedule",
                args.schedule,
                "--start-date",
                args.start_date,
                "--duration",
                args.duration,
                "--dt",
                args.dt,
                "--price_model_type",
                "constant",
                "--out-dir",
                out_root / tag / controller,
                "--device",
                args.device,
                python_executable=args.python,
            )
            if args.variable == "electricity_price":
                cmd.extend(["--electricity_price", str(value)])
            elif args.variable == "co2_price":
                cmd.extend(["--co2_price", str(value)])
            elif args.variable == "lettuce_price_fw":
                cmd.extend(["--lettuce_price_fw", str(value)])
            if controller == "rl":
                cmd.extend(["--load", str(args.load)])
            run_command(cmd, dry_run=bool(args.dry_run))


if __name__ == "__main__":
    main()
