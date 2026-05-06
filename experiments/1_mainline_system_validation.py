# -*- coding: utf-8 -*-
"""Run the mainline M1 system-validation simulations."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiment_suite_common import ROOT, build_python_command, run_command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run M1 system-validation simulations for code_0420."
    )
    parser.add_argument(
        "--controllers",
        nargs="+",
        default=["pid"],
        choices=["pid", "rl"],
        help="Controllers to run. Add rl only when --load is provided.",
    )
    parser.add_argument("--load", type=str, default=None, help="RL run directory for rl validation.")
    parser.add_argument("--device", type=str, default="cpu", help="RL inference device.")
    parser.add_argument("--start-date", type=str, default="2024-01-01")
    parser.add_argument("--duration", type=float, default=32.0)
    parser.add_argument("--dt", type=float, default=600.0)
    parser.add_argument(
        "--schedule",
        type=str,
        default="t1=14,t2=14,N1=20,rho2=36",
        help='Upper schedule string such as "t1=14,t2=14,N1=20,rho2=36".',
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT / "results" / "exp01_system_validation"),
    )
    parser.add_argument("--python", type=str, default=None, help="Python executable.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    controllers = [str(item).strip().lower() for item in args.controllers]
    if "rl" in controllers and not args.load:
        raise RuntimeError("--load is required when controllers include rl.")

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
            out_root / controller,
            "--device",
            args.device,
            python_executable=args.python,
        )
        if controller == "rl":
            cmd.extend(["--load", str(args.load)])
        run_command(cmd, dry_run=bool(args.dry_run))


if __name__ == "__main__":
    main()
