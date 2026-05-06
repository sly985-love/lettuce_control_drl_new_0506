# -*- coding: utf-8 -*-
"""Run the B2 TOU-tariff extension wrapper."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiment_suite_common import ROOT, build_python_command, run_command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run B2 TOU tariff extension.")
    parser.add_argument(
        "--task",
        type=str,
        default="simulate",
        choices=["simulate", "train"],
    )
    parser.add_argument(
        "--controller",
        type=str,
        default="pid",
        choices=["pid", "rl"],
    )
    parser.add_argument("--load", type=str, default=None, help="RL run directory for rl simulation.")
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
        "--tou-tariff-scenario",
        type=str,
        default="zhejiang_industrial_lt_1kv_2025_04",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="exp10_tou_train",
        help="Training experiment name when task=train.",
    )
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--include-price-observation", action="store_true")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT / "results" / "exp10_tou_tariff"),
    )
    parser.add_argument("--python", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if args.task == "simulate":
        if args.controller == "rl" and not args.load:
            raise RuntimeError("--load is required for rl simulation.")
        cmd = build_python_command(
            "simulate_hangzhou.py",
            "--controller",
            args.controller,
            "--schedule",
            args.schedule,
            "--start-date",
            args.start_date,
            "--duration",
            args.duration,
            "--dt",
            args.dt,
            "--price_model_type",
            "time_of_use",
            "--tou_tariff_scenario",
            args.tou_tariff_scenario,
            "--out-dir",
            out_root / "simulate" / args.controller,
            "--device",
            args.device,
            python_executable=args.python,
        )
        if args.controller == "rl":
            cmd.extend(["--load", str(args.load)])
        run_command(cmd, dry_run=bool(args.dry_run))
        return

    cmd = build_python_command(
        "train_pfal_contextual.py",
        "--experiment",
        args.experiment,
        "--device",
        args.device,
        "--price_model_type",
        "time_of_use",
        "--tou_tariff_scenario",
        args.tou_tariff_scenario,
        python_executable=args.python,
    )
    if args.epoch is not None:
        cmd.extend(["--epoch", str(args.epoch)])
    if args.include_price_observation:
        cmd.append("--include_electricity_price_observation")
    run_command(cmd, dry_run=bool(args.dry_run))


if __name__ == "__main__":
    main()
