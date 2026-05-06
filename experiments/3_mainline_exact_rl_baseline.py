# -*- coding: utf-8 -*-
"""Run the mainline M3 exact RL schedule baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiment_suite_common import ROOT, build_python_command, run_command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run M3 exact RL baseline.")
    parser.add_argument("--load", type=str, required=True, help="RL run directory.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--load-checkpoint", type=str, default="auto")
    parser.add_argument("--feasible-csv", type=str, default=None)
    parser.add_argument("--weather-path", type=str, default=None)
    parser.add_argument("--start-date", type=str, default="2024-01-01")
    parser.add_argument("--duration", type=float, default=365.0)
    parser.add_argument("--dt", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--merge-shards-only", action="store_true")
    parser.add_argument("--only-default", action="store_true")
    parser.add_argument("--schedule-key", type=str, default=None)
    parser.add_argument("--explore", action="store_true")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Optional output directory. Defaults to results/exp03_exact_rl_baseline.",
    )
    parser.add_argument("--save-detailed-traces", action="store_true")
    parser.add_argument("--save-batch-trajectories", action="store_true")
    parser.add_argument("--detailed-trace-dir", type=str, default=None)
    parser.add_argument("--batch-trace-dir", type=str, default=None)
    parser.add_argument("--I1_manual", type=float, default=None)
    parser.add_argument("--I2_manual", type=float, default=None)
    parser.add_argument("--photo_period_manual", type=int, default=None)
    parser.add_argument(
        "--light-control-mode",
        type=str,
        default=None,
        choices=["step", "daily_hold", "segmented_hold"],
    )
    parser.add_argument("--light-segments-per-photoperiod", type=int, default=None)
    parser.add_argument("--python", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (
        ROOT / "results" / "exp03_exact_rl_baseline"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_python_command(
        "exact_rl_schedule_baseline.py",
        "--load",
        args.load,
        "--device",
        args.device,
        "--load-checkpoint",
        args.load_checkpoint,
        "--start-date",
        args.start_date,
        "--duration",
        args.duration,
        "--dt",
        args.dt,
        "--seed",
        args.seed,
        "--num-shards",
        args.num_shards,
        "--shard-id",
        args.shard_id,
        "--price-model-type",
        "constant",
        "--out-dir",
        out_dir,
        python_executable=args.python,
    )
    if args.feasible_csv:
        cmd.extend(["--feasible-csv", str(args.feasible_csv)])
    if args.weather_path:
        cmd.extend(["--weather-path", str(args.weather_path)])
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    if args.overwrite:
        cmd.append("--overwrite")
    if args.merge_shards_only:
        cmd.append("--merge-shards-only")
    if args.only_default:
        cmd.append("--only-default")
    if args.schedule_key:
        cmd.extend(["--schedule-key", str(args.schedule_key)])
    if args.explore:
        cmd.append("--explore")
    if args.save_detailed_traces:
        cmd.append("--save-detailed-traces")
    if args.save_batch_trajectories:
        cmd.append("--save-batch-trajectories")
    if args.detailed_trace_dir:
        cmd.extend(["--detailed-trace-dir", str(args.detailed_trace_dir)])
    if args.batch_trace_dir:
        cmd.extend(["--batch-trace-dir", str(args.batch_trace_dir)])
    if args.I1_manual is not None:
        cmd.extend(["--I1_manual", str(args.I1_manual)])
    if args.I2_manual is not None:
        cmd.extend(["--I2_manual", str(args.I2_manual)])
    if args.photo_period_manual is not None:
        cmd.extend(["--photo-period-manual", str(args.photo_period_manual)])
    if args.light_control_mode:
        cmd.extend(["--light-control-mode", str(args.light_control_mode)])
    if args.light_segments_per_photoperiod is not None:
        cmd.extend(
            [
                "--light-segments-per-photoperiod",
                str(args.light_segments_per_photoperiod),
            ]
        )
    run_command(cmd, dry_run=bool(args.dry_run))


if __name__ == "__main__":
    main()
