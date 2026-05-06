# -*- coding: utf-8 -*-
"""Run the A3 RL-ablation wrapper."""

from __future__ import annotations

import argparse

from experiment_suite_common import build_python_command, run_command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run A3 RL ablation suite.")
    parser.add_argument("--suite-name", type=str, default="exp08_rl_ablation")
    parser.add_argument("--suite-profile", type=str, default="screening_core")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--runtime-profile", type=str, default="pilot_fast")
    parser.add_argument("--horizon-profile", type=str, default=None)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--train-num", type=int, default=None)
    parser.add_argument("--test-num", type=int, default=None)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--python", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cmd = build_python_command(
        "run_rl_ablation_suite.py",
        "--suite_name",
        args.suite_name,
        "--suite_profile",
        args.suite_profile,
        "--device",
        args.device,
        "--runtime_profile",
        args.runtime_profile,
        python_executable=args.python,
    )
    if args.horizon_profile:
        cmd.extend(["--horizon_profile", str(args.horizon_profile)])
    if args.epoch is not None:
        cmd.extend(["--epoch", str(args.epoch)])
    if args.batch_size is not None:
        cmd.extend(["--batch_size", str(args.batch_size)])
    if args.train_num is not None:
        cmd.extend(["--train_num", str(args.train_num)])
    if args.test_num is not None:
        cmd.extend(["--test_num", str(args.test_num)])
    if args.summary_only:
        cmd.append("--summary_only")
    if args.skip_existing:
        cmd.append("--skip_existing")
    run_command(cmd, dry_run=bool(args.dry_run))


if __name__ == "__main__":
    main()
