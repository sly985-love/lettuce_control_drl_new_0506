# -*- coding: utf-8 -*-
"""Run the mainline M2 exact PID schedule baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiment_suite_common import ROOT, build_python_command, run_command


DEFAULT_OUT_DIR = ROOT / "results" / "exp02_exact_pid_baseline"


def _format_float_token(value: float | None) -> str | None:
    if value is None:
        return None
    text = f"{float(value):.6g}"
    return text.replace("-", "m").replace(".", "p")


def _build_scenario_suffix(args: argparse.Namespace) -> str:
    parts: list[str] = []
    if args.photo_period_manual is not None:
        parts.append(f"pp{int(args.photo_period_manual)}h")
    if args.I1_manual is not None:
        parts.append(f"i1{_format_float_token(args.I1_manual)}")
    if args.I2_manual is not None:
        parts.append(f"i2{_format_float_token(args.I2_manual)}")
    if args.light_control_mode:
        parts.append(str(args.light_control_mode))
        if args.light_control_mode == "segmented_hold" and args.light_segments_per_photoperiod is not None:
            parts.append(f"seg{int(args.light_segments_per_photoperiod)}")
    has_economic_override = (
        args.price_model_type
        or args.tou_tariff_scenario
        or args.electricity_price is not None
        or args.co2_price is not None
        or args.lettuce_price_fw is not None
        or args.constant_price is not None
    )
    if has_economic_override:
        effective_price_model = str(
            args.price_model_type or ("time_of_use" if args.tou_tariff_scenario else "constant")
        )
        parts.append(f"pm_{effective_price_model}")
        if args.tou_tariff_scenario:
            parts.append(f"tou_{str(args.tou_tariff_scenario)}")
        if args.electricity_price is not None:
            parts.append(f"e{_format_float_token(args.electricity_price)}")
        if args.co2_price is not None:
            parts.append(f"co2{_format_float_token(args.co2_price)}")
        if args.lettuce_price_fw is not None:
            parts.append(f"l{_format_float_token(args.lettuce_price_fw)}")
        if args.constant_price is not None:
            parts.append(f"cp{_format_float_token(args.constant_price)}")
    return "__".join(parts)


def _resolve_effective_out_dir(args: argparse.Namespace) -> Path:
    out_dir = Path(args.out_dir).resolve()
    default_out_dir = DEFAULT_OUT_DIR.resolve()
    if out_dir != default_out_dir:
        return out_dir
    suffix = _build_scenario_suffix(args)
    if not suffix:
        return out_dir
    return out_dir.parent / f"{out_dir.name}__{suffix}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run M2 exact PID baseline.")
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
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_OUT_DIR),
    )
    parser.add_argument("--save-detailed-traces", action="store_true")
    parser.add_argument("--save-batch-trajectories", action="store_true")
    parser.add_argument(
        "--export-selected-traces-only",
        "--export_selected_traces_only",
        dest="export_selected_traces_only",
        action="store_true",
    )
    parser.add_argument(
        "--selected-trace-top-k-valid",
        "--selected_trace_top_k_valid",
        dest="selected_trace_top_k_valid",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--selected-trace-include-default",
        "--selected_trace_include_default",
        dest="selected_trace_include_default",
        action="store_true",
    )
    parser.add_argument(
        "--selected-trace-schedule-key",
        "--selected_trace_schedule_key",
        dest="selected_trace_schedule_keys",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--selected-trace-save-batch-trajectories",
        "--selected_trace_save_batch_trajectories",
        dest="selected_trace_save_batch_trajectories",
        action="store_true",
    )
    parser.add_argument("--detailed-trace-dir", type=str, default=None)
    parser.add_argument("--batch-trace-dir", type=str, default=None)
    parser.add_argument(
        "--selected-detailed-trace-dir",
        "--selected_detailed_trace_dir",
        dest="selected_detailed_trace_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--selected-batch-trace-dir",
        "--selected_batch_trace_dir",
        dest="selected_batch_trace_dir",
        type=str,
        default=None,
    )
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
    parser.add_argument(
        "--price-model-type",
        "--price_model_type",
        dest="price_model_type",
        type=str,
        default=None,
        choices=["constant", "time_of_use"],
    )
    parser.add_argument(
        "--tou-tariff-scenario",
        "--tou_tariff_scenario",
        dest="tou_tariff_scenario",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--electricity-price",
        "--electricity_price",
        dest="electricity_price",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--co2-price",
        "--co2_price",
        dest="co2_price",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--lettuce-price-fw",
        "--lettuce_price_fw",
        dest="lettuce_price_fw",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--constant-price",
        "--constant_price",
        dest="constant_price",
        type=float,
        default=None,
    )
    parser.add_argument("--python", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    effective_out_dir = _resolve_effective_out_dir(args)
    if not args.dry_run:
        effective_out_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_python_command(
        "exact_pid_schedule_baseline.py",
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
        str(args.price_model_type or "constant"),
        "--out-dir",
        str(effective_out_dir),
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
    if args.save_detailed_traces:
        cmd.append("--save-detailed-traces")
    if args.save_batch_trajectories:
        cmd.append("--save-batch-trajectories")
    if args.export_selected_traces_only:
        cmd.append("--export-selected-traces-only")
    if args.selected_trace_top_k_valid:
        cmd.extend(["--selected-trace-top-k-valid", str(args.selected_trace_top_k_valid)])
    if args.selected_trace_include_default:
        cmd.append("--selected-trace-include-default")
    for schedule_key in args.selected_trace_schedule_keys:
        cmd.extend(["--selected-trace-schedule-key", str(schedule_key)])
    if args.selected_trace_save_batch_trajectories:
        cmd.append("--selected-trace-save-batch-trajectories")
    if args.detailed_trace_dir:
        cmd.extend(["--detailed-trace-dir", str(args.detailed_trace_dir)])
    if args.batch_trace_dir:
        cmd.extend(["--batch-trace-dir", str(args.batch_trace_dir)])
    if args.selected_detailed_trace_dir:
        cmd.extend(["--selected-detailed-trace-dir", str(args.selected_detailed_trace_dir)])
    if args.selected_batch_trace_dir:
        cmd.extend(["--selected-batch-trace-dir", str(args.selected_batch_trace_dir)])
    if args.I1_manual is not None:
        cmd.extend(["--I1_manual", str(args.I1_manual)])
    if args.I2_manual is not None:
        cmd.extend(["--I2_manual", str(args.I2_manual)])
    if args.photo_period_manual is not None:
        cmd.extend(["--photo_period_manual", str(args.photo_period_manual)])
    if args.light_control_mode:
        cmd.extend(["--light-control-mode", str(args.light_control_mode)])
    if args.light_segments_per_photoperiod is not None:
        cmd.extend(
            [
                "--light-segments-per-photoperiod",
                str(args.light_segments_per_photoperiod),
            ]
        )
    if args.tou_tariff_scenario:
        cmd.extend(["--tou-tariff-scenario", str(args.tou_tariff_scenario)])
    if args.electricity_price is not None:
        cmd.extend(["--electricity-price", str(args.electricity_price)])
    if args.co2_price is not None:
        cmd.extend(["--co2-price", str(args.co2_price)])
    if args.lettuce_price_fw is not None:
        cmd.extend(["--lettuce-price-fw", str(args.lettuce_price_fw)])
    if args.constant_price is not None:
        cmd.extend(["--constant-price", str(args.constant_price)])
    run_command(cmd, dry_run=bool(args.dry_run))


if __name__ == "__main__":
    main()
