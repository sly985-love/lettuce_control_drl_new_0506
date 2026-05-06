# -*- coding: utf-8 -*-
"""
Exact RL schedule baseline over the feasible upper-schedule set.

This script mirrors `exact_pid_schedule_baseline.py`, but replaces the
lower-level PID controller with a trained RL policy loaded from a finished run.
The goal is research fairness: evaluate the same feasible schedule set, under
the same weather window, duration, and dt, then compare RL against PID
schedule-by-schedule.

Typical usage:

  python experiments/exact_rl_schedule_baseline.py ^
    --load C:\\path\\to\\trained_run ^
    --duration 365 --dt 600

  python experiments/exact_rl_schedule_baseline.py ^
    --load log/PFAL-contextual-SAC/sac_contextual/my_exp ^
    --only-default --duration 28
"""

from __future__ import annotations

import argparse
import csv
import importlib.machinery
import json
import os
import sys
import time
import types
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_SILENT", "true")
os.environ.setdefault("WANDB_DISABLE_CODE", "true")
os.environ.setdefault("WANDB_DISABLED", "true")

if "wandb" not in sys.modules:
    def _noop(*args, **kwargs):
        return None

    class _DummyRun:
        project = "disabled"
        name = "disabled"
        url = ""

        def log(self, *args, **kwargs):
            return None

        def finish(self, *args, **kwargs):
            return None

    wandb_stub = types.ModuleType("wandb")
    wandb_stub.init = lambda *args, **kwargs: _DummyRun()
    wandb_stub.log = _noop
    wandb_stub.finish = _noop
    wandb_stub.define_metric = _noop
    wandb_stub.Image = lambda *args, **kwargs: None
    wandb_stub.Artifact = lambda *args, **kwargs: None
    wandb_stub.Table = lambda *args, **kwargs: None
    wandb_stub.config = {}
    wandb_stub.run = None
    wandb_stub.__file__ = "<wandb_stub>"
    wandb_stub.__spec__ = importlib.machinery.ModuleSpec(
        name="wandb",
        loader=None,
    )
    sys.modules["wandb"] = wandb_stub

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.exact_pid_schedule_baseline import (  # noqa: E402
    DEFAULT_SCHEDULE,
    FEASIBLE_CSV_DEFAULT,
    RESULT_FIELDS,
    RESULT_SCHEMA_VERSION,
    _build_progress_timing_line,
    _ensure_results_schema,
    append_result,
    build_result_row,
    count_shard_candidates,
    generate_exact_baseline_plots,
    is_default_schedule,
    iter_schedules,
    load_completed_results,
    load_feasible_schedules,
    load_weather_window,
    save_schedule_trace_csvs,
    schedule_key,
    summarise_results,
)
from experiments.simulate_hangzhou import (  # noqa: E402
    PFALEnvContextual,
    _run_rl_simulation,
    build_env_config,
    build_price_override_kwargs,
    load_rl_policy,
)
from rl.drl_based_control import resolve_policy_checkpoint_path  # noqa: E402


RL_PLOT_FILENAMES = (
    "rl_exact_paper_summary.png",
    "rl_exact_design_heatmaps.png",
    "rl_exact_top10_valid.png",
    "rl_exact_termination_reasons.png",
)


def _resolve_default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _resolve_default_out_dir(load_ref: str | None) -> Path:
    label = "unknown_policy"
    if load_ref:
        label = Path(str(load_ref)).name or "unknown_policy"
    return ROOT / "results" / f"exact_rl_baseline_{label}"


def _summarise_episode_records(
    records: List[Dict[str, Any]],
    *,
    env: PFALEnvContextual,
    dt_seconds: float,
) -> Dict[str, Any]:
    """将一条 RL 闭环仿真轨迹汇总为与 PID exact baseline 一致的结果口径。"""
    if not records:
        return {
            "steps_executed": 0,
            "sim_days_executed": 0.0,
            "episode_length_steps": int(getattr(env, "episode_length", 0)),
            "episode_completion_ratio": 0.0,
            "termination_reason": "no_records",
            "terminated_early": True,
            "valid_full_horizon": False,
            "objective_value": -1.0e12,
            "net_profit": -1.0e12,
            "revenue": 0.0,
            "total_cost": 0.0,
            "energy_kwh": 0.0,
            "harvest_kg": 0.0,
            "harvest_dry_kg": 0.0,
            "harvest_fresh_kg": 0.0,
            "total_harvests": 0,
            "total_transplants": 0,
            "avg_harvest_kg_per_event": 0.0,
            "avg_harvest_dry_kg_per_event": 0.0,
            "avg_harvest_fresh_kg_per_event": 0.0,
            "avg_harvest_dry_g_per_plant": 0.0,
            "avg_harvest_fresh_g_per_plant": 0.0,
            "cost_per_kg": float("inf"),
            "revenue_per_kg": 0.0,
            "final_biomass_total_kg_m2": 0.0,
            "final_biomass_transplant_kg_m2": 0.0,
            "final_biomass_seedling_kg_m2": 0.0,
            "final_biomass_finishing_kg_m2": 0.0,
            "final_biomass_dense_kg_m2": 0.0,
            "final_temp_c": 0.0,
            "final_rh_pct": 0.0,
            "final_co2_ppm": 0.0,
            "final_vpd_kpa": 0.0,
            "cum_reward": 0.0,
        }

    last = records[-1]
    c_fw = float(getattr(env, "crop_params", {}).get("c_fw", 22.5))
    harvest_dry_kg = float(
        sum(float(r.get("harvest_dry_mass_g", r.get("harvest_mass_g", 0.0))) for r in records) / 1000.0
    )
    harvest_fresh_kg = float(
        sum(
            float(
                r.get(
                    "harvest_fresh_mass_equiv_g",
                    float(r.get("harvest_dry_mass_g", r.get("harvest_mass_g", 0.0))) * c_fw,
                )
            )
            for r in records
        )
        / 1000.0
    )
    total_cost = float(last.get("cum_cost", 0.0))
    energy_kwh = float(sum(float(r.get("E_step_kWh", 0.0)) for r in records))
    revenue = float(harvest_fresh_kg * float(getattr(env, "c_lettuce", 0.0)))
    net_profit = float(revenue - total_cost)
    termination_reason = str(getattr(env, "last_termination_reason", "unknown"))
    completion_ratio = float(
        getattr(
            env,
            "last_episode_completion_ratio",
            len(records) / max(float(getattr(env, "episode_length", len(records))), 1.0),
        )
    )
    terminated_early = bool(getattr(env, "last_episode_ended_early", False))
    valid_full_horizon = bool(
        termination_reason == "time_limit" and completion_ratio >= 0.999
    )
    objective_value = float(net_profit if valid_full_horizon else (-1.0e12 + net_profit))
    total_harvests = int(last.get("total_harvests", 0))
    total_transplants = int(last.get("total_transplants", 0))
    harvest_rows = [
        r for r in records if float(r.get("harvest_dry_mass_g", r.get("harvest_mass_g", 0.0))) > 0.0
    ]
    avg_harvest_kg_per_event = float(harvest_fresh_kg / total_harvests) if total_harvests > 0 else 0.0
    avg_harvest_dry_kg_per_event = float(harvest_dry_kg / total_harvests) if total_harvests > 0 else 0.0
    avg_harvest_fresh_kg_per_event = float(harvest_fresh_kg / total_harvests) if total_harvests > 0 else 0.0
    avg_harvest_dry_g_per_plant = float(
        sum(float(r.get("harvest_mean_dry_mass_per_plant_g", 0.0)) for r in harvest_rows) / len(harvest_rows)
    ) if harvest_rows else 0.0
    avg_harvest_fresh_g_per_plant = float(
        sum(float(r.get("harvest_mean_fresh_mass_per_plant_g", 0.0)) for r in harvest_rows) / len(harvest_rows)
    ) if harvest_rows else 0.0
    cost_per_kg = float(total_cost / harvest_fresh_kg) if harvest_fresh_kg > 0 else float("inf")
    revenue_per_kg = float(revenue / harvest_fresh_kg) if harvest_fresh_kg > 0 else 0.0

    return {
        "steps_executed": int(len(records)),
        "sim_days_executed": float(len(records) * dt_seconds / 86400.0),
        "episode_length_steps": int(getattr(env, "episode_length", len(records))),
        "episode_completion_ratio": completion_ratio,
        "termination_reason": termination_reason,
        "terminated_early": terminated_early,
        "valid_full_horizon": valid_full_horizon,
        "objective_value": objective_value,
        "net_profit": net_profit,
        "revenue": revenue,
        "total_cost": total_cost,
        "energy_kwh": energy_kwh,
        "harvest_kg": harvest_fresh_kg,
        "harvest_dry_kg": harvest_dry_kg,
        "harvest_fresh_kg": harvest_fresh_kg,
        "total_harvests": total_harvests,
        "total_transplants": total_transplants,
        "avg_harvest_kg_per_event": avg_harvest_kg_per_event,
        "avg_harvest_dry_kg_per_event": avg_harvest_dry_kg_per_event,
        "avg_harvest_fresh_kg_per_event": avg_harvest_fresh_kg_per_event,
        "avg_harvest_dry_g_per_plant": avg_harvest_dry_g_per_plant,
        "avg_harvest_fresh_g_per_plant": avg_harvest_fresh_g_per_plant,
        "cost_per_kg": cost_per_kg,
        "revenue_per_kg": revenue_per_kg,
        "final_biomass_total_kg_m2": float(last.get("biomass_total_kg_m2", 0.0)),
        "final_biomass_transplant_kg_m2": float(last.get("biomass_transplant_kg_m2", 0.0)),
        "final_biomass_seedling_kg_m2": float(last.get("biomass_seedling_kg_m2", 0.0)),
        "final_biomass_finishing_kg_m2": float(last.get("biomass_finishing_kg_m2", 0.0)),
        "final_biomass_dense_kg_m2": float(last.get("biomass_dense_kg_m2", 0.0)),
        "final_temp_c": float(last.get("T_in", 0.0)),
        "final_rh_pct": float(last.get("RH_pct", 0.0)),
        "final_co2_ppm": float(last.get("C_ppm", 0.0)),
        "final_vpd_kpa": float(last.get("VPD_kPa", 0.0)),
        "cum_reward": float(last.get("cum_reward", 0.0)),
    }


def evaluate_schedule_rl(
    schedule: Dict[str, Any],
    weather_rows: List[Dict[str, Any]],
    dt_seconds: float,
    seed: int,
    *,
    policy,
    run_cfg: Dict[str, Any] | None,
    photo_period_manual: int | None = None,
    I1_manual: float | None = None,
    I2_manual: float | None = None,
    light_control_mode: str | None = None,
    light_segments_per_photoperiod: int | None = None,
    price_override_kwargs: Dict[str, Any] | None = None,
    explore: bool = False,
    return_traces: bool = False,
) -> Dict[str, Any] | tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """在给定排程上运行 RL 闭环控制并返回 exact-baseline 摘要。"""
    env_cfg = build_env_config(
        schedule,
        dt_seconds,
        seed=seed,
        photo_period_override=photo_period_manual,
        light_control_mode=light_control_mode,
        light_segments_per_photoperiod=light_segments_per_photoperiod,
        run_config_overrides=run_cfg if isinstance(run_cfg, dict) else None,
        **dict(price_override_kwargs or {}),
    )
    env = PFALEnvContextual(env_cfg)
    records, batch_records = _run_rl_simulation(
        env,
        weather_rows,
        schedule,
        dt_seconds,
        seed,
        policy=policy,
        explore=explore,
        photo_period_manual=photo_period_manual,
        I1_manual=I1_manual,
        I2_manual=I2_manual,
    )
    summary = _summarise_episode_records(records, env=env, dt_seconds=dt_seconds)
    if return_traces:
        return summary, records, batch_records
    return summary


def shard_suffix(shard_id: int, num_shards: int) -> str:
    return f"shard_{shard_id + 1:02d}_of_{num_shards:02d}"


def resolve_output_paths(
    out_dir: Path,
    *,
    shard_id: int,
    num_shards: int,
    merged: bool = False,
) -> tuple[Path, Path, Path, List[Path]]:
    if num_shards <= 1 or merged:
        results_csv = out_dir / "rl_exact_schedule_results.csv"
        ranked_csv = out_dir / "rl_exact_schedule_results_ranked.csv"
        summary_json = out_dir / "rl_exact_schedule_summary.json"
        plot_paths = [out_dir / name for name in RL_PLOT_FILENAMES]
        return results_csv, ranked_csv, summary_json, plot_paths

    suffix = shard_suffix(shard_id, num_shards)
    results_csv = out_dir / f"rl_exact_schedule_results.{suffix}.csv"
    ranked_csv = out_dir / f"rl_exact_schedule_results_ranked.{suffix}.csv"
    summary_json = out_dir / f"rl_exact_schedule_summary.{suffix}.json"
    plot_paths = [
        out_dir / f"{Path(name).stem}.{suffix}{Path(name).suffix}" for name in RL_PLOT_FILENAMES
    ]
    return results_csv, ranked_csv, summary_json, plot_paths


def merge_shard_results(
    out_dir: Path,
    *,
    num_shards: int,
    merged_results_csv: Path,
) -> List[Path]:
    shard_paths = [
        resolve_output_paths(out_dir, shard_id=shard_idx, num_shards=num_shards, merged=False)[0]
        for shard_idx in range(num_shards)
    ]
    missing = [path for path in shard_paths if not path.exists()]
    if missing:
        raise RuntimeError(
            "Cannot merge shard results because some shard CSV files are missing: "
            + ", ".join(str(path) for path in missing)
        )

    frames = []
    for path in shard_paths:
        _ensure_results_schema(path)
        frames.append(pd.read_csv(path))
    merged = pd.concat(frames, ignore_index=True)
    if merged.empty:
        raise RuntimeError("Merged shard results are empty.")
    merged = merged.sort_values(
        by=["eval_index", "schedule_key"],
        ascending=[True, True],
    ).drop_duplicates(subset=["schedule_key"], keep="first")
    merged_results_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(merged_results_csv, index=False, encoding="utf-8")
    return shard_paths


def update_summary_metadata(
    summary_json: Path,
    *,
    load_ref: str | None,
    device: str,
    explore: bool,
    load_checkpoint: str | None = None,
    resolved_checkpoint_kind: str | None = None,
    resolved_checkpoint_path: str | None = None,
    I1_manual: float | None = None,
    I2_manual: float | None = None,
    photo_period_manual: int | None = None,
    light_control_mode: str | None = None,
    light_segments_per_photoperiod: int | None = None,
    controller_label: str = "rl",
) -> Dict[str, Any]:
    payload = {}
    if summary_json.exists():
        with open(summary_json, "r", encoding="utf-8") as f:
            payload = json.load(f) or {}
    payload["controller"] = str(controller_label)
    payload["load_ref"] = str(load_ref or "")
    payload["device"] = str(device)
    payload["policy_mode"] = "stochastic_sample" if explore else "deterministic_mean"
    payload["requested_load_checkpoint"] = str(load_checkpoint or "")
    payload["resolved_checkpoint_kind"] = str(resolved_checkpoint_kind or "")
    payload["resolved_checkpoint_path"] = str(resolved_checkpoint_path or "")
    payload["I1_manual"] = None if I1_manual is None else float(I1_manual)
    payload["I2_manual"] = None if I2_manual is None else float(I2_manual)
    payload["photo_period_manual"] = None if photo_period_manual is None else int(photo_period_manual)
    payload["light_control_mode"] = None if light_control_mode is None else str(light_control_mode)
    payload["light_segments_per_photoperiod"] = (
        None if light_segments_per_photoperiod is None else int(light_segments_per_photoperiod)
    )
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exact RL baseline over the feasible schedule set."
    )
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        help="Path to a trained RL run directory or experiment name.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=_resolve_default_device(),
        choices=["cpu", "cuda"],
        help="Inference device used to load the trained RL policy.",
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default="auto",
        choices=["best", "final", "selected", "auto"],
        help="Which checkpoint from the RL run to evaluate.",
    )
    parser.add_argument(
        "--explore",
        action="store_true",
        help="Sample from the learned policy distribution instead of using deterministic mean actions.",
    )
    parser.add_argument(
        "--feasible-csv",
        default=str(FEASIBLE_CSV_DEFAULT),
        help="Path to the feasible schedule CSV.",
    )
    parser.add_argument(
        "--weather-path",
        default=str(ROOT / "data" / "weather" / "weather_hangzhou_2024.csv"),
        help="Path to the Hangzhou weather CSV.",
    )
    parser.add_argument("--start-date", default="2024-01-01", help="Simulation start date YYYY-MM-DD.")
    parser.add_argument("--duration", type=float, default=365.0, help="Simulation duration [days].")
    parser.add_argument("--dt", type=float, default=600.0, help="Simulation step [s].")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the next N pending schedules.")
    parser.add_argument("--overwrite", action="store_true", help="Discard existing result CSV and restart.")
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split the feasible schedule list into this many disjoint shards for parallel exact runs.",
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help="0-based shard index to run when --num-shards > 1.",
    )
    parser.add_argument(
        "--merge-shards-only",
        action="store_true",
        help="Skip simulation, merge shard CSVs in --out-dir, and rebuild the global summary/plots.",
    )
    parser.add_argument(
        "--only-default",
        action="store_true",
        help="Evaluate only the default upper schedule x={14,14,20,36}.",
    )
    parser.add_argument(
        "--schedule-key",
        default=None,
        help='Evaluate only the specified schedule key, e.g. "t1=14|t2=14|N1=20|rho2=36".',
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory for exact-baseline outputs. Default: results/exact_rl_baseline_<experiment>.",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Skip simulation and regenerate ranked summary + paper-ready plots from an existing results CSV.",
    )
    parser.add_argument(
        "--save-detailed-traces",
        action="store_true",
        help="Save a simulate_hangzhou-style detailed step CSV for every evaluated schedule.",
    )
    parser.add_argument(
        "--save-batch-trajectories",
        action="store_true",
        help="Also save per-schedule batch-trajectory CSVs alongside detailed traces.",
    )
    parser.add_argument(
        "--detailed-trace-dir",
        default=None,
        help="Optional directory for per-schedule detailed trace CSVs. Default: <out-dir>/detailed_traces.",
    )
    parser.add_argument(
        "--batch-trace-dir",
        default=None,
        help="Optional directory for per-schedule batch trajectory CSVs. Default: <out-dir>/batch_trajectories.",
    )
    parser.add_argument("--I1_manual", type=float, default=None, help="Optional manual dense-zone PPFD override for fair RL evaluation.")
    parser.add_argument("--I2_manual", type=float, default=None, help="Optional manual finishing-zone PPFD override for fair RL evaluation.")
    parser.add_argument("--photo-period-manual", type=int, default=None, help="Optional manual photoperiod override [h/day].")
    parser.add_argument(
        "--light-control-mode",
        type=str,
        default=None,
        choices=["step", "daily_hold", "segmented_hold"],
        help="Optional light-intensity execution mode override.",
    )
    parser.add_argument(
        "--light-segments-per-photoperiod",
        type=int,
        default=None,
        help="Segments per photoperiod when --light-control-mode=segmented_hold.",
    )
    parser.add_argument(
        "--price-model-type",
        type=str,
        default=None,
        choices=["constant", "time_of_use"],
        help="Optional electricity-price model override.",
    )
    parser.add_argument(
        "--tou-tariff-scenario",
        type=str,
        default=None,
        help="Optional built-in TOU tariff scenario name.",
    )
    parser.add_argument(
        "--electricity-price",
        type=float,
        default=None,
        help="Optional fixed electricity-price override [RMB/kWh].",
    )
    parser.add_argument(
        "--co2-price",
        type=float,
        default=None,
        help="Optional CO2-price override [RMB/kg].",
    )
    parser.add_argument(
        "--lettuce-price-fw",
        type=float,
        default=None,
        help="Optional lettuce fresh-weight price override [RMB/kg].",
    )
    parser.add_argument(
        "--constant-price",
        type=float,
        default=None,
        help="Optional TOU flat/fallback electricity price [RMB/kWh].",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_shards < 1:
        raise RuntimeError("--num-shards must be at least 1.")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise RuntimeError("--shard-id must satisfy 0 <= shard-id < num-shards.")
    if args.merge_shards_only and args.num_shards <= 1:
        raise RuntimeError("--merge-shards-only requires --num-shards > 1.")
    if not args.plots_only and not args.merge_shards_only and not args.load:
        raise RuntimeError("--load is required unless only rebuilding plots or merging shards.")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else _resolve_default_out_dir(args.load)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_csv, ranked_csv, summary_json, plot_paths = resolve_output_paths(
        out_dir,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        merged=False,
    )
    merged_results_csv, merged_ranked_csv, merged_summary_json, merged_plot_paths = resolve_output_paths(
        out_dir,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        merged=True,
    )
    detailed_trace_dir = (
        Path(args.detailed_trace_dir).resolve()
        if args.detailed_trace_dir
        else (out_dir / "detailed_traces" if args.save_detailed_traces else None)
    )
    batch_trace_dir = (
        Path(args.batch_trace_dir).resolve()
        if args.batch_trace_dir
        else (out_dir / "batch_trajectories" if args.save_batch_trajectories else None)
    )

    if args.overwrite:
        for path in (results_csv, ranked_csv, summary_json, *plot_paths):
            if path.exists():
                path.unlink()
    if results_csv.exists():
        _ensure_results_schema(results_csv)

    if args.plots_only:
        if not results_csv.exists():
            raise RuntimeError(f"--plots-only requires an existing results CSV: {results_csv}")
        summary = summarise_results(results_csv, summary_json, ranked_csv)
        update_summary_metadata(
            summary_json,
            load_ref=args.load,
            device=args.device,
            explore=bool(args.explore),
            load_checkpoint=args.load_checkpoint,
        )
        generated = generate_exact_baseline_plots(
            results_csv,
            out_dir,
            plot_filenames=RL_PLOT_FILENAMES,
        )
        print("\n" + "=" * 72)
        print("Exact RL Baseline Plot Refresh")
        print("=" * 72)
        print(f"Results CSV        : {results_csv}")
        print(f"Ranked CSV         : {ranked_csv}")
        print(f"Summary JSON       : {summary_json}")
        print(f"Evaluated schedules: {summary['n_evaluated']}")
        print(f"Valid full horizon : {summary['n_valid_full_horizon']}")
        for path in generated:
            print(f"Saved plot         : {path}")
        print("=" * 72)
        return

    if args.merge_shards_only:
        shard_paths = merge_shard_results(
            out_dir,
            num_shards=args.num_shards,
            merged_results_csv=merged_results_csv,
        )
        summary = summarise_results(merged_results_csv, merged_summary_json, merged_ranked_csv)
        update_summary_metadata(
            merged_summary_json,
            load_ref=args.load,
            device=args.device,
            explore=bool(args.explore),
            load_checkpoint=args.load_checkpoint,
        )
        generated = generate_exact_baseline_plots(
            merged_results_csv,
            out_dir,
            plot_filenames=RL_PLOT_FILENAMES,
        )
        print("\n" + "=" * 72)
        print("Exact RL Baseline Shard Merge")
        print("=" * 72)
        print(f"Merged Results CSV : {merged_results_csv}")
        print(f"Merged Ranked CSV  : {merged_ranked_csv}")
        print(f"Merged Summary JSON: {merged_summary_json}")
        print(f"Shard CSV count    : {len(shard_paths)}")
        print(f"Evaluated schedules: {summary['n_evaluated']}")
        print(f"Valid full horizon : {summary['n_valid_full_horizon']}")
        for path in generated:
            print(f"Saved plot         : {path}")
        print("=" * 72)
        return

    resolved_checkpoint_path, resolved_checkpoint_kind = resolve_policy_checkpoint_path(
        args.load,
        checkpoint=args.load_checkpoint,
        project_root=ROOT,
    )
    policy, _action_space, run_cfg = load_rl_policy(
        args.load,
        device=args.device,
        checkpoint=args.load_checkpoint,
    )
    price_override_kwargs = build_price_override_kwargs(
        price_model_type=args.price_model_type,
        tou_tariff_scenario=args.tou_tariff_scenario,
        electricity_price=args.electricity_price,
        co2_price=args.co2_price,
        lettuce_price_fw=args.lettuce_price_fw,
        constant_price=args.constant_price,
    )

    schedules = load_feasible_schedules(Path(args.feasible_csv))
    if args.only_default:
        schedules = [s for s in schedules if is_default_schedule(s)]
    if args.schedule_key:
        schedules = [s for s in schedules if schedule_key(s) == str(args.schedule_key).strip()]
    if not schedules:
        raise RuntimeError("No schedules matched the requested filters.")

    completed = load_completed_results(results_csv)
    shard_candidate_count = count_shard_candidates(
        schedules,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
    )
    pending = list(
        iter_schedules(
            schedules,
            completed=completed,
            limit=args.limit,
            shard_id=args.shard_id,
            num_shards=args.num_shards,
        )
    )

    print(
        f"[INFO] Loaded {len(schedules)} feasible schedules; "
        f"shard candidates={shard_candidate_count}; "
        f"{len(completed)} already completed; {len(pending)} pending in this run."
    )
    if args.num_shards > 1:
        print(
            f"[INFO] Running shard {args.shard_id + 1}/{args.num_shards} "
            f"-> {results_csv.name}"
        )

    weather_rows: List[Dict[str, Any]] | None = None
    if pending:
        weather_rows = load_weather_window(
            Path(args.weather_path),
            args.start_date,
            args.duration,
            args.dt,
        )
        print(
            f"[INFO] Weather window: start={args.start_date}, duration={args.duration} d, "
            f"dt={args.dt}s -> {len(weather_rows)} steps."
        )
    else:
        print("[INFO] No pending schedules in this run; refreshing summary and plots only.")

    t0 = time.time()
    for local_idx, (global_idx, schedule) in enumerate(pending, start=1):
        key = schedule_key(schedule)
        schedule_start = time.time()
        evaluation = evaluate_schedule_rl(
            schedule,
            weather_rows or [],
            args.dt,
            args.seed,
            policy=policy,
            run_cfg=run_cfg,
            photo_period_manual=args.photo_period_manual,
            I1_manual=args.I1_manual,
            I2_manual=args.I2_manual,
            light_control_mode=args.light_control_mode,
            light_segments_per_photoperiod=args.light_segments_per_photoperiod,
            price_override_kwargs=price_override_kwargs,
            explore=bool(args.explore),
            return_traces=bool(args.save_detailed_traces or args.save_batch_trajectories),
        )
        if args.save_detailed_traces or args.save_batch_trajectories:
            summary, records, batch_records = evaluation
            save_schedule_trace_csvs(
                schedule,
                detailed_trace_dir=detailed_trace_dir,
                records=records,
                batch_trace_dir=batch_trace_dir,
                batch_records=batch_records,
            )
        else:
            summary = evaluation
        row = build_result_row(
            schedule,
            summary,
            eval_index=global_idx,
            duration_days=args.duration,
            dt_seconds=args.dt,
        )
        append_result(results_csv, row)
        elapsed = time.time() - schedule_start
        print(
            f"[RL-EXACT]  {local_idx:4d}/{len(pending):4d} "
            f"{key} | valid={row['valid_full_horizon']} "
            f"| reason={row['termination_reason']} "
            f"| profit={row['net_profit']:.2f} "
            f"| harvest_fw={row['harvest_fresh_kg']:.2f} kg "
            f"| harvest_dw={row['harvest_dry_kg']:.2f} kg "
            f"| cost={row['total_cost']:.2f} "
            f"| {elapsed:.1f}s"
        )
        print(
            _build_progress_timing_line(
                completed=local_idx,
                total=len(pending),
                current_elapsed=elapsed,
                started_at=t0,
            )
        )

    if not results_csv.exists():
        raise RuntimeError("Exact RL baseline produced no result CSV.")

    summary = summarise_results(results_csv, summary_json, ranked_csv)
    update_summary_metadata(
        summary_json,
        load_ref=args.load,
        device=args.device,
        explore=bool(args.explore),
        load_checkpoint=args.load_checkpoint,
        resolved_checkpoint_kind=resolved_checkpoint_kind,
        resolved_checkpoint_path=str(resolved_checkpoint_path),
        I1_manual=args.I1_manual,
        I2_manual=args.I2_manual,
        photo_period_manual=args.photo_period_manual,
        light_control_mode=args.light_control_mode,
        light_segments_per_photoperiod=args.light_segments_per_photoperiod,
    )
    generated = generate_exact_baseline_plots(
        results_csv,
        out_dir,
        plot_filenames=RL_PLOT_FILENAMES,
    )
    total_elapsed = time.time() - t0

    print("\n" + "=" * 72)
    print("Exact RL Baseline Summary")
    print("=" * 72)
    print(f"Policy run         : {args.load}")
    print(f"Checkpoint request : {args.load_checkpoint}")
    print(f"Checkpoint resolved: {resolved_checkpoint_kind}")
    if args.I1_manual is not None or args.I2_manual is not None:
        print(f"Manual light       : I1={args.I1_manual}, I2={args.I2_manual}")
    if args.photo_period_manual is not None:
        print(f"Manual photoperiod : {args.photo_period_manual} h")
    if args.light_control_mode is not None:
        print(
            f"Light control      : {args.light_control_mode} "
            f"(segments={args.light_segments_per_photoperiod})"
        )
    print(f"Results CSV        : {results_csv}")
    print(f"Ranked CSV         : {ranked_csv}")
    print(f"Summary JSON       : {summary_json}")
    print(f"Evaluated schedules: {summary['n_evaluated']}")
    print(f"Valid full horizon : {summary['n_valid_full_horizon']}")
    print(f"Invalid / early end: {summary['n_invalid_or_early_terminated']}")
    best_valid = summary.get("best_valid_by_profit")
    default_schedule = summary.get("default_schedule")
    if best_valid:
        print(
            "Best valid schedule: "
            f"{best_valid['schedule_key']} | profit={best_valid['net_profit']:.2f} "
            f"| harvest_fw={best_valid['harvest_fresh_kg']:.2f} kg "
            f"| harvest_dw={best_valid['harvest_dry_kg']:.2f} kg"
        )
    if default_schedule:
        print(
            "Default schedule   : "
            f"{default_schedule['schedule_key']} | profit={default_schedule['net_profit']:.2f} "
            f"| harvest_fw={default_schedule['harvest_fresh_kg']:.2f} kg "
            f"| harvest_dw={default_schedule['harvest_dry_kg']:.2f} kg "
            f"| valid={default_schedule['valid_full_horizon']}"
        )
    if args.num_shards > 1:
        print(
            "Merge command      : "
            f"python experiments/exact_rl_schedule_baseline.py --num-shards {args.num_shards} "
            f"--merge-shards-only --out-dir {out_dir} --load {args.load} "
            f"--load-checkpoint {args.load_checkpoint}"
        )
    for path in generated:
        print(f"Saved plot         : {path}")
    print(f"Elapsed            : {total_elapsed:.1f}s")
    print("=" * 72)


if __name__ == "__main__":
    main()
