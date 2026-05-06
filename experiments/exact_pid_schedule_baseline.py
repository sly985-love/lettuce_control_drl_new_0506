# -*- coding: utf-8 -*-
"""
Exact PID schedule baseline over the feasible upper-schedule set.

This script evaluates every feasible schedule in the current discrete recipe
space under the same Hangzhou weather window and the same baseline PID
controller, then ranks schedules by an objective that prioritises full-horizon
feasible operation and annual net profit.

Typical usage:

  python experiments/exact_pid_schedule_baseline.py

  python experiments/exact_pid_schedule_baseline.py --duration 364 --dt 600

  python experiments/exact_pid_schedule_baseline.py --limit 50 --overwrite
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib
import numpy as np
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LOCAL_FEASIBLE_CSV = ROOT / "data" / "feasibility" / "feasible_solutions.csv"
LEGACY_FEASIBLE_CSV = ROOT.parent / "results" / "feasibility" / "feasible_solutions.csv"
FEASIBLE_CSV_DEFAULT = LOCAL_FEASIBLE_CSV if LOCAL_FEASIBLE_CSV.exists() else LEGACY_FEASIBLE_CSV

from experiments.simulate_hangzhou import (  # noqa: E402
    BATCH_TRAJECTORY_FIELDS,
    FIELDS,
    PFALEnvContextual,
    _run_pid_simulation,
    build_env_config,
    build_price_override_kwargs,
    build_schedule,
    expand_weather_for_dt,
    load_weather_csv,
    slice_weather,
)
from rl.drl_based_control import load_schedule_bounds  # noqa: E402


DEFAULT_SCHEDULE = {
    "t1": 14,
    "t2": 14,
    "N1": 20,
    "rho2": 36.0,
}

DEFAULT_OUT_DIR = ROOT / "results" / "exact_pid_baseline"

RESULT_SCHEMA_VERSION = 5

PLOT_FILENAMES = (
    "pid_exact_paper_summary.png",
    "pid_exact_design_heatmaps.png",
    "pid_exact_top10_valid.png",
    "pid_exact_termination_reasons.png",
)

RESULT_FIELDS = [
    "result_schema_version",
    "schedule_key",
    "is_default_schedule",
    "eval_index",
    "t1",
    "t2",
    "N1",
    "N2",
    "rho2",
    "photo_period_hours",
    "manual_photo_period_hours",
    "manual_I1",
    "manual_I2",
    "light_control_mode",
    "light_segments_per_photoperiod",
    "price_model_type",
    "tou_tariff_scenario",
    "electricity_price",
    "co2_price",
    "lettuce_price_fw",
    "constant_price",
    "rho1",
    "A1",
    "A2",
    "A_total",
    "duration_days_requested",
    "dt_seconds",
    "steps_executed",
    "sim_days_executed",
    "episode_length_steps",
    "episode_completion_ratio",
    "termination_reason",
    "terminated_early",
    "valid_full_horizon",
    "objective_value",
    "net_profit",
    "revenue",
    "total_cost",
    "energy_kwh",
    "harvest_kg",
    "harvest_dry_kg",
    "harvest_fresh_kg",
    "total_harvests",
    "total_transplants",
    "avg_harvest_kg_per_event",
    "avg_harvest_dry_kg_per_event",
    "avg_harvest_fresh_kg_per_event",
    "avg_harvest_dry_g_per_plant",
    "avg_harvest_fresh_g_per_plant",
    "cost_per_kg",
    "revenue_per_kg",
    "final_biomass_total_kg_m2",
    "final_biomass_transplant_kg_m2",
    "final_biomass_seedling_kg_m2",
    "final_biomass_finishing_kg_m2",
    "final_biomass_dense_kg_m2",
    "final_temp_c",
    "final_rh_pct",
    "final_co2_ppm",
    "final_vpd_kpa",
    "cum_reward",
]


def _format_float_token(value: float | None) -> str | None:
    if value is None:
        return None
    text = f"{float(value):.6g}"
    return text.replace("-", "m").replace(".", "p")


def _build_exact_pid_scenario_metadata(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "manual_photo_period_hours": (
            int(args.photo_period_manual) if args.photo_period_manual is not None else None
        ),
        "manual_I1": float(args.I1_manual) if args.I1_manual is not None else None,
        "manual_I2": float(args.I2_manual) if args.I2_manual is not None else None,
        "light_control_mode": str(args.light_control_mode) if args.light_control_mode else None,
        "light_segments_per_photoperiod": (
            int(args.light_segments_per_photoperiod)
            if args.light_segments_per_photoperiod is not None
            else None
        ),
        "price_model_type": str(args.price_model_type) if args.price_model_type else None,
        "tou_tariff_scenario": (
            str(args.tou_tariff_scenario) if args.tou_tariff_scenario else None
        ),
        "electricity_price": (
            float(args.electricity_price) if args.electricity_price is not None else None
        ),
        "co2_price": float(args.co2_price) if args.co2_price is not None else None,
        "lettuce_price_fw": (
            float(args.lettuce_price_fw) if args.lettuce_price_fw is not None else None
        ),
        "constant_price": (
            float(args.constant_price) if args.constant_price is not None else None
        ),
    }


def _build_exact_pid_scenario_suffix(meta: Dict[str, Any]) -> str:
    parts: list[str] = []
    if meta.get("manual_photo_period_hours") is not None:
        parts.append(f"pp{int(meta['manual_photo_period_hours'])}h")
    if meta.get("manual_I1") is not None:
        parts.append(f"i1{_format_float_token(meta['manual_I1'])}")
    if meta.get("manual_I2") is not None:
        parts.append(f"i2{_format_float_token(meta['manual_I2'])}")
    if meta.get("light_control_mode"):
        parts.append(str(meta["light_control_mode"]))
        if (
            str(meta.get("light_control_mode")) == "segmented_hold"
            and meta.get("light_segments_per_photoperiod") is not None
        ):
            parts.append(f"seg{int(meta['light_segments_per_photoperiod'])}")
    has_economic_override = any(
        meta.get(key) is not None
        for key in (
            "tou_tariff_scenario",
            "electricity_price",
            "co2_price",
            "lettuce_price_fw",
            "constant_price",
        )
    )
    effective_price_model = str(
        meta.get("price_model_type")
        or ("time_of_use" if meta.get("tou_tariff_scenario") else "constant")
    ).strip().lower()
    if has_economic_override or effective_price_model != "constant":
        parts.append(f"pm_{effective_price_model}")
    if meta.get("tou_tariff_scenario"):
        parts.append(f"tou_{str(meta['tou_tariff_scenario'])}")
    if meta.get("electricity_price") is not None:
        parts.append(f"e{_format_float_token(meta['electricity_price'])}")
    if meta.get("co2_price") is not None:
        parts.append(f"co2{_format_float_token(meta['co2_price'])}")
    if meta.get("lettuce_price_fw") is not None:
        parts.append(f"l{_format_float_token(meta['lettuce_price_fw'])}")
    if meta.get("constant_price") is not None:
        parts.append(f"cp{_format_float_token(meta['constant_price'])}")
    return "__".join(parts)


def _resolve_exact_out_dir(base_out_dir: Path, meta: Dict[str, Any]) -> Path:
    out_dir = base_out_dir.resolve()
    if out_dir != DEFAULT_OUT_DIR.resolve():
        return out_dir
    suffix = _build_exact_pid_scenario_suffix(meta)
    if not suffix:
        return out_dir
    return out_dir.parent / f"{out_dir.name}__{suffix}"


def _write_run_metadata_json(out_dir: Path, payload: Dict[str, Any]) -> Path:
    metadata_path = out_dir / "pid_exact_run_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return metadata_path


def _validate_results_fixed_pp_semantics(
    df: pd.DataFrame,
    *,
    csv_path: Path,
    fixed_pp: int | None = None,
) -> None:
    if df.empty or "photo_period_hours" not in df.columns:
        return
    if fixed_pp is None:
        fixed_pp = int(
            load_schedule_bounds(str(ROOT / "configs" / "schedule_params.yaml"))["PP_fixed"]
        )
    pp_series = pd.to_numeric(df["photo_period_hours"], errors="coerce").dropna()
    if pp_series.empty:
        return
    unique_pp = sorted(pp_series.round().astype(int).unique().tolist())
    if len(unique_pp) != 1 or int(unique_pp[0]) != int(fixed_pp):
        raise RuntimeError(
            f"Existing results CSV uses stale photoperiod semantics: {csv_path}. "
            f"Expected fixed PP={fixed_pp} h, but found {unique_pp}. "
            "Please rerun with --overwrite to rebuild under the current fixed-PP mainline."
        )


def _ensure_results_schema(csv_path: Path) -> None:
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path, nrows=0)
    columns = set(df.columns)
    required = set(RESULT_FIELDS)
    if not required.issubset(columns):
        raise RuntimeError(
            f"Existing results CSV has stale schema: {csv_path}. "
            "Please rerun with --overwrite to rebuild under the current semantics."
        )
    if "photo_period_hours" in columns:
        full_df = pd.read_csv(csv_path, usecols=["photo_period_hours"])
        _validate_results_fixed_pp_semantics(full_df, csv_path=csv_path)


def _format_wall_time(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(float(seconds)):
        return "n/a"
    total_seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes > 0:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _build_progress_bar(current: int, total: int, width: int = 24) -> str:
    safe_total = max(int(total), 1)
    safe_current = min(max(int(current), 0), safe_total)
    if safe_current <= 0:
        return "[" + ("-" * int(width)) + "]"
    if safe_current >= safe_total:
        return "[" + ("#" * int(width)) + "]"
    filled = int(math.floor(width * safe_current / safe_total))
    filled = max(0, min(int(width) - 1, filled))
    return "[" + ("#" * filled) + ">" + ("-" * (int(width) - filled - 1)) + "]"


def _build_progress_timing_line(
    *,
    completed: int,
    total: int,
    current_elapsed: float,
    started_at: float,
) -> str:
    safe_total = max(int(total), 1)
    pct = 100.0 * float(completed) / float(safe_total)
    elapsed_total = max(0.0, float(time.time() - started_at))
    avg_elapsed = float(elapsed_total / max(int(completed), 1))
    eta = float(avg_elapsed * max(safe_total - int(completed), 0))
    bar = _build_progress_bar(completed, safe_total, width=28)
    return (
        f"[PROGRESS] {bar} {completed}/{safe_total} ({pct:5.1f}%) "
        f"| current={_format_wall_time(current_elapsed)} "
        f"| avg={_format_wall_time(avg_elapsed)} "
        f"| elapsed={_format_wall_time(elapsed_total)} "
        f"| eta={_format_wall_time(eta)}"
    )


def schedule_key(schedule: Dict[str, Any]) -> str:
    return (
        f"t1={int(schedule['t1'])}|t2={int(schedule['t2'])}|"
        f"N1={int(schedule['N1'])}|rho2={int(round(float(schedule['rho2'])))}"
    )


def _deduplicate_result_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "schedule_key" not in df.columns:
        return df.copy()
    ordered = df.copy()
    ordered["_source_row_index"] = np.arange(len(ordered), dtype=int)
    deduped = (
        ordered.sort_values("_source_row_index", ascending=True)
        .drop_duplicates(subset=["schedule_key"], keep="last")
        .sort_values("_source_row_index", ascending=True)
        .drop(columns=["_source_row_index"])
        .reset_index(drop=True)
    )
    return deduped


def _normalise_feasible_catalog_df(
    df: pd.DataFrame,
    *,
    bounds: Dict[str, Any],
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    required = ["t1", "t2", "N1", "rho2"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError(
            "Feasible schedule CSV is missing required columns: "
            + ", ".join(missing)
        )

    n_total = int(bounds.get("N_total", 80))
    a_board = float(bounds.get("A_board", 1.0))
    fixed_pp = int(bounds.get("PP_fixed", 16))

    work = df.copy()
    for col in required:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=required).copy()
    work["t1"] = work["t1"].astype(int)
    work["t2"] = work["t2"].astype(int)
    work["N1"] = work["N1"].astype(int)
    work["rho2"] = work["rho2"].astype(float).round().astype(int)
    work = (
        work.sort_values(by=["t1", "t2", "N1", "rho2"])
        .drop_duplicates(subset=["t1", "t2", "N1", "rho2"], keep="last")
        .reset_index(drop=True)
    )

    work["N2"] = n_total - work["N1"]
    work["PP"] = fixed_pp
    work["A1_m2"] = work["N1"].astype(float) * a_board
    work["A2_m2"] = work["N2"].astype(float) * a_board
    work["A1_A2_ratio"] = work["A1_m2"] / work["A2_m2"].replace(0.0, np.nan)
    work["rho1"] = (
        work["rho2"].astype(float)
        * work["N2"].astype(float)
        * work["t1"].astype(float)
        / (work["N1"].astype(float) * work["t2"].astype(float))
    )
    work["rho1_continuous"] = work["rho1"].astype(float)
    work["expansion_ratio"] = work["rho1"].astype(float) / work["rho2"].replace(0, np.nan)
    work["total_cycle_days"] = work["t1"] + work["t2"]
    work["delta_t"] = [
        math.gcd(int(t1), int(t2))
        for t1, t2 in zip(work["t1"].tolist(), work["t2"].tolist())
    ]
    return work


def is_default_schedule(schedule: Dict[str, Any]) -> bool:
    return (
        int(schedule["t1"]) == DEFAULT_SCHEDULE["t1"]
        and int(schedule["t2"]) == DEFAULT_SCHEDULE["t2"]
        and int(schedule["N1"]) == DEFAULT_SCHEDULE["N1"]
        and int(round(float(schedule["rho2"]))) == int(round(DEFAULT_SCHEDULE["rho2"]))
    )


def row_to_schedule(row: Dict[str, Any]) -> Dict[str, Any]:
    schedule = build_schedule(
        t1=int(row["t1"]),
        t2=int(row["t2"]),
        N1=int(row["N1"]),
        rho2=float(row["rho2"]),
        PP=int(row["PP"]) if "PP" in row and pd.notna(row["PP"]) else None,
    )
    if "N2" in row:
        schedule["N2"] = int(row["N2"])
    if "rho1" in row:
        schedule["rho1"] = float(row["rho1"])
    if "A1_m2" in row:
        schedule["A1"] = float(row["A1_m2"])
    if "A2_m2" in row:
        schedule["A2"] = float(row["A2_m2"])
    schedule["A_total"] = float(schedule["A1"] + schedule["A2"])
    return schedule


def load_feasible_schedules(csv_path: Path) -> List[Dict[str, Any]]:
    df = pd.read_csv(csv_path)
    bounds = load_schedule_bounds(str(ROOT / "configs" / "schedule_params.yaml"))
    df = _normalise_feasible_catalog_df(df, bounds=bounds)
    records = df.to_dict(orient="records")
    filtered: List[Dict[str, Any]] = []
    for row in records:
        schedule = row_to_schedule(row)
        if not (
            int(bounds["t1_min"]) <= int(schedule["t1"]) <= int(bounds["t1_max"])
            and int(bounds["t2_min"]) <= int(schedule["t2"]) <= int(bounds["t2_max"])
            and int(bounds["N1_min"]) <= int(schedule["N1"]) <= int(bounds["N1_max"])
            and float(bounds["rho2_min"]) <= float(schedule["rho2"]) <= float(bounds["rho2_max"])
        ):
            continue
        filtered.append(schedule)
    return filtered


def load_completed_results(csv_path: Path) -> Dict[str, Dict[str, Any]]:
    if not csv_path.exists():
        return {}
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return {}
    _validate_results_fixed_pp_semantics(df, csv_path=csv_path)
    completed: Dict[str, Dict[str, Any]] = {}
    for row in df.to_dict(orient="records"):
        version_raw = row.get("result_schema_version", -1)
        try:
            version = int(version_raw)
        except Exception:
            version = -1
        if version != RESULT_SCHEMA_VERSION:
            continue
        key = str(row.get("schedule_key", "")).strip()
        if key:
            completed[key] = row
    return completed


def append_result(csv_path: Path, row: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in RESULT_FIELDS})


def load_weather_window(weather_path: Path, start_date: str, duration_days: float, dt_seconds: float) -> List[Dict[str, Any]]:
    all_weather = load_weather_csv(str(weather_path))
    start_dt = dt.datetime.fromisoformat(start_date)
    end_dt = start_dt + dt.timedelta(days=float(duration_days))
    sliced = slice_weather(all_weather, start_dt, end_dt)
    if not sliced:
        raise RuntimeError(
            f"No weather data found for window start={start_date}, duration={duration_days} days."
        )
    return expand_weather_for_dt(sliced, dt_seconds)


def evaluate_schedule(
    schedule: Dict[str, Any],
    weather_rows: List[Dict[str, Any]],
    dt_seconds: float,
    seed: int,
    photo_period_manual: int | None = None,
    i1_manual: float | None = None,
    i2_manual: float | None = None,
    light_control_mode: str | None = None,
    light_segments_per_photoperiod: int | None = None,
    price_override_kwargs: Dict[str, Any] | None = None,
    return_traces: bool = False,
) -> Dict[str, Any] | tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    env_cfg = build_env_config(
        schedule,
        dt_seconds,
        seed=seed,
        photo_period_override=photo_period_manual,
        light_control_mode=light_control_mode,
        light_segments_per_photoperiod=light_segments_per_photoperiod,
        action_semantics_override="absolute",
        run_config_overrides=None,
        **dict(price_override_kwargs or {}),
    )
    env = PFALEnvContextual(env_cfg)
    records, batch_records = _run_pid_simulation(
        env,
        weather_rows,
        schedule,
        dt_seconds,
        seed,
        I1_manual=i1_manual,
        I2_manual=i2_manual,
        photo_period_manual=photo_period_manual,
    )
    summary = _summarise_pid_episode_records(
        records,
        env=env,
        dt_seconds=dt_seconds,
    )
    if return_traces:
        return summary, records, batch_records
    return summary


def _summarise_pid_episode_records(
    records: List[Dict[str, Any]],
    *,
    env: PFALEnvContextual,
    dt_seconds: float,
) -> Dict[str, Any]:
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
            "price_model_type": str(getattr(env, "electricity_price_model", "constant")),
            "tou_tariff_scenario": str(getattr(env, "tou_tariff_scenario", "") or ""),
            "electricity_price": float(getattr(env, "c_elec", 0.0)),
            "co2_price": float(getattr(env, "c_CO2", 0.0)),
            "lettuce_price_fw": float(getattr(env, "c_lettuce", 0.0)),
            "constant_price": float(
                getattr(env, "constant_electricity_price", getattr(env, "c_elec", 0.0))
            ),
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
        ) / 1000.0
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
    harvest_rows = [r for r in records if float(r.get("harvest_dry_mass_g", r.get("harvest_mass_g", 0.0))) > 0.0]
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
        "price_model_type": str(getattr(env, "electricity_price_model", "constant")),
        "tou_tariff_scenario": str(getattr(env, "tou_tariff_scenario", "") or ""),
        "electricity_price": float(getattr(env, "c_elec", 0.0)),
        "co2_price": float(getattr(env, "c_CO2", 0.0)),
        "lettuce_price_fw": float(getattr(env, "c_lettuce", 0.0)),
        "constant_price": float(
            getattr(env, "constant_electricity_price", getattr(env, "c_elec", 0.0))
        ),
        "cum_reward": float(last.get("cum_reward", 0.0)),
    }


def _trace_schedule_slug(schedule: Dict[str, Any]) -> str:
    return (
        f"t1-{int(schedule['t1'])}__"
        f"t2-{int(schedule['t2'])}__"
        f"N1-{int(schedule['N1'])}__"
        f"rho2-{int(round(float(schedule['rho2'])))}"
    )


def _write_trace_csv(
    csv_path: Path,
    *,
    rows: List[Dict[str, Any]],
    fieldnames: List[str],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_schedule_trace_csvs(
    schedule: Dict[str, Any],
    *,
    detailed_trace_dir: Path | None,
    records: List[Dict[str, Any]],
    batch_trace_dir: Path | None = None,
    batch_records: List[Dict[str, Any]] | None = None,
) -> Dict[str, Path]:
    saved: Dict[str, Path] = {}
    slug = _trace_schedule_slug(schedule)

    if detailed_trace_dir is not None:
        trace_path = detailed_trace_dir / f"{slug}.csv"
        _write_trace_csv(trace_path, rows=records, fieldnames=FIELDS)
        saved["detailed_trace_csv"] = trace_path

    if batch_trace_dir is not None:
        batch_path = batch_trace_dir / f"{slug}_batch_trajectory.csv"
        _write_trace_csv(
            batch_path,
            rows=list(batch_records or []),
            fieldnames=BATCH_TRAJECTORY_FIELDS,
        )
        saved["batch_trajectory_csv"] = batch_path

    return saved


def _ordered_unique_strings(values: Iterable[str]) -> List[str]:
    ordered: List[str] = []
    seen: set[str] = set()
    for value in values:
        key = str(value).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered


def _selected_trace_flags_active(args: argparse.Namespace) -> bool:
    return bool(
        args.export_selected_traces_only
        or int(args.selected_trace_top_k_valid or 0) > 0
        or bool(args.selected_trace_include_default)
        or bool(args.selected_trace_schedule_keys)
    )


def _resolve_selected_trace_dirs(
    out_dir: Path,
    *,
    selected_detailed_trace_dir: str | None,
    selected_batch_trace_dir: str | None,
    save_batch_trajectories: bool,
) -> tuple[Path, Path | None]:
    detailed_dir = (
        Path(selected_detailed_trace_dir).resolve()
        if selected_detailed_trace_dir
        else (out_dir / "selected_detailed_traces").resolve()
    )
    batch_dir = None
    if save_batch_trajectories:
        batch_dir = (
            Path(selected_batch_trace_dir).resolve()
            if selected_batch_trace_dir
            else (out_dir / "selected_batch_trajectories").resolve()
        )
    return detailed_dir, batch_dir


def _load_results_for_selection(results_csv: Path) -> pd.DataFrame:
    if not results_csv.exists():
        raise RuntimeError(
            f"Selected-trace export requires an existing results CSV: {results_csv}"
        )
    df = pd.read_csv(results_csv)
    if df.empty:
        raise RuntimeError(f"Results CSV is empty: {results_csv}")
    _validate_results_fixed_pp_semantics(df, csv_path=results_csv)
    df = _deduplicate_result_rows(df)
    df["valid_full_horizon"] = _coerce_bool_series(df["valid_full_horizon"])
    df["is_default_schedule"] = _coerce_bool_series(df["is_default_schedule"])
    return df


def _select_schedule_keys_for_trace_export(
    *,
    results_csv: Path | None,
    feasible_schedules: List[Dict[str, Any]],
    top_k_valid: int = 0,
    include_default: bool = False,
    explicit_schedule_keys: Iterable[str] = (),
) -> List[str]:
    schedule_map = {schedule_key(s): s for s in feasible_schedules}
    selected_keys: List[str] = []

    if include_default:
        default_keys = [key for key, sched in schedule_map.items() if is_default_schedule(sched)]
        if not default_keys:
            raise RuntimeError("Default schedule was not found in the feasible schedule set.")
        selected_keys.extend(default_keys[:1])

    if int(top_k_valid or 0) > 0:
        if results_csv is None:
            raise RuntimeError("Top-k trace export requires a resolved results CSV.")
        df = _load_results_for_selection(results_csv)
        df_valid = df[df["valid_full_horizon"]].sort_values(
            by=["net_profit", "harvest_kg", "energy_kwh"],
            ascending=[False, False, True],
        )
        if df_valid.empty:
            raise RuntimeError(
                f"Cannot export top-k valid traces because no valid schedules exist in {results_csv}."
            )
        selected_keys.extend(
            df_valid["schedule_key"].astype(str).head(int(top_k_valid)).tolist()
        )

    selected_keys.extend(list(explicit_schedule_keys or []))
    ordered_keys = _ordered_unique_strings(selected_keys)
    if not ordered_keys:
        raise RuntimeError(
            "No schedules were selected for trace export. "
            "Please enable --selected-trace-include-default, "
            "--selected-trace-top-k-valid, or --selected-trace-schedule-key."
        )

    missing = [key for key in ordered_keys if key not in schedule_map]
    if missing:
        raise RuntimeError(
            "The following selected schedule keys were not found in the feasible schedule set: "
            + ", ".join(missing)
        )
    return ordered_keys


def export_selected_schedule_traces(
    *,
    results_csv: Path | None,
    out_dir: Path,
    feasible_schedules: List[Dict[str, Any]],
    weather_path: Path,
    start_date: str,
    duration_days: float,
    dt_seconds: float,
    seed: int,
    photo_period_manual: int | None,
    i1_manual: float | None,
    i2_manual: float | None,
    light_control_mode: str | None,
    light_segments_per_photoperiod: int | None,
    price_override_kwargs: Dict[str, Any] | None,
    top_k_valid: int = 0,
    include_default: bool = False,
    explicit_schedule_keys: Iterable[str] = (),
    selected_detailed_trace_dir: str | None = None,
    selected_batch_trace_dir: str | None = None,
    save_batch_trajectories: bool = False,
    scenario_metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    selected_keys = _select_schedule_keys_for_trace_export(
        results_csv=results_csv,
        feasible_schedules=feasible_schedules,
        top_k_valid=top_k_valid,
        include_default=include_default,
        explicit_schedule_keys=explicit_schedule_keys,
    )
    schedule_map = {schedule_key(s): s for s in feasible_schedules}
    detailed_dir, batch_dir = _resolve_selected_trace_dirs(
        out_dir,
        selected_detailed_trace_dir=selected_detailed_trace_dir,
        selected_batch_trace_dir=selected_batch_trace_dir,
        save_batch_trajectories=save_batch_trajectories,
    )
    weather_rows = load_weather_window(weather_path, start_date, duration_days, dt_seconds)

    manifest_entries: List[Dict[str, Any]] = []
    for idx, key in enumerate(selected_keys, start=1):
        schedule = schedule_map[key]
        summary, records, batch_records = evaluate_schedule(
            schedule,
            weather_rows,
            dt_seconds,
            seed,
            photo_period_manual=photo_period_manual,
            i1_manual=i1_manual,
            i2_manual=i2_manual,
            light_control_mode=light_control_mode,
            light_segments_per_photoperiod=light_segments_per_photoperiod,
            price_override_kwargs=price_override_kwargs,
            return_traces=True,
        )
        saved = save_schedule_trace_csvs(
            schedule,
            detailed_trace_dir=detailed_dir,
            records=records,
            batch_trace_dir=batch_dir,
            batch_records=batch_records if save_batch_trajectories else None,
        )
        entry = {
            "rank_index": int(idx),
            "schedule_key": key,
            "saved_paths": {name: str(path) for name, path in saved.items()},
            "summary": summary,
        }
        manifest_entries.append(entry)
        print(
            f"[PID-TRACE] {idx:3d}/{len(selected_keys):3d} {key} "
            f"| valid={summary['valid_full_horizon']} "
            f"| profit={summary['net_profit']:.2f} "
            f"| detailed={saved.get('detailed_trace_csv', '')}"
        )

    manifest = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "results_csv": str(results_csv) if results_csv is not None else None,
        "selected_detailed_trace_dir": str(detailed_dir),
        "selected_batch_trace_dir": str(batch_dir) if batch_dir is not None else None,
        "selection": {
            "top_k_valid": int(top_k_valid or 0),
            "include_default": bool(include_default),
            "explicit_schedule_keys": _ordered_unique_strings(explicit_schedule_keys),
        },
        "scenario_metadata": dict(scenario_metadata or {}),
        "n_selected": int(len(manifest_entries)),
        "entries": manifest_entries,
    }
    manifest_path = out_dir / "selected_trace_export_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def build_result_row(
    schedule: Dict[str, Any],
    summary: Dict[str, Any],
    *,
    eval_index: int,
    duration_days: float,
    dt_seconds: float,
    scenario_metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    scenario_metadata = dict(scenario_metadata or {})
    row = {
        "result_schema_version": int(RESULT_SCHEMA_VERSION),
        "schedule_key": schedule_key(schedule),
        "is_default_schedule": bool(is_default_schedule(schedule)),
        "eval_index": int(eval_index),
        "t1": int(schedule["t1"]),
        "t2": int(schedule["t2"]),
        "N1": int(schedule["N1"]),
        "N2": int(schedule.get("N2", 80 - int(schedule["N1"]))),
        "rho2": float(schedule["rho2"]),
        "photo_period_hours": int(schedule["PP"]),
        "manual_photo_period_hours": scenario_metadata.get("manual_photo_period_hours"),
        "manual_I1": scenario_metadata.get("manual_I1"),
        "manual_I2": scenario_metadata.get("manual_I2"),
        "light_control_mode": scenario_metadata.get("light_control_mode"),
        "light_segments_per_photoperiod": scenario_metadata.get(
            "light_segments_per_photoperiod"
        ),
        "rho1": float(schedule.get("rho1", 0.0)),
        "A1": float(schedule.get("A1", 0.0)),
        "A2": float(schedule.get("A2", 0.0)),
        "A_total": float(schedule.get("A_total", 0.0)),
        "duration_days_requested": float(duration_days),
        "dt_seconds": float(dt_seconds),
    }
    row.update(summary)
    return row


def _first_non_null_value(df: pd.DataFrame, column: str) -> Any:
    if column not in df.columns:
        return None
    series = df[column]
    non_null = series[series.notna()]
    if non_null.empty:
        return None
    value = non_null.iloc[0]
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, str):
        value = value.strip()
        return value if value else None
    return value


def summarise_results(results_csv: Path, summary_json: Path, top_csv: Path) -> Dict[str, Any]:
    df = pd.read_csv(results_csv)
    if df.empty:
        raise RuntimeError(f"No results found in {results_csv}")
    _validate_results_fixed_pp_semantics(df, csv_path=results_csv)
    df = _deduplicate_result_rows(df)

    df["valid_full_horizon"] = df["valid_full_horizon"].astype(bool)
    df["is_default_schedule"] = df["is_default_schedule"].astype(bool)

    df_objective = df.sort_values(
        by=["objective_value", "net_profit"],
        ascending=[False, False],
    ).reset_index(drop=True)
    df_objective["rank_objective"] = df_objective.index + 1

    df_valid = df[df["valid_full_horizon"]].sort_values(
        by=["net_profit", "harvest_kg", "energy_kwh"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    if not df_valid.empty:
        df_valid["rank_valid_profit"] = df_valid.index + 1

    merged = df_objective.merge(
        df_valid[["schedule_key", "rank_valid_profit"]] if not df_valid.empty else pd.DataFrame(columns=["schedule_key", "rank_valid_profit"]),
        on="schedule_key",
        how="left",
    )
    merged.to_csv(top_csv, index=False, encoding="utf-8")

    default_subset = merged[merged["is_default_schedule"]]
    default_row = default_subset.iloc[0].to_dict() if not default_subset.empty else None
    best_objective = merged.iloc[0].to_dict()
    best_valid = df_valid.iloc[0].to_dict() if not df_valid.empty else None

    def _gap(best: Dict[str, Any] | None, baseline: Dict[str, Any]) -> Dict[str, Any] | None:
        if best is None:
            return None
        baseline_profit = float(baseline.get("net_profit", 0.0))
        best_profit = float(best.get("net_profit", 0.0))
        baseline_harvest = float(baseline.get("harvest_kg", 0.0))
        best_harvest = float(best.get("harvest_kg", 0.0))
        return {
            "profit_absolute": best_profit - baseline_profit,
            "profit_relative": (
                (best_profit - baseline_profit) / abs(baseline_profit)
                if abs(baseline_profit) > 1e-9 else None
            ),
            "harvest_absolute_kg": best_harvest - baseline_harvest,
            "harvest_relative": (
                (best_harvest - baseline_harvest) / max(abs(baseline_harvest), 1e-9)
                if abs(baseline_harvest) > 1e-9 else None
            ),
        }

    scenario_metadata = {
        "manual_photo_period_hours": _first_non_null_value(df, "manual_photo_period_hours"),
        "manual_I1": _first_non_null_value(df, "manual_I1"),
        "manual_I2": _first_non_null_value(df, "manual_I2"),
        "light_control_mode": _first_non_null_value(df, "light_control_mode"),
        "light_segments_per_photoperiod": _first_non_null_value(
            df, "light_segments_per_photoperiod"
        ),
        "price_model_type": _first_non_null_value(df, "price_model_type"),
        "tou_tariff_scenario": _first_non_null_value(df, "tou_tariff_scenario"),
        "electricity_price": _first_non_null_value(df, "electricity_price"),
        "co2_price": _first_non_null_value(df, "co2_price"),
        "lettuce_price_fw": _first_non_null_value(df, "lettuce_price_fw"),
        "constant_price": _first_non_null_value(df, "constant_price"),
    }

    summary = {
        "n_evaluated": int(len(df)),
        "n_valid_full_horizon": int(df["valid_full_horizon"].sum()),
        "n_invalid_or_early_terminated": int((~df["valid_full_horizon"]).sum()),
        "scenario_metadata": scenario_metadata,
        "best_by_objective": best_objective,
        "best_valid_by_profit": best_valid,
        "default_schedule": default_row,
        "gap_best_valid_vs_default": _gap(best_valid, default_row) if default_row else None,
        "gap_best_objective_vs_default": _gap(best_objective, default_row) if default_row else None,
        "termination_reason_counts": {
            str(k): int(v) for k, v in df["termination_reason"].value_counts().items()
        },
    }

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def _prepare_plot_dataframe(results_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(results_csv)
    if df.empty:
        raise RuntimeError(f"No results found in {results_csv}")
    _validate_results_fixed_pp_semantics(df, csv_path=results_csv)
    df = _deduplicate_result_rows(df)
    df["valid_full_horizon"] = _coerce_bool_series(df["valid_full_horizon"])
    df["is_default_schedule"] = _coerce_bool_series(df["is_default_schedule"])
    df["cycle_days"] = df["t1"].astype(float) + df["t2"].astype(float)
    return df


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.25, linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_alpha(0.3)


def _schedule_label(row: Dict[str, Any]) -> str:
    return (
        f"t1={int(row['t1'])}, t2={int(row['t2'])}, "
        f"N1={int(row['N1'])}, rho2={int(round(float(row['rho2'])))}"
    )


def _highlight_schedule(
    ax: plt.Axes,
    row: Dict[str, Any] | None,
    *,
    x_key: str,
    y_key: str,
    label: str,
    color: str,
    marker: str,
) -> None:
    if not row:
        return
    x = float(row.get(x_key, np.nan))
    y = float(row.get(y_key, np.nan))
    if not np.isfinite(x) or not np.isfinite(y):
        return
    ax.scatter(
        [x],
        [y],
        s=170,
        marker=marker,
        c=color,
        edgecolors="black",
        linewidths=1.0,
        zorder=5,
        label=label,
    )


def _plot_heatmap(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    row_key: str,
    col_key: str,
    value_key: str,
    title: str,
    cmap: str,
    cbar_label: str,
    default_row: Dict[str, Any] | None = None,
    best_row: Dict[str, Any] | None = None,
) -> None:
    pivot = df.pivot_table(index=row_key, columns=col_key, values=value_key, aggfunc="mean")
    pivot = pivot.sort_index().sort_index(axis=1)
    if pivot.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=11)
        ax.set_axis_off()
        return

    im = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap=cmap)
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlabel(col_key)
    ax.set_ylabel(row_key)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(v) for v in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(v) for v in pivot.index])
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)

    def _mark(row: Dict[str, Any] | None, color: str, marker: str, label: str) -> None:
        if not row:
            return
        row_val = row.get(row_key)
        col_val = row.get(col_key)
        if row_val not in pivot.index or col_val not in pivot.columns:
            return
        y = list(pivot.index).index(row_val)
        x = list(pivot.columns).index(col_val)
        ax.scatter(
            [x],
            [y],
            c=color,
            marker=marker,
            s=160,
            edgecolors="black",
            linewidths=1.0,
            zorder=5,
            label=label,
        )

    _mark(default_row, "#d62728", "s", "Default")
    _mark(best_row, "#111111", "*", "Best valid")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper right", fontsize=8, frameon=True)


def _plot_summary_delta(ax: plt.Axes, default_row: Dict[str, Any] | None, best_row: Dict[str, Any] | None) -> None:
    ax.set_title("Best Valid vs Default", fontsize=11, pad=8)
    if not default_row or not best_row:
        ax.text(0.5, 0.5, "Default or best-valid schedule unavailable", ha="center", va="center", fontsize=11)
        ax.set_axis_off()
        return

    metrics = [
        ("Net profit", "net_profit"),
        ("Harvest FW", "harvest_fresh_kg"),
        ("Energy", "energy_kwh"),
        ("Cost", "total_cost"),
    ]
    deltas_pct = []
    annotations = []
    for label, key in metrics:
        base = float(default_row.get(key, 0.0))
        best = float(best_row.get(key, 0.0))
        if abs(base) > 1e-9:
            delta_pct = (best - base) / abs(base) * 100.0
        else:
            delta_pct = np.nan
        deltas_pct.append(delta_pct)
        annotations.append(f"{best - base:+.1f}")

    colors = ["#2b8cbe", "#41ab5d", "#e6550d", "#756bb1"]
    bars = ax.bar(range(len(metrics)), deltas_pct, color=colors, alpha=0.88)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m[0] for m in metrics], rotation=20, ha="right")
    ax.set_ylabel("Relative delta vs default [%]")
    _style_axes(ax)
    for bar, txt, pct in zip(bars, annotations, deltas_pct):
        if np.isnan(pct):
            continue
        y = pct + (1.8 if pct >= 0 else -1.8)
        va = "bottom" if pct >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2.0, y, txt, ha="center", va=va, fontsize=9)


def save_paper_summary_plot(df: pd.DataFrame, out_png: Path) -> None:
    df_valid = df[df["valid_full_horizon"]].copy()
    plot_df = df_valid if not df_valid.empty else df.copy()
    default_row = df[df["is_default_schedule"]].iloc[0].to_dict() if df["is_default_schedule"].any() else None
    best_row = (
        df_valid.sort_values(by=["net_profit", "harvest_fresh_kg", "energy_kwh"], ascending=[False, False, True]).iloc[0].to_dict()
        if not df_valid.empty else None
    )

    fig, axes = plt.subplots(2, 2, figsize=(17, 12))
    ax1, ax2, ax3, ax4 = axes.flat

    sc1 = ax1.scatter(
        plot_df["harvest_fresh_kg"],
        plot_df["net_profit"],
        c=plot_df["energy_kwh"],
        s=40,
        cmap="viridis",
        alpha=0.85,
        edgecolors="none",
    )
    ax1.set_title("Profit vs Harvest Fresh Mass", fontsize=11, pad=8)
    ax1.set_xlabel("Harvest fresh mass [kg]")
    ax1.set_ylabel("Net profit")
    _style_axes(ax1)
    cb1 = fig.colorbar(sc1, ax=ax1, fraction=0.046, pad=0.04)
    cb1.set_label("Energy [kWh]")
    _highlight_schedule(ax1, default_row, x_key="harvest_fresh_kg", y_key="net_profit", label="Default", color="#d62728", marker="s")
    _highlight_schedule(ax1, best_row, x_key="harvest_fresh_kg", y_key="net_profit", label="Best valid", color="#111111", marker="*")
    if ax1.get_legend_handles_labels()[0]:
        ax1.legend(loc="lower right", frameon=True)

    sc2 = ax2.scatter(
        plot_df["energy_kwh"],
        plot_df["net_profit"],
        c=plot_df["cycle_days"],
        s=40,
        cmap="plasma",
        alpha=0.85,
        edgecolors="none",
    )
    ax2.set_title("Profit vs Energy Demand", fontsize=11, pad=8)
    ax2.set_xlabel("Energy [kWh]")
    ax2.set_ylabel("Net profit")
    _style_axes(ax2)
    cb2 = fig.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.04)
    cb2.set_label("Total cycle length [d]")
    _highlight_schedule(ax2, default_row, x_key="energy_kwh", y_key="net_profit", label="Default", color="#d62728", marker="s")
    _highlight_schedule(ax2, best_row, x_key="energy_kwh", y_key="net_profit", label="Best valid", color="#111111", marker="*")
    if ax2.get_legend_handles_labels()[0]:
        ax2.legend(loc="lower left", frameon=True)

    _plot_heatmap(
        ax3,
        plot_df,
        row_key="t1",
        col_key="t2",
        value_key="net_profit",
        title="Mean Net Profit over Stage Durations",
        cmap="YlGnBu",
        cbar_label="Net profit",
        default_row=default_row,
        best_row=best_row,
    )

    _plot_summary_delta(ax4, default_row, best_row)

    n_valid = int(df_valid.shape[0])
    fig.suptitle(
        f"Exact PID schedule baseline summary | evaluated={len(df)} | valid={n_valid}",
        fontsize=14,
        y=0.98,
    )
    fig.subplots_adjust(top=0.92, wspace=0.28, hspace=0.30)
    fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_design_heatmap_plot(df: pd.DataFrame, out_png: Path) -> None:
    df_valid = df[df["valid_full_horizon"]].copy()
    plot_df = df_valid if not df_valid.empty else df.copy()
    default_row = df[df["is_default_schedule"]].iloc[0].to_dict() if df["is_default_schedule"].any() else None
    best_row = (
        df_valid.sort_values(by=["net_profit", "harvest_fresh_kg", "energy_kwh"], ascending=[False, False, True]).iloc[0].to_dict()
        if not df_valid.empty else None
    )

    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    _plot_heatmap(
        axes[0, 0],
        plot_df,
        row_key="t1",
        col_key="t2",
        value_key="net_profit",
        title="Mean Net Profit by (t1, t2)",
        cmap="viridis",
        cbar_label="Net profit",
        default_row=default_row,
        best_row=best_row,
    )
    _plot_heatmap(
        axes[0, 1],
        plot_df,
        row_key="t1",
        col_key="t2",
        value_key="harvest_fresh_kg",
        title="Mean Harvest Fresh Mass by (t1, t2)",
        cmap="magma",
        cbar_label="Harvest fresh mass [kg]",
        default_row=default_row,
        best_row=best_row,
    )
    _plot_heatmap(
        axes[1, 0],
        plot_df,
        row_key="N1",
        col_key="rho2",
        value_key="net_profit",
        title="Mean Net Profit by (N1, rho2)",
        cmap="cividis",
        cbar_label="Net profit",
        default_row=default_row,
        best_row=best_row,
    )
    _plot_heatmap(
        axes[1, 1],
        plot_df,
        row_key="cycle_days",
        col_key="rho2",
        value_key="net_profit",
        title="Mean Net Profit by (total cycle, rho2)",
        cmap="coolwarm",
        cbar_label="Net profit",
        default_row=default_row,
        best_row=best_row,
    )
    fig.suptitle("Exact PID design-space heatmaps", fontsize=14, y=0.98)
    fig.subplots_adjust(top=0.92, wspace=0.28, hspace=0.30)
    fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_top10_plot(df: pd.DataFrame, out_png: Path) -> None:
    df_valid = df[df["valid_full_horizon"]].copy()
    if df_valid.empty:
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.text(0.5, 0.5, "No valid full-horizon schedules available", ha="center", va="center", fontsize=12)
        ax.set_axis_off()
        fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return

    top10 = df_valid.sort_values(
        by=["net_profit", "harvest_fresh_kg", "energy_kwh"],
        ascending=[False, False, True],
    ).head(10).copy()
    top10 = top10.iloc[::-1].reset_index(drop=True)
    labels = [_schedule_label(row) for row in top10.to_dict(orient="records")]
    profits = top10["net_profit"].astype(float).values
    harvest = top10["harvest_fresh_kg"].astype(float).values
    energy_mwh = top10["energy_kwh"].astype(float).values / 1000.0

    fig, ax = plt.subplots(figsize=(15, 8.5))
    y = np.arange(len(top10))
    bars = ax.barh(y, profits, color=plt.cm.Blues(np.linspace(0.35, 0.9, len(top10))), alpha=0.95)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Net profit")
    ax.set_title("Top-10 valid schedules ranked by net profit", fontsize=13, pad=10)
    _style_axes(ax)

    for bar, fw, emwh in zip(bars, harvest, energy_mwh):
        ax.text(
            bar.get_width() + max(profits) * 0.01,
            bar.get_y() + bar.get_height() / 2.0,
            f"FW={fw:.1f} kg | E={emwh:.2f} MWh",
            va="center",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_termination_reason_plot(df: pd.DataFrame, out_png: Path) -> None:
    counts = df["termination_reason"].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    if counts.empty:
        ax.text(0.5, 0.5, "No termination statistics available", ha="center", va="center", fontsize=12)
        ax.set_axis_off()
    else:
        bars = ax.barh(counts.index.astype(str), counts.values.astype(float), color="#6baed6", alpha=0.9)
        ax.set_xlabel("Schedule count")
        ax.set_title("Termination reasons across exact PID baseline", fontsize=12, pad=8)
        _style_axes(ax)
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_width() + max(counts.values) * 0.01, bar.get_y() + bar.get_height() / 2.0, str(int(val)), va="center")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_exact_baseline_plots(
    results_csv: Path,
    out_dir: Path,
    plot_filenames: Iterable[str] = PLOT_FILENAMES,
) -> List[Path]:
    df = _prepare_plot_dataframe(results_csv)
    out_paths = [out_dir / str(name) for name in plot_filenames]
    save_paper_summary_plot(df, out_paths[0])
    save_design_heatmap_plot(df, out_paths[1])
    save_top10_plot(df, out_paths[2])
    save_termination_reason_plot(df, out_paths[3])
    return out_paths


def iter_schedules(
    schedules: Iterable[Dict[str, Any]],
    *,
    completed: Dict[str, Dict[str, Any]],
    limit: int | None,
    shard_id: int = 0,
    num_shards: int = 1,
) -> Iterable[tuple[int, Dict[str, Any]]]:
    count = 0
    for idx, schedule in enumerate(schedules):
        if num_shards > 1 and idx % num_shards != shard_id:
            continue
        key = schedule_key(schedule)
        if key in completed:
            continue
        yield idx, schedule
        count += 1
        if limit is not None and count >= limit:
            return


def count_shard_candidates(schedules: Iterable[Dict[str, Any]], *, shard_id: int, num_shards: int) -> int:
    if num_shards <= 1:
        return sum(1 for _ in schedules)
    return sum(1 for idx, _ in enumerate(schedules) if idx % num_shards == shard_id)


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
        results_csv = out_dir / "pid_exact_schedule_results.csv"
        ranked_csv = out_dir / "pid_exact_schedule_results_ranked.csv"
        summary_json = out_dir / "pid_exact_schedule_summary.json"
        plot_paths = [out_dir / name for name in PLOT_FILENAMES]
        return results_csv, ranked_csv, summary_json, plot_paths

    suffix = shard_suffix(shard_id, num_shards)
    results_csv = out_dir / f"pid_exact_schedule_results.{suffix}.csv"
    ranked_csv = out_dir / f"pid_exact_schedule_results_ranked.{suffix}.csv"
    summary_json = out_dir / f"pid_exact_schedule_summary.{suffix}.json"
    plot_paths = [
        out_dir / f"{Path(name).stem}.{suffix}{Path(name).suffix}"
        for name in PLOT_FILENAMES
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exact PID baseline over the feasible schedule set."
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
        default=str(DEFAULT_OUT_DIR),
        help="Directory for exact-baseline outputs.",
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
        "--export-selected-traces-only",
        "--export_selected_traces_only",
        dest="export_selected_traces_only",
        action="store_true",
        help=(
            "Skip the full exact sweep and rerun only selected schedules to export "
            "detailed trace CSVs from an existing results directory."
        ),
    )
    parser.add_argument(
        "--selected-trace-top-k-valid",
        "--selected_trace_top_k_valid",
        dest="selected_trace_top_k_valid",
        type=int,
        default=0,
        help="After the exact sweep, rerun the top-k valid schedules (ranked by net profit) to export detailed traces.",
    )
    parser.add_argument(
        "--selected-trace-include-default",
        "--selected_trace_include_default",
        dest="selected_trace_include_default",
        action="store_true",
        help="Also export a detailed trace for the default schedule.",
    )
    parser.add_argument(
        "--selected-trace-schedule-key",
        "--selected_trace_schedule_key",
        dest="selected_trace_schedule_keys",
        action="append",
        default=[],
        help='Also export a detailed trace for the specified schedule key, e.g. "t1=14|t2=14|N1=20|rho2=36". Repeatable.',
    )
    parser.add_argument(
        "--selected-trace-save-batch-trajectories",
        "--selected_trace_save_batch_trajectories",
        dest="selected_trace_save_batch_trajectories",
        action="store_true",
        help="When exporting selected traces, also save the batch-trajectory CSVs.",
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
    parser.add_argument(
        "--selected-detailed-trace-dir",
        "--selected_detailed_trace_dir",
        dest="selected_detailed_trace_dir",
        default=None,
        help="Optional directory for selected per-schedule detailed trace CSVs. Default: <out-dir>/selected_detailed_traces.",
    )
    parser.add_argument(
        "--selected-batch-trace-dir",
        "--selected_batch_trace_dir",
        dest="selected_batch_trace_dir",
        default=None,
        help="Optional directory for selected per-schedule batch trajectory CSVs. Default: <out-dir>/selected_batch_trajectories.",
    )
    parser.add_argument("--I1_manual", type=float, default=None, help="Optional manual dense-zone PPFD override.")
    parser.add_argument("--I2_manual", type=float, default=None, help="Optional manual finishing-zone PPFD override.")
    parser.add_argument(
        "--photo-period-manual",
        "--photo_period_manual",
        dest="photo_period_manual",
        type=int,
        default=None,
        help="Optional manual photoperiod override [h/day].",
    )
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
        "--price_model_type",
        dest="price_model_type",
        type=str,
        default=None,
        choices=["constant", "time_of_use"],
        help="Optional electricity-price model override.",
    )
    parser.add_argument(
        "--tou-tariff-scenario",
        "--tou_tariff_scenario",
        dest="tou_tariff_scenario",
        type=str,
        default=None,
        help="Optional built-in TOU tariff scenario name.",
    )
    parser.add_argument(
        "--electricity-price",
        "--electricity_price",
        dest="electricity_price",
        type=float,
        default=None,
        help="Optional fixed electricity-price override [RMB/kWh].",
    )
    parser.add_argument(
        "--co2-price",
        "--co2_price",
        dest="co2_price",
        type=float,
        default=None,
        help="Optional CO2-price override [RMB/kg].",
    )
    parser.add_argument(
        "--lettuce-price-fw",
        "--lettuce_price_fw",
        dest="lettuce_price_fw",
        type=float,
        default=None,
        help="Optional lettuce fresh-weight price override [RMB/kg].",
    )
    parser.add_argument(
        "--constant-price",
        "--constant_price",
        dest="constant_price",
        type=float,
        default=None,
        help="Optional TOU flat/fallback electricity price [RMB/kWh].",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_trace_requested = _selected_trace_flags_active(args)
    if args.num_shards < 1:
        raise RuntimeError("--num-shards must be at least 1.")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise RuntimeError("--shard-id must satisfy 0 <= shard-id < num-shards.")
    if args.merge_shards_only and args.num_shards <= 1:
        raise RuntimeError("--merge-shards-only requires --num-shards > 1.")
    if args.export_selected_traces_only and args.merge_shards_only:
        raise RuntimeError(
            "--export-selected-traces-only cannot be combined with --merge-shards-only. "
            "Use --merge-shards-only first, then rerun with --export-selected-traces-only."
        )
    if (
        args.num_shards > 1
        and selected_trace_requested
        and not args.merge_shards_only
        and not args.export_selected_traces_only
    ):
        raise RuntimeError(
            "Selected-trace export for sharded exact runs should be done in the merge step. "
            "Please finish all shards, then rerun with --merge-shards-only together with the "
            "selected-trace options."
        )

    scenario_metadata = _build_exact_pid_scenario_metadata(args)
    out_dir = _resolve_exact_out_dir(Path(args.out_dir), scenario_metadata)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_payload = {
        "result_schema_version": int(RESULT_SCHEMA_VERSION),
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "scenario_metadata": scenario_metadata,
        "cli_args": vars(args),
        "effective_out_dir": str(out_dir),
    }
    metadata_path = _write_run_metadata_json(out_dir, metadata_payload)
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
    selected_trace_source_csv = merged_results_csv if args.num_shards > 1 else results_csv

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
        generated = generate_exact_baseline_plots(results_csv, out_dir)
        print("\n" + "=" * 72)
        print("Exact PID Baseline Plot Refresh")
        print("=" * 72)
        print(f"Results CSV        : {results_csv}")
        print(f"Ranked CSV         : {ranked_csv}")
        print(f"Summary JSON       : {summary_json}")
        print(f"Evaluated schedules: {summary['n_evaluated']}")
        print(f"Valid full horizon : {summary['n_valid_full_horizon']}")
        if selected_trace_requested:
            manifest = export_selected_schedule_traces(
                results_csv=selected_trace_source_csv,
                out_dir=out_dir,
                feasible_schedules=schedules,
                weather_path=Path(args.weather_path),
                start_date=args.start_date,
                duration_days=args.duration,
                dt_seconds=args.dt,
                seed=args.seed,
                photo_period_manual=args.photo_period_manual,
                i1_manual=args.I1_manual,
                i2_manual=args.I2_manual,
                light_control_mode=args.light_control_mode,
                light_segments_per_photoperiod=args.light_segments_per_photoperiod,
                price_override_kwargs=price_override_kwargs,
                top_k_valid=args.selected_trace_top_k_valid,
                include_default=args.selected_trace_include_default,
                explicit_schedule_keys=args.selected_trace_schedule_keys,
                selected_detailed_trace_dir=args.selected_detailed_trace_dir,
                selected_batch_trace_dir=args.selected_batch_trace_dir,
                save_batch_trajectories=bool(args.selected_trace_save_batch_trajectories),
                scenario_metadata=scenario_metadata,
            )
            print(f"Selected traces    : {manifest['n_selected']}")
            print(f"Trace manifest     : {manifest['manifest_path']}")
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
        generated = generate_exact_baseline_plots(merged_results_csv, out_dir)
        print("\n" + "=" * 72)
        print("Exact PID Baseline Shard Merge")
        print("=" * 72)
        print(f"Merged Results CSV : {merged_results_csv}")
        print(f"Merged Ranked CSV  : {merged_ranked_csv}")
        print(f"Merged Summary JSON: {merged_summary_json}")
        print(f"Shard CSV count    : {len(shard_paths)}")
        print(f"Evaluated schedules: {summary['n_evaluated']}")
        print(f"Valid full horizon : {summary['n_valid_full_horizon']}")
        if selected_trace_requested:
            manifest = export_selected_schedule_traces(
                results_csv=merged_results_csv,
                out_dir=out_dir,
                feasible_schedules=schedules,
                weather_path=Path(args.weather_path),
                start_date=args.start_date,
                duration_days=args.duration,
                dt_seconds=args.dt,
                seed=args.seed,
                photo_period_manual=args.photo_period_manual,
                i1_manual=args.I1_manual,
                i2_manual=args.I2_manual,
                light_control_mode=args.light_control_mode,
                light_segments_per_photoperiod=args.light_segments_per_photoperiod,
                price_override_kwargs=price_override_kwargs,
                top_k_valid=args.selected_trace_top_k_valid,
                include_default=args.selected_trace_include_default,
                explicit_schedule_keys=args.selected_trace_schedule_keys,
                selected_detailed_trace_dir=args.selected_detailed_trace_dir,
                selected_batch_trace_dir=args.selected_batch_trace_dir,
                save_batch_trajectories=bool(args.selected_trace_save_batch_trajectories),
                scenario_metadata=scenario_metadata,
            )
            print(f"Selected traces    : {manifest['n_selected']}")
            print(f"Trace manifest     : {manifest['manifest_path']}")
        for path in generated:
            print(f"Saved plot         : {path}")
        print("=" * 72)
        return

    if args.export_selected_traces_only:
        manifest = export_selected_schedule_traces(
            results_csv=selected_trace_source_csv,
            out_dir=out_dir,
            feasible_schedules=schedules,
            weather_path=Path(args.weather_path),
            start_date=args.start_date,
            duration_days=args.duration,
            dt_seconds=args.dt,
            seed=args.seed,
            photo_period_manual=args.photo_period_manual,
            i1_manual=args.I1_manual,
            i2_manual=args.I2_manual,
            light_control_mode=args.light_control_mode,
            light_segments_per_photoperiod=args.light_segments_per_photoperiod,
            price_override_kwargs=price_override_kwargs,
            top_k_valid=args.selected_trace_top_k_valid,
            include_default=args.selected_trace_include_default,
            explicit_schedule_keys=args.selected_trace_schedule_keys,
            selected_detailed_trace_dir=args.selected_detailed_trace_dir,
            selected_batch_trace_dir=args.selected_batch_trace_dir,
            save_batch_trajectories=bool(args.selected_trace_save_batch_trajectories),
            scenario_metadata=scenario_metadata,
        )
        print("\n" + "=" * 72)
        print("Exact PID Selected Trace Export")
        print("=" * 72)
        print(f"Results CSV        : {selected_trace_source_csv}")
        print(f"Detailed trace dir : {manifest['selected_detailed_trace_dir']}")
        if manifest.get("selected_batch_trace_dir"):
            print(f"Batch trace dir    : {manifest['selected_batch_trace_dir']}")
        print(f"Selected schedules : {manifest['n_selected']}")
        print(f"Manifest JSON      : {manifest['manifest_path']}")
        print("=" * 72)
        return
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
        evaluation = evaluate_schedule(
            schedule,
            weather_rows or [],
            args.dt,
            args.seed,
            photo_period_manual=args.photo_period_manual,
            i1_manual=args.I1_manual,
            i2_manual=args.I2_manual,
            light_control_mode=args.light_control_mode,
            light_segments_per_photoperiod=args.light_segments_per_photoperiod,
            price_override_kwargs=price_override_kwargs,
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
            scenario_metadata=scenario_metadata,
        )
        append_result(results_csv, row)
        elapsed = time.time() - schedule_start
        print(
            f"[PID-EXACT] {local_idx:4d}/{len(pending):4d} "
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
        raise RuntimeError("Exact PID baseline produced no result CSV.")

    summary = summarise_results(results_csv, summary_json, ranked_csv)
    generated = generate_exact_baseline_plots(results_csv, out_dir)
    total_elapsed = time.time() - t0

    print("\n" + "=" * 72)
    print("Exact PID Baseline Summary")
    print("=" * 72)
    print(f"Results CSV        : {results_csv}")
    print(f"Ranked CSV         : {ranked_csv}")
    print(f"Summary JSON       : {summary_json}")
    print(f"Run metadata       : {metadata_path}")
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
            f"python experiments/exact_pid_schedule_baseline.py --num-shards {args.num_shards} "
            f"--merge-shards-only --out-dir {out_dir}"
        )
    if selected_trace_requested:
        manifest = export_selected_schedule_traces(
            results_csv=results_csv,
            out_dir=out_dir,
            feasible_schedules=schedules,
            weather_path=Path(args.weather_path),
            start_date=args.start_date,
            duration_days=args.duration,
            dt_seconds=args.dt,
            seed=args.seed,
            photo_period_manual=args.photo_period_manual,
            i1_manual=args.I1_manual,
            i2_manual=args.I2_manual,
            light_control_mode=args.light_control_mode,
            light_segments_per_photoperiod=args.light_segments_per_photoperiod,
            price_override_kwargs=price_override_kwargs,
            top_k_valid=args.selected_trace_top_k_valid,
            include_default=args.selected_trace_include_default,
            explicit_schedule_keys=args.selected_trace_schedule_keys,
            selected_detailed_trace_dir=args.selected_detailed_trace_dir,
            selected_batch_trace_dir=args.selected_batch_trace_dir,
            save_batch_trajectories=bool(args.selected_trace_save_batch_trajectories),
            scenario_metadata=scenario_metadata,
        )
        print(f"Selected traces    : {manifest['n_selected']}")
        print(f"Trace manifest     : {manifest['manifest_path']}")
    for path in generated:
        print(f"Saved plot         : {path}")
    print(f"Elapsed            : {total_elapsed:.1f}s")
    print("=" * 72)


if __name__ == "__main__":
    main()
