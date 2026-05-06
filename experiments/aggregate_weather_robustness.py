# -*- coding: utf-8 -*-
"""Aggregate seasonal robustness results into schedule-level pass-rate summaries."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from envs.utils import load_all_configs  # noqa: E402


def _coerce_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def _schedule_key_from_row(row: Dict[str, Any]) -> str:
    return (
        f"t1={int(row['t1'])}|t2={int(row['t2'])}|"
        f"N1={int(row['N1'])}|rho2={int(round(float(row['rho2'])))}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate weather robustness outputs.")
    parser.add_argument(
        "--weather-root",
        type=str,
        required=True,
        help="Root directory produced by 6_extension_weather_robustness.py",
    )
    parser.add_argument(
        "--evidence-csv",
        type=str,
        default=None,
        help="Optional schedule evidence catalog to merge with robustness metrics.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory. Default: weather-root",
    )
    return parser.parse_args()


def _result_filename(controller_subdir: str) -> str:
    if controller_subdir == "pid":
        return "pid_exact_schedule_results.csv"
    if controller_subdir == "rl":
        return "rl_exact_schedule_results.csv"
    raise RuntimeError(f"Unsupported controller subdir: {controller_subdir}")


def _load_controller_records(
    weather_root: Path,
    *,
    controller_subdir: str,
    label: str,
    harvest_min_dry: float,
    harvest_target_dry: float,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    result_filename = _result_filename(controller_subdir)
    for season_dir in sorted([p for p in weather_root.iterdir() if p.is_dir()]):
        csv_path = season_dir / controller_subdir / result_filename
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        if "schedule_key" not in df.columns:
            df["schedule_key"] = df.apply(lambda row: _schedule_key_from_row(row.to_dict()), axis=1)
        df["valid_full_horizon"] = _coerce_bool(df["valid_full_horizon"])
        df["season"] = str(season_dir.name)
        avg_dry = pd.to_numeric(df.get("avg_harvest_dry_g_per_plant"), errors="coerce").fillna(0.0)
        df[f"{label}_min_pass"] = df["valid_full_horizon"] & (avg_dry >= float(harvest_min_dry))
        df[f"{label}_target_pass"] = df["valid_full_horizon"] & (avg_dry >= float(harvest_target_dry))
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=0, ignore_index=True)


def _summarize_label(records: pd.DataFrame, *, label: str) -> pd.DataFrame:
    if records.empty:
        return pd.DataFrame(columns=["schedule_key"])
    work = records.copy()
    grouped = work.groupby("schedule_key", as_index=False).agg(
        robust_eval_num_scenarios=("season", "nunique"),
        _pass_min=(f"{label}_min_pass", "mean"),
        _pass_target=(f"{label}_target_pass", "mean"),
        _profit_mean=("net_profit", "mean"),
        _profit_std=("net_profit", "std"),
        _energy_mean=("energy_kwh", "mean"),
        _energy_std=("energy_kwh", "std"),
        _harvest_fw_mean=("harvest_fresh_kg", "mean"),
        _harvest_fw_std=("harvest_fresh_kg", "std"),
        _valid_rate=("valid_full_horizon", "mean"),
    )
    rename_map = {
        "_pass_min": f"robust_min_pass_rate_{label}",
        "_pass_target": f"robust_target_pass_rate_{label}",
        "_profit_mean": f"robust_profit_mean_{label}",
        "_profit_std": f"robust_profit_std_{label}",
        "_energy_mean": f"robust_energy_mean_{label}",
        "_energy_std": f"robust_energy_std_{label}",
        "_harvest_fw_mean": f"robust_harvest_fw_mean_{label}",
        "_harvest_fw_std": f"robust_harvest_fw_std_{label}",
        "_valid_rate": f"robust_valid_full_horizon_rate_{label}",
    }
    out = grouped.rename(columns=rename_map)
    for col in rename_map.values():
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out


def build_summary(df: pd.DataFrame) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "n_schedules": int(len(df)),
        "columns": list(df.columns),
    }
    for col in [c for c in df.columns if c.startswith("robust_min_pass_rate_")]:
        summary[f"{col}_mean"] = float(pd.to_numeric(df[col], errors="coerce").fillna(0.0).mean())
    return summary


def main() -> None:
    args = parse_args()
    weather_root = Path(args.weather_root).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else weather_root
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_all_configs(str(ROOT / "configs"))
    reward_params = dict(cfg.get("reward_params", {}) or {})
    harvest_min_dry = float(reward_params.get("harvest_min_dry_mass_per_plant", 4.44))
    harvest_target_dry = float(reward_params.get("harvest_target_dry_mass_per_plant", 5.33))

    pid_records = _load_controller_records(
        weather_root,
        controller_subdir="pid",
        label="pid",
        harvest_min_dry=harvest_min_dry,
        harvest_target_dry=harvest_target_dry,
    )
    rl_records = _load_controller_records(
        weather_root,
        controller_subdir="rl",
        label="rl_main",
        harvest_min_dry=harvest_min_dry,
        harvest_target_dry=harvest_target_dry,
    )

    if pid_records.empty and rl_records.empty:
        raise RuntimeError(f"No seasonal exact-baseline outputs found under {weather_root}")

    robust_df: pd.DataFrame | None = None
    if not pid_records.empty:
        robust_df = _summarize_label(pid_records, label="pid")
    if not rl_records.empty:
        rl_summary = _summarize_label(rl_records, label="rl_main")
        robust_df = rl_summary if robust_df is None else robust_df.merge(rl_summary, on="schedule_key", how="outer")
    assert robust_df is not None

    out_csv = out_dir / "weather_robustness_summary.csv"
    robust_df.to_csv(out_csv, index=False, encoding="utf-8")

    merged_csv = None
    if args.evidence_csv:
        evidence_csv = Path(args.evidence_csv).resolve()
        evidence_df = pd.read_csv(evidence_csv)
        if "schedule_key" not in evidence_df.columns:
            evidence_df["schedule_key"] = evidence_df.apply(lambda row: _schedule_key_from_row(row.to_dict()), axis=1)
        merged_df = evidence_df.merge(robust_df, on="schedule_key", how="left")
        merged_csv = out_dir / "schedule_evidence_catalog_robust.csv"
        merged_df.to_csv(merged_csv, index=False, encoding="utf-8")

    summary = build_summary(robust_df)
    summary_path = out_dir / "weather_robustness_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[SAVE] CSV  -> {out_csv}")
    if merged_csv is not None:
        print(f"[SAVE] MERGED -> {merged_csv}")
    print(f"[SAVE] JSON -> {summary_path}")


if __name__ == "__main__":
    main()
