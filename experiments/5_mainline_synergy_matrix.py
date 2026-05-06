# -*- coding: utf-8 -*-
"""Build the four-combination synergy matrix for the mainline paper story."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEDULE_KEY = "t1=14|t2=14|N1=20|rho2=36"


def _coerce_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def _ensure_schedule_key(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if "schedule_key" not in work.columns:
        work["schedule_key"] = work.apply(
            lambda row: (
                f"t1={int(row['t1'])}|t2={int(row['t2'])}|"
                f"N1={int(row['N1'])}|rho2={int(round(float(row['rho2'])))}"
            ),
            axis=1,
        )
    work["valid_full_horizon"] = _coerce_bool(work["valid_full_horizon"])
    return work


def _pick_upper_optimized_schedule(pid_df: pd.DataFrame, requested: str | None) -> str:
    if requested:
        return str(requested).strip()
    valid = pid_df[pid_df["valid_full_horizon"]].copy()
    if valid.empty:
        raise RuntimeError("PID CSV contains no valid full-horizon schedules.")
    best = valid.sort_values(
        by=["net_profit", "harvest_fresh_kg", "energy_kwh"],
        ascending=[False, False, True],
    ).iloc[0]
    return str(best["schedule_key"])


def _extract_row(df: pd.DataFrame, schedule_key: str) -> dict[str, Any]:
    subset = df[df["schedule_key"] == schedule_key]
    if subset.empty:
        raise RuntimeError(f"Schedule '{schedule_key}' not found in {list(df.columns)}")
    return subset.iloc[0].to_dict()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the mainline synergy matrix.")
    parser.add_argument(
        "--pid-csv",
        type=str,
        default=str(ROOT / "results" / "exp02_exact_pid_baseline" / "pid_exact_schedule_results.csv"),
    )
    parser.add_argument(
        "--rl-csv",
        type=str,
        default=str(ROOT / "results" / "exp03_exact_rl_baseline" / "rl_exact_schedule_results.csv"),
    )
    parser.add_argument(
        "--default-schedule-key",
        type=str,
        default=DEFAULT_SCHEDULE_KEY,
    )
    parser.add_argument(
        "--optimized-schedule-key",
        type=str,
        default=None,
        help="Optional override for the upper-optimized schedule key.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT / "results" / "exp05_synergy_matrix"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pid_df = _ensure_schedule_key(pd.read_csv(args.pid_csv))
    rl_df = _ensure_schedule_key(pd.read_csv(args.rl_csv))

    default_key = str(args.default_schedule_key).strip()
    optimized_key = _pick_upper_optimized_schedule(pid_df, args.optimized_schedule_key)

    rows = [
        {
            "combination": "default_schedule + PID",
            "schedule_role": "default",
            "controller": "PID",
            **_extract_row(pid_df, default_key),
        },
        {
            "combination": "default_schedule + RL",
            "schedule_role": "default",
            "controller": "RL",
            **_extract_row(rl_df, default_key),
        },
        {
            "combination": "upper_optimized_schedule + PID",
            "schedule_role": "upper_optimized",
            "controller": "PID",
            **_extract_row(pid_df, optimized_key),
        },
        {
            "combination": "upper_optimized_schedule + RL",
            "schedule_role": "upper_optimized",
            "controller": "RL",
            **_extract_row(rl_df, optimized_key),
        },
    ]
    summary_df = pd.DataFrame(rows)
    summary_csv = out_dir / "synergy_matrix_summary.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8")

    default_pid = rows[0]
    default_rl = rows[1]
    opt_pid = rows[2]
    opt_rl = rows[3]
    summary_json = out_dir / "synergy_matrix_summary.json"
    payload = {
        "default_schedule_key": default_key,
        "optimized_schedule_key": optimized_key,
        "default_plus_pid_profit": float(default_pid["net_profit"]),
        "default_plus_rl_profit": float(default_rl["net_profit"]),
        "optimized_plus_pid_profit": float(opt_pid["net_profit"]),
        "optimized_plus_rl_profit": float(opt_rl["net_profit"]),
        "upper_only_gain_vs_default_pid": float(opt_pid["net_profit"] - default_pid["net_profit"]),
        "lower_only_gain_vs_default_pid": float(default_rl["net_profit"] - default_pid["net_profit"]),
        "combined_gain_vs_default_pid": float(opt_rl["net_profit"] - default_pid["net_profit"]),
        "rl_gain_on_default_schedule": float(default_rl["net_profit"] - default_pid["net_profit"]),
        "rl_gain_on_optimized_schedule": float(opt_rl["net_profit"] - opt_pid["net_profit"]),
    }
    summary_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary_md = out_dir / "synergy_matrix_summary.md"
    lines = [
        "# M5 Synergy Matrix",
        "",
        f"- Default schedule: `{default_key}`",
        f"- Upper-optimized schedule: `{optimized_key}`",
        "",
        "| Combination | Net profit | Harvest fresh kg | Energy kWh | Valid full horizon |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['combination']} | "
            f"{float(row['net_profit']):.2f} | "
            f"{float(row['harvest_fresh_kg']):.2f} | "
            f"{float(row['energy_kwh']):.2f} | "
            f"{bool(row['valid_full_horizon'])} |"
        )
    lines.extend(
        [
            "",
            f"- Upper-layer-only gain vs default+PID: {payload['upper_only_gain_vs_default_pid']:.2f}",
            f"- Lower-layer-only gain vs default+PID: {payload['lower_only_gain_vs_default_pid']:.2f}",
            f"- Combined gain vs default+PID: {payload['combined_gain_vs_default_pid']:.2f}",
            f"- RL gain on default schedule: {payload['rl_gain_on_default_schedule']:.2f}",
            f"- RL gain on optimized schedule: {payload['rl_gain_on_optimized_schedule']:.2f}",
        ]
    )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[SAVE] CSV  -> {summary_csv}")
    print(f"[SAVE] JSON -> {summary_json}")
    print(f"[SAVE] MD   -> {summary_md}")


if __name__ == "__main__":
    main()
