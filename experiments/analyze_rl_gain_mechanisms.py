# -*- coding: utf-8 -*-
"""Analyze where exact-RL gains come from relative to exact PID."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
LOCAL_FEASIBLE_CSV = ROOT / "data" / "feasibility" / "feasible_solutions.csv"
LEGACY_FEASIBLE_CSV = ROOT.parent / "results" / "feasibility" / "feasible_solutions.csv"
DEFAULT_FEASIBLE_CSV = (
    LOCAL_FEASIBLE_CSV if LOCAL_FEASIBLE_CSV.exists() else LEGACY_FEASIBLE_CSV
)


EPS = 1.0e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze whether exact RL gains over exact PID come primarily from "
            "higher revenue, lower total cost, lower energy per kg, or a mix."
        )
    )
    parser.add_argument("--pid-csv", type=str, required=True)
    parser.add_argument("--rl-csv", type=str, required=True)
    parser.add_argument(
        "--feasible-csv",
        type=str,
        default=str(DEFAULT_FEASIBLE_CSV),
    )
    parser.add_argument(
        "--pid-trace-csv",
        type=str,
        default=None,
        help="Optional detailed PID trajectory CSV for one representative schedule.",
    )
    parser.add_argument(
        "--rl-trace-csv",
        type=str,
        default=None,
        help="Optional detailed RL trajectory CSV for the same representative schedule.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Default: <rl_csv_dir>/analyze_rl_gain_mechanisms",
    )
    return parser.parse_args()


def _safe_divide(num: pd.Series | np.ndarray, den: pd.Series | np.ndarray) -> pd.Series:
    num_s = pd.Series(num, copy=False, dtype=float)
    den_s = pd.Series(den, copy=False, dtype=float)
    return num_s / den_s.clip(lower=EPS)


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.25, linewidth=0.6, linestyle="--")
    for spine in ax.spines.values():
        spine.set_alpha(0.3)


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def _schedule_key_from_row(row: Dict[str, Any]) -> str:
    return (
        f"t1={int(row['t1'])}|t2={int(row['t2'])}|"
        f"N1={int(row['N1'])}|rho2={int(round(float(row['rho2'])))}"
    )


def _deduplicate_schedule_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "schedule_key" not in df.columns:
        return df
    ordered = df.copy()
    ordered["_source_row_index"] = np.arange(len(ordered), dtype=int)
    deduped = ordered.drop_duplicates(subset=["schedule_key"], keep="last").copy()
    deduped["duplicate_count_for_schedule"] = (
        ordered.groupby("schedule_key")["schedule_key"].transform("size")
        .loc[deduped.index]
        .astype(int)
        .values
    )
    return deduped.sort_values("_source_row_index").reset_index(drop=True)


def load_baseline_results(csv_path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise RuntimeError(f"No rows found in {csv_path}")
    if "photo_period_hours" in df.columns:
        unique_pp = sorted(
            pd.to_numeric(df["photo_period_hours"], errors="coerce")
            .dropna()
            .round()
            .astype(int)
            .unique()
            .tolist()
        )
        if unique_pp and unique_pp != [16]:
            raise RuntimeError(
                f"Baseline CSV uses stale photoperiod semantics: {csv_path}. "
                f"Expected fixed PP=16, but found {unique_pp}."
            )
    if "valid_full_horizon" in df.columns:
        df["valid_full_horizon"] = _coerce_bool_series(df["valid_full_horizon"])
    else:
        df["valid_full_horizon"] = True
    if "is_default_schedule" in df.columns:
        df["is_default_schedule"] = _coerce_bool_series(df["is_default_schedule"])
    else:
        df["is_default_schedule"] = False
    if "schedule_key" not in df.columns:
        df["schedule_key"] = df.apply(lambda row: _schedule_key_from_row(row.to_dict()), axis=1)
    df = _deduplicate_schedule_rows(df)
    df["controller_label"] = str(label)
    df["cycle_days"] = df["t1"].astype(float) + df["t2"].astype(float)
    return df


def load_feasible_metadata(feasible_csv: Path) -> pd.DataFrame | None:
    if not feasible_csv.exists():
        return None
    df = pd.read_csv(feasible_csv)
    if df.empty:
        return None
    if "PP" in df.columns:
        unique_pp = sorted(
            pd.to_numeric(df["PP"], errors="coerce")
            .dropna()
            .round()
            .astype(int)
            .unique()
            .tolist()
        )
        if unique_pp and unique_pp != [16]:
            raise RuntimeError(
                f"Feasible schedule CSV uses stale photoperiod semantics: {feasible_csv}. "
                f"Expected fixed PP=16, but found {unique_pp}."
            )
    if "schedule_key" not in df.columns:
        df["schedule_key"] = df.apply(lambda row: _schedule_key_from_row(row.to_dict()), axis=1)
    df = _deduplicate_schedule_rows(df)
    keep_cols = [
        c
        for c in [
            "schedule_key",
            "reference_feasibility_class",
            "reference_target_feasible",
            "reference_min_feasible",
            "reference_harvest_fresh_mass_per_plant_g",
            "reference_harvest_dry_mass_per_plant_g",
            "reference_harvest_vs_target_ratio",
            "PP",
            "rho1",
            "rho1_continuous",
            "expansion_ratio",
            "total_cycle_days",
            "delta_t",
            "A1_m2",
            "A2_m2",
            "A1_A2_ratio",
        ]
        if c in df.columns
    ]
    return df[keep_cols].copy() if keep_cols else None


def merge_pid_rl(
    pid_df: pd.DataFrame,
    rl_df: pd.DataFrame,
    feasible_df: pd.DataFrame | None,
) -> pd.DataFrame:
    merge_keys = ["schedule_key", "t1", "t2", "N1", "rho2"]
    keep_cols = [
        *merge_keys,
        "is_default_schedule",
        "valid_full_horizon",
        "objective_value",
        "net_profit",
        "revenue",
        "harvest_fresh_kg",
        "harvest_dry_kg",
        "total_cost",
        "energy_kwh",
        "cum_reward",
        "termination_reason",
        "episode_completion_ratio",
    ]
    missing_pid = [c for c in keep_cols if c not in pid_df.columns]
    missing_rl = [c for c in keep_cols if c not in rl_df.columns]
    if missing_pid:
        raise RuntimeError(f"PID CSV missing required columns: {', '.join(missing_pid)}")
    if missing_rl:
        raise RuntimeError(f"RL CSV missing required columns: {', '.join(missing_rl)}")

    merged = pid_df[keep_cols].merge(
        rl_df[keep_cols],
        on=merge_keys,
        how="inner",
        suffixes=("_pid", "_rl"),
    )
    if merged.empty:
        raise RuntimeError("PID and RL result CSVs have no overlapping schedules.")

    merged["profit_gap_rl_minus_pid"] = merged["net_profit_rl"] - merged["net_profit_pid"]
    merged["harvest_gap_rl_minus_pid_kg"] = (
        merged["harvest_fresh_kg_rl"] - merged["harvest_fresh_kg_pid"]
    )
    merged["energy_gap_rl_minus_pid_kwh"] = (
        merged["energy_kwh_rl"] - merged["energy_kwh_pid"]
    )
    merged["cost_gap_rl_minus_pid"] = merged["total_cost_rl"] - merged["total_cost_pid"]
    merged["reward_gap_rl_minus_pid"] = merged["cum_reward_rl"] - merged["cum_reward_pid"]
    merged["objective_gap_rl_minus_pid"] = (
        merged["objective_value_rl"] - merged["objective_value_pid"]
    )
    merged["rl_wins_by_profit"] = merged["profit_gap_rl_minus_pid"] > 0.0
    merged["rl_wins_by_objective"] = merged["objective_gap_rl_minus_pid"] > 0.0
    merged["both_valid_full_horizon"] = (
        merged["valid_full_horizon_pid"] & merged["valid_full_horizon_rl"]
    )
    merged["cycle_days"] = merged["t1"].astype(float) + merged["t2"].astype(float)

    if feasible_df is not None:
        merged = merged.merge(feasible_df, on="schedule_key", how="left")
    if "reference_feasibility_class" not in merged.columns:
        merged["reference_feasibility_class"] = "unknown"

    return merged.sort_values(
        by=["profit_gap_rl_minus_pid", "objective_gap_rl_minus_pid"],
        ascending=[False, False],
    ).reset_index(drop=True)


def enrich_mechanism_metrics(merged: pd.DataFrame) -> pd.DataFrame:
    df = merged.copy()
    df["revenue_gap_rl_minus_pid"] = df["revenue_rl"] - df["revenue_pid"]
    df["profit_gap_reconstructed"] = (
        df["revenue_gap_rl_minus_pid"] - df["cost_gap_rl_minus_pid"]
    )
    df["profit_gap_reconstruction_error"] = (
        df["profit_gap_rl_minus_pid"] - df["profit_gap_reconstructed"]
    )

    df["energy_per_kg_pid"] = _safe_divide(df["energy_kwh_pid"], df["harvest_fresh_kg_pid"])
    df["energy_per_kg_rl"] = _safe_divide(df["energy_kwh_rl"], df["harvest_fresh_kg_rl"])
    df["cost_per_kg_pid"] = _safe_divide(df["total_cost_pid"], df["harvest_fresh_kg_pid"])
    df["cost_per_kg_rl"] = _safe_divide(df["total_cost_rl"], df["harvest_fresh_kg_rl"])
    df["profit_per_kg_pid"] = _safe_divide(df["net_profit_pid"], df["harvest_fresh_kg_pid"])
    df["profit_per_kg_rl"] = _safe_divide(df["net_profit_rl"], df["harvest_fresh_kg_rl"])

    df["energy_per_kg_gap_rl_minus_pid"] = df["energy_per_kg_rl"] - df["energy_per_kg_pid"]
    df["cost_per_kg_gap_rl_minus_pid"] = df["cost_per_kg_rl"] - df["cost_per_kg_pid"]
    df["profit_per_kg_gap_rl_minus_pid"] = df["profit_per_kg_rl"] - df["profit_per_kg_pid"]

    def _classify(row: pd.Series) -> str:
        profit_gap = float(row["profit_gap_rl_minus_pid"])
        revenue_gain = float(row["revenue_gap_rl_minus_pid"])
        cost_gain = float(-row["cost_gap_rl_minus_pid"])

        if (
            abs(profit_gap) <= EPS
            and abs(revenue_gain) <= EPS
            and abs(cost_gain) <= EPS
        ):
            return "tie_no_material_difference"

        if profit_gap <= 0.0:
            if revenue_gain < 0.0 and cost_gain <= 0.0:
                return "loss_revenue_down_cost_up"
            if revenue_gain < 0.0 and cost_gain > 0.0:
                return "loss_cost_saved_but_revenue_drop_dominant"
            if revenue_gain >= 0.0 and cost_gain <= 0.0:
                return "loss_revenue_up_but_cost_increase_dominant"
            return "loss_mixed_or_unclear"

        if revenue_gain > 0.0 and cost_gain > 0.0:
            if revenue_gain >= 1.5 * cost_gain:
                return "win_revenue_dominant"
            if cost_gain >= 1.5 * revenue_gain:
                return "win_cost_dominant"
            return "win_mixed_revenue_and_cost"
        if revenue_gain > 0.0 and cost_gain <= 0.0:
            return "win_revenue_offsets_cost_increase"
        if revenue_gain <= 0.0 and cost_gain > 0.0:
            return "win_cost_saving_offsets_revenue_loss"
        return "win_unclear"

    df["gain_mechanism_class"] = df.apply(_classify, axis=1)
    return df


def build_gain_summary(df: pd.DataFrame) -> dict[str, Any]:
    wins = df[df["profit_gap_rl_minus_pid"] > 0.0].copy()
    losses = df[df["profit_gap_rl_minus_pid"] <= 0.0].copy()

    def _avg(sub: pd.DataFrame, key: str) -> float:
        if sub.empty:
            return 0.0
        return float(sub[key].mean())

    summary: dict[str, Any] = {
        "n_common_schedules": int(len(df)),
        "n_rl_profit_wins": int(len(wins)),
        "n_rl_profit_losses_or_ties": int(len(losses)),
        "rl_profit_win_rate": float((df["profit_gap_rl_minus_pid"] > 0.0).mean()),
        "mean_profit_gap_rl_minus_pid": _avg(df, "profit_gap_rl_minus_pid"),
        "mean_revenue_gap_rl_minus_pid": _avg(df, "revenue_gap_rl_minus_pid"),
        "mean_cost_gap_rl_minus_pid": _avg(df, "cost_gap_rl_minus_pid"),
        "mean_energy_gap_rl_minus_pid_kwh": _avg(df, "energy_gap_rl_minus_pid_kwh"),
        "mean_harvest_gap_rl_minus_pid_kg": _avg(df, "harvest_gap_rl_minus_pid_kg"),
        "mean_energy_per_kg_gap_rl_minus_pid": _avg(df, "energy_per_kg_gap_rl_minus_pid"),
        "mean_cost_per_kg_gap_rl_minus_pid": _avg(df, "cost_per_kg_gap_rl_minus_pid"),
        "mean_profit_per_kg_gap_rl_minus_pid": _avg(df, "profit_per_kg_gap_rl_minus_pid"),
        "wins_mean_profit_gap": _avg(wins, "profit_gap_rl_minus_pid"),
        "wins_mean_revenue_gap": _avg(wins, "revenue_gap_rl_minus_pid"),
        "wins_mean_cost_gap": _avg(wins, "cost_gap_rl_minus_pid"),
        "wins_mean_energy_gap_kwh": _avg(wins, "energy_gap_rl_minus_pid_kwh"),
        "wins_mean_harvest_gap_kg": _avg(wins, "harvest_gap_rl_minus_pid_kg"),
        "wins_mean_energy_per_kg_gap": _avg(wins, "energy_per_kg_gap_rl_minus_pid"),
        "wins_mean_cost_per_kg_gap": _avg(wins, "cost_per_kg_gap_rl_minus_pid"),
        "wins_mean_profit_per_kg_gap": _avg(wins, "profit_per_kg_gap_rl_minus_pid"),
        "losses_mean_profit_gap": _avg(losses, "profit_gap_rl_minus_pid"),
        "losses_mean_revenue_gap": _avg(losses, "revenue_gap_rl_minus_pid"),
        "losses_mean_cost_gap": _avg(losses, "cost_gap_rl_minus_pid"),
        "losses_mean_energy_gap_kwh": _avg(losses, "energy_gap_rl_minus_pid_kwh"),
        "losses_mean_harvest_gap_kg": _avg(losses, "harvest_gap_rl_minus_pid_kg"),
        "gain_mechanism_counts": {
            str(k): int(v) for k, v in df["gain_mechanism_class"].value_counts().items()
        },
        "win_gain_mechanism_counts": {
            str(k): int(v) for k, v in wins["gain_mechanism_class"].value_counts().items()
        },
        "max_profit_win_schedule": None,
        "max_profit_loss_schedule": None,
    }

    if not wins.empty:
        best = wins.sort_values(by="profit_gap_rl_minus_pid", ascending=False).iloc[0]
        summary["max_profit_win_schedule"] = {
            "schedule_key": str(best["schedule_key"]),
            "profit_gap": float(best["profit_gap_rl_minus_pid"]),
            "revenue_gap": float(best["revenue_gap_rl_minus_pid"]),
            "cost_gap": float(best["cost_gap_rl_minus_pid"]),
            "energy_gap_kwh": float(best["energy_gap_rl_minus_pid_kwh"]),
            "harvest_gap_kg": float(best["harvest_gap_rl_minus_pid_kg"]),
            "gain_mechanism_class": str(best["gain_mechanism_class"]),
        }
    if not losses.empty:
        worst = losses.sort_values(by="profit_gap_rl_minus_pid", ascending=True).iloc[0]
        summary["max_profit_loss_schedule"] = {
            "schedule_key": str(worst["schedule_key"]),
            "profit_gap": float(worst["profit_gap_rl_minus_pid"]),
            "revenue_gap": float(worst["revenue_gap_rl_minus_pid"]),
            "cost_gap": float(worst["cost_gap_rl_minus_pid"]),
            "energy_gap_kwh": float(worst["energy_gap_rl_minus_pid_kwh"]),
            "harvest_gap_kg": float(worst["harvest_gap_rl_minus_pid_kg"]),
            "gain_mechanism_class": str(worst["gain_mechanism_class"]),
        }
    return summary


def _required_trace_columns() -> list[str]:
    return [
        "step_size_s",
        "P_LED_total_kW",
        "P_HVAC_kW",
        "P_heating_kW",
        "P_cooling_kW",
        "P_CO2_kW",
        "P_dehum_kW",
        "P_total_kW",
        "cost_elec_rmb",
        "cost_CO2_rmb",
        "cost_total_rmb",
        "I1",
        "I2",
        "Q_HVAC",
        "u_CO2",
        "m_dehum",
        "harvest_fresh_mass_equiv_g",
        "cum_reward",
    ]


def summarize_trace(csv_path: Path) -> dict[str, float]:
    df = pd.read_csv(csv_path)
    missing = [c for c in _required_trace_columns() if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Trace file {csv_path} is missing required columns: {', '.join(missing)}"
        )
    dt_h = pd.Series(df["step_size_s"], dtype=float) / 3600.0
    summary = {
        "led_kwh": float((pd.Series(df["P_LED_total_kW"], dtype=float) * dt_h).sum()),
        "hvac_kwh": float((pd.Series(df["P_HVAC_kW"], dtype=float) * dt_h).sum()),
        "heating_kwh": float((pd.Series(df["P_heating_kW"], dtype=float) * dt_h).sum()),
        "cooling_kwh": float((pd.Series(df["P_cooling_kW"], dtype=float) * dt_h).sum()),
        "co2_actuation_kwh": float((pd.Series(df["P_CO2_kW"], dtype=float) * dt_h).sum()),
        "dehum_kwh": float((pd.Series(df["P_dehum_kW"], dtype=float) * dt_h).sum()),
        "total_kwh": float((pd.Series(df["P_total_kW"], dtype=float) * dt_h).sum()),
        "electricity_cost_rmb": float(pd.Series(df["cost_elec_rmb"], dtype=float).sum()),
        "co2_cost_rmb": float(pd.Series(df["cost_CO2_rmb"], dtype=float).sum()),
        "total_cost_rmb": float(pd.Series(df["cost_total_rmb"], dtype=float).sum()),
        "mean_I1": float(pd.Series(df["I1"], dtype=float).mean()),
        "mean_I2": float(pd.Series(df["I2"], dtype=float).mean()),
        "mean_Q_HVAC": float(pd.Series(df["Q_HVAC"], dtype=float).mean()),
        "mean_abs_Q_HVAC": float(pd.Series(df["Q_HVAC"], dtype=float).abs().mean()),
        "mean_u_CO2": float(pd.Series(df["u_CO2"], dtype=float).mean()),
        "mean_m_dehum": float(pd.Series(df["m_dehum"], dtype=float).mean()),
        "final_cum_reward": float(pd.Series(df["cum_reward"], dtype=float).iloc[-1]),
        "total_harvest_fresh_kg": float(
            pd.Series(df["harvest_fresh_mass_equiv_g"], dtype=float).sum() / 1000.0
        ),
    }
    summary["energy_per_kg"] = summary["total_kwh"] / max(summary["total_harvest_fresh_kg"], EPS)
    summary["cost_per_kg"] = summary["total_cost_rmb"] / max(summary["total_harvest_fresh_kg"], EPS)
    return summary


def compare_traces(pid_trace: dict[str, float], rl_trace: dict[str, float]) -> pd.DataFrame:
    keys = [
        "led_kwh",
        "hvac_kwh",
        "heating_kwh",
        "cooling_kwh",
        "co2_actuation_kwh",
        "dehum_kwh",
        "total_kwh",
        "electricity_cost_rmb",
        "co2_cost_rmb",
        "total_cost_rmb",
        "total_harvest_fresh_kg",
        "energy_per_kg",
        "cost_per_kg",
        "mean_I1",
        "mean_I2",
        "mean_Q_HVAC",
        "mean_abs_Q_HVAC",
        "mean_u_CO2",
        "mean_m_dehum",
    ]
    rows = []
    for key in keys:
        pid_v = float(pid_trace.get(key, 0.0))
        rl_v = float(rl_trace.get(key, 0.0))
        rows.append(
            {
                "metric": key,
                "pid": pid_v,
                "rl": rl_v,
                "delta_rl_minus_pid": rl_v - pid_v,
                "relative_delta_vs_pid": (rl_v - pid_v) / max(abs(pid_v), EPS),
            }
        )
    return pd.DataFrame(rows)


def save_gain_driver_plot(trace_cmp: pd.DataFrame, out_png: Path) -> None:
    focus = trace_cmp[
        trace_cmp["metric"].isin(
            [
                "led_kwh",
                "heating_kwh",
                "cooling_kwh",
                "dehum_kwh",
                "co2_actuation_kwh",
                "total_kwh",
                "electricity_cost_rmb",
                "co2_cost_rmb",
                "total_cost_rmb",
            ]
        )
    ].copy()
    fig, ax = plt.subplots(figsize=(11, 6.4))
    colors = np.where(focus["delta_rl_minus_pid"].values <= 0.0, "#2f855a", "#c53030")
    ax.barh(focus["metric"], focus["delta_rl_minus_pid"], color=colors, alpha=0.9)
    ax.axvline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.set_title("Representative schedule: actuator/cost deltas (RL - PID)", fontsize=12, pad=10)
    ax.set_xlabel("Delta (negative is RL lower)")
    _style_axes(ax)
    fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_profit_source_scatter(df: pd.DataFrame, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 7.0))
    x = df["cost_gap_rl_minus_pid"].astype(float).values
    y = df["revenue_gap_rl_minus_pid"].astype(float).values
    c = df["profit_gap_rl_minus_pid"].astype(float).values
    sc = ax.scatter(x, y, c=c, cmap="coolwarm", s=42, alpha=0.88, edgecolors="none")
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.axvline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.set_xlabel("Total cost gap RL-PID")
    ax.set_ylabel("Revenue gap RL-PID")
    ax.set_title("Where RL profit gains come from", fontsize=12, pad=10)
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Profit gap RL-PID")
    _style_axes(ax)
    fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_mechanism_class_plot(df: pd.DataFrame, out_png: Path) -> None:
    counts = (
        df["gain_mechanism_class"]
        .value_counts()
        .rename_axis("gain_mechanism_class")
        .reset_index(name="count")
    )
    fig, ax = plt.subplots(figsize=(12.0, 5.8))
    ax.bar(counts["gain_mechanism_class"], counts["count"], color="#2b6cb0", alpha=0.9)
    ax.set_ylabel("Schedule count")
    ax.set_title("RL gain mechanism classes across schedules", fontsize=12, pad=10)
    ax.tick_params(axis="x", rotation=30)
    _style_axes(ax)
    fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_markdown_report(
    summary: dict[str, Any],
    df: pd.DataFrame,
    trace_cmp: pd.DataFrame | None,
) -> str:
    wins = df[df["profit_gap_rl_minus_pid"] > 0.0].copy()
    top_wins = wins.sort_values(by="profit_gap_rl_minus_pid", ascending=False).head(10)
    lines: list[str] = []
    lines.append("# RL Gain Mechanism Analysis")
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append(f"- Common schedules: {summary['n_common_schedules']}")
    lines.append(f"- RL profit win rate: {summary['rl_profit_win_rate']:.3f}")
    lines.append(f"- Mean RL-PID profit gap: {summary['mean_profit_gap_rl_minus_pid']:.2f}")
    lines.append(f"- Mean RL-PID revenue gap: {summary['mean_revenue_gap_rl_minus_pid']:.2f}")
    lines.append(f"- Mean RL-PID cost gap: {summary['mean_cost_gap_rl_minus_pid']:.2f}")
    lines.append(
        f"- Mean RL-PID energy-per-kg gap: {summary['mean_energy_per_kg_gap_rl_minus_pid']:.4f}"
    )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- If `revenue_gap_rl_minus_pid > 0` and `cost_gap_rl_minus_pid < 0`, RL wins through both "
        "higher revenue and lower cost."
    )
    lines.append(
        "- If `revenue_gap_rl_minus_pid <= 0` but `cost_gap_rl_minus_pid < 0`, RL is winning mainly "
        "through cost savings."
    )
    lines.append(
        "- If `revenue_gap_rl_minus_pid > 0` but `cost_gap_rl_minus_pid >= 0`, RL is winning mainly "
        "through higher harvest/revenue rather than cheaper operation."
    )
    lines.append(
        "- `energy_per_kg_gap_rl_minus_pid < 0` is the strongest schedule-level signal that RL is "
        "improving energy efficiency rather than simply using more energy to push yield."
    )
    lines.append("")
    lines.append("## Gain Mechanism Counts")
    lines.append("")
    for key, value in summary["gain_mechanism_counts"].items():
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    lines.append("## Top RL Profit Wins")
    lines.append("")
    lines.append(
        "| schedule | profit gap | revenue gap | cost gap | harvest gap kg | energy gap kWh | class |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for _, row in top_wins.iterrows():
        lines.append(
            f"| {row['schedule_key']} | {float(row['profit_gap_rl_minus_pid']):.2f} | "
            f"{float(row['revenue_gap_rl_minus_pid']):.2f} | {float(row['cost_gap_rl_minus_pid']):.2f} | "
            f"{float(row['harvest_gap_rl_minus_pid_kg']):.2f} | {float(row['energy_gap_rl_minus_pid_kwh']):.2f} | "
            f"{row['gain_mechanism_class']} |"
        )
    lines.append("")
    lines.append("## Recommended Next Step")
    lines.append("")
    lines.append(
        "- Schedule-level summary can already tell whether RL wins mainly through more harvest, lower total "
        "cost, or lower energy per kg."
    )
    lines.append(
        "- To attribute the gain to `lighting / heating / cooling / dehumidification / CO2`, run this script "
        "again with `--pid-trace-csv` and `--rl-trace-csv` for the same representative schedule."
    )
    if trace_cmp is not None:
        lines.append("")
        lines.append("## Representative Trace Decomposition")
        lines.append("")
        lines.append("| metric | PID | RL | RL-PID |")
        lines.append("| --- | ---: | ---: | ---: |")
        for _, row in trace_cmp.iterrows():
            lines.append(
                f"| {row['metric']} | {float(row['pid']):.4f} | {float(row['rl']):.4f} | "
                f"{float(row['delta_rl_minus_pid']):.4f} |"
            )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    pid_csv = Path(args.pid_csv).resolve()
    rl_csv = Path(args.rl_csv).resolve()
    feasible_csv = Path(args.feasible_csv).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else rl_csv.parent / "analyze_rl_gain_mechanisms"
    out_dir.mkdir(parents=True, exist_ok=True)

    pid_df = load_baseline_results(pid_csv, "pid")
    rl_df = load_baseline_results(rl_csv, "rl")
    feasible_df = load_feasible_metadata(feasible_csv)
    merged = merge_pid_rl(pid_df, rl_df, feasible_df)
    mechanism_df = enrich_mechanism_metrics(merged)
    summary = build_gain_summary(mechanism_df)

    mechanism_csv = out_dir / "pid_rl_gain_mechanism_schedule_table.csv"
    mechanism_df.to_csv(mechanism_csv, index=False, encoding="utf-8")

    summary_json = out_dir / "pid_rl_gain_mechanism_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    save_profit_source_scatter(mechanism_df, out_dir / "pid_rl_profit_source_scatter.png")
    save_mechanism_class_plot(mechanism_df, out_dir / "pid_rl_gain_mechanism_classes.png")

    trace_cmp: pd.DataFrame | None = None
    if args.pid_trace_csv and args.rl_trace_csv:
        pid_trace = summarize_trace(Path(args.pid_trace_csv).resolve())
        rl_trace = summarize_trace(Path(args.rl_trace_csv).resolve())
        trace_cmp = compare_traces(pid_trace, rl_trace)
        trace_cmp.to_csv(out_dir / "representative_trace_delta_table.csv", index=False, encoding="utf-8")
        save_gain_driver_plot(trace_cmp, out_dir / "representative_trace_gain_drivers.png")

    md_path = out_dir / "pid_rl_gain_mechanism_report.md"
    md_path.write_text(build_markdown_report(summary, mechanism_df, trace_cmp), encoding="utf-8")

    print("\n" + "=" * 72)
    print("RL Gain Mechanism Analysis")
    print("=" * 72)
    print(f"PID CSV            : {pid_csv}")
    print(f"RL CSV             : {rl_csv}")
    print(f"Output dir         : {out_dir}")
    print(f"RL profit win rate : {summary['rl_profit_win_rate']:.3f}")
    print(f"Mean profit gap    : {summary['mean_profit_gap_rl_minus_pid']:.2f}")
    print(f"Mean revenue gap   : {summary['mean_revenue_gap_rl_minus_pid']:.2f}")
    print(f"Mean cost gap      : {summary['mean_cost_gap_rl_minus_pid']:.2f}")
    print(f"Mean energy/kg gap : {summary['mean_energy_per_kg_gap_rl_minus_pid']:.4f}")
    print(f"Schedule table     : {mechanism_csv}")
    print(f"Summary JSON       : {summary_json}")
    print(f"Markdown report    : {md_path}")
    if trace_cmp is not None:
        print(f"Trace delta table  : {out_dir / 'representative_trace_delta_table.csv'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
