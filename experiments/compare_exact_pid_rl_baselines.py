# -*- coding: utf-8 -*-
"""
Compare exact PID and exact RL schedule baselines on the same feasible set.

This script assumes both controllers have already been evaluated over the same
schedule catalog. It aligns the two result CSV files by `schedule_key`, then
produces:

1. schedule-wise merged comparison tables
2. summary JSON / Markdown for paper writing
3. reviewer-oriented figures (scatter, heatmaps, gap distributions)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.exact_pid_schedule_baseline import (  # noqa: E402
    DEFAULT_SCHEDULE,
    FEASIBLE_CSV_DEFAULT,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare exact PID and exact RL schedule baselines."
    )
    parser.add_argument(
        "--pid-csv",
        type=str,
        required=True,
        help="Path to pid_exact_schedule_results.csv or equivalent.",
    )
    parser.add_argument(
        "--rl-csv",
        type=str,
        required=True,
        help="Path to rl_exact_schedule_results.csv or equivalent.",
    )
    parser.add_argument(
        "--feasible-csv",
        type=str,
        default=str(FEASIBLE_CSV_DEFAULT),
        help="Optional feasible schedule catalog with reference metadata.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory. Default: <rl_csv_dir>/compare_pid_rl",
    )
    return parser.parse_args()


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.25, linewidth=0.6, linestyle="--")
    for spine in ax.spines.values():
        spine.set_alpha(0.3)


def _schedule_key_from_row(row: Dict[str, Any]) -> str:
    return (
        f"t1={int(row['t1'])}|t2={int(row['t2'])}|"
        f"N1={int(row['N1'])}|rho2={int(round(float(row['rho2'])))}"
    )


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
    df["valid_full_horizon"] = _coerce_bool_series(df["valid_full_horizon"])
    df["is_default_schedule"] = _coerce_bool_series(df["is_default_schedule"])
    if "schedule_key" not in df.columns:
        df["schedule_key"] = df.apply(lambda row: _schedule_key_from_row(row.to_dict()), axis=1)
    df = df.drop_duplicates(subset=["schedule_key"], keep="last").reset_index(drop=True)
    df["controller_label"] = str(label)
    df["cycle_days"] = df["t1"].astype(float) + df["t2"].astype(float)
    return df


def load_feasible_metadata(feasible_csv: Path) -> pd.DataFrame | None:
    if not feasible_csv.exists():
        return None
    df = pd.read_csv(feasible_csv)
    if df.empty:
        return None
    if "schedule_key" not in df.columns:
        df["schedule_key"] = df.apply(lambda row: _schedule_key_from_row(row.to_dict()), axis=1)
    df = df.drop_duplicates(subset=["schedule_key"], keep="last").reset_index(drop=True)
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
        ]
        if c in df.columns
    ]
    return df[keep_cols].copy() if keep_cols else None


def merge_pid_rl(pid_df: pd.DataFrame, rl_df: pd.DataFrame, feasible_df: pd.DataFrame | None) -> pd.DataFrame:
    merge_keys = ["schedule_key", "t1", "t2", "N1", "rho2"]
    pid_keep = [
        *merge_keys,
        "is_default_schedule",
        "valid_full_horizon",
        "objective_value",
        "net_profit",
        "harvest_fresh_kg",
        "harvest_dry_kg",
        "total_cost",
        "energy_kwh",
        "cum_reward",
        "termination_reason",
        "episode_completion_ratio",
    ]
    rl_keep = list(pid_keep)
    merged = pid_df[pid_keep].merge(
        rl_df[rl_keep],
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
    merged["energy_gap_rl_minus_pid_kwh"] = merged["energy_kwh_rl"] - merged["energy_kwh_pid"]
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


def build_summary(merged: pd.DataFrame) -> Dict[str, Any]:
    default_subset = merged[merged["is_default_schedule_pid"] | merged["is_default_schedule_rl"]]
    default_row = default_subset.iloc[0].to_dict() if not default_subset.empty else None
    pid_best = merged.sort_values(
        by=["objective_value_pid", "net_profit_pid"],
        ascending=[False, False],
    ).iloc[0].to_dict()
    rl_best = merged.sort_values(
        by=["objective_value_rl", "net_profit_rl"],
        ascending=[False, False],
    ).iloc[0].to_dict()
    both_valid = merged[merged["both_valid_full_horizon"]].copy()

    summary = {
        "n_common_schedules": int(len(merged)),
        "n_both_valid_full_horizon": int(len(both_valid)),
        "rl_valid_full_horizon_count": int(merged["valid_full_horizon_rl"].sum()),
        "pid_valid_full_horizon_count": int(merged["valid_full_horizon_pid"].sum()),
        "rl_win_rate_by_profit_all": float(merged["rl_wins_by_profit"].mean()),
        "rl_win_rate_by_objective_all": float(merged["rl_wins_by_objective"].mean()),
        "mean_profit_gap_rl_minus_pid": float(merged["profit_gap_rl_minus_pid"].mean()),
        "median_profit_gap_rl_minus_pid": float(merged["profit_gap_rl_minus_pid"].median()),
        "mean_harvest_gap_rl_minus_pid_kg": float(merged["harvest_gap_rl_minus_pid_kg"].mean()),
        "mean_energy_gap_rl_minus_pid_kwh": float(merged["energy_gap_rl_minus_pid_kwh"].mean()),
        "mean_cost_gap_rl_minus_pid": float(merged["cost_gap_rl_minus_pid"].mean()),
        "best_pid_schedule": {
            "schedule_key": str(pid_best["schedule_key"]),
            "objective_value": float(pid_best["objective_value_pid"]),
            "net_profit": float(pid_best["net_profit_pid"]),
            "harvest_fresh_kg": float(pid_best["harvest_fresh_kg_pid"]),
        },
        "best_rl_schedule": {
            "schedule_key": str(rl_best["schedule_key"]),
            "objective_value": float(rl_best["objective_value_rl"]),
            "net_profit": float(rl_best["net_profit_rl"]),
            "harvest_fresh_kg": float(rl_best["harvest_fresh_kg_rl"]),
        },
        "default_schedule_comparison": default_row,
        "reference_feasibility_class_counts": {
            str(k): int(v)
            for k, v in merged["reference_feasibility_class"].value_counts().items()
        },
    }
    if not both_valid.empty:
        summary["rl_win_rate_by_profit_both_valid"] = float(both_valid["rl_wins_by_profit"].mean())
        summary["rl_win_rate_by_objective_both_valid"] = float(both_valid["rl_wins_by_objective"].mean())
    else:
        summary["rl_win_rate_by_profit_both_valid"] = 0.0
        summary["rl_win_rate_by_objective_both_valid"] = 0.0
    return summary


def save_scatter_plot(merged: pd.DataFrame, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 7.0))
    x = merged["net_profit_pid"].astype(float).values
    y = merged["net_profit_rl"].astype(float).values
    c = merged["profit_gap_rl_minus_pid"].astype(float).values
    sc = ax.scatter(
        x,
        y,
        c=c,
        cmap="coolwarm",
        s=42,
        alpha=0.88,
        edgecolors="none",
    )
    lo = float(min(np.min(x), np.min(y)))
    hi = float(max(np.max(x), np.max(y)))
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1.0, alpha=0.7)
    default_subset = merged[merged["is_default_schedule_pid"] | merged["is_default_schedule_rl"]]
    if not default_subset.empty:
        row = default_subset.iloc[0]
        ax.scatter(
            [float(row["net_profit_pid"])],
            [float(row["net_profit_rl"])],
            s=180,
            c="#111111",
            marker="*",
            edgecolors="black",
            linewidths=0.8,
            label="Default schedule",
        )
        ax.legend(frameon=True)
    ax.set_xlabel("PID net profit")
    ax.set_ylabel("RL net profit")
    ax.set_title("Exact RL vs PID: Schedule-wise Net Profit")
    _style_axes(ax)
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("RL - PID profit gap")
    fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


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


def save_gap_heatmaps(merged: pd.DataFrame, out_png: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(17, 13))
    _plot_heatmap(
        axes[0, 0],
        merged,
        row_key="t1",
        col_key="t2",
        value_key="profit_gap_rl_minus_pid",
        title="Mean Profit Gap by (t1, t2)",
        cmap="coolwarm",
        cbar_label="RL - PID profit gap",
    )
    _plot_heatmap(
        axes[0, 1],
        merged,
        row_key="N1",
        col_key="rho2",
        value_key="profit_gap_rl_minus_pid",
        title="Mean Profit Gap by (N1, rho2)",
        cmap="coolwarm",
        cbar_label="RL - PID profit gap",
    )
    _plot_heatmap(
        axes[1, 0],
        merged.assign(rl_win_rate=merged["rl_wins_by_profit"].astype(float)),
        row_key="cycle_days",
        col_key="N1",
        value_key="rl_win_rate",
        title="RL Win Rate by (total cycle, N1)",
        cmap="YlGn",
        cbar_label="RL win rate",
    )

    class_order = (
        merged.groupby("reference_feasibility_class", as_index=False)["rl_wins_by_profit"]
        .mean()
        .sort_values(by="rl_wins_by_profit", ascending=False)
    )
    ax = axes[1, 1]
    if class_order.empty:
        ax.text(0.5, 0.5, "No class metadata", ha="center", va="center", fontsize=11)
        ax.set_axis_off()
    else:
        ax.bar(
            class_order["reference_feasibility_class"].astype(str),
            class_order["rl_wins_by_profit"].astype(float),
            color="#2b6cb0",
            alpha=0.9,
        )
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("RL win rate")
        ax.set_title("RL Win Rate by Reference Feasibility Class", fontsize=11, pad=8)
        ax.tick_params(axis="x", rotation=20)
        _style_axes(ax)
    fig.suptitle("Exact RL vs PID: Design-Space Comparison", fontsize=14, y=0.98)
    fig.subplots_adjust(top=0.92, wspace=0.28, hspace=0.30)
    fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_gap_distribution_plot(merged: pd.DataFrame, out_png: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    specs = [
        ("profit_gap_rl_minus_pid", "Profit gap", "#2b6cb0"),
        ("harvest_gap_rl_minus_pid_kg", "Harvest fresh mass gap [kg]", "#2f855a"),
        ("cost_gap_rl_minus_pid", "Cost gap", "#dd6b20"),
        ("energy_gap_rl_minus_pid_kwh", "Energy gap [kWh]", "#805ad5"),
    ]
    for ax, (key, title, color) in zip(axes.flat, specs):
        vals = merged[key].astype(float).values
        ax.hist(vals, bins=30, color=color, alpha=0.88, edgecolor="white")
        ax.axvline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
        ax.set_title(title, fontsize=11, pad=8)
        ax.set_xlabel(key)
        ax.set_ylabel("Schedule count")
        _style_axes(ax)
    fig.suptitle("Exact RL vs PID: Gap Distributions", fontsize=14, y=0.98)
    fig.subplots_adjust(top=0.92, wspace=0.24, hspace=0.28)
    fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_markdown_summary(summary: Dict[str, Any], merged: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("# Exact PID vs RL Comparison")
    lines.append("")
    lines.append(f"- Common schedules: {summary['n_common_schedules']}")
    lines.append(f"- Both valid full horizon: {summary['n_both_valid_full_horizon']}")
    lines.append(f"- RL valid count: {summary['rl_valid_full_horizon_count']}")
    lines.append(f"- PID valid count: {summary['pid_valid_full_horizon_count']}")
    lines.append(f"- RL win rate by profit (all): {summary['rl_win_rate_by_profit_all']:.3f}")
    lines.append(f"- RL win rate by profit (both valid): {summary['rl_win_rate_by_profit_both_valid']:.3f}")
    lines.append(f"- Mean RL-PID profit gap: {summary['mean_profit_gap_rl_minus_pid']:.3f}")
    lines.append("")
    lines.append("## Best Schedules")
    lines.append("")
    lines.append(
        f"- PID best: `{summary['best_pid_schedule']['schedule_key']}` | "
        f"profit={summary['best_pid_schedule']['net_profit']:.2f}"
    )
    lines.append(
        f"- RL best: `{summary['best_rl_schedule']['schedule_key']}` | "
        f"profit={summary['best_rl_schedule']['net_profit']:.2f}"
    )
    lines.append("")

    top_wins = merged.sort_values(by="profit_gap_rl_minus_pid", ascending=False).head(10)
    top_losses = merged.sort_values(by="profit_gap_rl_minus_pid", ascending=True).head(10)

    lines.append("## Top-10 RL Wins")
    lines.append("")
    lines.append("| schedule | RL-PID profit gap | RL profit | PID profit | class |")
    lines.append("| --- | ---: | ---: | ---: | --- |")
    for _, row in top_wins.iterrows():
        lines.append(
            f"| {row['schedule_key']} | {float(row['profit_gap_rl_minus_pid']):.2f} | "
            f"{float(row['net_profit_rl']):.2f} | {float(row['net_profit_pid']):.2f} | "
            f"{row.get('reference_feasibility_class', 'unknown')} |"
        )
    lines.append("")
    lines.append("## Top-10 RL Losses")
    lines.append("")
    lines.append("| schedule | RL-PID profit gap | RL profit | PID profit | class |")
    lines.append("| --- | ---: | ---: | ---: | --- |")
    for _, row in top_losses.iterrows():
        lines.append(
            f"| {row['schedule_key']} | {float(row['profit_gap_rl_minus_pid']):.2f} | "
            f"{float(row['net_profit_rl']):.2f} | {float(row['net_profit_pid']):.2f} | "
            f"{row.get('reference_feasibility_class', 'unknown')} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    pid_csv = Path(args.pid_csv).resolve()
    rl_csv = Path(args.rl_csv).resolve()
    out_dir = (
        Path(args.out_dir).resolve()
        if args.out_dir
        else rl_csv.parent / "compare_pid_rl"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    pid_df = load_baseline_results(pid_csv, "pid")
    rl_df = load_baseline_results(rl_csv, "rl")
    feasible_df = load_feasible_metadata(Path(args.feasible_csv).resolve())
    merged = merge_pid_rl(pid_df, rl_df, feasible_df)
    summary = build_summary(merged)

    merged_csv = out_dir / "pid_rl_exact_schedule_comparison.csv"
    merged.to_csv(merged_csv, index=False, encoding="utf-8")

    merged_json = out_dir / "pid_rl_exact_schedule_comparison.json"
    merged.to_json(merged_json, orient="records", force_ascii=False, indent=2)

    summary_json = out_dir / "pid_rl_exact_comparison_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    top_win_csv = out_dir / "pid_rl_top20_rl_wins.csv"
    merged.sort_values(by="profit_gap_rl_minus_pid", ascending=False).head(20).to_csv(
        top_win_csv, index=False, encoding="utf-8"
    )
    top_loss_csv = out_dir / "pid_rl_top20_rl_losses.csv"
    merged.sort_values(by="profit_gap_rl_minus_pid", ascending=True).head(20).to_csv(
        top_loss_csv, index=False, encoding="utf-8"
    )

    md_path = out_dir / "pid_rl_exact_comparison.md"
    md_path.write_text(build_markdown_summary(summary, merged), encoding="utf-8")

    save_scatter_plot(merged, out_dir / "pid_rl_profit_scatter.png")
    save_gap_heatmaps(merged, out_dir / "pid_rl_gap_heatmaps.png")
    save_gap_distribution_plot(merged, out_dir / "pid_rl_gap_distributions.png")

    print("\n" + "=" * 72)
    print("Exact PID vs RL Comparison")
    print("=" * 72)
    print(f"PID CSV            : {pid_csv}")
    print(f"RL CSV             : {rl_csv}")
    print(f"Common schedules   : {summary['n_common_schedules']}")
    print(f"Both valid         : {summary['n_both_valid_full_horizon']}")
    print(f"RL win rate profit : {summary['rl_win_rate_by_profit_all']:.3f}")
    print(f"Mean profit gap    : {summary['mean_profit_gap_rl_minus_pid']:.3f}")
    print(f"Merged CSV         : {merged_csv}")
    print(f"Summary JSON       : {summary_json}")
    print(f"Markdown report    : {md_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
