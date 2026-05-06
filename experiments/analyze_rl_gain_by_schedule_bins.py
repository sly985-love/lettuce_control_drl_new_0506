# -*- coding: utf-8 -*-
"""Stratify exact RL-vs-PID gains by schedule structure."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.analyze_rl_gain_mechanisms import (  # noqa: E402
    DEFAULT_FEASIBLE_CSV,
    _style_axes,
    enrich_mechanism_metrics,
    load_baseline_results,
    load_feasible_metadata,
    merge_pid_rl,
)

EPS = 1.0e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze where exact RL gains concentrate across schedule structure, "
            "including rho2 / t2 / rho1 / throughput cadence strata."
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
        "--rho2-bins",
        type=int,
        default=4,
        help="Number of quantile bins for rho2 in overview/heatmap analysis.",
    )
    parser.add_argument(
        "--rho1-bins",
        type=int,
        default=4,
        help="Number of quantile bins for rho1 in overview/heatmap analysis.",
    )
    parser.add_argument(
        "--throughput-bins",
        type=int,
        default=4,
        help="Number of quantile bins for throughput (30/delta_t).",
    )
    parser.add_argument(
        "--min-group-size",
        type=int,
        default=3,
        help="Minimum group size before a group is treated as low-coverage in the report.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Default: <rl_csv_dir>/analyze_rl_gain_by_schedule_bins",
    )
    return parser.parse_args()


def _schedule_key_from_row(row: dict[str, Any]) -> str:
    return (
        f"t1={int(row['t1'])}|t2={int(row['t2'])}|"
        f"N1={int(row['N1'])}|rho2={int(round(float(row['rho2'])))}"
    )


def _deduplicate_schedule_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "schedule_key" not in df.columns:
        return df
    ordered = df.copy()
    ordered["_source_row_index"] = np.arange(len(ordered), dtype=int)
    deduped = ordered.drop_duplicates(subset=["schedule_key"], keep="last").copy()
    return deduped.sort_values("_source_row_index").drop(columns="_source_row_index").reset_index(
        drop=True
    )


def load_full_feasible_metadata(feasible_csv: Path) -> pd.DataFrame | None:
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


def attach_schedule_structure(
    df: pd.DataFrame,
    feasible_full_df: pd.DataFrame | None,
) -> pd.DataFrame:
    enriched = df.copy()
    if feasible_full_df is not None:
        merge_cols = [
            c for c in feasible_full_df.columns if c != "schedule_key" and c not in enriched.columns
        ]
        if merge_cols:
            enriched = enriched.merge(
                feasible_full_df[["schedule_key", *merge_cols]],
                on="schedule_key",
                how="left",
            )

    if "rho1" not in enriched.columns:
        enriched["rho1"] = np.nan
    enriched["rho1"] = pd.to_numeric(enriched["rho1"], errors="coerce")

    if "delta_t" not in enriched.columns:
        enriched["delta_t"] = np.gcd(
            enriched["t1"].astype(int).to_numpy(),
            enriched["t2"].astype(int).to_numpy(),
        )
    enriched["delta_t"] = pd.to_numeric(enriched["delta_t"], errors="coerce")

    if "total_cycle_days" not in enriched.columns:
        enriched["total_cycle_days"] = enriched["cycle_days"]
    enriched["total_cycle_days"] = pd.to_numeric(
        enriched["total_cycle_days"], errors="coerce"
    ).fillna(enriched["cycle_days"])

    if "PP" not in enriched.columns:
        enriched["PP"] = 16.0
    enriched["PP"] = pd.to_numeric(enriched["PP"], errors="coerce").fillna(16.0)
    unique_pp = sorted(
        enriched["PP"].dropna().round().astype(int).unique().tolist()
    )
    if unique_pp and unique_pp != [16]:
        raise RuntimeError(
            f"Schedule structure metadata is inconsistent with fixed PP=16: found {unique_pp}."
        )

    enriched["throughput_events_per_30d"] = 30.0 / enriched["delta_t"].clip(lower=EPS)
    enriched["schedule_density_load_index"] = (
        enriched["rho2"].astype(float) * enriched["total_cycle_days"].astype(float)
    )
    return enriched


def _format_interval_label(prefix: str, idx: int, left: float, right: float) -> str:
    left_str = f"{left:.1f}".rstrip("0").rstrip(".")
    right_str = f"{right:.1f}".rstrip("0").rstrip(".")
    return f"{prefix} Q{idx + 1} [{left_str}, {right_str}]"


def make_quantile_bins(
    series: pd.Series,
    n_bins: int,
    prefix: str,
) -> tuple[pd.Series, list[str]]:
    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna()
    if valid.empty:
        out = pd.Series(["missing"] * len(series), index=series.index, dtype="object")
        return out, ["missing"]

    unique_count = int(valid.nunique())
    if unique_count <= 1:
        only = float(valid.iloc[0])
        label = f"{prefix} all [{only:.1f}]"
        out = pd.Series([label] * len(series), index=series.index, dtype="object")
        return out, [label]

    q = max(1, min(int(n_bins), unique_count))
    raw = pd.qcut(valid, q=q, duplicates="drop")
    categories = list(raw.cat.categories)
    labels = [
        _format_interval_label(prefix, idx, float(cat.left), float(cat.right))
        for idx, cat in enumerate(categories)
    ]
    mapping = {cat: label for cat, label in zip(categories, labels)}

    out = pd.Series(pd.NA, index=series.index, dtype="object")
    out.loc[valid.index] = raw.cat.rename_categories(mapping).astype(str)
    out = out.fillna("missing")
    cat_type = pd.CategoricalDtype(categories=labels + ["missing"], ordered=True)
    return out.astype(cat_type), labels


def summarize_groups(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    grouped = df.groupby(group_col, dropna=False, observed=True)
    summary = grouped.agg(
        n_schedules=("schedule_key", "size"),
        mean_profit_gap_rl_minus_pid=("profit_gap_rl_minus_pid", "mean"),
        median_profit_gap_rl_minus_pid=("profit_gap_rl_minus_pid", "median"),
        mean_revenue_gap_rl_minus_pid=("revenue_gap_rl_minus_pid", "mean"),
        mean_cost_gap_rl_minus_pid=("cost_gap_rl_minus_pid", "mean"),
        mean_harvest_gap_rl_minus_pid_kg=("harvest_gap_rl_minus_pid_kg", "mean"),
        mean_energy_gap_rl_minus_pid_kwh=("energy_gap_rl_minus_pid_kwh", "mean"),
        mean_energy_per_kg_gap_rl_minus_pid=("energy_per_kg_gap_rl_minus_pid", "mean"),
        mean_cost_per_kg_gap_rl_minus_pid=("cost_per_kg_gap_rl_minus_pid", "mean"),
        rl_profit_win_rate=("rl_wins_by_profit", "mean"),
        rl_objective_win_rate=("rl_wins_by_objective", "mean"),
        both_valid_ratio=("both_valid_full_horizon", "mean"),
        mean_cycle_days=("cycle_days", "mean"),
        mean_delta_t=("delta_t", "mean"),
        mean_throughput_events_per_30d=("throughput_events_per_30d", "mean"),
        mean_rho2=("rho2", "mean"),
        mean_rho1=("rho1", "mean"),
    )
    class_counts = pd.crosstab(df[group_col], df["gain_mechanism_class"], dropna=False)
    if not class_counts.empty:
        class_counts = class_counts.rename(
            columns={col: f"class_count__{col}" for col in class_counts.columns}
        )
        summary = summary.join(class_counts, how="left")
        class_cols = [c for c in summary.columns if c.startswith("class_count__")]
        summary["dominant_gain_mechanism_class"] = (
            summary[class_cols].idxmax(axis=1).str.replace("class_count__", "", regex=False)
        )
        summary["dominant_gain_mechanism_class_count"] = (
            summary[class_cols].max(axis=1).fillna(0).astype(int)
        )
    else:
        summary["dominant_gain_mechanism_class"] = "unknown"
        summary["dominant_gain_mechanism_class_count"] = 0

    return summary.reset_index()


def compute_spearman_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    features = [
        "rho2",
        "t2",
        "rho1",
        "delta_t",
        "throughput_events_per_30d",
        "cycle_days",
        "schedule_density_load_index",
    ]
    targets = [
        "profit_gap_rl_minus_pid",
        "harvest_gap_rl_minus_pid_kg",
        "energy_gap_rl_minus_pid_kwh",
        "energy_per_kg_gap_rl_minus_pid",
    ]
    for feature in features:
        for target in targets:
            pair = df[[feature, target]].apply(pd.to_numeric, errors="coerce").dropna()
            rho = np.nan
            if len(pair) >= 3 and pair[feature].nunique() > 1 and pair[target].nunique() > 1:
                rho = float(pair.corr(method="spearman").iloc[0, 1])
            rows.append(
                {
                    "feature": feature,
                    "target": target,
                    "n": int(len(pair)),
                    "spearman_rho": rho,
                }
            )
    return pd.DataFrame(rows)


def _pick_group_extreme(group_df: pd.DataFrame, value_col: str, largest: bool) -> dict[str, Any] | None:
    if group_df.empty:
        return None
    ordered = group_df.sort_values(
        by=[value_col, "rl_profit_win_rate", "n_schedules"],
        ascending=[not largest, not largest, False],
    )
    row = ordered.iloc[0].to_dict()
    return {
        "group": str(row[group_df.columns[0]]),
        "n_schedules": int(row["n_schedules"]),
        "mean_profit_gap_rl_minus_pid": float(row["mean_profit_gap_rl_minus_pid"]),
        "rl_profit_win_rate": float(row["rl_profit_win_rate"]),
        "mean_energy_per_kg_gap_rl_minus_pid": float(
            row["mean_energy_per_kg_gap_rl_minus_pid"]
        ),
        "dominant_gain_mechanism_class": str(row["dominant_gain_mechanism_class"]),
    }


def build_summary(
    df: pd.DataFrame,
    group_tables: dict[str, pd.DataFrame],
    spearman_df: pd.DataFrame,
    min_group_size: int,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "n_common_schedules": int(len(df)),
        "n_rl_profit_wins": int(df["rl_wins_by_profit"].sum()),
        "rl_profit_win_rate": float(df["rl_wins_by_profit"].mean()),
        "mean_profit_gap_rl_minus_pid": float(df["profit_gap_rl_minus_pid"].mean()),
        "mean_harvest_gap_rl_minus_pid_kg": float(df["harvest_gap_rl_minus_pid_kg"].mean()),
        "mean_energy_gap_rl_minus_pid_kwh": float(df["energy_gap_rl_minus_pid_kwh"].mean()),
        "mean_energy_per_kg_gap_rl_minus_pid": float(
            df["energy_per_kg_gap_rl_minus_pid"].mean()
        ),
        "coverage_warning": None,
        "min_group_size_for_reporting": int(min_group_size),
        "group_extremes": {},
        "spearman": {},
    }

    if len(df) < 30:
        summary["coverage_warning"] = (
            "Current comparison set is still small. Stratified conclusions should be treated as "
            "pilot evidence until the exact RL baseline is run over the full feasible schedule set."
        )

    for name, table in group_tables.items():
        if table.empty:
            continue
        summary["group_extremes"][name] = {
            "best_mean_profit_gap_group": _pick_group_extreme(
                table, "mean_profit_gap_rl_minus_pid", largest=True
            ),
            "worst_mean_profit_gap_group": _pick_group_extreme(
                table, "mean_profit_gap_rl_minus_pid", largest=False
            ),
            "best_win_rate_group": _pick_group_extreme(
                table, "rl_profit_win_rate", largest=True
            ),
        }

    if not spearman_df.empty:
        for _, row in spearman_df.iterrows():
            summary["spearman"][f"{row['feature']}__{row['target']}"] = {
                "n": int(row["n"]),
                "spearman_rho": None
                if pd.isna(row["spearman_rho"])
                else float(row["spearman_rho"]),
            }
    return summary


def _plot_group_overview(
    ax: plt.Axes,
    group_df: pd.DataFrame,
    group_col: str,
    title: str,
    min_group_size: int,
) -> None:
    plot_df = group_df.copy()
    if plot_df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=11)
        ax.set_axis_off()
        return

    x_labels = plot_df[group_col].astype(str).tolist()
    x = np.arange(len(plot_df), dtype=float)
    colors = np.where(
        plot_df["n_schedules"].astype(int).values >= int(min_group_size),
        "#2b6cb0",
        "#a0aec0",
    )
    ax.bar(
        x,
        plot_df["mean_profit_gap_rl_minus_pid"].astype(float).values,
        color=colors,
        alpha=0.88,
    )
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=25, ha="right")
    ax.set_ylabel("Mean profit gap (RL-PID)")
    ax.set_title(title, fontsize=12, pad=10)
    _style_axes(ax)

    ax2 = ax.twinx()
    ax2.plot(
        x,
        plot_df["rl_profit_win_rate"].astype(float).values,
        color="#c53030",
        marker="o",
        linewidth=1.5,
        alpha=0.9,
    )
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_ylabel("RL profit win rate")

    for xi, yi, count in zip(
        x,
        plot_df["mean_profit_gap_rl_minus_pid"].astype(float).values,
        plot_df["n_schedules"].astype(int).values,
    ):
        y_text = yi + (0.02 * max(1.0, np.nanmax(np.abs(plot_df["mean_profit_gap_rl_minus_pid"])) or 1.0))
        ax.text(xi, y_text, f"n={count}", ha="center", va="bottom", fontsize=8, alpha=0.9)


def save_overview_figure(
    group_tables: dict[str, pd.DataFrame],
    min_group_size: int,
    out_png: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5))
    specs = [
        ("rho2_bin", "rho2_bin", "Gain vs finishing density strata"),
        ("t2", "t2", "Gain vs finishing-stage duration"),
        ("rho1_bin", "rho1_bin", "Gain vs dense-zone density strata"),
        ("delta_t", "delta_t", "Gain vs throughput cadence (delta_t)"),
    ]
    for ax, (table_key, group_col, title) in zip(axes.flat, specs):
        _plot_group_overview(ax, group_tables.get(table_key, pd.DataFrame()), group_col, title, min_group_size)
    fig.suptitle("Exact RL gain stratified by schedule structure", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _prepare_heatmap_inputs(
    df: pd.DataFrame,
    index_col: str,
    column_col: str,
    value_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    value_table = pd.pivot_table(
        df,
        index=index_col,
        columns=column_col,
        values=value_col,
        aggfunc="mean",
        observed=True,
    )
    count_table = pd.pivot_table(
        df,
        index=index_col,
        columns=column_col,
        values="schedule_key",
        aggfunc="count",
        observed=True,
    )
    return value_table, count_table


def _plot_heatmap(
    ax: plt.Axes,
    value_table: pd.DataFrame,
    count_table: pd.DataFrame,
    title: str,
    cmap: str,
    value_fmt: str,
    center_zero: bool = True,
) -> None:
    if value_table.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=11)
        ax.set_axis_off()
        return

    values = value_table.astype(float).to_numpy()
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        finite = np.array([0.0])

    norm = None
    if center_zero:
        vmax = float(np.nanmax(np.abs(finite)))
        vmax = max(vmax, EPS)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    im = ax.imshow(values, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(np.arange(value_table.shape[1]))
    ax.set_xticklabels([str(c) for c in value_table.columns], rotation=30, ha="right")
    ax.set_yticks(np.arange(value_table.shape[0]))
    ax.set_yticklabels([str(i) for i in value_table.index])
    ax.set_title(title, fontsize=12, pad=10)
    _style_axes(ax)
    ax.grid(False)

    if value_table.size <= 60:
        for r in range(value_table.shape[0]):
            for c in range(value_table.shape[1]):
                v = value_table.iloc[r, c]
                n = count_table.iloc[r, c]
                if pd.isna(v) or pd.isna(n) or int(n) <= 0:
                    continue
                txt = f"{format(float(v), value_fmt)}\n(n={int(n)})"
                ax.text(c, r, txt, ha="center", va="center", fontsize=8, color="black")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def save_heatmap_figure(df: pd.DataFrame, out_png: Path) -> None:
    profit_rho2_t2, count_rho2_t2 = _prepare_heatmap_inputs(
        df, "rho2_bin", "t2", "profit_gap_rl_minus_pid"
    )
    energy_rho2_t2, count_energy_rho2_t2 = _prepare_heatmap_inputs(
        df, "rho2_bin", "t2", "energy_per_kg_gap_rl_minus_pid"
    )
    profit_rho1_dt, count_rho1_dt = _prepare_heatmap_inputs(
        df, "rho1_bin", "delta_t", "profit_gap_rl_minus_pid"
    )
    win_rho1_dt, count_win_rho1_dt = _prepare_heatmap_inputs(
        df, "rho1_bin", "delta_t", "rl_wins_by_profit"
    )

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5))
    _plot_heatmap(
        axes[0, 0],
        profit_rho2_t2,
        count_rho2_t2,
        "Mean profit gap over rho2-bin x t2",
        cmap="coolwarm",
        value_fmt=".1f",
    )
    _plot_heatmap(
        axes[0, 1],
        energy_rho2_t2,
        count_energy_rho2_t2,
        "Mean energy/kg gap over rho2-bin x t2",
        cmap="coolwarm",
        value_fmt=".3f",
    )
    _plot_heatmap(
        axes[1, 0],
        profit_rho1_dt,
        count_rho1_dt,
        "Mean profit gap over rho1-bin x delta_t",
        cmap="coolwarm",
        value_fmt=".1f",
    )
    _plot_heatmap(
        axes[1, 1],
        win_rho1_dt,
        count_win_rho1_dt,
        "RL profit win rate over rho1-bin x delta_t",
        cmap="viridis",
        value_fmt=".2f",
        center_zero=False,
    )
    fig.suptitle("Where exact RL gains concentrate in schedule space", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_markdown_report(
    df: pd.DataFrame,
    summary: dict[str, Any],
    group_tables: dict[str, pd.DataFrame],
    spearman_df: pd.DataFrame,
    min_group_size: int,
) -> str:
    top_wins = df.sort_values(by="profit_gap_rl_minus_pid", ascending=False).head(10)
    top_losses = df.sort_values(by="profit_gap_rl_minus_pid", ascending=True).head(10)

    def _render_group_table(title: str, table: pd.DataFrame, group_col: str) -> list[str]:
        lines: list[str] = []
        lines.append(f"## {title}")
        lines.append("")
        if table.empty:
            lines.append("- No data")
            lines.append("")
            return lines
        lines.append(
            f"| {group_col} | n | mean profit gap | RL win rate | mean energy/kg gap | dominant class |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
        for _, row in table.iterrows():
            mark = " low_coverage" if int(row["n_schedules"]) < int(min_group_size) else ""
            lines.append(
                f"| {row[group_col]}{mark} | {int(row['n_schedules'])} | "
                f"{float(row['mean_profit_gap_rl_minus_pid']):.2f} | "
                f"{float(row['rl_profit_win_rate']):.3f} | "
                f"{float(row['mean_energy_per_kg_gap_rl_minus_pid']):.4f} | "
                f"{row['dominant_gain_mechanism_class']} |"
            )
        lines.append("")
        return lines

    lines: list[str] = []
    lines.append("# RL Gain Stratified Analysis")
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append(f"- Common schedules: {summary['n_common_schedules']}")
    lines.append(f"- RL profit win rate: {summary['rl_profit_win_rate']:.3f}")
    lines.append(f"- Mean RL-PID profit gap: {summary['mean_profit_gap_rl_minus_pid']:.2f}")
    lines.append(
        f"- Mean RL-PID harvest gap: {summary['mean_harvest_gap_rl_minus_pid_kg']:.2f} kg"
    )
    lines.append(
        f"- Mean RL-PID energy gap: {summary['mean_energy_gap_rl_minus_pid_kwh']:.2f} kWh"
    )
    lines.append(
        f"- Mean RL-PID energy/kg gap: {summary['mean_energy_per_kg_gap_rl_minus_pid']:.4f}"
    )
    if summary.get("coverage_warning"):
        lines.append(f"- Coverage warning: {summary['coverage_warning']}")
    lines.append("")
    lines.append("## What This Analysis Answers")
    lines.append("")
    lines.append("- Whether RL gains are concentrated in high-rho2 schedules or spread broadly.")
    lines.append("- Whether RL gains are stronger in shorter t2 / faster-throughput schedules.")
    lines.append("- Whether dense-zone load rho1 explains more of the gain than finishing-zone density alone.")
    lines.append(
        "- Whether RL wins mainly through lower energy per kg, lower total energy, or harvest/revenue expansion."
    )
    lines.append("")

    lines.extend(_render_group_table("Grouped by rho2 bins", group_tables["rho2_bin"], "rho2_bin"))
    lines.extend(_render_group_table("Grouped by t2", group_tables["t2"], "t2"))
    lines.extend(_render_group_table("Grouped by rho1 bins", group_tables["rho1_bin"], "rho1_bin"))
    lines.extend(_render_group_table("Grouped by delta_t", group_tables["delta_t"], "delta_t"))

    lines.append("## Spearman Correlations")
    lines.append("")
    lines.append("| feature | target | n | spearman rho |")
    lines.append("| --- | --- | ---: | ---: |")
    for _, row in spearman_df.iterrows():
        rho = "nan" if pd.isna(row["spearman_rho"]) else f"{float(row['spearman_rho']):.4f}"
        lines.append(
            f"| {row['feature']} | {row['target']} | {int(row['n'])} | {rho} |"
        )
    lines.append("")

    lines.append("## Top Profit Wins")
    lines.append("")
    lines.append("| schedule | profit gap | harvest gap kg | energy gap kWh | energy/kg gap | class |")
    lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
    for _, row in top_wins.iterrows():
        lines.append(
            f"| {row['schedule_key']} | {float(row['profit_gap_rl_minus_pid']):.2f} | "
            f"{float(row['harvest_gap_rl_minus_pid_kg']):.2f} | "
            f"{float(row['energy_gap_rl_minus_pid_kwh']):.2f} | "
            f"{float(row['energy_per_kg_gap_rl_minus_pid']):.4f} | "
            f"{row['gain_mechanism_class']} |"
        )
    lines.append("")

    lines.append("## Top Profit Losses")
    lines.append("")
    lines.append("| schedule | profit gap | harvest gap kg | energy gap kWh | energy/kg gap | class |")
    lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
    for _, row in top_losses.iterrows():
        lines.append(
            f"| {row['schedule_key']} | {float(row['profit_gap_rl_minus_pid']):.2f} | "
            f"{float(row['harvest_gap_rl_minus_pid_kg']):.2f} | "
            f"{float(row['energy_gap_rl_minus_pid_kwh']):.2f} | "
            f"{float(row['energy_per_kg_gap_rl_minus_pid']):.4f} | "
            f"{row['gain_mechanism_class']} |"
        )
    lines.append("")

    lines.append("## Publication Guidance")
    lines.append("")
    lines.append(
        "- If gains cluster mainly in high rho2 / high rho1 / small delta_t strata, the paper can argue that RL value emerges under high density and fast throughput pressure rather than in every schedule."
    )
    lines.append(
        "- If profit gaps are weak but energy/kg gaps are negative, the safer claim is economic-efficiency improvement rather than strong absolute-yield superiority."
    )
    lines.append(
        "- If neither win rate nor energy/kg improve in the challenging strata, the current RL design is not yet publication-ready for a strong superiority claim and should be repositioned as a negative or mixed result."
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    pid_csv = Path(args.pid_csv).resolve()
    rl_csv = Path(args.rl_csv).resolve()
    feasible_csv = Path(args.feasible_csv).resolve()
    out_dir = (
        Path(args.out_dir).resolve()
        if args.out_dir
        else rl_csv.parent / "analyze_rl_gain_by_schedule_bins"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    pid_df = load_baseline_results(pid_csv, "pid")
    rl_df = load_baseline_results(rl_csv, "rl")
    feasible_merge_df = load_feasible_metadata(feasible_csv)
    feasible_full_df = load_full_feasible_metadata(feasible_csv)

    merged = merge_pid_rl(pid_df, rl_df, feasible_merge_df)
    mechanism_df = enrich_mechanism_metrics(merged)
    mechanism_df = attach_schedule_structure(mechanism_df, feasible_full_df)
    mechanism_df["rho2_bin"], rho2_labels = make_quantile_bins(
        mechanism_df["rho2"], args.rho2_bins, "rho2"
    )
    mechanism_df["rho1_bin"], rho1_labels = make_quantile_bins(
        mechanism_df["rho1"], args.rho1_bins, "rho1"
    )
    mechanism_df["throughput_bin"], throughput_labels = make_quantile_bins(
        mechanism_df["throughput_events_per_30d"], args.throughput_bins, "throughput"
    )

    group_specs = {
        "rho2_exact": "rho2",
        "t2": "t2",
        "rho1_exact": "rho1",
        "delta_t": "delta_t",
        "rho2_bin": "rho2_bin",
        "rho1_bin": "rho1_bin",
        "throughput_bin": "throughput_bin",
    }
    group_tables: dict[str, pd.DataFrame] = {}
    for name, group_col in group_specs.items():
        group_tables[name] = summarize_groups(mechanism_df, group_col)
        group_tables[name].to_csv(
            out_dir / f"pid_rl_gain_grouped_by_{name}.csv",
            index=False,
            encoding="utf-8",
        )

    spearman_df = compute_spearman_summary(mechanism_df)
    summary = build_summary(
        mechanism_df,
        group_tables,
        spearman_df,
        min_group_size=args.min_group_size,
    )
    summary["rho2_bin_labels"] = rho2_labels
    summary["rho1_bin_labels"] = rho1_labels
    summary["throughput_bin_labels"] = throughput_labels

    mechanism_df.to_csv(
        out_dir / "pid_rl_gain_schedule_stratified_table.csv",
        index=False,
        encoding="utf-8",
    )
    spearman_df.to_csv(
        out_dir / "pid_rl_gain_spearman_summary.csv",
        index=False,
        encoding="utf-8",
    )
    with open(out_dir / "pid_rl_gain_by_schedule_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    save_overview_figure(
        {
            "rho2_bin": group_tables["rho2_bin"],
            "t2": group_tables["t2"],
            "rho1_bin": group_tables["rho1_bin"],
            "delta_t": group_tables["delta_t"],
        },
        min_group_size=args.min_group_size,
        out_png=out_dir / "pid_rl_gain_overview_by_schedule_structure.png",
    )
    save_heatmap_figure(
        mechanism_df,
        out_png=out_dir / "pid_rl_gain_heatmaps_by_schedule_structure.png",
    )

    report = build_markdown_report(
        mechanism_df,
        summary,
        {
            "rho2_bin": group_tables["rho2_bin"],
            "t2": group_tables["t2"],
            "rho1_bin": group_tables["rho1_bin"],
            "delta_t": group_tables["delta_t"],
        },
        spearman_df,
        min_group_size=args.min_group_size,
    )
    (out_dir / "pid_rl_gain_by_schedule_report.md").write_text(report, encoding="utf-8")

    print("\n" + "=" * 72)
    print("RL Gain Stratified Analysis")
    print("=" * 72)
    print(f"PID CSV          : {pid_csv}")
    print(f"RL CSV           : {rl_csv}")
    print(f"Feasible CSV     : {feasible_csv}")
    print(f"Output dir       : {out_dir}")
    print(f"Common schedules : {summary['n_common_schedules']}")
    print(f"RL profit wins   : {summary['n_rl_profit_wins']}")
    print(f"RL win rate      : {summary['rl_profit_win_rate']:.3f}")
    print(f"Mean profit gap  : {summary['mean_profit_gap_rl_minus_pid']:.2f}")
    print(
        f"Mean energy/kg   : {summary['mean_energy_per_kg_gap_rl_minus_pid']:.4f}"
    )
    print(f"Overview figure  : {out_dir / 'pid_rl_gain_overview_by_schedule_structure.png'}")
    print(f"Heatmap figure   : {out_dir / 'pid_rl_gain_heatmaps_by_schedule_structure.png'}")
    print(f"Summary JSON     : {out_dir / 'pid_rl_gain_by_schedule_summary.json'}")
    print(f"Markdown report  : {out_dir / 'pid_rl_gain_by_schedule_report.md'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
