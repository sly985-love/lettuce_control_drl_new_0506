from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

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

from experiments.exact_pid_schedule_baseline import (  # noqa: E402
    DEFAULT_SCHEDULE,
    generate_exact_baseline_plots,
    summarise_results,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a partial-progress analysis snapshot for the ongoing exact PID baseline."
    )
    parser.add_argument(
        "--results_csv",
        type=str,
        default=str(
            ROOT
            / "results"
            / "exact_pid_baseline_365d_coupled_20260417_v1"
            / "pid_exact_schedule_results.csv"
        ),
        help="Current exact-baseline results CSV that is being appended during the long run.",
    )
    parser.add_argument(
        "--feasible_csv",
        type=str,
        default=str(FEASIBLE_CSV_DEFAULT),
        help="Feasible schedule catalog for total-count and coverage axes.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory. Default: <results_csv_dir>/partial_analysis_current",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Top-k valid schedules to export in the current partial ranking.",
    )
    return parser.parse_args()


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.25, linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_alpha(0.3)


def _load_results_df(results_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(results_csv)
    if df.empty:
        raise RuntimeError(f"No rows found in {results_csv}")
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
                f"Results CSV uses stale photoperiod semantics: {results_csv}. "
                f"Expected fixed PP=16, but found {unique_pp}. Please rebuild with --overwrite."
            )
    if "schedule_key" in df.columns:
        df = df.drop_duplicates(subset=["schedule_key"], keep="last").reset_index(drop=True)
    df["valid_full_horizon"] = _coerce_bool_series(df["valid_full_horizon"])
    df["is_default_schedule"] = _coerce_bool_series(df["is_default_schedule"])
    df["terminated_early"] = _coerce_bool_series(df["terminated_early"])
    df["cycle_days"] = df["t1"].astype(float) + df["t2"].astype(float)
    return df


def _load_feasible_df(feasible_csv: Path) -> pd.DataFrame | None:
    if not feasible_csv.exists():
        return None
    df = pd.read_csv(feasible_csv)
    if df.empty:
        return None
    dedup_keys = [c for c in ("t1", "t2", "N1", "rho2") if c in df.columns]
    if dedup_keys:
        df = df.drop_duplicates(subset=dedup_keys, keep="last").reset_index(drop=True)
    return df


def _get_axis_values(feasible_df: pd.DataFrame | None, current_df: pd.DataFrame, key: str) -> list[float]:
    if feasible_df is not None and key in feasible_df.columns:
        values = feasible_df[key].dropna().astype(float).unique().tolist()
    else:
        values = current_df[key].dropna().astype(float).unique().tolist()
    return sorted(values)


def _schedule_label(row: dict[str, Any] | None) -> str:
    if not row:
        return "N/A"
    return (
        f"t1={int(row['t1'])}, t2={int(row['t2'])}, "
        f"N1={int(row['N1'])}, rho2={int(round(float(row['rho2'])))}"
    )


def _save_parameter_summary(
    df: pd.DataFrame,
    out_csv: Path,
) -> pd.DataFrame:
    plot_df = df[df["valid_full_horizon"]].copy()
    if plot_df.empty:
        plot_df = df.copy()

    parts: list[pd.DataFrame] = []
    plot_df["cycle_days"] = plot_df["t1"].astype(float) + plot_df["t2"].astype(float)

    for key in ("t1", "t2", "N1", "rho2", "cycle_days"):
        grouped = (
            plot_df.groupby(key, as_index=False)
            .agg(
                n_schedules=("schedule_key", "count"),
                mean_profit=("net_profit", "mean"),
                std_profit=("net_profit", "std"),
                mean_harvest_fresh_kg=("harvest_fresh_kg", "mean"),
                mean_energy_kwh=("energy_kwh", "mean"),
                mean_total_cost=("total_cost", "mean"),
            )
            .rename(columns={key: "value"})
        )
        grouped.insert(0, "parameter", key)
        parts.append(grouped)

    summary_df = pd.concat(parts, ignore_index=True)
    summary_df.to_csv(out_csv, index=False, encoding="utf-8")
    return summary_df


def _plot_parameter_trends(
    summary_df: pd.DataFrame,
    out_png: Path,
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes_flat = axes.flatten()
    parameter_order = ("t1", "t2", "N1", "rho2", "cycle_days")
    color_map = {
        "t1": "#2b6cb0",
        "t2": "#2f855a",
        "N1": "#dd6b20",
        "rho2": "#805ad5",
        "cycle_days": "#c53030",
    }

    for ax, key in zip(axes_flat, parameter_order):
        sub = summary_df[summary_df["parameter"] == key].copy()
        sub = sub.sort_values("value").reset_index(drop=True)
        x = sub["value"].astype(float).values
        y = sub["mean_profit"].astype(float).values
        std = sub["std_profit"].fillna(0.0).astype(float).values
        counts = sub["n_schedules"].astype(int).values

        ax.plot(x, y, marker="o", linewidth=2.0, color=color_map[key])
        ax.fill_between(x, y - std, y + std, color=color_map[key], alpha=0.18)
        ax.axvline(float(DEFAULT_SCHEDULE[key]), color="#d62728", linestyle="--", linewidth=1.3)
        ax.set_title(f"Mean net profit vs {key}", fontsize=11, pad=8)
        ax.set_xlabel(key)
        ax.set_ylabel("Mean net profit")
        _style_axes(ax)

        for xi, yi, count in zip(x, y, counts):
            ax.text(xi, yi, f"n={count}", fontsize=8, ha="center", va="bottom")

    note_ax = axes_flat[-1]
    note_ax.set_axis_off()
    note_ax.text(
        0.03,
        0.95,
        "Default schedule anchor\n"
        f"t1={DEFAULT_SCHEDULE['t1']}, t2={DEFAULT_SCHEDULE['t2']}, "
        f"N1={DEFAULT_SCHEDULE['N1']}, rho2={int(DEFAULT_SCHEDULE['rho2'])}, "
        f"cycle={DEFAULT_SCHEDULE['t1'] + DEFAULT_SCHEDULE['t2']}\n\n"
        "Each panel shows the current partial mean net profit across\n"
        "already evaluated schedules only. Dashed red line marks the\n"
        "manual default schedule coordinate for that parameter.",
        fontsize=11,
        va="top",
        ha="left",
    )

    fig.suptitle("Partial exact PID baseline: parameter trends", fontsize=14, y=0.98)
    fig.subplots_adjust(top=0.92, wspace=0.28, hspace=0.32)
    fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_coverage_heatmap(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    row_key: str,
    col_key: str,
    row_values: list[float],
    col_values: list[float],
    default_row_val: float,
    default_col_val: float,
    title: str,
) -> None:
    pivot = df.pivot_table(index=row_key, columns=col_key, values="schedule_key", aggfunc="count", fill_value=0)
    pivot = pivot.reindex(index=row_values, columns=col_values, fill_value=0)

    im = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="Blues")
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlabel(col_key)
    ax.set_ylabel(row_key)
    ax.set_xticks(range(len(col_values)))
    ax.set_xticklabels([str(int(v)) if float(v).is_integer() else f"{v:g}" for v in col_values], rotation=45, ha="right")
    ax.set_yticks(range(len(row_values)))
    ax.set_yticklabels([str(int(v)) if float(v).is_integer() else f"{v:g}" for v in row_values])
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Evaluated schedules")

    if default_row_val in row_values and default_col_val in col_values:
        y = row_values.index(default_row_val)
        x = col_values.index(default_col_val)
        ax.scatter(
            [x],
            [y],
            marker="s",
            s=180,
            c="#d62728",
            edgecolors="black",
            linewidths=1.0,
            zorder=5,
            label="Default schedule",
        )
        ax.legend(loc="upper right", fontsize=8, frameon=True)


def _save_coverage_plot(
    df: pd.DataFrame,
    feasible_df: pd.DataFrame | None,
    progress_summary: dict[str, Any],
    out_png: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    _plot_coverage_heatmap(
        axes[0, 0],
        df,
        row_key="t1",
        col_key="t2",
        row_values=_get_axis_values(feasible_df, df, "t1"),
        col_values=_get_axis_values(feasible_df, df, "t2"),
        default_row_val=float(DEFAULT_SCHEDULE["t1"]),
        default_col_val=float(DEFAULT_SCHEDULE["t2"]),
        title="Coverage over stage-duration grid",
    )
    _plot_coverage_heatmap(
        axes[0, 1],
        df,
        row_key="N1",
        col_key="rho2",
        row_values=_get_axis_values(feasible_df, df, "N1"),
        col_values=_get_axis_values(feasible_df, df, "rho2"),
        default_row_val=float(DEFAULT_SCHEDULE["N1"]),
        default_col_val=float(DEFAULT_SCHEDULE["rho2"]),
        title="Coverage over area-density grid",
    )

    cycle_values = _get_axis_values(feasible_df, df, "cycle_days")
    counts_cycle = (
        df.groupby("cycle_days")["schedule_key"].count().reindex(cycle_values, fill_value=0)
        if not df.empty
        else pd.Series(index=cycle_values, data=0)
    )
    axes[1, 0].bar(
        [str(int(v)) if float(v).is_integer() else f"{v:g}" for v in cycle_values],
        counts_cycle.values.astype(float),
        color="#6baed6",
        alpha=0.9,
    )
    axes[1, 0].axvline(
        (
            list(cycle_values).index(float(DEFAULT_SCHEDULE["t1"] + DEFAULT_SCHEDULE["t2"]))
            if float(DEFAULT_SCHEDULE["t1"] + DEFAULT_SCHEDULE["t2"]) in cycle_values
            else -1
        ),
        color="#d62728",
        linestyle="--",
        linewidth=1.2,
    )
    axes[1, 0].set_title("Coverage over total cycle length", fontsize=11, pad=8)
    axes[1, 0].set_xlabel("t1 + t2 [d]")
    axes[1, 0].set_ylabel("Evaluated schedules")
    _style_axes(axes[1, 0])

    axes[1, 1].set_axis_off()
    text = (
        f"Snapshot time: {progress_summary['generated_at']}\n"
        f"Evaluated schedules: {progress_summary['n_evaluated']} / {progress_summary['n_total_feasible']}\n"
        f"Completion ratio: {progress_summary['completion_ratio']:.2%}\n"
        f"Valid full-horizon: {progress_summary['n_valid_full_horizon']}\n"
        f"Termination reasons: {progress_summary['termination_reason_counts']}\n\n"
        f"Default evaluated: {progress_summary['default_schedule_evaluated']}\n"
        f"Default schedule: {progress_summary['default_schedule_key']}\n"
        f"Current best valid: {progress_summary['best_valid_schedule_key']}\n"
        f"Current best profit: {progress_summary['best_valid_net_profit']:.2f}\n"
        f"Current best harvest FW: {progress_summary['best_valid_harvest_fresh_kg']:.2f} kg\n"
    )
    axes[1, 1].text(0.03, 0.97, text, ha="left", va="top", fontsize=11)

    fig.suptitle("Partial exact PID baseline: current coverage and status", fontsize=14, y=0.98)
    fig.subplots_adjust(top=0.92, wspace=0.28, hspace=0.30)
    fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _build_progress_summary(
    df: pd.DataFrame,
    ranked_df: pd.DataFrame,
    *,
    results_csv: Path,
    feasible_df: pd.DataFrame | None,
) -> dict[str, Any]:
    n_total = int(len(feasible_df)) if feasible_df is not None else int(len(df))
    valid_df = df[df["valid_full_horizon"]].copy()
    default_df = ranked_df[ranked_df["is_default_schedule"]]
    default_row = default_df.iloc[0].to_dict() if not default_df.empty else None
    best_valid = (
        valid_df.sort_values(by=["net_profit", "harvest_fresh_kg", "energy_kwh"], ascending=[False, False, True]).iloc[0].to_dict()
        if not valid_df.empty
        else None
    )

    progress = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "results_csv": str(results_csv),
        "n_evaluated": int(len(df)),
        "n_total_feasible": n_total,
        "completion_ratio": float(len(df) / max(n_total, 1)),
        "n_valid_full_horizon": int(df["valid_full_horizon"].sum()),
        "n_invalid_or_early_terminated": int((~df["valid_full_horizon"]).sum()),
        "termination_reason_counts": {
            str(k): int(v) for k, v in df["termination_reason"].value_counts().items()
        },
        "default_schedule_key": (
            f"t1={DEFAULT_SCHEDULE['t1']}|t2={DEFAULT_SCHEDULE['t2']}|"
            f"N1={DEFAULT_SCHEDULE['N1']}|rho2={int(DEFAULT_SCHEDULE['rho2'])}"
        ),
        "default_schedule_evaluated": bool(default_row is not None),
        "default_schedule_current_rank_objective": (
            int(default_row["rank_objective"]) if default_row and pd.notna(default_row.get("rank_objective")) else None
        ),
        "default_schedule_current_rank_valid_profit": (
            int(default_row["rank_valid_profit"]) if default_row and pd.notna(default_row.get("rank_valid_profit")) else None
        ),
        "default_schedule_current_net_profit": (
            float(default_row["net_profit"]) if default_row is not None else None
        ),
        "best_valid_schedule_key": best_valid["schedule_key"] if best_valid else None,
        "best_valid_net_profit": float(best_valid["net_profit"]) if best_valid else float("nan"),
        "best_valid_harvest_fresh_kg": (
            float(best_valid["harvest_fresh_kg"]) if best_valid else float("nan")
        ),
    }
    return progress


def _save_progress_markdown(
    progress_summary: dict[str, Any],
    top_df: pd.DataFrame,
    out_md: Path,
) -> None:
    lines = [
        "# Partial Exact PID Baseline Snapshot",
        "",
        f"- Generated at: `{progress_summary['generated_at']}`",
        f"- Evaluated schedules: `{progress_summary['n_evaluated']}` / `{progress_summary['n_total_feasible']}`",
        f"- Completion ratio: `{progress_summary['completion_ratio']:.2%}`",
        f"- Valid full-horizon: `{progress_summary['n_valid_full_horizon']}`",
        f"- Default schedule evaluated: `{progress_summary['default_schedule_evaluated']}`",
        f"- Default schedule key: `{progress_summary['default_schedule_key']}`",
        "",
        "## Current Best Valid",
        "",
        f"- Schedule: `{progress_summary['best_valid_schedule_key']}`",
        f"- Net profit: `{progress_summary['best_valid_net_profit']:.2f}`",
        f"- Harvest fresh mass: `{progress_summary['best_valid_harvest_fresh_kg']:.2f} kg`",
        "",
        "## Current Top Schedules",
        "",
    ]
    for _, row in top_df.iterrows():
        lines.append(
            f"- `{row['schedule_key']}` | profit=`{float(row['net_profit']):.2f}` | "
            f"harvest_fw=`{float(row['harvest_fresh_kg']):.2f} kg` | "
            f"energy=`{float(row['energy_kwh']):.2f} kWh`"
        )
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    results_csv = Path(args.results_csv).resolve()
    feasible_csv = Path(args.feasible_csv).resolve()
    out_dir = (
        Path(args.out_dir).resolve()
        if args.out_dir is not None
        else results_csv.parent / "partial_analysis_current"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_results_df(results_csv)
    feasible_df = _load_feasible_df(feasible_csv)

    snapshot_csv = out_dir / "pid_exact_schedule_results_partial_snapshot.csv"
    df.to_csv(snapshot_csv, index=False, encoding="utf-8")

    ranked_csv = out_dir / "pid_exact_schedule_results_ranked.partial.csv"
    summary_json = out_dir / "pid_exact_schedule_summary.partial.json"
    summary = summarise_results(snapshot_csv, summary_json, ranked_csv)
    generated_plots = generate_exact_baseline_plots(snapshot_csv, out_dir)

    ranked_df = pd.read_csv(ranked_csv)
    valid_ranked = ranked_df[ranked_df["valid_full_horizon"]].copy()
    valid_ranked = valid_ranked.sort_values(
        by=["net_profit", "harvest_fresh_kg", "energy_kwh"],
        ascending=[False, False, True],
    )
    top_k = max(int(args.top_k), 1)
    top_current_csv = out_dir / "pid_exact_partial_top_current_valid.csv"
    top_current_df = valid_ranked.head(top_k).copy()
    top_current_df.to_csv(top_current_csv, index=False, encoding="utf-8")

    parameter_summary_csv = out_dir / "pid_exact_partial_parameter_summary.csv"
    parameter_summary_df = _save_parameter_summary(df, parameter_summary_csv)

    parameter_trends_png = out_dir / "pid_exact_partial_parameter_trends.png"
    _plot_parameter_trends(parameter_summary_df, parameter_trends_png)

    progress_summary = _build_progress_summary(
        df,
        ranked_df,
        results_csv=results_csv,
        feasible_df=feasible_df,
    )
    progress_summary["best_valid_schedule_key"] = summary.get("best_valid_by_profit", {}).get("schedule_key")
    progress_summary["best_valid_net_profit"] = float(
        summary.get("best_valid_by_profit", {}).get("net_profit", progress_summary["best_valid_net_profit"])
    )
    progress_summary["best_valid_harvest_fresh_kg"] = float(
        summary.get("best_valid_by_profit", {}).get(
            "harvest_fresh_kg",
            progress_summary["best_valid_harvest_fresh_kg"],
        )
    )

    coverage_png = out_dir / "pid_exact_partial_coverage.png"
    _save_coverage_plot(df, feasible_df, progress_summary, coverage_png)

    progress_json = out_dir / "pid_exact_partial_progress.json"
    progress_json.write_text(json.dumps(progress_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    progress_md = out_dir / "pid_exact_partial_progress.md"
    _save_progress_markdown(progress_summary, top_current_df, progress_md)

    print("[PARTIAL] Snapshot CSV  :", snapshot_csv)
    print("[PARTIAL] Ranked CSV    :", ranked_csv)
    print("[PARTIAL] Summary JSON  :", summary_json)
    print("[PARTIAL] Progress JSON :", progress_json)
    print("[PARTIAL] Progress MD   :", progress_md)
    print("[PARTIAL] Top-k CSV     :", top_current_csv)
    print("[PARTIAL] Param CSV     :", parameter_summary_csv)
    print("[PARTIAL] Extra plots   :", parameter_trends_png, "|", coverage_png)
    for path in generated_plots:
        print("[PARTIAL] Plot          :", path)
    print(
        "[PARTIAL] Progress      : "
        f"{progress_summary['n_evaluated']}/{progress_summary['n_total_feasible']} "
        f"({progress_summary['completion_ratio']:.2%})"
    )
    print(
        "[PARTIAL] Best valid    : "
        f"{progress_summary['best_valid_schedule_key']} | "
        f"profit={progress_summary['best_valid_net_profit']:.2f} | "
        f"harvest_fw={progress_summary['best_valid_harvest_fresh_kg']:.2f} kg"
    )
    print(
        "[PARTIAL] Default eval  : "
        f"{progress_summary['default_schedule_evaluated']}"
    )


if __name__ == "__main__":
    main()
