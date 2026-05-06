# -*- coding: utf-8 -*-
"""Build compact, manuscript-ready figures for the exact PID result review."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_style_academic import (
    COLORS,
    add_panel_label,
    apply_academic_style,
    apply_heatmap_frame,
    save_figure,
    style_axes,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = Path(r"C:\Users\29341\Desktop\fsdownload\results")
DEFAULT_OUT_DIR = ROOT / "paper" / "figures_result_review_20260426"


SCENARIO_SPECS = [
    {
        "name": "exp02_exact_pid_baseline_pp16h_i1250_i2300_pm_constant_e0p74_co20p54_l20",
        "label": "Const-L20",
    },
    {
        "name": "exp02_exact_pid_baseline_pp16h_i1250_i2300_pm_constant_e0p74_co20p54_l40",
        "label": "Const-L40",
    },
    {
        "name": "exp02_exact_pid_baseline_pp16h_i1250_i2300_pm_tou_zhejiang_lt1kv_l20",
        "label": "TOU-L20",
    },
    {
        "name": "exp02_exact_pid_baseline_pp16h_i1250_i2300_pm_tou_zhejiang_lt1kv_l40",
        "label": "TOU-L40",
    },
]

UPPER_VARS = ["t1", "t2", "N1", "rho2"]
METRICS = ["net_profit", "harvest_fresh_kg", "energy_kwh", "cost_per_kg"]
METRIC_LABELS = {
    "net_profit": "Net profit",
    "harvest_fresh_kg": "Annual fresh yield",
    "energy_kwh": "Annual electricity use",
    "cost_per_kg": "Fresh-mass cost",
}
VAR_LABELS = {
    "t1": r"Dense-stage duration, $t_1$",
    "t2": r"Finishing-stage duration, $t_2$",
    "N1": r"Dense-zone boards, $N_1$",
    "rho2": r"Finishing density, $\rho_2$",
}
VAR_AXIS_LABELS = {
    "t1": r"$t_1$ (d)",
    "t2": r"$t_2$ (d)",
    "N1": r"$N_1$ (-)",
    "rho2": r"$\rho_2$ (-)",
}
METRIC_COLORS = {
    "profit_gain_pct": COLORS["navy"],
    "harvest_gain_pct": COLORS["teal"],
    "energy_reduction_pct": COLORS["gold"],
    "cost_reduction_pct": COLORS["plum"],
}


def _load_scenario_df(exp_dir: Path) -> pd.DataFrame:
    merged_csv = exp_dir / "pid_exact_schedule_results.csv"
    if merged_csv.exists():
        df = pd.read_csv(merged_csv)
    else:
        shard_paths = sorted(exp_dir.glob("pid_exact_schedule_results.shard_*.csv"))
        if not shard_paths:
            raise FileNotFoundError(f"No schedule result csv found in {exp_dir}")
        df = pd.concat([pd.read_csv(p) for p in shard_paths], ignore_index=True)
    df["is_default_schedule"] = df["is_default_schedule"].astype(str).str.lower().eq("true")
    return df


def _compute_eta2(df: pd.DataFrame, variable: str, metric: str) -> float:
    y = df[metric].astype(float).to_numpy()
    y_bar = float(np.mean(y))
    ss_total = float(np.sum((y - y_bar) ** 2))
    if ss_total <= 0.0:
        return 0.0
    grouped = df.groupby(variable)[metric].agg(["mean", "count"]).reset_index()
    ss_between = float(np.sum(grouped["count"] * (grouped["mean"] - y_bar) ** 2))
    return ss_between / ss_total


def _collect_sensitivity_tables(data_map: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    marginal_rows = []
    for scenario_label, df in data_map.items():
        for variable in UPPER_VARS:
            grouped = df.groupby(variable).agg(
                net_profit=("net_profit", "mean"),
                harvest_fresh_kg=("harvest_fresh_kg", "mean"),
                energy_kwh=("energy_kwh", "mean"),
                cost_per_kg=("cost_per_kg", "mean"),
            ).reset_index()
            for metric in METRICS:
                rows.append(
                    {
                        "scenario": scenario_label,
                        "variable": variable,
                        "metric": metric,
                        "eta2": _compute_eta2(df, variable, metric),
                    }
                )
            if scenario_label == "Const-L40":
                for _, row in grouped.iterrows():
                    marginal_rows.append(
                        {
                            "variable": variable,
                            "level": row[variable],
                            "net_profit": row["net_profit"],
                            "energy_kwh": row["energy_kwh"],
                            "cost_per_kg": row["cost_per_kg"],
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(marginal_rows)


def _scenario_summary_rows(data_map: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    delta_rows = []
    for scenario_label, df in data_map.items():
        best = df.sort_values("net_profit", ascending=False).iloc[0]
        default = df[df["is_default_schedule"]].iloc[0]
        worst = df.sort_values("net_profit", ascending=True).iloc[0]
        for kind, row in [("best", best), ("default", default), ("worst", worst)]:
            summary_rows.append(
                {
                    "scenario": scenario_label,
                    "kind": kind,
                    "schedule_key": row["schedule_key"],
                    "net_profit": float(row["net_profit"]),
                }
            )
        delta_rows.append(
            {
                "scenario": scenario_label,
                "profit_gain_pct": 100.0 * (best["net_profit"] - default["net_profit"]) / default["net_profit"],
                "harvest_gain_pct": 100.0 * (best["harvest_fresh_kg"] - default["harvest_fresh_kg"]) / default["harvest_fresh_kg"],
                "energy_reduction_pct": 100.0 * (default["energy_kwh"] - best["energy_kwh"]) / default["energy_kwh"],
                "cost_reduction_pct": 100.0 * (default["cost_per_kg"] - best["cost_per_kg"]) / default["cost_per_kg"],
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(delta_rows)


def _build_core_summary_figure(
    out_path: Path,
    data_map: dict[str, pd.DataFrame],
    summary_df: pd.DataFrame,
    delta_df: pd.DataFrame,
) -> None:
    fig = plt.figure(figsize=(12.0, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.32)

    scenario_labels = [spec["label"] for spec in SCENARIO_SPECS]

    ax0 = fig.add_subplot(gs[0, 0])
    box_data = [data_map[label]["net_profit"].to_numpy() / 1.0e4 for label in scenario_labels]
    bp = ax0.boxplot(
        box_data,
        tick_labels=scenario_labels,
        patch_artist=True,
        widths=0.52,
        showfliers=False,
    )
    for patch in bp["boxes"]:
        patch.set(facecolor="white", edgecolor=COLORS["gray"], linewidth=1.0)
    for whisker in bp["whiskers"]:
        whisker.set(color=COLORS["gray"], linewidth=0.9)
    for cap in bp["caps"]:
        cap.set(color=COLORS["gray"], linewidth=0.9)
    for median in bp["medians"]:
        median.set(color=COLORS["ink"], linewidth=1.3)
    for x, scenario in enumerate(scenario_labels, start=1):
        s = summary_df[summary_df["scenario"] == scenario]
        best = float(s.loc[s["kind"] == "best", "net_profit"].iloc[0]) / 1.0e4
        default = float(s.loc[s["kind"] == "default", "net_profit"].iloc[0]) / 1.0e4
        worst = float(s.loc[s["kind"] == "worst", "net_profit"].iloc[0]) / 1.0e4
        ax0.plot([x, x], [default, best], color=COLORS["light_gray"], linewidth=1.2, zorder=2)
        ax0.scatter([x], [best], marker="D", s=36, color=COLORS["navy"], zorder=3, label="Best" if x == 1 else None)
        ax0.scatter([x], [default], marker="s", s=32, facecolor="white", edgecolor=COLORS["brick"], linewidth=1.0, zorder=3, label="Default" if x == 1 else None)
        ax0.scatter([x], [worst], marker="o", s=20, color=COLORS["gray"], zorder=3, label="Worst" if x == 1 else None)
    ax0.set_xlabel("Scenario")
    ax0.set_ylabel(r"Annual net profit ($10^4$ CNY)")
    ax0.legend(frameon=False, loc="upper left", ncol=1, handletextpad=0.5, borderaxespad=0.2)
    style_axes(ax0, grid_axis="y")
    add_panel_label(ax0, "a")

    ax1 = fig.add_subplot(gs[0, 1])
    metric_specs = [
        ("profit_gain_pct", "Net profit"),
        ("harvest_gain_pct", "Fresh yield"),
        ("energy_reduction_pct", "Electricity reduction"),
        ("cost_reduction_pct", "Cost reduction"),
    ]
    x = np.arange(len(scenario_labels))
    width = 0.17
    for idx, (col, label) in enumerate(metric_specs):
        values = delta_df.set_index("scenario").loc[scenario_labels, col].to_numpy()
        ax1.bar(
            x + (idx - 1.5) * width,
            values,
            width=width,
            color=METRIC_COLORS[col],
            label=label,
            edgecolor="none",
        )
    ax1.axhline(0.0, color=COLORS["gray"], linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenario_labels)
    ax1.set_xlabel("Scenario")
    ax1.set_ylabel("Relative change vs. default (%)")
    ax1.legend(frameon=False, ncol=2, loc="upper left", handletextpad=0.5, columnspacing=1.2, borderaxespad=0.2)
    style_axes(ax1, grid_axis="y")
    add_panel_label(ax1, "b")

    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def _build_sensitivity_heatmap(out_path: Path, eta_df: pd.DataFrame) -> pd.DataFrame:
    eta_avg = (
        eta_df.groupby(["variable", "metric"], as_index=False)["eta2"]
        .mean()
        .pivot(index="variable", columns="metric", values="eta2")
        .loc[UPPER_VARS, METRICS]
    )
    eta_avg = eta_avg.loc[eta_avg["net_profit"].sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    im = ax.imshow(eta_avg.to_numpy(), cmap="cividis", aspect="auto")
    ax.set_xticks(np.arange(len(METRICS)))
    ax.set_xticklabels(
        ["Net profit", "Fresh yield", "Electricity use", "Cost per kg"],
        rotation=0,
    )
    ax.set_yticks(np.arange(len(eta_avg.index)))
    ax.set_yticklabels([VAR_LABELS[v] for v in eta_avg.index])
    ax.set_xlabel("Response metric")
    ax.set_ylabel("Upper-level variable")
    apply_heatmap_frame(ax, nrows=len(eta_avg.index), ncols=len(METRICS))
    for i, variable in enumerate(eta_avg.index):
        for j, metric in enumerate(eta_avg.columns):
            value = float(eta_avg.loc[variable, metric])
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > 0.45 else COLORS["ink"],
                fontsize=8.1,
                fontweight="bold" if value >= 0.5 else None,
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label(r"Mean main-effect strength, $\eta^2$ (-)")
    add_panel_label(ax, "a")
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)
    return eta_avg.reset_index().rename(columns={"index": "variable"})


def _scaled_goodness(series: pd.Series, higher_is_better: bool) -> pd.Series:
    values = series.astype(float).to_numpy()
    low = float(np.min(values))
    high = float(np.max(values))
    if np.isclose(high, low):
        scaled = np.zeros_like(values)
    else:
        scaled = (values - low) / (high - low) if higher_is_better else (high - values) / (high - low)
    return pd.Series(scaled, index=series.index)


def _build_marginal_response_figure(out_path: Path, marginal_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.8), sharey=True)
    axes = axes.flatten()
    order = ["rho2", "N1", "t2", "t1"]
    line_specs = [
        ("net_profit", True, COLORS["navy"], "Net profit desirability"),
        ("energy_kwh", False, COLORS["teal"], "Electricity desirability"),
        ("cost_per_kg", False, COLORS["gold"], "Cost desirability"),
    ]

    for idx, (ax, variable) in enumerate(zip(axes, order)):
        sub = marginal_df[marginal_df["variable"] == variable].sort_values("level").reset_index(drop=True)
        x = sub["level"].to_numpy()
        for metric, higher_is_better, color, label in line_specs:
            desirability = _scaled_goodness(sub[metric], higher_is_better=higher_is_better)
            ax.plot(
                x,
                desirability,
                marker="o",
                markersize=3.6,
                linewidth=1.8,
                color=color,
                label=label,
            )
        ax.set_xlabel(VAR_AXIS_LABELS[variable])
        ax.set_ylim(-0.03, 1.03)
        ax.set_yticks([0.0, 0.5, 1.0])
        style_axes(ax, grid_axis="y")
        add_panel_label(ax, chr(ord("a") + idx))
    axes[0].set_ylabel("Standardized desirability (-)")
    axes[2].set_ylabel("Standardized desirability (-)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, columnspacing=1.2, handletextpad=0.5)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save_figure(fig, out_path)
    plt.close(fig)


def main() -> None:
    apply_academic_style()
    results_root = DEFAULT_RESULTS_ROOT
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    data_map: dict[str, pd.DataFrame] = {}
    for spec in SCENARIO_SPECS:
        exp_dir = results_root / spec["name"]
        data_map[spec["label"]] = _load_scenario_df(exp_dir)

    summary_df, delta_df = _scenario_summary_rows(data_map)
    eta_df, marginal_df = _collect_sensitivity_tables(data_map)

    _build_core_summary_figure(
        out_dir / "compact_core_summary.png",
        data_map=data_map,
        summary_df=summary_df,
        delta_df=delta_df,
    )
    eta_avg_df = _build_sensitivity_heatmap(
        out_dir / "compact_upper_sensitivity_heatmap.png",
        eta_df=eta_df,
    )
    _build_marginal_response_figure(
        out_dir / "compact_upper_marginal_responses.png",
        marginal_df=marginal_df,
    )

    summary_payload = {
        "scenario_summary": summary_df.to_dict(orient="records"),
        "best_vs_default_delta": delta_df.to_dict(orient="records"),
        "eta2_by_scenario": eta_df.to_dict(orient="records"),
        "eta2_average": eta_avg_df.to_dict(orient="records"),
    }
    (out_dir / "compact_result_review_summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_df.to_csv(out_dir / "compact_result_review_summary.csv", index=False, encoding="utf-8-sig")
    delta_df.to_csv(out_dir / "compact_result_review_deltas.csv", index=False, encoding="utf-8-sig")
    eta_df.to_csv(out_dir / "compact_upper_sensitivity_eta2.csv", index=False, encoding="utf-8-sig")

    print(f"[OK] Saved compact result-review figures to: {out_dir}")


if __name__ == "__main__":
    main()
