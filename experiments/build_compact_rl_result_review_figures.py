# -*- coding: utf-8 -*-
"""Build compact, high-information figures for the exact RL result review."""

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
    set_hour_ticks,
    style_axes,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RL_RESULTS_ROOT = Path(r"C:\Users\29341\Desktop\fsdownload\results_residual_pid_sac")
DEFAULT_PID_RESULTS_ROOT = Path(r"C:\Users\29341\Desktop\fsdownload\results")
DEFAULT_OUT_DIR = ROOT / "paper" / "figures_result_review_rl_20260427_supplement"
LEGACY_OUT_DIR = ROOT / "paper" / "figures_result_review_rl_20260427"


RL_SCENARIO_SPECS = [
    {
        "name": "exp03_exact_rl_baseline_pp16h_constant_e0p74_co20p54_l20",
        "label": "Seg-Const-L20",
    },
    {
        "name": "exp03_exact_rl_baseline_pp16h_constant_e0p74_co20p54_l20_daily",
        "label": "Daily-Const-L20",
    },
    {
        "name": "exp03_exact_rl_baseline_pp16h_constant_e0p74_co20p54_l40",
        "label": "Seg-Const-L40",
    },
    {
        "name": "exp03_exact_rl_baseline_pp16h_tou_zhejiang_lt1kv_l20",
        "label": "Seg-TOU-L20",
    },
    {
        "name": "exp03_exact_rl_baseline_pp16h_tou_zhejiang_lt1kv_l40",
        "label": "Seg-TOU-L40",
    },
]

MATCHED_CONTROLLER_SPECS = [
    {
        "label": "Const-L20",
        "pid_name": "exp02_exact_pid_baseline_pp16h_i1250_i2300_pm_constant_e0p74_co20p54_l20",
        "rl_name": "exp03_exact_rl_baseline_pp16h_constant_e0p74_co20p54_l20",
    },
    {
        "label": "Const-L40",
        "pid_name": "exp02_exact_pid_baseline_pp16h_i1250_i2300_pm_constant_e0p74_co20p54_l40",
        "rl_name": "exp03_exact_rl_baseline_pp16h_constant_e0p74_co20p54_l40",
    },
    {
        "label": "TOU-L20",
        "pid_name": "exp02_exact_pid_baseline_pp16h_i1250_i2300_pm_tou_zhejiang_lt1kv_l20",
        "rl_name": "exp03_exact_rl_baseline_pp16h_tou_zhejiang_lt1kv_l20",
    },
    {
        "label": "TOU-L40",
        "pid_name": "exp02_exact_pid_baseline_pp16h_i1250_i2300_pm_tou_zhejiang_lt1kv_l40",
        "rl_name": "exp03_exact_rl_baseline_pp16h_tou_zhejiang_lt1kv_l40",
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
ATTR_METRIC_LABELS = {
    "net_profit": "Net profit",
    "harvest_fresh_kg": "Fresh yield",
    "energy_kwh": "Total energy change",
}
VAR_LABELS = {
    "t1": r"Dense-stage duration, $t_1$",
    "t2": r"Finishing-stage duration, $t_2$",
    "N1": r"Dense-zone boards, $N_1$",
    "rho2": r"Finishing density, $\rho_2$",
}
DEFAULT_SCHEDULE_KEY = "t1=14|t2=14|N1=20|rho2=36"
SHARED_TOU_MECH_SCHEDULE_KEY = "t1=15|t2=13|N1=15|rho2=23"
DELTA_COLORS = {
    "profit_gain_pct": COLORS["navy"],
    "harvest_gain_pct": COLORS["teal"],
    "energy_change_pct": COLORS["gold"],
    "cost_reduction_pct": COLORS["plum"],
}
LAYER_COLORS = {
    "upper": COLORS["navy"],
    "lower": COLORS["green"],
}
MECHANISM_COLORS = {
    "dense": COLORS["blue"],
    "finishing": COLORS["brick"],
    "tou": COLORS["plum"],
    "price": COLORS["gray"],
}


def _schedule_key_to_trace_name(schedule_key: str) -> str:
    parts = dict(part.split("=") for part in schedule_key.split("|"))
    return (
        f"t1-{int(float(parts['t1']))}"
        f"__t2-{int(float(parts['t2']))}"
        f"__N1-{int(float(parts['N1']))}"
        f"__rho2-{int(float(parts['rho2']))}.csv"
    )


def _load_trace_df(exp_dir: Path, schedule_key: str, usecols: list[str] | None = None) -> pd.DataFrame:
    trace_path = exp_dir / "detailed_traces" / _schedule_key_to_trace_name(schedule_key)
    if not trace_path.exists():
        raise FileNotFoundError(f"Trace not found: {trace_path}")
    return pd.read_csv(trace_path, usecols=usecols)


def _hourly_profile(trace_df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    df = trace_df.copy()
    dt = pd.to_datetime(df["datetime"])
    df["hour"] = dt.dt.hour + dt.dt.minute / 60.0
    return df.groupby("hour", as_index=False)[value_cols].mean().sort_values("hour")


def _device_energy_breakdown(exp_dir: Path, schedule_key: str) -> pd.Series:
    trace_df = _load_trace_df(
        exp_dir,
        schedule_key,
        usecols=["E_step_kWh", "P_LED_total_kW", "P_HVAC_kW", "P_dehum_kW", "P_CO2_kW"],
    )
    total_power = (
        trace_df["P_LED_total_kW"]
        + trace_df["P_HVAC_kW"]
        + trace_df["P_dehum_kW"]
        + trace_df["P_CO2_kW"]
    ).replace(0, np.nan)
    shares = pd.DataFrame(
        {
            "led": trace_df["P_LED_total_kW"] / total_power,
            "hvac": trace_df["P_HVAC_kW"] / total_power,
            "dehum": trace_df["P_dehum_kW"] / total_power,
            "co2": trace_df["P_CO2_kW"] / total_power,
        }
    ).fillna(0.0)
    energies = shares.mul(trace_df["E_step_kWh"], axis=0).sum()
    energies["total"] = float(trace_df["E_step_kWh"].sum())
    return energies


def _style_secondary_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_color(COLORS["ink"])
    ax.spines["right"].set_linewidth(0.8)
    ax.tick_params(axis="y", which="major", colors=COLORS["ink"], pad=2.5)
    ax.tick_params(axis="y", which="minor", colors=COLORS["gray"])
    ax.grid(False)


def _load_schedule_df(exp_dir: Path, prefix: str) -> pd.DataFrame:
    merged_csv = exp_dir / f"{prefix}_schedule_results.csv"
    if merged_csv.exists():
        return pd.read_csv(merged_csv)
    shard_paths = sorted(exp_dir.glob(f"{prefix}_schedule_results.shard_*.csv"))
    if not shard_paths:
        raise FileNotFoundError(f"No schedule result csv found in {exp_dir}")
    return pd.concat([pd.read_csv(p) for p in shard_paths], ignore_index=True)


def _compute_eta2(df: pd.DataFrame, variable: str, metric: str) -> float:
    y = df[metric].astype(float).to_numpy()
    y_bar = float(np.mean(y))
    ss_total = float(np.sum((y - y_bar) ** 2))
    if ss_total <= 0.0:
        return 0.0
    grouped = df.groupby(variable)[metric].agg(["mean", "count"]).reset_index()
    ss_between = float(np.sum(grouped["count"] * (grouped["mean"] - y_bar) ** 2))
    return ss_between / ss_total


def _collect_sensitivity_tables(data_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for scenario_label, df in data_map.items():
        for variable in UPPER_VARS:
            for metric in METRICS:
                rows.append(
                    {
                        "scenario": scenario_label,
                        "variable": variable,
                        "metric": metric,
                        "eta2": _compute_eta2(df, variable, metric),
                    }
                )
    return pd.DataFrame(rows)


def _scenario_summary_rows(data_map: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    delta_rows = []
    for scenario_label, df in data_map.items():
        best = df.sort_values("net_profit", ascending=False).iloc[0]
        default = df[df["is_default_schedule"] == True].iloc[0]
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
                "energy_change_pct": 100.0 * (best["energy_kwh"] - default["energy_kwh"]) / default["energy_kwh"],
                "cost_reduction_pct": 100.0 * (default["cost_per_kg"] - best["cost_per_kg"]) / default["cost_per_kg"],
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(delta_rows)


def _controller_comparison_rows(
    pid_map: dict[str, pd.DataFrame],
    rl_map: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for spec in MATCHED_CONTROLLER_SPECS:
        label = spec["label"]
        pid_df = pid_map[label]
        rl_df = rl_map[label]

        pid_best = pid_df.sort_values("net_profit", ascending=False).iloc[0]
        rl_best = rl_df.sort_values("net_profit", ascending=False).iloc[0]
        pid_default = pid_df[pid_df["is_default_schedule"] == True].iloc[0]
        rl_default = rl_df[rl_df["is_default_schedule"] == True].iloc[0]

        for scope, pid_row, rl_row in [
            ("default_schedule", pid_default, rl_default),
            ("best_schedule", pid_best, rl_best),
        ]:
            rows.append(
                {
                    "scenario": label,
                    "scope": scope,
                    "profit_gain_pct": 100.0 * (rl_row["net_profit"] - pid_row["net_profit"]) / pid_row["net_profit"],
                    "harvest_gain_pct": 100.0 * (rl_row["harvest_fresh_kg"] - pid_row["harvest_fresh_kg"]) / pid_row["harvest_fresh_kg"],
                    "energy_change_pct": 100.0 * (rl_row["energy_kwh"] - pid_row["energy_kwh"]) / pid_row["energy_kwh"],
                    "cost_reduction_pct": 100.0 * (pid_row["cost_per_kg"] - rl_row["cost_per_kg"]) / pid_row["cost_per_kg"],
                    "pid_schedule_key": pid_row["schedule_key"],
                    "rl_schedule_key": rl_row["schedule_key"],
                }
            )
    return pd.DataFrame(rows)


def _build_rl_core_summary_figure(
    out_path: Path,
    data_map: dict[str, pd.DataFrame],
    summary_df: pd.DataFrame,
    delta_df: pd.DataFrame,
) -> None:
    fig = plt.figure(figsize=(12.2, 4.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.32)

    scenario_labels = [spec["label"] for spec in RL_SCENARIO_SPECS]

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
    ax0.legend(frameon=False, loc="upper left", handletextpad=0.5, borderaxespad=0.2)
    style_axes(ax0, grid_axis="y")
    add_panel_label(ax0, "a")

    ax1 = fig.add_subplot(gs[0, 1])
    metric_specs = [
        ("profit_gain_pct", "Net profit"),
        ("harvest_gain_pct", "Fresh yield"),
        ("energy_change_pct", "Electricity change"),
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
            color=DELTA_COLORS[col],
            label=label,
            edgecolor="none",
        )
    ax1.axhline(0.0, color=COLORS["gray"], linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenario_labels)
    ax1.set_xlabel("Scenario")
    ax1.set_ylabel("Relative change vs. RL default (%)")
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

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
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


def _build_controller_comparison_figure(out_path: Path, compare_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9), sharey=True)
    scopes = [
        ("default_schedule", "Scenario\n(fixed default upper schedule)"),
        ("best_schedule", "Scenario\n(controller-specific optimum)"),
    ]
    metrics = [
        ("profit_gain_pct", "Net profit"),
        ("harvest_gain_pct", "Fresh yield"),
        ("energy_change_pct", "Electricity change"),
        ("cost_reduction_pct", "Cost reduction"),
    ]
    all_values = compare_df[[col for col, _ in metrics]].to_numpy().ravel()
    y_min = min(float(np.min(all_values)), 0.0)
    y_max = max(float(np.max(all_values)), 0.0)
    y_pad = 0.08 * max(y_max - y_min, 5.0)

    for panel_idx, (ax, (scope, xlabel)) in enumerate(zip(axes, scopes)):
        sub = compare_df[compare_df["scope"] == scope].set_index("scenario")
        labels = [spec["label"] for spec in MATCHED_CONTROLLER_SPECS]
        x = np.arange(len(labels))
        width = 0.18
        for metric_idx, (col, metric_label) in enumerate(metrics):
            values = sub.loc[labels, col].to_numpy()
            ax.bar(
                x + (metric_idx - 1.5) * width,
                values,
                width=width,
                color=DELTA_COLORS[col],
                label=metric_label,
                edgecolor="none",
            )
        ax.axhline(0.0, color=COLORS["gray"], linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel(xlabel)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        style_axes(ax, grid_axis="y")
        add_panel_label(ax, chr(ord("a") + panel_idx))

    axes[0].set_ylabel("Relative change vs. matched PID baseline (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save_figure(fig, out_path)
    plt.close(fig)


def _gain_decomposition_rows(
    pid_map: dict[str, pd.DataFrame],
    rl_map: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for spec in MATCHED_CONTROLLER_SPECS:
        label = spec["label"]
        pid_df = pid_map[label]
        rl_df = rl_map[label]

        pid_best = pid_df.sort_values("net_profit", ascending=False).iloc[0]
        pid_default = pid_df[pid_df["is_default_schedule"] == True].iloc[0]
        rl_best = rl_df.sort_values("net_profit", ascending=False).iloc[0]
        rl_default = rl_df[rl_df["is_default_schedule"] == True].iloc[0]
        rl_on_pid_best = rl_df[rl_df["schedule_key"] == pid_best["schedule_key"]].iloc[0]

        base_profit = float(pid_default["net_profit"])
        rows.append(
            {
                "scenario": label,
                "upper_only_pid_pct": 100.0 * (pid_best["net_profit"] - pid_default["net_profit"]) / base_profit,
                "lower_fixed_default_pct": 100.0 * (rl_default["net_profit"] - pid_default["net_profit"]) / base_profit,
                "upper_reopt_after_rl_pct": 100.0 * (rl_best["net_profit"] - rl_default["net_profit"]) / base_profit,
                "total_joint_pct": 100.0 * (rl_best["net_profit"] - pid_default["net_profit"]) / base_profit,
                "controller_gain_on_pidbest_pct": 100.0 * (rl_on_pid_best["net_profit"] - pid_best["net_profit"]) / base_profit,
                "migration_gain_pct": 100.0 * (rl_best["net_profit"] - rl_on_pid_best["net_profit"]) / base_profit,
                "frontier_shift_total_pct": 100.0 * (rl_best["net_profit"] - pid_best["net_profit"]) / base_profit,
                "pid_best_schedule_key": pid_best["schedule_key"],
                "rl_best_schedule_key": rl_best["schedule_key"],
                "rl_on_pid_best_schedule_key": rl_on_pid_best["schedule_key"],
            }
        )
    return pd.DataFrame(rows)


def _build_gain_decomposition_figure(out_path: Path, gain_df: pd.DataFrame) -> None:
    labels = [spec["label"] for spec in MATCHED_CONTROLLER_SPECS]
    sub = gain_df.set_index("scenario").loc[labels]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0), sharey=False)

    ax0 = axes[0]
    lower_gain = sub["lower_fixed_default_pct"].to_numpy()
    reopt_gain = sub["upper_reopt_after_rl_pct"].to_numpy()
    pid_upper = sub["upper_only_pid_pct"].to_numpy()
    total_joint = sub["total_joint_pct"].to_numpy()

    ax0.bar(
        x,
        lower_gain,
        width=0.56,
        color=LAYER_COLORS["lower"],
        label="Lower-level gain at the default schedule",
        edgecolor="none",
    )
    ax0.bar(
        x,
        reopt_gain,
        width=0.56,
        bottom=lower_gain,
        color=LAYER_COLORS["upper"],
        label="Additional upper-level gain after RL re-optimization",
        edgecolor="none",
    )
    ax0.plot(
        x,
        pid_upper,
        color=COLORS["gray"],
        marker="s",
        linewidth=1.3,
        markersize=4.4,
        label="Upper-only gain under PID",
    )
    ax0.plot(
        x,
        total_joint,
        color=COLORS["ink"],
        marker="D",
        linewidth=1.5,
        markersize=4.6,
        label="Total joint gain",
    )
    for xi, value in zip(x, total_joint):
        ax0.text(xi, value + 0.55, f"{value:.1f}", ha="center", va="bottom", fontsize=7.8, color=COLORS["ink"])
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels)
    ax0.set_xlabel("Scenario")
    ax0.set_ylabel("Net-profit gain relative to PID default (%)")
    style_axes(ax0, grid_axis="y")
    add_panel_label(ax0, "a")
    ax0.legend(frameon=False, fontsize=7.8, loc="upper left", handletextpad=0.5)

    ax1 = axes[1]
    w = 0.28
    ctrl_gain = sub["controller_gain_on_pidbest_pct"].to_numpy()
    mig_gain = sub["migration_gain_pct"].to_numpy()
    frontier_total = sub["frontier_shift_total_pct"].to_numpy()
    bars1 = ax1.bar(
        x - w / 2.0,
        ctrl_gain,
        width=w,
        color=COLORS["plum"],
        label="Controller gain at the PID-best structure",
        edgecolor="none",
    )
    bars2 = ax1.bar(
        x + w / 2.0,
        mig_gain,
        width=w,
        color=COLORS["gold"],
        label="Additional gain from structure migration",
        edgecolor="none",
    )
    ax1.plot(
        x,
        frontier_total,
        color=COLORS["ink"],
        marker="o",
        linewidth=1.5,
        markersize=4.6,
        label="Total RL frontier shift",
    )
    for xi, value in zip(x, frontier_total):
        ax1.text(xi, value + 0.55, f"{value:.1f}", ha="center", va="bottom", fontsize=7.8, color=COLORS["ink"])
    ax1.axhline(0.0, color=COLORS["gray"], linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel("Scenario")
    ax1.set_ylabel("Net-profit gain relative to PID default (%)")
    style_axes(ax1, grid_axis="y")
    add_panel_label(ax1, "b")
    ax1.legend(frameon=False, fontsize=7.8, loc="upper left", handletextpad=0.5)

    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def _collect_mechanism_data(
    pid_map: dict[str, pd.DataFrame],
    rl_map: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pid_default_row = pid_map["Const-L20"][pid_map["Const-L20"]["is_default_schedule"] == True].iloc[0]
    rl_default_row = rl_map["Const-L20"][rl_map["Const-L20"]["is_default_schedule"] == True].iloc[0]

    pid_default_trace = _load_trace_df(
        DEFAULT_PID_RESULTS_ROOT / "exp02_exact_pid_baseline_pp16h_i1250_i2300_pm_constant_e0p74_co20p54_l20",
        DEFAULT_SCHEDULE_KEY,
        usecols=["datetime", "I1", "I2"],
    )
    rl_default_trace = _load_trace_df(
        DEFAULT_RL_RESULTS_ROOT / "exp03_exact_rl_baseline_pp16h_constant_e0p74_co20p54_l20",
        DEFAULT_SCHEDULE_KEY,
        usecols=["datetime", "I1", "I2"],
    )

    pid_profile = _hourly_profile(pid_default_trace, ["I1", "I2"]).rename(
        columns={"I1": "I1_pid", "I2": "I2_pid"}
    )
    rl_profile = _hourly_profile(rl_default_trace, ["I1", "I2"]).rename(
        columns={"I1": "I1_rl", "I2": "I2_rl"}
    )
    default_hourly = pid_profile.merge(rl_profile, on="hour", how="inner")

    default_relative = pd.DataFrame(
        [
            {
                "metric": "Dense-zone mean I1",
                "change_pct": 100.0 * (rl_default_trace["I1"].mean() - pid_default_trace["I1"].mean()) / pid_default_trace["I1"].mean(),
            },
            {
                "metric": "Finishing-zone mean I2",
                "change_pct": 100.0 * (rl_default_trace["I2"].mean() - pid_default_trace["I2"].mean()) / pid_default_trace["I2"].mean(),
            },
            {
                "metric": "Annual energy",
                "change_pct": 100.0 * (rl_default_row["energy_kwh"] - pid_default_row["energy_kwh"]) / pid_default_row["energy_kwh"],
            },
            {
                "metric": "Fresh g per plant",
                "change_pct": 100.0
                * (rl_default_row["avg_harvest_fresh_g_per_plant"] - pid_default_row["avg_harvest_fresh_g_per_plant"])
                / pid_default_row["avg_harvest_fresh_g_per_plant"],
            },
            {
                "metric": "Net profit",
                "change_pct": 100.0 * (rl_default_row["net_profit"] - pid_default_row["net_profit"]) / pid_default_row["net_profit"],
            },
        ]
    )

    const_tou_cols = ["datetime", "I1", "elec_price_rmb_kwh"]
    rl_const_trace = _load_trace_df(
        DEFAULT_RL_RESULTS_ROOT / "exp03_exact_rl_baseline_pp16h_constant_e0p74_co20p54_l40",
        SHARED_TOU_MECH_SCHEDULE_KEY,
        usecols=const_tou_cols,
    )
    rl_tou_trace = _load_trace_df(
        DEFAULT_RL_RESULTS_ROOT / "exp03_exact_rl_baseline_pp16h_tou_zhejiang_lt1kv_l40",
        SHARED_TOU_MECH_SCHEDULE_KEY,
        usecols=const_tou_cols,
    )
    rl_const_profile = _hourly_profile(rl_const_trace, ["I1", "elec_price_rmb_kwh"]).rename(
        columns={"I1": "I1_const", "elec_price_rmb_kwh": "price_const"}
    )
    rl_tou_profile = _hourly_profile(rl_tou_trace, ["I1", "elec_price_rmb_kwh"]).rename(
        columns={"I1": "I1_tou", "elec_price_rmb_kwh": "price_tou"}
    )
    tou_hourly = rl_const_profile.merge(rl_tou_profile, on="hour", how="inner")
    return default_hourly, default_relative, tou_hourly


def _build_mechanism_figure(
    out_path: Path,
    default_hourly: pd.DataFrame,
    default_relative: pd.DataFrame,
    tou_hourly: pd.DataFrame,
) -> None:
    fig = plt.figure(figsize=(13.6, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.28, 0.92, 1.18], wspace=0.34)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.axvspan(8.0, 24.0, color=COLORS["sand"], alpha=0.55, zorder=0)
    ax0.plot(
        default_hourly["hour"],
        default_hourly["I1_pid"],
        color=MECHANISM_COLORS["dense"],
        linestyle="--",
        linewidth=1.5,
        label=r"PID, $I_1$",
    )
    ax0.plot(
        default_hourly["hour"],
        default_hourly["I1_rl"],
        color=MECHANISM_COLORS["dense"],
        linewidth=1.8,
        label=r"RL, $I_1$",
    )
    ax0.plot(
        default_hourly["hour"],
        default_hourly["I2_pid"],
        color=MECHANISM_COLORS["finishing"],
        linestyle="--",
        linewidth=1.5,
        label=r"PID, $I_2$",
    )
    ax0.plot(
        default_hourly["hour"],
        default_hourly["I2_rl"],
        color=MECHANISM_COLORS["finishing"],
        linewidth=1.8,
        label=r"RL, $I_2$",
    )
    ax0.set_ylim(0.0, 320.0)
    ax0.set_xlabel("Time of day (h)")
    ax0.set_ylabel(r"Zone-specific PPFD ($\mu$mol m$^{-2}$ s$^{-1}$)")
    set_hour_ticks(ax0)
    style_axes(ax0, grid_axis="y")
    add_panel_label(ax0, "a")
    ax0.legend(frameon=False, ncol=2, loc="upper left", handletextpad=0.5, columnspacing=1.0)

    ax1 = fig.add_subplot(gs[0, 1])
    bar_values = default_relative["change_pct"].to_numpy()
    x = np.arange(len(bar_values))
    bar_labels = [
        r"Dense-zone" + "\n" + r"$I_1$",
        r"Finishing-zone" + "\n" + r"$I_2$",
        "Annual\n electricity",
        "Fresh mass\n per plant",
        "Net\n profit",
    ]
    bar_colors = [COLORS["brick"] if v < 0 else COLORS["teal"] for v in bar_values]
    ax1.bar(x, bar_values, color=bar_colors, width=0.62, edgecolor="none")
    ax1.axhline(0.0, color=COLORS["gray"], linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(bar_labels)
    ax1.set_ylabel("Relative change, RL vs. PID (%)")
    ax1.set_xlabel("Metric under the same default schedule")
    style_axes(ax1, grid_axis="y")
    add_panel_label(ax1, "b")

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axvspan(8.0, 24.0, color=COLORS["sand"], alpha=0.55, zorder=0)
    ax2.plot(
        tou_hourly["hour"],
        tou_hourly["I1_const"],
        color=COLORS["light_gray"],
        linewidth=1.8,
        label=r"Constant-price RL, $I_1$",
    )
    ax2.plot(
        tou_hourly["hour"],
        tou_hourly["I1_tou"],
        color=MECHANISM_COLORS["tou"],
        linewidth=1.9,
        label=r"TOU-aware RL, $I_1$",
    )
    ax2.set_ylim(0.0, 280.0)
    ax2.set_xlabel("Time of day (h)")
    ax2.set_ylabel(r"Dense-zone PPFD, $I_1$ ($\mu$mol m$^{-2}$ s$^{-1}$)")
    set_hour_ticks(ax2)
    style_axes(ax2, grid_axis="y")
    add_panel_label(ax2, "c")
    ax2b = ax2.twinx()
    ax2b.plot(
        tou_hourly["hour"],
        tou_hourly["price_tou"],
        color=MECHANISM_COLORS["price"],
        linestyle=":",
        linewidth=1.4,
        label="TOU electricity price",
    )
    ax2b.set_ylabel(r"Electricity price (CNY kWh$^{-1}$)")
    ax2b.set_ylim(0.0, max(1.25, float(tou_hourly["price_tou"].max()) * 1.08))
    _style_secondary_axis(ax2b)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(
        lines1 + lines2,
        labels1 + labels2,
        frameon=False,
        fontsize=7.9,
        loc="upper left",
        handletextpad=0.5,
    )

    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def _collect_layer_attribution_tables(
    pid_map: dict[str, pd.DataFrame],
    rl_map: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    shapley_rows = []
    direct_rows = []

    metric_specs = [
        ("net_profit", "Net profit"),
        ("harvest_fresh_kg", "Fresh yield"),
        ("energy_kwh", "Total energy change"),
    ]

    for spec in MATCHED_CONTROLLER_SPECS:
        label = spec["label"]
        pid_df = pid_map[label]
        rl_df = rl_map[label]

        pid_best = pid_df.sort_values("net_profit", ascending=False).iloc[0]
        pid_default = pid_df[pid_df["is_default_schedule"] == True].iloc[0]
        rl_best = rl_df.sort_values("net_profit", ascending=False).iloc[0]
        rl_default = rl_df[rl_df["is_default_schedule"] == True].iloc[0]

        for metric, metric_label in metric_specs:
            b = float(pid_default[metric])
            u = float(pid_best[metric])
            l = float(rl_default[metric])
            j = float(rl_best[metric])

            upper_only = u - b
            lower_only = l - b
            upper_shapley = 0.5 * ((u - b) + (j - l))
            lower_shapley = 0.5 * ((l - b) + (j - u))
            joint_total = j - b

            shapley_rows.append(
                {
                    "scenario": label,
                    "metric": metric,
                    "metric_label": metric_label,
                    "baseline_value": b,
                    "upper_only_delta": upper_only,
                    "lower_only_delta": lower_only,
                    "upper_shapley": upper_shapley,
                    "lower_shapley": lower_shapley,
                    "joint_total_delta": joint_total,
                    "upper_share_pct_of_joint": np.nan if abs(joint_total) < 1.0e-12 else 100.0 * upper_shapley / joint_total,
                    "lower_share_pct_of_joint": np.nan if abs(joint_total) < 1.0e-12 else 100.0 * lower_shapley / joint_total,
                }
            )

        device_keys = {
            "led": "LED energy",
            "hvac": "HVAC energy",
            "dehum": "Dehumidification energy",
            "total": "Total electricity",
        }

        e_b = _device_energy_breakdown(DEFAULT_PID_RESULTS_ROOT / spec["pid_name"], pid_default["schedule_key"])
        e_u = _device_energy_breakdown(DEFAULT_PID_RESULTS_ROOT / spec["pid_name"], pid_best["schedule_key"])
        e_l = _device_energy_breakdown(DEFAULT_RL_RESULTS_ROOT / spec["rl_name"], rl_default["schedule_key"])

        for comp_key, comp_label in device_keys.items():
            baseline = float(e_b[comp_key])
            upper_only = float(e_u[comp_key] - e_b[comp_key])
            lower_only = float(e_l[comp_key] - e_b[comp_key])
            direct_rows.append(
                {
                    "scenario": label,
                    "component": comp_key,
                    "component_label": comp_label,
                    "baseline_kwh": baseline,
                    "upper_only_delta_kwh": upper_only,
                    "lower_only_delta_kwh": lower_only,
                    "upper_only_pct": np.nan if abs(baseline) < 1.0e-12 else 100.0 * upper_only / baseline,
                    "lower_only_pct": np.nan if abs(baseline) < 1.0e-12 else 100.0 * lower_only / baseline,
                    "upper_reduction_pct": np.nan if abs(baseline) < 1.0e-12 else -100.0 * upper_only / baseline,
                    "lower_reduction_pct": np.nan if abs(baseline) < 1.0e-12 else -100.0 * lower_only / baseline,
                }
            )

    return pd.DataFrame(shapley_rows), pd.DataFrame(direct_rows)


def _build_layer_attribution_figure(
    out_path: Path,
    shapley_df: pd.DataFrame,
    direct_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    shapley_avg = (
        shapley_df.groupby(["metric", "metric_label"], as_index=False)[
            ["upper_share_pct_of_joint", "lower_share_pct_of_joint", "upper_shapley", "lower_shapley"]
        ]
        .mean()
    )
    shapley_avg["plot_order"] = shapley_avg["metric"].map(
        {"net_profit": 0, "harvest_fresh_kg": 1, "energy_kwh": 2}
    )
    shapley_avg = shapley_avg.sort_values("plot_order").reset_index(drop=True)

    direct_avg = (
        direct_df.groupby(["component", "component_label"], as_index=False)[
            ["upper_only_pct", "lower_only_pct", "upper_reduction_pct", "lower_reduction_pct"]
        ]
        .mean()
    )
    direct_avg["plot_order"] = direct_avg["component"].map(
        {"led": 0, "hvac": 1, "dehum": 2, "total": 3}
    )
    direct_avg = direct_avg.sort_values("plot_order").reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.9))

    ax0 = axes[0]
    x0 = np.arange(len(shapley_avg))
    w0 = 0.34
    bars_u = ax0.bar(
        x0 - w0 / 2.0,
        shapley_avg["upper_share_pct_of_joint"],
        width=w0,
        color=LAYER_COLORS["upper"],
        label="Upper-level contribution",
        edgecolor="none",
    )
    bars_l = ax0.bar(
        x0 + w0 / 2.0,
        shapley_avg["lower_share_pct_of_joint"],
        width=w0,
        color=LAYER_COLORS["lower"],
        label="Lower-level contribution",
        edgecolor="none",
    )
    for bars in [bars_u, bars_l]:
        for bar in bars:
            value = float(bar.get_height())
            ax0.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + 1.0,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.7,
            )
    ax0.set_xticks(x0)
    ax0.set_xticklabels(shapley_avg["metric_label"])
    ax0.set_xlabel("Outcome in the PID-to-RL joint improvement")
    ax0.set_ylabel("Mean Shapley contribution share (%)")
    ax0.set_ylim(
        0.0,
        max(100.0, float(shapley_avg[["upper_share_pct_of_joint", "lower_share_pct_of_joint"]].to_numpy().max()) + 12.0),
    )
    style_axes(ax0, grid_axis="y")
    add_panel_label(ax0, "a")
    ax0.legend(frameon=False, fontsize=7.9, loc="upper right", handletextpad=0.5)

    ax1 = axes[1]
    x1 = np.arange(len(direct_avg))
    w1 = 0.34
    bars_u2 = ax1.bar(
        x1 - w1 / 2.0,
        direct_avg["upper_reduction_pct"],
        width=w1,
        color=LAYER_COLORS["upper"],
        label="Upper-only effect",
        edgecolor="none",
    )
    bars_l2 = ax1.bar(
        x1 + w1 / 2.0,
        direct_avg["lower_reduction_pct"],
        width=w1,
        color=LAYER_COLORS["lower"],
        label="Lower-only effect",
        edgecolor="none",
    )
    for bars in [bars_u2, bars_l2]:
        for bar in bars:
            value = float(bar.get_height())
            offset = 0.45 if value >= 0 else -0.45
            va = "bottom" if value >= 0 else "top"
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + offset,
                f"{value:.1f}",
                ha="center",
                va=va,
                fontsize=7.5,
            )
    ax1.axhline(0.0, color=COLORS["gray"], linewidth=0.8)
    ax1.set_xticks(x1)
    ax1.set_xticklabels(direct_avg["component_label"], rotation=12)
    ax1.set_xlabel("Energy component with the other layer fixed")
    ax1.set_ylabel("Mean direct reduction relative to PID default (%)")
    style_axes(ax1, grid_axis="y")
    add_panel_label(ax1, "b")
    ax1.legend(frameon=False, fontsize=7.9, loc="upper right", handletextpad=0.5)

    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)
    return shapley_avg, direct_avg


def main() -> None:
    apply_academic_style()

    rl_map: dict[str, pd.DataFrame] = {}
    for spec in RL_SCENARIO_SPECS:
        exp_dir = DEFAULT_RL_RESULTS_ROOT / spec["name"]
        rl_map[spec["label"]] = _load_schedule_df(exp_dir, prefix="rl_exact")

    pid_map: dict[str, pd.DataFrame] = {}
    matched_rl_map: dict[str, pd.DataFrame] = {}
    for spec in MATCHED_CONTROLLER_SPECS:
        pid_map[spec["label"]] = _load_schedule_df(DEFAULT_PID_RESULTS_ROOT / spec["pid_name"], prefix="pid_exact")
        matched_rl_map[spec["label"]] = _load_schedule_df(DEFAULT_RL_RESULTS_ROOT / spec["rl_name"], prefix="rl_exact")

    summary_df, delta_df = _scenario_summary_rows(rl_map)
    eta_df = _collect_sensitivity_tables(rl_map)
    compare_df = _controller_comparison_rows(pid_map, matched_rl_map)
    gain_df = _gain_decomposition_rows(pid_map, matched_rl_map)
    default_hourly_df, default_relative_df, tou_hourly_df = _collect_mechanism_data(pid_map, matched_rl_map)
    layer_shapley_df, layer_direct_df = _collect_layer_attribution_tables(pid_map, matched_rl_map)
    out_dirs = [DEFAULT_OUT_DIR, LEGACY_OUT_DIR]
    for out_dir in out_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)

        _build_rl_core_summary_figure(
            out_dir / "compact_rl_core_summary.png",
            data_map=rl_map,
            summary_df=summary_df,
            delta_df=delta_df,
        )
        eta_avg_df = _build_sensitivity_heatmap(
            out_dir / "compact_rl_upper_sensitivity_heatmap.png",
            eta_df=eta_df,
        )
        _build_controller_comparison_figure(
            out_dir / "compact_rl_pid_controller_comparison.png",
            compare_df=compare_df,
        )
        _build_gain_decomposition_figure(
            out_dir / "compact_rl_gain_decomposition.png",
            gain_df=gain_df,
        )
        _build_mechanism_figure(
            out_dir / "compact_rl_mechanism_profiles.png",
            default_hourly=default_hourly_df,
            default_relative=default_relative_df,
            tou_hourly=tou_hourly_df,
        )
        layer_shapley_avg_df, layer_direct_avg_df = _build_layer_attribution_figure(
            out_dir / "compact_rl_layer_attribution.png",
            shapley_df=layer_shapley_df,
            direct_df=layer_direct_df,
        )

        summary_payload = {
            "rl_scenario_summary": summary_df.to_dict(orient="records"),
            "rl_best_vs_default_delta": delta_df.to_dict(orient="records"),
            "rl_eta2_by_scenario": eta_df.to_dict(orient="records"),
            "rl_eta2_average": eta_avg_df.to_dict(orient="records"),
            "pid_rl_comparison": compare_df.to_dict(orient="records"),
            "rl_gain_decomposition": gain_df.to_dict(orient="records"),
            "rl_mechanism_default_relative_changes": default_relative_df.to_dict(orient="records"),
            "rl_layer_shapley_attribution": layer_shapley_df.to_dict(orient="records"),
            "rl_layer_shapley_average": layer_shapley_avg_df.to_dict(orient="records"),
            "rl_layer_direct_energy_attribution": layer_direct_df.to_dict(orient="records"),
            "rl_layer_direct_energy_average": layer_direct_avg_df.to_dict(orient="records"),
        }
        (out_dir / "compact_rl_result_review_summary.json").write_text(
            json.dumps(summary_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summary_df.to_csv(out_dir / "compact_rl_result_review_summary.csv", index=False, encoding="utf-8-sig")
        delta_df.to_csv(out_dir / "compact_rl_result_review_deltas.csv", index=False, encoding="utf-8-sig")
        eta_df.to_csv(out_dir / "compact_rl_upper_sensitivity_eta2.csv", index=False, encoding="utf-8-sig")
        compare_df.to_csv(out_dir / "compact_rl_pid_controller_comparison.csv", index=False, encoding="utf-8-sig")
        gain_df.to_csv(out_dir / "compact_rl_gain_decomposition.csv", index=False, encoding="utf-8-sig")
        default_relative_df.to_csv(out_dir / "compact_rl_mechanism_default_relative_changes.csv", index=False, encoding="utf-8-sig")
        default_hourly_df.to_csv(out_dir / "compact_rl_mechanism_default_hourly_profile.csv", index=False, encoding="utf-8-sig")
        tou_hourly_df.to_csv(out_dir / "compact_rl_mechanism_tou_hourly_profile.csv", index=False, encoding="utf-8-sig")
        layer_shapley_df.to_csv(out_dir / "compact_rl_layer_shapley_attribution.csv", index=False, encoding="utf-8-sig")
        layer_shapley_avg_df.to_csv(out_dir / "compact_rl_layer_shapley_average.csv", index=False, encoding="utf-8-sig")
        layer_direct_df.to_csv(out_dir / "compact_rl_layer_direct_energy_attribution.csv", index=False, encoding="utf-8-sig")
        layer_direct_avg_df.to_csv(out_dir / "compact_rl_layer_direct_energy_average.csv", index=False, encoding="utf-8-sig")

        print(f"[OK] Saved compact RL result-review figures to: {out_dir}")


if __name__ == "__main__":
    main()
