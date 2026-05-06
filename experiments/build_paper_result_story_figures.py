# -*- coding: utf-8 -*-
"""Build paper-ready figures and compact source tables for the main results story."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from figure_style_academic import (
    COLORS,
    add_panel_label,
    apply_academic_style,
    apply_heatmap_frame,
    save_figure,
    style_axes,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper" / "figures_paper_story_20260428"

PID_CONST_L20_DIR = Path(
    r"C:\Users\29341\Desktop\fsdownload\results"
    r"\exp02_exact_pid_baseline_pp16h_i1250_i2300_pm_constant_e0p74_co20p54_l20"
)
RL_CONST_L20_DIR = Path(
    r"C:\Users\29341\Desktop\fsdownload\results_residual_pid_sac"
    r"\exp03_exact_rl_baseline_pp16h_constant_e0p74_co20p54_l20"
)
RL_CONST_L40_DIR = Path(
    r"C:\Users\29341\Desktop\fsdownload\results_residual_pid_sac"
    r"\exp03_exact_rl_baseline_pp16h_constant_e0p74_co20p54_l40"
)
RL_TOU_L20_DIR = Path(
    r"C:\Users\29341\Desktop\fsdownload\results_residual_pid_sac"
    r"\exp03_exact_rl_baseline_pp16h_tou_zhejiang_lt1kv_l20"
)
RL_DAILY_L20_DIR = Path(
    r"C:\Users\29341\Desktop\fsdownload\results_residual_pid_sac"
    r"\exp03_exact_rl_baseline_pp16h_constant_e0p74_co20p54_l20_daily"
)

RL_SUPP_DIR = ROOT / "paper" / "figures_result_review_rl_20260427_supplement"
CONTEXT_DIR = ROOT / "paper" / "figures_result_review_contextual_20260428"

SUMMARY_COLS = [
    "net_profit",
    "harvest_fresh_kg",
    "energy_kwh",
    "cost_per_kg",
    "avg_harvest_fresh_g_per_plant",
    "avg_harvest_fresh_kg_per_event",
    "total_harvests",
    "total_transplants",
    "rank_valid_profit",
]
TRACE_COLS = [
    "datetime",
    "elapsed_h",
    "elapsed_d",
    "T_in",
    "RH_pct",
    "C_ppm",
    "T_out",
    "RH_out_pct",
    "C_out_ppm",
    "I1",
    "I2",
    "Q_HVAC",
    "u_CO2",
    "m_dehum",
    "P_LED_total_kW",
    "P_HVAC_kW",
    "P_dehum_kW",
    "P_CO2_kW",
    "E_step_kWh",
    "biomass_total_kg_m2",
    "biomass_dense_kg_m2",
    "biomass_finishing_kg_m2",
    "harvest_fresh_mass_equiv_g",
    "harvest_event",
    "step_size_s",
]
MAINLINE_CONTROLLER_ORDER = ["pid", "residual", "climate", "contextual"]
CONTROLLER_LABELS = {
    "pid": "PID",
    "residual": "Residual-PID SAC",
    "gated": "Gated residual",
    "climate": "Climate-only residual",
    "contextual": "Contextual SAC",
}
CONTROLLER_TICK_LABELS = {
    "pid": "PID",
    "residual": "Residual-PID\nSAC",
    "gated": "Gated\nresidual",
    "climate": "Climate-only\nresidual",
    "contextual": "Contextual\nSAC",
}
CONTROLLER_COLORS = {
    "pid": COLORS["gray"],
    "residual": COLORS["teal"],
    "gated": COLORS["green"],
    "climate": COLORS["blue"],
    "contextual": COLORS["brick"],
}
CASE_SPECS = [
    {
        "case_id": "pid_best",
        "label": "PID best",
        "controller": "PID",
        "exp_dir": PID_CONST_L20_DIR,
        "prefix": "pid_exact",
        "schedule_key": "t1=13|t2=13|N1=16|rho2=24",
        "color": COLORS["navy"],
        "linestyle": "-",
    },
    {
        "case_id": "pid_default",
        "label": "PID default",
        "controller": "PID",
        "exp_dir": PID_CONST_L20_DIR,
        "prefix": "pid_exact",
        "schedule_key": "t1=14|t2=14|N1=20|rho2=36",
        "color": COLORS["gray"],
        "linestyle": "-",
    },
    {
        "case_id": "pid_worst",
        "label": "PID worst",
        "controller": "PID",
        "exp_dir": PID_CONST_L20_DIR,
        "prefix": "pid_exact",
        "schedule_key": "t1=16|t2=16|N1=20|rho2=48",
        "color": COLORS["brick"],
        "linestyle": "-",
    },
    {
        "case_id": "rl_default",
        "label": "Residual-PID SAC default",
        "controller": "Residual-PID SAC",
        "exp_dir": RL_CONST_L20_DIR,
        "prefix": "rl_exact",
        "schedule_key": "t1=14|t2=14|N1=20|rho2=36",
        "color": COLORS["teal"],
        "linestyle": "--",
    },
    {
        "case_id": "rl_best",
        "label": "Residual-PID SAC best",
        "controller": "Residual-PID SAC",
        "exp_dir": RL_CONST_L20_DIR,
        "prefix": "rl_exact",
        "schedule_key": "t1=14|t2=12|N1=14|rho2=24",
        "color": COLORS["green"],
        "linestyle": "--",
    },
]
METRIC_LABELS = {
    "net_profit": "Net profit",
    "harvest_fresh_kg": "Fresh yield",
    "energy_kwh": "Electricity use",
    "cost_per_kg": "Fresh-mass cost",
}
UPPER_VARS = ["t1", "t2", "N1", "rho2"]
UPPER_VAR_LABELS = {
    "t1": r"$t_1$",
    "t2": r"$t_2$",
    "N1": r"$N_1$",
    "rho2": r"$\rho_2$",
}


def _schedule_key_to_trace_name(schedule_key: str) -> str:
    parts = dict(part.split("=") for part in schedule_key.split("|"))
    return (
        f"t1-{int(float(parts['t1']))}"
        f"__t2-{int(float(parts['t2']))}"
        f"__N1-{int(float(parts['N1']))}"
        f"__rho2-{int(float(parts['rho2']))}.csv"
    )


def _load_schedule_df(exp_dir: Path, prefix: str) -> pd.DataFrame:
    merged_csv = exp_dir / f"{prefix}_schedule_results.csv"
    if merged_csv.exists():
        df = pd.read_csv(merged_csv)
    else:
        shard_paths = sorted(exp_dir.glob(f"{prefix}_schedule_results.shard_*.csv"))
        if not shard_paths:
            raise FileNotFoundError(f"No schedule results found in {exp_dir}")
        df = pd.concat([pd.read_csv(path) for path in shard_paths], ignore_index=True)

    if "is_default_schedule" in df.columns:
        df["is_default_schedule"] = df["is_default_schedule"].astype(str).str.lower().eq("true")
    for col in SUMMARY_COLS + UPPER_VARS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "schedule_key" not in df.columns:
        df["schedule_key"] = df.apply(
            lambda row: f"t1={int(row['t1'])}|t2={int(row['t2'])}|N1={int(row['N1'])}|rho2={int(row['rho2'])}",
            axis=1,
        )
    if "rank_valid_profit" not in df.columns:
        df = df.sort_values("net_profit", ascending=False).reset_index(drop=True)
        df["rank_valid_profit"] = np.arange(1, len(df) + 1)
    return df


def _load_trace_df(exp_dir: Path, schedule_key: str) -> pd.DataFrame:
    trace_path = exp_dir / "detailed_traces" / _schedule_key_to_trace_name(schedule_key)
    if not trace_path.exists():
        raise FileNotFoundError(f"Trace not found: {trace_path}")
    df = pd.read_csv(trace_path, usecols=TRACE_COLS)
    df["datetime"] = pd.to_datetime(df["datetime"])
    for col in TRACE_COLS:
        if col != "datetime":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    step_h = float(df["step_size_s"].dropna().iloc[0]) / 3600.0
    df["step_h"] = step_h
    df["E_led_kWh"] = df["P_LED_total_kW"] * step_h
    df["E_hvac_kWh"] = df["P_HVAC_kW"] * step_h
    df["E_dehum_kWh"] = df["P_dehum_kW"] * step_h
    df["E_co2_kWh"] = df["P_CO2_kW"] * step_h
    df["harvest_kg_step"] = df["harvest_fresh_mass_equiv_g"].fillna(0.0) / 1000.0
    return df


def _summary_row_by_key(df: pd.DataFrame, schedule_key: str) -> pd.Series:
    row = df.loc[df["schedule_key"] == schedule_key]
    if row.empty:
        raise KeyError(f"Schedule key {schedule_key} not found")
    return row.iloc[0]


def _best_default_worst(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    best = df.sort_values("net_profit", ascending=False).iloc[0]
    default = df.loc[df["is_default_schedule"] == True].iloc[0]
    worst = df.sort_values("net_profit", ascending=True).iloc[0]
    return best, default, worst


def _compute_eta2(df: pd.DataFrame, variable: str, metric: str) -> float:
    y = df[metric].dropna().astype(float).to_numpy()
    y_mean = float(np.mean(y))
    ss_total = float(np.sum((y - y_mean) ** 2))
    if ss_total <= 0.0:
        return 0.0
    grouped = df.groupby(variable)[metric].agg(["count", "mean"]).reset_index()
    ss_between = float(np.sum(grouped["count"] * (grouped["mean"] - y_mean) ** 2))
    return ss_between / ss_total


def _k_rmb_formatter(x: float, _: int) -> str:
    return f"{x / 1000.0:.0f}"


def _mwh_formatter(x: float, _: int) -> str:
    return f"{x / 1000.0:.1f}"


def _percent_formatter(x: float, _: int) -> str:
    return f"{x:.0f}%"


def _round_frame(df: pd.DataFrame, digits: int = 4) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].round(digits)
    return out


def _build_case_bundle(summary_df: pd.DataFrame) -> dict[str, dict[str, object]]:
    case_map: dict[str, dict[str, object]] = {}
    for spec in CASE_SPECS:
        summary_row = _summary_row_by_key(summary_df if spec["prefix"] == "rl_exact" else PID_MAIN_DF, spec["schedule_key"])
        if spec["prefix"] == "rl_exact":
            summary_row = _summary_row_by_key(RL_MAIN_DF, spec["schedule_key"])
        trace_df = _load_trace_df(spec["exp_dir"], spec["schedule_key"])
        case_map[spec["case_id"]] = {
            **spec,
            "summary": summary_row,
            "trace": trace_df,
        }
    return case_map


def _case_metrics_frame(case_map: dict[str, dict[str, object]]) -> pd.DataFrame:
    rows = []
    for case_id, bundle in case_map.items():
        row = bundle["summary"]
        trace = bundle["trace"]
        rows.append(
            {
                "case_id": case_id,
                "label": bundle["label"],
                "controller": bundle["controller"],
                "schedule_key": bundle["schedule_key"],
                "net_profit": float(row["net_profit"]),
                "harvest_fresh_kg": float(row["harvest_fresh_kg"]),
                "energy_kwh": float(row["energy_kwh"]),
                "cost_per_kg": float(row["cost_per_kg"]),
                "avg_harvest_fresh_g_per_plant": float(row["avg_harvest_fresh_g_per_plant"]),
                "total_harvests": int(round(float(row["total_harvests"]))),
                "total_transplants": int(round(float(row["total_transplants"]))),
                "led_mwh": float(trace["E_led_kWh"].sum() / 1000.0),
                "hvac_mwh": float(trace["E_hvac_kWh"].sum() / 1000.0),
                "dehum_mwh": float(trace["E_dehum_kWh"].sum() / 1000.0),
                "co2_mwh": float(trace["E_co2_kWh"].sum() / 1000.0),
                "harvest_mass_trace_kg": float(trace["harvest_kg_step"].sum()),
                "mean_temp_c": float(trace["T_in"].mean()),
                "mean_rh_pct": float(trace["RH_pct"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _prepare_dashboard_trace(trace_df: pd.DataFrame, n_days: int = 32) -> pd.DataFrame:
    short = trace_df.loc[trace_df["elapsed_d"] <= float(n_days)].copy()
    short = short.set_index("datetime").resample("1h").mean(numeric_only=True).reset_index()
    short["elapsed_d"] = (short["datetime"] - short["datetime"].iloc[0]).dt.total_seconds() / 86400.0
    return short


def _select_case_bundles(
    case_map: dict[str, dict[str, object]],
    case_ids: list[str],
) -> list[dict[str, object]]:
    return [case_map[case_id] for case_id in case_ids]


def _plot_upper_need(df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    best, default, worst = _best_default_worst(df)
    ranked = df.sort_values("net_profit", ascending=False).reset_index(drop=True).copy()
    ranked["rank"] = np.arange(1, len(ranked) + 1)

    delta_rows = pd.DataFrame(
        [
            {
                "metric": "Net profit",
                "delta_pct": 100.0 * (best["net_profit"] - default["net_profit"]) / default["net_profit"],
                "color": COLORS["navy"],
            },
            {
                "metric": "Fresh yield",
                "delta_pct": 100.0 * (best["harvest_fresh_kg"] - default["harvest_fresh_kg"]) / default["harvest_fresh_kg"],
                "color": COLORS["teal"],
            },
            {
                "metric": "Electricity use",
                "delta_pct": 100.0 * (best["energy_kwh"] - default["energy_kwh"]) / default["energy_kwh"],
                "color": COLORS["gold"],
            },
            {
                "metric": "Fresh-mass cost",
                "delta_pct": -100.0 * (default["cost_per_kg"] - best["cost_per_kg"]) / default["cost_per_kg"],
                "color": COLORS["brick"],
            },
        ]
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.2), gridspec_kw={"width_ratios": [1.45, 1.0]})
    ax0, ax1 = axes

    ax0.plot(ranked["rank"], ranked["net_profit"], color=COLORS["light_gray"], linewidth=2.1, zorder=1)
    ax0.scatter(ranked["rank"], ranked["net_profit"], s=7, color=COLORS["light_gray"], zorder=2)
    for row, label, color in [
        (best, "Best", COLORS["green"]),
        (default, "Default", COLORS["navy"]),
        (worst, "Worst", COLORS["brick"]),
    ]:
        ax0.scatter(
            float(row["rank_valid_profit"]),
            float(row["net_profit"]),
            s=54,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
            label=label,
        )
    ax0.set_xlabel("Profit rank among 368 feasible schedules")
    ax0.set_ylabel("Net profit (10$^3$ RMB y$^{-1}$)")
    ax0.yaxis.set_major_formatter(FuncFormatter(_k_rmb_formatter))
    style_axes(ax0, grid_axis="y")
    ax0.legend(loc="upper right", ncol=3, handletextpad=0.4, columnspacing=1.0)
    add_panel_label(ax0, "a")

    y = np.arange(len(delta_rows))
    ax1.axvline(0.0, color=COLORS["ink"], linewidth=0.8)
    ax1.barh(
        y,
        delta_rows["delta_pct"],
        color=delta_rows["color"],
        edgecolor="none",
        height=0.56,
    )
    ax1.set_yticks(y)
    ax1.set_yticklabels(delta_rows["metric"])
    ax1.invert_yaxis()
    ax1.set_xlabel("Best vs default change (%)")
    style_axes(ax1, grid_axis="x")
    for yi, value in enumerate(delta_rows["delta_pct"]):
        ha = "left" if value >= 0 else "right"
        x = value + (0.8 if value >= 0 else -0.8)
        ax1.text(x, yi, f"{value:+.1f}%", va="center", ha=ha, fontsize=8.5, color=COLORS["ink"])
    add_panel_label(ax1, "b")

    save_figure(fig, out_path)
    plt.close(fig)
    return delta_rows.drop(columns=["color"])


def _plot_lower_need(
    mechanism_hourly: pd.DataFrame,
    default_compare: pd.DataFrame,
    direct_energy_const: pd.DataFrame,
    out_path: Path,
) -> pd.DataFrame:
    default_row = default_compare.loc[
        (default_compare["scenario"] == "Const-L20") & (default_compare["scope"] == "default_schedule")
    ].iloc[0]
    energy_row = direct_energy_const.loc[
        (direct_energy_const["scenario"] == "Const-L20") & (direct_energy_const["component"] != "total")
    ].copy()

    panel_a = pd.DataFrame(
        [
            {"metric": "Net profit", "value": float(default_row["profit_gain_pct"]), "color": COLORS["navy"]},
            {"metric": "Fresh yield", "value": float(default_row["harvest_gain_pct"]), "color": COLORS["teal"]},
            {"metric": "Electricity reduction", "value": -float(default_row["energy_change_pct"]), "color": COLORS["green"]},
            {"metric": "Cost reduction", "value": float(default_row["cost_reduction_pct"]), "color": COLORS["gold"]},
        ]
    )

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), gridspec_kw={"width_ratios": [0.92, 0.9, 1.35]})
    ax0, ax1, ax2 = axes

    x = np.arange(len(panel_a))
    ax0.bar(x, panel_a["value"], color=panel_a["color"], edgecolor="none", width=0.62)
    ax0.set_xticks(x)
    ax0.set_xticklabels(panel_a["metric"], rotation=12, ha="right")
    ax0.set_ylabel("Relative change (%)")
    style_axes(ax0, grid_axis="y")
    for xi, value in zip(x, panel_a["value"]):
        ax0.text(xi, value + 0.6, f"{value:+.1f}%", ha="center", va="bottom", fontsize=8.5)
    add_panel_label(ax0, "a")

    energy_labels = ["LED", "HVAC", "Dehumidification"]
    energy_values = [
        float(energy_row.loc[energy_row["component"] == "led", "lower_only_pct"].iloc[0]),
        float(energy_row.loc[energy_row["component"] == "hvac", "lower_only_pct"].iloc[0]),
        float(energy_row.loc[energy_row["component"] == "dehum", "lower_only_pct"].iloc[0]),
    ]
    energy_colors = [COLORS["gold"], COLORS["blue"], COLORS["teal"]]
    x1 = np.arange(len(energy_labels))
    ax1.axhline(0.0, color=COLORS["ink"], linewidth=0.8)
    ax1.bar(x1, energy_values, color=energy_colors, edgecolor="none", width=0.62)
    ax1.set_xticks(x1)
    ax1.set_xticklabels(energy_labels, rotation=10)
    ax1.set_ylabel("Device electricity change (%)")
    style_axes(ax1, grid_axis="y")
    for xi, value in zip(x1, energy_values):
        va = "bottom" if value >= 0 else "top"
        y = value + (0.8 if value >= 0 else -0.8)
        ax1.text(xi, y, f"{value:+.1f}%", ha="center", va=va, fontsize=8.5)
    add_panel_label(ax1, "b")

    ax2.plot(
        mechanism_hourly["hour"],
        mechanism_hourly["I1_pid"],
        color=COLORS["navy"],
        linewidth=1.7,
        label=r"PID dense-zone $I_1$",
    )
    ax2.plot(
        mechanism_hourly["hour"],
        mechanism_hourly["I2_pid"],
        color=COLORS["navy"],
        linewidth=1.7,
        linestyle="--",
        alpha=0.72,
        label=r"PID finishing-zone $I_2$",
    )
    ax2.plot(
        mechanism_hourly["hour"],
        mechanism_hourly["I1_rl"],
        color=COLORS["teal"],
        linewidth=1.9,
        label=r"Residual-PID SAC dense-zone $I_1$",
    )
    ax2.plot(
        mechanism_hourly["hour"],
        mechanism_hourly["I2_rl"],
        color=COLORS["teal"],
        linewidth=1.9,
        linestyle="--",
        alpha=0.72,
        label=r"Residual-PID SAC finishing-zone $I_2$",
    )
    ax2.set_xlim(0.0, 24.0)
    ax2.set_xticks(np.arange(0, 25, 4))
    ax2.set_xlabel("Hour of day")
    ax2.set_ylabel("PPFD ($\\mu$mol m$^{-2}$ s$^{-1}$)")
    style_axes(ax2, grid_axis="y")
    ax2.legend(loc="upper left", ncol=2, fontsize=7.8, handlelength=2.1, columnspacing=0.9)
    add_panel_label(ax2, "c")

    save_figure(fig, out_path)
    plt.close(fig)
    return pd.concat(
        [
            panel_a.rename(columns={"value": "value_pct"})[["metric", "value_pct"]],
            energy_row.rename(columns={"component_label": "metric", "lower_only_pct": "value_pct"})[["metric", "value_pct"]],
        ],
        ignore_index=True,
    )


def _plot_synergy(
    pid_df: pd.DataFrame,
    rl_df: pd.DataFrame,
    gain_df: pd.DataFrame,
    out_path: Path,
) -> pd.DataFrame:
    pid_best, pid_default, _ = _best_default_worst(pid_df)
    rl_best, rl_default, _ = _best_default_worst(rl_df)
    gain_row = gain_df.loc[gain_df["scenario"] == "Const-L20"].iloc[0]

    strategy_df = pd.DataFrame(
        [
            {"label": "PID default", "net_profit": float(pid_default["net_profit"]), "color": COLORS["gray"]},
            {"label": "PID + upper opt.", "net_profit": float(pid_best["net_profit"]), "color": COLORS["navy"]},
            {"label": "Residual-PID SAC\non default", "net_profit": float(rl_default["net_profit"]), "color": COLORS["teal"]},
            {"label": "Residual-PID SAC\njoint best", "net_profit": float(rl_best["net_profit"]), "color": COLORS["green"]},
        ]
    )

    frontier_shift_df = pd.DataFrame(
        [
            {"label": "Control gain\non PID-best", "value": float(gain_row["controller_gain_on_pidbest_pct"]), "color": COLORS["teal"]},
            {"label": "Schedule migration\ngain", "value": float(gain_row["migration_gain_pct"]), "color": COLORS["green"]},
        ]
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2), gridspec_kw={"width_ratios": [1.15, 0.95]})
    ax0, ax1 = axes

    x = np.arange(len(strategy_df))
    ax0.bar(x, strategy_df["net_profit"], color=strategy_df["color"], edgecolor="none", width=0.64)
    ax0.set_xticks(x)
    ax0.set_xticklabels(strategy_df["label"], rotation=12, ha="right")
    ax0.set_ylabel("Net profit (10$^3$ RMB y$^{-1}$)")
    ax0.yaxis.set_major_formatter(FuncFormatter(_k_rmb_formatter))
    style_axes(ax0, grid_axis="y")
    for xi, value in zip(x, strategy_df["net_profit"]):
        ax0.text(xi, value + 2200.0, f"{value / 1000.0:.1f}", ha="center", va="bottom", fontsize=8.4)
    add_panel_label(ax0, "a")

    baseline = float(pid_best["net_profit"])
    cumulative = baseline
    ax1.bar(0, baseline, color=COLORS["light_gray"], edgecolor="none", width=0.6)
    ax1.text(0, baseline + 2200.0, f"{baseline / 1000.0:.1f}", ha="center", va="bottom", fontsize=8.4)
    for idx, row in frontier_shift_df.iterrows():
        absolute_delta = baseline * row["value"] / 100.0
        ax1.bar(idx + 1, absolute_delta, bottom=cumulative, color=row["color"], edgecolor="none", width=0.6)
        cumulative += absolute_delta
        ax1.text(idx + 1, cumulative + 2000.0, f"+{row['value']:.2f}%", ha="center", va="bottom", fontsize=8.3)
    ax1.bar(3, float(rl_best["net_profit"]), color="none", edgecolor=COLORS["ink"], linewidth=1.0, width=0.6)
    ax1.text(3, float(rl_best["net_profit"]) + 2200.0, f"{float(rl_best['net_profit']) / 1000.0:.1f}", ha="center", va="bottom", fontsize=8.4)
    ax1.set_xticks([0, 1, 2, 3])
    ax1.set_xticklabels(["PID best", "Control gain", "Migration gain", "Residual-PID SAC\nbest"], rotation=10)
    ax1.set_ylabel("Net profit (10$^3$ RMB y$^{-1}$)")
    ax1.yaxis.set_major_formatter(FuncFormatter(_k_rmb_formatter))
    style_axes(ax1, grid_axis="y")
    add_panel_label(ax1, "b")

    save_figure(fig, out_path)
    plt.close(fig)
    return pd.DataFrame(
        [
            {"metric": "Upper-only gain", "value_pct": float(gain_row["upper_only_pid_pct"])},
            {"metric": "Lower-only gain", "value_pct": float(gain_row["lower_fixed_default_pct"])},
            {"metric": "Joint gain", "value_pct": float(gain_row["total_joint_pct"])},
            {"metric": "Frontier shift total", "value_pct": float(gain_row["frontier_shift_total_pct"])},
            {"metric": "Control gain on PID-best", "value_pct": float(gain_row["controller_gain_on_pidbest_pct"])},
            {"metric": "Schedule migration gain", "value_pct": float(gain_row["migration_gain_pct"])},
        ]
    )


def _plot_case_dashboard(
    case_map: dict[str, dict[str, object]],
    case_ids: list[str],
    out_path: Path,
) -> None:
    bundles = _select_case_bundles(case_map, case_ids)
    short_map = {
        bundle["case_id"]: _prepare_dashboard_trace(bundle["trace"], n_days=32)
        for bundle in bundles
    }

    fig, axes = plt.subplots(2, 4, figsize=(13.8, 7.2), sharex=True)
    axes = axes.ravel()

    for bundle in bundles:
        short = short_map[bundle["case_id"]]
        color = bundle["color"]
        ls = bundle["linestyle"]
        label = bundle["label"]
        axes[0].plot(short["elapsed_d"], short["I1"], color=color, linestyle=ls, linewidth=1.55, alpha=0.97, label=label)
        axes[0].plot(short["elapsed_d"], short["I2"], color=color, linestyle=":", linewidth=1.35, alpha=0.84)
        axes[1].plot(short["elapsed_d"], short["T_in"], color=color, linestyle=ls, linewidth=1.45)
        axes[2].plot(short["elapsed_d"], short["RH_pct"], color=color, linestyle=ls, linewidth=1.45)
        axes[3].plot(short["elapsed_d"], short["C_ppm"], color=color, linestyle=ls, linewidth=1.45)
        axes[4].plot(short["elapsed_d"], short["biomass_total_kg_m2"], color=color, linestyle=ls, linewidth=1.65)
        axes[5].plot(short["elapsed_d"], short["Q_HVAC"], color=color, linestyle=ls, linewidth=1.35)
        axes[6].plot(short["elapsed_d"], short["m_dehum"] * 1.0e6, color=color, linestyle=ls, linewidth=1.35)
        axes[7].plot(short["elapsed_d"], short["u_CO2"] * 1.0e6, color=color, linestyle=ls, linewidth=1.35)

    ref = short_map[bundles[0]["case_id"]]
    axes[1].plot(ref["elapsed_d"], ref["T_out"], color=COLORS["light_gray"], linewidth=1.2, label="Outdoor")
    axes[2].plot(ref["elapsed_d"], ref["RH_out_pct"], color=COLORS["light_gray"], linewidth=1.2, label="Outdoor")
    axes[3].plot(ref["elapsed_d"], ref["C_out_ppm"], color=COLORS["light_gray"], linewidth=1.2, label="Outdoor")
    axes[5].axhline(0.0, color=COLORS["ink"], linewidth=0.8)

    ylabels = [
        "PPFD ($\\mu$mol m$^{-2}$ s$^{-1}$)",
        "Temperature ($^\\circ$C)",
        "Relative humidity (%)",
        "CO$_2$ (ppm)",
        "Total dry biomass (kg m$^{-2}$)",
        "HVAC load (W m$^{-2}$)",
        "Dehumidification rate (mg m$^{-2}$ s$^{-1}$)",
        "CO$_2$ supply rate (mg m$^{-2}$ s$^{-1}$)",
    ]
    letters = list("abcdefgh")
    for ax, ylabel, letter in zip(axes, ylabels, letters):
        ax.set_xlim(0.0, 32.0)
        ax.set_xticks(np.arange(0, 33, 8))
        ax.set_xlabel("Simulation day")
        ax.set_ylabel(ylabel)
        style_axes(ax, grid_axis="y")
        add_panel_label(ax, letter)

    case_handles = [
        Line2D([0], [0], color=bundle["color"], linestyle=bundle["linestyle"], linewidth=2.0, label=bundle["label"])
        for bundle in bundles
    ]
    zone_handles = [
        Line2D([0], [0], color=COLORS["ink"], linestyle="-", linewidth=1.6, label=r"Dense-zone $I_1$"),
        Line2D([0], [0], color=COLORS["ink"], linestyle=":", linewidth=1.6, label=r"Finishing-zone $I_2$"),
    ]
    outdoor_handle = Line2D([0], [0], color=COLORS["light_gray"], linewidth=2.0, label="Outdoor reference")
    fig.legend(
        handles=case_handles + zone_handles + [outdoor_handle],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=3,
        fontsize=8.2,
        columnspacing=1.3,
        handlelength=2.4,
    )

    save_figure(fig, out_path)
    plt.close(fig)


def _plot_case_energy_stacking(case_metrics: pd.DataFrame, out_path: Path) -> None:
    order = ["pid_best", "pid_default", "pid_worst", "rl_default", "rl_best"]
    plot_df = case_metrics.set_index("case_id").loc[order].reset_index()
    x = np.arange(len(plot_df))
    tick_labels = [
        "PID\nbest",
        "PID\ndefault",
        "PID\nworst",
        "Residual-PID SAC\ndefault",
        "Residual-PID SAC\nbest",
    ]

    fig, ax = plt.subplots(figsize=(10.4, 4.6))
    led = plot_df["led_mwh"].to_numpy()
    hvac = plot_df["hvac_mwh"].to_numpy()
    dehum = plot_df["dehum_mwh"].to_numpy()
    ax.bar(x, led, color=COLORS["gold"], edgecolor="none", width=0.64, label="LED")
    ax.bar(x, hvac, bottom=led, color=COLORS["blue"], edgecolor="none", width=0.64, label="HVAC")
    ax.bar(x, dehum, bottom=led + hvac, color=COLORS["teal"], edgecolor="none", width=0.64, label="Dehumidification")
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel("Annual electricity use (MWh y$^{-1}$)")
    style_axes(ax, grid_axis="y")
    add_panel_label(ax, "a")

    ax2 = ax.twinx()
    ax2.plot(
        x,
        plot_df["net_profit"] / 1000.0,
        color=COLORS["ink"],
        linewidth=1.5,
        marker="o",
        markersize=4.0,
        label="Net profit",
    )
    ax2.set_ylabel("Net profit (10$^3$ RMB y$^{-1}$)")
    ax2.spines["top"].set_visible(False)
    ax2.tick_params(axis="y", colors=COLORS["ink"])

    legend_handles = [
        Patch(facecolor=COLORS["gold"], edgecolor="none", label="LED"),
        Patch(facecolor=COLORS["blue"], edgecolor="none", label="HVAC"),
        Patch(facecolor=COLORS["teal"], edgecolor="none", label="Dehumidification"),
        Line2D([0], [0], color=COLORS["ink"], marker="o", linewidth=1.5, markersize=4.0, label="Net profit"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", ncol=2)

    save_figure(fig, out_path)
    plt.close(fig)


def _plot_upper_sensitivity(df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    eta_rows = []
    for variable in UPPER_VARS:
        for metric in ["net_profit", "harvest_fresh_kg", "energy_kwh", "cost_per_kg"]:
            eta_rows.append(
                {
                    "variable": variable,
                    "metric": metric,
                    "eta2": _compute_eta2(df, variable, metric),
                }
            )
    eta_df = pd.DataFrame(eta_rows)
    heatmap_df = eta_df.pivot(index="variable", columns="metric", values="eta2").loc[UPPER_VARS]
    heatmap_df = heatmap_df[["net_profit", "harvest_fresh_kg", "energy_kwh", "cost_per_kg"]]

    rho2_curve = df.groupby("rho2", as_index=False)["net_profit"].mean().sort_values("rho2")
    n1_curve = df.groupby("N1", as_index=False)["energy_kwh"].mean().sort_values("N1")

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.3), gridspec_kw={"width_ratios": [1.05, 0.9, 0.9]})
    ax0, ax1, ax2 = axes

    im = ax0.imshow(heatmap_df.to_numpy(), cmap="YlGnBu", vmin=0.0, vmax=max(0.85, float(heatmap_df.to_numpy().max())))
    ax0.set_xticks(np.arange(heatmap_df.shape[1]))
    ax0.set_xticklabels(["Net profit", "Fresh yield", "Electricity", "Cost per kg"], rotation=18, ha="right")
    ax0.set_yticks(np.arange(heatmap_df.shape[0]))
    ax0.set_yticklabels([UPPER_VAR_LABELS[v] for v in heatmap_df.index])
    apply_heatmap_frame(ax0, heatmap_df.shape[0], heatmap_df.shape[1])
    for i in range(heatmap_df.shape[0]):
        for j in range(heatmap_df.shape[1]):
            ax0.text(j, i, f"{heatmap_df.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8.1, color=COLORS["ink"])
    add_panel_label(ax0, "a")

    cbar = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
    cbar.set_label(r"Main-effect $\eta^2$")

    ax1.plot(rho2_curve["rho2"], rho2_curve["net_profit"], color=COLORS["navy"], linewidth=1.9, marker="o", markersize=4.2)
    ax1.set_xlabel(r"Finishing density, $\rho_2$")
    ax1.set_ylabel("Mean net profit (10$^3$ RMB y$^{-1}$)")
    ax1.yaxis.set_major_formatter(FuncFormatter(_k_rmb_formatter))
    style_axes(ax1, grid_axis="y")
    add_panel_label(ax1, "b")

    ax2.plot(n1_curve["N1"], n1_curve["energy_kwh"], color=COLORS["gold"], linewidth=1.9, marker="o", markersize=4.2)
    ax2.set_xlabel(r"Dense-zone boards, $N_1$")
    ax2.set_ylabel("Mean electricity use (MWh y$^{-1}$)")
    ax2.yaxis.set_major_formatter(FuncFormatter(_mwh_formatter))
    style_axes(ax2, grid_axis="y")
    add_panel_label(ax2, "c")

    save_figure(fig, out_path)
    plt.close(fig)
    return eta_df


def _plot_controller_positioning(controller_df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    main_df = controller_df.loc[controller_df["scenario"] == "Const-L20"].copy()
    best_df = (
        main_df.loc[main_df["level"] == "best"]
        .set_index("controller")
        .loc[MAINLINE_CONTROLLER_ORDER]
        .reset_index()
    )
    default_df = (
        main_df.loc[main_df["level"] == "default"]
        .set_index("controller")
        .loc[MAINLINE_CONTROLLER_ORDER]
        .reset_index()
    )
    for frame in (best_df, default_df):
        frame["controller_label"] = frame["controller"].map(CONTROLLER_LABELS)

    fig, axes = plt.subplots(1, 3, figsize=(12.9, 4.25))
    best_colors = [CONTROLLER_COLORS[c] for c in best_df["controller"]]

    x = np.arange(len(best_df))
    axes[0].bar(x, best_df["net_profit"], color=best_colors, edgecolor="none", width=0.64)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([CONTROLLER_TICK_LABELS[c] for c in best_df["controller"]])
    axes[0].set_ylabel("Best net profit (10$^3$ RMB y$^{-1}$)")
    axes[0].yaxis.set_major_formatter(FuncFormatter(_k_rmb_formatter))
    style_axes(axes[0], grid_axis="y")
    add_panel_label(axes[0], "a")

    x1 = np.arange(len(default_df))
    axes[1].bar(x1, default_df["net_profit"], color=[CONTROLLER_COLORS[c] for c in default_df["controller"]], edgecolor="none", width=0.64)
    axes[1].set_xticks(x1)
    axes[1].set_xticklabels([CONTROLLER_TICK_LABELS[c] for c in default_df["controller"]])
    axes[1].set_ylabel("Default-schedule net profit (10$^3$ RMB y$^{-1}$)")
    axes[1].yaxis.set_major_formatter(FuncFormatter(_k_rmb_formatter))
    style_axes(axes[1], grid_axis="y")
    add_panel_label(axes[1], "b")

    axes[2].bar(x, best_df["total_harvests"], color=best_colors, edgecolor="none", width=0.64)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([CONTROLLER_TICK_LABELS[c] for c in best_df["controller"]])
    axes[2].set_ylabel("Annual harvest count of best schedule")
    style_axes(axes[2], grid_axis="y")
    add_panel_label(axes[2], "c")

    for ax in axes:
        ax.tick_params(axis="x", labelsize=8.7, pad=2.0)
    fig.subplots_adjust(bottom=0.16, wspace=0.32)

    save_figure(fig, out_path)
    plt.close(fig)
    out_df = pd.concat([default_df, best_df], ignore_index=True)
    column_order = [
        "controller",
        "controller_label",
        "scenario",
        "level",
        "schedule_key",
        "schedule_tuple",
        "net_profit",
        "harvest_fresh_kg",
        "energy_kwh",
        "cost_per_kg",
        "total_harvests",
        "rank_valid_profit",
    ]
    return out_df[column_order]


def _plot_layer_attribution(
    shapley_df: pd.DataFrame,
    direct_energy_df: pd.DataFrame,
    out_path: Path,
) -> pd.DataFrame:
    shapley_main = shapley_df.loc[shapley_df["scenario"] == "Const-L20"].copy()
    energy_main = direct_energy_df.loc[
        (direct_energy_df["scenario"] == "Const-L20") & (direct_energy_df["component"].isin(["led", "hvac", "dehum", "total"]))
    ].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.3), gridspec_kw={"width_ratios": [1.0, 1.0]})
    ax0, ax1 = axes

    metric_labels = ["Net profit", "Fresh yield", "Total energy change"]
    upper_share = shapley_main["upper_share_pct_of_joint"].to_numpy()
    lower_share = shapley_main["lower_share_pct_of_joint"].to_numpy()
    y = np.arange(len(metric_labels))
    ax0.barh(y, upper_share, color=COLORS["navy"], edgecolor="none", height=0.56, label="Upper layer")
    ax0.barh(y, lower_share, left=upper_share, color=COLORS["green"], edgecolor="none", height=0.56, label="Lower layer")
    ax0.set_yticks(y)
    ax0.set_yticklabels(metric_labels)
    ax0.invert_yaxis()
    ax0.set_xlim(0.0, 100.0)
    ax0.set_xlabel("Share of joint change (%)")
    style_axes(ax0, grid_axis="x")
    ax0.xaxis.set_major_formatter(FuncFormatter(_percent_formatter))
    for yi, up, lo in zip(y, upper_share, lower_share):
        ax0.text(up / 2.0, yi, f"{up:.1f}%", ha="center", va="center", fontsize=8.0, color="white")
        ax0.text(up + lo / 2.0, yi, f"{lo:.1f}%", ha="center", va="center", fontsize=8.0, color=COLORS["ink"])
    ax0.legend(loc="lower right")
    add_panel_label(ax0, "a")

    comp_labels = ["LED", "HVAC", "Dehumidification", "Total"]
    x = np.arange(len(comp_labels))
    width = 0.34
    ax1.bar(x - width / 2.0, energy_main["upper_reduction_pct"], width, color=COLORS["navy"], edgecolor="none", label="Upper-only reduction")
    ax1.bar(x + width / 2.0, energy_main["lower_reduction_pct"], width, color=COLORS["green"], edgecolor="none", label="Lower-only reduction")
    ax1.set_xticks(x)
    ax1.set_xticklabels(comp_labels, rotation=12)
    ax1.set_ylabel("Direct electricity reduction (%)")
    style_axes(ax1, grid_axis="y")
    for xi, up, lo in zip(x, energy_main["upper_reduction_pct"], energy_main["lower_reduction_pct"]):
        ax1.text(xi - width / 2.0, up + 0.6, f"{up:.1f}", ha="center", va="bottom", fontsize=7.9)
        ax1.text(xi + width / 2.0, lo + 0.6, f"{lo:.1f}", ha="center", va="bottom", fontsize=7.9)
    ax1.legend(loc="upper right")
    add_panel_label(ax1, "b")

    save_figure(fig, out_path)
    plt.close(fig)
    return pd.concat([shapley_main, energy_main], ignore_index=True)


def _plot_price_extension(const_l20_df: pd.DataFrame, const_l40_df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    merged = const_l20_df.merge(
        const_l40_df,
        on="schedule_key",
        suffixes=("_l20", "_l40"),
    )
    merged["profit_delta"] = merged["net_profit_l40"] - merged["net_profit_l20"]
    merged["rank_shift_abs"] = (merged["rank_valid_profit_l40"] - merged["rank_valid_profit_l20"]).abs()

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.2))
    ax0, ax1 = axes

    ax0.scatter(
        merged["harvest_fresh_kg_l20"] / 1000.0,
        merged["profit_delta"] / 1000.0,
        s=22,
        color=COLORS["navy"],
        alpha=0.72,
        edgecolors="white",
        linewidth=0.35,
    )
    coef = np.polyfit(merged["harvest_fresh_kg_l20"] / 1000.0, merged["profit_delta"] / 1000.0, deg=1)
    xfit = np.linspace(float(merged["harvest_fresh_kg_l20"].min() / 1000.0), float(merged["harvest_fresh_kg_l20"].max() / 1000.0), 100)
    yfit = coef[0] * xfit + coef[1]
    ax0.plot(xfit, yfit, color=COLORS["gold"], linewidth=1.8)
    ax0.set_xlabel("Annual fresh yield under L20 (t y$^{-1}$)")
    ax0.set_ylabel("Profit increase from L20 to L40 (10$^3$ RMB y$^{-1}$)")
    style_axes(ax0, grid_axis="y")
    add_panel_label(ax0, "a")

    ax1.scatter(
        merged["rank_valid_profit_l20"],
        merged["rank_valid_profit_l40"],
        s=20,
        color=COLORS["teal"],
        alpha=0.72,
        edgecolors="white",
        linewidth=0.35,
    )
    diag_max = int(max(merged["rank_valid_profit_l20"].max(), merged["rank_valid_profit_l40"].max()))
    ax1.plot([1, diag_max], [1, diag_max], color=COLORS["gray"], linewidth=1.0, linestyle="--")
    ax1.set_xlabel("Profit rank under Const-L20")
    ax1.set_ylabel("Profit rank under Const-L40")
    style_axes(ax1, grid_axis="both")
    add_panel_label(ax1, "b")

    save_figure(fig, out_path)
    plt.close(fig)
    return merged


def _plot_tou_extension(
    const_l20_df: pd.DataFrame,
    tou_l20_df: pd.DataFrame,
    tou_hourly_df: pd.DataFrame,
    out_path: Path,
) -> pd.DataFrame:
    merged = const_l20_df.merge(
        tou_l20_df,
        on="schedule_key",
        suffixes=("_const", "_tou"),
    )
    merged["rank_shift_abs"] = (merged["rank_valid_profit_tou"] - merged["rank_valid_profit_const"]).abs()
    merged["profit_delta_pct"] = 100.0 * (merged["net_profit_tou"] - merged["net_profit_const"]) / merged["net_profit_const"]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.2), gridspec_kw={"width_ratios": [1.0, 1.05]})
    ax0, ax1 = axes

    ax0.scatter(
        merged["rank_valid_profit_const"],
        merged["rank_valid_profit_tou"],
        s=20,
        color=COLORS["teal"],
        alpha=0.72,
        edgecolors="white",
        linewidth=0.35,
    )
    diag_max = int(max(merged["rank_valid_profit_const"].max(), merged["rank_valid_profit_tou"].max()))
    ax0.plot([1, diag_max], [1, diag_max], color=COLORS["gray"], linewidth=1.0, linestyle="--")
    ax0.set_xlabel("Profit rank under Const-L20")
    ax0.set_ylabel("Profit rank under TOU-L20")
    style_axes(ax0, grid_axis="both")
    add_panel_label(ax0, "a")

    ax1.plot(tou_hourly_df["hour"], tou_hourly_df["I1_const"], color=COLORS["navy"], linewidth=1.8, label=r"Const dense-zone $I_1$")
    ax1.plot(tou_hourly_df["hour"], tou_hourly_df["I1_tou"], color=COLORS["teal"], linewidth=1.9, label=r"TOU dense-zone $I_1$")
    ax1.set_xlim(0.0, 24.0)
    ax1.set_xticks(np.arange(0, 25, 4))
    ax1.set_xlabel("Hour of day")
    ax1.set_ylabel("Dense-zone PPFD ($\\mu$mol m$^{-2}$ s$^{-1}$)")
    style_axes(ax1, grid_axis="y")
    ax1b = ax1.twinx()
    ax1b.plot(tou_hourly_df["hour"], tou_hourly_df["price_tou"], color=COLORS["gold"], linewidth=1.4, linestyle="--", label="TOU electricity price")
    ax1b.set_ylabel("Electricity price (RMB kWh$^{-1}$)")
    ax1b.spines["top"].set_visible(False)
    ax1b.tick_params(axis="y", colors=COLORS["ink"])
    handles = [
        Line2D([0], [0], color=COLORS["navy"], linewidth=1.8, label=r"Const dense-zone $I_1$"),
        Line2D([0], [0], color=COLORS["teal"], linewidth=1.9, label=r"TOU dense-zone $I_1$"),
        Line2D([0], [0], color=COLORS["gold"], linewidth=1.4, linestyle="--", label="TOU electricity price"),
    ]
    ax1.legend(handles=handles, loc="upper left")
    add_panel_label(ax1, "b")

    save_figure(fig, out_path)
    plt.close(fig)
    return merged


def _plot_daily_extension(seg_df: pd.DataFrame, daily_df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    ranked_seg = seg_df.sort_values("net_profit", ascending=False).reset_index(drop=True).copy()
    ranked_seg["rank"] = np.arange(1, len(ranked_seg) + 1)
    ranked_day = daily_df.sort_values("net_profit", ascending=False).reset_index(drop=True).copy()
    ranked_day["rank"] = np.arange(1, len(ranked_day) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.2))
    ax0, ax1 = axes

    ax0.plot(ranked_seg["rank"], ranked_seg["net_profit"], color=COLORS["navy"], linewidth=1.9, label="Three-segment hold")
    ax0.plot(ranked_day["rank"], ranked_day["net_profit"], color=COLORS["green"], linewidth=1.9, label="Daily hold")
    ax0.set_xlabel("Profit rank among 368 feasible schedules")
    ax0.set_ylabel("Net profit (10$^3$ RMB y$^{-1}$)")
    ax0.yaxis.set_major_formatter(FuncFormatter(_k_rmb_formatter))
    style_axes(ax0, grid_axis="y")
    ax0.legend(loc="upper right")
    add_panel_label(ax0, "a")

    ax1.scatter(
        seg_df["total_harvests"],
        seg_df["net_profit"] / 1000.0,
        s=22,
        color=COLORS["navy"],
        alpha=0.55,
        edgecolors="white",
        linewidth=0.3,
        label="Three-segment hold",
    )
    ax1.scatter(
        daily_df["total_harvests"],
        daily_df["net_profit"] / 1000.0,
        s=22,
        color=COLORS["green"],
        alpha=0.55,
        edgecolors="white",
        linewidth=0.3,
        label="Daily hold",
    )
    ax1.set_xlabel("Annual harvest count")
    ax1.set_ylabel("Net profit (10$^3$ RMB y$^{-1}$)")
    style_axes(ax1, grid_axis="y")
    ax1.legend(loc="lower right")
    add_panel_label(ax1, "b")

    save_figure(fig, out_path)
    plt.close(fig)
    return pd.concat(
        [
            ranked_seg.head(10).assign(mode="segmented"),
            ranked_day.head(10).assign(mode="daily"),
        ],
        ignore_index=True,
    )


def build_story() -> None:
    apply_academic_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    global PID_MAIN_DF
    global RL_MAIN_DF
    PID_MAIN_DF = _load_schedule_df(PID_CONST_L20_DIR, "pid_exact")
    RL_MAIN_DF = _load_schedule_df(RL_CONST_L20_DIR, "rl_exact")
    RL_L40_DF = _load_schedule_df(RL_CONST_L40_DIR, "rl_exact")
    RL_TOU_DF = _load_schedule_df(RL_TOU_L20_DIR, "rl_exact")
    RL_DAILY_DF = _load_schedule_df(RL_DAILY_L20_DIR, "rl_exact")

    gain_df = pd.read_csv(RL_SUPP_DIR / "compact_rl_gain_decomposition.csv")
    default_compare_df = pd.read_csv(RL_SUPP_DIR / "compact_rl_pid_controller_comparison.csv")
    direct_energy_df = pd.read_csv(RL_SUPP_DIR / "compact_rl_layer_direct_energy_attribution.csv")
    shapley_df = pd.read_csv(RL_SUPP_DIR / "compact_rl_layer_shapley_attribution.csv")
    mechanism_hourly = pd.read_csv(RL_SUPP_DIR / "compact_rl_mechanism_default_hourly_profile.csv")
    tou_hourly = pd.read_csv(RL_SUPP_DIR / "compact_rl_mechanism_tou_hourly_profile.csv")
    controller_df = pd.read_csv(CONTEXT_DIR / "compact_contextual_controller_comparison.csv")

    numbers: list[dict[str, object]] = []

    upper_delta_df = _plot_upper_need(RL_MAIN_DF, OUT_DIR / "paper_mainline_upper_need.png")
    for _, row in upper_delta_df.iterrows():
        numbers.append({"section": "upper_need", "metric": row["metric"], "value_pct": row["delta_pct"]})

    lower_df = _plot_lower_need(
        mechanism_hourly,
        default_compare_df,
        direct_energy_df,
        OUT_DIR / "paper_mainline_lower_need.png",
    )
    for _, row in lower_df.iterrows():
        numbers.append({"section": "lower_need", "metric": row["metric"], "value_pct": row["value_pct"]})

    synergy_df = _plot_synergy(PID_MAIN_DF, RL_MAIN_DF, gain_df, OUT_DIR / "paper_mainline_synergy.png")
    for _, row in synergy_df.iterrows():
        numbers.append({"section": "synergy", "metric": row["metric"], "value_pct": row["value_pct"]})

    case_map = _build_case_bundle(RL_MAIN_DF)
    case_metrics = _case_metrics_frame(case_map)
    _round_frame(case_metrics).to_csv(OUT_DIR / "paper_mainline_case_metrics_constl20.csv", index=False)
    _plot_case_dashboard(
        case_map,
        ["pid_best", "pid_default", "pid_worst"],
        OUT_DIR / "paper_case_dashboard_constl20.png",
    )
    _plot_case_dashboard(
        case_map,
        ["pid_best", "pid_default", "pid_worst"],
        OUT_DIR / "paper_case_upper_dashboard_constl20.png",
    )
    _plot_case_dashboard(
        case_map,
        ["pid_default", "rl_default"],
        OUT_DIR / "paper_case_lower_dashboard_constl20.png",
    )
    _plot_case_dashboard(
        case_map,
        ["pid_best", "rl_best"],
        OUT_DIR / "paper_case_synergy_dashboard_constl20.png",
    )
    _plot_case_energy_stacking(case_metrics, OUT_DIR / "paper_case_energy_stacking_constl20.png")

    eta_df = _plot_upper_sensitivity(RL_MAIN_DF, OUT_DIR / "paper_mainline_upper_sensitivity.png")
    _round_frame(eta_df).to_csv(OUT_DIR / "paper_mainline_upper_sensitivity_eta2.csv", index=False)

    controller_main_df = _plot_controller_positioning(controller_df, OUT_DIR / "paper_mainline_controller_positioning.png")
    _round_frame(controller_main_df).to_csv(OUT_DIR / "paper_mainline_controller_positioning_constl20.csv", index=False)

    layer_df = _plot_layer_attribution(shapley_df, direct_energy_df, OUT_DIR / "paper_mainline_layer_attribution.png")
    _round_frame(layer_df).to_csv(OUT_DIR / "paper_mainline_layer_attribution_constl20.csv", index=False)

    price_df = _plot_price_extension(RL_MAIN_DF, RL_L40_DF, OUT_DIR / "paper_extension_lettuce_price.png")
    price_summary = {
        "mean_profit_delta_rmb": float(price_df["profit_delta"].mean()),
        "mean_abs_rank_shift": float(price_df["rank_shift_abs"].mean()),
        "max_abs_rank_shift": float(price_df["rank_shift_abs"].max()),
    }
    for key, value in price_summary.items():
        numbers.append({"section": "price_extension", "metric": key, "value": value})
    _round_frame(price_df[["schedule_key", "profit_delta", "rank_shift_abs"]]).to_csv(
        OUT_DIR / "paper_extension_lettuce_price_source.csv", index=False
    )

    tou_df = _plot_tou_extension(RL_MAIN_DF, RL_TOU_DF, tou_hourly, OUT_DIR / "paper_extension_tou.png")
    tou_summary = {
        "mean_abs_rank_shift": float(tou_df["rank_shift_abs"].mean()),
        "max_abs_rank_shift": float(tou_df["rank_shift_abs"].max()),
        "mean_profit_delta_pct": float(tou_df["profit_delta_pct"].mean()),
    }
    for key, value in tou_summary.items():
        numbers.append({"section": "tou_extension", "metric": key, "value": value})
    _round_frame(tou_df[["schedule_key", "rank_shift_abs", "profit_delta_pct"]]).to_csv(
        OUT_DIR / "paper_extension_tou_source.csv", index=False
    )

    daily_df = _plot_daily_extension(RL_MAIN_DF, RL_DAILY_DF, OUT_DIR / "paper_extension_daily_hold.png")
    _round_frame(daily_df).to_csv(OUT_DIR / "paper_extension_daily_hold_source.csv", index=False)
    best_seg = RL_MAIN_DF.sort_values("net_profit", ascending=False).iloc[0]
    best_day = RL_DAILY_DF.sort_values("net_profit", ascending=False).iloc[0]
    numbers.extend(
        [
            {"section": "daily_extension", "metric": "best_profit_gain_daily_vs_segmented_pct", "value_pct": 100.0 * (best_day["net_profit"] - best_seg["net_profit"]) / best_seg["net_profit"]},
            {"section": "daily_extension", "metric": "best_harvest_count_segmented", "value": float(best_seg["total_harvests"])},
            {"section": "daily_extension", "metric": "best_harvest_count_daily", "value": float(best_day["total_harvests"])},
        ]
    )

    numbers_df = pd.DataFrame(numbers)
    _round_frame(numbers_df).to_csv(OUT_DIR / "paper_mainline_story_numbers.csv", index=False)


if __name__ == "__main__":
    build_story()
