# -*- coding: utf-8 -*-
"""Build manuscript-style representative case trajectory figures."""

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
    save_figure,
    set_day_ticks,
    set_hour_ticks,
    style_axes,
)


ROOT = Path(__file__).resolve().parents[1]
PID_EXP_DIR = Path(
    r"C:\Users\29341\Desktop\fsdownload\results"
    r"\exp02_exact_pid_baseline_pp16h_i1250_i2300_pm_constant_e0p74_co20p54_l40"
)
RL_EXP_DIR = Path(
    r"C:\Users\29341\Desktop\fsdownload\results_residual_pid_sac"
    r"\exp03_exact_rl_baseline_pp16h_constant_e0p74_co20p54_l40"
)
OUT_DIR = ROOT / "paper" / "figures_case_review_20260427"

TRACE_COLS = [
    "datetime",
    "elapsed_d",
    "T_in",
    "RH_pct",
    "VPD_kPa",
    "T_out",
    "RH_out_pct",
    "C_ppm",
    "I1",
    "I2",
    "P_LED_total_kW",
    "P_HVAC_kW",
    "P_dehum_kW",
    "P_CO2_kW",
    "E_step_kWh",
    "E_transp_kg",
    "dehum_removed_kg",
    "env_condensation_removed_kg",
    "harvest_event",
    "harvest_fresh_mass_equiv_g",
    "harvest_mean_fresh_mass_per_plant_g",
    "step_size_s",
]

SUMMARY_NUMERIC_COLS = [
    "net_profit",
    "harvest_fresh_kg",
    "energy_kwh",
    "cost_per_kg",
    "avg_harvest_fresh_kg_per_event",
    "avg_harvest_fresh_g_per_plant",
    "total_harvests",
    "total_transplants",
]

CASE_SPECS = [
    {
        "case_id": "pid_best",
        "label": "PID best",
        "controller": "PID",
        "exp_dir": PID_EXP_DIR,
        "prefix": "pid_exact",
        "schedule_key": "t1=13|t2=13|N1=16|rho2=24",
        "color": COLORS["navy"],
        "linestyle": "-",
    },
    {
        "case_id": "pid_default",
        "label": "PID default",
        "controller": "PID",
        "exp_dir": PID_EXP_DIR,
        "prefix": "pid_exact",
        "schedule_key": "t1=14|t2=14|N1=20|rho2=36",
        "color": COLORS["gray"],
        "linestyle": "-",
    },
    {
        "case_id": "pid_worst",
        "label": "PID worst",
        "controller": "PID",
        "exp_dir": PID_EXP_DIR,
        "prefix": "pid_exact",
        "schedule_key": "t1=16|t2=16|N1=20|rho2=48",
        "color": COLORS["brick"],
        "linestyle": "-",
    },
    {
        "case_id": "rl_default",
        "label": "RL default",
        "controller": "RL",
        "exp_dir": RL_EXP_DIR,
        "prefix": "rl_exact",
        "schedule_key": "t1=14|t2=14|N1=20|rho2=36",
        "color": COLORS["teal"],
        "linestyle": "--",
    },
    {
        "case_id": "rl_best",
        "label": "RL best",
        "controller": "RL",
        "exp_dir": RL_EXP_DIR,
        "prefix": "rl_exact",
        "schedule_key": "t1=15|t2=13|N1=15|rho2=23",
        "color": COLORS["green"],
        "linestyle": "--",
    },
]

DEVICE_COLORS = {
    "LED": COLORS["gold"],
    "HVAC": COLORS["blue"],
    "Dehumidification": COLORS["teal"],
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
            raise FileNotFoundError(f"No schedule result csv found in {exp_dir}")
        df = pd.concat([pd.read_csv(path) for path in shard_paths], ignore_index=True)

    df["is_default_schedule"] = df["is_default_schedule"].astype(str).str.lower().eq("true")
    for col in SUMMARY_NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
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
    step_hours = float(df["step_size_s"].dropna().iloc[0]) / 3600.0 if df["step_size_s"].notna().any() else 1.0 / 6.0
    df["step_h"] = step_hours
    df["E_led_kWh"] = df["P_LED_total_kW"] * step_hours
    df["E_hvac_kWh"] = df["P_HVAC_kW"] * step_hours
    df["E_dehum_kWh"] = df["P_dehum_kW"] * step_hours
    df["E_co2_kWh"] = df["P_CO2_kW"] * step_hours
    df["harvest_kg_step"] = df["harvest_fresh_mass_equiv_g"].fillna(0.0) / 1000.0
    return df


def _daily_summary(trace_df: pd.DataFrame) -> pd.DataFrame:
    df = trace_df.set_index("datetime").sort_index()
    daily = pd.DataFrame(index=pd.date_range(df.index.min().floor("D"), df.index.max().floor("D"), freq="D"))
    daily["harvest_kg"] = df["harvest_kg_step"].resample("D").sum().reindex(daily.index, fill_value=0.0)
    daily["harvest_events"] = df["harvest_event"].resample("D").sum().reindex(daily.index, fill_value=0.0)
    daily["transp_kg"] = df["E_transp_kg"].resample("D").sum().reindex(daily.index, fill_value=0.0)
    daily["dehum_removed_kg"] = df["dehum_removed_kg"].resample("D").sum().reindex(daily.index, fill_value=0.0)
    daily["cond_removed_kg"] = df["env_condensation_removed_kg"].resample("D").sum().reindex(daily.index, fill_value=0.0)
    daily["energy_total_kwh"] = df["E_step_kWh"].resample("D").sum().reindex(daily.index, fill_value=0.0)
    daily["energy_led_kwh"] = df["E_led_kWh"].resample("D").sum().reindex(daily.index, fill_value=0.0)
    daily["energy_hvac_kwh"] = df["E_hvac_kWh"].resample("D").sum().reindex(daily.index, fill_value=0.0)
    daily["energy_dehum_kwh"] = df["E_dehum_kWh"].resample("D").sum().reindex(daily.index, fill_value=0.0)
    daily["RH_pct"] = df["RH_pct"].resample("D").mean().reindex(daily.index)
    daily["VPD_kPa"] = df["VPD_kPa"].resample("D").mean().reindex(daily.index)
    daily["cum_harvest_t"] = daily["harvest_kg"].cumsum() / 1000.0
    daily["cum_harvest_events"] = daily["harvest_events"].cumsum()
    daily["cum_transp_t"] = daily["transp_kg"].cumsum() / 1000.0
    daily["cum_dehum_removed_t"] = daily["dehum_removed_kg"].cumsum() / 1000.0
    daily["cum_energy_total_mwh"] = daily["energy_total_kwh"].cumsum() / 1000.0
    daily["cum_energy_dehum_mwh"] = daily["energy_dehum_kwh"].cumsum() / 1000.0
    daily["rh_roll7"] = daily["RH_pct"].rolling(7, min_periods=1).mean()
    daily["vpd_roll7"] = daily["VPD_kPa"].rolling(7, min_periods=1).mean()
    daily["elapsed_d"] = np.arange(len(daily), dtype=float)
    return daily.reset_index().rename(columns={"index": "date"})


def _event_summary(trace_df: pd.DataFrame) -> pd.DataFrame:
    events = trace_df.loc[
        trace_df["harvest_event"] > 0,
        ["elapsed_d", "harvest_kg_step", "harvest_mean_fresh_mass_per_plant_g"],
    ].copy()
    events["event_idx"] = np.arange(1, len(events) + 1)
    return events.rename(
        columns={
            "harvest_kg_step": "harvest_kg",
            "harvest_mean_fresh_mass_per_plant_g": "fresh_g_per_plant",
        }
    )


def _hourly_light_profile(trace_df: pd.DataFrame) -> pd.DataFrame:
    hours = trace_df["datetime"].dt.hour + trace_df["datetime"].dt.minute / 60.0
    return (
        pd.DataFrame({"hour": hours, "I1": trace_df["I1"], "I2": trace_df["I2"]})
        .groupby("hour", as_index=False)
        .mean()
        .sort_values("hour")
    )


def _build_case_record(spec: dict[str, object], summary_row: pd.Series, trace_df: pd.DataFrame) -> dict[str, object]:
    return {
        "case_id": spec["case_id"],
        "label": spec["label"],
        "controller": spec["controller"],
        "schedule_key": spec["schedule_key"],
        "net_profit_rmb": float(summary_row["net_profit"]),
        "harvest_kg": float(summary_row["harvest_fresh_kg"]),
        "energy_kwh": float(summary_row["energy_kwh"]),
        "cost_per_kg_rmb": float(summary_row["cost_per_kg"]),
        "avg_fresh_g_per_plant": float(summary_row["avg_harvest_fresh_g_per_plant"]),
        "total_harvests": int(round(float(summary_row["total_harvests"]))),
        "total_transplants": int(round(float(summary_row["total_transplants"]))),
        "avg_harvest_kg_per_event": float(summary_row["avg_harvest_fresh_kg_per_event"]),
        "led_mwh": float(trace_df["E_led_kWh"].sum() / 1000.0),
        "hvac_mwh": float(trace_df["E_hvac_kWh"].sum() / 1000.0),
        "dehum_mwh": float(trace_df["E_dehum_kWh"].sum() / 1000.0),
        "co2_mwh": float(trace_df["E_co2_kWh"].sum() / 1000.0),
        "mean_rh_pct": float(trace_df["RH_pct"].mean()),
        "mean_vpd_kpa": float(trace_df["VPD_kPa"].mean()),
        "mean_temp_c": float(trace_df["T_in"].mean()),
        "transp_kg": float(trace_df["E_transp_kg"].sum()),
        "dehum_removed_kg": float(trace_df["dehum_removed_kg"].sum()),
        "cond_removed_kg": float(trace_df["env_condensation_removed_kg"].sum()),
    }


def _build_comparison_rows(metrics_df: pd.DataFrame) -> pd.DataFrame:
    metric_map = metrics_df.set_index("case_id")
    pairs = [
        ("PID best vs PID default", "pid_best", "pid_default"),
        ("PID worst vs PID default", "pid_worst", "pid_default"),
        ("RL default vs PID default", "rl_default", "pid_default"),
        ("RL best vs PID best", "rl_best", "pid_best"),
        ("RL best vs RL default", "rl_best", "rl_default"),
    ]
    rows = []
    for label, lhs_key, rhs_key in pairs:
        lhs = metric_map.loc[lhs_key]
        rhs = metric_map.loc[rhs_key]
        rows.append(
            {
                "comparison": label,
                "profit_change_pct": 100.0 * (lhs["net_profit_rmb"] - rhs["net_profit_rmb"]) / rhs["net_profit_rmb"],
                "harvest_change_pct": 100.0 * (lhs["harvest_kg"] - rhs["harvest_kg"]) / rhs["harvest_kg"],
                "energy_change_pct": 100.0 * (lhs["energy_kwh"] - rhs["energy_kwh"]) / rhs["energy_kwh"],
                "cost_per_kg_change_pct": 100.0 * (lhs["cost_per_kg_rmb"] - rhs["cost_per_kg_rmb"]) / rhs["cost_per_kg_rmb"],
                "avg_fresh_change_pct": 100.0 * (lhs["avg_fresh_g_per_plant"] - rhs["avg_fresh_g_per_plant"]) / rhs["avg_fresh_g_per_plant"],
                "harvest_count_change_pct": 100.0 * (lhs["total_harvests"] - rhs["total_harvests"]) / rhs["total_harvests"],
                "led_change_pct": 100.0 * (lhs["led_mwh"] - rhs["led_mwh"]) / rhs["led_mwh"],
                "hvac_change_pct": 100.0 * (lhs["hvac_mwh"] - rhs["hvac_mwh"]) / rhs["hvac_mwh"],
                "dehum_change_pct": 100.0 * (lhs["dehum_mwh"] - rhs["dehum_mwh"]) / rhs["dehum_mwh"],
                "transp_change_pct": 100.0 * (lhs["transp_kg"] - rhs["transp_kg"]) / rhs["transp_kg"],
                "dehum_removed_change_pct": 100.0 * (lhs["dehum_removed_kg"] - rhs["dehum_removed_kg"]) / rhs["dehum_removed_kg"],
            }
        )
    return pd.DataFrame(rows)


def _plot_grouped_device_bars(ax: plt.Axes, metrics_df: pd.DataFrame, case_ids: list[str]) -> None:
    subset = metrics_df.set_index("case_id").loc[case_ids]
    labels = subset["label"].tolist()
    x = np.arange(len(labels))
    width = 0.22
    ax.bar(x - width, subset["led_mwh"], width, label="LED", color=DEVICE_COLORS["LED"], edgecolor="none")
    ax.bar(x, subset["hvac_mwh"], width, label="HVAC", color=DEVICE_COLORS["HVAC"], edgecolor="none")
    ax.bar(x + width, subset["dehum_mwh"], width, label="Dehumidification", color=DEVICE_COLORS["Dehumidification"], edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_xlabel("Case")
    ax.set_ylabel("Electricity use (MWh)")
    style_axes(ax, grid_axis="y")


def _apply_common_line_axis(ax: plt.Axes, ylabel: str, xlabel: str) -> None:
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    style_axes(ax, grid_axis="y")


def _plot_pid_structure_figure(out_path: Path, case_map: dict[str, dict[str, object]], metrics_df: pd.DataFrame) -> None:
    case_ids = ["pid_best", "pid_default", "pid_worst"]
    fig, axes = plt.subplots(2, 4, figsize=(13.4, 7.2))
    axes = axes.ravel()

    for case_id in case_ids:
        bundle = case_map[case_id]
        daily = bundle["daily"]
        events = bundle["events"]
        axes[0].plot(daily["elapsed_d"], daily["cum_harvest_t"], label=bundle["label"], color=bundle["color"], linestyle=bundle["linestyle"], linewidth=1.8)
        axes[1].plot(daily["elapsed_d"], daily["cum_harvest_events"], label=bundle["label"], color=bundle["color"], linestyle=bundle["linestyle"], linewidth=1.8)
        axes[2].plot(events["event_idx"], events["fresh_g_per_plant"], label=bundle["label"], color=bundle["color"], linestyle=bundle["linestyle"], linewidth=1.5, marker="o", markersize=2.3)
        axes[3].plot(daily["elapsed_d"], daily["cum_energy_total_mwh"], label=bundle["label"], color=bundle["color"], linestyle=bundle["linestyle"], linewidth=1.8)
        axes[4].plot(daily["elapsed_d"], daily["cum_transp_t"], label=bundle["label"], color=bundle["color"], linestyle=bundle["linestyle"], linewidth=1.8)
        axes[5].plot(daily["elapsed_d"], daily["cum_dehum_removed_t"], label=bundle["label"], color=bundle["color"], linestyle=bundle["linestyle"], linewidth=1.8)
        axes[6].plot(daily["elapsed_d"], daily["cum_energy_dehum_mwh"], label=bundle["label"], color=bundle["color"], linestyle=bundle["linestyle"], linewidth=1.8)

    ylabels = [
        "Cumulative fresh yield (t)",
        "Cumulative harvest events (-)",
        r"Harvest fresh mass (g plant$^{-1}$)",
        "Cumulative electricity use (MWh)",
        r"Cumulative transpiration (t H$_2$O)",
        r"Cumulative dehumidification removal (t H$_2$O)",
        "Cumulative dehumidification electricity (MWh)",
    ]
    xlabels = ["Time (d)", "Time (d)", "Harvest event index (-)", "Time (d)", "Time (d)", "Time (d)", "Time (d)"]
    for idx, (ax, ylabel, xlabel) in enumerate(zip(axes[:7], ylabels, xlabels)):
        _apply_common_line_axis(ax, ylabel=ylabel, xlabel=xlabel)
        set_day_ticks(ax) if idx != 2 else None
        add_panel_label(ax, chr(ord("a") + idx))
    _plot_grouped_device_bars(axes[7], metrics_df, case_ids)
    axes[7].legend(frameon=False, loc="upper left", handletextpad=0.5, borderaxespad=0.2)
    add_panel_label(axes[7], "h")
    case_handles, case_labels = axes[0].get_legend_handles_labels()
    fig.legend(case_handles, case_labels, loc="upper center", ncol=3, frameon=False, columnspacing=1.4, handletextpad=0.6)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, out_path)
    plt.close(fig)


def _plot_default_pid_vs_rl_figure(out_path: Path, case_map: dict[str, dict[str, object]], metrics_df: pd.DataFrame) -> None:
    case_ids = ["pid_default", "rl_default"]
    fig, axes = plt.subplots(2, 4, figsize=(13.4, 7.2))
    axes = axes.ravel()

    for case_id in case_ids:
        bundle = case_map[case_id]
        daily = bundle["daily"]
        hourly = bundle["hourly"]
        axes[0].plot(hourly["hour"], hourly["I1"], label=bundle["label"], color=bundle["color"], linestyle=bundle["linestyle"], linewidth=1.8)
        axes[1].plot(hourly["hour"], hourly["I2"], label=bundle["label"], color=bundle["color"], linestyle=bundle["linestyle"], linewidth=1.8)
        axes[2].plot(daily["elapsed_d"], daily["rh_roll7"], label=bundle["label"], color=bundle["color"], linestyle=bundle["linestyle"], linewidth=1.8)
        axes[3].plot(daily["elapsed_d"], daily["vpd_roll7"], label=bundle["label"], color=bundle["color"], linestyle=bundle["linestyle"], linewidth=1.8)
        axes[4].plot(daily["elapsed_d"], daily["cum_harvest_t"], label=bundle["label"], color=bundle["color"], linestyle=bundle["linestyle"], linewidth=1.8)
        axes[5].plot(daily["elapsed_d"], daily["cum_dehum_removed_t"], label=bundle["label"], color=bundle["color"], linestyle=bundle["linestyle"], linewidth=1.8)
        axes[6].plot(daily["elapsed_d"], daily["cum_energy_total_mwh"], label=bundle["label"], color=bundle["color"], linestyle=bundle["linestyle"], linewidth=1.8)

    ylabels = [
        r"Dense-zone PPFD, $I_1$ ($\mu$mol m$^{-2}$ s$^{-1}$)",
        r"Finishing-zone PPFD, $I_2$ ($\mu$mol m$^{-2}$ s$^{-1}$)",
        "7-d mean relative humidity (%)",
        "7-d mean VPD (kPa)",
        "Cumulative fresh yield (t)",
        r"Cumulative dehumidification removal (t H$_2$O)",
        "Cumulative electricity use (MWh)",
    ]
    xlabels = ["Time of day (h)", "Time of day (h)", "Time (d)", "Time (d)", "Time (d)", "Time (d)", "Time (d)"]
    for idx, (ax, ylabel, xlabel) in enumerate(zip(axes[:7], ylabels, xlabels)):
        _apply_common_line_axis(ax, ylabel=ylabel, xlabel=xlabel)
        if idx < 2:
            set_hour_ticks(ax)
        else:
            set_day_ticks(ax)
        add_panel_label(ax, chr(ord("a") + idx))
    _plot_grouped_device_bars(axes[7], metrics_df, case_ids)
    axes[7].legend(frameon=False, loc="upper left", handletextpad=0.5, borderaxespad=0.2)
    add_panel_label(axes[7], "h")
    case_handles, case_labels = axes[0].get_legend_handles_labels()
    fig.legend(case_handles, case_labels, loc="upper center", ncol=2, frameon=False, columnspacing=1.4, handletextpad=0.6)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, out_path)
    plt.close(fig)


def _plot_pidbest_vs_rlbest_figure(out_path: Path, case_map: dict[str, dict[str, object]], metrics_df: pd.DataFrame) -> None:
    case_ids = ["pid_best", "rl_best"]
    fig, axes = plt.subplots(2, 4, figsize=(13.4, 7.2))
    axes = axes.ravel()

    for case_id in case_ids:
        bundle = case_map[case_id]
        daily = bundle["daily"]
        hourly = bundle["hourly"]
        events = bundle["events"]
        axes[0].plot(hourly["hour"], hourly["I1"], label=bundle["label"], color=bundle["color"], linestyle=bundle["linestyle"], linewidth=1.8)
        axes[1].plot(hourly["hour"], hourly["I2"], label=bundle["label"], color=bundle["color"], linestyle=bundle["linestyle"], linewidth=1.8)
        axes[2].plot(daily["elapsed_d"], daily["cum_harvest_t"], label=bundle["label"], color=bundle["color"], linestyle=bundle["linestyle"], linewidth=1.8)
        axes[3].plot(daily["elapsed_d"], daily["cum_harvest_events"], label=bundle["label"], color=bundle["color"], linestyle=bundle["linestyle"], linewidth=1.8)
        axes[4].plot(events["elapsed_d"], events["fresh_g_per_plant"], label=bundle["label"], color=bundle["color"], linestyle="none", marker="o", markersize=2.3, alpha=0.85)
        axes[5].plot(daily["elapsed_d"], daily["cum_energy_total_mwh"], label=bundle["label"], color=bundle["color"], linestyle=bundle["linestyle"], linewidth=1.8)
        axes[6].plot(daily["elapsed_d"], daily["cum_dehum_removed_t"], label=bundle["label"], color=bundle["color"], linestyle=bundle["linestyle"], linewidth=1.8)

    ylabels = [
        r"Dense-zone PPFD, $I_1$ ($\mu$mol m$^{-2}$ s$^{-1}$)",
        r"Finishing-zone PPFD, $I_2$ ($\mu$mol m$^{-2}$ s$^{-1}$)",
        "Cumulative fresh yield (t)",
        "Cumulative harvest events (-)",
        r"Harvest fresh mass (g plant$^{-1}$)",
        "Cumulative electricity use (MWh)",
        r"Cumulative dehumidification removal (t H$_2$O)",
    ]
    xlabels = ["Time of day (h)", "Time of day (h)", "Time (d)", "Time (d)", "Time (d)", "Time (d)", "Time (d)"]
    for idx, (ax, ylabel, xlabel) in enumerate(zip(axes[:7], ylabels, xlabels)):
        _apply_common_line_axis(ax, ylabel=ylabel, xlabel=xlabel)
        if idx < 2:
            set_hour_ticks(ax)
        else:
            set_day_ticks(ax)
        add_panel_label(ax, chr(ord("a") + idx))
    _plot_grouped_device_bars(axes[7], metrics_df, case_ids)
    axes[7].legend(frameon=False, loc="upper left", handletextpad=0.5, borderaxespad=0.2)
    add_panel_label(axes[7], "h")
    case_handles, case_labels = axes[0].get_legend_handles_labels()
    fig.legend(case_handles, case_labels, loc="upper center", ncol=2, frameon=False, columnspacing=1.4, handletextpad=0.6)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, out_path)
    plt.close(fig)


def main() -> None:
    apply_academic_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pid_df = _load_schedule_df(PID_EXP_DIR, "pid_exact")
    rl_df = _load_schedule_df(RL_EXP_DIR, "rl_exact")

    case_map: dict[str, dict[str, object]] = {}
    records: list[dict[str, object]] = []

    for spec in CASE_SPECS:
        schedule_df = pid_df if spec["controller"] == "PID" else rl_df
        row = schedule_df.loc[schedule_df["schedule_key"] == spec["schedule_key"]]
        if row.empty:
            raise KeyError(f"Schedule not found: {spec['schedule_key']}")
        summary_row = row.iloc[0]
        trace_df = _load_trace_df(Path(spec["exp_dir"]), str(spec["schedule_key"]))
        record = _build_case_record(spec, summary_row, trace_df)
        case_bundle = dict(spec)
        case_bundle["trace"] = trace_df
        case_bundle["daily"] = _daily_summary(trace_df)
        case_bundle["events"] = _event_summary(trace_df)
        case_bundle["hourly"] = _hourly_light_profile(trace_df)
        case_map[str(spec["case_id"])] = case_bundle
        records.append(record)

    metrics_df = pd.DataFrame(records)
    comparison_df = _build_comparison_rows(metrics_df)

    _plot_pid_structure_figure(OUT_DIR / "compact_case_pid_structure_constl40.png", case_map, metrics_df)
    _plot_default_pid_vs_rl_figure(OUT_DIR / "compact_case_default_pid_vs_rl_constl40.png", case_map, metrics_df)
    _plot_pidbest_vs_rlbest_figure(OUT_DIR / "compact_case_pidbest_vs_rlbest_constl40.png", case_map, metrics_df)

    metrics_df.to_csv(OUT_DIR / "case_key_metrics_constl40.csv", index=False, encoding="utf-8-sig")
    comparison_df.to_csv(OUT_DIR / "case_comparison_constl40.csv", index=False, encoding="utf-8-sig")
    payload = {
        "case_key_metrics": metrics_df.to_dict(orient="records"),
        "case_comparisons": comparison_df.to_dict(orient="records"),
    }
    (OUT_DIR / "case_key_metrics_constl40.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved case figures to: {OUT_DIR}")


if __name__ == "__main__":
    main()
