from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper" / "figures_humidity_stress_20260506"

STYLE_DIR = Path(r"C:\Users\29341\Desktop\code_0420\lettuce_control_drl_new_0422\experiments")
sys.path.insert(0, str(STYLE_DIR))

from figure_style_academic import (  # noqa: E402
    COLORS,
    add_panel_label,
    apply_academic_style,
    save_figure,
    style_axes,
)


PID_DIR = Path(
    r"C:\Users\29341\Desktop\fsdownload\results"
    r"\exp02_exact_pid_baseline_pp16h_i1250_i2300_pm_constant_e0p74_co20p54_l20"
)
RL_DIR = Path(
    r"C:\Users\29341\Desktop\fsdownload\results_residual_pid_sac"
    r"\exp03_exact_rl_baseline_pp16h_constant_e0p74_co20p54_l20"
)
SCHEDULE_SLUG = "t1-14__t2-14__N1-20__rho2-36.csv"
SCHEDULE_KEY = "t1=14|t2=14|N1=20|rho2=36"

PID_COLOR = COLORS["navy"]
RL_COLOR = COLORS["teal"]
DEHUM_COLOR = COLORS["brick"]
TRANSP_COLOR = COLORS["gold"]
HVAC_COLOR = COLORS["blue"]
LED_COLOR = COLORS["gold"]


def _load_schedule_row(exp_dir: Path, prefix: str) -> pd.Series:
    df = pd.read_csv(exp_dir / f"{prefix}_schedule_results.csv")
    return df.loc[df["schedule_key"].astype(str).eq(SCHEDULE_KEY)].iloc[0]


def _load_trace(exp_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(exp_dir / "detailed_traces" / SCHEDULE_SLUG)
    df["datetime"] = pd.to_datetime(df["datetime"])
    step_h = float(df["step_size_s"].iloc[0]) / 3600.0
    df["E_led_kWh_step"] = df["P_LED_total_kW"] * step_h
    df["E_hvac_kWh_step"] = df["P_HVAC_kW"] * step_h
    df["E_dehum_kWh_step"] = df["P_dehum_kW"] * step_h
    df["E_total_kWh_step"] = df["E_step_kWh"]
    df["cum_led_mwh"] = df["E_led_kWh_step"].cumsum() / 1000.0
    df["cum_hvac_mwh"] = df["E_hvac_kWh_step"].cumsum() / 1000.0
    df["cum_dehum_mwh"] = df["E_dehum_kWh_step"].cumsum() / 1000.0
    df["cum_transp_t"] = df["E_transp_kg"].cumsum() / 1000.0
    df["cum_dehum_removed_t"] = df["dehum_removed_kg"].cumsum() / 1000.0
    df["m_dehum_umol"] = df["m_dehum"] * 1e6
    return df


def _hourly(df: pd.DataFrame, days: float = 32.0) -> pd.DataFrame:
    short = df.loc[df["elapsed_d"] <= days].copy()
    cols = [
        "elapsed_d",
        "I1",
        "I2",
        "T_in",
        "RH_pct",
        "VPD_kPa",
        "Q_HVAC",
        "m_dehum_umol",
        "P_dehum_kW",
        "cum_transp_t",
        "cum_dehum_removed_t",
        "cum_dehum_mwh",
        "biomass_dense_kg_m2",
        "biomass_finishing_kg_m2",
    ]
    agg = {col: "last" for col in cols if col in short.columns}
    hourly = short.set_index("datetime").resample("1h").agg(agg).dropna(how="all").reset_index()
    hourly["elapsed_d"] = (hourly["datetime"] - hourly["datetime"].iloc[0]).dt.total_seconds() / 86400.0
    return hourly


def _daily_light_on_mean(df: pd.DataFrame, days: float = 32.0) -> pd.DataFrame:
    short = df.loc[df["elapsed_d"] <= days].copy()
    short["day"] = np.floor(short["elapsed_d"]).astype(int)
    rows = []
    for day, group in short.groupby("day"):
        row = {"elapsed_d": float(day) + 0.5}
        for col in ["I1", "I2"]:
            light_on = group.loc[group[col] > 0, col]
            row[col] = float(light_on.mean()) if not light_on.empty else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _metric_summary(pid_row: pd.Series, rl_row: pd.Series, pid: pd.DataFrame, rl: pd.DataFrame) -> pd.DataFrame:
    def annual_metrics(row: pd.Series, trace: pd.DataFrame, controller: str) -> dict[str, float | str]:
        return {
            "controller": controller,
            "net_profit_rmb": float(row["net_profit"]),
            "fresh_yield_kg": float(row["harvest_fresh_kg"]),
            "energy_kwh": float(row["energy_kwh"]),
            "cost_per_kg": float(row["cost_per_kg"]),
            "mean_fresh_g_per_plant": float(row["avg_harvest_fresh_g_per_plant"]),
            "led_mwh": float(trace["E_led_kWh_step"].sum() / 1000.0),
            "hvac_mwh": float(trace["E_hvac_kWh_step"].sum() / 1000.0),
            "dehum_mwh": float(trace["E_dehum_kWh_step"].sum() / 1000.0),
            "transp_t": float(trace["E_transp_kg"].sum() / 1000.0),
            "dehum_removed_t": float(trace["dehum_removed_kg"].sum() / 1000.0),
            "mean_rh_pct": float(trace["RH_pct"].mean()),
            "p90_rh_pct": float(trace["RH_pct"].quantile(0.9)),
            "rh_gt90_h": float((trace["RH_pct"] > 90.0).sum() * float(trace["step_size_s"].iloc[0]) / 3600.0),
            "mean_vpd_kpa": float(trace["VPD_kPa"].mean()),
            "p10_vpd_kpa": float(trace["VPD_kPa"].quantile(0.1)),
            "vpd_lt04_h": float((trace["VPD_kPa"] < 0.4).sum() * float(trace["step_size_s"].iloc[0]) / 3600.0),
            "mean_i1_light": float(trace.loc[trace["I1"] > 0, "I1"].mean()),
            "mean_i2_light": float(trace.loc[trace["I2"] > 0, "I2"].mean()),
            "mean_dehum_command_umol": float(trace["m_dehum_umol"].mean()),
        }

    base = pd.DataFrame(
        [
            annual_metrics(pid_row, pid, "PID"),
            annual_metrics(rl_row, rl, "Residual-PID SAC"),
        ]
    )
    delta_rows = []
    pid_vals = base.loc[base["controller"] == "PID"].iloc[0]
    rl_vals = base.loc[base["controller"] == "Residual-PID SAC"].iloc[0]
    for metric in base.columns:
        if metric == "controller":
            continue
        pid_value = float(pid_vals[metric])
        rl_value = float(rl_vals[metric])
        delta_rows.append(
            {
                "metric": metric,
                "pid_value": pid_value,
                "rl_value": rl_value,
                "delta": rl_value - pid_value,
                "delta_pct": 100.0 * (rl_value - pid_value) / pid_value if abs(pid_value) > 1e-12 else np.nan,
            }
        )
    delta = pd.DataFrame(delta_rows)
    base.to_csv(OUT_DIR / "humidity_stress_default_case_metrics.csv", index=False)
    delta.to_csv(OUT_DIR / "humidity_stress_default_case_deltas.csv", index=False)
    return delta


def _pct_label(value: float) -> str:
    return f"{value:+.1f}%"


def _plot_mechanism_overview(pid: pd.DataFrame, rl: pd.DataFrame, delta: pd.DataFrame, out_path: Path) -> None:
    pid_h = _hourly(pid)
    rl_h = _hourly(rl)
    pid_light = _daily_light_on_mean(pid)
    rl_light = _daily_light_on_mean(rl)
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.0))
    axes = axes.flatten()

    axes[0].plot(pid_light["elapsed_d"], pid_light["I1"], color=PID_COLOR, linewidth=1.75, label=r"PID $I_1$")
    axes[0].plot(pid_light["elapsed_d"], pid_light["I2"], color=PID_COLOR, linestyle="--", linewidth=1.75, label=r"PID $I_2$")
    axes[0].plot(rl_light["elapsed_d"], rl_light["I1"], color=RL_COLOR, linewidth=1.85, label=r"Residual-PID SAC $I_1$")
    axes[0].plot(rl_light["elapsed_d"], rl_light["I2"], color=RL_COLOR, linestyle="--", linewidth=1.85, label=r"Residual-PID SAC $I_2$")
    axes[0].set_ylabel(r"Light-on PPFD ($\mu$mol m$^{-2}$ s$^{-1}$)")
    axes[0].legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2, fontsize=7.0, frameon=False)

    axes[1].plot(pid_h["elapsed_d"], pid_h["cum_transp_t"], color=PID_COLOR, label="PID transpiration")
    axes[1].plot(pid_h["elapsed_d"], pid_h["cum_dehum_removed_t"], color=PID_COLOR, linestyle="--", label="PID dehumidified water")
    axes[1].plot(rl_h["elapsed_d"], rl_h["cum_transp_t"], color=RL_COLOR, label="Residual transpiration")
    axes[1].plot(rl_h["elapsed_d"], rl_h["cum_dehum_removed_t"], color=RL_COLOR, linestyle="--", label="Residual dehumidified water")
    axes[1].set_ylabel("Cumulative water flux (t)")
    axes[1].legend(loc="upper left", fontsize=6.8, frameon=False)

    axes[2].plot(pid_h["elapsed_d"], pid_h["cum_dehum_mwh"], color=PID_COLOR, label="PID")
    axes[2].plot(rl_h["elapsed_d"], rl_h["cum_dehum_mwh"], color=RL_COLOR, label="Residual-PID SAC")
    axes[2].set_ylabel("Cumulative dehumidification electricity (MWh)")
    axes[2].legend(loc="upper left", fontsize=7.2, frameon=False)

    bins_rh = np.linspace(45, 100, 45)
    axes[3].hist(pid["RH_pct"], bins=bins_rh, density=True, histtype="stepfilled", alpha=0.35, color=PID_COLOR, label="PID")
    axes[3].hist(rl["RH_pct"], bins=bins_rh, density=True, histtype="step", linewidth=2.0, color=RL_COLOR, label="Residual-PID SAC")
    axes[3].axvline(90, color=COLORS["gray"], linestyle="--", linewidth=1.0)
    axes[3].set_xlabel("Relative humidity (%)")
    axes[3].set_ylabel("Density")
    axes[3].legend(loc="upper left", fontsize=7.2, frameon=False)

    bins_vpd = np.linspace(0.05, 1.6, 45)
    axes[4].hist(pid["VPD_kPa"], bins=bins_vpd, density=True, histtype="stepfilled", alpha=0.35, color=PID_COLOR, label="PID")
    axes[4].hist(rl["VPD_kPa"], bins=bins_vpd, density=True, histtype="step", linewidth=2.0, color=RL_COLOR, label="Residual-PID SAC")
    axes[4].axvline(0.4, color=COLORS["gray"], linestyle="--", linewidth=1.0)
    axes[4].set_xlabel("VPD (kPa)")
    axes[4].set_ylabel("Density")

    energy_metrics = [
        ("led_mwh", "LED", LED_COLOR),
        ("hvac_mwh", "HVAC", HVAC_COLOR),
        ("dehum_mwh", "Dehum.", DEHUM_COLOR),
        ("energy_kwh", "Total", COLORS["gray"]),
    ]
    x = np.arange(len(energy_metrics))
    vals = [float(delta.loc[delta["metric"].eq(metric), "delta_pct"].iloc[0]) for metric, _, _ in energy_metrics]
    colors = [color for _, _, color in energy_metrics]
    axes[5].axhline(0, color=COLORS["ink"], linewidth=0.8)
    axes[5].bar(x, vals, color=colors, edgecolor="none", width=0.62)
    axes[5].set_xticks(x)
    axes[5].set_xticklabels([label for _, label, _ in energy_metrics], rotation=12, ha="right")
    axes[5].set_ylabel("Residual vs PID change (%)")
    for xi, value in zip(x, vals):
        va = "bottom" if value >= 0 else "top"
        axes[5].text(xi, value + (0.9 if value >= 0 else -0.9), _pct_label(value), ha="center", va=va, fontsize=8.0)

    labels = list("abcdef")
    for idx, ax in enumerate(axes):
        style_axes(ax, grid_axis="y")
        add_panel_label(ax, labels[idx])
        if idx <= 2:
            ax.set_xlim(0, 32)
            ax.set_xlabel("Elapsed time (d)")
    fig.subplots_adjust(wspace=0.34, hspace=0.46)
    save_figure(fig, out_path)
    plt.close(fig)


def _plot_annual_water_energy(pid: pd.DataFrame, rl: pd.DataFrame, delta: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.4), gridspec_kw={"width_ratios": [1.1, 1.0, 1.25]})
    ax0, ax1, ax2 = axes

    for trace, label, color in [(pid, "PID", PID_COLOR), (rl, "Residual-PID SAC", RL_COLOR)]:
        monthly = trace.set_index("datetime").resample("ME").agg(
            transp_t=("E_transp_kg", lambda x: x.sum() / 1000.0),
            dehum_removed_t=("dehum_removed_kg", lambda x: x.sum() / 1000.0),
            dehum_mwh=("E_dehum_kWh_step", lambda x: x.sum() / 1000.0),
        )
        month = np.arange(1, len(monthly) + 1)
        ax0.plot(month, monthly["transp_t"], color=color, marker="o", markersize=3.5, label=f"{label} transp.")
        ax0.plot(month, monthly["dehum_removed_t"], color=color, linestyle="--", marker="s", markersize=3.2, label=f"{label} dehum. water")
        ax1.plot(month, monthly["dehum_mwh"], color=color, marker="o", markersize=3.5, label=label)

    ax0.set_xlabel("Month")
    ax0.set_ylabel("Monthly water flux (t)")
    ax0.legend(loc="upper left", fontsize=6.9, ncol=1, frameon=False)
    style_axes(ax0, grid_axis="y")
    add_panel_label(ax0, "a")

    ax1.set_xlabel("Month")
    ax1.set_ylabel("Monthly dehumidification electricity (MWh)")
    ax1.legend(loc="upper left", fontsize=7.2, frameon=False)
    style_axes(ax1, grid_axis="y")
    add_panel_label(ax1, "b")

    metrics = [
        ("fresh_yield_kg", "Yield", COLORS["green"]),
        ("net_profit_rmb", "Profit", COLORS["navy"]),
        ("transp_t", "Transp.", TRANSP_COLOR),
        ("dehum_removed_t", "Dehum.\nwater", RL_COLOR),
        ("dehum_mwh", "Dehum.\nelec.", DEHUM_COLOR),
    ]
    vals = [float(delta.loc[delta["metric"].eq(metric), "delta_pct"].iloc[0]) for metric, _, _ in metrics]
    x = np.arange(len(metrics))
    ax2.axhline(0, color=COLORS["ink"], linewidth=0.8)
    ax2.bar(x, vals, color=[color for _, _, color in metrics], edgecolor="none", width=0.62)
    ax2.set_xticks(x)
    ax2.set_xticklabels([label for _, label, _ in metrics], rotation=0)
    ax2.set_ylabel("Residual vs PID change (%)")
    ymin = min(vals) - 7.0
    ymax = max(vals) + 7.0
    ax2.set_ylim(ymin, ymax)
    for xi, value in zip(x, vals):
        va = "bottom" if value >= 0 else "top"
        offset = 1.0 if value >= 0 else -1.0
        ax2.text(xi, value + offset, _pct_label(value), ha="center", va=va, fontsize=8.0)
    style_axes(ax2, grid_axis="y")
    add_panel_label(ax2, "c")

    fig.subplots_adjust(left=0.055, right=0.985, wspace=0.34, bottom=0.17)
    save_figure(fig, out_path)
    plt.close(fig)


def _plot_daily_profile(pid: pd.DataFrame, rl: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(14.8, 4.0), sharex=True)
    ax0, ax1, ax2, ax3 = axes

    def profile(trace: pd.DataFrame) -> pd.DataFrame:
        temp = trace.copy()
        temp["hour"] = temp["datetime"].dt.hour + temp["datetime"].dt.minute / 60.0
        return temp.groupby("hour", as_index=False).agg(
            I1=("I1", "mean"),
            I2=("I2", "mean"),
            RH_pct=("RH_pct", "mean"),
            VPD_kPa=("VPD_kPa", "mean"),
            dehum_kW=("P_dehum_kW", "mean"),
        )

    pid_p = profile(pid)
    rl_p = profile(rl)
    ax0.plot(pid_p["hour"], pid_p["I1"], color=PID_COLOR, label=r"PID $I_1$")
    ax0.plot(pid_p["hour"], pid_p["I2"], color=PID_COLOR, linestyle="--", label=r"PID $I_2$")
    ax0.plot(rl_p["hour"], rl_p["I1"], color=RL_COLOR, label=r"Residual-PID SAC $I_1$")
    ax0.plot(rl_p["hour"], rl_p["I2"], color=RL_COLOR, linestyle="--", label=r"Residual-PID SAC $I_2$")
    ax0.set_xlabel("Hour of day")
    ax0.set_ylabel(r"Mean PPFD ($\mu$mol m$^{-2}$ s$^{-1}$)")
    ax0.set_xlim(0, 24)
    ax0.set_xticks(np.arange(0, 25, 4))
    ax0.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2, fontsize=6.8, frameon=False)

    ax1.plot(pid_p["hour"], pid_p["RH_pct"], color=PID_COLOR, label="PID RH")
    ax1.plot(rl_p["hour"], rl_p["RH_pct"], color=RL_COLOR, label="Residual-PID SAC RH")
    ax1.set_xlabel("Hour of day")
    ax1.set_ylabel("Relative humidity (%)")
    ax1.set_xlim(0, 24)
    ax1.set_xticks(np.arange(0, 25, 4))
    ax1.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), fontsize=6.8, frameon=False)

    ax2.plot(pid_p["hour"], pid_p["VPD_kPa"], color=PID_COLOR, label="PID VPD")
    ax2.plot(rl_p["hour"], rl_p["VPD_kPa"], color=RL_COLOR, label="Residual-PID SAC VPD")
    ax2.set_xlabel("Hour of day")
    ax2.set_ylabel("VPD (kPa)")
    ax2.set_xlim(0, 24)
    ax2.set_xticks(np.arange(0, 25, 4))
    ax2.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), fontsize=6.8, frameon=False)

    ax3.plot(pid_p["hour"], pid_p["dehum_kW"], color=PID_COLOR, label="PID")
    ax3.plot(rl_p["hour"], rl_p["dehum_kW"], color=RL_COLOR, label="Residual-PID SAC")
    ax3.set_xlabel("Hour of day")
    ax3.set_ylabel("Mean dehumidification power (kW)")
    ax3.set_xlim(0, 24)
    ax3.set_xticks(np.arange(0, 25, 4))
    ax3.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), fontsize=6.8, frameon=False)

    for idx, ax in enumerate(axes):
        style_axes(ax, grid_axis="y")
        add_panel_label(ax, list("abcd")[idx])
    fig.subplots_adjust(wspace=0.36, top=0.82)
    save_figure(fig, out_path)
    plt.close(fig)


def build_humidity_story() -> None:
    apply_academic_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pid_row = _load_schedule_row(PID_DIR, "pid_exact")
    rl_row = _load_schedule_row(RL_DIR, "rl_exact")
    pid = _load_trace(PID_DIR)
    rl = _load_trace(RL_DIR)
    delta = _metric_summary(pid_row, rl_row, pid, rl)
    _plot_mechanism_overview(pid, rl, delta, OUT_DIR / "paper_humidity_stress_mechanism.png")
    _plot_annual_water_energy(pid, rl, delta, OUT_DIR / "paper_humidity_stress_annual_balance.png")
    _plot_daily_profile(pid, rl, OUT_DIR / "paper_humidity_stress_daily_profile.png")


if __name__ == "__main__":
    build_humidity_story()
