from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper" / "figures_system_validation_20260428"

STYLE_DIR = Path(r"C:\Users\29341\Desktop\code_0420\lettuce_control_drl_new_0422\experiments")
sys.path.insert(0, str(STYLE_DIR))

from figure_style_academic import (  # noqa: E402
    COLORS,
    add_panel_label,
    apply_academic_style,
    save_figure,
    style_axes,
)


CURRENT_PID_DIR = Path(
    r"C:\Users\29341\Desktop\fsdownload\results"
    r"\exp02_exact_pid_baseline_pp16h_i1250_i2300_pm_constant_e0p74_co20p54_l20"
)
LEGACY_PPFD_DIR = Path(
    r"C:\Users\29341\Desktop\code_0420\lettuce_control_drl_new_xiu_PP_PPFD\results"
)

CURRENT_DEFAULT_TRACE = CURRENT_PID_DIR / "detailed_traces" / "t1-14__t2-14__N1-20__rho2-36.csv"
CURRENT_SUMMARY_CSV = CURRENT_PID_DIR / "pid_exact_schedule_results.csv"

DENSITY_CASES = {
    "Low density": CURRENT_PID_DIR / "detailed_traces" / "t1-14__t2-14__N1-16__rho2-20.csv",
    "Default": CURRENT_PID_DIR / "detailed_traces" / "t1-14__t2-14__N1-20__rho2-36.csv",
    "High density": CURRENT_PID_DIR / "detailed_traces" / "t1-14__t2-14__N1-20__rho2-48.csv",
}

LIGHT_CASES = {
    "100+150": LEGACY_PPFD_DIR / "pid_ppfd_min" / "pid_2024-01-01_d32_dt600_manual_light_pp16h.csv",
    "150+200": LEGACY_PPFD_DIR / "pid_default_32d_15_30" / "pid_2024-01-01_d32_dt600_manual_light_pp16h.csv",
    "250+300": LEGACY_PPFD_DIR / "pid_default_32d_25_30" / "pid_2024-01-01_d32_dt600_manual_light_pp16h.csv",
}

DENSITY_COLORS = {
    "Low density": COLORS["sky"],
    "Default": COLORS["navy"],
    "High density": COLORS["brick"],
}
LIGHT_COLORS = {
    "100+150": COLORS["sky"],
    "150+200": COLORS["teal"],
    "250+300": COLORS["gold"],
}


def _prepare_trace(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    if "biomass_dense_kg_m2" not in df.columns and "biomass_seedling_kg_m2" in df.columns:
        df["biomass_dense_kg_m2"] = df["biomass_seedling_kg_m2"]
    if "biomass_finishing_kg_m2" not in df.columns and "biomass_transplant_kg_m2" in df.columns:
        df["biomass_finishing_kg_m2"] = df["biomass_transplant_kg_m2"]
    if "dehum_removed_kg" not in df.columns:
        df["dehum_removed_kg"] = np.nan
    if "env_condensation_removed_kg" not in df.columns:
        df["env_condensation_removed_kg"] = np.nan
    if "VPD_kPa" not in df.columns:
        df["VPD_kPa"] = np.nan
    step_h = float(df["step_size_s"].iloc[0]) / 3600.0
    df["harvest_fw_kg_step"] = df["harvest_fresh_mass_equiv_g"] / 1000.0
    df["cum_harvest_fw_kg"] = df["harvest_fw_kg_step"].cumsum()
    df["cum_transp_kg"] = df["E_transp_kg"].cumsum()
    df["cum_dehum_removed_kg"] = df["dehum_removed_kg"].fillna(0.0).cumsum()
    df["cum_led_kwh"] = (df["P_LED_total_kW"] * step_h).cumsum()
    df["cum_hvac_kwh"] = (df["P_HVAC_kW"] * step_h).cumsum()
    df["cum_dehum_kwh"] = (df["P_dehum_kW"] * step_h).cumsum()
    df["cum_co2_kwh"] = (df["P_CO2_kW"] * step_h).cumsum()
    df["m_dehum_umol"] = df["m_dehum"] * 1e6
    df["u_co2_umol"] = df["u_CO2"] * 1e6
    return df


def _hourly_slice(df: pd.DataFrame, days: float = 32.0) -> pd.DataFrame:
    short = df.loc[df["elapsed_d"] <= float(days)].copy()
    agg_last = {
        "elapsed_d",
        "I1",
        "I2",
        "T_in",
        "T_out",
        "RH_pct",
        "RH_out_pct",
        "C_ppm",
        "C_out_ppm",
        "Q_HVAC",
        "m_dehum_umol",
        "u_co2_umol",
        "P_dehum_kW",
        "biomass_dense_kg_m2",
        "biomass_finishing_kg_m2",
        "cum_harvest_fw_kg",
        "cum_transp_kg",
        "cum_dehum_removed_kg",
    }
    agg_map = {}
    for col in short.columns:
        if col == "datetime":
            continue
        if col in agg_last:
            agg_map[col] = "last"
        elif pd.api.types.is_numeric_dtype(short[col]):
            agg_map[col] = "mean"
    hourly = short.set_index("datetime").resample("1h").agg(agg_map).dropna(how="all").reset_index()
    hourly["elapsed_d"] = (hourly["datetime"] - hourly["datetime"].iloc[0]).dt.total_seconds() / 86400.0
    return hourly


def _load_current_summary() -> pd.DataFrame:
    df = pd.read_csv(CURRENT_SUMMARY_CSV).copy()
    for col in ["t1", "t2", "N1", "rho2", "net_profit", "harvest_fresh_kg", "energy_kwh", "cost_per_kg", "avg_harvest_fresh_g_per_plant", "total_harvests"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _default_annual_metrics(summary_df: pd.DataFrame, trace_df: pd.DataFrame) -> pd.DataFrame:
    row = summary_df.loc[
        (summary_df["t1"] == 14) & (summary_df["t2"] == 14) & (summary_df["N1"] == 20) & (summary_df["rho2"] == 36)
    ].iloc[0]
    step_h = float(trace_df["step_size_s"].iloc[0]) / 3600.0
    led_mwh = float((trace_df["P_LED_total_kW"] * step_h).sum() / 1000.0)
    hvac_mwh = float((trace_df["P_HVAC_kW"] * step_h).sum() / 1000.0)
    dehum_mwh = float((trace_df["P_dehum_kW"] * step_h).sum() / 1000.0)
    co2_mwh = float((trace_df["P_CO2_kW"] * step_h).sum() / 1000.0)
    total_mwh = led_mwh + hvac_mwh + dehum_mwh + co2_mwh
    rows = [
        ("Net profit", float(row["net_profit"]), "RMB y^-1"),
        ("Fresh yield", float(row["harvest_fresh_kg"]), "kg y^-1"),
        ("Electricity use", float(row["energy_kwh"]), "kWh y^-1"),
        ("Fresh-mass cost", float(row["cost_per_kg"]), "RMB kg^-1"),
        ("Mean fresh mass per plant", float(row["avg_harvest_fresh_g_per_plant"]), "g plant^-1"),
        ("Annual harvest count", float(row["total_harvests"]), "count y^-1"),
        ("Mean indoor temperature", float(trace_df["T_in"].mean()), "degC"),
        ("Mean indoor relative humidity", float(trace_df["RH_pct"].mean()), "%"),
        ("Mean indoor CO2", float(trace_df["C_ppm"].mean()), "ppm"),
        ("Mean VPD", float(trace_df["VPD_kPa"].mean()), "kPa"),
        ("LED electricity", led_mwh, "MWh y^-1"),
        ("HVAC electricity", hvac_mwh, "MWh y^-1"),
        ("Dehumidification electricity", dehum_mwh, "MWh y^-1"),
        ("CO2 electricity", co2_mwh, "MWh y^-1"),
        ("LED share", 100.0 * led_mwh / total_mwh, "%"),
        ("HVAC share", 100.0 * hvac_mwh / total_mwh, "%"),
        ("Dehumidification share", 100.0 * dehum_mwh / total_mwh, "%"),
        ("Annual transpiration", float(trace_df["E_transp_kg"].sum()), "kg y^-1"),
        ("Annual dehumidified water", float(trace_df["dehum_removed_kg"].sum()), "kg y^-1"),
    ]
    return pd.DataFrame(rows, columns=["metric", "value", "unit"])


def _density_annual_metrics(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, path in DENSITY_CASES.items():
        trace_df = _prepare_trace(path)
        file_name = path.name
        parts = file_name.replace(".csv", "").split("__")
        t1 = int(parts[0].split("-")[1])
        t2 = int(parts[1].split("-")[1])
        n1 = int(parts[2].split("-")[1])
        rho2 = int(parts[3].split("-")[1])
        row = summary_df.loc[
            (summary_df["t1"] == t1)
            & (summary_df["t2"] == t2)
            & (summary_df["N1"] == n1)
            & (summary_df["rho2"] == rho2)
        ].iloc[0]
        step_h = float(trace_df["step_size_s"].iloc[0]) / 3600.0
        rows.append(
            {
                "case": label,
                "net_profit_rmb": float(row["net_profit"]),
                "harvest_fresh_kg": float(row["harvest_fresh_kg"]),
                "energy_kwh": float(row["energy_kwh"]),
                "cost_per_kg": float(row["cost_per_kg"]),
                "avg_harvest_fresh_g_per_plant": float(row["avg_harvest_fresh_g_per_plant"]),
                "total_harvests": float(row["total_harvests"]),
                "mean_temp_c": float(trace_df["T_in"].mean()),
                "mean_rh_pct": float(trace_df["RH_pct"].mean()),
                "mean_co2_ppm": float(trace_df["C_ppm"].mean()),
                "transp_kg": float(trace_df["E_transp_kg"].sum()),
                "dehum_removed_kg": float(trace_df["dehum_removed_kg"].sum()),
                "dehum_mwh": float((trace_df["P_dehum_kW"] * step_h).sum() / 1000.0),
                "led_mwh": float((trace_df["P_LED_total_kW"] * step_h).sum() / 1000.0),
                "hvac_mwh": float((trace_df["P_HVAC_kW"] * step_h).sum() / 1000.0),
            }
        )
    return pd.DataFrame(rows)


def _light_32d_metrics() -> pd.DataFrame:
    rows = []
    for label, path in LIGHT_CASES.items():
        trace_df = _prepare_trace(path)
        trace_df = trace_df.loc[trace_df["elapsed_d"] <= 32.0].copy()
        step_h = float(trace_df["step_size_s"].iloc[0]) / 3600.0
        rows.append(
            {
                "case": label,
                "mean_temp_c": float(trace_df["T_in"].mean()),
                "mean_rh_pct": float(trace_df["RH_pct"].mean()),
                "mean_co2_ppm": float(trace_df["C_ppm"].mean()),
                "final_dense_biomass_kg_m2": float(trace_df["biomass_dense_kg_m2"].iloc[-1]),
                "final_finishing_biomass_kg_m2": float(trace_df["biomass_finishing_kg_m2"].iloc[-1]),
                "cum_harvest_fw_kg": float(trace_df["harvest_fw_kg_step"].sum()),
                "cum_led_kwh": float((trace_df["P_LED_total_kW"] * step_h).sum()),
                "cum_hvac_kwh": float((trace_df["P_HVAC_kW"] * step_h).sum()),
                "cum_dehum_kwh": float((trace_df["P_dehum_kW"] * step_h).sum()),
                "cum_transp_kg": float(trace_df["E_transp_kg"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _plot_default_dashboard(trace_df: pd.DataFrame, out_path: Path) -> None:
    hourly = _hourly_slice(trace_df, days=32.0)
    fig, axes = plt.subplots(2, 4, figsize=(13.6, 6.6))
    axes = axes.flatten()

    axes[0].plot(hourly["elapsed_d"], hourly["I1"], color=COLORS["sky"], label=r"Dense-zone $I_1$")
    axes[0].plot(hourly["elapsed_d"], hourly["I2"], color=COLORS["gold"], label=r"Finishing-zone $I_2$")
    axes[0].set_ylabel(r"PPFD ($\mu$mol m$^{-2}$ s$^{-1}$)")

    axes[1].plot(hourly["elapsed_d"], hourly["T_in"], color=COLORS["navy"], label="Indoor")
    axes[1].plot(hourly["elapsed_d"], hourly["T_out"], color=COLORS["gray"], linestyle="--", label="Outdoor")
    axes[1].set_ylabel(r"Temperature ($^\circ$C)")

    axes[2].plot(hourly["elapsed_d"], hourly["RH_pct"], color=COLORS["teal"], label="Indoor")
    axes[2].plot(hourly["elapsed_d"], hourly["RH_out_pct"], color=COLORS["gray"], linestyle="--", label="Outdoor")
    axes[2].set_ylabel("Relative humidity (%)")

    axes[3].plot(hourly["elapsed_d"], hourly["C_ppm"], color=COLORS["green"], label="Indoor")
    axes[3].plot(hourly["elapsed_d"], hourly["C_out_ppm"], color=COLORS["gray"], linestyle="--", label="Outdoor")
    axes[3].set_ylabel(r"CO$_2$ (ppm)")

    axes[4].plot(hourly["elapsed_d"], hourly["biomass_dense_kg_m2"], color=COLORS["sky"], label="Dense zone")
    axes[4].plot(hourly["elapsed_d"], hourly["biomass_finishing_kg_m2"], color=COLORS["gold"], label="Finishing zone")
    axes[4].set_ylabel(r"Biomass (kg m$^{-2}$)")

    axes[5].plot(hourly["elapsed_d"], hourly["Q_HVAC"], color=COLORS["blue"])
    axes[5].set_ylabel(r"HVAC command (W m$^{-2}$)")

    axes[6].plot(hourly["elapsed_d"], hourly["m_dehum_umol"], color=COLORS["brick"])
    axes[6].set_ylabel(r"Dehumidifier command ($10^{-6}$ kg s$^{-1}$)")

    axes[7].plot(hourly["elapsed_d"], hourly["u_co2_umol"], color=COLORS["olive"])
    axes[7].set_ylabel(r"CO$_2$ supply ($10^{-6}$ kg s$^{-1}$)")

    labels = list("abcdefgh")
    for idx, ax in enumerate(axes):
        style_axes(ax, grid_axis="y")
        add_panel_label(ax, labels[idx])
        ax.set_xlim(0.0, 32.0)
        if idx >= 4:
            ax.set_xlabel("Elapsed time (d)")
    axes[0].legend(loc="upper left", ncol=2, fontsize=7.8)
    axes[1].legend(loc="upper right", ncol=2, fontsize=7.8)
    axes[2].legend(loc="upper right", ncol=2, fontsize=7.8)
    axes[3].legend(loc="upper right", ncol=2, fontsize=7.8)
    axes[4].legend(loc="upper left", ncol=2, fontsize=7.8)

    save_figure(fig, out_path)
    plt.close(fig)


def _plot_density_response(case_frames: dict[str, pd.DataFrame], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(13.8, 6.6))
    axes = axes.flatten()

    for label, df in case_frames.items():
        color = DENSITY_COLORS[label]
        axes[0].plot(df["elapsed_d"], df["cum_harvest_fw_kg"], color=color, label=label)
        axes[1].plot(df["elapsed_d"], df["biomass_dense_kg_m2"], color=color, label=label)
        axes[2].plot(df["elapsed_d"], df["biomass_finishing_kg_m2"], color=color, label=label)
        axes[3].plot(df["elapsed_d"], df["cum_transp_kg"], color=color, label=label)
        axes[4].plot(df["elapsed_d"], df["T_in"], color=color, label=label)
        axes[5].plot(df["elapsed_d"], df["RH_pct"], color=color, label=label)
        axes[6].plot(df["elapsed_d"], df["Q_HVAC"], color=color, label=label)
        axes[7].plot(df["elapsed_d"], df["m_dehum_umol"], color=color, label=label)

    ylabels = [
        r"Cumulative harvest (kg)",
        r"Dense-zone biomass (kg m$^{-2}$)",
        r"Finishing-zone biomass (kg m$^{-2}$)",
        r"Cumulative transpiration (kg)",
        r"Indoor temperature ($^\circ$C)",
        "Indoor relative humidity (%)",
        r"HVAC command (W m$^{-2}$)",
        r"Dehumidifier command ($10^{-6}$ kg s$^{-1}$)",
    ]

    labels = list("abcdefgh")
    for idx, ax in enumerate(axes):
        ax.set_ylabel(ylabels[idx])
        style_axes(ax, grid_axis="y")
        add_panel_label(ax, labels[idx])
        ax.set_xlim(0.0, 32.0)
        if idx >= 4:
            ax.set_xlabel("Elapsed time (d)")
    axes[0].legend(loc="upper left", ncol=1, fontsize=7.8)

    save_figure(fig, out_path)
    plt.close(fig)


def _plot_light_response(case_frames: dict[str, pd.DataFrame], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(13.8, 6.6))
    axes = axes.flatten()

    for label, df in case_frames.items():
        color = LIGHT_COLORS[label]
        axes[0].plot(df["elapsed_d"], df["I1"], color=color, label=label)
        axes[1].plot(df["elapsed_d"], df["I2"], color=color, label=label)
        axes[2].plot(df["elapsed_d"], df["biomass_dense_kg_m2"], color=color, label=label)
        axes[3].plot(df["elapsed_d"], df["biomass_finishing_kg_m2"], color=color, label=label)
        axes[4].plot(df["elapsed_d"], df["cum_harvest_fw_kg"], color=color, label=label)
        axes[5].plot(df["elapsed_d"], df["T_in"], color=color, label=label)
        axes[6].plot(df["elapsed_d"], df["RH_pct"], color=color, label=label)
        axes[7].plot(df["elapsed_d"], df["cum_transp_kg"], color=color, label=label)

    ylabels = [
        r"Dense-zone PPFD ($\mu$mol m$^{-2}$ s$^{-1}$)",
        r"Finishing-zone PPFD ($\mu$mol m$^{-2}$ s$^{-1}$)",
        r"Dense-zone biomass (kg m$^{-2}$)",
        r"Finishing-zone biomass (kg m$^{-2}$)",
        r"Cumulative harvest (kg)",
        r"Indoor temperature ($^\circ$C)",
        "Indoor relative humidity (%)",
        r"Cumulative transpiration (kg)",
    ]

    labels = list("abcdefgh")
    for idx, ax in enumerate(axes):
        ax.set_ylabel(ylabels[idx])
        style_axes(ax, grid_axis="y")
        add_panel_label(ax, labels[idx])
        ax.set_xlim(0.0, 32.0)
        if idx >= 4:
            ax.set_xlabel("Elapsed time (d)")
    axes[0].legend(loc="upper left", ncol=1, fontsize=7.8)

    save_figure(fig, out_path)
    plt.close(fig)


def _plot_summary_metrics(
    default_metrics: pd.DataFrame,
    density_metrics: pd.DataFrame,
    light_metrics: pd.DataFrame,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.2))
    ax0, ax1, ax2 = axes

    energy_order = [
        ("LED electricity", COLORS["gold"]),
        ("HVAC electricity", COLORS["blue"]),
        ("Dehumidification electricity", COLORS["teal"]),
    ]
    left = 0.0
    for metric, color in energy_order:
        value = float(default_metrics.loc[default_metrics["metric"] == metric, "value"].iloc[0])
        ax0.barh(["Default annual electricity"], [value], left=left, color=color, edgecolor="none", height=0.55, label=metric.replace(" electricity", ""))
        left += value
    ax0.set_xlabel("Electricity use (MWh y$^{-1}$)")
    style_axes(ax0, grid_axis="x")
    ax0.legend(loc="lower right", fontsize=7.7)
    add_panel_label(ax0, "a")

    density_plot = density_metrics.set_index("case").loc[["Low density", "Default", "High density"]]
    density_x = np.arange(len(density_plot))
    density_series = {
        "Fresh yield": density_plot["harvest_fresh_kg"] / density_plot.loc["Default", "harvest_fresh_kg"],
        "Mean plant FW": density_plot["avg_harvest_fresh_g_per_plant"] / density_plot.loc["Default", "avg_harvest_fresh_g_per_plant"],
        "Transpiration": density_plot["transp_kg"] / density_plot.loc["Default", "transp_kg"],
        "Dehumidification": density_plot["dehum_mwh"] / density_plot.loc["Default", "dehum_mwh"],
    }
    density_metric_colors = {
        "Fresh yield": COLORS["navy"],
        "Mean plant FW": COLORS["green"],
        "Transpiration": COLORS["gold"],
        "Dehumidification": COLORS["brick"],
    }
    for label, series in density_series.items():
        ax1.plot(density_x, series.values, marker="o", markersize=4.5, linewidth=1.7, color=density_metric_colors[label], label=label)
    ax1.axhline(1.0, color=COLORS["gray"], linewidth=1.0, linestyle="--")
    ax1.set_xticks(density_x)
    ax1.set_xticklabels(["Low", "Default", "High"])
    ax1.set_ylabel("Relative to default")
    style_axes(ax1, grid_axis="y")
    ax1.legend(loc="upper left", ncol=2, fontsize=7.4)
    add_panel_label(ax1, "b")

    light_plot = light_metrics.set_index("case").loc[["100+150", "150+200", "250+300"]]
    light_x = np.arange(len(light_plot))
    light_series = {
        "Finishing biomass": light_plot["final_finishing_biomass_kg_m2"] / light_plot.loc["250+300", "final_finishing_biomass_kg_m2"],
        "Cumulative harvest": light_plot["cum_harvest_fw_kg"] / light_plot.loc["250+300", "cum_harvest_fw_kg"],
        "Transpiration": light_plot["cum_transp_kg"] / light_plot.loc["250+300", "cum_transp_kg"],
        "Dehumidification": light_plot["cum_dehum_kwh"] / light_plot.loc["250+300", "cum_dehum_kwh"],
    }
    light_metric_colors = {
        "Finishing biomass": COLORS["navy"],
        "Cumulative harvest": COLORS["green"],
        "Transpiration": COLORS["gold"],
        "Dehumidification": COLORS["brick"],
    }
    for label, series in light_series.items():
        ax2.plot(light_x, series.values, marker="o", markersize=4.5, linewidth=1.7, color=light_metric_colors[label], label=label)
    ax2.axhline(1.0, color=COLORS["gray"], linewidth=1.0, linestyle="--")
    ax2.set_xticks(light_x)
    ax2.set_xticklabels(["100+150", "150+200", "250+300"])
    ax2.set_ylabel("Relative to 250+300")
    style_axes(ax2, grid_axis="y")
    ax2.legend(loc="upper left", ncol=2, fontsize=7.2)
    add_panel_label(ax2, "c")

    save_figure(fig, out_path)
    plt.close(fig)


def build_validation_story() -> None:
    apply_academic_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_df = _load_current_summary()
    default_trace = _prepare_trace(CURRENT_DEFAULT_TRACE)
    default_metrics = _default_annual_metrics(summary_df, default_trace)
    default_metrics.to_csv(OUT_DIR / "validation_default_annual_metrics.csv", index=False)

    density_metrics = _density_annual_metrics(summary_df)
    density_metrics.to_csv(OUT_DIR / "validation_density_annual_metrics.csv", index=False)

    light_metrics = _light_32d_metrics()
    light_metrics.to_csv(OUT_DIR / "validation_light_32d_metrics.csv", index=False)

    density_frames = {label: _hourly_slice(_prepare_trace(path), days=32.0) for label, path in DENSITY_CASES.items()}
    light_frames = {label: _hourly_slice(_prepare_trace(path), days=32.0) for label, path in LIGHT_CASES.items()}

    _plot_default_dashboard(default_trace, OUT_DIR / "paper_validation_default_dashboard.png")
    _plot_density_response(density_frames, OUT_DIR / "paper_validation_density_response.png")
    _plot_light_response(light_frames, OUT_DIR / "paper_validation_light_response.png")
    _plot_summary_metrics(default_metrics, density_metrics, light_metrics, OUT_DIR / "paper_validation_summary_metrics.png")


if __name__ == "__main__":
    build_validation_story()
