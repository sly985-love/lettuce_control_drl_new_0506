# -*- coding: utf-8 -*-
"""Build compact figures and summary tables for climate-only residual PID RL results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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
PID_RESULTS_ROOT = Path(r"C:\Users\29341\Desktop\fsdownload\results")
RESIDUAL_RESULTS_ROOT = Path(r"C:\Users\29341\Desktop\fsdownload\results_residual_pid_sac")
GATED_RESULTS_ROOT = Path(r"C:\Users\29341\Desktop\fsdownload\results_gated_residual_pid_sac")
CLIMATE_RESULTS_ROOT = Path(r"C:\Users\29341\Desktop\fsdownload\result_climate_only_residual_pid_sac")
OUT_DIR = ROOT / "paper" / "figures_result_review_climate_20260427"

DEFAULT_SCHEDULE_KEY = "t1=14|t2=14|N1=20|rho2=36"
REASONABLE_SCHEDULE_KEY = "t1=13|t2=13|N1=15|rho2=24"
GATED_FAVORED_SCHEDULE_KEY = "t1=13|t2=13|N1=16|rho2=23"
AGGRESSIVE_SCHEDULE_KEY = "t1=15|t2=13|N1=15|rho2=23"
UPPER_VARS = ["t1", "t2", "N1", "rho2"]
METRICS = ["net_profit", "harvest_fresh_kg", "energy_kwh", "cost_per_kg"]
METRIC_LABELS = {
    "net_profit": "Net profit",
    "harvest_fresh_kg": "Fresh yield",
    "energy_kwh": "Electricity use",
    "cost_per_kg": "Cost per kg",
}
VAR_LABELS = {
    "t1": r"Dense-stage duration, $t_1$",
    "t2": r"Finishing-stage duration, $t_2$",
    "N1": r"Dense-zone boards, $N_1$",
    "rho2": r"Finishing density, $\rho_2$",
}
SCENARIO_COLORS = {
    "Const-L20": COLORS["navy"],
    "Const-L40": COLORS["teal"],
    "TOU-L40": COLORS["gold"],
}
CONTROLLER_SPECS = [
    {"key": "pid", "label": "PID", "color": COLORS["gray"]},
    {"key": "climate", "label": "Climate-only", "color": COLORS["navy"]},
    {"key": "gated", "label": "Gated residual", "color": COLORS["teal"]},
    {"key": "residual", "label": "Full residual", "color": COLORS["brick"]},
]
ENERGY_COMPONENT_COLORS = {
    "led_mwh": COLORS["gold"],
    "hvac_mwh": COLORS["teal"],
    "dehum_mwh": COLORS["navy"],
}
DELTA_COLORS = {
    "profit_gain_pct": COLORS["navy"],
    "harvest_gain_pct": COLORS["teal"],
    "energy_change_pct": COLORS["gold"],
    "cost_reduction_pct": COLORS["plum"],
}

SCENARIOS = [
    {
        "label": "Const-L20",
        "climate_name": "exp03_exact_climaterespid_rl_pp16h_constant_e0p74_co20p54_l20",
        "residual_name": "exp03_exact_rl_baseline_pp16h_constant_e0p74_co20p54_l20",
        "gated_name": "exp03_exact_gatedrespid_rl_pp16h_constant_e0p74_co20p54_l20",
        "pid_name": "exp02_exact_pid_baseline_pp16h_i1250_i2300_pm_constant_e0p74_co20p54_l20",
    },
    {
        "label": "Const-L40",
        "climate_name": "exp03_exact_climaterespid_rl_pp16h_constant_e0p74_co20p54_l40",
        "residual_name": "exp03_exact_rl_baseline_pp16h_constant_e0p74_co20p54_l40",
        "gated_name": "exp03_exact_gatedrespid_rl_pp16h_constant_e0p74_co20p54_l40",
        "pid_name": "exp02_exact_pid_baseline_pp16h_i1250_i2300_pm_constant_e0p74_co20p54_l40",
    },
    {
        "label": "TOU-L40",
        "climate_name": "exp03_climaterespid_rl_pp16h_tou_zhejiang_lt1kv_l40",
        "residual_name": "exp03_exact_rl_baseline_pp16h_tou_zhejiang_lt1kv_l40",
        "gated_name": "exp03_exact_gatedrespid_rl_pp16h_tou_zhejiang_lt1kv_l40",
        "pid_name": "exp02_exact_pid_baseline_pp16h_i1250_i2300_pm_tou_zhejiang_lt1kv_l40",
    },
]


def _schedule_key_to_trace_name(schedule_key: str) -> str:
    parts = dict(part.split("=") for part in schedule_key.split("|"))
    return (
        f"t1-{int(float(parts['t1']))}"
        f"__t2-{int(float(parts['t2']))}"
        f"__N1-{int(float(parts['N1']))}"
        f"__rho2-{int(float(parts['rho2']))}.csv"
    )


def _format_schedule_tuple(row: pd.Series) -> str:
    return (
        f"({int(row['t1'])}, {int(row['t2'])}, "
        f"{int(row['N1'])}, {int(float(row['rho2']))})"
    )


def _load_schedule_df(exp_dir: Path, prefix: str) -> pd.DataFrame:
    candidate_prefixes = [prefix]
    if not prefix.endswith("_exact"):
        candidate_prefixes.append(f"{prefix}_exact")

    for candidate_prefix in candidate_prefixes:
        for candidate in [
            exp_dir / f"{candidate_prefix}_schedule_results_ranked.csv",
            exp_dir / f"{candidate_prefix}_schedule_results.csv",
        ]:
            if candidate.exists():
                df = pd.read_csv(candidate)
                if "rank_valid_profit" not in df.columns:
                    df = df.sort_values("net_profit", ascending=False).reset_index(drop=True)
                    df["rank_valid_profit"] = np.arange(1, len(df) + 1)
                return df

        shard_paths = sorted(exp_dir.glob(f"{candidate_prefix}_schedule_results.shard_*.csv"))
        if shard_paths:
            df = pd.concat([pd.read_csv(path) for path in shard_paths], ignore_index=True)
            df = df.sort_values("net_profit", ascending=False).reset_index(drop=True)
            df["rank_valid_profit"] = np.arange(1, len(df) + 1)
            return df

        ranked_shards = sorted(exp_dir.glob(f"{candidate_prefix}_schedule_results_ranked.shard_*.csv"))
        if ranked_shards:
            return pd.concat([pd.read_csv(path) for path in ranked_shards], ignore_index=True)

    raise FileNotFoundError(f"No result csv found in {exp_dir}")


def _load_trace_df(exp_dir: Path, schedule_key: str, usecols: list[str] | None = None) -> pd.DataFrame:
    trace_path = exp_dir / "detailed_traces" / _schedule_key_to_trace_name(schedule_key)
    if not trace_path.exists():
        raise FileNotFoundError(trace_path)
    return pd.read_csv(trace_path, usecols=usecols)


def _compute_eta2(df: pd.DataFrame, variable: str, metric: str) -> float:
    y = df[metric].astype(float).to_numpy()
    y_bar = float(np.mean(y))
    ss_total = float(np.sum((y - y_bar) ** 2))
    if ss_total <= 0.0:
        return 0.0
    grouped = df.groupby(variable)[metric].agg(["mean", "count"]).reset_index()
    ss_between = float(np.sum(grouped["count"] * (grouped["mean"] - y_bar) ** 2))
    return ss_between / ss_total


def _hourly_profile(trace_df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    df = trace_df.copy()
    dt = pd.to_datetime(df["datetime"])
    df["hour"] = dt.dt.hour + dt.dt.minute / 60.0
    return df.groupby("hour", as_index=False)[value_cols].mean().sort_values("hour")


def _energy_from_trace(trace_df: pd.DataFrame, power_col: str) -> float:
    return float((trace_df[power_col] * trace_df["step_size_s"] / 3600.0).sum() / 1000.0)


def _trace_summary(trace_df: pd.DataFrame) -> dict[str, float]:
    harvest_mask = trace_df["harvest_event"] == 1
    return {
        "I1_mean": float(trace_df["I1"].mean()),
        "I2_mean": float(trace_df["I2"].mean()),
        "I1_unique_n": int(trace_df["I1"].nunique()),
        "I2_unique_n": int(trace_df["I2"].nunique()),
        "T_mean": float(trace_df["T_in"].mean()),
        "RH_mean": float(trace_df["RH_pct"].mean()),
        "VPD_mean": float(trace_df["VPD_kPa"].mean()),
        "led_mwh": _energy_from_trace(trace_df, "P_LED_total_kW"),
        "hvac_mwh": _energy_from_trace(trace_df, "P_HVAC_kW"),
        "dehum_mwh": _energy_from_trace(trace_df, "P_dehum_kW"),
        "transp_kg": float(trace_df["E_transp_kg"].sum()),
        "dehum_removed_kg": float(trace_df["dehum_removed_kg"].sum()),
        "avg_fw_g_per_plant": float(
            trace_df.loc[harvest_mask, "harvest_mean_fresh_mass_per_plant_g"].mean()
        ),
        "harvest_events": int(trace_df["harvest_event"].sum()),
    }


def _collect_integrity_table() -> pd.DataFrame:
    rows = []
    for spec in SCENARIOS:
        exp_dir = CLIMATE_RESULTS_ROOT / spec["climate_name"]
        rows.append(
            {
                "scenario": spec["label"],
                "folder_name": spec["climate_name"],
                "csv_count": len(list(exp_dir.rglob("*.csv"))),
                "json_count": len(list(exp_dir.rglob("*.json"))),
                "png_count": len(list(exp_dir.rglob("*.png"))),
                "trace_count": len(list((exp_dir / "detailed_traces").glob("*.csv"))),
                "shard_result_count": len(list(exp_dir.glob("rl_exact_schedule_results.shard_*.csv"))),
                "shard_summary_count": len(list(exp_dir.glob("rl_exact_schedule_summary.shard_*.json"))),
                "has_merged_csv": (exp_dir / "rl_exact_schedule_results.csv").exists(),
                "has_ranked_csv": (exp_dir / "rl_exact_schedule_results_ranked.csv").exists(),
                "has_merged_json": (exp_dir / "rl_exact_schedule_summary.json").exists(),
            }
        )
    return pd.DataFrame(rows)


def _collect_metadata_nulls(climate_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    meta_cols = [
        "manual_photo_period_hours",
        "manual_I1",
        "manual_I2",
        "light_control_mode",
        "light_segments_per_photoperiod",
        "price_model_type",
        "tou_tariff_scenario",
        "electricity_price",
        "co2_price",
        "lettuce_price_fw",
        "constant_price",
    ]
    rows = []
    for scenario, df in climate_map.items():
        for col in meta_cols:
            non_null_values = [value for value in pd.unique(df[col]) if pd.notna(value)]
            rows.append(
                {
                    "scenario": scenario,
                    "column": col,
                    "all_null": len(non_null_values) == 0,
                    "non_null_values": json.dumps(non_null_values[:8], ensure_ascii=False),
                }
            )
    return pd.DataFrame(rows)


def _collect_climate_summary_rows(
    climate_map: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    delta_rows = []
    for spec in SCENARIOS:
        scenario = spec["label"]
        df = climate_map[scenario]
        best = df.sort_values("net_profit", ascending=False).iloc[0]
        default = df[df["is_default_schedule"] == True].iloc[0]
        worst = df.sort_values("net_profit", ascending=True).iloc[0]

        for status, row in [("best", best), ("default", default), ("worst", worst)]:
            summary_rows.append(
                {
                    "scenario": scenario,
                    "status": status,
                    "schedule_key": row["schedule_key"],
                    "schedule_tuple": _format_schedule_tuple(row),
                    "t1": int(row["t1"]),
                    "t2": int(row["t2"]),
                    "N1": int(row["N1"]),
                    "rho2": float(row["rho2"]),
                    "rank_valid_profit": int(row["rank_valid_profit"]),
                    "net_profit": float(row["net_profit"]),
                    "harvest_fresh_kg": float(row["harvest_fresh_kg"]),
                    "energy_kwh": float(row["energy_kwh"]),
                    "total_cost": float(row["total_cost"]),
                    "cost_per_kg": float(row["cost_per_kg"]),
                    "avg_harvest_fresh_g_per_plant": float(row["avg_harvest_fresh_g_per_plant"]),
                    "total_harvests": int(row["total_harvests"]),
                    "total_transplants": int(row["total_transplants"]),
                }
            )

        delta_rows.append(
            {
                "scenario": scenario,
                "best_schedule_key": best["schedule_key"],
                "default_schedule_key": default["schedule_key"],
                "profit_gain_pct": float((best["net_profit"] / default["net_profit"] - 1.0) * 100.0),
                "harvest_gain_pct": float(
                    (best["harvest_fresh_kg"] / default["harvest_fresh_kg"] - 1.0) * 100.0
                ),
                "energy_change_pct": float((best["energy_kwh"] / default["energy_kwh"] - 1.0) * 100.0),
                "cost_reduction_pct": float((1.0 - best["cost_per_kg"] / default["cost_per_kg"]) * 100.0),
                "avg_fw_per_plant_gain_pct": float(
                    (best["avg_harvest_fresh_g_per_plant"] / default["avg_harvest_fresh_g_per_plant"] - 1.0)
                    * 100.0
                ),
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(delta_rows)


def _collect_eta_table(
    climate_map: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for spec in SCENARIOS:
        scenario = spec["label"]
        df = climate_map[scenario]
        for metric in METRICS:
            for variable in UPPER_VARS:
                rows.append(
                    {
                        "scenario": scenario,
                        "metric": metric,
                        "metric_label": METRIC_LABELS[metric],
                        "variable": variable,
                        "variable_label": VAR_LABELS[variable],
                        "eta2": _compute_eta2(df, variable, metric),
                    }
                )
    eta_df = pd.DataFrame(rows)
    eta_avg_df = (
        eta_df.groupby(["metric", "metric_label", "variable", "variable_label"], as_index=False)["eta2"]
        .mean()
        .rename(columns={"eta2": "eta2_mean"})
    )
    return eta_df, eta_avg_df


def _collect_controller_ablation_rows(
    pid_map: dict[str, pd.DataFrame],
    residual_map: dict[str, pd.DataFrame],
    gated_map: dict[str, pd.DataFrame],
    climate_map: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    same_schedule_rows = []
    controller_maps = {
        "pid": pid_map,
        "residual": residual_map,
        "gated": gated_map,
        "climate": climate_map,
    }
    for spec in SCENARIOS:
        scenario = spec["label"]
        for controller_key, df_map in controller_maps.items():
            df = df_map[scenario]
            best = df.sort_values("net_profit", ascending=False).iloc[0]
            default = df[df["is_default_schedule"] == True].iloc[0]
            for level, row in [("default", default), ("best", best)]:
                rows.append(
                    {
                        "scenario": scenario,
                        "controller": controller_key,
                        "level": level,
                        "schedule_key": row["schedule_key"],
                        "schedule_tuple": _format_schedule_tuple(row),
                        "net_profit": float(row["net_profit"]),
                        "harvest_fresh_kg": float(row["harvest_fresh_kg"]),
                        "energy_kwh": float(row["energy_kwh"]),
                        "cost_per_kg": float(row["cost_per_kg"]),
                        "total_harvests": int(row["total_harvests"]),
                        "rank_valid_profit": int(row["rank_valid_profit"]),
                    }
                )

            for schedule_case, schedule_key in [
                ("default", DEFAULT_SCHEDULE_KEY),
                ("reasonable", REASONABLE_SCHEDULE_KEY),
                ("gated_like", GATED_FAVORED_SCHEDULE_KEY),
                ("aggressive", AGGRESSIVE_SCHEDULE_KEY),
            ]:
                row = df[df["schedule_key"] == schedule_key].iloc[0]
                same_schedule_rows.append(
                    {
                        "scenario": scenario,
                        "controller": controller_key,
                        "schedule_case": schedule_case,
                        "schedule_key": row["schedule_key"],
                        "schedule_tuple": _format_schedule_tuple(row),
                        "net_profit": float(row["net_profit"]),
                        "harvest_fresh_kg": float(row["harvest_fresh_kg"]),
                        "energy_kwh": float(row["energy_kwh"]),
                        "cost_per_kg": float(row["cost_per_kg"]),
                        "total_harvests": int(row["total_harvests"]),
                        "rank_valid_profit": int(row["rank_valid_profit"]),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(same_schedule_rows)


def _collect_mechanism_tables(
    pid_map: dict[str, pd.DataFrame],
    residual_map: dict[str, pd.DataFrame],
    gated_map: dict[str, pd.DataFrame],
    climate_map: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scenario = "Const-L40"
    controller_specs = [
        ("pid", "PID", PID_RESULTS_ROOT / next(spec["pid_name"] for spec in SCENARIOS if spec["label"] == scenario), pid_map[scenario]),
        (
            "climate",
            "Climate-only",
            CLIMATE_RESULTS_ROOT / next(spec["climate_name"] for spec in SCENARIOS if spec["label"] == scenario),
            climate_map[scenario],
        ),
        (
            "gated",
            "Gated residual",
            GATED_RESULTS_ROOT / next(spec["gated_name"] for spec in SCENARIOS if spec["label"] == scenario),
            gated_map[scenario],
        ),
        (
            "residual",
            "Full residual",
            RESIDUAL_RESULTS_ROOT / next(spec["residual_name"] for spec in SCENARIOS if spec["label"] == scenario),
            residual_map[scenario],
        ),
    ]

    hourly_rows = []
    metrics_rows = []
    for controller_key, controller_label, exp_dir, df in controller_specs:
        schedule_row = df[df["schedule_key"] == DEFAULT_SCHEDULE_KEY].iloc[0]
        trace_df = _load_trace_df(exp_dir, DEFAULT_SCHEDULE_KEY)
        hourly_df = _hourly_profile(trace_df, ["I1", "I2", "Q_HVAC", "m_dehum"])
        hourly_df["controller"] = controller_key
        hourly_df["controller_label"] = controller_label
        hourly_rows.append(hourly_df)

        summary = _trace_summary(trace_df)
        metrics_rows.append(
            {
                "controller": controller_key,
                "controller_label": controller_label,
                "net_profit": float(schedule_row["net_profit"]),
                "harvest_fresh_kg": float(schedule_row["harvest_fresh_kg"]),
                "energy_kwh": float(schedule_row["energy_kwh"]),
                "cost_per_kg": float(schedule_row["cost_per_kg"]),
                **summary,
            }
        )

    hourly_df = pd.concat(hourly_rows, ignore_index=True)
    metrics_df = pd.DataFrame(metrics_rows)
    pid_row = metrics_df[metrics_df["controller"] == "pid"].iloc[0]
    rel_rows = []
    metric_map = {
        "transp_kg": "Transpiration",
        "dehum_removed_kg": "Dehumidified water",
        "dehum_mwh": "Dehum electricity",
        "avg_fw_g_per_plant": "Fresh mass per plant",
        "net_profit": "Net profit",
    }
    for _, row in metrics_df.iterrows():
        if row["controller"] == "pid":
            continue
        for metric_key, metric_label in metric_map.items():
            rel_rows.append(
                {
                    "controller": row["controller"],
                    "controller_label": row["controller_label"],
                    "metric_key": metric_key,
                    "metric_label": metric_label,
                    "delta_pct_vs_pid": float((row[metric_key] / pid_row[metric_key] - 1.0) * 100.0),
                }
            )
    return hourly_df, metrics_df, pd.DataFrame(rel_rows)


def _collect_tou_tables(
    residual_map: dict[str, pd.DataFrame],
    gated_map: dict[str, pd.DataFrame],
    climate_map: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rank_shift_rows = []
    comparison_specs = [
        (
            "residual",
            "Full residual",
            residual_map["Const-L40"],
            residual_map["TOU-L40"],
        ),
        (
            "gated",
            "Gated residual",
            gated_map["Const-L40"],
            gated_map["TOU-L40"],
        ),
        (
            "climate",
            "Climate-only",
            climate_map["Const-L40"],
            climate_map["TOU-L40"],
        ),
    ]
    for controller_key, controller_label, const_df, tou_df in comparison_specs:
        merged = const_df[["schedule_key", "rank_valid_profit"]].merge(
            tou_df[["schedule_key", "rank_valid_profit"]],
            on="schedule_key",
            suffixes=("_const", "_tou"),
        )
        shift = (merged["rank_valid_profit_tou"] - merged["rank_valid_profit_const"]).abs()
        rank_shift_rows.append(
            {
                "controller": controller_key,
                "controller_label": controller_label,
                "mean_abs_rank_shift": float(shift.mean()),
                "max_rank_shift": int(shift.max()),
            }
        )

    const_df = climate_map["Const-L40"]
    tou_df = climate_map["TOU-L40"]
    const_row = const_df[const_df["schedule_key"] == AGGRESSIVE_SCHEDULE_KEY].iloc[0]
    tou_row = tou_df[tou_df["schedule_key"] == AGGRESSIVE_SCHEDULE_KEY].iloc[0]
    delta_rows = []
    for metric_key, metric_label in [
        ("harvest_fresh_kg", "Fresh yield"),
        ("energy_kwh", "Electricity use"),
        ("total_cost", "Total cost"),
        ("net_profit", "Net profit"),
        ("cost_per_kg", "Cost per kg"),
    ]:
        delta_rows.append(
            {
                "metric_key": metric_key,
                "metric_label": metric_label,
                "const_value": float(const_row[metric_key]),
                "tou_value": float(tou_row[metric_key]),
                "delta_pct": float((tou_row[metric_key] / const_row[metric_key] - 1.0) * 100.0),
            }
        )

    const_trace = _load_trace_df(
        CLIMATE_RESULTS_ROOT / next(spec["climate_name"] for spec in SCENARIOS if spec["label"] == "Const-L40"),
        AGGRESSIVE_SCHEDULE_KEY,
    )
    tou_trace = _load_trace_df(
        CLIMATE_RESULTS_ROOT / next(spec["climate_name"] for spec in SCENARIOS if spec["label"] == "TOU-L40"),
        AGGRESSIVE_SCHEDULE_KEY,
    )
    for metric_key, metric_label, const_value, tou_value in [
        ("transp_kg", "Transpiration", float(const_trace["E_transp_kg"].sum()), float(tou_trace["E_transp_kg"].sum())),
        ("dehum_mwh", "Dehum electricity", _energy_from_trace(const_trace, "P_dehum_kW"), _energy_from_trace(tou_trace, "P_dehum_kW")),
    ]:
        delta_rows.append(
            {
                "metric_key": metric_key,
                "metric_label": metric_label,
                "const_value": const_value,
                "tou_value": tou_value,
                "delta_pct": float((tou_value / const_value - 1.0) * 100.0),
            }
        )

    const_hourly = _hourly_profile(const_trace, ["I1", "I2", "Q_HVAC", "m_dehum", "elec_price_rmb_kwh"])
    const_hourly["tariff"] = "Constant"
    tou_hourly = _hourly_profile(tou_trace, ["I1", "I2", "Q_HVAC", "m_dehum", "elec_price_rmb_kwh"])
    tou_hourly["tariff"] = "TOU"
    return pd.DataFrame(rank_shift_rows), pd.DataFrame(delta_rows), pd.concat([const_hourly, tou_hourly], ignore_index=True)


def _build_climate_core_summary_figure(
    out_path: Path,
    climate_map: dict[str, pd.DataFrame],
    summary_df: pd.DataFrame,
    delta_df: pd.DataFrame,
) -> None:
    apply_academic_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.3), gridspec_kw={"width_ratios": [1.15, 1.0]})

    ax = axes[0]
    for spec in SCENARIOS:
        scenario = spec["label"]
        df = climate_map[scenario]
        ax.scatter(
            df["total_harvests"],
            df["net_profit"] / 1000.0,
            s=16,
            alpha=0.26,
            color=SCENARIO_COLORS[scenario],
            edgecolors="none",
        )
        for status, marker, edge_color, face_color in [
            ("best", "D", COLORS["ink"], SCENARIO_COLORS[scenario]),
            ("default", "o", COLORS["ink"], "white"),
            ("worst", "s", COLORS["ink"], COLORS["light_gray"]),
        ]:
            row = summary_df[(summary_df["scenario"] == scenario) & (summary_df["status"] == status)].iloc[0]
            ax.scatter(
                [row["total_harvests"]],
                [row["net_profit"] / 1000.0],
                s=48,
                marker=marker,
                facecolor=face_color,
                edgecolor=edge_color,
                linewidth=0.9,
                zorder=5,
            )
    style_axes(ax, grid_axis="both")
    ax.set_xlabel("Annual harvest events")
    ax.set_ylabel(r"Net profit (10$^3$ RMB)")
    ax.set_xlim(15, 380)
    scenario_handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=6.2, color=color, label=label)
        for label, color in SCENARIO_COLORS.items()
    ]
    marker_handles = [
        Line2D([0], [0], marker="D", linestyle="", markersize=5.6, markerfacecolor="white", markeredgecolor=COLORS["ink"], label="Best"),
        Line2D([0], [0], marker="o", linestyle="", markersize=5.6, markerfacecolor="white", markeredgecolor=COLORS["ink"], label="Default"),
        Line2D([0], [0], marker="s", linestyle="", markersize=5.6, markerfacecolor=COLORS["light_gray"], markeredgecolor=COLORS["ink"], label="Worst"),
    ]
    legend1 = ax.legend(handles=scenario_handles, loc="upper left", frameon=False, ncol=1, handletextpad=0.5)
    ax.add_artist(legend1)
    ax.legend(handles=marker_handles, loc="lower right", frameon=False, ncol=1, handletextpad=0.5)
    add_panel_label(ax, "a")

    ax = axes[1]
    x = np.arange(len(SCENARIOS))
    width = 0.18
    for idx, metric_key in enumerate(
        ["profit_gain_pct", "harvest_gain_pct", "energy_change_pct", "cost_reduction_pct"]
    ):
        values = [
            float(delta_df[delta_df["scenario"] == spec["label"]][metric_key].iloc[0])
            for spec in SCENARIOS
        ]
        ax.bar(
            x + (idx - 1.5) * width,
            values,
            width=width,
            color=DELTA_COLORS[metric_key],
            edgecolor="white",
            linewidth=0.5,
            label={
                "profit_gain_pct": "Profit gain",
                "harvest_gain_pct": "Yield gain",
                "energy_change_pct": "Electricity change",
                "cost_reduction_pct": "Cost-per-kg reduction",
            }[metric_key],
        )
    style_axes(ax)
    ax.axhline(0.0, color=COLORS["ink"], linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([spec["label"] for spec in SCENARIOS])
    ax.set_ylabel("Relative change (%)")
    ax.legend(loc="upper right", frameon=False, ncol=1, handletextpad=0.5)
    add_panel_label(ax, "b")

    fig.tight_layout(pad=0.8, w_pad=1.0)
    save_figure(fig, out_path)
    plt.close(fig)


def _build_climate_sensitivity_heatmap(out_path: Path, eta_avg_df: pd.DataFrame) -> None:
    apply_academic_style()
    metric_order = METRICS
    variable_order = UPPER_VARS
    heat = np.zeros((len(variable_order), len(metric_order)))
    for i, variable in enumerate(variable_order):
        for j, metric in enumerate(metric_order):
            heat[i, j] = float(
                eta_avg_df[
                    (eta_avg_df["variable"] == variable)
                    & (eta_avg_df["metric"] == metric)
                ]["eta2_mean"].iloc[0]
            )

    fig, ax = plt.subplots(figsize=(6.0, 3.9))
    im = ax.imshow(heat, cmap="YlGnBu", vmin=0.0, vmax=max(0.85, float(heat.max())))
    ax.set_xticks(np.arange(len(metric_order)))
    ax.set_xticklabels([METRIC_LABELS[m] for m in metric_order], rotation=0)
    ax.set_yticks(np.arange(len(variable_order)))
    ax.set_yticklabels([VAR_LABELS[v] for v in variable_order])
    apply_heatmap_frame(ax, len(variable_order), len(metric_order))
    for i in range(len(variable_order)):
        for j in range(len(metric_order)):
            ax.text(
                j,
                i,
                f"{heat[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8.0,
                color="white" if heat[i, j] >= 0.45 else COLORS["ink"],
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    cbar.set_label(r"Main-effect $\eta^2$")
    add_panel_label(ax, "a")
    fig.tight_layout(pad=0.8)
    save_figure(fig, out_path)
    plt.close(fig)


def _build_grouped_controller_bars(
    ax: plt.Axes,
    df: pd.DataFrame,
    level: str,
    value_col: str,
    ylabel: str,
    annotate_harvests: bool = False,
) -> None:
    subset = df[df["level"] == level].copy()
    x = np.arange(len(SCENARIOS))
    width = 0.18
    for idx, controller_spec in enumerate(CONTROLLER_SPECS):
        controller = controller_spec["key"]
        color = controller_spec["color"]
        values = []
        harvests = []
        for spec in SCENARIOS:
            row = subset[
                (subset["scenario"] == spec["label"])
                & (subset["controller"] == controller)
            ].iloc[0]
            values.append(float(row[value_col]))
            harvests.append(int(row["total_harvests"]))
        bars = ax.bar(
            x + (idx - 1.5) * width,
            values,
            width=width,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            label=controller_spec["label"],
        )
        if annotate_harvests:
            for bar, harvest_count in zip(bars, harvests):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 4.0,
                    str(harvest_count),
                    ha="center",
                    va="bottom",
                    fontsize=7.1,
                    color=COLORS["ink"],
                    rotation=0,
                )
    style_axes(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([spec["label"] for spec in SCENARIOS])
    ax.set_ylabel(ylabel)


def _build_climate_controller_ablation_figure(
    out_path: Path,
    controller_ablation_df: pd.DataFrame,
) -> None:
    apply_academic_style()
    df = controller_ablation_df.copy()
    df["net_profit_k"] = df["net_profit"] / 1000.0

    fig, axes = plt.subplots(1, 3, figsize=(12.3, 4.1), gridspec_kw={"width_ratios": [1.0, 1.0, 0.9]})

    _build_grouped_controller_bars(
        axes[0],
        df,
        "default",
        "net_profit_k",
        r"Default-schedule profit (10$^3$ RMB)",
    )
    add_panel_label(axes[0], "a")

    _build_grouped_controller_bars(
        axes[1],
        df,
        "best",
        "net_profit_k",
        r"Best-schedule profit (10$^3$ RMB)",
    )
    axes[1].legend(loc="upper left", frameon=False, ncol=2, handletextpad=0.5, columnspacing=1.0)
    add_panel_label(axes[1], "b")

    _build_grouped_controller_bars(
        axes[2],
        df,
        "best",
        "total_harvests",
        "Best-schedule harvest events",
        annotate_harvests=True,
    )
    axes[2].set_ylim(0, 390)
    add_panel_label(axes[2], "c")

    fig.tight_layout(pad=0.8, w_pad=1.0)
    save_figure(fig, out_path)
    plt.close(fig)


def _build_climate_mechanism_figure(
    out_path: Path,
    hourly_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    relative_df: pd.DataFrame,
) -> None:
    apply_academic_style()
    fig, axes = plt.subplots(2, 2, figsize=(11.1, 6.9))

    for ax, signal, label, panel in [
        (axes[0, 0], "I1", r"Dense-zone light, $I_1$ ($\mu$mol m$^{-2}$ s$^{-1}$)", "a"),
        (axes[0, 1], "I2", r"Finishing-zone light, $I_2$ ($\mu$mol m$^{-2}$ s$^{-1}$)", "b"),
    ]:
        for controller_spec in CONTROLLER_SPECS:
            controller = controller_spec["key"]
            if controller == "pid" or controller == "climate" or controller == "gated" or controller == "residual":
                subset = hourly_df[hourly_df["controller"] == controller]
                ax.plot(
                    subset["hour"],
                    subset[signal],
                    color=controller_spec["color"],
                    linewidth=1.5,
                    label=controller_spec["label"],
                )
        style_axes(ax)
        set_hour_ticks(ax)
        ax.set_xlabel("Hour of day")
        ax.set_ylabel(label)
        add_panel_label(ax, panel)
    axes[0, 1].legend(loc="upper left", frameon=False, ncol=2, handletextpad=0.5, columnspacing=1.0)

    ax = axes[1, 0]
    order = [spec["key"] for spec in CONTROLLER_SPECS]
    ordered_df = metrics_df.set_index("controller").loc[order].reset_index()
    x = np.arange(len(ordered_df))
    bottom = np.zeros(len(ordered_df))
    for component, label in [
        ("led_mwh", "LED"),
        ("hvac_mwh", "HVAC"),
        ("dehum_mwh", "Dehumidification"),
    ]:
        ax.bar(
            x,
            ordered_df[component],
            bottom=bottom,
            color=ENERGY_COMPONENT_COLORS[component],
            edgecolor="white",
            linewidth=0.5,
            label=label,
        )
        bottom += ordered_df[component].to_numpy()
    style_axes(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(ordered_df["controller_label"], rotation=0)
    ax.set_ylabel("Annual electricity use (MWh)")
    ax.legend(loc="upper right", frameon=False, ncol=1, handletextpad=0.5)
    add_panel_label(ax, "c")

    ax = axes[1, 1]
    metric_order = [
        "transp_kg",
        "dehum_removed_kg",
        "dehum_mwh",
        "avg_fw_g_per_plant",
        "net_profit",
    ]
    x = np.arange(len(metric_order))
    width = 0.24
    rel_df = relative_df.copy()
    label_map = {
        "transp_kg": "Transpiration",
        "dehum_removed_kg": "Dehumidified\nwater",
        "dehum_mwh": "Dehum\nelectricity",
        "avg_fw_g_per_plant": "Fresh mass\nper plant",
        "net_profit": "Net profit",
    }
    for idx, controller_spec in enumerate(CONTROLLER_SPECS[1:]):
        controller = controller_spec["key"]
        color = controller_spec["color"]
        values = []
        for metric_key in metric_order:
            row = rel_df[
                (rel_df["controller"] == controller) & (rel_df["metric_key"] == metric_key)
            ].iloc[0]
            values.append(float(row["delta_pct_vs_pid"]))
        ax.bar(
            x + (idx - 1.0) * width,
            values,
            width=width,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            label=controller_spec["label"],
        )
    style_axes(ax)
    ax.axhline(0.0, color=COLORS["ink"], linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([label_map[key] for key in metric_order])
    ax.set_ylabel("Relative change vs PID (%)")
    ax.legend(loc="upper right", frameon=False, ncol=1, handletextpad=0.5)
    add_panel_label(ax, "d")

    fig.tight_layout(pad=0.8, w_pad=1.1, h_pad=1.0)
    save_figure(fig, out_path)
    plt.close(fig)


def _style_secondary_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_color(COLORS["gray"])
    ax.spines["right"].set_linewidth(0.7)
    ax.tick_params(axis="y", colors=COLORS["gray"], labelsize=7.8)


def _build_climate_tou_figure(
    out_path: Path,
    rank_shift_df: pd.DataFrame,
    tou_delta_df: pd.DataFrame,
    tou_hourly_df: pd.DataFrame,
) -> None:
    apply_academic_style()
    fig, axes = plt.subplots(2, 2, figsize=(11.1, 6.8))

    ax = axes[0, 0]
    ordered = rank_shift_df.set_index("controller").loc[["residual", "gated", "climate"]].reset_index()
    ax.bar(
        ordered["controller_label"],
        ordered["mean_abs_rank_shift"],
        color=[COLORS["brick"], COLORS["teal"], COLORS["navy"]],
        edgecolor="white",
        linewidth=0.5,
    )
    for idx, row in ordered.iterrows():
        ax.text(
            idx,
            row["mean_abs_rank_shift"] + 1.2,
            f"max={int(row['max_rank_shift'])}",
            ha="center",
            va="bottom",
            fontsize=7.3,
            color=COLORS["ink"],
        )
    style_axes(ax)
    ax.set_ylabel("Mean absolute rank shift")
    add_panel_label(ax, "a")

    ax = axes[0, 1]
    for tariff, color, ls in [("Constant", COLORS["navy"], "-"), ("TOU", COLORS["brick"], "--")]:
        subset = tou_hourly_df[tou_hourly_df["tariff"] == tariff]
        ax.plot(
            subset["hour"],
            subset["I1"],
            color=color,
            linestyle=ls,
            linewidth=1.5,
            label=fr"{tariff} $I_1$",
        )
        ax.plot(
            subset["hour"],
            subset["I2"],
            color=color,
            linestyle=":" if tariff == "Constant" else "-.",
            linewidth=1.2,
            label=fr"{tariff} $I_2$",
        )
    style_axes(ax)
    set_hour_ticks(ax)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel(r"Light intensity ($\mu$mol m$^{-2}$ s$^{-1}$)")
    ax.legend(loc="upper left", frameon=False, ncol=2, handletextpad=0.5, columnspacing=0.9)
    add_panel_label(ax, "b")

    ax = axes[1, 0]
    for tariff, color, ls in [("Constant", COLORS["navy"], "-"), ("TOU", COLORS["brick"], "--")]:
        subset = tou_hourly_df[tou_hourly_df["tariff"] == tariff]
        ax.plot(
            subset["hour"],
            subset["Q_HVAC"],
            color=color,
            linestyle=ls,
            linewidth=1.5,
            label=tariff,
        )
    style_axes(ax)
    set_hour_ticks(ax)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel(r"HVAC heat flow, $Q_{\mathrm{HVAC}}$ (W m$^{-2}$)")
    ax2 = ax.twinx()
    tou_subset = tou_hourly_df[tou_hourly_df["tariff"] == "TOU"]
    ax2.step(
        tou_subset["hour"],
        tou_subset["elec_price_rmb_kwh"],
        where="mid",
        color=COLORS["gray"],
        linewidth=1.0,
        alpha=0.85,
    )
    ax2.set_ylabel(r"Electricity price (RMB kWh$^{-1}$)")
    _style_secondary_axis(ax2)
    ax.legend(loc="upper left", frameon=False, ncol=1, handletextpad=0.5)
    add_panel_label(ax, "c")

    ax = axes[1, 1]
    order = [
        "harvest_fresh_kg",
        "energy_kwh",
        "total_cost",
        "net_profit",
        "transp_kg",
        "dehum_mwh",
    ]
    label_map = {
        "harvest_fresh_kg": "Yield",
        "energy_kwh": "Electricity",
        "total_cost": "Total cost",
        "net_profit": "Net profit",
        "transp_kg": "Transpiration",
        "dehum_mwh": "Dehum\nelectricity",
    }
    subset = tou_delta_df.set_index("metric_key").loc[order].reset_index()
    colors = [
        COLORS["teal"] if value >= 0 else COLORS["brick"]
        for value in subset["delta_pct"]
    ]
    ax.bar(
        np.arange(len(subset)),
        subset["delta_pct"],
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )
    style_axes(ax)
    ax.axhline(0.0, color=COLORS["ink"], linewidth=0.8)
    ax.set_xticks(np.arange(len(subset)))
    ax.set_xticklabels([label_map[key] for key in subset["metric_key"]])
    ax.set_ylabel("TOU vs constant (%)")
    add_panel_label(ax, "d")

    fig.tight_layout(pad=0.8, w_pad=1.1, h_pad=1.0)
    save_figure(fig, out_path)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pid_map = {
        spec["label"]: _load_schedule_df(PID_RESULTS_ROOT / spec["pid_name"], "pid")
        for spec in SCENARIOS
    }
    residual_map = {
        spec["label"]: _load_schedule_df(RESIDUAL_RESULTS_ROOT / spec["residual_name"], "rl")
        for spec in SCENARIOS
    }
    gated_map = {
        spec["label"]: _load_schedule_df(GATED_RESULTS_ROOT / spec["gated_name"], "rl")
        for spec in SCENARIOS
    }
    climate_map = {
        spec["label"]: _load_schedule_df(CLIMATE_RESULTS_ROOT / spec["climate_name"], "rl")
        for spec in SCENARIOS
    }

    integrity_df = _collect_integrity_table()
    metadata_nulls_df = _collect_metadata_nulls(climate_map)
    summary_df, delta_df = _collect_climate_summary_rows(climate_map)
    eta_df, eta_avg_df = _collect_eta_table(climate_map)
    controller_ablation_df, same_schedule_df = _collect_controller_ablation_rows(
        pid_map,
        residual_map,
        gated_map,
        climate_map,
    )
    mechanism_hourly_df, mechanism_metrics_df, mechanism_relative_df = _collect_mechanism_tables(
        pid_map,
        residual_map,
        gated_map,
        climate_map,
    )
    tou_rank_shift_df, tou_delta_df, tou_hourly_df = _collect_tou_tables(
        residual_map,
        gated_map,
        climate_map,
    )

    _build_climate_core_summary_figure(
        OUT_DIR / "compact_climate_core_summary.png",
        climate_map,
        summary_df,
        delta_df,
    )
    _build_climate_sensitivity_heatmap(
        OUT_DIR / "compact_climate_upper_sensitivity_heatmap.png",
        eta_avg_df,
    )
    _build_climate_controller_ablation_figure(
        OUT_DIR / "compact_climate_controller_ablation.png",
        controller_ablation_df,
    )
    _build_climate_mechanism_figure(
        OUT_DIR / "compact_climate_light_fixed_mechanism.png",
        mechanism_hourly_df,
        mechanism_metrics_df,
        mechanism_relative_df,
    )
    _build_climate_tou_figure(
        OUT_DIR / "compact_climate_tou_response.png",
        tou_rank_shift_df,
        tou_delta_df,
        tou_hourly_df,
    )

    integrity_df.to_csv(OUT_DIR / "compact_climate_integrity.csv", index=False, encoding="utf-8-sig")
    metadata_nulls_df.to_csv(OUT_DIR / "compact_climate_metadata_nulls.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUT_DIR / "compact_climate_summary.csv", index=False, encoding="utf-8-sig")
    delta_df.to_csv(OUT_DIR / "compact_climate_deltas.csv", index=False, encoding="utf-8-sig")
    eta_df.to_csv(OUT_DIR / "compact_climate_upper_sensitivity_eta2.csv", index=False, encoding="utf-8-sig")
    eta_avg_df.to_csv(OUT_DIR / "compact_climate_upper_sensitivity_average.csv", index=False, encoding="utf-8-sig")
    controller_ablation_df.to_csv(OUT_DIR / "compact_climate_controller_ablation.csv", index=False, encoding="utf-8-sig")
    same_schedule_df.to_csv(OUT_DIR / "compact_climate_same_schedule_comparison.csv", index=False, encoding="utf-8-sig")
    mechanism_hourly_df.to_csv(OUT_DIR / "compact_climate_default_hourly_profile.csv", index=False, encoding="utf-8-sig")
    mechanism_metrics_df.to_csv(OUT_DIR / "compact_climate_default_mechanism_metrics.csv", index=False, encoding="utf-8-sig")
    mechanism_relative_df.to_csv(OUT_DIR / "compact_climate_default_relative_changes.csv", index=False, encoding="utf-8-sig")
    tou_rank_shift_df.to_csv(OUT_DIR / "compact_climate_tou_rank_shift.csv", index=False, encoding="utf-8-sig")
    tou_delta_df.to_csv(OUT_DIR / "compact_climate_tou_shared_schedule_delta.csv", index=False, encoding="utf-8-sig")
    tou_hourly_df.to_csv(OUT_DIR / "compact_climate_tou_hourly_profile.csv", index=False, encoding="utf-8-sig")

    summary_payload = {
        "integrity_rows": len(integrity_df),
        "scenarios": [spec["label"] for spec in SCENARIOS],
        "output_dir": str(OUT_DIR),
        "best_schedules": {
            row["scenario"]: row["schedule_key"]
            for _, row in summary_df[summary_df["status"] == "best"].iterrows()
        },
    }
    (OUT_DIR / "compact_climate_result_review_summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[OK] Saved compact climate result-review figures to: {OUT_DIR}")


if __name__ == "__main__":
    main()
