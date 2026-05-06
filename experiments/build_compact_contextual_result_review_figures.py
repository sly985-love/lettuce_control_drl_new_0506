# -*- coding: utf-8 -*-
"""Build compact figures and summary tables for contextual SAC results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

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
CONTEXTUAL_RESULTS_ROOT = Path(r"C:\Users\29341\Desktop\fsdownload\results_contextual_sac")
OUT_DIR = ROOT / "paper" / "figures_result_review_contextual_20260428"

DEFAULT_SCHEDULE_KEY = "t1=14|t2=14|N1=20|rho2=36"
REASONABLE_SCHEDULE_KEY = "t1=13|t2=13|N1=15|rho2=24"
PIDLIKE_SCHEDULE_KEY = "t1=13|t2=13|N1=16|rho2=24"
GATEDLIKE_SCHEDULE_KEY = "t1=13|t2=13|N1=16|rho2=23"
AGGRESSIVE_SCHEDULE_KEY = "t1=15|t2=13|N1=15|rho2=23"
CONTEXTUAL_BEST_L20_KEY = "t1=16|t2=16|N1=12|rho2=21"
CONTEXTUAL_BEST_L40_KEY = "t1=15|t2=13|N1=15|rho2=26"
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
}
CONTROLLER_SPECS = [
    {"key": "pid", "label": "PID", "color": COLORS["gray"]},
    {"key": "residual", "label": "Full residual", "color": COLORS["brick"]},
    {"key": "gated", "label": "Gated residual", "color": COLORS["teal"]},
    {"key": "climate", "label": "Climate-only", "color": COLORS["navy"]},
    {"key": "contextual", "label": "Contextual SAC", "color": COLORS["gold"]},
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
        "contextual_name": "exp03_exact_contextualsac_rl_pp16h_constant_e0p74_co20p54_l20",
        "residual_name": "exp03_exact_rl_baseline_pp16h_constant_e0p74_co20p54_l20",
        "gated_name": "exp03_exact_gatedrespid_rl_pp16h_constant_e0p74_co20p54_l20",
        "climate_name": "exp03_exact_climaterespid_rl_pp16h_constant_e0p74_co20p54_l20",
        "pid_name": "exp02_exact_pid_baseline_pp16h_i1250_i2300_pm_constant_e0p74_co20p54_l20",
    },
    {
        "label": "Const-L40",
        "contextual_name": "exp03_exact_contextualsa_rl_pp16h_constant_e0p74_co20p54_l40",
        "residual_name": "exp03_exact_rl_baseline_pp16h_constant_e0p74_co20p54_l40",
        "gated_name": "exp03_exact_gatedrespid_rl_pp16h_constant_e0p74_co20p54_l40",
        "climate_name": "exp03_exact_climaterespid_rl_pp16h_constant_e0p74_co20p54_l40",
        "pid_name": "exp02_exact_pid_baseline_pp16h_i1250_i2300_pm_constant_e0p74_co20p54_l40",
    },
]

RESIDUAL_GATED_SHARED_SCENARIOS = [
    {
        "label": "Const-L20",
        "residual_name": "exp03_exact_rl_baseline_pp16h_constant_e0p74_co20p54_l20",
        "gated_name": "exp03_exact_gatedrespid_rl_pp16h_constant_e0p74_co20p54_l20",
    },
    {
        "label": "Daily-Const-L20",
        "residual_name": "exp03_exact_rl_baseline_pp16h_constant_e0p74_co20p54_l20_daily",
        "gated_name": "exp03_exact_gatedrespid_rl_pp16h_constant_e0p74_co20p54_l20_daily",
    },
    {
        "label": "Const-L40",
        "residual_name": "exp03_exact_rl_baseline_pp16h_constant_e0p74_co20p54_l40",
        "gated_name": "exp03_exact_gatedrespid_rl_pp16h_constant_e0p74_co20p54_l40",
    },
    {
        "label": "TOU-L20",
        "residual_name": "exp03_exact_rl_baseline_pp16h_tou_zhejiang_lt1kv_l20",
        "gated_name": "exp03_exact_gatedrespid_rl_pp16h_tou_zhejiang_lt1kv_l20",
    },
    {
        "label": "TOU-L40",
        "residual_name": "exp03_exact_rl_baseline_pp16h_tou_zhejiang_lt1kv_l40",
        "gated_name": "exp03_exact_gatedrespid_rl_pp16h_tou_zhejiang_lt1kv_l40",
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
    dt_h = float(trace_df["step_size_s"].iloc[0]) / 3600.0
    horizon_h = len(trace_df) * dt_h
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
        "tv_I1_per_h": float(np.abs(np.diff(trace_df["I1"].to_numpy())).sum() / horizon_h),
        "tv_I2_per_h": float(np.abs(np.diff(trace_df["I2"].to_numpy())).sum() / horizon_h),
        "tv_Q_HVAC_per_h": float(np.abs(np.diff(trace_df["Q_HVAC"].to_numpy())).sum() / horizon_h),
    }


def _load_all_maps() -> tuple[
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
]:
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
    contextual_map = {
        spec["label"]: _load_schedule_df(CONTEXTUAL_RESULTS_ROOT / spec["contextual_name"], "rl")
        for spec in SCENARIOS
    }
    return pid_map, residual_map, gated_map, climate_map, contextual_map


def _collect_integrity_table() -> pd.DataFrame:
    rows = []
    for spec in SCENARIOS:
        exp_dir = CONTEXTUAL_RESULTS_ROOT / spec["contextual_name"]
        rows.append(
            {
                "scenario": spec["label"],
                "folder_name": spec["contextual_name"],
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


def _collect_metadata_nulls(contextual_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
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
    for scenario, df in contextual_map.items():
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


def _collect_contextual_summary_rows(
    contextual_map: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    delta_rows = []
    for spec in SCENARIOS:
        scenario = spec["label"]
        df = contextual_map[scenario]
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
    contextual_map: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for spec in SCENARIOS:
        scenario = spec["label"]
        df = contextual_map[scenario]
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


def _collect_controller_constant_rows(
    pid_map: dict[str, pd.DataFrame],
    residual_map: dict[str, pd.DataFrame],
    gated_map: dict[str, pd.DataFrame],
    climate_map: dict[str, pd.DataFrame],
    contextual_map: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    controller_maps = {
        "pid": pid_map,
        "residual": residual_map,
        "gated": gated_map,
        "climate": climate_map,
        "contextual": contextual_map,
    }
    rows = []
    same_schedule_rows = []
    contextual_best_keys = {
        "Const-L20": CONTEXTUAL_BEST_L20_KEY,
        "Const-L40": CONTEXTUAL_BEST_L40_KEY,
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

            case_specs = [
                ("default", DEFAULT_SCHEDULE_KEY),
                ("reasonable", REASONABLE_SCHEDULE_KEY),
                ("pid_like", PIDLIKE_SCHEDULE_KEY),
                ("gated_like", GATEDLIKE_SCHEDULE_KEY),
                ("aggressive", AGGRESSIVE_SCHEDULE_KEY),
                ("contextual_best", contextual_best_keys[scenario]),
            ]
            for schedule_case, schedule_key in case_specs:
                row = df[df["schedule_key"] == schedule_key]
                if len(row) == 0:
                    continue
                row = row.iloc[0]
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
    contextual_map: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scenario = "Const-L40"
    exp_dirs = {
        "pid": PID_RESULTS_ROOT / next(spec["pid_name"] for spec in SCENARIOS if spec["label"] == scenario),
        "residual": RESIDUAL_RESULTS_ROOT / next(spec["residual_name"] for spec in SCENARIOS if spec["label"] == scenario),
        "gated": GATED_RESULTS_ROOT / next(spec["gated_name"] for spec in SCENARIOS if spec["label"] == scenario),
        "climate": CLIMATE_RESULTS_ROOT / next(spec["climate_name"] for spec in SCENARIOS if spec["label"] == scenario),
        "contextual": CONTEXTUAL_RESULTS_ROOT / next(spec["contextual_name"] for spec in SCENARIOS if spec["label"] == scenario),
    }
    df_maps = {
        "pid": pid_map[scenario],
        "residual": residual_map[scenario],
        "gated": gated_map[scenario],
        "climate": climate_map[scenario],
        "contextual": contextual_map[scenario],
    }

    hourly_rows = []
    metrics_rows = []
    for controller_spec in CONTROLLER_SPECS:
        controller = controller_spec["key"]
        schedule_row = df_maps[controller][df_maps[controller]["schedule_key"] == DEFAULT_SCHEDULE_KEY].iloc[0]
        trace_df = _load_trace_df(exp_dirs[controller], DEFAULT_SCHEDULE_KEY)
        hourly_df = _hourly_profile(trace_df, ["I1", "I2", "Q_HVAC"])
        hourly_df["controller"] = controller
        hourly_df["controller_label"] = controller_spec["label"]
        hourly_rows.append(hourly_df)

        summary = _trace_summary(trace_df)
        metrics_rows.append(
            {
                "controller": controller,
                "controller_label": controller_spec["label"],
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
    for _, row in metrics_df.iterrows():
        if row["controller"] == "pid":
            continue
        for metric_key, metric_label in [
            ("harvest_fresh_kg", "Fresh yield"),
            ("led_mwh", "LED electricity"),
            ("hvac_mwh", "HVAC electricity"),
            ("dehum_mwh", "Dehumidification electricity"),
            ("net_profit", "Net profit"),
            ("tv_Q_HVAC_per_h", r"HVAC variation"),
        ]:
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


def _collect_risk_tables(
    pid_map: dict[str, pd.DataFrame],
    residual_map: dict[str, pd.DataFrame],
    gated_map: dict[str, pd.DataFrame],
    climate_map: dict[str, pd.DataFrame],
    contextual_map: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    controller_maps = {
        "pid": pid_map,
        "residual": residual_map,
        "gated": gated_map,
        "climate": climate_map,
        "contextual": contextual_map,
    }
    dist_rows = []
    for spec in SCENARIOS:
        scenario = spec["label"]
        for controller in controller_maps:
            df = controller_maps[controller][scenario]
            dist_rows.append(
                {
                    "scenario": scenario,
                    "controller": controller,
                    "mean_profit": float(df["net_profit"].mean()),
                    "median_profit": float(df["net_profit"].median()),
                    "p10_profit": float(df["net_profit"].quantile(0.1)),
                    "p90_profit": float(df["net_profit"].quantile(0.9)),
                    "negative_profit_n": int((df["net_profit"] < 0).sum()),
                    "profit_range": float(df["net_profit"].max() - df["net_profit"].min()),
                }
            )

    contextual_case_specs = [
        ("Const-L20", "best", CONTEXTUAL_BEST_L20_KEY),
        ("Const-L20", "worst", contextual_map["Const-L20"].sort_values("net_profit", ascending=True).iloc[0]["schedule_key"]),
        ("Const-L40", "best", CONTEXTUAL_BEST_L40_KEY),
        ("Const-L40", "worst", contextual_map["Const-L40"].sort_values("net_profit", ascending=True).iloc[0]["schedule_key"]),
    ]
    stress_rows = []
    for scenario, case, schedule_key in contextual_case_specs:
        exp_dir = CONTEXTUAL_RESULTS_ROOT / next(
            spec["contextual_name"] for spec in SCENARIOS if spec["label"] == scenario
        )
        df = _load_trace_df(exp_dir, schedule_key)
        stress_rows.append(
            {
                "scenario": scenario,
                "case": case,
                "schedule_key": schedule_key,
                "I1_zero_pct": float((df["I1"] == 0.0).mean() * 100.0),
                "I1_max_pct": float((df["I1"] >= 249.9).mean() * 100.0),
                "I2_max_pct": float((df["I2"] >= 299.9).mean() * 100.0),
                "HVAC_cool_sat_pct": float((df["Q_HVAC"] <= -211.9).mean() * 100.0),
                "RH_lt40_pct": float((df["RH_pct"] < 40.0).mean() * 100.0),
                "RH_gt90_pct": float((df["RH_pct"] > 90.0).mean() * 100.0),
                "VPD_gt1p5_pct": float((df["VPD_kPa"] > 1.5).mean() * 100.0),
            }
        )
    return pd.DataFrame(dist_rows), pd.DataFrame(stress_rows)


def _collect_model_selection_tables(
    pid_map: dict[str, pd.DataFrame],
    residual_map: dict[str, pd.DataFrame],
    gated_map: dict[str, pd.DataFrame],
    climate_map: dict[str, pd.DataFrame],
    contextual_map: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for spec in RESIDUAL_GATED_SHARED_SCENARIOS:
        residual_df = _load_schedule_df(RESIDUAL_RESULTS_ROOT / spec["residual_name"], "rl")
        gated_df = _load_schedule_df(GATED_RESULTS_ROOT / spec["gated_name"], "rl")
        residual_best = residual_df.sort_values("net_profit", ascending=False).iloc[0]
        gated_best = gated_df.sort_values("net_profit", ascending=False).iloc[0]
        rows.append(
            {
                "scenario": spec["label"],
                "residual_best_schedule": residual_best["schedule_key"],
                "gated_best_schedule": gated_best["schedule_key"],
                "residual_best_profit": float(residual_best["net_profit"]),
                "gated_best_profit": float(gated_best["net_profit"]),
                "gated_vs_residual_profit_pct": float(
                    (gated_best["net_profit"] / residual_best["net_profit"] - 1.0) * 100.0
                ),
                "residual_best_harvests": int(residual_best["total_harvests"]),
                "gated_best_harvests": int(gated_best["total_harvests"]),
            }
        )

    controller_constant_df, _ = _collect_controller_constant_rows(
        pid_map, residual_map, gated_map, climate_map, contextual_map
    )
    return pd.DataFrame(rows), controller_constant_df


def _build_contextual_core_summary_figure(
    out_path: Path,
    contextual_map: dict[str, pd.DataFrame],
    summary_df: pd.DataFrame,
    delta_df: pd.DataFrame,
) -> None:
    apply_academic_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.3, 4.2), gridspec_kw={"width_ratios": [1.15, 1.0]})

    ax = axes[0]
    for spec in SCENARIOS:
        scenario = spec["label"]
        df = contextual_map[scenario]
        ax.scatter(
            df["total_harvests"],
            df["net_profit"] / 1000.0,
            s=16,
            alpha=0.25,
            color=SCENARIO_COLORS[scenario],
            edgecolors="none",
        )
        for status, marker, edge_color, face_color in [
            ("best", "D", COLORS["ink"], SCENARIO_COLORS[scenario]),
            ("default", "o", COLORS["ink"], "white"),
            ("worst", "s", COLORS["ink"], COLORS["light_gray"]),
        ]:
            row = summary_df[
                (summary_df["scenario"] == scenario) & (summary_df["status"] == status)
            ].iloc[0]
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
    ax.set_xlim(0, 390)
    scenario_handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=6.2, color=color, label=label)
        for label, color in SCENARIO_COLORS.items()
    ]
    marker_handles = [
        Line2D([0], [0], marker="D", linestyle="", markersize=5.6, markerfacecolor="white", markeredgecolor=COLORS["ink"], label="Best"),
        Line2D([0], [0], marker="o", linestyle="", markersize=5.6, markerfacecolor="white", markeredgecolor=COLORS["ink"], label="Default"),
        Line2D([0], [0], marker="s", linestyle="", markersize=5.6, markerfacecolor=COLORS["light_gray"], markeredgecolor=COLORS["ink"], label="Worst"),
    ]
    legend1 = ax.legend(handles=scenario_handles, loc="upper left", frameon=False, handletextpad=0.5)
    ax.add_artist(legend1)
    ax.legend(handles=marker_handles, loc="lower right", frameon=False, handletextpad=0.5)
    add_panel_label(ax, "a")

    ax = axes[1]
    x = np.arange(len(SCENARIOS))
    width = 0.18
    metric_order = ["profit_gain_pct", "harvest_gain_pct", "energy_change_pct", "cost_reduction_pct"]
    label_map = {
        "profit_gain_pct": "Profit gain",
        "harvest_gain_pct": "Yield gain",
        "energy_change_pct": "Electricity change",
        "cost_reduction_pct": "Cost-per-kg reduction",
    }
    for idx, metric_key in enumerate(metric_order):
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
            label=label_map[metric_key],
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


def _build_contextual_sensitivity_heatmap(out_path: Path, eta_avg_df: pd.DataFrame) -> None:
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
    im = ax.imshow(heat, cmap="YlOrBr", vmin=0.0, vmax=max(0.70, float(heat.max())))
    ax.set_xticks(np.arange(len(metric_order)))
    ax.set_xticklabels([METRIC_LABELS[m] for m in metric_order])
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
                color="white" if heat[i, j] >= 0.42 else COLORS["ink"],
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
    annotate_value_col: str | None = None,
) -> None:
    subset = df[df["level"] == level].copy()
    x = np.arange(len(SCENARIOS))
    width = 0.15
    for idx, controller_spec in enumerate(CONTROLLER_SPECS):
        controller = controller_spec["key"]
        values = []
        anno_values = []
        for spec in SCENARIOS:
            row = subset[
                (subset["scenario"] == spec["label"]) & (subset["controller"] == controller)
            ].iloc[0]
            values.append(float(row[value_col]))
            anno_values.append(float(row[annotate_value_col]) if annotate_value_col else np.nan)
        bars = ax.bar(
            x + (idx - 2.0) * width,
            values,
            width=width,
            color=controller_spec["color"],
            edgecolor="white",
            linewidth=0.5,
            label=controller_spec["label"],
        )
        if annotate_value_col:
            for bar, anno in zip(bars, anno_values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 1.8,
                    f"{int(round(anno))}",
                    ha="center",
                    va="bottom",
                    fontsize=7.0,
                    color=COLORS["ink"],
                )
    style_axes(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([spec["label"] for spec in SCENARIOS])
    ax.set_ylabel(ylabel)


def _build_contextual_controller_comparison_figure(
    out_path: Path,
    controller_constant_df: pd.DataFrame,
    risk_dist_df: pd.DataFrame,
) -> None:
    apply_academic_style()
    df = controller_constant_df.copy()
    df["net_profit_k"] = df["net_profit"] / 1000.0

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.2))

    _build_grouped_controller_bars(
        axes[0, 0],
        df,
        "default",
        "net_profit_k",
        r"Default-schedule profit (10$^3$ RMB)",
    )
    add_panel_label(axes[0, 0], "a")

    _build_grouped_controller_bars(
        axes[0, 1],
        df,
        "best",
        "net_profit_k",
        r"Best-schedule profit (10$^3$ RMB)",
        annotate_value_col="total_harvests",
    )
    axes[0, 1].legend(loc="upper left", frameon=False, ncol=2, handletextpad=0.5, columnspacing=0.8)
    add_panel_label(axes[0, 1], "b")

    ax = axes[1, 0]
    dist_subset = risk_dist_df.copy()
    x = np.arange(len(CONTROLLER_SPECS))
    offsets = {"Const-L20": -0.12, "Const-L40": 0.12}
    for scenario in ["Const-L20", "Const-L40"]:
        for idx, controller_spec in enumerate(CONTROLLER_SPECS):
            row = dist_subset[
                (dist_subset["scenario"] == scenario)
                & (dist_subset["controller"] == controller_spec["key"])
            ].iloc[0]
            xpos = idx + offsets[scenario]
            ax.plot(
                [xpos, xpos],
                [row["p10_profit"] / 1000.0, row["p90_profit"] / 1000.0],
                color=SCENARIO_COLORS[scenario],
                linewidth=2.0,
                alpha=0.9,
            )
            ax.scatter(
                [xpos],
                [row["median_profit"] / 1000.0],
                color=SCENARIO_COLORS[scenario],
                s=26,
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
    style_axes(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([spec["label"] for spec in CONTROLLER_SPECS], rotation=0)
    ax.set_ylabel(r"Profit distribution (10$^3$ RMB)")
    legend_handles = [
        Line2D([0], [0], color=SCENARIO_COLORS["Const-L20"], linewidth=2.0, label="Const-L20"),
        Line2D([0], [0], color=SCENARIO_COLORS["Const-L40"], linewidth=2.0, label="Const-L40"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", frameon=False, ncol=1, handletextpad=0.5)
    add_panel_label(ax, "c")

    ax = axes[1, 1]
    neg_counts = (
        risk_dist_df.groupby("controller", as_index=False)["negative_profit_n"]
        .sum()
        .set_index("controller")
        .loc[[spec["key"] for spec in CONTROLLER_SPECS]]
        .reset_index()
    )
    ax.bar(
        np.arange(len(neg_counts)),
        neg_counts["negative_profit_n"],
        color=[spec["color"] for spec in CONTROLLER_SPECS],
        edgecolor="white",
        linewidth=0.5,
    )
    style_axes(ax)
    ax.set_xticks(np.arange(len(neg_counts)))
    ax.set_xticklabels([spec["label"] for spec in CONTROLLER_SPECS], rotation=0)
    ax.set_ylabel("Negative-profit schedules (2 scenarios)")
    add_panel_label(ax, "d")

    fig.tight_layout(pad=0.8, w_pad=1.0, h_pad=1.0)
    save_figure(fig, out_path)
    plt.close(fig)


def _build_contextual_mechanism_figure(
    out_path: Path,
    hourly_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    relative_df: pd.DataFrame,
) -> None:
    apply_academic_style()
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.0))

    ax = axes[0, 0]
    for controller_spec in CONTROLLER_SPECS:
        subset = hourly_df[hourly_df["controller"] == controller_spec["key"]]
        ax.plot(
            subset["hour"],
            subset["I1"],
            color=controller_spec["color"],
            linewidth=1.5,
            label=controller_spec["label"],
        )
    style_axes(ax)
    set_hour_ticks(ax)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel(r"Dense-zone light, $I_1$ ($\mu$mol m$^{-2}$ s$^{-1}$)")
    add_panel_label(ax, "a")

    ax = axes[0, 1]
    for controller_spec in CONTROLLER_SPECS:
        subset = hourly_df[hourly_df["controller"] == controller_spec["key"]]
        ax.plot(
            subset["hour"],
            subset["Q_HVAC"],
            color=controller_spec["color"],
            linewidth=1.5,
            label=controller_spec["label"],
        )
    style_axes(ax)
    set_hour_ticks(ax)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel(r"HVAC heat flow, $Q_{\mathrm{HVAC}}$ (W m$^{-2}$)")
    ax.legend(loc="upper left", frameon=False, ncol=2, handletextpad=0.5, columnspacing=0.8)
    add_panel_label(ax, "b")

    ax = axes[1, 0]
    ordered_df = metrics_df.set_index("controller").loc[[spec["key"] for spec in CONTROLLER_SPECS]].reset_index()
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
    ax.set_xticklabels(ordered_df["controller_label"])
    ax.set_ylabel("Annual electricity use (MWh)")
    ax.legend(loc="upper right", frameon=False, ncol=1, handletextpad=0.5)
    add_panel_label(ax, "c")

    ax = axes[1, 1]
    metric_order = [
        "harvest_fresh_kg",
        "led_mwh",
        "hvac_mwh",
        "dehum_mwh",
        "net_profit",
        "tv_Q_HVAC_per_h",
    ]
    label_map = {
        "harvest_fresh_kg": "Yield",
        "led_mwh": "LED",
        "hvac_mwh": "HVAC",
        "dehum_mwh": "Dehum",
        "net_profit": "Profit",
        "tv_Q_HVAC_per_h": "HVAC\nvariation",
    }
    x = np.arange(len(metric_order))
    width = 0.18
    for idx, controller_spec in enumerate(CONTROLLER_SPECS[1:]):
        controller = controller_spec["key"]
        values = []
        for metric_key in metric_order:
            row = relative_df[
                (relative_df["controller"] == controller)
                & (relative_df["metric_key"] == metric_key)
            ].iloc[0]
            values.append(float(row["delta_pct_vs_pid"]))
        ax.bar(
            x + (idx - 1.5) * width,
            values,
            width=width,
            color=controller_spec["color"],
            edgecolor="white",
            linewidth=0.5,
            label=controller_spec["label"],
        )
    style_axes(ax)
    ax.axhline(0.0, color=COLORS["ink"], linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([label_map[key] for key in metric_order])
    ax.set_ylabel("Relative change vs PID (%)")
    add_panel_label(ax, "d")

    fig.tight_layout(pad=0.8, w_pad=1.1, h_pad=1.0)
    save_figure(fig, out_path)
    plt.close(fig)


def _build_contextual_tail_risk_figure(
    out_path: Path,
    risk_dist_df: pd.DataFrame,
    contextual_stress_df: pd.DataFrame,
) -> None:
    apply_academic_style()
    fig, axes = plt.subplots(2, 2, figsize=(11.3, 7.0))

    for ax, scenario, panel in [
        (axes[0, 0], "Const-L20", "a"),
        (axes[0, 1], "Const-L40", "b"),
    ]:
        subset = risk_dist_df[risk_dist_df["scenario"] == scenario].set_index("controller")
        xs = np.arange(len(CONTROLLER_SPECS))
        for idx, controller_spec in enumerate(CONTROLLER_SPECS):
            row = subset.loc[controller_spec["key"]]
            ax.plot(
                [idx, idx],
                [row["p10_profit"] / 1000.0, row["p90_profit"] / 1000.0],
                color=controller_spec["color"],
                linewidth=2.0,
            )
            ax.scatter(
                [idx],
                [row["median_profit"] / 1000.0],
                color=controller_spec["color"],
                s=28,
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
        style_axes(ax)
        ax.set_xticks(xs)
        ax.set_xticklabels([spec["label"] for spec in CONTROLLER_SPECS], rotation=0)
        ax.set_ylabel(r"Profit interval (10$^3$ RMB)")
        add_panel_label(ax, panel)

    ax = axes[1, 0]
    subset = contextual_stress_df.copy()
    metric_order = ["I1_zero_pct", "HVAC_cool_sat_pct", "RH_lt40_pct", "VPD_gt1p5_pct"]
    label_map = {
        "I1_zero_pct": r"$I_1=0$ time",
        "HVAC_cool_sat_pct": "Cooling saturation",
        "RH_lt40_pct": r"RH < 40%",
        "VPD_gt1p5_pct": r"VPD > 1.5 kPa",
    }
    x = np.arange(len(metric_order))
    width = 0.18
    color_map = {
        ("Const-L20", "best"): COLORS["navy"],
        ("Const-L20", "worst"): COLORS["brick"],
        ("Const-L40", "best"): COLORS["teal"],
        ("Const-L40", "worst"): COLORS["gold"],
    }
    for idx, (_, row) in enumerate(subset.iterrows()):
        values = [float(row[metric]) for metric in metric_order]
        ax.bar(
            x + (idx - 1.5) * width,
            values,
            width=width,
            color=color_map[(row["scenario"], row["case"])],
            edgecolor="white",
            linewidth=0.5,
            label=f"{row['scenario']} {row['case']}",
        )
    style_axes(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([label_map[key] for key in metric_order])
    ax.set_ylabel("Time share (%)")
    ax.legend(loc="upper right", frameon=False, ncol=2, handletextpad=0.5, columnspacing=0.8)
    add_panel_label(ax, "c")

    ax = axes[1, 1]
    subset = contextual_stress_df.copy()
    x = np.arange(len(subset))
    values = subset["RH_gt90_pct"].to_numpy()
    ax.bar(
        x,
        values,
        color=[color_map[(row["scenario"], row["case"])] for _, row in subset.iterrows()],
        edgecolor="white",
        linewidth=0.5,
    )
    style_axes(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{row['scenario']} {row['case']}" for _, row in subset.iterrows()], rotation=0)
    ax.set_ylabel(r"RH > 90% time (%)")
    add_panel_label(ax, "d")

    fig.tight_layout(pad=0.8, w_pad=1.0, h_pad=1.0)
    save_figure(fig, out_path)
    plt.close(fig)


def _build_controller_model_selection_figure(
    out_path: Path,
    residual_gated_df: pd.DataFrame,
    controller_constant_df: pd.DataFrame,
) -> None:
    apply_academic_style()
    df = controller_constant_df.copy()
    df["net_profit_k"] = df["net_profit"] / 1000.0
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.2))

    _build_grouped_controller_bars(
        axes[0, 0],
        df,
        "best",
        "net_profit_k",
        r"Best-schedule profit (10$^3$ RMB)",
        annotate_value_col="total_harvests",
    )
    axes[0, 0].legend(loc="upper left", frameon=False, ncol=2, handletextpad=0.5, columnspacing=0.8)
    add_panel_label(axes[0, 0], "a")

    _build_grouped_controller_bars(
        axes[0, 1],
        df,
        "default",
        "net_profit_k",
        r"Default-schedule profit (10$^3$ RMB)",
    )
    add_panel_label(axes[0, 1], "b")

    ax = axes[1, 0]
    ordered = residual_gated_df.copy()
    x = np.arange(len(ordered))
    ax.bar(
        x,
        100.0 + ordered["gated_vs_residual_profit_pct"],
        color=COLORS["teal"],
        edgecolor="white",
        linewidth=0.5,
    )
    style_axes(ax)
    ax.axhline(100.0, color=COLORS["ink"], linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(ordered["scenario"], rotation=0)
    ax.set_ylabel("Gated profit vs residual (%)")
    add_panel_label(ax, "c")

    ax = axes[1, 1]
    width = 0.34
    x = np.arange(len(ordered))
    ax.bar(
        x - width / 2.0,
        ordered["residual_best_harvests"],
        width=width,
        color=COLORS["brick"],
        edgecolor="white",
        linewidth=0.5,
        label="Residual best",
    )
    ax.bar(
        x + width / 2.0,
        ordered["gated_best_harvests"],
        width=width,
        color=COLORS["teal"],
        edgecolor="white",
        linewidth=0.5,
        label="Gated best",
    )
    style_axes(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(ordered["scenario"], rotation=0)
    ax.set_ylabel("Best-schedule harvest events")
    ax.legend(loc="upper left", frameon=False, ncol=1, handletextpad=0.5)
    add_panel_label(ax, "d")

    fig.tight_layout(pad=0.8, w_pad=1.0, h_pad=1.0)
    save_figure(fig, out_path)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pid_map, residual_map, gated_map, climate_map, contextual_map = _load_all_maps()

    integrity_df = _collect_integrity_table()
    metadata_nulls_df = _collect_metadata_nulls(contextual_map)
    summary_df, delta_df = _collect_contextual_summary_rows(contextual_map)
    eta_df, eta_avg_df = _collect_eta_table(contextual_map)
    controller_constant_df, same_schedule_df = _collect_controller_constant_rows(
        pid_map, residual_map, gated_map, climate_map, contextual_map
    )
    mechanism_hourly_df, mechanism_metrics_df, mechanism_relative_df = _collect_mechanism_tables(
        pid_map, residual_map, gated_map, climate_map, contextual_map
    )
    risk_dist_df, contextual_stress_df = _collect_risk_tables(
        pid_map, residual_map, gated_map, climate_map, contextual_map
    )
    residual_gated_df, _ = _collect_model_selection_tables(
        pid_map, residual_map, gated_map, climate_map, contextual_map
    )

    _build_contextual_core_summary_figure(
        OUT_DIR / "compact_contextual_core_summary.png",
        contextual_map,
        summary_df,
        delta_df,
    )
    _build_contextual_sensitivity_heatmap(
        OUT_DIR / "compact_contextual_upper_sensitivity_heatmap.png",
        eta_avg_df,
    )
    _build_contextual_controller_comparison_figure(
        OUT_DIR / "compact_contextual_controller_comparison.png",
        controller_constant_df,
        risk_dist_df,
    )
    _build_contextual_mechanism_figure(
        OUT_DIR / "compact_contextual_mechanism_profiles.png",
        mechanism_hourly_df,
        mechanism_metrics_df,
        mechanism_relative_df,
    )
    _build_contextual_tail_risk_figure(
        OUT_DIR / "compact_contextual_tail_risk.png",
        risk_dist_df,
        contextual_stress_df,
    )
    _build_controller_model_selection_figure(
        OUT_DIR / "compact_controller_model_selection.png",
        residual_gated_df,
        controller_constant_df,
    )

    integrity_df.to_csv(OUT_DIR / "compact_contextual_integrity.csv", index=False, encoding="utf-8-sig")
    metadata_nulls_df.to_csv(OUT_DIR / "compact_contextual_metadata_nulls.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUT_DIR / "compact_contextual_summary.csv", index=False, encoding="utf-8-sig")
    delta_df.to_csv(OUT_DIR / "compact_contextual_deltas.csv", index=False, encoding="utf-8-sig")
    eta_df.to_csv(OUT_DIR / "compact_contextual_upper_sensitivity_eta2.csv", index=False, encoding="utf-8-sig")
    eta_avg_df.to_csv(OUT_DIR / "compact_contextual_upper_sensitivity_average.csv", index=False, encoding="utf-8-sig")
    controller_constant_df.to_csv(OUT_DIR / "compact_contextual_controller_comparison.csv", index=False, encoding="utf-8-sig")
    same_schedule_df.to_csv(OUT_DIR / "compact_contextual_same_schedule_comparison.csv", index=False, encoding="utf-8-sig")
    mechanism_hourly_df.to_csv(OUT_DIR / "compact_contextual_default_hourly_profile.csv", index=False, encoding="utf-8-sig")
    mechanism_metrics_df.to_csv(OUT_DIR / "compact_contextual_default_mechanism_metrics.csv", index=False, encoding="utf-8-sig")
    mechanism_relative_df.to_csv(OUT_DIR / "compact_contextual_default_relative_changes.csv", index=False, encoding="utf-8-sig")
    risk_dist_df.to_csv(OUT_DIR / "compact_contextual_risk_distribution.csv", index=False, encoding="utf-8-sig")
    contextual_stress_df.to_csv(OUT_DIR / "compact_contextual_stress_cases.csv", index=False, encoding="utf-8-sig")
    residual_gated_df.to_csv(OUT_DIR / "compact_controller_model_selection_summary.csv", index=False, encoding="utf-8-sig")

    summary_payload = {
        "scenarios": [spec["label"] for spec in SCENARIOS],
        "output_dir": str(OUT_DIR),
        "best_schedules": {
            row["scenario"]: row["schedule_key"]
            for _, row in summary_df[summary_df["status"] == "best"].iterrows()
        },
    }
    (OUT_DIR / "compact_contextual_result_review_summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[OK] Saved compact contextual result-review figures to: {OUT_DIR}")


if __name__ == "__main__":
    main()
