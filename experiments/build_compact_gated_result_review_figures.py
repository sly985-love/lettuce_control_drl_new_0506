# -*- coding: utf-8 -*-
"""Build compact figures and summary tables for gated residual PID RL results."""

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
PID_RESULTS_ROOT = Path(r"C:\Users\29341\Desktop\fsdownload\results")
RESIDUAL_RESULTS_ROOT = Path(r"C:\Users\29341\Desktop\fsdownload\results_residual_pid_sac")
GATED_RESULTS_ROOT = Path(r"C:\Users\29341\Desktop\fsdownload\results_gated_residual_pid_sac")
OUT_DIR = ROOT / "paper" / "figures_result_review_gated_20260427"

DEFAULT_SCHEDULE_KEY = "t1=14|t2=14|N1=20|rho2=36"
REASONABLE_SCHEDULE_KEY = "t1=13|t2=13|N1=16|rho2=23"
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
DELTA_COLORS = {
    "profit_gain_pct": COLORS["navy"],
    "harvest_gain_pct": COLORS["teal"],
    "energy_change_pct": COLORS["gold"],
    "cost_reduction_pct": COLORS["plum"],
}

SCENARIOS = [
    {
        "label": "Const-L20",
        "gated_name": "exp03_exact_gatedrespid_rl_pp16h_constant_e0p74_co20p54_l20",
        "residual_name": "exp03_exact_rl_baseline_pp16h_constant_e0p74_co20p54_l20",
        "pid_name": "exp02_exact_pid_baseline_pp16h_i1250_i2300_pm_constant_e0p74_co20p54_l20",
        "price_level": "L20",
        "mode": "segmented",
    },
    {
        "label": "Daily-Const-L20",
        "gated_name": "exp03_exact_gatedrespid_rl_pp16h_constant_e0p74_co20p54_l20_daily",
        "residual_name": "exp03_exact_rl_baseline_pp16h_constant_e0p74_co20p54_l20_daily",
        "pid_name": None,
        "price_level": "L20",
        "mode": "daily",
    },
    {
        "label": "Const-L40",
        "gated_name": "exp03_exact_gatedrespid_rl_pp16h_constant_e0p74_co20p54_l40",
        "residual_name": "exp03_exact_rl_baseline_pp16h_constant_e0p74_co20p54_l40",
        "pid_name": "exp02_exact_pid_baseline_pp16h_i1250_i2300_pm_constant_e0p74_co20p54_l40",
        "price_level": "L40",
        "mode": "segmented",
    },
    {
        "label": "TOU-L20",
        "gated_name": "exp03_exact_gatedrespid_rl_pp16h_tou_zhejiang_lt1kv_l20",
        "residual_name": "exp03_exact_rl_baseline_pp16h_tou_zhejiang_lt1kv_l20",
        "pid_name": "exp02_exact_pid_baseline_pp16h_i1250_i2300_pm_tou_zhejiang_lt1kv_l20",
        "price_level": "L20",
        "mode": "segmented",
    },
    {
        "label": "TOU-L40",
        "gated_name": "exp03_exact_gatedrespid_rl_pp16h_tou_zhejiang_lt1kv_l40",
        "residual_name": "exp03_exact_rl_baseline_pp16h_tou_zhejiang_lt1kv_l40",
        "pid_name": "exp02_exact_pid_baseline_pp16h_i1250_i2300_pm_tou_zhejiang_lt1kv_l40",
        "price_level": "L40",
        "mode": "segmented",
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


def _load_schedule_df(exp_dir: Path, prefix: str) -> pd.DataFrame:
    for candidate in [
        exp_dir / f"{prefix}_schedule_results_ranked.csv",
        exp_dir / f"{prefix}_schedule_results.csv",
    ]:
        if candidate.exists():
            return pd.read_csv(candidate)

    shard_paths = sorted(exp_dir.glob(f"{prefix}_schedule_results.shard_*.csv"))
    if shard_paths:
        return pd.concat([pd.read_csv(path) for path in shard_paths], ignore_index=True)

    ranked_shards = sorted(exp_dir.glob(f"{prefix}_schedule_results_ranked.shard_*.csv"))
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


def _control_quality_metrics(trace_df: pd.DataFrame) -> dict[str, float]:
    dt_h = float(trace_df["step_size_s"].iloc[0]) / 3600.0
    horizon_h = len(trace_df) * dt_h
    metrics = {}
    for col in ["I1", "I2", "Q_HVAC", "u_CO2", "m_dehum"]:
        metrics[f"tv_{col}_per_h"] = float(np.abs(np.diff(trace_df[col].to_numpy())).sum() / horizon_h)
    metrics["mean_temp_c"] = float(trace_df["T_in"].mean())
    metrics["mean_rh_pct"] = float(trace_df["RH_pct"].mean())
    metrics["mean_vpd_kpa"] = float(trace_df["VPD_kPa"].mean())
    metrics["sd_temp_c"] = float(trace_df["T_in"].std())
    metrics["sd_rh_pct"] = float(trace_df["RH_pct"].std())
    metrics["sd_vpd_kpa"] = float(trace_df["VPD_kPa"].std())
    return metrics


def _energy_from_trace(trace_df: pd.DataFrame, power_col: str) -> float:
    return float((trace_df[power_col] * trace_df["step_size_s"] / 3600.0).sum() / 1000.0)


def _trace_summary(trace_df: pd.DataFrame) -> dict[str, float]:
    summary = {
        "I1_mean": float(trace_df["I1"].mean()),
        "I2_mean": float(trace_df["I2"].mean()),
        "led_mwh": _energy_from_trace(trace_df, "P_LED_total_kW"),
        "hvac_mwh": _energy_from_trace(trace_df, "P_HVAC_kW"),
        "dehum_mwh": _energy_from_trace(trace_df, "P_dehum_kW"),
        "transp_kg": float(trace_df["E_transp_kg"].sum()),
        "dehum_removed_kg": float(trace_df["dehum_removed_kg"].sum()),
    }
    summary.update(_control_quality_metrics(trace_df))
    return summary


def _collect_integrity_table() -> pd.DataFrame:
    rows = []
    for spec in SCENARIOS:
        exp_dir = GATED_RESULTS_ROOT / spec["gated_name"]
        rows.append(
            {
                "scenario": spec["label"],
                "folder_name": spec["gated_name"],
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


def _collect_metadata_nulls(gated_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
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
    for label, df in gated_map.items():
        row = {"scenario": label}
        for col in meta_cols:
            row[f"{col}_nonnull"] = int(df[col].notna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _collect_gated_summary_rows(gated_map: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    delta_rows = []
    for spec in SCENARIOS:
        label = spec["label"]
        df = gated_map[label]
        best = df.sort_values("net_profit", ascending=False).iloc[0]
        default = df[df["is_default_schedule"] == True].iloc[0]
        worst = df.sort_values("net_profit", ascending=True).iloc[0]
        for kind, row in [("best", best), ("default", default), ("worst", worst)]:
            summary_rows.append(
                {
                    "scenario": label,
                    "kind": kind,
                    "schedule_key": row["schedule_key"],
                    "net_profit": float(row["net_profit"]),
                    "harvest_fresh_kg": float(row["harvest_fresh_kg"]),
                    "energy_kwh": float(row["energy_kwh"]),
                    "cost_per_kg": float(row["cost_per_kg"]),
                    "total_harvests": int(row["total_harvests"]),
                    "avg_harvest_fresh_g_per_plant": float(row["avg_harvest_fresh_g_per_plant"]),
                    "rank_valid_profit": int(row.get("rank_valid_profit", np.nan))
                    if pd.notna(row.get("rank_valid_profit", np.nan))
                    else np.nan,
                }
            )
        delta_rows.append(
            {
                "scenario": label,
                "profit_gain_pct": 100.0 * (best["net_profit"] - default["net_profit"]) / default["net_profit"],
                "harvest_gain_pct": 100.0
                * (best["harvest_fresh_kg"] - default["harvest_fresh_kg"])
                / default["harvest_fresh_kg"],
                "energy_change_pct": 100.0 * (best["energy_kwh"] - default["energy_kwh"]) / default["energy_kwh"],
                "cost_reduction_pct": 100.0 * (default["cost_per_kg"] - best["cost_per_kg"]) / default["cost_per_kg"],
                "default_rank": int(default.get("rank_valid_profit", np.nan))
                if pd.notna(default.get("rank_valid_profit", np.nan))
                else np.nan,
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(delta_rows)


def _collect_eta_table(gated_map: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for label, df in gated_map.items():
        for variable in UPPER_VARS:
            for metric in METRICS:
                rows.append(
                    {
                        "scenario": label,
                        "variable": variable,
                        "metric": metric,
                        "eta2": _compute_eta2(df, variable, metric),
                    }
                )
    eta_df = pd.DataFrame(rows)
    eta_avg = (
        eta_df.groupby(["variable", "metric"], as_index=False)["eta2"]
        .mean()
        .pivot(index="variable", columns="metric", values="eta2")
        .loc[UPPER_VARS, METRICS]
    )
    eta_avg = eta_avg.loc[eta_avg["net_profit"].sort_values(ascending=False).index]
    return eta_df, eta_avg.reset_index().rename(columns={"index": "variable"})


def _collect_regularization_rows(
    residual_map: dict[str, pd.DataFrame],
    gated_map: dict[str, pd.DataFrame],
    pid_map: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for spec in SCENARIOS:
        label = spec["label"]
        residual_df = residual_map[label].sort_values("net_profit", ascending=False).reset_index(drop=True)
        gated_df = gated_map[label].sort_values("net_profit", ascending=False).reset_index(drop=True)
        residual_df["rank_profit"] = residual_df.index + 1
        gated_df["rank_profit"] = gated_df.index + 1
        residual_best = residual_df.iloc[0]
        gated_best = gated_df.iloc[0]
        pid_best_profit = np.nan
        pid_best_schedule = ""
        if spec["pid_name"] is not None:
            pid_df = pid_map[label]
            pid_best = pid_df.sort_values("net_profit", ascending=False).iloc[0]
            pid_best_profit = float(pid_best["net_profit"])
            pid_best_schedule = str(pid_best["schedule_key"])

        top10_residual = residual_df.head(10)
        top10_gated = gated_df.head(10)
        rows.append(
            {
                "scenario": label,
                "mode": spec["mode"],
                "price_level": spec["price_level"],
                "pid_best_schedule": pid_best_schedule,
                "pid_best_profit": pid_best_profit,
                "residual_best_schedule": residual_best["schedule_key"],
                "gated_best_schedule": gated_best["schedule_key"],
                "residual_best_profit": float(residual_best["net_profit"]),
                "gated_best_profit": float(gated_best["net_profit"]),
                "best_profit_gap_gated_vs_residual_pct": 100.0
                * (gated_best["net_profit"] - residual_best["net_profit"])
                / residual_best["net_profit"],
                "residual_best_harvests": int(residual_best["total_harvests"]),
                "gated_best_harvests": int(gated_best["total_harvests"]),
                "residual_top10_mean_harvests": float(top10_residual["total_harvests"].mean()),
                "gated_top10_mean_harvests": float(top10_gated["total_harvests"].mean()),
                "residual_top10_median_harvests": float(top10_residual["total_harvests"].median()),
                "gated_top10_median_harvests": float(top10_gated["total_harvests"].median()),
                "residual_best_rank_in_gated": int(
                    gated_df.loc[gated_df["schedule_key"] == residual_best["schedule_key"], "rank_profit"].iloc[0]
                ),
                "gated_best_rank_in_residual": int(
                    residual_df.loc[residual_df["schedule_key"] == gated_best["schedule_key"], "rank_profit"].iloc[0]
                ),
            }
        )
    return pd.DataFrame(rows)


def _collect_same_schedule_rows(
    residual_map: dict[str, pd.DataFrame],
    gated_map: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for spec in SCENARIOS:
        label = spec["label"]
        residual_df = residual_map[label]
        gated_df = gated_map[label]
        residual_best_key = str(residual_df.sort_values("net_profit", ascending=False).iloc[0]["schedule_key"])
        gated_best_key = str(gated_df.sort_values("net_profit", ascending=False).iloc[0]["schedule_key"])

        for case, schedule_key in [
            ("default", DEFAULT_SCHEDULE_KEY),
            ("gated_best", gated_best_key),
            ("residual_best", residual_best_key),
        ]:
            residual_row = residual_df[residual_df["schedule_key"] == schedule_key].iloc[0]
            gated_row = gated_df[gated_df["schedule_key"] == schedule_key].iloc[0]
            rows.append(
                {
                    "scenario": label,
                    "mode": spec["mode"],
                    "schedule_case": case,
                    "schedule_key": schedule_key,
                    "profit_delta_pct": 100.0
                    * (gated_row["net_profit"] - residual_row["net_profit"])
                    / residual_row["net_profit"],
                    "yield_delta_pct": 100.0
                    * (gated_row["harvest_fresh_kg"] - residual_row["harvest_fresh_kg"])
                    / residual_row["harvest_fresh_kg"],
                    "energy_delta_pct": 100.0
                    * (gated_row["energy_kwh"] - residual_row["energy_kwh"])
                    / residual_row["energy_kwh"],
                    "cost_delta_pct": 100.0
                    * (gated_row["cost_per_kg"] - residual_row["cost_per_kg"])
                    / residual_row["cost_per_kg"],
                    "residual_harvests": int(residual_row["total_harvests"]),
                    "gated_harvests": int(gated_row["total_harvests"]),
                }
            )
    return pd.DataFrame(rows)


def _collect_tou_shift_rows(
    residual_map: dict[str, pd.DataFrame],
    gated_map: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for controller_name, data_map in [("residual", residual_map), ("gated", gated_map)]:
        for price_level in ["L20", "L40"]:
            const_df = data_map[f"Const-{price_level}"].copy()
            tou_df = data_map[f"TOU-{price_level}"].copy()

            merged = const_df.merge(tou_df, on="schedule_key", suffixes=("_const", "_tou"))
            rank_const = const_df.sort_values("net_profit", ascending=False).reset_index()[["schedule_key"]]
            rank_const["rank_const"] = rank_const.index + 1
            rank_tou = tou_df.sort_values("net_profit", ascending=False).reset_index()[["schedule_key"]]
            rank_tou["rank_tou"] = rank_tou.index + 1
            ranks = rank_const.merge(rank_tou, on="schedule_key")

            rows.append(
                {
                    "controller": controller_name,
                    "price_level": price_level,
                    "mean_profit_delta_rmb": float((merged["net_profit_tou"] - merged["net_profit_const"]).mean()),
                    "mean_cost_delta_rmb": float((merged["total_cost_tou"] - merged["total_cost_const"]).mean()),
                    "mean_energy_delta_kwh": float((merged["energy_kwh_tou"] - merged["energy_kwh_const"]).mean()),
                    "mean_yield_delta_kg": float((merged["harvest_fresh_kg_tou"] - merged["harvest_fresh_kg_const"]).mean()),
                    "mean_abs_rank_shift": float((ranks["rank_tou"] - ranks["rank_const"]).abs().mean()),
                    "max_abs_rank_shift": int((ranks["rank_tou"] - ranks["rank_const"]).abs().max()),
                }
            )
    return pd.DataFrame(rows)


def _collect_mechanism_tables(
    residual_map: dict[str, pd.DataFrame],
    gated_map: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    residual_const_l40_dir = RESIDUAL_RESULTS_ROOT / "exp03_exact_rl_baseline_pp16h_constant_e0p74_co20p54_l40"
    gated_const_l40_dir = GATED_RESULTS_ROOT / "exp03_exact_gatedrespid_rl_pp16h_constant_e0p74_co20p54_l40"
    gated_tou_l40_dir = GATED_RESULTS_ROOT / "exp03_exact_gatedrespid_rl_pp16h_tou_zhejiang_lt1kv_l40"

    reasonable_res_trace = _load_trace_df(
        residual_const_l40_dir,
        REASONABLE_SCHEDULE_KEY,
        usecols=["datetime", "I1", "I2", "step_size_s", "P_LED_total_kW", "P_HVAC_kW", "P_dehum_kW", "E_transp_kg", "dehum_removed_kg", "T_in", "RH_pct", "VPD_kPa", "Q_HVAC", "u_CO2", "m_dehum"],
    )
    reasonable_gated_trace = _load_trace_df(
        gated_const_l40_dir,
        REASONABLE_SCHEDULE_KEY,
        usecols=["datetime", "I1", "I2", "step_size_s", "P_LED_total_kW", "P_HVAC_kW", "P_dehum_kW", "E_transp_kg", "dehum_removed_kg", "T_in", "RH_pct", "VPD_kPa", "Q_HVAC", "u_CO2", "m_dehum"],
    )
    aggressive_res_trace = _load_trace_df(
        residual_const_l40_dir,
        AGGRESSIVE_SCHEDULE_KEY,
        usecols=["datetime", "I1", "I2", "step_size_s", "P_LED_total_kW", "P_HVAC_kW", "P_dehum_kW", "E_transp_kg", "dehum_removed_kg", "T_in", "RH_pct", "VPD_kPa", "Q_HVAC", "u_CO2", "m_dehum"],
    )
    aggressive_gated_trace = _load_trace_df(
        gated_const_l40_dir,
        AGGRESSIVE_SCHEDULE_KEY,
        usecols=["datetime", "I1", "I2", "step_size_s", "P_LED_total_kW", "P_HVAC_kW", "P_dehum_kW", "E_transp_kg", "dehum_removed_kg", "T_in", "RH_pct", "VPD_kPa", "Q_HVAC", "u_CO2", "m_dehum"],
    )
    gated_tou_trace = _load_trace_df(
        gated_tou_l40_dir,
        REASONABLE_SCHEDULE_KEY,
        usecols=["datetime", "I1", "I2", "elec_price_rmb_kwh", "step_size_s", "P_LED_total_kW", "P_HVAC_kW", "P_dehum_kW", "E_transp_kg", "dehum_removed_kg", "T_in", "RH_pct", "VPD_kPa", "Q_HVAC", "u_CO2", "m_dehum"],
    )

    reasonable_hourly = (
        _hourly_profile(reasonable_res_trace, ["I1", "I2"])
        .rename(columns={"I1": "I1_residual", "I2": "I2_residual"})
        .merge(
            _hourly_profile(reasonable_gated_trace, ["I1", "I2"]).rename(
                columns={"I1": "I1_gated", "I2": "I2_gated"}
            ),
            on="hour",
            how="inner",
        )
    )
    aggressive_hourly = (
        _hourly_profile(aggressive_res_trace, ["I1", "I2"])
        .rename(columns={"I1": "I1_residual", "I2": "I2_residual"})
        .merge(
            _hourly_profile(aggressive_gated_trace, ["I1", "I2"]).rename(
                columns={"I1": "I1_gated", "I2": "I2_gated"}
            ),
            on="hour",
            how="inner",
        )
    )
    tou_hourly = (
        _hourly_profile(reasonable_gated_trace.assign(elec_price_rmb_kwh=0.74), ["I1", "elec_price_rmb_kwh"])
        .rename(columns={"I1": "I1_const", "elec_price_rmb_kwh": "price_const"})
        .merge(
            _hourly_profile(gated_tou_trace, ["I1", "elec_price_rmb_kwh"]).rename(
                columns={"I1": "I1_tou", "elec_price_rmb_kwh": "price_tou"}
            ),
            on="hour",
            how="inner",
        )
    )

    case_rows = []
    case_specs = [
        ("reasonable_residual", "Const-L40", "residual", REASONABLE_SCHEDULE_KEY, reasonable_res_trace),
        ("reasonable_gated", "Const-L40", "gated", REASONABLE_SCHEDULE_KEY, reasonable_gated_trace),
        ("aggressive_residual", "Const-L40", "residual", AGGRESSIVE_SCHEDULE_KEY, aggressive_res_trace),
        ("aggressive_gated", "Const-L40", "gated", AGGRESSIVE_SCHEDULE_KEY, aggressive_gated_trace),
        ("default_residual", "Const-L40", "residual", DEFAULT_SCHEDULE_KEY, _load_trace_df(residual_const_l40_dir, DEFAULT_SCHEDULE_KEY)),
        ("default_gated", "Const-L40", "gated", DEFAULT_SCHEDULE_KEY, _load_trace_df(gated_const_l40_dir, DEFAULT_SCHEDULE_KEY)),
        ("reasonable_gated_tou", "TOU-L40", "gated", REASONABLE_SCHEDULE_KEY, gated_tou_trace),
    ]
    for case_id, scenario_label, controller, schedule_key, trace_df in case_specs:
        source_df = residual_map[scenario_label] if controller == "residual" else gated_map[scenario_label]
        summary_row = source_df[source_df["schedule_key"] == schedule_key].iloc[0]
        row = {
            "case_id": case_id,
            "scenario": scenario_label,
            "controller": controller,
            "schedule_key": schedule_key,
            "net_profit": float(summary_row["net_profit"]),
            "harvest_fresh_kg": float(summary_row["harvest_fresh_kg"]),
            "energy_kwh": float(summary_row["energy_kwh"]),
            "cost_per_kg": float(summary_row["cost_per_kg"]),
            "total_harvests": int(summary_row["total_harvests"]),
            "avg_harvest_fresh_g_per_plant": float(summary_row["avg_harvest_fresh_g_per_plant"]),
        }
        row.update(_trace_summary(trace_df))
        case_rows.append(row)

    return (
        reasonable_hourly,
        aggressive_hourly,
        tou_hourly,
        pd.DataFrame(case_rows),
    )


def _build_gated_core_summary_figure(
    out_path: Path,
    gated_map: dict[str, pd.DataFrame],
    summary_df: pd.DataFrame,
    delta_df: pd.DataFrame,
) -> None:
    scenario_labels = [spec["label"] for spec in SCENARIOS]

    fig = plt.figure(figsize=(12.0, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.32)

    ax0 = fig.add_subplot(gs[0, 0])
    box_data = [gated_map[label]["net_profit"].to_numpy() / 1.0e4 for label in scenario_labels]
    bp = ax0.boxplot(
        box_data,
        tick_labels=scenario_labels,
        patch_artist=True,
        widths=0.52,
        showfliers=False,
    )
    for patch in bp["boxes"]:
        patch.set(facecolor="white", edgecolor=COLORS["gray"], linewidth=1.0)
    for key in ["whiskers", "caps"]:
        for artist in bp[key]:
            artist.set(color=COLORS["gray"], linewidth=0.9)
    for median in bp["medians"]:
        median.set(color=COLORS["ink"], linewidth=1.3)

    for x_pos, label in enumerate(scenario_labels, start=1):
        sub = summary_df[summary_df["scenario"] == label]
        best = float(sub.loc[sub["kind"] == "best", "net_profit"].iloc[0]) / 1.0e4
        default = float(sub.loc[sub["kind"] == "default", "net_profit"].iloc[0]) / 1.0e4
        worst = float(sub.loc[sub["kind"] == "worst", "net_profit"].iloc[0]) / 1.0e4
        ax0.plot([x_pos, x_pos], [default, best], color=COLORS["light_gray"], linewidth=1.2, zorder=2)
        ax0.scatter([x_pos], [best], marker="D", s=36, color=COLORS["navy"], zorder=3, label="Best" if x_pos == 1 else None)
        ax0.scatter([x_pos], [default], marker="s", s=32, facecolor="white", edgecolor=COLORS["brick"], linewidth=1.0, zorder=3, label="Default" if x_pos == 1 else None)
        ax0.scatter([x_pos], [worst], marker="o", s=20, color=COLORS["gray"], zorder=3, label="Worst" if x_pos == 1 else None)
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
    ordered = delta_df.set_index("scenario").loc[scenario_labels]
    for idx, (col, label) in enumerate(metric_specs):
        ax1.bar(
            x + (idx - 1.5) * width,
            ordered[col].to_numpy(),
            width=width,
            color=DELTA_COLORS[col],
            label=label,
            edgecolor="none",
        )
    ax1.axhline(0.0, color=COLORS["gray"], linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenario_labels)
    ax1.set_xlabel("Scenario")
    ax1.set_ylabel("Relative change vs. gated default (%)")
    ax1.legend(frameon=False, ncol=2, loc="upper left", handletextpad=0.5, columnspacing=1.2, borderaxespad=0.2)
    style_axes(ax1, grid_axis="y")
    add_panel_label(ax1, "b")

    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def _build_gated_sensitivity_heatmap(out_path: Path, eta_avg_df: pd.DataFrame) -> None:
    eta_avg = eta_avg_df.set_index("variable").loc[:, METRICS]

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    im = ax.imshow(eta_avg.to_numpy(), cmap="cividis", aspect="auto")
    ax.set_xticks(np.arange(len(METRICS)))
    ax.set_xticklabels(["Net profit", "Fresh yield", "Electricity use", "Cost per kg"])
    ax.set_yticks(np.arange(len(eta_avg.index)))
    ax.set_yticklabels([VAR_LABELS[var] for var in eta_avg.index])
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


def _build_gated_regularization_figure(out_path: Path, regularization_df: pd.DataFrame) -> None:
    plot_df = regularization_df.copy()
    labels = plot_df["scenario"].tolist()
    x = np.arange(len(labels))
    width = 0.34

    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.8))

    ax0 = axes[0]
    ax0.bar(x - width / 2.0, plot_df["residual_best_profit"] / 1.0e4, width=width, color=COLORS["plum"], label="Residual best", edgecolor="none")
    ax0.bar(x + width / 2.0, plot_df["gated_best_profit"] / 1.0e4, width=width, color=COLORS["navy"], label="Gated best", edgecolor="none")
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, rotation=12)
    ax0.set_xlabel("Scenario")
    ax0.set_ylabel(r"Best annual net profit ($10^4$ CNY)")
    style_axes(ax0, grid_axis="y")
    add_panel_label(ax0, "a")
    ax0.legend(frameon=False, loc="upper left", handletextpad=0.5)

    ax1 = axes[1]
    ax1.bar(x - width / 2.0, plot_df["residual_best_harvests"], width=width, color=COLORS["plum"], label="Residual best", edgecolor="none")
    ax1.bar(x + width / 2.0, plot_df["gated_best_harvests"], width=width, color=COLORS["navy"], label="Gated best", edgecolor="none")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=12)
    ax1.set_xlabel("Scenario")
    ax1.set_ylabel("Annual harvest events at the best schedule (-)")
    style_axes(ax1, grid_axis="y")
    add_panel_label(ax1, "b")

    ax2 = axes[2]
    ax2.bar(x - width / 2.0, plot_df["residual_top10_mean_harvests"], width=width, color=COLORS["plum"], label="Residual top-10", edgecolor="none")
    ax2.bar(x + width / 2.0, plot_df["gated_top10_mean_harvests"], width=width, color=COLORS["navy"], label="Gated top-10", edgecolor="none")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=12)
    ax2.set_xlabel("Scenario")
    ax2.set_ylabel("Mean annual harvest events in the top-10 schedules (-)")
    style_axes(ax2, grid_axis="y")
    add_panel_label(ax2, "c")

    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def _build_gated_same_schedule_figure(out_path: Path, same_schedule_df: pd.DataFrame) -> None:
    metric_specs = [
        ("profit_delta_pct", "Net profit"),
        ("yield_delta_pct", "Fresh yield"),
        ("energy_delta_pct", "Electricity"),
        ("cost_delta_pct", "Cost per kg"),
    ]
    metric_colors = {
        "profit_delta_pct": COLORS["navy"],
        "yield_delta_pct": COLORS["teal"],
        "energy_delta_pct": COLORS["gold"],
        "cost_delta_pct": COLORS["plum"],
    }
    case_order = [
        ("default", "Scenario\n(fixed default schedule)"),
        ("gated_best", "Scenario\n(fixed gated-best schedule)"),
        ("residual_best", "Scenario\n(fixed residual-best schedule)"),
    ]
    scenario_labels = [spec["label"] for spec in SCENARIOS]
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.9), sharey=True)

    values_all = same_schedule_df[[col for col, _ in metric_specs]].to_numpy().ravel()
    y_min = min(float(np.min(values_all)), 0.0)
    y_max = max(float(np.max(values_all)), 0.0)
    y_pad = 0.08 * max(y_max - y_min, 6.0)

    for panel_idx, (ax, (case, xlabel)) in enumerate(zip(axes, case_order)):
        sub = same_schedule_df[same_schedule_df["schedule_case"] == case].set_index("scenario").loc[scenario_labels]
        x = np.arange(len(scenario_labels))
        width = 0.18
        for metric_idx, (col, label) in enumerate(metric_specs):
            ax.bar(
                x + (metric_idx - 1.5) * width,
                sub[col].to_numpy(),
                width=width,
                color=metric_colors[col],
                edgecolor="none",
                label=label,
            )
        ax.axhline(0.0, color=COLORS["gray"], linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_labels, rotation=12)
        ax.set_xlabel(xlabel)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        style_axes(ax, grid_axis="y")
        add_panel_label(ax, chr(ord("a") + panel_idx))

    axes[0].set_ylabel("Relative change, gated vs. residual (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, columnspacing=1.2, handletextpad=0.5)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save_figure(fig, out_path)
    plt.close(fig)


def _build_gated_tou_figure(out_path: Path, tou_shift_df: pd.DataFrame) -> None:
    levels = ["L20", "L40"]
    x = np.arange(len(levels))
    width = 0.34

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))

    ax0 = axes[0]
    for offset, controller, color in [(-width / 2.0, "residual", COLORS["plum"]), (width / 2.0, "gated", COLORS["navy"])]:
        sub = tou_shift_df[tou_shift_df["controller"] == controller].set_index("price_level").loc[levels]
        ax0.bar(x + offset, sub["mean_abs_rank_shift"], width=width, color=color, edgecolor="none", label=controller.capitalize())
    ax0.set_xticks(x)
    ax0.set_xticklabels(levels)
    ax0.set_xlabel("Lettuce-price level")
    ax0.set_ylabel("Mean absolute ranking shift: TOU vs. constant (-)")
    style_axes(ax0, grid_axis="y")
    add_panel_label(ax0, "a")
    ax0.legend(frameon=False, loc="upper right", handletextpad=0.5)

    ax1 = axes[1]
    for offset, controller, color in [(-width / 2.0, "residual", COLORS["plum"]), (width / 2.0, "gated", COLORS["navy"])]:
        sub = tou_shift_df[tou_shift_df["controller"] == controller].set_index("price_level").loc[levels]
        ax1.bar(x + offset, sub["mean_profit_delta_rmb"] / 1.0e4, width=width, color=color, edgecolor="none", label=controller.capitalize())
    ax1.axhline(0.0, color=COLORS["gray"], linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(levels)
    ax1.set_xlabel("Lettuce-price level")
    ax1.set_ylabel(r"Mean TOU-induced profit change ($10^4$ CNY)")
    style_axes(ax1, grid_axis="y")
    add_panel_label(ax1, "b")

    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def _style_secondary_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_color(COLORS["ink"])
    ax.spines["right"].set_linewidth(0.8)
    ax.tick_params(axis="y", which="major", colors=COLORS["ink"], pad=2.5)
    ax.tick_params(axis="y", which="minor", colors=COLORS["gray"])
    ax.grid(False)


def _build_gated_mechanism_figure(
    out_path: Path,
    reasonable_hourly: pd.DataFrame,
    aggressive_hourly: pd.DataFrame,
    tou_hourly: pd.DataFrame,
) -> None:
    fig = plt.figure(figsize=(13.6, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.05, 1.10], wspace=0.34)

    for panel_idx, (ax, hourly_df, xlabel) in enumerate(
        [
            (
                fig.add_subplot(gs[0, 0]),
                reasonable_hourly,
                "Time of day (h)\n(constant price, schedule 13-13-16-23)",
            ),
            (
                fig.add_subplot(gs[0, 1]),
                aggressive_hourly,
                "Time of day (h)\n(constant price, schedule 15-13-15-23)",
            ),
        ]
    ):
        ax.axvspan(8.0, 24.0, color=COLORS["sand"], alpha=0.55, zorder=0)
        ax.plot(hourly_df["hour"], hourly_df["I1_residual"], color=COLORS["blue"], linestyle="--", linewidth=1.5, label=r"Residual, $I_1$")
        ax.plot(hourly_df["hour"], hourly_df["I1_gated"], color=COLORS["blue"], linewidth=1.8, label=r"Gated, $I_1$")
        ax.plot(hourly_df["hour"], hourly_df["I2_residual"], color=COLORS["brick"], linestyle="--", linewidth=1.5, label=r"Residual, $I_2$")
        ax.plot(hourly_df["hour"], hourly_df["I2_gated"], color=COLORS["brick"], linewidth=1.8, label=r"Gated, $I_2$")
        ax.set_ylim(0.0, 320.0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"Zone-specific PPFD ($\mu$mol m$^{-2}$ s$^{-1}$)")
        set_hour_ticks(ax)
        style_axes(ax, grid_axis="y")
        add_panel_label(ax, chr(ord("a") + panel_idx))
        if panel_idx == 0:
            ax.legend(frameon=False, ncol=2, loc="upper left", handletextpad=0.5, columnspacing=1.0)

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axvspan(8.0, 24.0, color=COLORS["sand"], alpha=0.55, zorder=0)
    ax2.plot(tou_hourly["hour"], tou_hourly["I1_const"], color=COLORS["light_gray"], linewidth=1.8, label=r"Constant-price gated, $I_1$")
    ax2.plot(tou_hourly["hour"], tou_hourly["I1_tou"], color=COLORS["plum"], linewidth=1.9, label=r"TOU-aware gated, $I_1$")
    ax2.set_ylim(0.0, 280.0)
    ax2.set_xlabel("Time of day (h)\n(gated controller, schedule 13-13-16-23)")
    ax2.set_ylabel(r"Dense-zone PPFD, $I_1$ ($\mu$mol m$^{-2}$ s$^{-1}$)")
    set_hour_ticks(ax2)
    style_axes(ax2, grid_axis="y")
    add_panel_label(ax2, "c")

    ax2b = ax2.twinx()
    ax2b.plot(tou_hourly["hour"], tou_hourly["price_tou"], color=COLORS["gray"], linestyle=":", linewidth=1.4, label="TOU electricity price")
    ax2b.set_ylabel(r"Electricity price (CNY kWh$^{-1}$)")
    ax2b.set_ylim(0.0, max(1.25, float(tou_hourly["price_tou"].max()) * 1.08))
    _style_secondary_axis(ax2b)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, frameon=False, fontsize=7.8, loc="upper left", handletextpad=0.5)

    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def main() -> None:
    apply_academic_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gated_map: dict[str, pd.DataFrame] = {}
    residual_map: dict[str, pd.DataFrame] = {}
    pid_map: dict[str, pd.DataFrame] = {}
    for spec in SCENARIOS:
        gated_map[spec["label"]] = _load_schedule_df(GATED_RESULTS_ROOT / spec["gated_name"], prefix="rl_exact")
        residual_map[spec["label"]] = _load_schedule_df(RESIDUAL_RESULTS_ROOT / spec["residual_name"], prefix="rl_exact")
        if spec["pid_name"] is not None:
            pid_map[spec["label"]] = _load_schedule_df(PID_RESULTS_ROOT / spec["pid_name"], prefix="pid_exact")

    integrity_df = _collect_integrity_table()
    metadata_nulls_df = _collect_metadata_nulls(gated_map)
    summary_df, delta_df = _collect_gated_summary_rows(gated_map)
    eta_df, eta_avg_df = _collect_eta_table(gated_map)
    regularization_df = _collect_regularization_rows(residual_map, gated_map, pid_map)
    same_schedule_df = _collect_same_schedule_rows(residual_map, gated_map)
    tou_shift_df = _collect_tou_shift_rows(residual_map, gated_map)
    reasonable_hourly_df, aggressive_hourly_df, tou_hourly_df, case_metrics_df = _collect_mechanism_tables(residual_map, gated_map)

    _build_gated_core_summary_figure(
        OUT_DIR / "compact_gated_core_summary.png",
        gated_map=gated_map,
        summary_df=summary_df,
        delta_df=delta_df,
    )
    _build_gated_sensitivity_heatmap(
        OUT_DIR / "compact_gated_upper_sensitivity_heatmap.png",
        eta_avg_df=eta_avg_df,
    )
    _build_gated_regularization_figure(
        OUT_DIR / "compact_gated_regularization_summary.png",
        regularization_df=regularization_df,
    )
    _build_gated_same_schedule_figure(
        OUT_DIR / "compact_gated_same_schedule_comparison.png",
        same_schedule_df=same_schedule_df,
    )
    _build_gated_tou_figure(
        OUT_DIR / "compact_gated_tou_sensitivity.png",
        tou_shift_df=tou_shift_df,
    )
    _build_gated_mechanism_figure(
        OUT_DIR / "compact_gated_mechanism_profiles.png",
        reasonable_hourly=reasonable_hourly_df,
        aggressive_hourly=aggressive_hourly_df,
        tou_hourly=tou_hourly_df,
    )

    payload = {
        "integrity": integrity_df.to_dict(orient="records"),
        "metadata_nulls": metadata_nulls_df.to_dict(orient="records"),
        "gated_summary": summary_df.to_dict(orient="records"),
        "gated_delta_vs_default": delta_df.to_dict(orient="records"),
        "gated_eta2": eta_df.to_dict(orient="records"),
        "gated_eta2_average": eta_avg_df.to_dict(orient="records"),
        "regularization": regularization_df.to_dict(orient="records"),
        "same_schedule_comparison": same_schedule_df.to_dict(orient="records"),
        "tou_shift_comparison": tou_shift_df.to_dict(orient="records"),
        "case_metrics": case_metrics_df.to_dict(orient="records"),
    }
    (OUT_DIR / "compact_gated_result_review_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    integrity_df.to_csv(OUT_DIR / "compact_gated_integrity.csv", index=False, encoding="utf-8-sig")
    metadata_nulls_df.to_csv(OUT_DIR / "compact_gated_metadata_nulls.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUT_DIR / "compact_gated_summary.csv", index=False, encoding="utf-8-sig")
    delta_df.to_csv(OUT_DIR / "compact_gated_deltas.csv", index=False, encoding="utf-8-sig")
    eta_df.to_csv(OUT_DIR / "compact_gated_upper_sensitivity_eta2.csv", index=False, encoding="utf-8-sig")
    eta_avg_df.to_csv(OUT_DIR / "compact_gated_upper_sensitivity_average.csv", index=False, encoding="utf-8-sig")
    regularization_df.to_csv(OUT_DIR / "compact_gated_regularization_summary.csv", index=False, encoding="utf-8-sig")
    same_schedule_df.to_csv(OUT_DIR / "compact_gated_same_schedule_comparison.csv", index=False, encoding="utf-8-sig")
    tou_shift_df.to_csv(OUT_DIR / "compact_gated_tou_sensitivity.csv", index=False, encoding="utf-8-sig")
    reasonable_hourly_df.to_csv(OUT_DIR / "compact_gated_mechanism_reasonable_hourly.csv", index=False, encoding="utf-8-sig")
    aggressive_hourly_df.to_csv(OUT_DIR / "compact_gated_mechanism_aggressive_hourly.csv", index=False, encoding="utf-8-sig")
    tou_hourly_df.to_csv(OUT_DIR / "compact_gated_mechanism_tou_hourly.csv", index=False, encoding="utf-8-sig")
    case_metrics_df.to_csv(OUT_DIR / "compact_gated_case_metrics.csv", index=False, encoding="utf-8-sig")

    print(f"[OK] Saved compact gated result-review figures to: {OUT_DIR}")


if __name__ == "__main__":
    main()
