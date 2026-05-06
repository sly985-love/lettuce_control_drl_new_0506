# -*- coding: utf-8 -*-
"""Build a journal-grade schedule evidence catalog for the PFAL mainline."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLED", "true")


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from envs.schedule_sampler import ScheduleSampler  # noqa: E402
from envs.utils import load_all_configs, prepare_runtime_config  # noqa: E402
from experiments.exact_pid_schedule_baseline import (  # noqa: E402
    FEASIBLE_CSV_DEFAULT,
    _normalise_feasible_catalog_df,
    schedule_key as schedule_key_from_schedule,
)
from rl.drl_based_control import load_schedule_bounds  # noqa: E402


def _coerce_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    return value


def _build_bounds(schedule_params: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "t1_min": int(schedule_params.get("t1_min", 10)),
        "t1_max": int(schedule_params.get("t1_max", 18)),
        "t2_min": int(schedule_params.get("t2_min", 10)),
        "t2_max": int(schedule_params.get("t2_max", 18)),
        "N1_min": int(schedule_params.get("N1_min", 8)),
        "N1_max": int(schedule_params.get("N1_max", 20)),
        "rho2_min": float(schedule_params.get("rho2_min", 20.0)),
        "rho2_max": float(schedule_params.get("rho2_max", 52.0)),
        "PP_fixed": int(schedule_params.get("PP_fixed", 16)),
        "rho1_min": float(schedule_params.get("rho1_min", 72.0)),
        "rho1_max": float(schedule_params.get("rho1_max", 144.0)),
        "N_total": int(schedule_params.get("N_total", 80)),
        "A_board": float(schedule_params.get("A_board", 1.0)),
        "er_min": float(schedule_params.get("er_min", 3.0)),
        "er_max": float(schedule_params.get("er_max", 6.0)),
        "total_cycle_min": float(schedule_params.get("total_cycle_min", 24.0)),
        "total_cycle_max": float(schedule_params.get("total_cycle_max", 32.0)),
        "DT_values": list(
            schedule_params.get(
                "DT_values",
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
            )
        ),
    }


def _schedule_key_from_row(row: Dict[str, Any]) -> str:
    return schedule_key_from_schedule(
        {
            "t1": int(row["t1"]),
            "t2": int(row["t2"]),
            "N1": int(row["N1"]),
            "rho2": float(row["rho2"]),
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the schedule evidence catalog.")
    parser.add_argument(
        "--feasible-csv",
        type=str,
        default=str(FEASIBLE_CSV_DEFAULT),
        help="Structural feasible schedule CSV.",
    )
    parser.add_argument("--pid-csv", type=str, default=None, help="Exact PID result CSV.")
    parser.add_argument("--rl-main-csv", type=str, default=None, help="Exact RL-main result CSV.")
    parser.add_argument(
        "--rl-ablation-csv",
        type=str,
        default=None,
        help="Exact RL-ablation result CSV.",
    )
    parser.add_argument(
        "--robust-csv",
        type=str,
        default=None,
        help="Optional robustness summary CSV to merge.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT / "results" / "schedule_evidence_catalog"),
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help="Optional explicit output CSV path. Defaults to <out-dir>/schedule_evidence_catalog.csv",
    )
    return parser.parse_args()


def load_structural_catalog(feasible_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(feasible_csv)
    bounds = load_schedule_bounds(str(ROOT / "configs" / "schedule_params.yaml"))
    work = _normalise_feasible_catalog_df(df, bounds=bounds)
    work["schedule_key"] = work.apply(lambda row: _schedule_key_from_row(row.to_dict()), axis=1)
    if "PP" in work.columns:
        work["fixed_photoperiod_hours"] = pd.to_numeric(work["PP"], errors="coerce").fillna(bounds["PP_fixed"]).astype(int)
    else:
        work["fixed_photoperiod_hours"] = int(bounds["PP_fixed"])
    work["is_default_schedule"] = (
        (work["t1"].astype(int) == 14)
        & (work["t2"].astype(int) == 14)
        & (work["N1"].astype(int) == 20)
        & (work["rho2"].astype(float).round().astype(int) == 36)
    )
    keep_cols = [
        "schedule_key",
        "is_default_schedule",
        "t1",
        "t2",
        "N1",
        "N2",
        "rho2",
        "rho1",
        "A1_m2",
        "A2_m2",
        "A1_A2_ratio",
        "expansion_ratio",
        "total_cycle_days",
        "delta_t",
        "fixed_photoperiod_hours",
    ]
    if "PP" in work.columns:
        keep_cols.append("PP")
    return work[keep_cols].copy()


def compute_reference_catalog() -> tuple[pd.DataFrame, Dict[str, Any]]:
    cfg = load_all_configs(str(ROOT / "configs"))
    runtime = prepare_runtime_config(cfg)
    bounds = _build_bounds(dict(cfg.get("schedule_params", {}) or {}))
    sampler = ScheduleSampler(
        bounds,
        container_params=runtime["container_params"],
        crop_params=runtime["crop_params"],
        reward_params=runtime["reward_params"],
        steady_state_params=runtime["steady_state_params"],
    )
    rows = sampler.get_all_feasible_schedules(include_reference=True)
    ref_df = pd.DataFrame(rows)
    if ref_df.empty:
        raise RuntimeError("Reference sampler returned no schedules.")
    ref_df["schedule_key"] = ref_df.apply(lambda row: _schedule_key_from_row(row.to_dict()), axis=1)
    keep_cols = [
        "schedule_key",
        "reference_feasibility_class",
        "reference_min_feasible",
        "reference_target_feasible",
        "reference_harvest_dry_mass_per_plant_g",
        "reference_harvest_fresh_mass_per_plant_g",
        "reference_harvest_vs_min_ratio",
        "reference_harvest_vs_target_ratio",
    ]
    keep_cols = [c for c in keep_cols if c in ref_df.columns]
    summary = {
        "reference_class_counts": sampler.get_reference_class_counts(),
        "config_warnings": list(runtime.get("config_warnings", []) or []),
        "fixed_photoperiod_hours": int(bounds["PP_fixed"]),
    }
    return ref_df[keep_cols].drop_duplicates(subset=["schedule_key"], keep="last").reset_index(drop=True), summary


def _prepare_exact_df(
    csv_path: Path,
    *,
    prefix: str,
    harvest_min_dry: float,
    harvest_target_dry: float,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise RuntimeError(f"No rows found in exact result CSV: {csv_path}")
    if "schedule_key" not in df.columns:
        df["schedule_key"] = df.apply(lambda row: _schedule_key_from_row(row.to_dict()), axis=1)
    df = df.sort_values(by=["schedule_key"]).drop_duplicates(subset=["schedule_key"], keep="last").reset_index(drop=True)
    df["valid_full_horizon"] = _coerce_bool(df["valid_full_horizon"])
    df["terminated_early"] = _coerce_bool(df["terminated_early"]) if "terminated_early" in df.columns else False
    avg_dry = pd.to_numeric(df.get("avg_harvest_dry_g_per_plant"), errors="coerce").fillna(0.0)
    avg_fresh = pd.to_numeric(df.get("avg_harvest_fresh_g_per_plant"), errors="coerce").fillna(0.0)
    df[f"{prefix}_exact_harvest_vs_min_ratio"] = avg_dry / max(float(harvest_min_dry), 1.0e-9)
    df[f"{prefix}_exact_harvest_vs_target_ratio"] = avg_dry / max(float(harvest_target_dry), 1.0e-9)
    df[f"{prefix}_exact_min_feasible"] = df["valid_full_horizon"] & (avg_dry >= float(harvest_min_dry))
    df[f"{prefix}_exact_target_feasible"] = df["valid_full_horizon"] & (avg_dry >= float(harvest_target_dry))

    mapping = {
        "objective_value": f"{prefix}_exact_objective_value",
        "net_profit": f"{prefix}_exact_net_profit_rmb",
        "revenue": f"{prefix}_exact_revenue_rmb",
        "total_cost": f"{prefix}_exact_total_cost_rmb",
        "energy_kwh": f"{prefix}_exact_energy_kwh",
        "harvest_fresh_kg": f"{prefix}_exact_harvest_fw_kg",
        "harvest_dry_kg": f"{prefix}_exact_harvest_dry_kg",
        "avg_harvest_fresh_g_per_plant": f"{prefix}_exact_avg_harvest_fw_g_per_plant",
        "avg_harvest_dry_g_per_plant": f"{prefix}_exact_avg_harvest_dry_g_per_plant",
        "episode_completion_ratio": f"{prefix}_exact_episode_completion_ratio",
        "termination_reason": f"{prefix}_exact_termination_reason",
        "valid_full_horizon": f"{prefix}_exact_valid_full_horizon",
        "terminated_early": f"{prefix}_exact_terminated_early",
        "cost_per_kg": f"{prefix}_exact_cost_per_kg_rmb",
        "revenue_per_kg": f"{prefix}_exact_revenue_per_kg_rmb",
    }
    keep = ["schedule_key", *mapping.keys(), f"{prefix}_exact_harvest_vs_min_ratio", f"{prefix}_exact_harvest_vs_target_ratio", f"{prefix}_exact_min_feasible", f"{prefix}_exact_target_feasible"]
    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy()
    return out.rename(columns={k: v for k, v in mapping.items() if k in out.columns})


def merge_exact_results(
    base_df: pd.DataFrame,
    *,
    pid_csv: Path | None,
    rl_main_csv: Path | None,
    rl_ablation_csv: Path | None,
    harvest_min_dry: float,
    harvest_target_dry: float,
) -> pd.DataFrame:
    merged = base_df.copy()
    if pid_csv is not None and pid_csv.exists():
        pid_df = _prepare_exact_df(
            pid_csv,
            prefix="pid",
            harvest_min_dry=harvest_min_dry,
            harvest_target_dry=harvest_target_dry,
        )
        merged = merged.merge(pid_df, on="schedule_key", how="left")
    if rl_main_csv is not None and rl_main_csv.exists():
        rl_df = _prepare_exact_df(
            rl_main_csv,
            prefix="rl_main",
            harvest_min_dry=harvest_min_dry,
            harvest_target_dry=harvest_target_dry,
        )
        merged = merged.merge(rl_df, on="schedule_key", how="left")
    if rl_ablation_csv is not None and rl_ablation_csv.exists():
        rl_ablation_df = _prepare_exact_df(
            rl_ablation_csv,
            prefix="rl_ablation",
            harvest_min_dry=harvest_min_dry,
            harvest_target_dry=harvest_target_dry,
        )
        merged = merged.merge(rl_ablation_df, on="schedule_key", how="left")
    return merged


def merge_robust_summary(base_df: pd.DataFrame, robust_csv: Path | None) -> pd.DataFrame:
    if robust_csv is None or not robust_csv.exists():
        return base_df
    robust_df = pd.read_csv(robust_csv)
    if robust_df.empty:
        return base_df
    if "schedule_key" not in robust_df.columns:
        robust_df["schedule_key"] = robust_df.apply(lambda row: _schedule_key_from_row(row.to_dict()), axis=1)
    robust_df = robust_df.drop_duplicates(subset=["schedule_key"], keep="last").reset_index(drop=True)
    merge_cols = [c for c in robust_df.columns if c != "schedule_key"]
    return base_df.merge(robust_df[["schedule_key", *merge_cols]], on="schedule_key", how="left")


def add_delta_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    pairs = [
        ("rl_main_exact_net_profit_rmb", "pid_exact_net_profit_rmb", "delta_profit_rl_main_vs_pid"),
        ("rl_main_exact_energy_kwh", "pid_exact_energy_kwh", "delta_energy_rl_main_vs_pid"),
        ("rl_main_exact_harvest_fw_kg", "pid_exact_harvest_fw_kg", "delta_harvest_fw_rl_main_vs_pid"),
        ("rl_main_exact_cost_per_kg_rmb", "pid_exact_cost_per_kg_rmb", "delta_cost_per_kg_rl_main_vs_pid"),
        ("rl_main_exact_revenue_per_kg_rmb", "pid_exact_revenue_per_kg_rmb", "delta_revenue_per_kg_rl_main_vs_pid"),
        ("rl_main_exact_harvest_vs_min_ratio", "pid_exact_harvest_vs_min_ratio", "delta_harvest_vs_min_ratio_rl_main_vs_pid"),
        ("rl_main_exact_harvest_vs_target_ratio", "pid_exact_harvest_vs_target_ratio", "delta_harvest_vs_target_ratio_rl_main_vs_pid"),
    ]
    for left, right, target in pairs:
        if left in out.columns and right in out.columns:
            out[target] = pd.to_numeric(out[left], errors="coerce") - pd.to_numeric(out[right], errors="coerce")
    if "delta_profit_rl_main_vs_pid" in out.columns:
        out["rl_main_wins_by_profit_vs_pid"] = pd.to_numeric(out["delta_profit_rl_main_vs_pid"], errors="coerce") > 0.0
    return out


def build_summary(df: pd.DataFrame, reference_summary: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "n_schedules": int(len(df)),
        "reference_summary": _json_safe(reference_summary),
    }
    if "reference_feasibility_class" in df.columns:
        summary["reference_feasibility_class_counts"] = {
            str(k): int(v)
            for k, v in df["reference_feasibility_class"].fillna("unknown").value_counts().items()
        }
    for prefix in ("pid", "rl_main", "rl_ablation"):
        profit_col = f"{prefix}_exact_net_profit_rmb"
        min_col = f"{prefix}_exact_min_feasible"
        if profit_col in df.columns:
            summary[f"{prefix}_exact_profit_mean"] = float(pd.to_numeric(df[profit_col], errors="coerce").dropna().mean())
        if min_col in df.columns:
            bool_series = _coerce_bool(df[min_col])
            summary[f"{prefix}_exact_min_feasible_count"] = int(bool_series.sum())
            summary[f"{prefix}_exact_min_feasible_rate"] = float(bool_series.mean())
    robust_cols = [c for c in df.columns if c.startswith("robust_")]
    if robust_cols:
        summary["robust_columns"] = list(sorted(robust_cols))
    return summary


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    feasible_csv = Path(args.feasible_csv).resolve()
    base_df = load_structural_catalog(feasible_csv)
    reference_df, reference_summary = compute_reference_catalog()
    merged = base_df.merge(reference_df, on="schedule_key", how="left")

    cfg = load_all_configs(str(ROOT / "configs"))
    reward_params = dict(cfg.get("reward_params", {}) or {})
    harvest_min_dry = float(reward_params.get("harvest_min_dry_mass_per_plant", 4.44))
    harvest_target_dry = float(reward_params.get("harvest_target_dry_mass_per_plant", 5.33))

    merged = merge_exact_results(
        merged,
        pid_csv=Path(args.pid_csv).resolve() if args.pid_csv else None,
        rl_main_csv=Path(args.rl_main_csv).resolve() if args.rl_main_csv else None,
        rl_ablation_csv=Path(args.rl_ablation_csv).resolve() if args.rl_ablation_csv else None,
        harvest_min_dry=harvest_min_dry,
        harvest_target_dry=harvest_target_dry,
    )
    merged = merge_robust_summary(merged, Path(args.robust_csv).resolve() if args.robust_csv else None)
    merged = add_delta_columns(merged)
    merged = merged.sort_values(by=["t1", "t2", "N1", "rho2"]).reset_index(drop=True)

    out_csv = Path(args.out_csv).resolve() if args.out_csv else (out_dir / "schedule_evidence_catalog.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False, encoding="utf-8")

    summary = build_summary(merged, reference_summary)
    summary_path = out_dir / "schedule_evidence_catalog_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# Schedule Evidence Catalog",
        "",
        f"- Structural schedule count: `{summary['n_schedules']}`",
        f"- Fixed photoperiod: `{reference_summary.get('fixed_photoperiod_hours', 16)} h`",
        "",
    ]
    if "reference_feasibility_class_counts" in summary:
        md_lines.append("## Reference Class Counts")
        md_lines.append("")
        md_lines.append("| Class | Count |")
        md_lines.append("| --- | ---: |")
        for key, value in summary["reference_feasibility_class_counts"].items():
            md_lines.append(f"| {key} | {value} |")
        md_lines.append("")
    for prefix, label in (("pid", "PID"), ("rl_main", "RL main"), ("rl_ablation", "RL ablation")):
        key = f"{prefix}_exact_min_feasible_rate"
        if key in summary:
            md_lines.append(f"- {label} exact min-feasible rate: `{summary[key]:.4f}`")
    (out_dir / "schedule_evidence_catalog_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[SAVE] CSV  -> {out_csv}")
    print(f"[SAVE] JSON -> {summary_path}")
    print(f"[INFO] n_schedules={len(merged)} | reference_counts={summary.get('reference_feasibility_class_counts', {})}")


if __name__ == "__main__":
    main()
