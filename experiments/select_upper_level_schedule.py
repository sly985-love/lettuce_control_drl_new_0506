# -*- coding: utf-8 -*-
"""Select a publication-grade upper-level schedule from an evidence catalog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _coerce_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select the upper-level schedule from a catalog.")
    parser.add_argument("--catalog-csv", type=str, required=True)
    parser.add_argument(
        "--controller-scope",
        type=str,
        default="rl_main",
        choices=["pid", "rl_main", "rl_ablation"],
    )
    parser.add_argument(
        "--objective-mode",
        type=str,
        default="auto",
        choices=["auto", "robust_profit_mean", "exact_net_profit"],
    )
    parser.add_argument("--require-reference-min-feasible", action="store_true")
    parser.add_argument("--require-reference-target-feasible", action="store_true")
    parser.add_argument("--require-controller-exact-min-feasible", action="store_true")
    parser.add_argument("--require-controller-exact-target-feasible", action="store_true")
    parser.add_argument("--min-robust-min-pass-rate", type=float, default=None)
    parser.add_argument("--min-robust-target-pass-rate", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--out-dir", type=str, default=None)
    return parser.parse_args()


def _select_objective_column(df: pd.DataFrame, controller_scope: str, objective_mode: str) -> str:
    exact_col = f"{controller_scope}_exact_net_profit_rmb"
    robust_col = f"robust_profit_mean_{controller_scope}"
    if objective_mode == "robust_profit_mean":
        if robust_col not in df.columns:
            raise RuntimeError(f"Requested robust objective column missing: {robust_col}")
        return robust_col
    if objective_mode == "exact_net_profit":
        if exact_col not in df.columns:
            raise RuntimeError(f"Requested exact objective column missing: {exact_col}")
        return exact_col
    if robust_col in df.columns:
        return robust_col
    if exact_col in df.columns:
        return exact_col
    raise RuntimeError(
        f"Could not resolve objective column for controller={controller_scope}. "
        f"Expected one of: {robust_col}, {exact_col}"
    )


def _apply_bool_constraint(df: pd.DataFrame, column: str, required: bool) -> pd.DataFrame:
    if not required:
        return df
    if column not in df.columns:
        raise RuntimeError(f"Required constraint column missing: {column}")
    mask = _coerce_bool(df[column])
    return df[mask].copy()


def _apply_numeric_floor(df: pd.DataFrame, column: str, floor_value: float | None) -> pd.DataFrame:
    if floor_value is None:
        return df
    if column not in df.columns:
        raise RuntimeError(f"Required robustness column missing: {column}")
    mask = pd.to_numeric(df[column], errors="coerce").fillna(0.0) >= float(floor_value)
    return df[mask].copy()


def _schedule_payload(row: pd.Series) -> dict[str, Any]:
    payload = {
        "schedule_key": str(row["schedule_key"]),
        "t1": int(row["t1"]),
        "t2": int(row["t2"]),
        "N1": int(row["N1"]),
        "rho2": int(round(float(row["rho2"]))),
    }
    payload["schedule_string"] = (
        f"t1={payload['t1']},t2={payload['t2']},N1={payload['N1']},rho2={payload['rho2']}"
    )
    return payload


def main() -> None:
    args = parse_args()
    catalog_csv = Path(args.catalog_csv).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else catalog_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(catalog_csv)
    if df.empty:
        raise RuntimeError(f"Catalog is empty: {catalog_csv}")
    if "schedule_key" not in df.columns:
        raise RuntimeError(f"Catalog missing schedule_key: {catalog_csv}")

    objective_col = _select_objective_column(df, args.controller_scope, args.objective_mode)
    work = df.copy()
    work = _apply_bool_constraint(work, "reference_min_feasible", args.require_reference_min_feasible)
    work = _apply_bool_constraint(work, "reference_target_feasible", args.require_reference_target_feasible)
    work = _apply_bool_constraint(
        work,
        f"{args.controller_scope}_exact_min_feasible",
        args.require_controller_exact_min_feasible,
    )
    work = _apply_bool_constraint(
        work,
        f"{args.controller_scope}_exact_target_feasible",
        args.require_controller_exact_target_feasible,
    )
    work = _apply_numeric_floor(
        work,
        f"robust_min_pass_rate_{args.controller_scope}",
        args.min_robust_min_pass_rate,
    )
    work = _apply_numeric_floor(
        work,
        f"robust_target_pass_rate_{args.controller_scope}",
        args.min_robust_target_pass_rate,
    )

    if work.empty:
        raise RuntimeError(
            "No schedules remain after applying the current upper-level selection constraints."
        )

    tie_breakers = [objective_col]
    for col in [
        f"robust_min_pass_rate_{args.controller_scope}",
        f"{args.controller_scope}_exact_net_profit_rmb",
        f"{args.controller_scope}_exact_harvest_vs_target_ratio",
        f"{args.controller_scope}_exact_energy_kwh",
    ]:
        if col in work.columns and col not in tie_breakers:
            tie_breakers.append(col)
    ascending = [False] * len(tie_breakers)
    if tie_breakers and tie_breakers[-1].endswith("_energy_kwh"):
        ascending[-1] = True
    ranked = work.sort_values(by=tie_breakers, ascending=ascending).reset_index(drop=True)

    top_k = max(int(args.top_k), 1)
    shortlist = ranked.head(top_k).copy()
    selected = shortlist.iloc[0]
    payload = {
        "controller_scope": str(args.controller_scope),
        "objective_column": str(objective_col),
        "objective_mode_requested": str(args.objective_mode),
        "selection_constraints": {
            "require_reference_min_feasible": bool(args.require_reference_min_feasible),
            "require_reference_target_feasible": bool(args.require_reference_target_feasible),
            "require_controller_exact_min_feasible": bool(args.require_controller_exact_min_feasible),
            "require_controller_exact_target_feasible": bool(args.require_controller_exact_target_feasible),
            "min_robust_min_pass_rate": args.min_robust_min_pass_rate,
            "min_robust_target_pass_rate": args.min_robust_target_pass_rate,
        },
        "n_candidates_after_filtering": int(len(work)),
        "selected_schedule": _schedule_payload(selected),
        "selected_objective_value": float(pd.to_numeric(pd.Series([selected[objective_col]]), errors="coerce").iloc[0]),
    }

    json_path = out_dir / "upper_level_schedule_selection.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    shortlist_csv = out_dir / "upper_level_schedule_shortlist.csv"
    shortlist.to_csv(shortlist_csv, index=False, encoding="utf-8")

    md_lines = [
        "# Upper-Level Schedule Selection",
        "",
        f"- Controller scope: `{args.controller_scope}`",
        f"- Objective column: `{objective_col}`",
        f"- Candidates after filtering: `{len(work)}`",
        f"- Selected schedule: `{payload['selected_schedule']['schedule_key']}`",
        f"- Selected schedule string: `{payload['selected_schedule']['schedule_string']}`",
        f"- Objective value: `{payload['selected_objective_value']:.4f}`",
        "",
        "## Top Shortlist",
        "",
        "| Rank | Schedule key | Objective |",
        "| --- | --- | ---: |",
    ]
    for idx, (_, row) in enumerate(shortlist.iterrows(), start=1):
        md_lines.append(
            f"| {idx} | {row['schedule_key']} | {float(pd.to_numeric(pd.Series([row[objective_col]]), errors='coerce').iloc[0]):.4f} |"
        )
    (out_dir / "upper_level_schedule_selection.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[SAVE] JSON -> {json_path}")
    print(f"[SAVE] CSV  -> {shortlist_csv}")
    print(f"[INFO] Selected schedule: {payload['selected_schedule']['schedule_string']}")


if __name__ == "__main__":
    main()
