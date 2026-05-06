# -*- coding: utf-8 -*-
"""
Export a top-k schedule shortlist CSV from an existing ranking/result table.

Typical use:

  python experiments/export_schedule_shortlist.py ^
    --source-csv results/approx_pid_search_fast/approx_stage3_exact.csv ^
    --topk 8 ^
    --out-csv results/approx_pid_search_fast/shortlist_top8.csv

The output CSV is compatible with `exact_pid_schedule_baseline.py` and
`exact_rl_schedule_baseline.py` via `--feasible-csv`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_SORT_CANDIDATES = (
    "annualized_profit",
    "net_profit",
    "stage_objective",
    "objective_value",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a top-k schedule shortlist CSV from a ranking/result table."
    )
    parser.add_argument(
        "--source-csv",
        required=True,
        help="Input CSV such as approx_stage3_exact.csv or pid_exact_schedule_results_ranked.csv.",
    )
    parser.add_argument(
        "--out-csv",
        required=True,
        help="Output shortlist CSV path.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=8,
        help="How many top schedules to keep.",
    )
    parser.add_argument(
        "--sort-by",
        default="auto",
        help="Metric used for ranking. Default: auto-detect annualized_profit/net_profit/stage_objective/objective_value.",
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Sort ascending instead of descending.",
    )
    parser.add_argument(
        "--include-default",
        action="store_true",
        help="Always keep the default schedule if present.",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional JSON summary path. Default: <out-csv stem>_summary.json",
    )
    return parser.parse_args()


def resolve_sort_key(df: pd.DataFrame, requested: str) -> str:
    if requested and requested != "auto":
        if requested not in df.columns:
            raise RuntimeError(f"Requested sort key '{requested}' not found in source CSV.")
        return requested
    for key in DEFAULT_SORT_CANDIDATES:
        if key in df.columns:
            return key
    raise RuntimeError(
        "Could not auto-detect a ranking column. "
        f"Tried: {', '.join(DEFAULT_SORT_CANDIDATES)}"
    )


def main() -> None:
    args = parse_args()
    source_csv = Path(args.source_csv).resolve()
    out_csv = Path(args.out_csv).resolve()
    summary_json = (
        Path(args.summary_json).resolve()
        if args.summary_json
        else out_csv.with_name(f"{out_csv.stem}_summary.json")
    )

    df = pd.read_csv(source_csv)
    if df.empty:
        raise RuntimeError(f"Source CSV is empty: {source_csv}")

    required_cols = ["t1", "t2", "N1", "rho2"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Source CSV missing required schedule columns: {', '.join(missing)}"
        )

    sort_key = resolve_sort_key(df, args.sort_by)
    work = df.copy()
    if "schedule_key" not in work.columns:
        work["schedule_key"] = work.apply(
            lambda row: (
                f"t1={int(row['t1'])}|t2={int(row['t2'])}|"
                f"N1={int(row['N1'])}|rho2={int(round(float(row['rho2'])))}"
            ),
            axis=1,
        )

    work = work.drop_duplicates(subset=["schedule_key"], keep="first")
    work = work.sort_values(by=sort_key, ascending=bool(args.ascending)).reset_index(drop=True)

    selected = work.head(max(int(args.topk), 0)).copy()
    if args.include_default and "is_default_schedule" in work.columns:
        default_subset = work[work["is_default_schedule"].astype(bool)]
        if not default_subset.empty:
            default_row = default_subset.iloc[[0]]
            selected = (
                pd.concat([selected, default_row], ignore_index=True)
                .drop_duplicates(subset=["schedule_key"], keep="first")
                .reset_index(drop=True)
            )

    if selected.empty:
        raise RuntimeError("No schedules selected for shortlist export.")

    keep_cols = [
        c
        for c in [
            "schedule_key",
            "t1",
            "t2",
            "N1",
            "N2",
            "rho2",
            "rho1",
            "A1_m2",
            "A2_m2",
            "A1",
            "A2",
            "is_default_schedule",
            sort_key,
        ]
        if c in selected.columns
    ]
    shortlist = selected[keep_cols].copy()
    shortlist["source_rank"] = range(1, len(shortlist) + 1)
    shortlist["source_metric_name"] = sort_key
    shortlist["source_metric_value"] = selected[sort_key].astype(float).values

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    shortlist.to_csv(out_csv, index=False, encoding="utf-8")

    summary: dict[str, Any] = {
        "source_csv": str(source_csv),
        "out_csv": str(out_csv),
        "topk_requested": int(args.topk),
        "n_exported": int(len(shortlist)),
        "sort_key": str(sort_key),
        "ascending": bool(args.ascending),
        "include_default": bool(args.include_default),
        "schedule_keys": shortlist["schedule_key"].astype(str).tolist(),
    }
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("Schedule Shortlist Export")
    print("=" * 72)
    print(f"Source CSV         : {source_csv}")
    print(f"Sort key           : {sort_key}")
    print(f"Exported schedules : {len(shortlist)}")
    print(f"Shortlist CSV      : {out_csv}")
    print(f"Summary JSON       : {summary_json}")
    print("=" * 72)


if __name__ == "__main__":
    main()
