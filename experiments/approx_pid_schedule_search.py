# -*- coding: utf-8 -*-
"""
Multi-fidelity approximate PID schedule search over the feasible recipe set.

This script is designed as a practical alternative to exhaustive 365-day exact
evaluation when the user needs a near-optimal schedule quickly. It uses a
successive-halving style workflow:

1. Evaluate all feasible schedules with a short-horizon proxy simulation.
2. Promote the best schedules to a medium-horizon refinement stage.
3. Exact-evaluate only a small finalist set over the full 365-day horizon.

The final recommendation is still approximate, because only the finalist set is
fully evaluated. That is intentional: the goal is to get a strong near-optimal
candidate quickly, plus a ranked candidate set for downstream RL planning.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LOCAL_FEASIBLE_CSV = ROOT / "data" / "feasibility" / "feasible_solutions.csv"
LEGACY_FEASIBLE_CSV = ROOT.parent / "results" / "feasibility" / "feasible_solutions.csv"
FEASIBLE_CSV_DEFAULT = LOCAL_FEASIBLE_CSV if LOCAL_FEASIBLE_CSV.exists() else LEGACY_FEASIBLE_CSV

from experiments.exact_pid_schedule_baseline import (  # noqa: E402
    DEFAULT_SCHEDULE,
    build_result_row,
    evaluate_schedule,
    is_default_schedule,
    load_feasible_schedules,
    load_weather_window,
    schedule_key,
)


ANNUAL_DAYS = 364.0
STAGE_NAMES = ("stage1_proxy", "stage2_refine", "stage3_exact")
SUMMARY_PLOT_NAME = "approx_pid_search_summary.png"

_WEATHER_CACHE: Dict[tuple[str, str, float, float], List[Dict[str, Any]]] = {}


def _load_weather_window_cached(
    weather_path: str,
    start_date: str,
    duration_days: float,
    dt_seconds: float,
) -> List[Dict[str, Any]]:
    key = (str(weather_path), str(start_date), float(duration_days), float(dt_seconds))
    if key not in _WEATHER_CACHE:
        _WEATHER_CACHE[key] = load_weather_window(
            Path(weather_path),
            start_date,
            duration_days,
            dt_seconds,
        )
    return _WEATHER_CACHE[key]


def annualize_metric(value: float, sim_days: float) -> float:
    if sim_days <= 1e-9:
        return 0.0
    return float(value) * ANNUAL_DAYS / float(sim_days)


def stage_sort_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    ranked = df.sort_values(
        by=[
            "stage_objective",
            "annualized_profit",
            "annualized_harvest_fresh_kg",
            "annualized_energy_kwh",
        ],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    ranked["stage_rank"] = ranked.index + 1
    return ranked


def _evaluate_stage_task(task: Dict[str, Any]) -> Dict[str, Any]:
    schedule = dict(task["schedule"])
    duration_days = float(task["duration_days"])
    dt_seconds = float(task["dt_seconds"])
    weather_rows = _load_weather_window_cached(
        str(task["weather_path"]),
        str(task["start_date"]),
        duration_days,
        dt_seconds,
    )
    summary = evaluate_schedule(
        schedule,
        weather_rows,
        dt_seconds,
        int(task["seed"]),
        photo_period_manual=task.get("photo_period_manual"),
        i1_manual=task.get("I1_manual"),
        i2_manual=task.get("I2_manual"),
    )
    row = build_result_row(
        schedule,
        summary,
        eval_index=int(task["eval_index"]),
        duration_days=duration_days,
        dt_seconds=dt_seconds,
    )
    sim_days = max(float(row.get("sim_days_executed", duration_days)), 1e-9)
    annualized_profit = annualize_metric(float(row.get("net_profit", 0.0)), sim_days)
    annualized_revenue = annualize_metric(float(row.get("revenue", 0.0)), sim_days)
    annualized_cost = annualize_metric(float(row.get("total_cost", 0.0)), sim_days)
    annualized_energy = annualize_metric(float(row.get("energy_kwh", 0.0)), sim_days)
    annualized_harvest_fw = annualize_metric(float(row.get("harvest_fresh_kg", 0.0)), sim_days)
    annualized_harvest_dw = annualize_metric(float(row.get("harvest_dry_kg", 0.0)), sim_days)
    stage_objective = (
        annualized_profit
        if bool(row.get("valid_full_horizon", False))
        else (-1.0e12 + annualized_profit)
    )
    row.update(
        {
            "stage_name": str(task["stage_name"]),
            "stage_order": int(task["stage_order"]),
            "stage_duration_days": duration_days,
            "stage_objective": float(stage_objective),
            "annualized_profit": float(annualized_profit),
            "annualized_revenue": float(annualized_revenue),
            "annualized_cost": float(annualized_cost),
            "annualized_energy_kwh": float(annualized_energy),
            "annualized_harvest_fresh_kg": float(annualized_harvest_fw),
            "annualized_harvest_dry_kg": float(annualized_harvest_dw),
        }
    )
    return row


def run_stage(
    *,
    stage_name: str,
    stage_order: int,
    schedules: List[Dict[str, Any]],
    duration_days: float,
    dt_seconds: float,
    start_date: str,
    weather_path: str,
    seed: int,
    num_workers: int,
    out_csv: Path,
    photo_period_manual: int | None,
    i1_manual: float | None,
    i2_manual: float | None,
) -> pd.DataFrame:
    tasks = []
    for item in schedules:
        tasks.append(
            {
                "schedule": item["schedule"],
                "eval_index": int(item["eval_index"]),
                "stage_name": stage_name,
                "stage_order": int(stage_order),
                "duration_days": float(duration_days),
                "dt_seconds": float(dt_seconds),
                "start_date": str(start_date),
                "weather_path": str(weather_path),
                "seed": int(seed),
                "photo_period_manual": photo_period_manual,
                "I1_manual": i1_manual,
                "I2_manual": i2_manual,
            }
        )

    if not tasks:
        empty = pd.DataFrame()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        empty.to_csv(out_csv, index=False, encoding="utf-8")
        return empty

    print(
        f"[{stage_name}] evaluating {len(tasks)} schedules | "
        f"duration={duration_days}d | workers={num_workers}"
    )
    t0 = time.time()
    results: List[Dict[str, Any]] = []
    if num_workers <= 1:
        iterator = (_evaluate_stage_task(task) for task in tasks)
    else:
        chunksize = max(1, len(tasks) // max(num_workers * 4, 1))
        executor = ProcessPoolExecutor(max_workers=num_workers)
        iterator = executor.map(_evaluate_stage_task, tasks, chunksize=chunksize)

    try:
        for idx, row in enumerate(iterator, start=1):
            results.append(row)
            if idx == 1 or idx == len(tasks) or idx % max(1, len(tasks) // 10) == 0:
                print(
                    f"[{stage_name}] {idx:4d}/{len(tasks):4d} "
                    f"{row['schedule_key']} | valid={row['valid_full_horizon']} "
                    f"| ann_profit={row['annualized_profit']:.2f} "
                    f"| ann_fw={row['annualized_harvest_fresh_kg']:.2f} kg"
                )
    finally:
        if num_workers > 1:
            executor.shutdown(wait=True, cancel_futures=False)

    df = stage_sort_dataframe(pd.DataFrame(results))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[{stage_name}] saved -> {out_csv} | elapsed={time.time() - t0:.1f}s")
    return df


def build_schedule_catalog(
    schedules: Iterable[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    catalog: List[Dict[str, Any]] = []
    lookup: Dict[str, Dict[str, Any]] = {}
    for idx, schedule in enumerate(schedules):
        key = schedule_key(schedule)
        item = {"schedule_key": key, "eval_index": idx, "schedule": dict(schedule)}
        catalog.append(item)
        lookup[key] = item
    return catalog, lookup


def ensure_keys(
    selected_keys: List[str],
    *,
    required_keys: Iterable[str],
    lookup: Dict[str, Dict[str, Any]],
) -> List[str]:
    enriched = list(selected_keys)
    for key in required_keys:
        if key in lookup and key not in enriched:
            enriched.append(key)
    return enriched


def select_candidates_from_stage(
    stage_df: pd.DataFrame,
    *,
    topk: int,
    lookup: Dict[str, Dict[str, Any]],
    required_keys: Iterable[str],
) -> List[Dict[str, Any]]:
    if stage_df.empty:
        return []
    ordered_keys = stage_df["schedule_key"].astype(str).tolist()
    selected_keys = ordered_keys[: min(topk, len(ordered_keys))]
    selected_keys = ensure_keys(selected_keys, required_keys=required_keys, lookup=lookup)
    return [lookup[key] for key in selected_keys if key in lookup]


def _gap(best: Dict[str, Any] | None, baseline: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if not best or not baseline:
        return None
    gap = {}
    for key in ("annualized_profit", "annualized_harvest_fresh_kg", "annualized_energy_kwh", "annualized_cost"):
        best_val = float(best.get(key, 0.0))
        base_val = float(baseline.get(key, 0.0))
        gap[f"{key}_absolute"] = best_val - base_val
        gap[f"{key}_relative"] = ((best_val - base_val) / abs(base_val)) if abs(base_val) > 1e-9 else None
    return gap


def _short_schedule_label(row: Dict[str, Any]) -> str:
    return (
        f"t1={int(row['t1'])}, t2={int(row['t2'])}, "
        f"N1={int(row['N1'])}, rho2={int(round(float(row['rho2'])))}"
    )


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.25, linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_alpha(0.3)


def save_summary_plot(
    stage1_df: pd.DataFrame,
    stage2_df: pd.DataFrame,
    stage3_df: pd.DataFrame,
    out_png: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(17, 12))
    ax1, ax2, ax3, ax4 = axes.flat

    if not stage1_df.empty:
        ax1.hist(stage1_df["annualized_profit"].astype(float).values, bins=40, color="#6baed6", alpha=0.9)
        ax1.set_title("Stage-1 proxy annualized profit distribution", fontsize=11, pad=8)
        ax1.set_xlabel("Annualized net profit")
        ax1.set_ylabel("Schedule count")
        _style_axes(ax1)
    else:
        ax1.text(0.5, 0.5, "No Stage-1 results", ha="center", va="center")
        ax1.set_axis_off()

    if not stage2_df.empty and not stage3_df.empty:
        join23 = stage2_df[["schedule_key", "annualized_profit"]].merge(
            stage3_df[["schedule_key", "annualized_profit", "is_default_schedule"]],
            on="schedule_key",
            suffixes=("_stage2", "_stage3"),
        )
        ax2.scatter(
            join23["annualized_profit_stage2"],
            join23["annualized_profit_stage3"],
            c=np.where(join23["is_default_schedule"], "#d62728", "#3182bd"),
            s=np.where(join23["is_default_schedule"], 120, 55),
            alpha=0.9,
            edgecolors="black",
            linewidths=0.6,
        )
        ax2.set_title("Stage-2 vs Stage-3 annualized profit", fontsize=11, pad=8)
        ax2.set_xlabel("Stage-2 annualized net profit")
        ax2.set_ylabel("Stage-3 annualized net profit")
        _style_axes(ax2)
    else:
        ax2.text(0.5, 0.5, "Need Stage-2 and Stage-3 results", ha="center", va="center")
        ax2.set_axis_off()

    if not stage3_df.empty:
        top10 = stage3_df.head(min(10, len(stage3_df))).iloc[::-1]
        bars = ax3.barh(
            range(len(top10)),
            top10["annualized_profit"].astype(float).values,
            color=plt.cm.Blues(np.linspace(0.35, 0.9, len(top10))),
            alpha=0.95,
        )
        ax3.set_yticks(range(len(top10)))
        ax3.set_yticklabels([_short_schedule_label(r) for r in top10.to_dict(orient="records")], fontsize=9)
        ax3.set_title("Finalist set: top annualized profit candidates", fontsize=11, pad=8)
        ax3.set_xlabel("Annualized net profit")
        _style_axes(ax3)
        for bar, row in zip(bars, top10.to_dict(orient="records")):
            ax3.text(
                bar.get_width() + max(top10["annualized_profit"].astype(float).max() * 0.01, 1.0),
                bar.get_y() + bar.get_height() / 2.0,
                f"FW={float(row['annualized_harvest_fresh_kg']):.1f} kg/y",
                va="center",
                fontsize=9,
            )
    else:
        ax3.text(0.5, 0.5, "No Stage-3 results", ha="center", va="center")
        ax3.set_axis_off()

    default_df = stage3_df[stage3_df["is_default_schedule"] == True] if not stage3_df.empty else pd.DataFrame()
    best_row = stage3_df.iloc[0].to_dict() if not stage3_df.empty else None
    default_row = default_df.iloc[0].to_dict() if not default_df.empty else None
    if best_row and default_row:
        metrics = [
            ("Profit", "annualized_profit"),
            ("Harvest FW", "annualized_harvest_fresh_kg"),
            ("Energy", "annualized_energy_kwh"),
            ("Cost", "annualized_cost"),
        ]
        deltas_pct = []
        annotations = []
        for _, key in metrics:
            best_val = float(best_row.get(key, 0.0))
            base_val = float(default_row.get(key, 0.0))
            deltas_pct.append(((best_val - base_val) / abs(base_val) * 100.0) if abs(base_val) > 1e-9 else 0.0)
            annotations.append(f"{best_val - base_val:+.1f}")
        bars = ax4.bar(
            range(len(metrics)),
            deltas_pct,
            color=["#2b8cbe", "#41ab5d", "#e6550d", "#756bb1"],
            alpha=0.9,
        )
        ax4.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
        ax4.set_xticks(range(len(metrics)))
        ax4.set_xticklabels([m[0] for m in metrics], rotation=20, ha="right")
        ax4.set_ylabel("Best-final vs default [%]")
        ax4.set_title("Best finalist vs default schedule", fontsize=11, pad=8)
        _style_axes(ax4)
        for bar, txt, pct in zip(bars, annotations, deltas_pct):
            y = pct + (1.5 if pct >= 0 else -1.5)
            va = "bottom" if pct >= 0 else "top"
            ax4.text(bar.get_x() + bar.get_width() / 2.0, y, txt, ha="center", va=va, fontsize=9)
    else:
        ax4.text(0.5, 0.5, "Default schedule not present in final stage", ha="center", va="center")
        ax4.set_axis_off()

    fig.suptitle("Approximate PID schedule search summary", fontsize=14, y=0.98)
    fig.subplots_adjust(top=0.92, wspace=0.28, hspace=0.30)
    fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-fidelity approximate PID schedule search."
    )
    parser.add_argument(
        "--feasible-csv",
        default=str(FEASIBLE_CSV_DEFAULT),
        help="Path to the feasible schedule CSV.",
    )
    parser.add_argument(
        "--weather-path",
        default=str(ROOT / "data" / "weather" / "weather_hangzhou_2024.csv"),
        help="Path to the Hangzhou weather CSV.",
    )
    parser.add_argument("--start-date", default="2024-01-01", help="Simulation start date YYYY-MM-DD.")
    parser.add_argument("--dt", type=float, default=600.0, help="Simulation step [s].")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--stage1-days", type=float, default=56.0, help="Short-horizon proxy duration [days].")
    parser.add_argument("--stage2-days", type=float, default=140.0, help="Medium-horizon refinement duration [days].")
    parser.add_argument("--final-days", type=float, default=365.0, help="Final exact duration [days].")
    parser.add_argument("--topk-stage2", type=int, default=160, help="Promote top-K schedules from stage 1 to stage 2.")
    parser.add_argument("--topk-stage3", type=int, default=24, help="Promote top-K schedules from stage 2 to final exact stage.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, min(4, (os.cpu_count() or 1) - 1)),
        help="Parallel worker count used inside each stage.",
    )
    parser.add_argument(
        "--max-schedules",
        type=int,
        default=None,
        help="Optional debug limit: only evaluate the first N feasible schedules.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "results" / "approx_pid_search"),
        help="Directory for approximate-search outputs.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Delete old approximate-search outputs in --out-dir first.")
    parser.add_argument("--I1_manual", type=float, default=None, help="Optional manual dense-zone PPFD override.")
    parser.add_argument("--I2_manual", type=float, default=None, help="Optional manual finishing-zone PPFD override.")
    parser.add_argument("--photo-period-manual", type=int, default=None, help="Optional manual photoperiod override [h/day].")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.topk_stage2 < 1 or args.topk_stage3 < 1:
        raise RuntimeError("top-k values must be at least 1.")
    if args.num_workers < 1:
        raise RuntimeError("--num-workers must be at least 1.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stage1_csv = out_dir / "approx_stage1_proxy.csv"
    stage2_csv = out_dir / "approx_stage2_refine.csv"
    stage3_csv = out_dir / "approx_stage3_exact.csv"
    summary_json = out_dir / "approx_pid_search_summary.json"
    summary_plot = out_dir / SUMMARY_PLOT_NAME

    if args.overwrite:
        for path in (stage1_csv, stage2_csv, stage3_csv, summary_json, summary_plot):
            if path.exists():
                path.unlink()

    schedules = load_feasible_schedules(Path(args.feasible_csv))
    if args.max_schedules is not None:
        schedules = schedules[: max(0, int(args.max_schedules))]
    if not schedules:
        raise RuntimeError("No feasible schedules available for approximate search.")

    catalog, lookup = build_schedule_catalog(schedules)
    default_keys = [item["schedule_key"] for item in catalog if is_default_schedule(item["schedule"])]
    default_key = default_keys[0] if default_keys else None

    print(
        f"[APPROX] loaded {len(catalog)} feasible schedules | "
        f"stage1={args.stage1_days}d | stage2={args.stage2_days}d | final={args.final_days}d"
    )

    stage1_df = run_stage(
        stage_name=STAGE_NAMES[0],
        stage_order=1,
        schedules=catalog,
        duration_days=args.stage1_days,
        dt_seconds=args.dt,
        start_date=args.start_date,
        weather_path=args.weather_path,
        seed=args.seed,
        num_workers=args.num_workers,
        out_csv=stage1_csv,
        photo_period_manual=args.photo_period_manual,
        i1_manual=args.I1_manual,
        i2_manual=args.I2_manual,
    )
    stage2_candidates = select_candidates_from_stage(
        stage1_df,
        topk=args.topk_stage2,
        lookup=lookup,
        required_keys=[default_key] if default_key else [],
    )
    stage2_df = run_stage(
        stage_name=STAGE_NAMES[1],
        stage_order=2,
        schedules=stage2_candidates,
        duration_days=args.stage2_days,
        dt_seconds=args.dt,
        start_date=args.start_date,
        weather_path=args.weather_path,
        seed=args.seed,
        num_workers=args.num_workers,
        out_csv=stage2_csv,
        photo_period_manual=args.photo_period_manual,
        i1_manual=args.I1_manual,
        i2_manual=args.I2_manual,
    )
    stage3_candidates = select_candidates_from_stage(
        stage2_df,
        topk=args.topk_stage3,
        lookup=lookup,
        required_keys=[default_key] if default_key else [],
    )
    stage3_df = run_stage(
        stage_name=STAGE_NAMES[2],
        stage_order=3,
        schedules=stage3_candidates,
        duration_days=args.final_days,
        dt_seconds=args.dt,
        start_date=args.start_date,
        weather_path=args.weather_path,
        seed=args.seed,
        num_workers=args.num_workers,
        out_csv=stage3_csv,
        photo_period_manual=args.photo_period_manual,
        i1_manual=args.I1_manual,
        i2_manual=args.I2_manual,
    )

    stage1_best = stage1_df.iloc[0].to_dict() if not stage1_df.empty else None
    stage2_best = stage2_df.iloc[0].to_dict() if not stage2_df.empty else None
    stage3_best = stage3_df.iloc[0].to_dict() if not stage3_df.empty else None
    stage3_default = None
    if not stage3_df.empty and default_key is not None:
        default_subset = stage3_df[stage3_df["schedule_key"] == default_key]
        if not default_subset.empty:
            stage3_default = default_subset.iloc[0].to_dict()

    save_summary_plot(stage1_df, stage2_df, stage3_df, summary_plot)

    summary = {
        "method": "multi_fidelity_successive_halving",
        "note": (
            "Final recommendation is approximate because only the promoted finalist set "
            "is exact-evaluated over the full horizon."
        ),
        "config": {
            "stage1_days": float(args.stage1_days),
            "stage2_days": float(args.stage2_days),
            "final_days": float(args.final_days),
            "topk_stage2": int(args.topk_stage2),
            "topk_stage3": int(args.topk_stage3),
            "num_workers": int(args.num_workers),
            "dt_seconds": float(args.dt),
            "start_date": str(args.start_date),
            "seed": int(args.seed),
        },
        "n_feasible_loaded": int(len(catalog)),
        "n_stage1": int(len(stage1_df)),
        "n_stage2": int(len(stage2_df)),
        "n_stage3": int(len(stage3_df)),
        "best_stage1": stage1_best,
        "best_stage2": stage2_best,
        "best_final_approx": stage3_best,
        "default_in_final_stage": stage3_default,
        "gap_best_final_vs_default": _gap(stage3_best, stage3_default),
        "artifacts": {
            "stage1_csv": str(stage1_csv),
            "stage2_csv": str(stage2_csv),
            "stage3_csv": str(stage3_csv),
            "summary_plot": str(summary_plot),
        },
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 72)
    print("Approximate PID Schedule Search Summary")
    print("=" * 72)
    print(f"Stage-1 CSV        : {stage1_csv}")
    print(f"Stage-2 CSV        : {stage2_csv}")
    print(f"Stage-3 CSV        : {stage3_csv}")
    print(f"Summary JSON       : {summary_json}")
    print(f"Summary Plot       : {summary_plot}")
    print(f"Loaded schedules   : {len(catalog)}")
    print(f"Stage sizes        : {len(stage1_df)} -> {len(stage2_df)} -> {len(stage3_df)}")
    if stage3_best:
        print(
            "Best final approx  : "
            f"{stage3_best['schedule_key']} | ann_profit={stage3_best['annualized_profit']:.2f} "
            f"| ann_fw={stage3_best['annualized_harvest_fresh_kg']:.2f} kg "
            f"| energy={stage3_best['annualized_energy_kwh']:.2f} kWh"
        )
    if stage3_default:
        print(
            "Default in finalists: "
            f"{stage3_default['schedule_key']} | ann_profit={stage3_default['annualized_profit']:.2f} "
            f"| ann_fw={stage3_default['annualized_harvest_fresh_kg']:.2f} kg"
        )
    print("=" * 72)


if __name__ == "__main__":
    main()
