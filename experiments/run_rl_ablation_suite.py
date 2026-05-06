# -*- coding: utf-8 -*-
"""
Run a small RL ablation suite over controller design and curriculum profile.

Typical usage:

  python experiments/run_rl_ablation_suite.py --suite_profile screening_core --runtime_profile pilot_fast --device cpu

  python experiments/run_rl_ablation_suite.py --suite_profile screening_core --runtime_profile pilot_fast --device cuda --skip_existing

  python experiments/run_rl_ablation_suite.py --suite_name rl_screen_v1 --summary_only
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "experiments" / "train_pfal_contextual.py"
LOG_ROOT = ROOT / "log" / "PFAL-contextual-SAC" / "sac_contextual"
RESULTS_ROOT = ROOT / "results"

VARIANTS = [
    {
        "name": "direct_contextual_sac",
        "controller_design": "contextual_sac",
        "curriculum_profile": "off",
    },
    {
        "name": "curriculum_contextual_sac",
        "controller_design": "contextual_sac",
        "curriculum_profile": "target_to_full",
    },
    {
        "name": "direct_residual_pid_sac",
        "controller_design": "residual_pid_sac",
        "curriculum_profile": "off",
    },
    {
        "name": "direct_gated_residual_pid_sac",
        "controller_design": "gated_residual_pid_sac",
        "curriculum_profile": "off",
    },
    {
        "name": "direct_climate_only_residual_pid_sac",
        "controller_design": "climate_only_residual_pid_sac",
        "curriculum_profile": "off",
    },
    {
        "name": "curriculum_residual_pid_sac",
        "controller_design": "residual_pid_sac",
        "curriculum_profile": "target_to_full",
    },
    {
        "name": "curriculum_gated_residual_pid_sac",
        "controller_design": "gated_residual_pid_sac",
        "curriculum_profile": "target_to_full",
    },
    {
        "name": "curriculum_climate_only_residual_pid_sac",
        "controller_design": "climate_only_residual_pid_sac",
        "curriculum_profile": "target_to_full",
    },
]

SUITE_PROFILES = {
    "screening_core": [
        "direct_contextual_sac",
        "curriculum_contextual_sac",
        "direct_residual_pid_sac",
        "curriculum_residual_pid_sac",
    ],
    "residual_family": [
        "direct_residual_pid_sac",
        "curriculum_residual_pid_sac",
        "direct_gated_residual_pid_sac",
        "curriculum_gated_residual_pid_sac",
        "direct_climate_only_residual_pid_sac",
        "curriculum_climate_only_residual_pid_sac",
    ],
    "control_quality_core": [
        "direct_residual_pid_sac",
        "direct_gated_residual_pid_sac",
        "direct_climate_only_residual_pid_sac",
    ],
    "all": [entry["name"] for entry in VARIANTS],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run RL ablations for controller design, including residual, gated residual, "
            "and climate-only residual SAC variants."
        )
    )
    parser.add_argument("--suite_name", type=str, default=None, help="Suite name prefix.")
    parser.add_argument(
        "--suite_profile",
        type=str,
        default="screening_core",
        choices=sorted(SUITE_PROFILES.keys()),
        help="Named subset of variants to run before any explicit --variants override.",
    )
    parser.add_argument(
        "--variants",
        type=str,
        nargs="*",
        default=None,
        choices=[entry["name"] for entry in VARIANTS],
        help="Optional subset of variants to run.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="Random seeds.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--runtime_profile",
        type=str,
        default="pilot_fast",
        choices=["default", "pilot_fast", "pilot_ultrafast"],
        help="Training runtime preset.",
    )
    parser.add_argument(
        "--horizon_profile",
        type=str,
        default=None,
        choices=["config", "fast_t2max", "mainline_long_horizon"],
        help="Optional episode-horizon preset passed to the trainer.",
    )
    parser.add_argument("--epoch", type=int, default=None, help="Optional epoch override.")
    parser.add_argument("--batch_size", type=int, default=None, help="Optional batch size override.")
    parser.add_argument("--train_num", type=int, default=None, help="Optional train env override.")
    parser.add_argument("--test_num", type=int, default=None, help="Optional test env override.")
    parser.add_argument("--train_episode_days", type=float, default=None)
    parser.add_argument("--test_episode_days", type=float, default=None)
    parser.add_argument(
        "--eval_episode_length_mode",
        type=str,
        default=None,
        choices=[
            "schedule_t2",
            "max_t2",
            "fixed_days",
            "total_cycle",
            "max_total_cycle",
            "mixed",
            "mixed_horizon",
            "mixed_episode",
            "curriculum",
        ],
    )
    parser.add_argument("--eval_episode_days", type=float, default=None)
    parser.add_argument(
        "--step_per_collect_env_steps",
        type=int,
        default=None,
        help="Optional rollout-step override.",
    )
    parser.add_argument(
        "--update_per_step",
        type=float,
        default=None,
        help="Optional gradient-update intensity override.",
    )
    parser.add_argument("--nstep_factor", type=float, default=None, help="Auto nstep factor.")
    parser.add_argument(
        "--constraint_selection_interval_epochs",
        type=int,
        default=None,
        help="Constraint-selection interval override; 0 disables.",
    )
    parser.add_argument("--final_eval_schedules", type=int, default=6)
    parser.add_argument("--final_eval_episodes_per_schedule", type=int, default=1)
    parser.add_argument(
        "--summary_only",
        action="store_true",
        help="Do not launch training; only summarize finished runs for the selected suite.",
    )
    parser.add_argument("--skip_existing", action="store_true", help="Skip finished runs.")
    parser.add_argument("--continue_on_error", action="store_true", help="Continue after failures.")
    return parser.parse_args()


def resolve_variants(
    selected_names: list[str] | None,
    suite_profile: str | None,
) -> list[dict]:
    if selected_names:
        selected_set = set(selected_names)
    else:
        profile_name = str(suite_profile or "screening_core").strip().lower()
        selected_set = set(SUITE_PROFILES.get(profile_name, SUITE_PROFILES["screening_core"]))
    selected = []
    for entry in VARIANTS:
        if entry["name"] in selected_set:
            selected.append(dict(entry))
    return selected


def run_command(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        if isinstance(value, str):
            match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value)
            if match:
                try:
                    return float(match.group(0))
                except ValueError:
                    pass
        return float(default)


def extract_run_record(experiment_name: str, seed: int, variant_name: str | None = None) -> dict:
    log_dir = LOG_ROOT / experiment_name
    training_summary = load_json(log_dir / "training_summary.json") or {}
    selected_eval = load_json(log_dir / "generalization_eval_selected.json")
    final_eval = load_json(log_dir / "generalization_eval.json")
    eval_payload = selected_eval or final_eval or {}
    eval_source = "selected" if selected_eval is not None else "final"
    trainer_result = training_summary.get("trainer_result", {}) or {}

    record = {
        "experiment": experiment_name,
        "variant_name": str(variant_name or ""),
        "seed": int(seed),
        "controller_design": str(training_summary.get("controller_design", "")),
        "curriculum_profile": str(training_summary.get("curriculum_profile", "")),
        "runtime_profile": str(training_summary.get("runtime_profile", "")),
        "horizon_profile": str(training_summary.get("horizon_profile", "")),
        "action_semantics": str(training_summary.get("action_semantics", "")),
        "nstep": int(training_summary.get("nstep", 0) or 0),
        "resolved_train_episode_steps": int(
            training_summary.get("resolved_train_episode_steps", 0) or 0
        ),
        "resolved_step_per_epoch_total": int(
            training_summary.get("resolved_step_per_epoch_total", 0) or 0
        ),
        "resolved_step_per_collect_total": int(
            training_summary.get("resolved_step_per_collect_total", 0) or 0
        ),
        "resolved_update_per_step": _safe_float(
            training_summary.get("resolved_update_per_step", 0.0), 0.0
        ),
        "resolved_updates_per_epoch": _safe_float(
            training_summary.get("resolved_updates_per_epoch", 0.0), 0.0
        ),
        "batch_size": int(training_summary.get("batch_size", 0) or 0),
        "train_num": int(training_summary.get("train_num", 0) or 0),
        "test_num": int(training_summary.get("test_num", 0) or 0),
        "elapsed_seconds": _safe_float(training_summary.get("elapsed_seconds", 0.0), 0.0),
        "trainer_duration_seconds": _safe_float(trainer_result.get("duration", 0.0), 0.0),
        "trainer_model_time_seconds": _safe_float(
            trainer_result.get("train_time/model", 0.0), 0.0
        ),
        "trainer_collector_time_seconds": _safe_float(
            trainer_result.get("train_time/collector", 0.0), 0.0
        ),
        "trainer_test_time_seconds": _safe_float(
            trainer_result.get("test_time", 0.0), 0.0
        ),
        "eval_source": eval_source,
        "mean_reward": float(eval_payload.get("mean_reward", 0.0) or 0.0),
        "std_reward": float(eval_payload.get("std_reward", 0.0) or 0.0),
        "mean_reward_per_day": float(eval_payload.get("mean_reward_per_day", 0.0) or 0.0),
        "mean_constraint_cost": float(eval_payload.get("mean_constraint_cost", 0.0) or 0.0),
        "mean_constraint_cost_per_day": float(
            eval_payload.get("mean_constraint_cost_per_day", 0.0) or 0.0
        ),
        "mean_constraint_active_ratio": float(
            eval_payload.get("mean_constraint_active_ratio", 0.0) or 0.0
        ),
        "mean_completion_ratio": float(eval_payload.get("mean_completion_ratio", 0.0) or 0.0),
        "early_termination_ratio": float(eval_payload.get("early_termination_ratio", 0.0) or 0.0),
        "harvest_fail_episode_ratio": float(
            eval_payload.get("harvest_fail_episode_ratio", 0.0) or 0.0
        ),
        "safety_override_episode_ratio": float(
            eval_payload.get("safety_override_episode_ratio", 0.0) or 0.0
        ),
        "mean_safety_overrides_per_episode": float(
            eval_payload.get("mean_safety_overrides_per_episode", 0.0) or 0.0
        ),
        "mean_harvest_g_per_day": float(eval_payload.get("mean_harvest_g_per_day", 0.0) or 0.0),
        "mean_cost_per_day": float(eval_payload.get("mean_cost_per_day", 0.0) or 0.0),
    }
    record["variant_label"] = (
        f"{record['controller_design']}|{record['curriculum_profile']}|seed{seed}"
    )
    return record


def save_suite_manifest(
    *,
    args: argparse.Namespace,
    suite_name: str,
    variants: list[dict],
    out_dir: Path,
) -> None:
    manifest = {
        "suite_name": suite_name,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "suite_profile": args.suite_profile,
        "variants": [dict(item) for item in variants],
        "seeds": list(args.seeds),
        "device": args.device,
        "runtime_profile": args.runtime_profile,
        "horizon_profile": args.horizon_profile,
        "epoch": args.epoch,
        "batch_size": args.batch_size,
        "train_num": args.train_num,
        "test_num": args.test_num,
        "summary_only": bool(args.summary_only),
        "skip_existing": bool(args.skip_existing),
    }
    with open(out_dir / "suite_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def aggregate_records(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in records:
        key = (
            str(row.get("variant_name", "")),
            str(row.get("controller_design", "")),
            str(row.get("curriculum_profile", "")),
            str(row.get("horizon_profile", "")),
        )
        grouped[key].append(row)

    metric_names = [
        "mean_reward",
        "mean_reward_per_day",
        "mean_constraint_cost",
        "mean_constraint_cost_per_day",
        "mean_constraint_active_ratio",
        "mean_completion_ratio",
        "early_termination_ratio",
        "harvest_fail_episode_ratio",
        "safety_override_episode_ratio",
        "mean_safety_overrides_per_episode",
        "mean_harvest_g_per_day",
        "mean_cost_per_day",
        "elapsed_seconds",
        "trainer_duration_seconds",
        "trainer_model_time_seconds",
        "trainer_collector_time_seconds",
        "trainer_test_time_seconds",
    ]

    aggregated = []
    for key, rows in grouped.items():
        variant_name, controller_design, curriculum_profile, horizon_profile = key
        agg = {
            "variant_name": variant_name,
            "controller_design": controller_design,
            "curriculum_profile": curriculum_profile,
            "horizon_profile": horizon_profile,
            "seed_count": len(rows),
            "seeds": ",".join(str(int(row.get("seed", 0))) for row in rows),
            "runtime_profile": str(rows[0].get("runtime_profile", "")),
            "action_semantics": str(rows[0].get("action_semantics", "")),
            "nstep_mean": float(np.mean([float(row.get("nstep", 0)) for row in rows])),
            "resolved_train_episode_steps_mean": float(
                np.mean([float(row.get("resolved_train_episode_steps", 0)) for row in rows])
            ),
        }
        for metric in metric_names:
            values = np.array([float(row.get(metric, 0.0) or 0.0) for row in rows], dtype=float)
            agg[f"{metric}_mean"] = float(values.mean()) if len(values) else 0.0
            agg[f"{metric}_std"] = float(values.std(ddof=0)) if len(values) else 0.0
        aggregated.append(agg)

    aggregated.sort(
        key=lambda row: (
            -float(row.get("mean_reward_per_day_mean", 0.0)),
            float(row.get("early_termination_ratio_mean", 0.0)),
            float(row.get("harvest_fail_episode_ratio_mean", 0.0)),
            float(row.get("mean_constraint_cost_per_day_mean", 0.0)),
        )
    )
    return aggregated


def save_aggregate_summary(records: list[dict], out_dir: Path) -> None:
    if not records:
        return
    fieldnames = list(records[0].keys())
    csv_path = out_dir / "suite_summary_aggregated.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    with open(out_dir / "suite_summary_aggregated.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] {csv_path}")


def save_leaderboard(records: list[dict], out_dir: Path) -> None:
    if not records:
        return
    leaderboard = []
    for rank, row in enumerate(records, start=1):
        leaderboard.append(
            {
                "rank": rank,
                "variant_name": row["variant_name"],
                "controller_design": row["controller_design"],
                "curriculum_profile": row["curriculum_profile"],
                "seed_count": row["seed_count"],
                "mean_reward_per_day_mean": row["mean_reward_per_day_mean"],
                "mean_reward_per_day_std": row["mean_reward_per_day_std"],
                "mean_constraint_cost_per_day_mean": row["mean_constraint_cost_per_day_mean"],
                "early_termination_ratio_mean": row["early_termination_ratio_mean"],
                "harvest_fail_episode_ratio_mean": row["harvest_fail_episode_ratio_mean"],
                "mean_harvest_g_per_day_mean": row["mean_harvest_g_per_day_mean"],
                "mean_cost_per_day_mean": row["mean_cost_per_day_mean"],
                "elapsed_seconds_mean": row["elapsed_seconds_mean"],
            }
        )
    csv_path = out_dir / "suite_leaderboard.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(leaderboard[0].keys()))
        writer.writeheader()
        writer.writerows(leaderboard)
    with open(out_dir / "suite_leaderboard.json", "w", encoding="utf-8") as f:
        json.dump(leaderboard, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] {csv_path}")


def select_screening_winner(aggregate_rows: list[dict]) -> dict | None:
    if not aggregate_rows:
        return None
    safe_pool = [
        row
        for row in aggregate_rows
        if float(row.get("early_termination_ratio_mean", 1.0)) <= 0.05
        and float(row.get("harvest_fail_episode_ratio_mean", 1.0)) <= 0.05
    ]
    pool = safe_pool or aggregate_rows
    return sorted(
        pool,
        key=lambda row: (
            -float(row.get("mean_reward_per_day_mean", 0.0)),
            float(row.get("mean_constraint_cost_per_day_mean", 0.0)),
            float(row.get("mean_cost_per_day_mean", 0.0)),
            -float(row.get("mean_completion_ratio_mean", 0.0)),
        ),
    )[0]


def save_screening_winner(aggregate_rows: list[dict], out_dir: Path) -> None:
    winner = select_screening_winner(aggregate_rows)
    if winner is None:
        return
    payload = {
        "winner_variant": winner["variant_name"],
        "controller_design": winner["controller_design"],
        "curriculum_profile": winner["curriculum_profile"],
        "horizon_profile": winner["horizon_profile"],
        "mean_reward_per_day_mean": winner["mean_reward_per_day_mean"],
        "mean_constraint_cost_per_day_mean": winner["mean_constraint_cost_per_day_mean"],
        "early_termination_ratio_mean": winner["early_termination_ratio_mean"],
        "harvest_fail_episode_ratio_mean": winner["harvest_fail_episode_ratio_mean"],
        "mean_harvest_g_per_day_mean": winner["mean_harvest_g_per_day_mean"],
        "mean_cost_per_day_mean": winner["mean_cost_per_day_mean"],
    }
    with open(out_dir / "screening_recommendation.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(
        "[RECOMMEND]",
        payload["winner_variant"],
        f"reward/day={payload['mean_reward_per_day_mean']:.3f}",
        f"constraint/day={payload['mean_constraint_cost_per_day_mean']:.3f}",
    )


def save_summary(records: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not records:
        return
    fieldnames = list(records[0].keys())
    csv_path = out_dir / "suite_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    with open(out_dir / "suite_summary.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] {csv_path}")


def plot_reward_constraint(records: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    x = [row["mean_constraint_cost"] for row in records]
    y = [row["mean_reward"] for row in records]
    ax.scatter(x, y, s=90, c="#2b6cb0", alpha=0.9)
    for row in records:
        ax.annotate(
            row["variant_label"],
            (row["mean_constraint_cost"], row["mean_reward"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("Mean Constraint Cost")
    ax.set_ylabel("Mean Reward")
    ax.set_title("RL Ablation: Reward vs Constraint Cost")
    ax.grid(alpha=0.25, linestyle="--")
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_risk_profile(records: list[dict], out_path: Path) -> None:
    labels = [row["variant_label"] for row in records]
    x = np.arange(len(labels))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10.6, 6.0))
    ax.bar(
        x - width,
        [row["early_termination_ratio"] for row in records],
        width=width,
        label="Early termination",
        color="#c53030",
    )
    ax.bar(
        x,
        [row["harvest_fail_episode_ratio"] for row in records],
        width=width,
        label="Harvest fail",
        color="#dd6b20",
    )
    ax.bar(
        x + width,
        [row["safety_override_episode_ratio"] for row in records],
        width=width,
        label="Safety override",
        color="#2f855a",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("Episode ratio")
    ax.set_title("RL Ablation: Risk Profile")
    ax.set_ylim(0.0, max(1.0, ax.get_ylim()[1]))
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_productivity(records: list[dict], out_path: Path) -> None:
    labels = [row["variant_label"] for row in records]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(3, 1, figsize=(10.6, 10.2), sharex=True)
    axes[0].bar(x, [row["mean_reward_per_day"] for row in records], color="#2b6cb0")
    axes[0].set_ylabel("Reward/day")
    axes[0].set_title("RL Ablation: Productivity and Cost")
    axes[0].grid(axis="y", alpha=0.25, linestyle="--")

    axes[1].bar(x, [row["mean_harvest_g_per_day"] for row in records], color="#2f855a")
    axes[1].set_ylabel("Harvest g/day")
    axes[1].grid(axis="y", alpha=0.25, linestyle="--")

    axes[2].bar(x, [row["mean_cost_per_day"] for row in records], color="#805ad5")
    axes[2].set_ylabel("Cost/day")
    axes[2].grid(axis="y", alpha=0.25, linestyle="--")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=18, ha="right")

    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_training_efficiency(records: list[dict], out_path: Path) -> None:
    labels = [row["variant_label"] for row in records]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 1, figsize=(10.8, 8.4), sharex=True)
    axes[0].bar(x, [row["elapsed_seconds"] / 60.0 for row in records], color="#1a202c")
    axes[0].set_ylabel("Elapsed min")
    axes[0].set_title("RL Ablation: Training Efficiency")
    axes[0].grid(axis="y", alpha=0.25, linestyle="--")

    model_times = np.array([row["trainer_model_time_seconds"] for row in records], dtype=float)
    collector_times = np.array([row["trainer_collector_time_seconds"] for row in records], dtype=float)
    test_times = np.array([row["trainer_test_time_seconds"] for row in records], dtype=float)
    axes[1].bar(x, model_times / 60.0, label="Model", color="#2b6cb0")
    axes[1].bar(x, collector_times / 60.0, bottom=model_times / 60.0, label="Collector", color="#2f855a")
    axes[1].bar(
        x,
        test_times / 60.0,
        bottom=(model_times + collector_times) / 60.0,
        label="Test",
        color="#dd6b20",
    )
    axes[1].set_ylabel("Trainer min")
    axes[1].grid(axis="y", alpha=0.25, linestyle="--")
    axes[1].legend(frameon=False)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=18, ha="right")

    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_aggregate_productivity(records: list[dict], out_path: Path) -> None:
    labels = [row["variant_name"] for row in records]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(3, 1, figsize=(10.8, 10.4), sharex=True)

    axes[0].bar(
        x,
        [row["mean_reward_per_day_mean"] for row in records],
        yerr=[row["mean_reward_per_day_std"] for row in records],
        color="#2b6cb0",
        alpha=0.92,
        capsize=4,
    )
    axes[0].set_ylabel("Reward/day")
    axes[0].set_title("RL Ablation Aggregate: Productivity, Cost, Risk")
    axes[0].grid(axis="y", alpha=0.25, linestyle="--")

    axes[1].bar(
        x,
        [row["mean_harvest_g_per_day_mean"] for row in records],
        yerr=[row["mean_harvest_g_per_day_std"] for row in records],
        color="#2f855a",
        alpha=0.92,
        capsize=4,
    )
    axes[1].set_ylabel("Harvest g/day")
    axes[1].grid(axis="y", alpha=0.25, linestyle="--")

    axes[2].bar(
        x,
        [row["mean_cost_per_day_mean"] for row in records],
        yerr=[row["mean_cost_per_day_std"] for row in records],
        color="#805ad5",
        alpha=0.92,
        capsize=4,
    )
    axes[2].set_ylabel("Cost/day")
    axes[2].grid(axis="y", alpha=0.25, linestyle="--")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=18, ha="right")

    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_aggregate_risk(records: list[dict], out_path: Path) -> None:
    labels = [row["variant_name"] for row in records]
    x = np.arange(len(labels))
    width = 0.24

    fig, ax = plt.subplots(figsize=(11.0, 6.2))
    ax.bar(
        x - width,
        [row["early_termination_ratio_mean"] for row in records],
        yerr=[row["early_termination_ratio_std"] for row in records],
        width=width,
        label="Early termination",
        color="#c53030",
        capsize=4,
    )
    ax.bar(
        x,
        [row["harvest_fail_episode_ratio_mean"] for row in records],
        yerr=[row["harvest_fail_episode_ratio_std"] for row in records],
        width=width,
        label="Harvest fail",
        color="#dd6b20",
        capsize=4,
    )
    ax.bar(
        x + width,
        [row["safety_override_episode_ratio_mean"] for row in records],
        yerr=[row["safety_override_episode_ratio_std"] for row in records],
        width=width,
        label="Safety override",
        color="#2f855a",
        capsize=4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("Episode ratio")
    ax.set_title("RL Ablation Aggregate: Risk Profile")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_aggregate_efficiency_tradeoff(records: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 6.4))
    x = [row["elapsed_seconds_mean"] / 60.0 for row in records]
    y = [row["mean_reward_per_day_mean"] for row in records]
    sizes = np.array([row["mean_harvest_g_per_day_mean"] for row in records], dtype=float)
    if sizes.size:
        sizes = 80.0 + 180.0 * (sizes - sizes.min()) / max(1e-6, sizes.ptp())
    else:
        sizes = np.array([120.0 for _ in records], dtype=float)
    colors = [row["mean_constraint_cost_per_day_mean"] for row in records]

    sc = ax.scatter(x, y, s=sizes, c=colors, cmap="viridis", alpha=0.92, edgecolor="black")
    for row, xi, yi in zip(records, x, y):
        ax.annotate(
            row["variant_name"],
            (xi, yi),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
        )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Constraint cost/day")
    ax.set_xlabel("Training elapsed time (min)")
    ax.set_ylabel("Reward/day")
    ax.set_title("RL Ablation Aggregate: Efficiency vs Performance")
    ax.grid(alpha=0.25, linestyle="--")
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    suite_name = args.suite_name or f"rl_ablation_suite_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    suite_out_dir = RESULTS_ROOT / suite_name
    suite_out_dir.mkdir(parents=True, exist_ok=True)

    if args.horizon_profile is not None and args.runtime_profile in {"pilot_fast", "pilot_ultrafast"}:
        print(
            "[WARN] Explicit --horizon_profile will override the short fixed-day episode budget "
            f"embedded in runtime_profile={args.runtime_profile}. "
            "For fastest screening, omit --horizon_profile and let the runtime profile control episode length."
        )

    variants = resolve_variants(args.variants, args.suite_profile)
    save_suite_manifest(args=args, suite_name=suite_name, variants=variants, out_dir=suite_out_dir)
    summary_records = []

    for seed in args.seeds:
        for variant in variants:
            experiment_name = f"{suite_name}_{variant['name']}_seed{seed}"
            log_dir = LOG_ROOT / experiment_name
            summary_path = log_dir / "training_summary.json"

            if args.summary_only:
                print(f"[SUMMARY_ONLY] {experiment_name}")
            elif args.skip_existing and summary_path.exists():
                print(f"[SKIP] {experiment_name} already finished.")
            else:
                cmd = [
                    sys.executable,
                    str(TRAIN_SCRIPT),
                    "--experiment",
                    experiment_name,
                    "--device",
                    args.device,
                    "--no_wandb",
                    "--controller_design",
                    variant["controller_design"],
                    "--curriculum_profile",
                    variant["curriculum_profile"],
                    "--runtime_profile",
                    args.runtime_profile,
                    "--seed",
                    str(seed),
                    "--auto_nstep",
                    "--final_eval_schedules",
                    str(args.final_eval_schedules),
                    "--final_eval_episodes_per_schedule",
                    str(args.final_eval_episodes_per_schedule),
                ]
                if args.epoch is not None:
                    cmd.extend(["--epoch", str(args.epoch)])
                if args.horizon_profile is not None:
                    cmd.extend(["--horizon_profile", str(args.horizon_profile)])
                if args.batch_size is not None:
                    cmd.extend(["--batch_size", str(args.batch_size)])
                if args.train_num is not None:
                    cmd.extend(["--train_num", str(args.train_num)])
                if args.test_num is not None:
                    cmd.extend(["--test_num", str(args.test_num)])
                if args.step_per_collect_env_steps is not None:
                    cmd.extend(
                        [
                            "--step_per_collect_env_steps",
                            str(args.step_per_collect_env_steps),
                        ]
                    )
                if args.update_per_step is not None:
                    cmd.extend(["--update_per_step", str(args.update_per_step)])
                if args.train_episode_days is not None:
                    cmd.extend(["--train_episode_length_mode", "fixed_days"])
                    cmd.extend(["--train_episode_days", str(args.train_episode_days)])
                if args.test_episode_days is not None:
                    cmd.extend(["--test_episode_length_mode", "fixed_days"])
                    cmd.extend(["--test_episode_days", str(args.test_episode_days)])
                if args.eval_episode_length_mode is not None:
                    cmd.extend(["--eval_episode_length_mode", str(args.eval_episode_length_mode)])
                if args.eval_episode_days is not None:
                    cmd.extend(["--eval_episode_days", str(args.eval_episode_days)])
                if args.nstep_factor is not None:
                    cmd.extend(["--nstep_factor", str(args.nstep_factor)])
                if args.constraint_selection_interval_epochs is not None:
                    cmd.extend(
                        [
                            "--constraint_selection_interval_epochs",
                            str(args.constraint_selection_interval_epochs),
                        ]
                    )
                try:
                    run_command(cmd)
                except subprocess.CalledProcessError as exc:
                    print(f"[FAIL] {experiment_name}: {exc}")
                    if not args.continue_on_error:
                        raise
                    continue

            if not summary_path.exists():
                message = f"[WARN] Missing training_summary.json for {experiment_name}, skipping summary."
                print(message)
                if not args.continue_on_error:
                    raise FileNotFoundError(message)
                continue

            try:
                summary_records.append(
                    extract_run_record(
                        experiment_name,
                        seed,
                        variant_name=variant["name"],
                    )
                )
            except Exception as exc:  # pragma: no cover - defensive aggregation
                print(f"[WARN] Failed to summarize {experiment_name}: {exc}")
                if not args.continue_on_error:
                    raise

    save_summary(summary_records, suite_out_dir)
    if summary_records:
        aggregate_rows = aggregate_records(summary_records)
        save_aggregate_summary(aggregate_rows, suite_out_dir)
        save_leaderboard(aggregate_rows, suite_out_dir)
        save_screening_winner(aggregate_rows, suite_out_dir)
        plot_reward_constraint(summary_records, suite_out_dir / "rl_ablation_reward_constraint.png")
        plot_risk_profile(summary_records, suite_out_dir / "rl_ablation_risk_profile.png")
        plot_productivity(summary_records, suite_out_dir / "rl_ablation_productivity.png")
        plot_training_efficiency(summary_records, suite_out_dir / "rl_ablation_training_efficiency.png")
        plot_aggregate_productivity(
            aggregate_rows,
            suite_out_dir / "rl_ablation_aggregate_productivity.png",
        )
        plot_aggregate_risk(
            aggregate_rows,
            suite_out_dir / "rl_ablation_aggregate_risk.png",
        )
        plot_aggregate_efficiency_tradeoff(
            aggregate_rows,
            suite_out_dir / "rl_ablation_aggregate_efficiency_tradeoff.png",
        )
        print(f"[SAVE] {suite_out_dir / 'rl_ablation_reward_constraint.png'}")
        print(f"[SAVE] {suite_out_dir / 'rl_ablation_risk_profile.png'}")
        print(f"[SAVE] {suite_out_dir / 'rl_ablation_productivity.png'}")
        print(f"[SAVE] {suite_out_dir / 'rl_ablation_training_efficiency.png'}")
        print(f"[SAVE] {suite_out_dir / 'rl_ablation_aggregate_productivity.png'}")
        print(f"[SAVE] {suite_out_dir / 'rl_ablation_aggregate_risk.png'}")
        print(f"[SAVE] {suite_out_dir / 'rl_ablation_aggregate_efficiency_tradeoff.png'}")


if __name__ == "__main__":
    main()
