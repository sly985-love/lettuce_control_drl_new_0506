# -*- coding: utf-8 -*-
"""
Batch-evaluate multiple checkpoints from one RL run and compare them side-by-side.

Typical usage:

  python experiments/compare_rl_checkpoints.py ^
    --load rl_t2max_residual_direct_20260419_v1_500 ^
    --device cpu ^
    --n_eval_schedules 20 ^
    --n_eval_episodes_per_schedule 1 ^
    --eval_selection reference_stratified
"""

from __future__ import annotations

import argparse
import importlib.machinery
import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import pandas as pd

# Disable wandb side effects for pure evaluation scripts.
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_SILENT", "true")
os.environ.setdefault("WANDB_DISABLE_CODE", "true")
os.environ.setdefault("WANDB_DISABLED", "true")

if "wandb" not in sys.modules:
    def _noop(*args, **kwargs):
        return None

    class _DummyRun:
        project = "disabled"
        name = "disabled"
        url = ""

        def log(self, *args, **kwargs):
            return None

        def finish(self, *args, **kwargs):
            return None

    wandb_stub = types.ModuleType("wandb")
    wandb_stub.init = lambda *args, **kwargs: _DummyRun()
    wandb_stub.log = _noop
    wandb_stub.finish = _noop
    wandb_stub.define_metric = _noop
    wandb_stub.Image = lambda *args, **kwargs: None
    wandb_stub.Artifact = lambda *args, **kwargs: None
    wandb_stub.Table = lambda *args, **kwargs: None
    wandb_stub.config = {}
    wandb_stub.run = None
    wandb_stub.__file__ = "<wandb_stub>"
    wandb_stub.__spec__ = importlib.machinery.ModuleSpec(
        name="wandb",
        loader=None,
    )
    sys.modules["wandb"] = wandb_stub


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.train_pfal_contextual import (  # noqa: E402
    apply_controller_design,
    apply_curriculum_profile,
    apply_horizon_profile,
    apply_runtime_profile,
    attach_inlet_seedling_metadata,
    evaluate_on_schedules,
    is_climate_only_residual_scale,
    load_rl_params,
    resolve_phase_episode_settings_from_params,
)
from rl.drl_based_control import (  # noqa: E402
    compute_constraint_aware_selection_score,
    create_policy,
    load_policy,
    load_saved_run_config,
    resolve_experiment_dir,
    sync_inlet_seedling_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate and compare multiple checkpoints from one RL run."
    )
    parser.add_argument(
        "--load",
        type=str,
        required=True,
        help="RL run directory or experiment name under log/PFAL-contextual-SAC/sac_contextual.",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="*",
        default=["best", "selected", "final", "auto"],
        help="Checkpoint labels to evaluate. Supported: best final selected auto",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Inference device, typically cpu or cuda.",
    )
    parser.add_argument(
        "--n_eval_schedules",
        type=int,
        default=20,
        help="How many schedules to evaluate for each checkpoint.",
    )
    parser.add_argument(
        "--n_eval_episodes_per_schedule",
        type=int,
        default=1,
        help="How many episodes per schedule.",
    )
    parser.add_argument(
        "--eval_seed",
        type=int,
        default=42,
        help="Shared evaluation seed for all checkpoints.",
    )
    parser.add_argument(
        "--eval_selection",
        type=str,
        default=None,
        choices=["coverage", "random", "reference_stratified"],
        help="Optional override for schedule selection strategy.",
    )
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
        help="Optional override for evaluation horizon mode.",
    )
    parser.add_argument(
        "--eval_episode_days",
        type=float,
        default=None,
        help="Optional override when eval episode mode is fixed_days.",
    )
    parser.add_argument(
        "--hour_of_day_mode",
        type=str,
        default=None,
        choices=["fixed", "random"],
        help="Optional override for evaluation reset hour mode.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory. Default: <run_dir>/checkpoint_eval_compare",
    )
    parser.add_argument(
        "--strict_missing",
        action="store_true",
        help="Fail immediately if any requested checkpoint is missing.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce per-schedule console output.",
    )
    return parser.parse_args()


def _selection_weights_from_params(params: dict[str, Any]) -> dict[str, Any]:
    return {
        "reward_weight": float(params.get("constraint_selection_reward_weight", 1.0)),
        "constraint_cost_weight": float(
            params.get("constraint_selection_constraint_cost_weight", 1.0)
        ),
        "constraint_active_ratio_weight": float(
            params.get("constraint_selection_constraint_active_ratio_weight", 0.0)
        ),
        "constraint_cost_mode": str(
            params.get("constraint_selection_cost_mode", "overall")
        ),
        "early_termination_weight": float(
            params.get("constraint_selection_early_termination_weight", 10.0)
        ),
        "harvest_fail_weight": float(
            params.get("constraint_selection_harvest_fail_weight", 1.0)
        ),
        "safety_override_weight": float(
            params.get("constraint_selection_safety_override_weight", 0.1)
        ),
    }


def _load_run_params_for_eval(load_ref: str) -> tuple[dict[str, Any], dict[str, Any]]:
    params = load_rl_params()
    phase_episode_keys = [
        "train_episode_length_mode",
        "train_episode_days",
        "train_episode_length_mix",
        "test_episode_length_mode",
        "test_episode_days",
        "test_episode_length_mix",
        "eval_episode_length_mode",
        "eval_episode_days",
        "eval_episode_length_mix",
    ]
    saved_run_cfg = load_saved_run_config(load_ref, ROOT)
    if not saved_run_cfg:
        raise RuntimeError(
            f"run_config.json not found for {load_ref}. "
            "Please point --load to a finished or in-progress RL run directory."
        )

    params.update(saved_run_cfg)
    missing_phase_keys = [key for key in phase_episode_keys if key not in saved_run_cfg]
    if missing_phase_keys:
        for key in missing_phase_keys:
            params[key] = None
    if "observation_semantics" not in saved_run_cfg:
        params["observation_semantics"] = "legacy31"
    if "auto_nstep" not in saved_run_cfg:
        params["auto_nstep"] = False
    if "controller_design" not in saved_run_cfg:
        action_semantics = str(params.get("action_semantics", "absolute")).lower()
        if action_semantics == "absolute":
            params["controller_design"] = "contextual_sac"
        elif action_semantics in {"residual_pid_gated", "pid_residual_gated"}:
            params["controller_design"] = "gated_residual_pid_sac"
        elif is_climate_only_residual_scale(params.get("residual_action_scale")):
            params["controller_design"] = "climate_only_residual_pid_sac"
        else:
            params["controller_design"] = "residual_pid_sac"
    if "curriculum_profile" not in saved_run_cfg:
        params["curriculum_profile"] = (
            "config" if params.get("context_curriculum") not in (None, "", []) else "off"
        )
    if "runtime_profile" not in saved_run_cfg:
        params["runtime_profile"] = "default"
    if "horizon_profile" not in saved_run_cfg:
        params["horizon_profile"] = "config"

    apply_horizon_profile(params, params.get("horizon_profile", "config"))
    apply_runtime_profile(params, params.get("runtime_profile", "default"))
    apply_controller_design(params, params.get("controller_design"))
    apply_curriculum_profile(params, params.get("curriculum_profile"))
    if str(params.get("curriculum_profile", "off")).strip().lower() != "config":
        apply_curriculum_profile(params, params.get("curriculum_profile"))

    sync_inlet_seedling_metadata(
        params,
        project_root=ROOT,
        fallback_preset="external_nursery_proxy",
    )
    return params, saved_run_cfg


def _build_policy(params: dict[str, Any], device: str):
    return create_policy(
        hidden_sizes=tuple(params["hidden_sizes"]),
        gamma=params["gamma"],
        actor_lr=params["actor_lr"],
        critic_lr=params["critic_lr"],
        alpha_lr=params["alpha_lr"],
        auto_alpha=params["auto_alpha"],
        device=device,
        run_params=params,
        project_root=ROOT,
    )


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _flatten_summary_row(result: dict[str, Any]) -> dict[str, Any]:
    score_components = result.get("selection_score_components", {}) or {}
    return {
        "requested_checkpoint": str(result.get("requested_checkpoint", "")),
        "loaded_checkpoint_kind": str(result.get("loaded_checkpoint_kind", "")),
        "loaded_checkpoint_path": str(result.get("loaded_checkpoint_path", "")),
        "duplicate_of_request": str(result.get("duplicate_of_request", "")),
        "selection_score": _safe_float(result.get("selection_score")),
        "selection_reward_component": _safe_float(score_components.get("reward")),
        "selection_constraint_cost_component": _safe_float(
            score_components.get("constraint_cost")
        ),
        "selection_constraint_active_ratio_component": _safe_float(
            score_components.get("constraint_active_ratio")
        ),
        "selection_early_termination_component": _safe_float(
            score_components.get("early_termination")
        ),
        "selection_harvest_fail_component": _safe_float(
            score_components.get("harvest_fail")
        ),
        "selection_safety_override_component": _safe_float(
            score_components.get("safety_override")
        ),
        "mean_reward": _safe_float(result.get("mean_reward")),
        "std_reward": _safe_float(result.get("std_reward")),
        "mean_reward_per_day": _safe_float(result.get("mean_reward_per_day")),
        "mean_constraint_cost": _safe_float(result.get("mean_constraint_cost")),
        "mean_constraint_cost_per_day": _safe_float(
            result.get("mean_constraint_cost_per_day")
        ),
        "mean_constraint_active_ratio": _safe_float(
            result.get("mean_constraint_active_ratio")
        ),
        "mean_completion_ratio": _safe_float(result.get("mean_completion_ratio")),
        "early_termination_ratio": _safe_float(result.get("early_termination_ratio")),
        "harvest_fail_episode_ratio": _safe_float(
            result.get("harvest_fail_episode_ratio")
        ),
        "safety_override_episode_ratio": _safe_float(
            result.get("safety_override_episode_ratio")
        ),
        "mean_safety_overrides_per_episode": _safe_float(
            result.get("mean_safety_overrides_per_episode")
        ),
        "mean_harvest_g": _safe_float(result.get("mean_harvest_g")),
        "mean_harvest_g_per_day": _safe_float(result.get("mean_harvest_g_per_day")),
        "mean_cost": _safe_float(result.get("mean_cost")),
        "mean_cost_per_day": _safe_float(result.get("mean_cost_per_day")),
        "mean_sim_days": _safe_float(result.get("mean_sim_days")),
        "n_schedules": _safe_int(result.get("n_schedules")),
        "n_episodes_per_schedule": _safe_int(result.get("n_episodes_per_schedule")),
        "eval_schedule_selection": str(result.get("eval_schedule_selection", "")),
        "episode_length_mode": str(result.get("episode_length_mode", "")),
        "hour_of_day_mode": str(result.get("hour_of_day_mode", "")),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_markdown(
    path: Path,
    *,
    run_name: str,
    run_dir: Path,
    recommended: dict[str, Any] | None,
    evaluated_rows: list[dict[str, Any]],
    skipped_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    lines: list[str] = []
    lines.append(f"# RL Checkpoint Comparison: {run_name}")
    lines.append("")
    lines.append(f"- Run dir: `{run_dir}`")
    lines.append(f"- Device: `{args.device}`")
    lines.append(
        f"- Protocol: `{args.n_eval_schedules}` schedules x "
        f"`{args.n_eval_episodes_per_schedule}` episode(s) per schedule"
    )
    lines.append(f"- Eval seed: `{args.eval_seed}`")
    lines.append("")
    if recommended is not None:
        lines.append("## Recommended Checkpoint")
        lines.append("")
        lines.append(
            f"- Recommended request: `{recommended['requested_checkpoint']}`"
        )
        lines.append(
            f"- Resolved checkpoint: `{recommended['loaded_checkpoint_kind']}`"
        )
        lines.append(
            f"- Reason: highest constraint-aware selection score = "
            f"`{recommended['selection_score']:.4f}`"
        )
        lines.append(
            f"- Loaded path: `{recommended['loaded_checkpoint_path']}`"
        )
        lines.append("")

    if evaluated_rows:
        df = pd.DataFrame(evaluated_rows)[
            [
                "requested_checkpoint",
                "loaded_checkpoint_kind",
                "selection_score",
                "mean_reward",
                "mean_reward_per_day",
                "mean_constraint_cost",
                "mean_constraint_cost_per_day",
                "mean_completion_ratio",
                "early_termination_ratio",
                "loaded_checkpoint_path",
            ]
        ].copy()
        lines.append("## Evaluated Checkpoints")
        lines.append("")
        lines.append(df.to_markdown(index=False))
        lines.append("")

    if skipped_rows:
        lines.append("## Skipped Checkpoints")
        lines.append("")
        for row in skipped_rows:
            lines.append(
                f"- `{row['requested_checkpoint']}`: {row.get('reason', 'missing')}"
            )
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = resolve_experiment_dir(args.load, ROOT)
    if not run_dir.exists():
        raise RuntimeError(f"Run directory not found: {run_dir}")
    out_dir = Path(args.out_dir).resolve() if args.out_dir else run_dir / "checkpoint_eval_compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    params, saved_run_cfg = _load_run_params_for_eval(args.load)
    eval_episode_mode, eval_episode_days, eval_episode_mix = (
        resolve_phase_episode_settings_from_params(params, "eval")
    )
    if args.eval_selection is not None:
        params["eval_schedule_selection"] = args.eval_selection
    if args.eval_episode_length_mode is not None:
        eval_episode_mode = args.eval_episode_length_mode
    if args.eval_episode_days is not None:
        eval_episode_days = float(args.eval_episode_days)
    hour_of_day_mode = (
        args.hour_of_day_mode
        if args.hour_of_day_mode is not None
        else params.get("eval_hour_of_day_mode", "fixed")
    )

    selection_weights = _selection_weights_from_params(params)
    evaluated_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    first_request_by_path: dict[str, str] = {}

    for requested_checkpoint in args.checkpoints:
        try:
            policy = _build_policy(params, device=args.device)
            policy = load_policy(
                policy,
                args.load,
                device=args.device,
                checkpoint=requested_checkpoint,
                project_root=ROOT,
            )
        except FileNotFoundError as exc:
            skip_payload = {
                "requested_checkpoint": str(requested_checkpoint),
                "reason": str(exc),
            }
            skipped_rows.append(skip_payload)
            print(f"[skip] checkpoint={requested_checkpoint}: missing")
            if args.strict_missing:
                raise
            continue

        eval_results = evaluate_on_schedules(
            policy,
            n_schedules=args.n_eval_schedules,
            n_episodes_per_schedule=args.n_eval_episodes_per_schedule,
            seed=args.eval_seed,
            context_sampling_phase=params.get("context_sampling_phase", "full"),
            fixed_schedule=params.get("context_fixed_schedule"),
            narrow_bounds=params.get("context_narrow_bounds"),
            eval_schedule_selection=params.get("eval_schedule_selection", "coverage"),
            episode_length_mode=eval_episode_mode,
            episode_days=eval_episode_days,
            episode_length_mix=eval_episode_mix,
            hour_of_day_mode=hour_of_day_mode,
            rl_param_overrides=params,
            verbose=not args.quiet,
        )
        attach_inlet_seedling_metadata(eval_results, params)
        selection_score = compute_constraint_aware_selection_score(
            eval_results,
            **selection_weights,
        )
        loaded_path = str(getattr(policy, "_loaded_checkpoint_path", ""))
        loaded_kind = str(
            getattr(policy, "_loaded_checkpoint_kind", requested_checkpoint)
        )
        duplicate_of_request = first_request_by_path.get(loaded_path, "")
        if loaded_path and not duplicate_of_request:
            first_request_by_path[loaded_path] = str(requested_checkpoint)

        eval_results["requested_checkpoint"] = str(requested_checkpoint)
        eval_results["loaded_checkpoint_kind"] = loaded_kind
        eval_results["loaded_checkpoint_path"] = loaded_path
        eval_results["duplicate_of_request"] = str(duplicate_of_request)
        eval_results["selection_score"] = float(selection_score["score"])
        eval_results["selection_score_components"] = dict(selection_score["components"])
        eval_results["selection_score_weights"] = dict(selection_score["weights"])
        eval_results["run_dir"] = str(run_dir)
        eval_results["experiment"] = str(run_dir.name)
        eval_results["saved_run_config"] = {
            "controller_design": saved_run_cfg.get("controller_design"),
            "action_semantics": saved_run_cfg.get("action_semantics"),
            "curriculum_profile": saved_run_cfg.get("curriculum_profile"),
            "runtime_profile": saved_run_cfg.get("runtime_profile"),
            "horizon_profile": saved_run_cfg.get("horizon_profile"),
        }

        json_name = f"checkpoint_eval_{requested_checkpoint}.json"
        _write_json(out_dir / json_name, eval_results)
        evaluated_rows.append(eval_results)
        print(
            f"[ok] checkpoint={requested_checkpoint:<8s} "
            f"resolved={loaded_kind:<8s} "
            f"score={eval_results['selection_score']:.4f} "
            f"reward={_safe_float(eval_results.get('mean_reward')):.4f} "
            f"constraint={_safe_float(eval_results.get('mean_constraint_cost')):.4f}"
        )

    if not evaluated_rows:
        raise RuntimeError("No checkpoints were evaluated successfully.")

    comparison_rows = [_flatten_summary_row(row) for row in evaluated_rows]
    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        by=["selection_score", "mean_reward", "mean_constraint_cost"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    comparison_df["rank_by_selection_score"] = comparison_df["selection_score"].rank(
        method="dense",
        ascending=False,
    ).astype(int)
    comparison_df["rank_by_mean_reward"] = comparison_df["mean_reward"].rank(
        method="dense",
        ascending=False,
    ).astype(int)
    comparison_df["rank_by_constraint_cost"] = comparison_df["mean_constraint_cost"].rank(
        method="dense",
        ascending=True,
    ).astype(int)

    recommended_row = comparison_df.iloc[0].to_dict()
    comparison_csv = out_dir / "checkpoint_comparison.csv"
    comparison_json = out_dir / "checkpoint_comparison.json"
    comparison_md = out_dir / "checkpoint_comparison.md"
    request_json = out_dir / "checkpoint_eval_request.json"

    comparison_df.to_csv(comparison_csv, index=False, encoding="utf-8")
    comparison_payload = {
        "run_dir": str(run_dir),
        "experiment": str(run_dir.name),
        "recommended_checkpoint": recommended_row,
        "evaluated_checkpoints": evaluated_rows,
        "skipped_checkpoints": skipped_rows,
        "evaluation_protocol": {
            "device": str(args.device),
            "checkpoints": [str(item) for item in args.checkpoints],
            "n_eval_schedules": int(args.n_eval_schedules),
            "n_eval_episodes_per_schedule": int(args.n_eval_episodes_per_schedule),
            "eval_seed": int(args.eval_seed),
            "eval_schedule_selection": str(params.get("eval_schedule_selection", "")),
            "eval_episode_length_mode": str(eval_episode_mode),
            "eval_episode_days": (
                None if eval_episode_days is None else float(eval_episode_days)
            ),
            "hour_of_day_mode": str(hour_of_day_mode),
            "selection_weights": dict(selection_weights),
        },
    }
    _write_json(comparison_json, comparison_payload)
    _write_json(
        request_json,
        {
            "load": str(args.load),
            "resolved_run_dir": str(run_dir),
            "device": str(args.device),
            "checkpoints": [str(item) for item in args.checkpoints],
            "n_eval_schedules": int(args.n_eval_schedules),
            "n_eval_episodes_per_schedule": int(args.n_eval_episodes_per_schedule),
            "eval_seed": int(args.eval_seed),
            "eval_selection_override": args.eval_selection,
            "eval_episode_length_mode_override": args.eval_episode_length_mode,
            "eval_episode_days_override": args.eval_episode_days,
            "hour_of_day_mode_override": args.hour_of_day_mode,
        },
    )
    _write_markdown(
        comparison_md,
        run_name=run_dir.name,
        run_dir=run_dir,
        recommended=recommended_row,
        evaluated_rows=comparison_rows,
        skipped_rows=skipped_rows,
        args=args,
    )

    print("\n" + "=" * 72)
    print("RL Checkpoint Comparison Complete")
    print("=" * 72)
    print(f"Run dir               : {run_dir}")
    print(f"Output dir            : {out_dir}")
    print(f"Recommended request   : {recommended_row['requested_checkpoint']}")
    print(f"Resolved checkpoint   : {recommended_row['loaded_checkpoint_kind']}")
    print(f"Selection score       : {recommended_row['selection_score']:.4f}")
    print(f"Mean reward           : {recommended_row['mean_reward']:.4f}")
    print(f"Mean constraint cost  : {recommended_row['mean_constraint_cost']:.4f}")
    print(f"Comparison CSV        : {comparison_csv}")
    print(f"Comparison JSON       : {comparison_json}")
    print(f"Comparison Markdown   : {comparison_md}")
    if skipped_rows:
        print(f"Skipped checkpoints   : {len(skipped_rows)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
