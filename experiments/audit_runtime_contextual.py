# -*- coding: utf-8 -*-
"""Run a contextual PFAL episode and audit batch events plus env-step clipping."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from controllers.pfal_conventional_controller import PFALConventionalController
from envs.plant_factory_env_new import PFALEnvContextual
from envs.utils import (
    apply_inlet_seedling_metadata,
    extract_inlet_seedling_metadata,
    has_inlet_seedling_metadata,
    load_all_configs,
    prepare_runtime_config,
)


CONFIG_DIR = ROOT / "configs"
LEGACY_LOAD_INLET_PRESET = "external_nursery_proxy"


def _parse_schedule_arg(schedule_text: str | None) -> dict | None:
    if not schedule_text:
        return None
    schedule = {}
    for part in schedule_text.split(","):
        token = part.strip()
        if not token:
            continue
        key, value = token.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key in {"t1", "t2", "N1", "PP"}:
            schedule[key] = int(float(value))
        else:
            schedule[key] = float(value)
    return schedule


def _slot_audit(slots: list[int], expected_count: int) -> dict:
    expected = set(range(max(expected_count, 0)))
    actual = list(sorted(int(s) for s in slots))
    actual_set = set(actual)
    return {
        "slots": actual,
        "unique_count": len(actual_set),
        "has_duplicates": len(actual) != len(actual_set),
        "missing_slots": sorted(expected - actual_set),
        "unexpected_slots": sorted(actual_set - expected),
    }


def _resolve_run_config_path(load_ref: str | None) -> Path | None:
    if not load_ref:
        return None
    candidate = Path(str(load_ref))
    if candidate.is_file():
        if candidate.name.lower() == "run_config.json":
            return candidate.resolve()
        return (candidate.parent / "run_config.json").resolve()
    if candidate.exists():
        return (candidate / "run_config.json").resolve()
    return (
        ROOT
        / "log"
        / "PFAL-contextual-SAC"
        / "sac_contextual"
        / str(load_ref)
        / "run_config.json"
    ).resolve()


def _load_run_config_with_legacy_fallback(load_ref: str | None) -> dict:
    run_cfg_path = _resolve_run_config_path(load_ref)
    run_cfg: dict = {}
    if run_cfg_path and run_cfg_path.exists():
        with open(run_cfg_path, "r", encoding="utf-8") as f:
            payload = json.load(f) or {}
        if isinstance(payload, dict):
            run_cfg = dict(payload)
    if load_ref and not has_inlet_seedling_metadata(run_cfg):
        run_cfg["initial_seedling_mass_preset"] = LEGACY_LOAD_INLET_PRESET
        run_cfg["inlet_seedling_metadata_source"] = str(
            run_cfg.get("inlet_seedling_metadata_source") or "legacy_load_fallback"
        )
    return run_cfg


def _build_env(schedule: dict | None, load_ref: str | None) -> tuple[PFALEnvContextual, dict]:
    cfg = load_all_configs(str(CONFIG_DIR))
    run_cfg = _load_run_config_with_legacy_fallback(load_ref)
    cfg = apply_inlet_seedling_metadata(cfg, run_cfg)
    runtime_cfg = prepare_runtime_config(cfg, schedule=schedule)
    env = PFALEnvContextual(config=runtime_cfg)
    return env, extract_inlet_seedling_metadata(runtime_cfg, assume_runtime=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit runtime batch events and environment clipping for the contextual PFAL simulator."
    )
    parser.add_argument(
        "--schedule",
        type=str,
        default=None,
        help='Optional schedule string like "t1=14,t2=14,N1=20,rho2=36".',
    )
    parser.add_argument(
        "--episode_days",
        type=float,
        default=35.0,
        help="Episode length in days for the audit run.",
    )
    parser.add_argument(
        "--hour_of_day",
        type=float,
        default=10.0,
        help="Reset hour of day.",
    )
    parser.add_argument(
        "--max_events",
        type=int,
        default=20,
        help="Maximum number of event rows to print.",
    )
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        help="Optional experiment dir/name or run_config.json path used to restore saved runtime metadata.",
    )
    args = parser.parse_args()

    schedule = _parse_schedule_arg(args.schedule)
    env, inlet_meta = _build_env(schedule, args.load)
    controller = PFALConventionalController(env)
    obs, info = env.reset(
        options={
            "schedule": schedule,
            "hour_of_day": float(args.hour_of_day),
            "episode_length_mode": "fixed_days",
            "episode_days": float(args.episode_days),
        }
    )
    controller.reset()

    prev_harvests = int(info.get("n_harvests", 0))
    prev_transplants = int(info.get("n_transplants", 0))
    event_rows: list[dict] = []
    reward_sum = 0.0
    delta_seed_max = 0.0
    delta_trans_max = 0.0
    delta_seed_mean_abs = 0.0
    delta_trans_mean_abs = 0.0
    n_steps = 0

    while True:
        action = controller.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        reward_sum += float(reward)
        n_steps += 1

        lumped = env.batch_manager._extract_lumped_features()
        delta_seed = float(lumped.get("delta_density_seedling", 0.0))
        delta_trans = float(lumped.get("delta_density_transplant", 0.0))
        delta_seed_max = max(delta_seed_max, abs(delta_seed))
        delta_trans_max = max(delta_trans_max, abs(delta_trans))
        delta_seed_mean_abs += abs(delta_seed)
        delta_trans_mean_abs += abs(delta_trans)

        harvests = int(info.get("n_harvests", 0))
        transplants = int(info.get("n_transplants", 0))
        if (
            float(info.get("harvest_mass_g", 0.0)) > 0.0
            or harvests != prev_harvests
            or transplants != prev_transplants
        ):
            event_rows.append(
                {
                    "step": int(info.get("time_step", 0)),
                    "day": float(info.get("sim_days_elapsed", 0.0)),
                    "harvests": harvests,
                    "transplants": transplants,
                    "harvest_mass_g": float(info.get("harvest_mass_g", 0.0)),
                    "harvest_fail": bool(info.get("harvest_fail", False)),
                    "harvest_mean_dry_mass_g": float(
                        info.get("harvest_mean_dry_mass_per_plant_g", 0.0)
                    ),
                    "harvest_mean_fresh_mass_g": float(
                        info.get("harvest_mean_fresh_mass_per_plant_g", 0.0)
                    ),
                    "oldest_dry_mass_g": float(info.get("oldest_dry_mass_per_plant_g", 0.0)),
                    "seedling_slots": sorted(
                        int(b.pipeline_slot) for b in env.batch_manager.seedling_batches
                    ),
                    "transplant_slots": sorted(
                        int(b.pipeline_slot) for b in env.batch_manager.transplant_batches
                    ),
                }
            )
        prev_harvests = harvests
        prev_transplants = transplants

        if terminated or truncated:
            break

    batch_mgr = env.batch_manager
    seedling_audit = _slot_audit(
        [int(b.pipeline_slot) for b in batch_mgr.seedling_batches],
        batch_mgr.k1,
    )
    transplant_audit = _slot_audit(
        [int(b.pipeline_slot) for b in batch_mgr.transplant_batches],
        batch_mgr.k2,
    )

    print("=== Runtime Audit Summary ===")
    print(
        "Inlet seedling metadata: "
        f"preset={inlet_meta.get('initial_seedling_mass_preset', '')} "
        f"basis={inlet_meta.get('initial_seedling_mass_basis', '')}"
    )
    print(
        f"Schedule: t1={env.t1} t2={env.t2} N1={env.N1} rho2={env.rho2:.1f} PP={env.PP}"
    )
    print(
        f"Episode: {info.get('actual_episode_days', 0.0):.2f} d, "
        f"{info.get('actual_episode_steps', 0)} steps, dt={env.dt:.1f} s"
    )
    print(
        f"Termination: reason={info.get('termination_reason', 'unknown')} "
        f"terminated={info.get('terminated', False)} truncated={info.get('truncated', False)}"
    )
    print(f"Reward sum: {reward_sum:.3f}")
    print(f"Harvests: {info.get('n_harvests', 0)} | Transplants: {info.get('n_transplants', 0)}")
    print(f"Total harvest mass: {env.total_harvest_mass_g:.1f} g DW")
    print(
        f"Minimum fresh mass: {info.get('min_fresh_mass_per_plant_g', 0.0):.1f} g/plant | "
        f"Target fresh mass: {info.get('target_fresh_mass_per_plant_g', 0.0):.1f} g/plant"
    )
    print(
        f"Current oldest fresh mass: {info.get('oldest_fresh_mass_per_plant_g', 0.0):.1f} g/plant"
    )
    print(
        f"Reference harvest fresh mass: {info.get('reference_harvest_fresh_mass_per_plant_g', 0.0):.1f} g/plant | "
        f"vs min={info.get('reference_harvest_vs_min_ratio', 0.0):.3f}x | "
        f"vs target={info.get('reference_harvest_vs_target_ratio', 0.0):.3f}x"
    )
    print(
        f"Reference class: {info.get('schedule_reference_feasibility_class', '')} | "
        f"min_feasible={info.get('schedule_reference_min_feasible', False)} | "
        f"target_feasible={info.get('schedule_reference_target_feasible', False)}"
    )
    schedule_warnings = list(info.get("schedule_reference_warnings", []) or [])
    if schedule_warnings:
        print("")
        print("=== Schedule Reference Warnings ===")
        for warning in schedule_warnings:
            print(f"- {warning}")
    print(
        f"Delta density | seedling max abs={delta_seed_max:.3f}, mean abs={delta_seed_mean_abs / max(n_steps, 1):.3f} g/m^2/h"
    )
    print(
        f"Delta density | transplant max abs={delta_trans_max:.3f}, mean abs={delta_trans_mean_abs / max(n_steps, 1):.3f} g/m^2/h"
    )

    clip_counts = dict(info.get("env_step_clip_counts", {}) or {})
    print("")
    print("=== Env-Step Clip Counts ===")
    for key in (
        "temperature",
        "co2_density",
        "co2_step_delta",
        "condensation",
        "humidity_floor",
        "humidity_saturation",
        "fallback",
        "safe_default",
    ):
        print(f"{key}: {int(clip_counts.get(key, 0))}")

    print("")
    print("=== Slot Audit ===")
    print(
        f"Dense slots: {seedling_audit['slots']} | duplicates={seedling_audit['has_duplicates']} "
        f"| missing={seedling_audit['missing_slots']} | unexpected={seedling_audit['unexpected_slots']}"
    )
    print(
        f"Finishing slots: {transplant_audit['slots']} | duplicates={transplant_audit['has_duplicates']} "
        f"| missing={transplant_audit['missing_slots']} | unexpected={transplant_audit['unexpected_slots']}"
    )

    if event_rows:
        print("")
        print("=== Event Trace ===")
        for row in event_rows[: max(int(args.max_events), 0)]:
            print(
                f"step={row['step']:5d} day={row['day']:6.2f} "
                f"harvests={row['harvests']:3d} transplants={row['transplants']:3d} "
                f"harvest_mass_g={row['harvest_mass_g']:8.2f} harvest_fail={row['harvest_fail']} "
                f"harvest_fw_g={row['harvest_mean_fresh_mass_g']:6.1f} "
                f"harvest_dw_g={row['harvest_mean_dry_mass_g']:5.2f} "
                f"oldest_dry_mass_g={row['oldest_dry_mass_g']:6.2f}"
            )
            print(
                f"  seedling_slots={row['seedling_slots']} transplant_slots={row['transplant_slots']}"
            )
    else:
        print("")
        print("=== Event Trace ===")
        print("No transplant/harvest events were observed in the requested horizon.")


if __name__ == "__main__":
    main()
