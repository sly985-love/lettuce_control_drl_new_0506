# -*- coding: utf-8 -*-
"""Scan the full feasible schedule catalogue under multiple inlet-mass presets."""

from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
import statistics
import sys
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from envs.schedule_sampler import (
    REFERENCE_CLASS_INFEASIBLE,
    REFERENCE_CLASS_MARGINAL,
    REFERENCE_CLASS_TARGET,
    ScheduleSampler,
)
from envs.utils import DEFAULT_SCHEDULE, load_all_configs, prepare_runtime_config


REFERENCE_CLASSES = (
    REFERENCE_CLASS_TARGET,
    REFERENCE_CLASS_MARGINAL,
    REFERENCE_CLASS_INFEASIBLE,
)


def _schedule_key(schedule: Dict[str, Any]) -> Tuple[int, int, int, int]:
    return (
        int(schedule["t1"]),
        int(schedule["t2"]),
        int(schedule["N1"]),
        int(round(float(schedule["rho2"]))),
    )


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


def _make_runtime_for_preset(base_config: Dict[str, Any], preset_name: str) -> Dict[str, Any]:
    cfg = deepcopy(base_config)
    cfg.setdefault("container_params", {})
    cfg["container_params"]["initial_seedling_mass_preset"] = str(preset_name)
    return prepare_runtime_config(cfg)


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    return float(sum(vals) / max(len(vals), 1))


def _median(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    if not vals:
        return 0.0
    return float(statistics.median(vals))


def _top_schedules(rows: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    ranked = sorted(
        rows,
        key=lambda row: (
            float(row.get("reference_harvest_fresh_mass_per_plant_g", 0.0)),
            float(row.get("reference_harvest_vs_target_ratio", 0.0)),
            -float(row.get("total_cycle_days", row.get("t1", 0) + row.get("t2", 0))),
        ),
        reverse=True,
    )
    top_rows: List[Dict[str, Any]] = []
    for row in ranked[:top_k]:
        top_rows.append(
            {
                "schedule": {
                    "t1": int(row["t1"]),
                    "t2": int(row["t2"]),
                    "N1": int(row["N1"]),
                    "rho2": int(round(float(row["rho2"]))),
                },
                "rho1": float(row["rho1"]),
                "reference_class": str(row.get("reference_feasibility_class", "")),
                "harvest_fresh_g_per_plant": float(
                    row.get("reference_harvest_fresh_mass_per_plant_g", 0.0)
                ),
                "harvest_dry_g_per_plant": float(
                    row.get("reference_harvest_dry_mass_per_plant_g", 0.0)
                ),
                "target_ratio": float(row.get("reference_harvest_vs_target_ratio", 0.0)),
                "min_ratio": float(row.get("reference_harvest_vs_min_ratio", 0.0)),
            }
        )
    return top_rows


def _summarize_rows(
    rows: List[Dict[str, Any]],
    *,
    runtime: Dict[str, Any],
    default_key: Tuple[int, int, int, int],
    top_k: int,
) -> Dict[str, Any]:
    class_counts = {cls: 0 for cls in REFERENCE_CLASSES}
    harvest_fw = []
    harvest_dw = []
    target_ratio = []
    min_ratio = []

    default_row = None
    for row in rows:
        cls = str(row.get("reference_feasibility_class", ""))
        if cls in class_counts:
            class_counts[cls] += 1
        harvest_fw.append(float(row.get("reference_harvest_fresh_mass_per_plant_g", 0.0)))
        harvest_dw.append(float(row.get("reference_harvest_dry_mass_per_plant_g", 0.0)))
        target_ratio.append(float(row.get("reference_harvest_vs_target_ratio", 0.0)))
        min_ratio.append(float(row.get("reference_harvest_vs_min_ratio", 0.0)))
        if _schedule_key(row) == default_key:
            default_row = row

    return {
        "n_schedules": int(len(rows)),
        "reference_class_counts": class_counts,
        "harvest_fresh_g_per_plant": {
            "mean": _mean(harvest_fw),
            "median": _median(harvest_fw),
            "min": min(harvest_fw) if harvest_fw else 0.0,
            "max": max(harvest_fw) if harvest_fw else 0.0,
        },
        "harvest_dry_g_per_plant": {
            "mean": _mean(harvest_dw),
            "median": _median(harvest_dw),
            "min": min(harvest_dw) if harvest_dw else 0.0,
            "max": max(harvest_dw) if harvest_dw else 0.0,
        },
        "target_ratio": {
            "mean": _mean(target_ratio),
            "median": _median(target_ratio),
            "min": min(target_ratio) if target_ratio else 0.0,
            "max": max(target_ratio) if target_ratio else 0.0,
        },
        "min_ratio": {
            "mean": _mean(min_ratio),
            "median": _median(min_ratio),
            "min": min(min_ratio) if min_ratio else 0.0,
            "max": max(min_ratio) if min_ratio else 0.0,
        },
        "default_schedule": _json_safe(default_row) if default_row is not None else None,
        "top_schedules_by_harvest_fresh_mass": _top_schedules(rows, top_k=top_k),
        "config_warning_count": int(len(runtime.get("config_warnings", []) or [])),
        "config_warnings": list(runtime.get("config_warnings", []) or []),
    }


def _compare_presets(
    rows_by_preset: Dict[str, List[Dict[str, Any]]],
    *,
    base_preset: str,
    compare_preset: str,
) -> Dict[str, Any]:
    base_rows = {_schedule_key(row): row for row in rows_by_preset[base_preset]}
    compare_rows = {_schedule_key(row): row for row in rows_by_preset[compare_preset]}
    shared_keys = sorted(set(base_rows.keys()) & set(compare_rows.keys()))

    class_transition_counts: Dict[str, int] = {}
    delta_fw = []
    delta_target_ratio = []
    upgrades_to_target = 0
    upgrades_to_min = 0
    downgrades_from_target = 0
    biggest_gain = None
    biggest_loss = None

    for key in shared_keys:
        row_a = base_rows[key]
        row_b = compare_rows[key]
        class_a = str(row_a.get("reference_feasibility_class", ""))
        class_b = str(row_b.get("reference_feasibility_class", ""))
        transition = f"{class_a} -> {class_b}"
        class_transition_counts[transition] = class_transition_counts.get(transition, 0) + 1

        fw_delta = float(row_b.get("reference_harvest_fresh_mass_per_plant_g", 0.0)) - float(
            row_a.get("reference_harvest_fresh_mass_per_plant_g", 0.0)
        )
        target_delta = float(row_b.get("reference_harvest_vs_target_ratio", 0.0)) - float(
            row_a.get("reference_harvest_vs_target_ratio", 0.0)
        )
        delta_fw.append(fw_delta)
        delta_target_ratio.append(target_delta)

        if class_b == REFERENCE_CLASS_TARGET and class_a != REFERENCE_CLASS_TARGET:
            upgrades_to_target += 1
        if class_b != REFERENCE_CLASS_INFEASIBLE and class_a == REFERENCE_CLASS_INFEASIBLE:
            upgrades_to_min += 1
        if class_a == REFERENCE_CLASS_TARGET and class_b != REFERENCE_CLASS_TARGET:
            downgrades_from_target += 1

        payload = {
            "schedule": {
                "t1": key[0],
                "t2": key[1],
                "N1": key[2],
                "rho2": key[3],
            },
            "base_class": class_a,
            "compare_class": class_b,
            "base_harvest_fresh_g_per_plant": float(
                row_a.get("reference_harvest_fresh_mass_per_plant_g", 0.0)
            ),
            "compare_harvest_fresh_g_per_plant": float(
                row_b.get("reference_harvest_fresh_mass_per_plant_g", 0.0)
            ),
            "delta_harvest_fresh_g_per_plant": float(fw_delta),
            "base_target_ratio": float(row_a.get("reference_harvest_vs_target_ratio", 0.0)),
            "compare_target_ratio": float(row_b.get("reference_harvest_vs_target_ratio", 0.0)),
            "delta_target_ratio": float(target_delta),
        }
        if biggest_gain is None or fw_delta > biggest_gain["delta_harvest_fresh_g_per_plant"]:
            biggest_gain = payload
        if biggest_loss is None or fw_delta < biggest_loss["delta_harvest_fresh_g_per_plant"]:
            biggest_loss = payload

    return {
        "base_preset": base_preset,
        "compare_preset": compare_preset,
        "n_shared_schedules": int(len(shared_keys)),
        "delta_harvest_fresh_g_per_plant": {
            "mean": _mean(delta_fw),
            "median": _median(delta_fw),
            "min": min(delta_fw) if delta_fw else 0.0,
            "max": max(delta_fw) if delta_fw else 0.0,
        },
        "delta_target_ratio": {
            "mean": _mean(delta_target_ratio),
            "median": _median(delta_target_ratio),
            "min": min(delta_target_ratio) if delta_target_ratio else 0.0,
            "max": max(delta_target_ratio) if delta_target_ratio else 0.0,
        },
        "class_transition_counts": dict(sorted(class_transition_counts.items())),
        "upgrades_to_target": int(upgrades_to_target),
        "upgrades_from_below_minimum": int(upgrades_to_min),
        "downgrades_from_target": int(downgrades_from_target),
        "biggest_gain_schedule": biggest_gain,
        "biggest_loss_schedule": biggest_loss,
    }


def _write_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan the full feasible schedule set under alternative inlet seedling-mass presets."
    )
    parser.add_argument(
        "--presets",
        type=str,
        default="strict_ref2,external_nursery_uniform,external_nursery_proxy",
        help="Comma-separated preset names defined in container_params.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top schedules to keep in the JSON summary for each preset.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/reference_catalog_by_inlet_preset",
        help="Output directory relative to repo root.",
    )
    args = parser.parse_args()

    preset_names = [token.strip() for token in str(args.presets).split(",") if token.strip()]
    if not preset_names:
        raise ValueError("At least one preset name is required.")

    base_config = load_all_configs(str(ROOT / "configs"))
    bounds = _build_bounds(dict(base_config.get("schedule_params", {}) or {}))
    default_key = _schedule_key(dict(DEFAULT_SCHEDULE))
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    rows_by_preset: Dict[str, List[Dict[str, Any]]] = {}
    summary: Dict[str, Any] = {
        "presets": {},
        "pairwise_comparison": None,
        "pairwise_comparisons": [],
        "schedule_bounds": _json_safe(bounds),
        "default_schedule": dict(DEFAULT_SCHEDULE),
    }

    for preset_name in preset_names:
        runtime = _make_runtime_for_preset(base_config, preset_name)
        sampler = ScheduleSampler(
            bounds,
            container_params=runtime["container_params"],
            crop_params=runtime["crop_params"],
            reward_params=runtime["reward_params"],
            steady_state_params=runtime["steady_state_params"],
        )
        rows = sampler.get_all_feasible_schedules(include_reference=True)
        rows_by_preset[preset_name] = rows

        resolved_preset = str(
            runtime["container_params"].get(
                "_resolved_initial_seedling_mass_preset",
                runtime["container_params"].get("initial_seedling_mass_preset", ""),
            )
        )
        for row in rows:
            row_local = dict(row)
            row_local["preset"] = preset_name
            row_local["resolved_preset"] = resolved_preset
            row_local["is_default_schedule"] = bool(_schedule_key(row_local) == default_key)
            all_rows.append(row_local)

        summary["presets"][preset_name] = _summarize_rows(
            rows,
            runtime=runtime,
            default_key=default_key,
            top_k=max(int(args.top_k), 1),
        )
        summary["presets"][preset_name]["resolved_preset"] = resolved_preset
        summary["presets"][preset_name]["reference_class_counts_from_sampler"] = (
            sampler.get_reference_class_counts()
        )

        print(
            f"[PRESET] {preset_name} -> resolved={resolved_preset} | "
            f"n={len(rows)} | class_counts={summary['presets'][preset_name]['reference_class_counts']}"
        )
        default_row = summary["presets"][preset_name]["default_schedule"]
        if default_row is not None:
            print(
                "          default x={14,14,20,36,16}: "
                f"harvest_fw={float(default_row.get('reference_harvest_fresh_mass_per_plant_g', 0.0)):.2f} g/plant | "
                f"class={default_row.get('reference_feasibility_class', '')}"
            )

    comparisons: List[Dict[str, Any]] = []
    if len(preset_names) >= 2:
        for i, base_preset in enumerate(preset_names[:-1]):
            for compare_preset in preset_names[i + 1:]:
                comparison = _compare_presets(
                    rows_by_preset,
                    base_preset=base_preset,
                    compare_preset=compare_preset,
                )
                comparisons.append(comparison)
                print(
                    f"[COMPARE] {base_preset} -> {compare_preset} | "
                    f"mean_delta_fw={comparison['delta_harvest_fresh_g_per_plant']['mean']:.2f} g/plant | "
                    f"upgrades_to_target={comparison['upgrades_to_target']}"
                )
        summary["pairwise_comparisons"] = comparisons
        summary["pairwise_comparison"] = comparisons[0]

    csv_path = out_dir / "reference_catalog_by_inlet_preset.csv"
    json_path = out_dir / "reference_catalog_by_inlet_preset_summary.json"
    _write_csv(all_rows, csv_path)
    json_path.write_text(json.dumps(_json_safe(summary), ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[SAVE] CSV -> {csv_path}")
    print(f"[SAVE] JSON -> {json_path}")


if __name__ == "__main__":
    main()
