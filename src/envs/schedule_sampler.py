# -*- coding: utf-8 -*-
"""Discrete feasible schedule sampler for contextual RL."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from envs.utils import (
    load_all_configs,
    prepare_runtime_config,
    resolve_fixed_photoperiod_hours,
)
from models.batch_manager import BatchManager


REFERENCE_CLASS_TARGET = "target_feasible"
REFERENCE_CLASS_MARGINAL = "min_feasible_only"
REFERENCE_CLASS_INFEASIBLE = "below_minimum"

REFERENCE_FILTER_ALIASES = {
    "min_feasible": "min_feasible",
    "feasible_min": "min_feasible",
    "target_feasible": "target_feasible",
    "feasible_target": "target_feasible",
    "marginal": "marginal",
    "min_feasible_only": "marginal",
    "infeasible": "infeasible",
    "below_minimum": "infeasible",
}

REFERENCE_CLASS_WEIGHT_ALIASES = {
    REFERENCE_CLASS_TARGET: REFERENCE_CLASS_TARGET,
    "target": REFERENCE_CLASS_TARGET,
    "target_feasible": REFERENCE_CLASS_TARGET,
    "feasible_target": REFERENCE_CLASS_TARGET,
    REFERENCE_CLASS_MARGINAL: REFERENCE_CLASS_MARGINAL,
    "marginal": REFERENCE_CLASS_MARGINAL,
    "min_feasible_only": REFERENCE_CLASS_MARGINAL,
    "minimum_only": REFERENCE_CLASS_MARGINAL,
    REFERENCE_CLASS_INFEASIBLE: REFERENCE_CLASS_INFEASIBLE,
    "infeasible": REFERENCE_CLASS_INFEASIBLE,
    "below_minimum": REFERENCE_CLASS_INFEASIBLE,
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _cache_key(payload: Dict[str, Any]) -> str:
    return json.dumps(_json_safe(payload), sort_keys=True, ensure_ascii=True)


def phase_requires_reference_metadata(phase: Optional[str]) -> bool:
    _, reference_filter = _parse_curriculum_phase(phase)
    return reference_filter is not None


def _parse_curriculum_phase(phase: Optional[str]) -> Tuple[str, Optional[str]]:
    phase_text = str(phase or "full").strip().lower()
    if phase_text in {"fixed", "narrow", "full"}:
        return phase_text, None

    for base_phase in ("fixed", "narrow", "full"):
        prefix = f"{base_phase}_"
        if phase_text.startswith(prefix):
            suffix = phase_text[len(prefix) :]
            reference_filter = REFERENCE_FILTER_ALIASES.get(suffix)
            if reference_filter is None:
                raise ValueError(f"Unknown contextual schedule phase: {phase_text}")
            return base_phase, reference_filter

    if phase_text in REFERENCE_FILTER_ALIASES:
        return "full", REFERENCE_FILTER_ALIASES[phase_text]

    raise ValueError(f"Unknown contextual schedule phase: {phase_text}")


def _schedule_key(schedule: Dict[str, Any]) -> Tuple[int, int, int, int]:
    return (
        int(schedule["t1"]),
        int(schedule["t2"]),
        int(schedule["N1"]),
        int(round(float(schedule["rho2"]))),
    )


def _coerce_positive_weight(value: Any, default: float = 1.0) -> float:
    try:
        weight = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(weight) or weight <= 0.0:
        return float(default)
    return float(weight)


def _normalise_reference_weight_map(raw: Optional[Dict[str, Any]]) -> Dict[str, float]:
    normalised: Dict[str, float] = {}
    for key, value in dict(raw or {}).items():
        canonical = REFERENCE_CLASS_WEIGHT_ALIASES.get(str(key).strip().lower())
        if canonical is None:
            continue
        normalised[canonical] = _coerce_positive_weight(value, default=1.0)
    return normalised


def _sampling_needs_reference_metadata(sampling_weights: Optional[Dict[str, Any]]) -> bool:
    weights = dict(sampling_weights or {})
    reference_weights = _normalise_reference_weight_map(
        weights.get("reference_class", weights.get("reference"))
    )
    return bool(reference_weights)


def _enumerate_feasible_schedules_from_bounds(bounds_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    t1_min = int(bounds_dict["t1_min"])
    t1_max = int(bounds_dict["t1_max"])
    t2_min = int(bounds_dict["t2_min"])
    t2_max = int(bounds_dict["t2_max"])
    n1_min = int(bounds_dict["N1_min"])
    n1_max = int(bounds_dict["N1_max"])
    rho2_min = float(bounds_dict["rho2_min"])
    rho2_max = float(bounds_dict["rho2_max"])
    fixed_pp = int(resolve_fixed_photoperiod_hours(bounds_dict))
    rho1_min = float(bounds_dict["rho1_min"])
    rho1_max = float(bounds_dict["rho1_max"])
    n_total = int(bounds_dict["N_total"])
    er_min = float(bounds_dict.get("er_min", 3.0))
    er_max = float(bounds_dict.get("er_max", 6.0))
    total_cycle_min = float(bounds_dict.get("total_cycle_min", 24.0))
    total_cycle_max = float(bounds_dict.get("total_cycle_max", 32.0))
    dt_values = tuple(
        int(v)
        for v in bounds_dict.get(
            "DT_values",
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        )
    )

    feasible: List[Dict[str, Any]] = []
    rho2_lo = int(math.ceil(rho2_min))
    rho2_hi = int(math.floor(rho2_max))

    for t1 in range(t1_min, t1_max + 1):
        for t2 in range(t2_min, t2_max + 1):
            for n1 in range(n1_min, n1_max + 1):
                for rho2 in range(rho2_lo, rho2_hi + 1):
                    total_cycle = float(t1 + t2)
                    if not (total_cycle_min <= total_cycle <= total_cycle_max):
                        continue

                    delta_t = math.gcd(t1, t2)
                    if delta_t not in dt_values:
                        continue

                    n2 = n_total - n1
                    k1 = t1 // delta_t
                    k2 = t2 // delta_t
                    if k1 < 1 or k2 < 1:
                        continue
                    if n1 % k1 != 0 or n2 % k2 != 0:
                        continue

                    denominator = n1 * t2
                    if denominator == 0:
                        continue
                    numerator = rho2 * n2 * t1
                    if numerator % denominator != 0:
                        continue

                    rho1 = numerator / denominator
                    if not (rho1_min <= rho1 <= rho1_max):
                        continue

                    er = rho1 / max(rho2, 1e-12)
                    if not (er_min <= er <= er_max):
                        continue

                    feasible.append(
                        {
                            "t1": int(t1),
                            "t2": int(t2),
                            "N1": int(n1),
                            "N2": int(n2),
                            "rho2": int(rho2),
                            "PP": int(fixed_pp),
                            "rho1": float(rho1),
                            "delta_t": int(delta_t),
                            "k1": int(k1),
                            "k2": int(k2),
                        }
                    )

    feasible.sort(key=_schedule_key)
    return feasible


def _classify_reference_profile(reference_growth_profile: Dict[str, Any]) -> str:
    min_ratio = float(reference_growth_profile.get("reference_harvest_vs_min_ratio", 0.0))
    target_ratio = float(reference_growth_profile.get("reference_harvest_vs_target_ratio", 0.0))
    if target_ratio >= 1.0:
        return REFERENCE_CLASS_TARGET
    if min_ratio >= 1.0:
        return REFERENCE_CLASS_MARGINAL
    return REFERENCE_CLASS_INFEASIBLE


@lru_cache(maxsize=1)
def _load_default_reference_bundle() -> Dict[str, Any]:
    config_dir = str(Path(__file__).resolve().parents[2] / "configs")
    runtime = prepare_runtime_config(load_all_configs(config_dir))
    return {
        "container_params": dict(runtime.get("container_params", {}) or {}),
        "crop_params": dict(runtime.get("crop_params", {}) or {}),
        "reward_params": dict(runtime.get("reward_params", {}) or {}),
        "steady_state_params": dict(runtime.get("steady_state_params", {}) or {}),
    }


@lru_cache(maxsize=8)
def _build_reference_catalog_cached(
    bounds_key: str,
    reference_bundle_key: str,
) -> Tuple[str, ...]:
    bounds_dict = json.loads(bounds_key)
    reference_bundle = json.loads(reference_bundle_key)
    container_params = dict(reference_bundle.get("container_params", {}) or {})
    crop_params = dict(reference_bundle.get("crop_params", {}) or {})
    reward_params = dict(reference_bundle.get("reward_params", {}) or {})
    steady_state_params = dict(reference_bundle.get("steady_state_params", {}) or {})

    feasible_schedules = _enumerate_feasible_schedules_from_bounds(bounds_dict)
    n_total = int(bounds_dict.get("N_total", 80))
    a_board = float(
        bounds_dict.get(
            "A_board",
            container_params.get(
                "A_board",
                float(container_params.get("c_total_plant_area", float(n_total))) / max(n_total, 1),
            ),
        )
    )
    rng_seed = int(reference_bundle.get("seed", 0))
    rng = np.random.default_rng(rng_seed)

    catalog_rows: List[str] = []
    for schedule in feasible_schedules:
        schedule_local = dict(schedule)
        n1 = int(schedule_local["N1"])
        n2 = int(schedule_local.get("N2", n_total - n1))
        a1 = float(n1 * a_board)
        a2 = float(n2 * a_board)
        a_total = float(a1 + a2)

        container_local = dict(container_params)
        container_local["A1"] = a1
        container_local["A2"] = a2
        container_local["_A1"] = a1
        container_local["_A2"] = a2
        container_local["_A_total"] = a_total
        container_local["c_total_plant_area"] = a_total
        container_local["disturb_factor_max"] = 0.0

        batch_manager = BatchManager(
            schedule_local,
            container_local,
            crop_params,
            rng=rng,
            steady_state_params=steady_state_params,
            reward_params=reward_params,
            initialise_batches=False,
        )
        reference_growth_profile = dict(
            getattr(batch_manager, "reference_growth_profile", {}) or {}
        )
        reference_class = _classify_reference_profile(reference_growth_profile)
        schedule_local.update(reference_growth_profile)
        schedule_local["reference_feasibility_class"] = reference_class
        schedule_local["reference_min_feasible"] = bool(
            float(reference_growth_profile.get("reference_harvest_vs_min_ratio", 0.0)) >= 1.0
        )
        schedule_local["reference_target_feasible"] = bool(
            float(reference_growth_profile.get("reference_harvest_vs_target_ratio", 0.0)) >= 1.0
        )
        catalog_rows.append(json.dumps(schedule_local, sort_keys=True))

    return tuple(catalog_rows)


class ScheduleSampler:
    """
    Sample schedules from the discrete feasible set.

    The design target in this project is:
      x = {t1, t2, N1, rho2} in Z^4 with fixed PP=16 injected at runtime

    The sampler enumerates all discrete schedules that satisfy the structural
    constraints once, then optionally augments them with nominal reference
    growth metadata so training and evaluation can distinguish:
      - target-feasible schedules
      - minimum-only feasible schedules
      - below-minimum schedules
    """

    def __init__(
        self,
        bounds_dict: Dict[str, Any],
        *,
        container_params: Optional[Dict[str, Any]] = None,
        crop_params: Optional[Dict[str, Any]] = None,
        reward_params: Optional[Dict[str, Any]] = None,
        steady_state_params: Optional[Dict[str, Any]] = None,
    ):
        self.bounds_dict = dict(bounds_dict or {})
        self.t1_min = int(self.bounds_dict["t1_min"])
        self.t1_max = int(self.bounds_dict["t1_max"])
        self.t2_min = int(self.bounds_dict["t2_min"])
        self.t2_max = int(self.bounds_dict["t2_max"])
        self.N1_min = int(self.bounds_dict["N1_min"])
        self.N1_max = int(self.bounds_dict["N1_max"])
        self.rho2_min = float(self.bounds_dict["rho2_min"])
        self.rho2_max = float(self.bounds_dict["rho2_max"])
        self.fixed_pp = int(resolve_fixed_photoperiod_hours(self.bounds_dict))
        self.PP_min = int(self.fixed_pp)
        self.PP_max = int(self.fixed_pp)
        self.rho1_min = float(self.bounds_dict["rho1_min"])
        self.rho1_max = float(self.bounds_dict["rho1_max"])
        self.N_total = int(self.bounds_dict["N_total"])
        self.er_min = float(self.bounds_dict.get("er_min", 3.0))
        self.er_max = float(self.bounds_dict.get("er_max", 6.0))
        self.total_cycle_min = float(self.bounds_dict.get("total_cycle_min", 24.0))
        self.total_cycle_max = float(self.bounds_dict.get("total_cycle_max", 32.0))
        self.DT_values = tuple(
            int(v)
            for v in self.bounds_dict.get(
                "DT_values",
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
            )
        )

        self._feasible_schedules = _enumerate_feasible_schedules_from_bounds(self.bounds_dict)
        if not self._feasible_schedules:
            raise RuntimeError("No feasible schedules found under the configured discrete bounds.")

        if (
            container_params is None
            or crop_params is None
            or reward_params is None
            or steady_state_params is None
        ):
            self._reference_bundle = deepcopy(_load_default_reference_bundle())
        else:
            self._reference_bundle = {
                "container_params": dict(container_params or {}),
                "crop_params": dict(crop_params or {}),
                "reward_params": dict(reward_params or {}),
                "steady_state_params": dict(steady_state_params or {}),
            }
        self._reference_catalog: Optional[List[Dict[str, Any]]] = None
        self._reference_index: Optional[Dict[Tuple[int, int, int, int], Dict[str, Any]]] = None

    def sample(
        self,
        curriculum_phase: str = "full",
        fixed_schedule: Optional[Dict[str, Any]] = None,
        narrow_bounds: Optional[Dict[str, Any]] = None,
        sampling_weights: Optional[Dict[str, Any]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, Any]:
        """Sample one schedule from the candidate catalogue."""
        if rng is None:
            rng = np.random.default_rng()
        candidates = self.get_candidate_pool(
            curriculum_phase=curriculum_phase,
            fixed_schedule=fixed_schedule,
            narrow_bounds=narrow_bounds,
            include_reference=_sampling_needs_reference_metadata(sampling_weights),
        )
        return self._sample_from_candidates(
            candidates,
            rng,
            phase=str(curriculum_phase or "full"),
            sampling_weights=sampling_weights,
        )

    def get_candidate_pool(
        self,
        curriculum_phase: str = "full",
        fixed_schedule: Optional[Dict[str, Any]] = None,
        narrow_bounds: Optional[Dict[str, Any]] = None,
        *,
        include_reference: bool = False,
    ) -> List[Dict[str, Any]]:
        base_phase, reference_filter = _parse_curriculum_phase(curriculum_phase)
        need_reference = bool(include_reference or reference_filter is not None)

        if base_phase == "fixed":
            schedule = self._sample_fixed(fixed_schedule, include_reference=need_reference)
            return [schedule]

        if base_phase == "narrow":
            candidates = self._filter_narrow_candidates(
                self._reference_schedules() if need_reference else self._feasible_schedules,
                narrow_bounds,
            )
        else:
            candidates = self._reference_schedules() if need_reference else self._feasible_schedules

        if reference_filter is not None:
            candidates = self._filter_reference_candidates(candidates, reference_filter)

        return [dict(s) for s in candidates]

    def _sample_fixed(
        self,
        fixed_schedule: Optional[Dict[str, Any]],
        *,
        include_reference: bool = False,
    ) -> Dict[str, Any]:
        if fixed_schedule is None:
            fixed_schedule = self.get_default_schedule()
        schedule = self._enforce_constraints(fixed_schedule)
        if schedule is None:
            raise ValueError(f"Fixed schedule is infeasible: {fixed_schedule}")
        if not include_reference:
            return schedule
        schedule_ref = self._reference_schedule_by_key().get(_schedule_key(schedule))
        if schedule_ref is None:
            return schedule
        return dict(schedule_ref)

    def _filter_narrow_candidates(
        self,
        candidates: List[Dict[str, Any]],
        narrow_bounds: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if narrow_bounds is None:
            narrow_bounds = {
                "t1_min": 12,
                "t1_max": 16,
                "t2_min": 12,
                "t2_max": 16,
                "N1_min": 16,
                "N1_max": 20,
                "rho2_min": 28.0,
                "rho2_max": 40.0,
            }

        t1_min = int(narrow_bounds["t1_min"])
        t1_max = int(narrow_bounds["t1_max"])
        t2_min = int(narrow_bounds["t2_min"])
        t2_max = int(narrow_bounds["t2_max"])
        n1_min = int(narrow_bounds["N1_min"])
        n1_max = int(narrow_bounds["N1_max"])
        rho2_min = int(math.ceil(float(narrow_bounds["rho2_min"])))
        rho2_max = int(math.floor(float(narrow_bounds["rho2_max"])))

        return [
            dict(schedule)
            for schedule in candidates
            if (
                t1_min <= int(schedule["t1"]) <= t1_max
                and t2_min <= int(schedule["t2"]) <= t2_max
                and n1_min <= int(schedule["N1"]) <= n1_max
                and rho2_min <= int(round(float(schedule["rho2"]))) <= rho2_max
            )
        ]

    def _filter_reference_candidates(
        self,
        candidates: List[Dict[str, Any]],
        reference_filter: str,
    ) -> List[Dict[str, Any]]:
        if reference_filter == "min_feasible":
            return [
                dict(schedule)
                for schedule in candidates
                if bool(schedule.get("reference_min_feasible", False))
            ]
        if reference_filter == "target_feasible":
            return [
                dict(schedule)
                for schedule in candidates
                if bool(schedule.get("reference_target_feasible", False))
            ]
        if reference_filter == "marginal":
            return [
                dict(schedule)
                for schedule in candidates
                if str(schedule.get("reference_feasibility_class", "")) == REFERENCE_CLASS_MARGINAL
            ]
        if reference_filter == "infeasible":
            return [
                dict(schedule)
                for schedule in candidates
                if str(schedule.get("reference_feasibility_class", "")) == REFERENCE_CLASS_INFEASIBLE
            ]
        raise ValueError(f"Unknown reference schedule filter: {reference_filter}")

    def _sample_from_candidates(
        self,
        candidates: List[Dict[str, Any]],
        rng: np.random.Generator,
        phase: str,
        sampling_weights: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not candidates:
            raise RuntimeError(f"No feasible schedules available for phase '{phase}'.")
        weights = dict(sampling_weights or {})
        reference_weights = _normalise_reference_weight_map(
            weights.get("reference_class", weights.get("reference"))
        )
        if not reference_weights:
            idx = int(rng.integers(0, len(candidates)))
            return dict(candidates[idx])

        sample_weights = np.ones(len(candidates), dtype=np.float64)
        for i, schedule in enumerate(candidates):
            weight = 1.0
            if reference_weights:
                cls = str(schedule.get("reference_feasibility_class", "")).strip().lower()
                canonical_cls = REFERENCE_CLASS_WEIGHT_ALIASES.get(cls, cls)
                weight *= reference_weights.get(canonical_cls, 1.0)
            sample_weights[i] = max(float(weight), 1e-9)

        weight_sum = float(sample_weights.sum())
        if not np.isfinite(weight_sum) or weight_sum <= 0.0:
            idx = int(rng.integers(0, len(candidates)))
            return dict(candidates[idx])
        idx = int(rng.choice(len(candidates), p=sample_weights / weight_sum))
        return dict(candidates[idx])

    def _enforce_constraints(self, raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        t1 = int(raw["t1"])
        t2 = int(raw["t2"])
        n1 = int(raw["N1"])
        rho2 = int(round(float(raw["rho2"])))
        pp = int(raw.get("PP", self.fixed_pp))

        if not (self.t1_min <= t1 <= self.t1_max):
            return None
        if not (self.t2_min <= t2 <= self.t2_max):
            return None
        if not (self.N1_min <= n1 <= self.N1_max):
            return None
        if pp != self.fixed_pp:
            return None
        if not (self.rho2_min <= rho2 <= self.rho2_max):
            return None

        total_cycle = float(t1 + t2)
        if not (self.total_cycle_min <= total_cycle <= self.total_cycle_max):
            return None

        delta_t = math.gcd(t1, t2)
        if delta_t not in self.DT_values:
            return None

        n2 = self.N_total - n1
        k1 = t1 // delta_t
        k2 = t2 // delta_t
        if k1 < 1 or k2 < 1:
            return None
        if n1 % k1 != 0 or n2 % k2 != 0:
            return None

        denominator = n1 * t2
        if denominator == 0:
            return None
        numerator = rho2 * n2 * t1
        if numerator % denominator != 0:
            return None

        rho1 = numerator / denominator
        if not (self.rho1_min <= rho1 <= self.rho1_max):
            return None

        er = rho1 / max(rho2, 1e-12)
        if not (self.er_min <= er <= self.er_max):
            return None

        return {
            "t1": int(t1),
            "t2": int(t2),
            "N1": int(n1),
            "N2": int(n2),
            "rho2": int(rho2),
            "PP": int(pp),
            "rho1": float(rho1),
            "delta_t": int(delta_t),
            "k1": int(k1),
            "k2": int(k2),
        }

    def _reference_schedules(self) -> List[Dict[str, Any]]:
        if self._reference_catalog is None:
            bounds_key = _cache_key(self.bounds_dict)
            reference_bundle_key = _cache_key(self._reference_bundle)
            rows = _build_reference_catalog_cached(bounds_key, reference_bundle_key)
            self._reference_catalog = [json.loads(row) for row in rows]
            self._reference_catalog.sort(key=_schedule_key)
        return [dict(schedule) for schedule in self._reference_catalog]

    def _reference_schedule_by_key(self) -> Dict[Tuple[int, int, int, int], Dict[str, Any]]:
        if self._reference_index is None:
            self._reference_index = {
                _schedule_key(schedule): dict(schedule)
                for schedule in self._reference_schedules()
            }
        return self._reference_index

    def get_reference_class_counts(self) -> Dict[str, int]:
        counts = {
            REFERENCE_CLASS_TARGET: 0,
            REFERENCE_CLASS_MARGINAL: 0,
            REFERENCE_CLASS_INFEASIBLE: 0,
        }
        for schedule in self._reference_schedules():
            cls = str(schedule.get("reference_feasibility_class", ""))
            if cls in counts:
                counts[cls] += 1
        return counts

    def validate_schedule(self, schedule: Dict[str, Any]) -> Tuple[bool, str]:
        keys = ["t1", "t2", "N1", "rho2"]
        if not all(k in schedule for k in keys):
            return False, "Missing keys"

        validated = self._enforce_constraints(schedule)
        if validated is None:
            return False, "Schedule violates one or more discrete feasibility constraints"
        return True, "OK"

    def get_default_schedule(self) -> Dict[str, Any]:
        return {
            "t1": 14,
            "t2": 14,
            "N1": 20,
            "rho2": 36,
        }

    def get_all_feasible_schedules(self, *, include_reference: bool = False) -> List[Dict[str, Any]]:
        if include_reference:
            return self._reference_schedules()
        return [dict(schedule) for schedule in self._feasible_schedules]
