# -*- coding: utf-8 -*-
"""Contextual RL environment for the dual-zone PFAL simulator."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import os
import sys

import gymnasium as gym
from gymnasium import spaces
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    BatchManager,
    absolute_humidity_to_relative,
    absolute_humidity_to_vpd,
    calculate_energy_cost,
    describe_physical_setup,
    calculate_total_power,
    co2_density_to_ppm,
    co2_ppm_to_density,
    relative_humidity_to_absolute,
    resolve_led_params,
    simulate_environment_step,
    vpd_to_absolute_humidity,
)
from envs.schedule_sampler import ScheduleSampler
from envs.utils import (
    get_action_bounds,
    load_all_configs,
    prepare_runtime_config,
    resolve_fixed_photoperiod_hours,
    resolve_simulator_economics,
)
from utils import compute_electricity_price


class PFALEnvContextual(gym.Env):
    """Contextual RL environment for PFAL optimal control."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        default_config = load_all_configs(
            str(Path(__file__).resolve().parents[2] / "configs")
        )
        default_config.setdefault(
            "schedule",
            {"t1": 14, "t2": 14, "N1": 20, "rho2": 36.0},
        )
        default_config.setdefault("seed", 42)
        default_config.setdefault("dt", 600.0)
        merged_config = self._deep_update_dict(default_config, config or {})
        self.config = prepare_runtime_config(merged_config)

        self.schedule = self.config["schedule"]
        self.container_params = self.config["container_params"]
        self.crop_params = self.config["crop_params"]
        self.equipment_params = self.config["equipment_params"]
        self.reward_params = self.config["reward_params"]
        self.experiment_params = self.config.get("experiment_params", {})
        self.controller_params = self.config.get("controller_params", {})
        self.schedule_params = self.config.get("schedule_params", {})
        self.rl_params = self.config.get("rl_params", {})
        self.config_warnings = list(self.config.get("config_warnings", []) or [])
        self.reward_param_warnings = list(
            self.config.get("reward_param_warnings", []) or []
        )
        self.dt = float(self.config.get("dt", 600.0))
        pid_safety_cfg = self.controller_params.get("pid_controller", {})
        self.enable_action_safety_projection = bool(
            self.rl_params.get("enable_action_safety_projection", True)
        )
        self.observation_semantics = str(
            self.rl_params.get("observation_semantics", "target31_v2")
        ).lower()
        self.action_semantics = str(
            self.rl_params.get("action_semantics", "absolute")
        ).lower()
        self._residual_action_scale = self._resolve_residual_action_scale(
            self.rl_params.get("residual_action_scale", 0.75)
        )
        self.residual_gate_min = float(
            np.clip(self.rl_params.get("residual_gate_min", 0.0), 0.0, 1.0)
        )
        self.safety_temp_guard_band_c = float(
            self.rl_params.get("safety_temp_guard_band_c", 3.0)
        )
        self.safety_temp_projection_margin_c = float(
            self.rl_params.get("safety_temp_projection_margin_c", 0.5)
        )
        self.safety_temp_kp = float(
            self.rl_params.get(
                "safety_temp_kp",
                max(float(pid_safety_cfg.get("kp_temp", 25.0)), 60.0),
            )
        )
        self.info_detail_level = str(
            self.rl_params.get("info_detail_level", "rich")
        ).strip().lower()
        self.mask_schedule_context_observation = bool(
            self.rl_params.get("mask_schedule_context_observation", False)
        )
        self.light_control_mode = "step"
        self.light_segments_per_photoperiod = 3
        self.zone_semantics = {
            "seedling": "dense",
            "transplant": "finishing",
        }
        self.daily_dli_penalty_mode = self._normalise_daily_dli_penalty_mode(
            self.reward_params.get("daily_dli_penalty_mode", "disabled")
        )
        self.daily_photoperiod_penalty_mode = (
            self._normalise_daily_photoperiod_penalty_mode(
                self.reward_params.get(
                    "daily_photoperiod_penalty_mode",
                    "fixed_schedule_guard",
                )
            )
        )
        self.photoperiod_guard_tolerance_h = max(
            0.0,
            float(self.reward_params.get("photoperiod_guard_tolerance_h", 0.25)),
        )
        self.harvest_fail_penalty_mode = self._normalise_harvest_fail_penalty_mode(
            self.reward_params.get("harvest_fail_penalty_mode", "constraint_only")
        )

        cp = self.container_params

        self.c_alpha = float(cp.get("c_alpha", 0.68))
        self.c_beta = float(cp.get("c_beta", 0.8))
        self.c_bnd = float(cp.get("c_bnd", 0.004))
        self.c_car_1 = float(cp.get("c_car_1", -1.32e-5))
        self.c_car_2 = float(cp.get("c_car_2", 5.94e-4))
        self.c_car_3 = float(cp.get("c_car_3", -2.64e-3))
        self.c_eps = float(cp.get("c_eps", 17e-9))
        self.c_fw = float(cp.get("c_fw", 22.5))
        self.c_Gamma = float(cp.get("c_Gamma", 7.32e-5))
        self.c_k = float(cp.get("c_k", 0.9))
        self.c_lar_s = float(cp.get("c_lar_s", 75.0))
        self.c_par = float(cp.get("c_par", 1.0))
        self.c_Q10_Gamma = float(cp.get("c_Q10_Gamma", 2.0))
        self.c_Q10_gr = float(cp.get("c_Q10_gr", 1.6))
        self.c_Q10_resp = float(cp.get("c_Q10_resp", 2.0))
        self.c_rad_rf = float(cp.get("c_rad_rf", 1.0))
        self.c_r_gr_max = float(cp.get("c_r_gr_max", 5e-6))
        self.c_resp_s = float(cp.get("c_resp_s", 3.47e-7))
        self.c_resp_r = float(cp.get("c_resp_r", 1.16e-7))
        self.c_stm = float(cp.get("c_stm", 0.007))
        self.c_tau = float(cp.get("c_tau", 0.07))
        c_d2f_cfg = cp.get("c_d2f", None)
        self.c_d2f = (
            float(c_d2f_cfg)
            if c_d2f_cfg is not None
            else 1.0 / max(self.c_fw, 1e-12)
        )

        self.c_v_0 = float(cp.get("c_v_0", 0.85))
        self.c_v_1 = float(cp.get("c_v_1", 611.0))
        self.c_v_2 = float(cp.get("c_v_2", 17.4))
        self.c_v_3 = float(cp.get("c_v_3", 239.0))
        self.c_a_pl = float(cp.get("c_a_pl", 62.8))
        self.c_v_pl_ai = float(cp.get("c_v_pl_ai", 3.6e-3))
        self.mw_water = float(cp.get("mw_water", 18.0))
        self.c_R = float(cp.get("c_R", 8314.0))
        self.c_T_abs = float(cp.get("c_T_abs", 273.0))

        self.c_Length = float(cp.get("c_Length", 12.2))
        self.c_Width = float(cp.get("c_Width", 2.5))
        self.c_Height = float(cp.get("c_Height", 3.0))
        self.c_surface_area = float(cp.get("c_surface_area", 143.2))
        self.c_volume = float(cp.get("c_volume", 91.5))
        self.c_grow_area = float(cp.get("c_total_plant_area", 80.0))
        self.c_cap_q = float(cp.get("c_cap_q", 30000.0))
        self.c_cap_q_p = float(cp.get("c_cap_q_p", 1000.0))
        self.c_cap_q_v = float(cp.get("c_cap_q_v", 1290.0))
        self.c_lat_water = float(cp.get("c_lat_water", 2256.4))
        self.c_U = float(cp.get("c_U", 0.3))

        self.economics = self.config.get("economics") or resolve_simulator_economics(
            self.config, self.equipment_params
        )
        self.c_CO2 = float(self.economics["co2_price"])
        self.c_lettuce = float(self.economics["lettuce_price_fw"])
        self.c_elec = float(self.economics["electricity_price"])
        self.constant_electricity_price = float(
            self.experiment_params.get("constant_price", self.c_elec)
        )
        self.electricity_price_model = str(
            self.experiment_params.get(
                "price_model_type",
                self.equipment_params.get("elec_price_model", "constant"),
            )
        ).strip().lower()
        self.time_of_use_periods = dict(
            self.experiment_params.get("time_of_use_periods", {}) or {}
        )
        self.time_of_use_prices = dict(
            self.experiment_params.get("time_of_use_prices", {}) or {}
        )
        elec_norm_cfg = dict(
            self.experiment_params.get("elec_price_normalization", {}) or {}
        )
        self.elec_price_norm_min = float(
            elec_norm_cfg.get("min", min(self.time_of_use_prices.values(), default=self.constant_electricity_price))
        )
        self.elec_price_norm_max = float(
            elec_norm_cfg.get("max", max(self.time_of_use_prices.values(), default=self.constant_electricity_price))
        )
        self.tou_tariff_scenario = str(
            self.experiment_params.get("tou_tariff_scenario", "") or ""
        )
        self.include_electricity_price_observation = bool(
            self.rl_params.get("include_electricity_price_observation", False)
        )
        self.c_optical_eff = float(cp.get("c_optical_eff", 4.6))

        pid_anchor_light = self.rl_params.get("pid_anchor_light", None)
        if pid_anchor_light is not None:
            anchor_arr = np.asarray(pid_anchor_light, dtype=float).ravel()
            if anchor_arr.size != 2:
                raise ValueError("rl_params.pid_anchor_light must contain exactly two values: I1 I2.")
            self.I_target_seedling = float(anchor_arr[0])
            self.I_target_transplant = float(anchor_arr[1])
        else:
            self.I_target_seedling = float(cp.get("I_target_seedling", 200.0))
            self.I_target_transplant = float(cp.get("I_target_transplant", 300.0))

        ep = self.equipment_params
        self.c_COP = float(ep.get("c_COP", 3.0))
        self.c_dehum_cap = float(ep.get("c_dehum_cap", 2.083e-5))
        self.c_dehum_eev = float(ep.get("c_dehum_eev", 3.0))
        self.dehum_latent_recovery_fraction = float(
            ep.get("dehum_latent_recovery_fraction", 1.0)
        )
        self.dehum_electric_heat_fraction = float(
            ep.get("dehum_electric_heat_fraction", 1.0)
        )
        self.c_vent_fan_cap = float(ep.get("c_vent_fan_cap", 6.36e-5))
        self.co2_supply_max = float(ep.get("co2_supply_max", 1.0e-6))
        self.fan_eff = float(ep.get("fan_eff", 7.07))
        self.I1_min = float(ep.get("I1_min", ep.get("I_min", 0.0)))
        self.I1_max = float(ep.get("I1_max", ep.get("I_max", 400.0)))
        self.I2_min = float(ep.get("I2_min", ep.get("I_min", 0.0)))
        self.I2_max = float(ep.get("I2_max", ep.get("I_max", 400.0)))
        self.I_max = max(self.I1_max, self.I2_max)
        self.lighting_constraint_mode = self._normalise_lighting_constraint_mode(
            ep.get("lighting_constraint_mode", ep.get("light_constraint_mode", "hardware_only"))
        )
        self.static_agronomic_I1_max = self._coerce_optional_positive_float(
            ep.get("I1_agronomic_max", ep.get("static_agronomic_I1_max"))
        )
        self.static_agronomic_I2_max = self._coerce_optional_positive_float(
            ep.get("I2_agronomic_max", ep.get("static_agronomic_I2_max"))
        )
        self.dli_max_seedling_mol_m2_d = self._coerce_optional_positive_float(
            ep.get(
                "dli_max_seedling_mol_m2_d",
                ep.get("dli_aware_seedling_max_mol_m2_d", 14.4),
            )
        )
        self.dli_max_transplant_mol_m2_d = self._coerce_optional_positive_float(
            ep.get(
                "dli_max_transplant_mol_m2_d",
                ep.get("dli_aware_transplant_max_mol_m2_d", 17.28),
            )
        )
        self._light_bound_info: Dict[str, Any] = {
            "mode": str(self.lighting_constraint_mode),
            "photoperiod_h": float(self.schedule.get("PP", 16.0)),
            "hardware_I1_max": float(self.I1_max),
            "hardware_I2_max": float(self.I2_max),
            "effective_I1_max": float(self.I1_max),
            "effective_I2_max": float(self.I2_max),
            "dli_seedling_mol_m2_d": (
                None
                if self.dli_max_seedling_mol_m2_d is None
                else float(self.dli_max_seedling_mol_m2_d)
            ),
            "dli_transplant_mol_m2_d": (
                None
                if self.dli_max_transplant_mol_m2_d is None
                else float(self.dli_max_transplant_mol_m2_d)
            ),
        }
        self.c_led_ppe, self.c_optical_eff, self.c_led_eff = resolve_led_params(cp)
        self._V_vent_fixed = float(
            cp.get("V_vent_fixed", cp.get("vent_leak_rate", 6.36e-5))
        )
        self.container_params["c_dehum_eev"] = self.c_dehum_eev
        self.container_params[
            "dehum_latent_recovery_fraction"
        ] = self.dehum_latent_recovery_fraction
        self.container_params[
            "dehum_electric_heat_fraction"
        ] = self.dehum_electric_heat_fraction

        self.temp_range = ((18.0, 20.0), (22.0, 25.0))
        pid_cfg = self.controller_params.get("pid_controller", {})
        self.vpd_range = (
            (
                float(pid_cfg.get("vpd_night_lo_kpa", 0.25)),
                float(pid_cfg.get("vpd_night_hi_kpa", 0.60)),
            ),
            (
                float(pid_cfg.get("vpd_day_lo_kpa", 0.55)),
                float(pid_cfg.get("vpd_day_hi_kpa", 0.95)),
            ),
        )
        self.humidity_range = (
            relative_humidity_to_absolute(self.temp_range[0][0], 0.7, cp),
            relative_humidity_to_absolute(self.temp_range[1][1], 0.8, cp),
        )
        self.carbon_range = (
            co2_ppm_to_density(800.0),
            co2_ppm_to_density(1200.0),
        )
        self.photo_period = (16.0, 8.0)
        self._configure_light_control_from_rl_params(reset_state=True)

        self.uscale = np.array(
            [
                self.I_max / 2.0,
                self.co2_supply_max / 2.0,
                self.c_dehum_cap / 2.0,
                212.0,
                self.c_vent_fan_cap / 2.0,
            ],
            dtype=np.float32,
        )
        self.ushift = np.array([1.0, 1.0, 1.0, 0.0, 1.0], dtype=np.float32)
        self.xscale = np.array(
            [
                1.0,
                1.0,
                co2_ppm_to_density(3000.0),
                40.0,
                relative_humidity_to_absolute(40.0, 0.95, cp),
                40.0,
                40.0,
                1.0,
                40.0,
                co2_ppm_to_density(3000.0),
                relative_humidity_to_absolute(40.0, 0.98, cp),
                float(24 * 3600 - int(self.dt)),
            ],
            dtype=np.float64,
        )

        self._sync_schedule_geometry_and_diagnostics()

        rng = np.random.default_rng(self.config.get("seed", 42))
        self.batch_manager = BatchManager(
            self.schedule,
            self.container_params,
            self.crop_params,
            rng,
            self.config.get("steady_state_params", None),
            self.reward_params,
        )
        self._sync_schedule_reference_metadata()

        self._init_observation_space()
        self._init_action_space()

        self.state = None
        self.obs_state = None
        self.external = None
        self.elec_price = self.constant_electricity_price
        self._last_applied_elec_price = self.elec_price
        self.time_step = 0
        self.total_steps = 0
        self.tvp = None

        self.hours_continuous_light = 0.0
        self.hours_continuous_dark = 0.0
        self._was_light_on = True
        self.daily_light_hours = 0.0
        self.daily_DLI_dense = 0.0
        self.daily_DLI_finishing = 0.0
        self.daily_DLI = 0.0
        self._initial_hour_of_day = 0.0
        self._episode_hours_elapsed = 0.0
        self.hour_of_day = 0
        self.day_of_period = 0

        self.prev_action_4d = None
        self._last_applied_action_phys = None
        self._last_daily_settlement: Dict[str, Any] = {}
        self._last_settled_day_index = -1
        self._last_constraint_info: Dict[str, Any] = {}
        self._last_env_step_diagnostics: Dict[str, Any] = {}
        self._env_step_clip_counts: Dict[str, int] = {}

        self.episode_reward = 0.0
        self.total_cost = 0.0
        self.total_harvest_mass_g = 0.0
        self.total_E = 0.0
        self.total_P = 0.0
        self.total_R = 0.0
        self._prev_xD = None
        self.total_resets = 0
        self.contextual_reset_count = 0
        self.schedule_visit_counts: Dict[str, int] = {}
        self.schedule_visit_meta: Dict[str, Dict[str, Any]] = {}
        self.completed_episode_count = 0
        self.early_termination_count = 0
        self.failure_termination_count = 0
        self.time_limit_completion_count = 0
        self.episode_completion_ratio_sum = 0.0
        self.termination_reason_counts: Dict[str, int] = {}
        self.safety_override_count = 0
        self.safety_override_reason_counts: Dict[str, int] = {}
        self.episode_safety_override_count = 0
        self.episode_safety_override_reason_counts: Dict[str, int] = {}
        self._action_anchor_controller = None
        self._last_policy_action_norm = np.zeros(5, dtype=np.float32)
        self._last_action_anchor_norm = np.zeros(5, dtype=np.float32)
        self._last_effective_action_norm = np.zeros(5, dtype=np.float32)
        self._last_residual_gate = {
            "overall": 1.0,
            "temperature": 1.0,
            "co2": 1.0,
            "humidity": 1.0,
            "actuator_vector": [1.0] * 5,
        }
        self._thermal_violation_duration_s = 0.0
        self._thermal_violation_peak_c = 0.0
        self._thermal_violation_side = "none"
        self._last_thermal_meltdown_info: Dict[str, Any] = {
            "active": False,
            "reason": "none",
            "temperature_c": 0.0,
            "violation_duration_s": 0.0,
            "violation_peak_c": 0.0,
            "hold_seconds": 0.0,
            "side": "none",
        }
        self.completed_constraint_cost_totals = self._make_constraint_total_dict()
        self.completed_constraint_raw_totals = self._make_constraint_raw_dict()
        self.completed_constraint_counts = self._make_constraint_count_dict()
        self.completed_constraint_active_ratio_sum = 0.0
        self.last_episode_constraint_cost_totals = self._make_constraint_total_dict()
        self.last_episode_constraint_raw_totals = self._make_constraint_raw_dict()
        self.last_episode_constraint_counts = self._make_constraint_count_dict()
        self.last_episode_constraint_active_ratio = 0.0
        self._reset_constraint_tracking()
        self._reset_environment_step_tracking()

        self._ctx_low = np.array(
            [
                float(self.schedule_params.get("t1_min", 10)),
                float(self.schedule_params.get("t2_min", 10)),
                float(self.schedule_params.get("N1_min", 8)),
                float(self.schedule_params.get("rho2_min", 20.0)),
            ],
            dtype=np.float32,
        )
        self._ctx_high = np.array(
            [
                float(self.schedule_params.get("t1_max", 18)),
                float(self.schedule_params.get("t2_max", 18)),
                float(self.schedule_params.get("N1_max", 20)),
                float(self.schedule_params.get("rho2_max", 52.0)),
            ],
            dtype=np.float32,
        )
        self._schedule_sampler = self._build_schedule_sampler()
        self._persistent_reset_options: Dict[str, Any] = {}
        self._context_sampling_cycle_state: Dict[str, Dict[str, Any]] = {}

        dt_hours = max(self.dt / 3600.0, 1e-9)
        self.enforce_action_rate_limits = bool(
            self.config.get("enforce_action_rate_limits", True)
        )
        light_rate_limit = float(self.schedule_params.get("rate_limit_I", np.inf)) * dt_hours
        self._action_rate_limits = np.array(
            [
                light_rate_limit,
                light_rate_limit,
                float(pid_cfg.get("rate_limit_Q", np.inf)) * dt_hours,
                float(pid_cfg.get("rate_limit_CO2", np.inf)) * dt_hours,
                float(pid_cfg.get("rate_limit_dehum", np.inf)) * dt_hours,
            ],
            dtype=np.float32,
        )
        self.episode_length_mode = "schedule_t2"
        self.episode_length_days = float(self.t2)
        self.episode_length = self._resolve_episode_length(schedule=self.schedule)
        self._reset_episode_outcome_tracking()
        self._reset_safety_override_tracking()

    def _sync_schedule_geometry_and_diagnostics(self) -> None:
        """Refresh schedule-dependent geometry and physical diagnostics."""
        self.t1 = int(self.schedule.get("t1", 14))
        self.t2 = int(self.schedule.get("t2", 14))
        self.N1 = int(self.schedule.get("N1", 20))
        self.rho2 = float(self.schedule.get("rho2", 36.0))
        self.PP = int(
            self.schedule.get(
                "PP",
                resolve_fixed_photoperiod_hours(self.schedule_params, self.schedule),
            )
        )

        n_total = int(self.schedule_params.get("N_total", 80))
        a_board = float(self.schedule_params.get("A_board", 1.0))
        self.N2 = n_total - self.N1
        self.A1 = float(self.N1 * a_board)
        self.A2 = float(self.N2 * a_board)
        self.A_total = self.A1 + self.A2
        self.A1_A2 = self.A1 / self.A2 if self.A2 > 0 else 0.0

        self.container_params["_A1"] = self.A1
        self.container_params["_A2"] = self.A2
        self.container_params["_A_total"] = self.A_total
        self.physics_diagnostics = describe_physical_setup(
            self.container_params,
            self.equipment_params,
            A1=self.A1,
            A2=self.A2,
        )

    def _sync_schedule_reference_metadata(self) -> None:
        """Attach reference-growth feasibility metadata to the active schedule."""
        reference_growth = dict(
            getattr(self.batch_manager, "reference_growth_profile", {}) or {}
        )
        min_ratio = float(reference_growth.get("reference_harvest_vs_min_ratio", 0.0))
        target_ratio = float(reference_growth.get("reference_harvest_vs_target_ratio", 0.0))
        if target_ratio >= 1.0:
            reference_class = "target_feasible"
        elif min_ratio >= 1.0:
            reference_class = "min_feasible_only"
        else:
            reference_class = "below_minimum"

        schedule = dict(self.schedule or {})
        schedule.update(reference_growth)
        schedule["reference_feasibility_class"] = str(reference_class)
        schedule["reference_min_feasible"] = bool(min_ratio >= 1.0)
        schedule["reference_target_feasible"] = bool(target_ratio >= 1.0)
        self.schedule = schedule

    @staticmethod
    def _uses_rich_info(info_detail_level: str) -> bool:
        level = str(info_detail_level or "rich").strip().lower()
        return level in {"rich", "full", "debug", "diagnostic", "diagnostics"}

    @staticmethod
    def _deep_update_dict(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        merged = deepcopy(base)
        for key, value in dict(updates or {}).items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = PFALEnvContextual._deep_update_dict(merged[key], value)
            else:
                merged[key] = deepcopy(value)
        return merged

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        applied_elec_price = self._resolve_current_electricity_price()
        self._last_applied_elec_price = float(applied_elec_price)
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        u_raw = self._denormalise_action(action)
        u_applied = self._apply_action_rate_limit(u_raw)
        u_applied = self._apply_light_control_hold(u_applied)
        u_applied, safety_info = self._apply_safety_projection(u_applied)
        self._last_safety_override = dict(safety_info)
        I1, I2, Q_HVAC, u_CO2, m_dehum = u_applied

        T_cur = float(self.state[1])
        C_cur = float(self.state[0])
        xH_cur = float(self.state[2])
        RH_cur = absolute_humidity_to_relative(
            T_cur, xH_cur, self.container_params
        )
        actions_6d = np.array(
            [I1, I2, Q_HVAC, u_CO2, self._V_vent_fixed, m_dehum],
            dtype=np.float32,
        )
        preview_exchange = self.batch_manager.estimate_exchange_rates(
            I1, I2, T_cur, C_cur, RH_cur
        )
        preview_state, _, _preview_diag = simulate_environment_step(
            self.state,
            actions_6d,
            self.external,
            float(preview_exchange.get("total_E_rate", 0.0)),
            float(preview_exchange.get("total_P_rate", 0.0)),
            float(preview_exchange.get("total_resp_rate", 0.0)),
            self.container_params,
            dt=self.dt,
            return_diagnostics=True,
        )

        T_eff = 0.5 * (T_cur + float(preview_state[1]))
        C_eff = 0.5 * (C_cur + float(preview_state[0]))
        xH_eff = 0.5 * (xH_cur + float(preview_state[2]))
        RH_eff = absolute_humidity_to_relative(T_eff, xH_eff, self.container_params)
        RH_eff = float(np.clip(RH_eff, 0.0, 0.995))

        batch_info = self.batch_manager.update(self.dt, I1, I2, T_eff, C_eff, RH_eff)
        self.total_E = float(batch_info.get("total_E_rate", 0.0))
        self.total_P = float(batch_info.get("total_P_rate", 0.0))
        self.total_R = float(batch_info.get("total_resp_rate", 0.0))

        next_state, status, env_step_diag = simulate_environment_step(
            self.state,
            actions_6d,
            self.external,
            self.total_E,
            self.total_P,
            self.total_R,
            self.container_params,
            dt=self.dt,
            return_diagnostics=True,
        )
        env_step_diag["crop_env_coupling_mode"] = "predictor_corrector_midpoint"
        env_step_diag["crop_preview_total_E_rate"] = float(
            preview_exchange.get("total_E_rate", 0.0)
        )
        env_step_diag["crop_preview_total_P_rate"] = float(
            preview_exchange.get("total_P_rate", 0.0)
        )
        env_step_diag["crop_preview_total_resp_rate"] = float(
            preview_exchange.get("total_resp_rate", 0.0)
        )
        env_step_diag["crop_effective_temp_c"] = float(T_eff)
        env_step_diag["crop_effective_co2_density"] = float(C_eff)
        env_step_diag["crop_effective_rh"] = float(RH_eff)
        self._record_environment_step_diagnostics(env_step_diag)
        self.state = next_state
        self._update_photoperiod_trackers(I1, I2)

        harvest_mass_g = float(batch_info.get("harvest_mass", 0.0))
        self.total_harvest_mass_g += harvest_mass_g
        harvest_mass_kg_m2 = harvest_mass_g / 1000.0 / max(self.c_grow_area, 1e-12)
        harvest_fail = bool(batch_info.get("harvest_fail", False))

        reward, reward_info = self._compute_reward_pfal(
            u_applied,
            batch_info=batch_info,
            harvest_mass_kg_m2=harvest_mass_kg_m2,
            harvest_fail=harvest_fail,
            safety_info=safety_info,
            elec_price=applied_elec_price,
        )

        self.prev_action_4d = np.array([I1, I2, Q_HVAC, u_CO2, m_dehum], dtype=np.float32)
        self._last_applied_action_phys = self.prev_action_4d.copy()

        self.time_step += 1
        self.total_steps += 1
        self._episode_hours_elapsed += self.dt / 3600.0
        self._update_time_trackers()
        self._update_electricity_price()

        daily_reward, daily_info = self._settle_daily_reward_if_needed()
        if daily_info:
            if daily_reward != 0.0:
                reward += daily_reward
            reward_info.update(daily_info)
            self._last_daily_settlement = dict(daily_info)
        else:
            self._last_daily_settlement = {}

        thermal_info = self._check_thermal_meltdown()
        thermal_meltdown = bool(thermal_info.get("active", False))
        if thermal_meltdown:
            meltdown_penalty = float(
                self.reward_params.get("thermal_meltdown_penalty", -100.0)
            )
            reward += meltdown_penalty
            reward_info["thermal_meltdown_penalty"] = meltdown_penalty
            reward_info["thermal_meltdown"] = True
            reward_info["thermal_meltdown_reason"] = str(
                thermal_info.get("reason", "thermal_meltdown")
            )
        else:
            reward_info["thermal_meltdown_penalty"] = 0.0
            reward_info["thermal_meltdown"] = False
            reward_info["thermal_meltdown_reason"] = "none"
        reward_info["thermal_violation_duration_s"] = float(
            thermal_info.get("violation_duration_s", 0.0)
        )
        reward_info["thermal_violation_peak_c"] = float(
            thermal_info.get("violation_peak_c", 0.0)
        )

        if status != 0:
            reward -= 10.0
            reward_info["ode_status_penalty"] = -10.0
            reward_info["p_cons"] = reward_info.get("p_cons", 0.0) + 10.0
        else:
            reward_info["ode_status_penalty"] = 0.0

        constraint_info = self._build_constraint_info(reward_info)
        self._last_constraint_info = dict(constraint_info)
        self._accumulate_constraint_info(constraint_info)

        self.episode_reward += reward
        self.total_cost += float(reward_info.get("c_control", 0.0))

        horizon_reached = self.time_step >= self.episode_length
        ode_failure = status != 0
        terminated = bool(ode_failure or thermal_meltdown)
        truncated = bool(horizon_reached and not terminated)
        if thermal_meltdown:
            termination_reason = "thermal_meltdown"
        elif ode_failure:
            termination_reason = "ode_failure"
        elif truncated:
            termination_reason = "time_limit"
        else:
            termination_reason = "running"

        if terminated or truncated:
            self._record_episode_end(
                termination_reason=termination_reason,
                terminated=terminated,
                truncated=truncated,
                status=int(status),
            )

        obs = self._get_observation()
        info = self._get_info(
            batch_info,
            terminated=terminated,
            truncated=truncated,
            termination_reason=termination_reason,
            ode_status=int(status),
            thermal_meltdown=thermal_meltdown,
        )
        if self._uses_rich_info(self.info_detail_level):
            info["_reward_info"] = reward_info
            info["_constraint_info"] = constraint_info
        return obs, float(reward), terminated, truncated, info

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self._update_persistent_reset_options(options or {})
        options = dict(self._persistent_reset_options)
        rng = getattr(self, "np_random", None)
        if rng is None:
            fallback_seed = int(seed if seed is not None else self.config.get("seed", 42))
            rng = np.random.default_rng(fallback_seed)

        if "schedule" in options:
            new_sched = options["schedule"]
        elif options.get("sample_context", False):
            new_sched = self._sample_context_schedule(options, rng=rng)
        else:
            new_sched = None

        if new_sched is not None:
            self.schedule = new_sched
            self._sync_schedule_geometry_and_diagnostics()

            batch_seed = int(rng.integers(0, np.iinfo(np.uint32).max))
            batch_rng = np.random.default_rng(batch_seed)
            self.batch_manager = BatchManager(
                self.schedule,
                self.container_params,
                self.crop_params,
                batch_rng,
                self.config.get("steady_state_params", None),
                self.reward_params,
            )
            self._sync_schedule_reference_metadata()

        self.episode_length = self._resolve_episode_length(
            schedule=self.schedule,
            options=options,
        )
        self._reset_episode_outcome_tracking()
        self._record_schedule_visit(self.schedule, options)

        cp = self.container_params
        self.elec_price = self.constant_electricity_price
        self._last_applied_elec_price = self.elec_price
        if options.get("external") is not None:
            self.external = np.array(options["external"], dtype=np.float64)
        else:
            frac = float(rng.uniform(0.0, 1.0))
            T_summer = float(cp.get("ext_temp_summer", 40.0))
            T_winter = float(cp.get("ext_temp_winter", -12.0))
            RH_summer = float(cp.get("ext_rh_summer", 0.80))
            RH_winter = float(cp.get("ext_rh_winter", 0.70))
            C_summer_ppm = float(cp.get("ext_co2_summer", 400.0))
            C_winter_ppm = float(cp.get("ext_co2_winter", 400.0))

            T_out = T_winter + frac * (T_summer - T_winter)
            xH_out_summer = relative_humidity_to_absolute(T_summer, RH_summer, cp)
            xH_out_winter = relative_humidity_to_absolute(T_winter, RH_winter, cp)
            xH_out = xH_out_winter + frac * (xH_out_summer - xH_out_winter)
            C_out_ppm = C_winter_ppm + frac * (C_summer_ppm - C_winter_ppm)
            self.external = np.array(
                [T_out, xH_out, co2_ppm_to_density(C_out_ppm)],
                dtype=np.float64,
            )

        if "photo_period" in options:
            pp_hours = float(options["photo_period"])
        else:
            pp_hours = float(self.schedule.get("PP", getattr(self, "PP", 16.0)))
        self.photo_period = (pp_hours, max(4.0, 24.0 - pp_hours))
        self._refresh_action_bounds()

        self.tvp = self._tv_data()
        self._i_max = int(self.tvp.shape[0])

        T_init = 23.0
        C_init = co2_ppm_to_density(1000.0)
        RH_init = 0.75
        xH_init = relative_humidity_to_absolute(T_init, RH_init, cp)
        xH_init = float(np.clip(xH_init, 1e-7, 0.5))
        self.state = np.array([C_init, T_init, xH_init], dtype=np.float64)

        default_action_phys = np.array(
            [
                float(cp.get("default_I1", self.I_target_seedling)),
                float(cp.get("default_I2", self.I_target_transplant)),
                0.0,
                0.0,
                self.c_dehum_cap * 0.5,
            ],
            dtype=np.float32,
        )
        self.prev_action_4d = np.clip(
            default_action_phys,
            self._act_low,
            self._act_high,
        ).astype(np.float32)
        self._last_applied_action_phys = None
        self._last_daily_settlement = {}
        self._last_settled_day_index = -1
        self._reset_constraint_tracking()
        self._reset_thermal_meltdown_tracking()

        self.time_step = 0
        hour_mode = str(options.get("hour_of_day_mode", "fixed")).lower()
        if options.get("hour_of_day") is not None:
            initial_hour = float(options["hour_of_day"])
        elif hour_mode in {"random", "uniform"}:
            initial_hour = float(rng.uniform(0.0, 24.0))
        else:
            initial_hour = 0.0
        self._initial_hour_of_day = float(initial_hour % 24.0)
        self._episode_hours_elapsed = 0.0
        self._update_time_trackers()
        self._update_electricity_price()
        self._last_applied_elec_price = self.elec_price
        self._reset_light_control_tracking()

        self.hours_continuous_light = 0.0
        self.hours_continuous_dark = 0.0
        self._was_light_on = False
        self.daily_light_hours = 0.0
        self.daily_DLI_dense = 0.0
        self.daily_DLI_finishing = 0.0
        self.daily_DLI = 0.0
        self.episode_reward = 0.0
        self.total_cost = 0.0
        self.total_harvest_mass_g = 0.0
        self.total_E = 0.0
        self.total_P = 0.0
        self.total_R = 0.0

        self.batch_manager.reset_episode()
        self._prev_xD = self._get_total_dry_mass_kg_m2()
        self._reset_safety_override_tracking()
        self._reset_thermal_meltdown_tracking()
        self._reset_constraint_tracking()
        self._reset_environment_step_tracking()
        if self._action_anchor_controller is not None:
            self._action_anchor_controller.reset()
        self._last_policy_action_norm = np.zeros(5, dtype=np.float32)
        self._last_action_anchor_norm = np.zeros(5, dtype=np.float32)
        self._last_effective_action_norm = np.zeros(5, dtype=np.float32)
        self._last_residual_gate = {
            "overall": 1.0,
            "temperature": 1.0,
            "co2": 1.0,
            "humidity": 1.0,
            "actuator_vector": [1.0] * 5,
        }

        batch_info = {"harvest_mass": 0.0, "harvest_fail": False}
        return self._get_observation(), self._get_info(
            batch_info,
            termination_reason="reset",
        )

    def render(self):
        pass

    def close(self):
        pass

    def seed(self, seed: int):
        np.random.seed(seed)

    def _compute_reward_pfal(
        self,
        u: np.ndarray,
        batch_info: Dict[str, Any],
        harvest_mass_kg_m2: float = 0.0,
        harvest_fail: bool = False,
        safety_info: Optional[Dict[str, Any]] = None,
        elec_price: Optional[float] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        rp = self.reward_params
        reward_scale = float(rp.get("reward_scale", 1.0))
        alpha_growth = float(rp.get("alpha_growth", 0.5))
        weight_growth = float(rp.get("weight_growth", 1.0))
        weight_cost = float(rp.get("weight_cost", 1.0))
        weight_penalty = float(rp.get("weight_penalty", 1.0))
        weight_harvest_target = float(rp.get("weight_harvest_target", 1.0))
        economic_reference = max(
            float(rp.get("economic_reward_reference", 0.10)), 1e-9
        )
        climate_penalty_reference = max(
            float(rp.get("climate_penalty_reference", 5.0)), 1e-9
        )
        harvest_event_penalty_reference = max(
            float(rp.get("harvest_event_penalty_reference", 50.0)), 1e-9
        )
        safety_override_reference = max(
            float(rp.get("safety_override_reference", 0.25)), 1e-9
        )

        power_dict = calculate_total_power(u, self.A1, self.A2, self.equipment_params)
        resolved_elec_price = float(
            self._last_applied_elec_price if elec_price is None else elec_price
        )
        cost_dict = calculate_energy_cost(
            power_dict,
            self.dt,
            resolved_elec_price,
            u_CO2_density=float(u[3]),
            A_total=self.A_total,
            p_CO2=self.c_CO2,
        )
        c_control = float(cost_dict["cost_electric"] + cost_dict["cost_CO2"])

        T = float(self.state[1])
        C_density = float(self.state[0])
        xH = float(self.state[2])
        RH = absolute_humidity_to_relative(T, xH, self.container_params)
        RH_pct = 100.0 * RH
        C_ppm = co2_density_to_ppm(C_density, T)
        vpd_kpa = absolute_humidity_to_vpd(T, xH, self.container_params)

        xD_after = self._get_total_dry_mass_kg_m2()
        if self._prev_xD is None:
            self._prev_xD = xD_after
        xD_before = float(self._prev_xD)
        self._prev_xD = xD_after

        harvest_mass_total_kg = float(batch_info.get("harvest_mass", 0.0)) / 1000.0
        delta_xD_corrected_total_kg = max(
            0.0,
            ((xD_after + harvest_mass_kg_m2) - xD_before) * self.c_grow_area,
        )
        dry_mass_unit_price = self.c_lettuce / max(self.c_d2f, 1e-12)
        growth_value_raw = dry_mass_unit_price * delta_xD_corrected_total_kg
        harvest_value_raw = dry_mass_unit_price * harvest_mass_total_kg
        harvest_target_window_days = max(
            float(rp.get("harvest_target_shaping_window_days", 3.0)),
            1e-9,
        )
        harvest_target_shaping_gain = float(
            rp.get("harvest_target_shaping_gain", 1.0)
        )
        oldest_target_gap_g = max(float(batch_info.get("oldest_target_gap_g", 0.0)), 0.0)
        oldest_plant_count = max(float(batch_info.get("oldest_plant_count", 0.0)), 0.0)
        days_left_oldest = max(float(batch_info.get("days_left_oldest", 0.0)), 0.0)
        harvest_target_urgency = 0.0
        preharvest_target_shortfall_kg = oldest_target_gap_g * oldest_plant_count / 1000.0
        preharvest_target_penalty_raw = 0.0
        if (
            oldest_target_gap_g > 0.0
            and oldest_plant_count > 0.0
            and days_left_oldest <= harvest_target_window_days
        ):
            harvest_target_urgency = 1.0 - (
                days_left_oldest / harvest_target_window_days
            )
            preharvest_target_penalty_raw = (
                harvest_target_shaping_gain
                * dry_mass_unit_price
                * preharvest_target_shortfall_kg
                * harvest_target_urgency
                * (self.dt / 86400.0 / harvest_target_window_days)
            )

        photo_idx = int(self.time_step) % self._i_max
        targets = self.get_climate_targets(photo_idx=photo_idx, T_current=T)
        temp_error = T - float(targets["temp_setpoint_c"])
        co2_error_ppm = C_ppm - float(targets["co2_setpoint_ppm"])
        vpd_error = vpd_kpa - float(targets["vpd_target_kpa"])
        light_on = bool(targets["light_on"])

        if light_on:
            temp_penalty = self._piecewise_penalty(
                T,
                float(rp.get("temp_light_ideal_lo", 22.0)),
                float(rp.get("temp_light_ideal_hi", 25.0)),
                float(rp.get("temp_light_accept_lo", 20.0)),
                float(rp.get("temp_light_accept_hi", 26.0)),
                float(rp.get("temp_mild_penalty", -0.5)),
                float(rp.get("temp_severe_penalty", -2.0)),
            )
            co2_penalty = self._piecewise_penalty(
                C_ppm,
                float(rp.get("co2_light_ideal_lo", 800.0)),
                float(rp.get("co2_light_ideal_hi", 1000.0)),
                float(rp.get("co2_light_accept_lo", 400.0)),
                float(rp.get("co2_light_accept_hi", 1100.0)),
                float(rp.get("co2_mild_penalty", -0.05)),
                float(rp.get("co2_severe_penalty", -0.20)),
            )
            rh_penalty = self._piecewise_penalty(
                RH_pct,
                float(rp.get("rh_light_ideal_lo", 65.0)),
                float(rp.get("rh_light_ideal_hi", 80.0)),
                float(rp.get("rh_light_accept_lo", 50.0)),
                float(rp.get("rh_light_accept_hi", 90.0)),
                float(rp.get("rh_penalty", -0.05)),
                2.0 * float(rp.get("rh_penalty", -0.05)),
            )
        else:
            temp_penalty = self._piecewise_penalty(
                T,
                float(rp.get("temp_night_ideal_lo", 18.0)),
                float(rp.get("temp_night_ideal_hi", 21.0)),
                float(rp.get("temp_night_accept_lo", 16.0)),
                float(rp.get("temp_night_accept_hi", 24.0)),
                float(rp.get("temp_mild_penalty", -0.5)),
                float(rp.get("temp_severe_penalty", -2.0)),
            )
            co2_penalty = self._linear_band_penalty(
                C_ppm,
                0.0,
                float(rp.get("co2_night_max", 1500.0)),
                float(rp.get("co2_mild_penalty", -0.05)),
            )
            rh_penalty = self._piecewise_penalty(
                RH_pct,
                float(rp.get("rh_night_ideal_lo", 70.0)),
                float(rp.get("rh_night_ideal_hi", 85.0)),
                float(rp.get("rh_night_accept_lo", 55.0)),
                float(rp.get("rh_night_accept_hi", 90.0)),
                float(rp.get("rh_penalty", -0.05)),
                2.0 * float(rp.get("rh_penalty", -0.05)),
            )

        harvest_fail_constraint_raw = (
            abs(float(rp.get("harvest_fail_penalty", 50.0))) if harvest_fail else 0.0
        )
        if self.harvest_fail_penalty_mode == "reward":
            harvest_fail_penalty = -float(harvest_fail_constraint_raw)
        else:
            harvest_fail_penalty = 0.0
        safety_override_penalty_raw = 0.0
        if safety_info and bool(safety_info.get("active", False)):
            q_span = max(float(self._act_high[2] - self._act_low[2]), 1e-12)
            q_delta_ratio = abs(float(safety_info.get("q_hvac_delta", 0.0))) / q_span
            safety_override_penalty_raw = (
                -abs(float(rp.get("safety_override_base_penalty", 0.05)))
                - abs(float(rp.get("safety_override_q_hvac_penalty", 0.20))) * q_delta_ratio
            )

        economic_reward_raw = (
            weight_growth * (alpha_growth * growth_value_raw + harvest_value_raw)
            - weight_cost * c_control
            - weight_harvest_target * preharvest_target_penalty_raw
        )
        climate_penalty_raw = temp_penalty + co2_penalty + rh_penalty
        event_penalty_raw = harvest_fail_penalty

        economic_reward_norm = self._normalize_reward_component(
            economic_reward_raw, economic_reference
        )
        climate_penalty_norm = self._normalize_reward_component(
            climate_penalty_raw, climate_penalty_reference
        )
        event_penalty_norm = self._normalize_reward_component(
            event_penalty_raw, harvest_event_penalty_reference
        )
        safety_override_penalty_norm = self._normalize_reward_component(
            safety_override_penalty_raw, safety_override_reference
        )

        economic_reward_scaled = reward_scale * economic_reward_norm
        climate_penalty_scaled = reward_scale * weight_penalty * climate_penalty_norm
        event_penalty_scaled = reward_scale * weight_penalty * event_penalty_norm
        safety_override_penalty = (
            reward_scale * weight_penalty * safety_override_penalty_norm
        )
        reward = (
            economic_reward_scaled
            + climate_penalty_scaled
            + event_penalty_scaled
            + safety_override_penalty
        )

        p_temp_scaled = reward_scale * weight_penalty * self._normalize_reward_component(
            temp_penalty, climate_penalty_reference
        )
        p_co2_scaled = reward_scale * weight_penalty * self._normalize_reward_component(
            co2_penalty, climate_penalty_reference
        )
        p_hum_scaled = reward_scale * weight_penalty * self._normalize_reward_component(
            rh_penalty, climate_penalty_reference
        )
        p_target_scaled = reward_scale * self._normalize_reward_component(
            -weight_harvest_target * preharvest_target_penalty_raw,
            economic_reference,
        )
        harvest_bonus_scaled = reward_scale * self._normalize_reward_component(
            weight_growth * harvest_value_raw,
            economic_reference,
        )

        return float(reward), {
            "reward_scale": reward_scale,
            "economic_reward_reference": economic_reference,
            "climate_penalty_reference": climate_penalty_reference,
            "harvest_event_penalty_reference": harvest_event_penalty_reference,
            "safety_override_reference": safety_override_reference,
            "c_control": c_control,
            "elec_price": float(resolved_elec_price),
            "c_CO2": float(cost_dict["cost_CO2"]),
            "c_light": float(cost_dict["cost_led"]),
            "c_dehum": float(cost_dict["cost_dehum"]),
            "c_HVAC": float(cost_dict["cost_hvac"]),
            "E_led_kWh": float(cost_dict["E_led1_kWh"] + cost_dict["E_led2_kWh"]),
            "E_heating_kWh": float(cost_dict["E_heating_kWh"]),
            "E_cooling_kWh": float(cost_dict["E_cooling_kWh"]),
            "E_dehum_kWh": float(cost_dict["E_dehum_kWh"]),
            "p_temp": float(temp_penalty),
            "p_CO2": float(co2_penalty),
            "p_hum": float(rh_penalty),
            "p_temp_scaled": float(p_temp_scaled),
            "p_CO2_scaled": float(p_co2_scaled),
            "p_hum_scaled": float(p_hum_scaled),
            "p_cons": float(abs(harvest_fail_penalty)),
            "p_target": float(p_target_scaled),
            "p_safety": float(safety_override_penalty),
            "temp_error_c": float(temp_error),
            "co2_error_ppm": float(co2_error_ppm),
            "vpd_error_kpa": float(vpd_error),
            "p_growth": float(delta_xD_corrected_total_kg),
            "economic_reward_raw": float(economic_reward_raw),
            "economic_reward_norm": float(economic_reward_norm),
            "economic_reward_scaled": float(economic_reward_scaled),
            "climate_penalty_raw": float(climate_penalty_raw),
            "climate_penalty_norm": float(climate_penalty_norm),
            "climate_penalty_scaled": float(climate_penalty_scaled),
            "event_penalty_raw": float(event_penalty_raw),
            "event_penalty_norm": float(event_penalty_norm),
            "event_penalty_scaled": float(event_penalty_scaled),
            "growth_value_raw": float(growth_value_raw),
            "harvest_bonus": float(harvest_bonus_scaled),
            "harvest_value_raw": float(harvest_value_raw),
            "preharvest_target_penalty_raw": float(preharvest_target_penalty_raw),
            "preharvest_target_shortfall_kg": float(preharvest_target_shortfall_kg),
            "harvest_target_urgency": float(harvest_target_urgency),
            "harvest_target_window_days": float(harvest_target_window_days),
            "oldest_target_ratio": float(batch_info.get("oldest_target_ratio", 0.0)),
            "oldest_target_gap_g": float(oldest_target_gap_g),
            "oldest_target_surplus_g": float(batch_info.get("oldest_target_surplus_g", 0.0)),
            "oldest_plant_count": float(oldest_plant_count),
            "harvest_fail_penalty": float(harvest_fail_penalty),
            "harvest_fail_constraint_raw": float(harvest_fail_constraint_raw),
            "harvest_fail_penalty_mode": str(self.harvest_fail_penalty_mode),
            "safety_override_penalty": float(safety_override_penalty_raw),
            "safety_override_penalty_scaled": float(safety_override_penalty),
            "harvest_mass_kg_m2": float(harvest_mass_kg_m2),
            "harvest_mass_kg_total": float(harvest_mass_total_kg),
            "harvest_target_mass_g": float(batch_info.get("harvest_target_mass_g", 0.0)),
            "harvest_target_shortfall_g": float(batch_info.get("harvest_target_shortfall_g", 0.0)),
            "harvest_target_surplus_g": float(batch_info.get("harvest_target_surplus_g", 0.0)),
            "harvest_mean_target_ratio": float(batch_info.get("harvest_mean_target_ratio", 0.0)),
            "harvest_fail_n_batches_this_step": int(batch_info.get("harvest_fail_n_batches_this_step", 0)),
            "xD_after": float(xD_after),
            "xD_before": float(xD_before),
            "dry_mass_unit_price": float(dry_mass_unit_price),
            "light_on": light_on,
            "VPD_kPa": float(vpd_kpa),
            "I1": float(u[0]),
            "I2": float(u[1]),
            "Q_HVAC": float(u[2]),
            "u_CO2": float(u[3]),
            "m_dehum": float(u[4]),
            "reward_soft_total": float(reward),
        }

    @staticmethod
    def _normalize_reward_component(value: float, reference: float) -> float:
        return float(value) / max(abs(float(reference)), 1e-12)

    def _get_total_dry_mass_kg_m2(self) -> float:
        lumped = self.batch_manager._extract_lumped_features()
        dm_seedling_kg = float(lumped.get("density_seedling", 0.0)) / 1000.0 * self.A1
        dm_transplant_kg = (
            float(lumped.get("density_transplant", 0.0)) / 1000.0 * self.A2
        )
        return (dm_seedling_kg + dm_transplant_kg) / max(self.c_grow_area, 1e-12)

    def _init_observation_space(self):
        self._obs_limits = {
            "lai": 6.0,
            "density": 300.0,
            "delta_density": 50.0,
            "W_old": 15.0,
            "target_ratio": 1.0,
            "days_left": 40.0,
            "cycle_days_left": 56.0,
            "harvest_count": float(max(self.batch_manager.k1 + self.batch_manager.k2, 1)),
            "temp_err": 12.0,
            "co2_err": 1200.0,
            "vpd_err": 1.5,
            "vpd": 2.0,
            "dli_progress_err": max(
                float(self._get_daily_zone_dli_caps().get("weighted", 20.0)),
                1.0,
            ),
        }
        self._act_low, self._act_high = self._get_action_physical_bounds()
        obs_dim = 31 if self.include_electricity_price_observation else 30
        self.observation_space = spaces.Box(
            low=-np.ones(obs_dim, dtype=np.float32),
            high=np.ones(obs_dim, dtype=np.float32),
            dtype=np.float32,
        )

    def _init_action_space(self):
        self.action_space = spaces.Box(
            low=np.array([-1.0] * 5, dtype=np.float32),
            high=np.array([1.0] * 5, dtype=np.float32),
            dtype=np.float32,
        )

    def _get_action_physical_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        action_low, action_high = get_action_bounds(self.equipment_params)
        i1_low, i1_high, i2_low, i2_high, light_bound_info = (
            self._resolve_effective_light_bounds()
        )
        action_low[0] = float(i1_low)
        action_high[0] = float(i1_high)
        action_low[1] = float(i2_low)
        action_high[1] = float(i2_high)
        self._light_bound_info = dict(light_bound_info)
        return action_low, action_high

    @staticmethod
    def _coerce_optional_positive_float(value: Any) -> Optional[float]:
        try:
            resolved = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(resolved) or resolved <= 0.0:
            return None
        return float(resolved)

    @staticmethod
    def _normalise_lighting_constraint_mode(value: Any) -> str:
        aliases = {
            "hardware": "hardware_only",
            "hardware_only": "hardware_only",
            "wide": "hardware_only",
            "static": "static_agronomic",
            "agronomic": "static_agronomic",
            "static_agronomic": "static_agronomic",
            "dli": "dli_aware_agronomic",
            "dli_aware": "dli_aware_agronomic",
            "dli_aware_agronomic": "dli_aware_agronomic",
        }
        mode = str(value or "hardware_only").strip().lower()
        return aliases.get(mode, "hardware_only")

    @staticmethod
    def _normalise_daily_dli_penalty_mode(value: Any) -> str:
        aliases = {
            "off": "disabled",
            "disable": "disabled",
            "disabled": "disabled",
            "none": "disabled",
            "legacy": "legacy_average_band",
            "band": "legacy_average_band",
            "legacy_average": "legacy_average_band",
            "legacy_average_band": "legacy_average_band",
            "zone_upper": "zone_upper_guard",
            "zone_cap": "zone_upper_guard",
            "zone_upper_guard": "zone_upper_guard",
        }
        mode = str(value or "disabled").strip().lower()
        return aliases.get(mode, "disabled")

    @staticmethod
    def _normalise_daily_photoperiod_penalty_mode(value: Any) -> str:
        aliases = {
            "off": "disabled",
            "disable": "disabled",
            "disabled": "disabled",
            "none": "disabled",
            "legacy": "band",
            "band": "band",
            "guard": "fixed_schedule_guard",
            "fixed": "fixed_schedule_guard",
            "fixed_schedule_guard": "fixed_schedule_guard",
        }
        mode = str(value or "fixed_schedule_guard").strip().lower()
        return aliases.get(mode, "fixed_schedule_guard")

    @staticmethod
    def _normalise_harvest_fail_penalty_mode(value: Any) -> str:
        aliases = {
            "reward": "reward",
            "in_reward": "reward",
            "constraint": "constraint_only",
            "constraint_only": "constraint_only",
            "selection_only": "constraint_only",
            "diag": "constraint_only",
            "diagnostic": "constraint_only",
            "disabled": "disabled",
            "off": "disabled",
            "none": "disabled",
        }
        mode = str(value or "constraint_only").strip().lower()
        return aliases.get(mode, "constraint_only")

    def _resolve_current_photoperiod_hours(self) -> float:
        if getattr(self, "photo_period", None) is not None:
            try:
                return max(float(self.photo_period[0]), 1e-9)
            except Exception:
                pass
        if self.config.get("photo_period_override") is not None:
            return max(float(self.config.get("photo_period_override", 16.0)), 1e-9)
        return max(float(self.schedule.get("PP", getattr(self, "PP", 16.0))), 1e-9)

    def _compute_area_weighted_daily_dli(
        self,
        dense_dli: float,
        finishing_dli: float,
    ) -> float:
        area_total = max(float(self.A1) + float(self.A2), 1e-9)
        return float(
            (float(dense_dli) * float(self.A1) + float(finishing_dli) * float(self.A2))
            / area_total
        )

    def _get_daily_zone_dli_caps(self) -> Dict[str, float]:
        pp_hours = self._resolve_current_photoperiod_hours()
        _, i1_high, _, i2_high, _ = self._resolve_effective_light_bounds()
        dense_cap = max(float(i1_high), 0.0) * 0.0036 * pp_hours
        finishing_cap = max(float(i2_high), 0.0) * 0.0036 * pp_hours
        weighted_cap = self._compute_area_weighted_daily_dli(dense_cap, finishing_cap)
        return {
            "dense": float(dense_cap),
            "finishing": float(finishing_cap),
            "weighted": float(weighted_cap),
        }

    def _get_daily_light_progress_metrics(self) -> Dict[str, float]:
        planned_light_hours = self._get_planned_light_hours_elapsed()
        pp_hours = max(float(self.photo_period[0]), 1e-9)
        dli_caps = self._get_daily_zone_dli_caps()
        weighted_cap = float(dli_caps["weighted"])
        weighted_cap_scale = max(weighted_cap, 1.0)
        realized_weighted = self._compute_area_weighted_daily_dli(
            self.daily_DLI_dense,
            self.daily_DLI_finishing,
        )
        expected_dli_so_far = weighted_cap * planned_light_hours / pp_hours
        dli_progress_error = realized_weighted - expected_dli_so_far
        return {
            "planned_light_hours": float(planned_light_hours),
            "weighted_cap": float(weighted_cap),
            "expected_dli_so_far": float(expected_dli_so_far),
            "dli_progress_error": float(dli_progress_error),
            "dli_progress": float(
                self._scale_zero_to_one(realized_weighted, weighted_cap_scale)
            ),
            "realized_weighted": float(realized_weighted),
            "dense_cap": float(dli_caps["dense"]),
            "finishing_cap": float(dli_caps["finishing"]),
        }

    def _resolve_effective_light_bounds(
        self,
    ) -> Tuple[float, float, float, float, Dict[str, Any]]:
        i1_low = float(self.I1_min)
        i1_high = float(self.I1_max)
        i2_low = float(self.I2_min)
        i2_high = float(self.I2_max)
        pp_hours = self._resolve_current_photoperiod_hours()
        mode = str(self.lighting_constraint_mode)

        if mode == "static_agronomic":
            if self.static_agronomic_I1_max is not None:
                i1_high = min(i1_high, float(self.static_agronomic_I1_max))
            if self.static_agronomic_I2_max is not None:
                i2_high = min(i2_high, float(self.static_agronomic_I2_max))
        elif mode == "dli_aware_agronomic":
            if self.dli_max_seedling_mol_m2_d is not None:
                i1_high = min(
                    i1_high,
                    float(self.dli_max_seedling_mol_m2_d) / (0.0036 * pp_hours),
                )
            if self.dli_max_transplant_mol_m2_d is not None:
                i2_high = min(
                    i2_high,
                    float(self.dli_max_transplant_mol_m2_d) / (0.0036 * pp_hours),
                )

        i1_high = max(i1_low, float(i1_high))
        i2_high = max(i2_low, float(i2_high))
        metadata = {
            "mode": mode,
            "photoperiod_h": float(pp_hours),
            "hardware_I1_max": float(self.I1_max),
            "hardware_I2_max": float(self.I2_max),
            "effective_I1_max": float(i1_high),
            "effective_I2_max": float(i2_high),
            "dli_seedling_mol_m2_d": (
                None
                if self.dli_max_seedling_mol_m2_d is None
                else float(self.dli_max_seedling_mol_m2_d)
            ),
            "dli_transplant_mol_m2_d": (
                None
                if self.dli_max_transplant_mol_m2_d is None
                else float(self.dli_max_transplant_mol_m2_d)
            ),
            "static_I1_max": (
                None
                if self.static_agronomic_I1_max is None
                else float(self.static_agronomic_I1_max)
            ),
            "static_I2_max": (
                None
                if self.static_agronomic_I2_max is None
                else float(self.static_agronomic_I2_max)
            ),
        }
        return i1_low, i1_high, i2_low, i2_high, metadata

    def _refresh_action_bounds(self) -> None:
        self._act_low, self._act_high = self._get_action_physical_bounds()
        if self._action_anchor_controller is not None:
            try:
                self._action_anchor_controller.bind_env(self)
            except Exception:
                pass

    @staticmethod
    def _resolve_residual_action_scale(scale_value: Any) -> np.ndarray:
        if np.isscalar(scale_value):
            return np.full(5, float(scale_value), dtype=np.float32)
        arr = np.asarray(scale_value, dtype=np.float32).ravel()
        if arr.size == 1:
            return np.full(5, float(arr[0]), dtype=np.float32)
        if arr.size != 5:
            raise ValueError(
                "residual_action_scale must be a scalar or length-5 sequence."
            )
        return arr.astype(np.float32)

    def _ensure_action_anchor_controller(self):
        if self.action_semantics not in {
            "residual_pid",
            "pid_residual",
            "residual_pid_gated",
            "pid_residual_gated",
        }:
            return None
        if self._action_anchor_controller is None:
            from controllers.pfal_conventional_controller import PFALConventionalController

            self._action_anchor_controller = PFALConventionalController(self)
        return self._action_anchor_controller

    @staticmethod
    def _band_membership_gate(
        value: float,
        ideal_lo: float,
        ideal_hi: float,
        accept_lo: float,
        accept_hi: float,
    ) -> float:
        accept_lo_v = float(min(accept_lo, accept_hi))
        accept_hi_v = float(max(accept_lo, accept_hi))
        ideal_lo_v = float(min(ideal_lo, ideal_hi))
        ideal_hi_v = float(max(ideal_lo, ideal_hi))
        value_v = float(value)
        if value_v <= accept_lo_v or value_v >= accept_hi_v:
            return 0.0
        if ideal_lo_v <= value_v <= ideal_hi_v:
            return 1.0
        if value_v < ideal_lo_v:
            span = max(ideal_lo_v - accept_lo_v, 1e-9)
            return float(np.clip((value_v - accept_lo_v) / span, 0.0, 1.0))
        span = max(accept_hi_v - ideal_hi_v, 1e-9)
        return float(np.clip((accept_hi_v - value_v) / span, 0.0, 1.0))

    def _compute_residual_gate(self) -> Dict[str, Any]:
        gate_default = {
            "overall": 1.0,
            "temperature": 1.0,
            "co2": 1.0,
            "humidity": 1.0,
            "actuator_vector": np.ones(5, dtype=np.float32),
        }
        if self.state is None:
            return gate_default

        T = float(self.state[1])
        xH = float(self.state[2])
        C_ppm = float(co2_density_to_ppm(float(self.state[0]), T))
        RH_pct = float(
            absolute_humidity_to_relative(T, xH, self.container_params) * 100.0
        )
        photo_idx = int(self.time_step) % self._i_max if getattr(self, "_i_max", 0) > 0 else 0
        targets = self.get_climate_targets(photo_idx=photo_idx, T_current=T)
        light_on = bool(targets.get("light_on", True))
        rp = self.reward_params

        if light_on:
            temp_gate = self._band_membership_gate(
                T,
                float(rp.get("temp_light_ideal_lo", 22.0)),
                float(rp.get("temp_light_ideal_hi", 25.0)),
                float(rp.get("temp_light_accept_lo", 20.0)),
                float(rp.get("temp_light_accept_hi", 26.0)),
            )
            co2_gate = self._band_membership_gate(
                C_ppm,
                float(rp.get("co2_light_ideal_lo", 800.0)),
                float(rp.get("co2_light_ideal_hi", 1000.0)),
                float(rp.get("co2_light_accept_lo", 400.0)),
                float(rp.get("co2_light_accept_hi", 1100.0)),
            )
            rh_gate = self._band_membership_gate(
                RH_pct,
                float(rp.get("rh_light_ideal_lo", 65.0)),
                float(rp.get("rh_light_ideal_hi", 80.0)),
                float(rp.get("rh_light_accept_lo", 50.0)),
                float(rp.get("rh_light_accept_hi", 90.0)),
            )
        else:
            temp_gate = self._band_membership_gate(
                T,
                float(rp.get("temp_night_ideal_lo", 18.0)),
                float(rp.get("temp_night_ideal_hi", 21.0)),
                float(rp.get("temp_night_accept_lo", 16.0)),
                float(rp.get("temp_night_accept_hi", 24.0)),
            )
            co2_gate = 1.0
            rh_gate = self._band_membership_gate(
                RH_pct,
                float(rp.get("rh_night_ideal_lo", 70.0)),
                float(rp.get("rh_night_ideal_hi", 85.0)),
                float(rp.get("rh_night_accept_lo", 55.0)),
                float(rp.get("rh_night_accept_hi", 90.0)),
            )

        gate_floor = float(self.residual_gate_min)
        temp_gate = max(temp_gate, gate_floor)
        co2_gate = max(co2_gate, gate_floor)
        rh_gate = max(rh_gate, gate_floor)
        climate_gate = min(temp_gate, rh_gate)
        actuator_gate = np.array(
            [climate_gate, climate_gate, climate_gate, co2_gate, rh_gate],
            dtype=np.float32,
        )
        return {
            "overall": float(np.min(actuator_gate)),
            "temperature": float(temp_gate),
            "co2": float(co2_gate),
            "humidity": float(rh_gate),
            "actuator_vector": actuator_gate,
        }

    def _get_action_anchor_norm(self) -> np.ndarray:
        controller = self._ensure_action_anchor_controller()
        if controller is None:
            return np.zeros(5, dtype=np.float32)
        obs = self._get_observation()
        anchor = np.asarray(controller.predict(obs), dtype=np.float32).ravel()[:5]
        if not np.all(np.isfinite(anchor)):
            anchor = np.zeros(5, dtype=np.float32)
        return np.clip(anchor, -1.0, 1.0).astype(np.float32)

    def _denormalise_action(self, action_norm: np.ndarray) -> np.ndarray:
        if not np.all(np.isfinite(action_norm)):
            action_norm = np.zeros(5, dtype=np.float32)
        action_norm = np.asarray(action_norm, dtype=np.float32).ravel()[:5]
        self._last_policy_action_norm = action_norm.copy()

        if self.action_semantics in {
            "residual_pid",
            "pid_residual",
            "residual_pid_gated",
            "pid_residual_gated",
        }:
            anchor_norm = self._get_action_anchor_norm()
            gate_info = self._compute_residual_gate()
            residual_gate = np.asarray(gate_info["actuator_vector"], dtype=np.float32)
            if self.action_semantics in {"residual_pid", "pid_residual"}:
                residual_gate = np.ones(5, dtype=np.float32)
                gate_info = {
                    "overall": 1.0,
                    "temperature": 1.0,
                    "co2": 1.0,
                    "humidity": 1.0,
                    "actuator_vector": residual_gate,
                }
            effective_norm = np.clip(
                anchor_norm + residual_gate * self._residual_action_scale * action_norm,
                -1.0,
                1.0,
            ).astype(np.float32)
        else:
            anchor_norm = np.zeros(5, dtype=np.float32)
            gate_info = {
                "overall": 1.0,
                "temperature": 1.0,
                "co2": 1.0,
                "humidity": 1.0,
                "actuator_vector": np.ones(5, dtype=np.float32),
            }
            effective_norm = np.clip(action_norm, -1.0, 1.0).astype(np.float32)

        self._last_action_anchor_norm = anchor_norm.copy()
        self._last_effective_action_norm = effective_norm.copy()
        self._last_residual_gate = {
            "overall": float(gate_info["overall"]),
            "temperature": float(gate_info["temperature"]),
            "co2": float(gate_info["co2"]),
            "humidity": float(gate_info["humidity"]),
            "actuator_vector": np.asarray(
                gate_info["actuator_vector"], dtype=np.float32
            ).astype(float).tolist(),
        }
        al, ah = self._get_action_physical_bounds()
        raw = (effective_norm + 1.0) * 0.5
        physical = al + raw * (ah - al)

        if self.tvp is not None and getattr(self, "_i_max", 0) > 0:
            photo_idx = int(self.time_step) % self._i_max
            is_light_on = bool(self.tvp[photo_idx, 2])
        else:
            is_light_on = True

        if not is_light_on:
            physical[0] = 0.0
            physical[1] = 0.0
            physical[3] = 0.0
        return np.asarray(physical, dtype=np.float32)

    def _apply_action_rate_limit(self, action_phys: np.ndarray) -> np.ndarray:
        action_phys = np.asarray(action_phys, dtype=np.float32).copy()
        if not self.enforce_action_rate_limits or self._last_applied_action_phys is None:
            return action_phys

        out = action_phys.copy()
        prev = self._last_applied_action_phys
        mode = str(getattr(self, "light_control_mode", "step"))
        for idx, limit in enumerate(self._action_rate_limits):
            limit = float(limit)
            if not np.isfinite(limit):
                continue
            if idx in (0, 1) and mode != "step":
                continue
            delta = float(np.clip(out[idx] - prev[idx], -limit, limit))
            out[idx] = prev[idx] + delta

        if self.tvp is not None and getattr(self, "_i_max", 0) > 0:
            photo_idx = int(self.time_step) % self._i_max
            if not bool(self.tvp[photo_idx, 2]):
                out[0] = 0.0
                out[1] = 0.0
                out[3] = 0.0

        al, ah = self._get_action_physical_bounds()
        return np.clip(out, al, ah).astype(np.float32)

    @staticmethod
    def _normalise_light_control_mode(value: Any) -> str:
        aliases = {
            "daily": "daily_hold",
            "daily_once": "daily_hold",
            "per_day": "daily_hold",
            "segmented": "segmented_hold",
            "segment": "segmented_hold",
            "segments": "segmented_hold",
            "per_segment": "segmented_hold",
            "per_step": "step",
            "legacy": "step",
        }
        mode = str(value or "step").strip().lower()
        mode = aliases.get(mode, mode)
        if mode not in {"step", "daily_hold", "segmented_hold"}:
            return "step"
        return mode

    def _configure_light_control_from_rl_params(self, reset_state: bool = True) -> None:
        self.light_control_mode = self._normalise_light_control_mode(
            self.rl_params.get("light_control_mode", "step")
        )
        try:
            segments = int(self.rl_params.get("light_segments_per_photoperiod", 3))
        except Exception:
            segments = 3
        self.light_segments_per_photoperiod = max(1, segments)
        if reset_state:
            self._reset_light_control_tracking()

    def _reset_light_control_tracking(self) -> None:
        self._held_light_setpoints = np.zeros(2, dtype=np.float32)
        self._light_hold_initialized = False
        self._last_light_update_day_index = -1
        self._last_light_update_segment_key = None
        self._last_light_control_info = {
            "mode": str(getattr(self, "light_control_mode", "step")),
            "update_allowed": False,
            "light_on": False,
            "segment_index": -1,
            "day_index": 0,
            "hour_of_day": 0.0,
            "held_I1_target": 0.0,
            "held_I2_target": 0.0,
            "initialized": False,
        }

    def _get_current_time_markers(self) -> Tuple[int, float]:
        total_h = float(self._initial_hour_of_day + self._episode_hours_elapsed)
        return int(total_h // 24.0), float(total_h % 24.0)

    def _resolve_current_electricity_price(
        self,
        hour_of_day: Optional[float] = None,
    ) -> float:
        if hour_of_day is None:
            _, hour_of_day = self._get_current_time_markers()
        return float(
            compute_electricity_price(
                hour=float(hour_of_day),
                price_model=self.electricity_price_model,
                time_of_use_periods=self.time_of_use_periods,
                time_of_use_prices=self.time_of_use_prices,
                constant_price=self.constant_electricity_price,
            )
        )

    def _update_electricity_price(
        self,
        hour_of_day: Optional[float] = None,
    ) -> float:
        self.elec_price = self._resolve_current_electricity_price(hour_of_day)
        return float(self.elec_price)

    def _scale_electricity_price_for_observation(self, value: float) -> float:
        lo = float(min(self.elec_price_norm_min, self.elec_price_norm_max))
        hi = float(max(self.elec_price_norm_min, self.elec_price_norm_max))
        if hi - lo <= 1e-9:
            return 0.0
        scaled = 2.0 * (float(value) - lo) / (hi - lo) - 1.0
        return float(np.clip(scaled, -1.0, 1.0))

    def _get_light_segment_index(self, hour_of_day: Optional[float] = None) -> Optional[int]:
        if hour_of_day is None:
            _, hour_of_day = self._get_current_time_markers()
        pp_hours = max(float(self.photo_period[0]), 1e-9)
        dawn = float(self.photo_period[1])
        dusk = dawn + pp_hours
        if hour_of_day < dawn or hour_of_day >= dusk:
            return None
        segments = max(int(self.light_segments_per_photoperiod), 1)
        seg_len = pp_hours / segments
        progress = min(max(hour_of_day - dawn, 0.0), max(pp_hours - 1e-9, 0.0))
        return min(int(progress / max(seg_len, 1e-9)), segments - 1)

    def _apply_light_control_hold(self, action_phys: np.ndarray) -> np.ndarray:
        action_phys = np.asarray(action_phys, dtype=np.float32).copy()
        mode = str(getattr(self, "light_control_mode", "step"))
        day_idx, hour_of_day = self._get_current_time_markers()
        segment_idx = self._get_light_segment_index(hour_of_day)

        if self.tvp is not None and getattr(self, "_i_max", 0) > 0:
            photo_idx = int(self.time_step) % self._i_max
            light_on = bool(self.tvp[photo_idx, 2])
        else:
            light_on = bool(segment_idx is not None)

        update_allowed = bool(light_on)
        if not light_on:
            action_phys[0] = 0.0
            action_phys[1] = 0.0
        elif mode == "daily_hold":
            update_allowed = (
                (not self._light_hold_initialized)
                or int(day_idx) != int(self._last_light_update_day_index)
            )
        elif mode == "segmented_hold":
            current_key = (int(day_idx), int(segment_idx if segment_idx is not None else -1))
            update_allowed = (
                (not self._light_hold_initialized)
                or current_key != self._last_light_update_segment_key
            )
        else:
            update_allowed = bool(light_on)

        if light_on and mode in {"daily_hold", "segmented_hold"}:
            if update_allowed:
                self._held_light_setpoints = np.asarray(action_phys[:2], dtype=np.float32).copy()
                self._light_hold_initialized = True
                self._last_light_update_day_index = int(day_idx)
                self._last_light_update_segment_key = (
                    int(day_idx),
                    int(segment_idx if segment_idx is not None else -1),
                )
            else:
                action_phys[0] = float(self._held_light_setpoints[0])
                action_phys[1] = float(self._held_light_setpoints[1])

        self._last_light_control_info = {
            "mode": mode,
            "update_allowed": bool(update_allowed),
            "light_on": bool(light_on),
            "segment_index": int(segment_idx if segment_idx is not None else -1),
            "day_index": int(day_idx),
            "hour_of_day": float(hour_of_day),
            "held_I1_target": float(self._held_light_setpoints[0]),
            "held_I2_target": float(self._held_light_setpoints[1]),
            "initialized": bool(self._light_hold_initialized),
        }
        return action_phys.astype(np.float32)

    def _reset_safety_override_tracking(self) -> None:
        self.episode_safety_override_count = 0
        self.episode_safety_override_reason_counts = {}
        self._last_safety_override = {
            "active": False,
            "reason": "none",
            "q_hvac_before": 0.0,
            "q_hvac_after": 0.0,
            "q_hvac_delta": 0.0,
        }

    def _reset_thermal_meltdown_tracking(self) -> None:
        self._thermal_violation_duration_s = 0.0
        self._thermal_violation_peak_c = 0.0
        self._thermal_violation_side = "none"
        self._last_thermal_meltdown_info = {
            "active": False,
            "reason": "none",
            "temperature_c": float(self.state[1]) if self.state is not None else 0.0,
            "violation_duration_s": 0.0,
            "violation_peak_c": 0.0,
            "hold_seconds": 0.0,
            "side": "none",
        }

    @staticmethod
    def _make_env_step_clip_count_dict() -> Dict[str, int]:
        return {
            "temperature": 0,
            "co2_density": 0,
            "co2_step_delta": 0,
            "condensation": 0,
            "humidity_floor": 0,
            "humidity_saturation": 0,
            "fallback": 0,
            "safe_default": 0,
        }

    def _reset_environment_step_tracking(self) -> None:
        self._last_env_step_diagnostics = {
            "integration_method": "none",
            "used_euler_fallback": False,
            "used_safe_defaults": False,
            "temperature_clipped": False,
            "co2_density_bounded": False,
            "co2_rate_limited": False,
            "condensation_active": False,
            "humidity_floor_clipped": False,
            "humidity_saturation_clipped": False,
        }
        self._env_step_clip_counts = self._make_env_step_clip_count_dict()

    def _record_environment_step_diagnostics(self, diagnostics: Optional[Dict[str, Any]]) -> None:
        diag = dict(diagnostics or {})
        if not diag:
            return
        self._last_env_step_diagnostics = diag
        if diag.get("temperature_clipped", False):
            self._env_step_clip_counts["temperature"] += 1
        if diag.get("co2_density_bounded", False):
            self._env_step_clip_counts["co2_density"] += 1
        if diag.get("co2_rate_limited", False):
            self._env_step_clip_counts["co2_step_delta"] += 1
        if diag.get("condensation_active", False):
            self._env_step_clip_counts["condensation"] += 1
        if diag.get("humidity_floor_clipped", False):
            self._env_step_clip_counts["humidity_floor"] += 1
        if diag.get("humidity_saturation_clipped", False):
            self._env_step_clip_counts["humidity_saturation"] += 1
        if diag.get("used_euler_fallback", False):
            self._env_step_clip_counts["fallback"] += 1
        if diag.get("used_safe_defaults", False):
            self._env_step_clip_counts["safe_default"] += 1

    def _build_schedule_reference_warnings(self) -> list[str]:
        ref = dict(getattr(self.batch_manager, "reference_growth_profile", {}) or {})
        warnings: list[str] = []
        ref_min_ratio = float(ref.get("reference_harvest_vs_min_ratio", 0.0))
        ref_target_ratio = float(ref.get("reference_harvest_vs_target_ratio", 0.0))
        ref_fw = float(ref.get("reference_harvest_fresh_mass_per_plant_g", 0.0))
        min_fw = float(
            self.reward_params.get("harvest_min_dry_mass_per_plant", 4.44) * self.c_fw
        )
        target_fw = float(
            self.reward_params.get(
                "harvest_target_dry_mass_per_plant",
                self.reward_params.get("harvest_min_dry_mass_per_plant", 4.44),
            ) * self.c_fw
        )
        if ref_min_ratio < 1.0:
            warnings.append(
                f"Nominal reference growth for the current schedule reaches only {ref_fw:.1f} g FW/plant, "
                f"below the configured minimum line {min_fw:.1f} g FW/plant."
            )
        elif ref_target_ratio < 1.0:
            warnings.append(
                f"Nominal reference growth for the current schedule reaches {ref_fw:.1f} g FW/plant, "
                f"above the minimum line but below the configured target {target_fw:.1f} g FW/plant."
            )
        return warnings

    def _apply_safety_projection(
        self,
        action_phys: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        action_phys = np.asarray(action_phys, dtype=np.float32).copy()
        info = {
            "active": False,
            "reason": "none",
            "q_hvac_before": float(action_phys[2]),
            "q_hvac_after": float(action_phys[2]),
            "q_hvac_delta": 0.0,
        }
        if not self.enable_action_safety_projection or self.state is None:
            return action_phys, info

        T = float(self.state[1])
        T_lo = float(self.reward_params.get("thermal_meltdown_temp_lo", 15.0))
        T_hi = float(self.reward_params.get("thermal_meltdown_temp_hi", 32.0))
        guard_band = max(float(self.safety_temp_guard_band_c), 0.0)
        guard_lo = min(T_hi, T_lo + guard_band)
        guard_hi = max(T_lo, T_hi - guard_band)
        targets = self.get_climate_targets(T_current=T)
        temp_sp = float(targets["temp_setpoint_c"])

        q_before = float(action_phys[2])
        q_after = q_before
        q_min = float(self._act_low[2])
        q_max = float(self._act_high[2])
        kp = max(float(self.safety_temp_kp), 1e-6)
        projection_margin = max(float(self.safety_temp_projection_margin_c), 0.0)
        c_cap_q = max(float(self.container_params.get("c_cap_q", 30000.0)), 1e-9)
        dt_s = max(float(self.dt), 1e-9)
        reason = "none"

        low_guard_trigger = T <= guard_lo or (q_before < 0.0 and T <= guard_lo + projection_margin)
        high_guard_trigger = T >= guard_hi or (q_before > 0.0 and T >= guard_hi - projection_margin)

        if low_guard_trigger:
            req_delta_c = max(temp_sp - T, guard_lo - T)
            req_heat = float(np.clip(kp * req_delta_c, 0.0, q_max))
            req_heat = min(req_heat, c_cap_q * max(req_delta_c, 0.0) / dt_s)
            q_after = max(max(q_before, 0.0), req_heat)
            reason = "temp_low_guard"
        elif high_guard_trigger:
            req_delta_c = max(T - temp_sp, T - guard_hi)
            req_cool = float(np.clip(kp * req_delta_c, 0.0, abs(q_min)))
            req_cool = min(req_cool, c_cap_q * max(req_delta_c, 0.0) / dt_s)
            q_after = min(min(q_before, 0.0), -req_cool)
            reason = "temp_high_guard"

        q_after = float(np.clip(q_after, q_min, q_max))
        action_phys[2] = q_after
        q_delta = q_after - q_before
        if abs(q_delta) > 1e-9:
            info["active"] = True
            info["reason"] = reason
            info["q_hvac_after"] = q_after
            info["q_hvac_delta"] = q_delta
            self.safety_override_count += 1
            self.safety_override_reason_counts[reason] = int(
                self.safety_override_reason_counts.get(reason, 0)
            ) + 1
            self.episode_safety_override_count += 1
            self.episode_safety_override_reason_counts[reason] = int(
                self.episode_safety_override_reason_counts.get(reason, 0)
            ) + 1
        return action_phys, info

    def _linear_band_penalty(self, value: float, lo: float, hi: float, coef: float) -> float:
        coef_abs = abs(float(coef))
        if value < lo:
            return -coef_abs * (lo - value)
        if value > hi:
            return -coef_abs * (value - hi)
        return 0.0

    def _piecewise_penalty(
        self,
        value: float,
        ideal_lo: float,
        ideal_hi: float,
        accept_lo: float,
        accept_hi: float,
        mild_coef: float,
        severe_coef: float,
    ) -> float:
        mild_abs = abs(float(mild_coef))
        severe_abs = abs(float(severe_coef))
        if value < accept_lo:
            return (
                -mild_abs * max(0.0, ideal_lo - accept_lo)
                - severe_abs * (accept_lo - value)
            )
        if value < ideal_lo:
            return -mild_abs * (ideal_lo - value)
        if value <= ideal_hi:
            return 0.0
        if value <= accept_hi:
            return -mild_abs * (value - ideal_hi)
        return (
            -mild_abs * max(0.0, accept_hi - ideal_hi)
            - severe_abs * (value - accept_hi)
        )

    def _check_thermal_meltdown(self) -> Dict[str, Any]:
        T = float(self.state[1])
        T_lo = float(self.reward_params.get("thermal_meltdown_temp_lo", 15.0))
        T_hi = float(self.reward_params.get("thermal_meltdown_temp_hi", 32.0))
        hold_seconds = max(
            float(
                self.reward_params.get(
                    "thermal_meltdown_hold_seconds",
                    max(float(self.dt), 1800.0),
                )
            ),
            max(float(self.dt), 1.0),
        )
        T_lo_hard = float(
            self.reward_params.get("thermal_meltdown_temp_lo_hard", T_lo - 3.0)
        )
        T_hi_hard = float(
            self.reward_params.get("thermal_meltdown_temp_hi_hard", T_hi + 3.0)
        )

        active = False
        reason = "none"
        side = "none"
        exceedance = 0.0
        if T < T_lo:
            side = "low"
            exceedance = T_lo - T
        elif T > T_hi:
            side = "high"
            exceedance = T - T_hi

        if side == "none":
            self._thermal_violation_duration_s = 0.0
            self._thermal_violation_peak_c = 0.0
            self._thermal_violation_side = "none"
        else:
            if self._thermal_violation_side not in {side, "none"}:
                self._thermal_violation_duration_s = 0.0
                self._thermal_violation_peak_c = 0.0
            self._thermal_violation_side = side
            self._thermal_violation_duration_s += float(self.dt)
            self._thermal_violation_peak_c = max(
                float(self._thermal_violation_peak_c), float(exceedance)
            )
            if side == "low" and T <= T_lo_hard:
                active = True
                reason = "thermal_meltdown_low_immediate"
            elif side == "high" and T >= T_hi_hard:
                active = True
                reason = "thermal_meltdown_high_immediate"
            elif self._thermal_violation_duration_s >= hold_seconds:
                active = True
                reason = f"thermal_meltdown_{side}_duration"

        info = {
            "active": bool(active),
            "reason": str(reason),
            "temperature_c": float(T),
            "violation_duration_s": float(self._thermal_violation_duration_s),
            "violation_peak_c": float(self._thermal_violation_peak_c),
            "hold_seconds": float(hold_seconds),
            "side": str(self._thermal_violation_side),
        }
        self._last_thermal_meltdown_info = dict(info)
        return info

    def _reset_constraint_tracking(self) -> None:
        self._last_constraint_info = {}
        self._episode_constraint_cost_totals = self._make_constraint_total_dict()
        self._episode_constraint_raw_totals = self._make_constraint_raw_dict()
        self._episode_constraint_counts = self._make_constraint_count_dict()

    @staticmethod
    def _make_constraint_total_dict() -> Dict[str, float]:
        return {
            "temp": 0.0,
            "co2": 0.0,
            "rh": 0.0,
            "dli": 0.0,
            "photoperiod": 0.0,
            "target_progress": 0.0,
            "harvest_fail": 0.0,
            "safety_override": 0.0,
            "thermal_meltdown": 0.0,
            "ode_failure": 0.0,
            "climate": 0.0,
            "daily_light": 0.0,
            "event": 0.0,
            "termination": 0.0,
            "overall": 0.0,
        }

    @staticmethod
    def _make_constraint_raw_dict() -> Dict[str, float]:
        return {
            "temp": 0.0,
            "co2": 0.0,
            "rh": 0.0,
            "dli": 0.0,
            "photoperiod": 0.0,
            "target_progress": 0.0,
            "harvest_fail": 0.0,
            "safety_override": 0.0,
            "thermal_meltdown": 0.0,
            "ode_failure": 0.0,
        }

    @staticmethod
    def _make_constraint_count_dict() -> Dict[str, int]:
        return {
            "steps": 0,
            "any": 0,
            "temp": 0,
            "co2": 0,
            "rh": 0,
            "dli": 0,
            "photoperiod": 0,
            "target_progress": 0,
            "harvest_fail": 0,
            "safety_override": 0,
            "thermal_meltdown": 0,
            "ode_failure": 0,
            "climate": 0,
            "daily_light": 0,
            "event": 0,
            "termination": 0,
        }

    def _accumulate_constraint_info(self, constraint_info: Dict[str, Any]) -> None:
        cost = constraint_info.get("cost", {})
        raw = constraint_info.get("raw", {})
        totals = constraint_info.get("totals", {})
        active = constraint_info.get("active", {})

        self._episode_constraint_counts["steps"] += 1
        for key in self._episode_constraint_raw_totals:
            self._episode_constraint_raw_totals[key] += float(raw.get(key, 0.0))
        for key in self._episode_constraint_cost_totals:
            if key in totals:
                self._episode_constraint_cost_totals[key] += float(totals.get(key, 0.0))
            else:
                self._episode_constraint_cost_totals[key] += float(cost.get(key, 0.0))
        for key in self._episode_constraint_counts:
            if key == "steps":
                continue
            self._episode_constraint_counts[key] += int(bool(active.get(key, False)))

    def _build_constraint_info(self, reward_info: Dict[str, Any]) -> Dict[str, Any]:
        rp = self.reward_params
        raw = {
            "temp": max(-float(reward_info.get("p_temp", 0.0)), 0.0),
            "co2": max(-float(reward_info.get("p_CO2", 0.0)), 0.0),
            "rh": max(-float(reward_info.get("p_hum", 0.0)), 0.0),
            "dli": max(
                float(
                    reward_info.get(
                        "daily_dli_constraint_raw",
                        -float(reward_info.get("daily_dli_low_penalty", 0.0))
                        - float(reward_info.get("daily_dli_high_penalty", 0.0)),
                    )
                ),
                0.0,
            ),
            "photoperiod": max(
                float(
                    reward_info.get(
                        "daily_photoperiod_constraint_raw",
                        -float(reward_info.get("daily_photoperiod_low_penalty", 0.0))
                        - float(reward_info.get("daily_photoperiod_high_penalty", 0.0)),
                    )
                ),
                0.0,
            ),
            "target_progress": max(
                float(reward_info.get("preharvest_target_penalty_raw", 0.0)),
                0.0,
            ),
            "harvest_fail": max(
                float(
                    reward_info.get(
                        "harvest_fail_constraint_raw",
                        -float(reward_info.get("harvest_fail_penalty", 0.0)),
                    )
                ),
                0.0,
            ),
            "safety_override": max(
                -float(reward_info.get("safety_override_penalty", 0.0)),
                0.0,
            ),
            "thermal_meltdown": max(
                -float(reward_info.get("thermal_meltdown_penalty", 0.0)),
                0.0,
            ),
            "ode_failure": max(
                -float(reward_info.get("ode_status_penalty", 0.0)),
                0.0,
            ),
        }

        references = {
            "temp": max(float(reward_info.get("climate_penalty_reference", 5.0)), 1e-9),
            "co2": max(float(reward_info.get("climate_penalty_reference", 5.0)), 1e-9),
            "rh": max(float(reward_info.get("climate_penalty_reference", 5.0)), 1e-9),
            "dli": max(float(rp.get("daily_penalty_reference", 10.0)), 1e-9),
            "photoperiod": max(float(rp.get("daily_penalty_reference", 10.0)), 1e-9),
            "target_progress": max(
                float(reward_info.get("economic_reward_reference", 0.10)),
                1e-9,
            ),
            "harvest_fail": max(
                float(reward_info.get("harvest_event_penalty_reference", 50.0)),
                1e-9,
            ),
            "safety_override": max(
                float(reward_info.get("safety_override_reference", 0.25)),
                1e-9,
            ),
            "thermal_meltdown": max(
                float(
                    rp.get(
                        "thermal_meltdown_reference",
                        abs(float(rp.get("thermal_meltdown_penalty", -100.0))),
                    )
                ),
                1e-9,
            ),
            "ode_failure": max(float(rp.get("ode_failure_reference", 10.0)), 1e-9),
        }
        cost = {
            key: float(raw[key]) / references[key]
            for key in raw
        }
        totals = {
            "climate": cost["temp"] + cost["co2"] + cost["rh"],
            "daily_light": cost["dli"] + cost["photoperiod"],
            "event": cost["harvest_fail"] + cost["safety_override"],
            "termination": cost["thermal_meltdown"] + cost["ode_failure"],
        }
        totals["overall"] = (
            totals["climate"]
            + totals["daily_light"]
            + cost["target_progress"]
            + totals["event"]
            + totals["termination"]
        )
        active = {
            "any": totals["overall"] > 0.0,
            "temp": cost["temp"] > 0.0,
            "co2": cost["co2"] > 0.0,
            "rh": cost["rh"] > 0.0,
            "dli": cost["dli"] > 0.0,
            "photoperiod": cost["photoperiod"] > 0.0,
            "target_progress": cost["target_progress"] > 0.0,
            "harvest_fail": cost["harvest_fail"] > 0.0,
            "safety_override": cost["safety_override"] > 0.0,
            "thermal_meltdown": cost["thermal_meltdown"] > 0.0,
            "ode_failure": cost["ode_failure"] > 0.0,
            "climate": totals["climate"] > 0.0,
            "daily_light": totals["daily_light"] > 0.0,
            "event": totals["event"] > 0.0,
            "termination": totals["termination"] > 0.0,
        }
        return {
            "raw": raw,
            "reference": references,
            "cost": cost,
            "totals": totals,
            "active": active,
        }

    def _settle_daily_reward_if_needed(self) -> Tuple[float, Dict[str, Any]]:
        completed_days = int(np.floor(self._episode_hours_elapsed / 24.0))
        if completed_days <= self._last_settled_day_index + 1:
            return 0.0, {}

        rp = self.reward_params
        reward_scale = float(rp.get("reward_scale", 1.0))
        weight_penalty = float(rp.get("weight_penalty", 1.0))
        daily_penalty_reference = max(
            float(rp.get("daily_penalty_reference", 10.0)), 1e-9
        )
        dli_dense = float(self.daily_DLI_dense)
        dli_finishing = float(self.daily_DLI_finishing)
        dli = self._compute_area_weighted_daily_dli(dli_dense, dli_finishing)
        light_hours = float(self.daily_light_hours)
        dli_caps = self._get_daily_zone_dli_caps()

        dli_low_penalty = 0.0
        dli_high_penalty = 0.0
        if self.daily_dli_penalty_mode == "legacy_average_band":
            dli_ideal_min = float(rp.get("dli_ideal_min", 12.0))
            dli_ideal_max = float(rp.get("dli_ideal_max", 17.0))
            dli_accept_min = float(rp.get("dli_accept_min", 8.0))
            dli_accept_max = float(rp.get("dli_accept_max", 20.0))
            if dli < dli_ideal_min:
                coef = abs(float(rp.get("dli_low_penalty", -10.0)))
                mild_gap = max(0.0, dli_ideal_min - max(dli, dli_accept_min))
                severe_gap = max(0.0, dli_accept_min - dli)
                dli_low_penalty = -(0.5 * coef * mild_gap + coef * severe_gap)
            elif dli > dli_ideal_max:
                coef = abs(float(rp.get("dli_high_penalty", -2.0)))
                mild_gap = max(0.0, min(dli, dli_accept_max) - dli_ideal_max)
                severe_gap = max(0.0, dli - dli_accept_max)
                dli_high_penalty = -(0.5 * coef * mild_gap + coef * severe_gap)
        elif self.daily_dli_penalty_mode == "zone_upper_guard":
            coef = abs(float(rp.get("dli_high_penalty", -2.0)))
            dense_excess = max(0.0, dli_dense - float(dli_caps["dense"]))
            finishing_excess = max(
                0.0,
                dli_finishing - float(dli_caps["finishing"]),
            )
            dli_high_penalty = -coef * (dense_excess + finishing_excess)

        dli_constraint_raw = abs(min(dli_low_penalty + dli_high_penalty, 0.0))

        photoperiod_low_penalty = 0.0
        photoperiod_high_penalty = 0.0
        photoperiod_target_hours = max(float(self.photo_period[0]), 0.0)
        if self.daily_photoperiod_penalty_mode == "band":
            pp_ideal_min = float(rp.get("photoperiod_ideal_min", 16.0))
            pp_ideal_max = float(rp.get("photoperiod_ideal_max", 16.0))
            pp_accept_min = float(rp.get("photoperiod_accept_min", 16.0))
            pp_accept_max = float(rp.get("photoperiod_accept_max", 16.0))
            if light_hours < pp_ideal_min:
                coef = abs(float(rp.get("photoperiod_low_penalty", -10.0)))
                mild_gap = max(0.0, pp_ideal_min - max(light_hours, pp_accept_min))
                severe_gap = max(0.0, pp_accept_min - light_hours)
                photoperiod_low_penalty = -(0.5 * coef * mild_gap + coef * severe_gap)
            elif light_hours > pp_ideal_max:
                coef = abs(float(rp.get("photoperiod_high_penalty", -3.0)))
                mild_gap = max(0.0, min(light_hours, pp_accept_max) - pp_ideal_max)
                severe_gap = max(0.0, light_hours - pp_accept_max)
                photoperiod_high_penalty = -(0.5 * coef * mild_gap + coef * severe_gap)
        elif self.daily_photoperiod_penalty_mode == "fixed_schedule_guard":
            tol = max(float(self.photoperiod_guard_tolerance_h), 0.0)
            low_gap = max(0.0, photoperiod_target_hours - tol - light_hours)
            high_gap = max(0.0, light_hours - (photoperiod_target_hours + tol))
            if low_gap > 0.0:
                photoperiod_low_penalty = -abs(
                    float(rp.get("photoperiod_low_penalty", -10.0))
                ) * low_gap
            if high_gap > 0.0:
                photoperiod_high_penalty = -abs(
                    float(rp.get("photoperiod_high_penalty", -3.0))
                ) * high_gap

        photoperiod_constraint_raw = abs(
            min(photoperiod_low_penalty + photoperiod_high_penalty, 0.0)
        )

        reward_raw = (
            dli_low_penalty
            + dli_high_penalty
            + photoperiod_low_penalty
            + photoperiod_high_penalty
        )
        reward_norm = self._normalize_reward_component(
            reward_raw, daily_penalty_reference
        )
        reward = reward_scale * weight_penalty * reward_norm
        self._last_settled_day_index = completed_days - 1
        info = {
            "daily_settlement_reward": float(reward),
            "daily_settlement_reward_raw": float(reward_raw),
            "daily_settlement_reward_norm": float(reward_norm),
            "daily_penalty_weight": float(weight_penalty),
            "daily_dli_penalty_mode": str(self.daily_dli_penalty_mode),
            "daily_photoperiod_penalty_mode": str(
                self.daily_photoperiod_penalty_mode
            ),
            "daily_DLI_dense_realized": float(dli_dense),
            "daily_DLI_finishing_realized": float(dli_finishing),
            "daily_DLI_realized": float(dli),
            "daily_light_hours_realized": light_hours,
            "daily_dli_low_penalty": float(dli_low_penalty),
            "daily_dli_high_penalty": float(dli_high_penalty),
            "daily_dli_constraint_raw": float(dli_constraint_raw),
            "daily_dli_dense_cap": float(dli_caps["dense"]),
            "daily_dli_finishing_cap": float(dli_caps["finishing"]),
            "daily_dli_weighted_cap": float(dli_caps["weighted"]),
            "daily_photoperiod_low_penalty": float(photoperiod_low_penalty),
            "daily_photoperiod_high_penalty": float(photoperiod_high_penalty),
            "daily_photoperiod_constraint_raw": float(photoperiod_constraint_raw),
            "daily_photoperiod_target_hours": float(photoperiod_target_hours),
            "daily_photoperiod_guard_tolerance_h": float(
                self.photoperiod_guard_tolerance_h
            ),
            "daily_settlement_day_index": int(completed_days - 1),
        }

        self.daily_DLI_dense = 0.0
        self.daily_DLI_finishing = 0.0
        self.daily_DLI = 0.0
        self.daily_light_hours = 0.0
        return float(reward), info

    def _clip_obs(self, value: float) -> float:
        return float(np.clip(value, -1.0, 1.0))

    def _scale_zero_to_one(self, value: float, max_value: float) -> float:
        return self._clip_obs(value / max(max_value, 1e-12))

    def _scale_symmetric(self, value: float, span: float) -> float:
        return self._clip_obs(value / max(span, 1e-12))

    def _scale_centered(self, value: float, center: float, span: float) -> float:
        return self._clip_obs((value - center) / max(span, 1e-12))

    def _get_planned_light_hours_elapsed(self) -> float:
        elapsed_in_day = float(self._episode_hours_elapsed % 24.0)
        if elapsed_in_day <= 0.0:
            return 0.0

        light_hours = max(float(self.photo_period[0]), 1e-9)
        start_hour = float(self._initial_hour_of_day % 24.0)
        dawn = float(self.photo_period[1])
        dusk = dawn + light_hours

        if dusk <= 24.0:
            light_segments = [(dawn, dusk)]
        else:
            light_segments = [(dawn, 24.0), (0.0, dusk - 24.0)]

        end_hour = start_hour + elapsed_in_day
        observation_segments = [(start_hour, min(end_hour, 24.0))]
        if end_hour > 24.0:
            observation_segments.append((0.0, end_hour - 24.0))

        planned_light_hours = 0.0
        for seg_lo, seg_hi in observation_segments:
            for light_lo, light_hi in light_segments:
                planned_light_hours += max(
                    0.0,
                    min(seg_hi, light_hi) - max(seg_lo, light_lo),
                )
        return float(np.clip(planned_light_hours, 0.0, light_hours))

    def _get_observation(self) -> np.ndarray:
        limits = self._obs_limits
        lumped = self.batch_manager._extract_lumped_features()

        C_density = float(self.state[0])
        T = float(self.state[1])
        xH = float(self.state[2])
        RH = absolute_humidity_to_relative(T, xH, self.container_params)
        C_ppm = co2_density_to_ppm(C_density, T)
        vpd_kpa = absolute_humidity_to_vpd(T, xH, self.container_params)

        T_out = float(self.external[0])
        xH_out = float(self.external[1])
        RH_out = absolute_humidity_to_relative(T_out, xH_out, self.container_params)

        photo_idx = int(self.time_step) % self._i_max
        targets = self.get_climate_targets(photo_idx=photo_idx, T_current=T)
        temp_error = T - float(targets["temp_setpoint_c"])
        co2_error_ppm = C_ppm - float(targets["co2_setpoint_ppm"])
        vpd_error = vpd_kpa - float(targets["vpd_target_kpa"])
        light_on = 1.0 if bool(targets["light_on"]) else -1.0
        day_progress = float(
            (self._initial_hour_of_day + self._episode_hours_elapsed) % 24.0
        ) / 24.0
        daily_light_metrics = self._get_daily_light_progress_metrics()
        dli_progress = float(daily_light_metrics["dli_progress"])
        dli_progress_error = float(daily_light_metrics["dli_progress_error"])

        obs_env = np.array(
            [
                self._scale_centered(T, 22.0, 12.0),
                self._scale_centered(RH, 0.70, 0.30),
                self._scale_centered(C_ppm, 900.0, 900.0),
                self._scale_centered(T_out, 15.0, 30.0),
                self._scale_centered(RH_out, 0.70, 0.30),
                self._scale_zero_to_one(vpd_kpa, limits["vpd"]),
                self._scale_symmetric(temp_error, limits["temp_err"]),
                self._scale_symmetric(co2_error_ppm, limits["co2_err"]),
                self._scale_symmetric(vpd_error, limits["vpd_err"]),
                light_on,
                self._clip_obs(2.0 * day_progress - 1.0),
                dli_progress,
            ],
            dtype=np.float32,
        )
        if self.include_electricity_price_observation:
            obs_env = np.concatenate(
                [
                    obs_env,
                    np.array(
                        [self._scale_electricity_price_for_observation(self.elec_price)],
                        dtype=np.float32,
                    ),
                ]
            ).astype(np.float32)

        if self.observation_semantics == "legacy31":
            obs_crop = np.array(
                [
                    self._scale_zero_to_one(float(lumped.get("lai_seedling", 0.0)), limits["lai"]),
                    self._scale_zero_to_one(float(lumped.get("lai_transplant", 0.0)), limits["lai"]),
                    self._scale_zero_to_one(
                        float(lumped.get("density_seedling", 0.0)), limits["density"]
                    ),
                    self._scale_zero_to_one(
                        float(lumped.get("density_transplant", 0.0)), limits["density"]
                    ),
                    self._scale_symmetric(
                        float(lumped.get("delta_density_seedling", 0.0)),
                        limits["delta_density"],
                    ),
                    self._scale_symmetric(
                        float(lumped.get("delta_density_transplant", 0.0)),
                        limits["delta_density"],
                    ),
                    self._scale_zero_to_one(
                        float(lumped.get("W_oldest_per_plant", 0.0)), limits["W_old"]
                    ),
                    self._scale_zero_to_one(
                        float(lumped.get("days_left_oldest", 0.0)), limits["days_left"]
                    ),
                    self._scale_zero_to_one(
                        float(lumped.get("total_cycle_days_left", float(self.t1 + self.t2))),
                        limits["cycle_days_left"],
                    ),
                ],
                dtype=np.float32,
            )
        elif self.observation_semantics == "target31_v2":
            obs_crop = np.array(
                [
                    self._scale_zero_to_one(float(lumped.get("lai_seedling", 0.0)), limits["lai"]),
                    self._scale_zero_to_one(float(lumped.get("lai_transplant", 0.0)), limits["lai"]),
                    self._scale_zero_to_one(
                        float(lumped.get("density_seedling", 0.0)), limits["density"]
                    ),
                    self._scale_zero_to_one(
                        float(lumped.get("density_transplant", 0.0)), limits["density"]
                    ),
                    self._scale_symmetric(
                        float(lumped.get("delta_density_seedling", 0.0)),
                        limits["delta_density"],
                    ),
                    self._scale_symmetric(
                        float(lumped.get("delta_density_transplant", 0.0)),
                        limits["delta_density"],
                    ),
                    self._scale_centered(
                        float(lumped.get("oldest_target_ratio", 0.0)),
                        1.0,
                        limits["target_ratio"],
                    ),
                    self._scale_zero_to_one(
                        float(lumped.get("days_left_oldest", 0.0)), limits["days_left"]
                    ),
                    self._scale_symmetric(
                        float(dli_progress_error),
                        limits["dli_progress_err"],
                    ),
                ],
                dtype=np.float32,
            )
        else:
            obs_crop = np.array(
                [
                    self._scale_zero_to_one(float(lumped.get("lai_seedling", 0.0)), limits["lai"]),
                    self._scale_zero_to_one(float(lumped.get("lai_transplant", 0.0)), limits["lai"]),
                    self._scale_zero_to_one(
                        float(lumped.get("density_seedling", 0.0)), limits["density"]
                    ),
                    self._scale_zero_to_one(
                        float(lumped.get("density_transplant", 0.0)), limits["density"]
                    ),
                    self._scale_symmetric(
                        float(lumped.get("delta_density_seedling", 0.0)),
                        limits["delta_density"],
                    ),
                    self._scale_symmetric(
                        float(lumped.get("delta_density_transplant", 0.0)),
                        limits["delta_density"],
                    ),
                    self._scale_centered(
                        float(lumped.get("oldest_target_ratio", 0.0)),
                        1.0,
                        limits["target_ratio"],
                    ),
                    self._scale_zero_to_one(
                        float(lumped.get("days_left_oldest", 0.0)), limits["days_left"]
                    ),
                    self._scale_zero_to_one(
                        float(lumped.get("n_harvests", 0.0)), limits["harvest_count"]
                    ),
                ],
                dtype=np.float32,
            )

        ctx_span = np.maximum(self._ctx_high - self._ctx_low, 1e-12)
        obs_context = np.array(
            [
                2.0 * (float(self.t1) - self._ctx_low[0]) / ctx_span[0] - 1.0,
                2.0 * (float(self.t2) - self._ctx_low[1]) / ctx_span[1] - 1.0,
                2.0 * (float(self.N1) - self._ctx_low[2]) / ctx_span[2] - 1.0,
                2.0 * (float(self.rho2) - self._ctx_low[3]) / ctx_span[3] - 1.0,
            ],
            dtype=np.float32,
        )
        obs_context = np.clip(obs_context, -1.0, 1.0)
        if self.mask_schedule_context_observation:
            obs_context[:] = 0.0

        prev = (
            self.prev_action_4d.astype(np.float32)
            if self.prev_action_4d is not None
            else np.zeros(5, dtype=np.float32)
        )
        act_span = np.maximum(self._act_high - self._act_low, 1e-12)
        obs_action = 2.0 * (prev - self._act_low) / act_span - 1.0
        obs_action = np.clip(obs_action.astype(np.float32), -1.0, 1.0)

        obs = np.concatenate([obs_env, obs_crop, obs_context, obs_action]).astype(
            np.float32
        )
        if not np.all(np.isfinite(obs)):
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(obs, -1.0, 1.0)

    def _get_info(
        self,
        batch_info=None,
        terminated: bool = False,
        truncated: bool = False,
        termination_reason: str = "running",
        ode_status: int = 0,
        thermal_meltdown: bool = False,
    ) -> Dict[str, Any]:
        lumped = self.batch_manager._extract_lumped_features()
        T = float(self.state[1])
        xH = float(self.state[2])
        vpd_kpa = absolute_humidity_to_vpd(T, xH, self.container_params)
        photo_idx = int(self.time_step) % self._i_max
        targets = self.get_climate_targets(photo_idx=photo_idx, T_current=T)
        actual_steps = int(self.time_step)
        planned_steps = int(self.episode_length)
        actual_days = float(actual_steps * self.dt / 86400.0)
        planned_days = float(planned_steps * self.dt / 86400.0)
        completion_ratio = float(actual_steps / max(planned_steps, 1))
        ended_early = bool((terminated or truncated) and actual_steps < planned_steps)
        safety_info = getattr(self, "_last_safety_override", None) or {}
        residual_gate_info = getattr(self, "_last_residual_gate", None) or {}
        thermal_info = getattr(self, "_last_thermal_meltdown_info", None) or {}
        constraint_info = getattr(self, "_last_constraint_info", None) or {}
        env_step_diag = getattr(self, "_last_env_step_diagnostics", None) or {}
        env_clip_counts = dict(getattr(self, "_env_step_clip_counts", {}) or {})
        light_control_info = dict(getattr(self, "_last_light_control_info", {}) or {})
        constraint_totals = dict(getattr(self, "_episode_constraint_cost_totals", {}) or {})
        constraint_counts = dict(getattr(self, "_episode_constraint_counts", {}) or {})
        reference_growth = dict(
            getattr(self.batch_manager, "reference_growth_profile", {}) or {}
        )
        rich_info = self._uses_rich_info(self.info_detail_level)
        residual_gate_vector = np.asarray(
            residual_gate_info.get("actuator_vector", [1.0] * 5),
            dtype=np.float32,
        ).reshape(-1)
        if residual_gate_vector.size < 5:
            residual_gate_vector = np.pad(
                residual_gate_vector,
                (0, 5 - residual_gate_vector.size),
                constant_values=1.0,
            )
        else:
            residual_gate_vector = residual_gate_vector[:5]
        reference_min_ratio = float(
            reference_growth.get("reference_harvest_vs_min_ratio", 0.0)
        )
        reference_target_ratio = float(
            reference_growth.get("reference_harvest_vs_target_ratio", 0.0)
        )
        schedule_reference_warning_count = int(
            (reference_min_ratio < 1.0) or (reference_target_ratio < 1.0)
        )
        schedule_reference_warnings = (
            self._build_schedule_reference_warnings() if rich_info else []
        )
        daily_light_metrics = self._get_daily_light_progress_metrics()
        planned_light_hours = float(daily_light_metrics["planned_light_hours"])
        expected_dli_so_far = float(daily_light_metrics["expected_dli_so_far"])
        dli_progress_error = float(daily_light_metrics["dli_progress_error"])
        instant_total_P_g_m2_h = self.total_P * 1000.0 * 3600.0 / max(self.A_total, 1e-12)
        instant_total_R_g_m2_h = self.total_R * 1000.0 * 3600.0 / max(self.A_total, 1e-12)
        instant_total_E_g_m2_h = self.total_E * 1000.0 * 3600.0 / max(self.A_total, 1e-12)
        vent_hold_g_m2_h = float(
            getattr(self, "physics_diagnostics", {}).get("co2_hold_1000ppm_g_m2_h", 0.0)
        )
        co2_supply_max_g_m2_h = float(
            getattr(self, "physics_diagnostics", {}).get("co2_supply_max_g_m2_h", 0.0)
        )
        instant_net_canopy_co2_demand_g_m2_h = max(
            instant_total_P_g_m2_h - instant_total_R_g_m2_h,
            0.0,
        )
        co2_supply_headroom_ratio = co2_supply_max_g_m2_h / max(
            vent_hold_g_m2_h + instant_net_canopy_co2_demand_g_m2_h,
            1e-12,
        )
        total_harvest_mass_kg = float(
            getattr(self.batch_manager, "total_harvest_mass", 0.0)
        )
        info = {
            "time_step": self.time_step,
            "hour_of_day": self.hour_of_day,
            "initial_hour_of_day": float(self._initial_hour_of_day),
            "day_of_period": self.day_of_period,
            "zone_semantics_seedling": str(self.zone_semantics["seedling"]),
            "zone_semantics_transplant": str(self.zone_semantics["transplant"]),
            "episode_length_steps": int(self.episode_length),
            "episode_length_days": float(self.episode_length_days),
            "episode_length_mode": str(self.episode_length_mode),
            "episode_length_request_mode": str(
                getattr(self, "episode_length_request_mode", self.episode_length_mode)
            ),
            "planned_episode_steps": planned_steps,
            "planned_episode_days": planned_days,
            "actual_episode_steps": actual_steps,
            "actual_episode_days": actual_days,
            "sim_days_elapsed": actual_days,
            "episode_completion_ratio": completion_ratio,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "termination_reason": str(termination_reason),
            "ended_early": ended_early,
            "ode_status": int(ode_status),
            "env_integration_method": str(env_step_diag.get("integration_method", "unknown")),
            "env_temperature_clipped": bool(env_step_diag.get("temperature_clipped", False)),
            "env_co2_density_bounded": bool(env_step_diag.get("co2_density_bounded", False)),
            "env_co2_rate_limited": bool(env_step_diag.get("co2_rate_limited", False)),
            "env_condensation_active": bool(env_step_diag.get("condensation_active", False)),
            "env_condensation_removed_kg": float(
                env_step_diag.get("condensation_removed_kg", 0.0)
            ),
            "env_condensation_delta_temp_c": float(
                env_step_diag.get("condensation_delta_temp_c", 0.0)
            ),
            "env_humidity_floor_clipped": bool(
                env_step_diag.get("humidity_floor_clipped", False)
            ),
            "env_humidity_saturation_clipped": bool(
                env_step_diag.get("humidity_saturation_clipped", False)
            ),
            "env_used_euler_fallback": bool(env_step_diag.get("used_euler_fallback", False)),
            "env_used_safe_defaults": bool(env_step_diag.get("used_safe_defaults", False)),
            "thermal_meltdown": bool(thermal_meltdown),
            "thermal_meltdown_reason": str(thermal_info.get("reason", "none")),
            "thermal_violation_duration_s": float(
                thermal_info.get("violation_duration_s", 0.0)
            ),
            "thermal_violation_peak_c": float(
                thermal_info.get("violation_peak_c", 0.0)
            ),
            "thermal_meltdown_hold_seconds": float(
                thermal_info.get("hold_seconds", 0.0)
            ),
            "safety_override_active": bool(safety_info.get("active", False)),
            "safety_override_reason": str(safety_info.get("reason", "none")),
            "safety_override_q_hvac_delta": float(safety_info.get("q_hvac_delta", 0.0)),
            "safety_override_count": int(self.episode_safety_override_count),
            "safety_override_reason_unique_count": int(
                len(self.episode_safety_override_reason_counts)
            ),
            "lifetime_safety_override_count": int(self.safety_override_count),
            "residual_gate": float(residual_gate_info.get("overall", 1.0)),
            "residual_gate_temp": float(residual_gate_info.get("temperature", 1.0)),
            "residual_gate_co2": float(residual_gate_info.get("co2", 1.0)),
            "residual_gate_humidity": float(residual_gate_info.get("humidity", 1.0)),
            "residual_gate_act_0": float(residual_gate_vector[0]),
            "residual_gate_act_1": float(residual_gate_vector[1]),
            "residual_gate_act_2": float(residual_gate_vector[2]),
            "residual_gate_act_3": float(residual_gate_vector[3]),
            "residual_gate_act_4": float(residual_gate_vector[4]),
            "observation_semantics": str(self.observation_semantics),
            "action_semantics": str(self.action_semantics),
            "mask_schedule_context_observation": bool(
                self.mask_schedule_context_observation
            ),
            "pid_anchor_I1": float(self.I_target_seedling),
            "pid_anchor_I2": float(self.I_target_transplant),
            "T": T,
            "RH": absolute_humidity_to_relative(T, xH, self.container_params) * 100.0,
            "C_ppm": co2_density_to_ppm(float(self.state[0]), T),
            "VPD_kPa": vpd_kpa,
            "elec_price": float(self._last_applied_elec_price),
            "next_elec_price": float(self.elec_price),
            "price_model_type": str(self.electricity_price_model),
            "tou_tariff_scenario": str(self.tou_tariff_scenario),
            "include_electricity_price_observation": bool(
                self.include_electricity_price_observation
            ),
            "fixed_vent_rate_m3_m2_s": float(self._V_vent_fixed),
            "fixed_vent_ach": float(
                getattr(self, "physics_diagnostics", {}).get("fixed_vent_ach", 0.0)
            ),
            "co2_supply_max_g_m2_h": float(co2_supply_max_g_m2_h),
            "co2_hold_1000ppm_g_m2_h": float(vent_hold_g_m2_h),
            "co2_supply_headroom_ratio": float(co2_supply_headroom_ratio),
            "instant_total_P_g_m2_h": float(instant_total_P_g_m2_h),
            "instant_total_R_g_m2_h": float(instant_total_R_g_m2_h),
            "instant_total_E_g_m2_h": float(instant_total_E_g_m2_h),
            "instant_net_canopy_co2_demand_g_m2_h": float(
                instant_net_canopy_co2_demand_g_m2_h
            ),
            "led_ppe_umol_per_j": float(
                getattr(self, "physics_diagnostics", {}).get("led_ppe_umol_per_j", 0.0)
            ),
            "led_radiant_efficiency": float(
                getattr(self, "physics_diagnostics", {}).get("led_radiant_efficiency", 0.0)
            ),
            "led_heat_fraction": float(
                getattr(self, "physics_diagnostics", {}).get("led_heat_fraction", 0.0)
            ),
            "led_room_heat_fraction": float(
                getattr(self, "physics_diagnostics", {}).get("led_room_heat_fraction", 0.0)
            ),
            "target_led_power_total_W": float(
                getattr(self, "physics_diagnostics", {}).get("P_led_total_W", 0.0)
            ),
            "target_led_driver_loss_W": float(
                getattr(self, "physics_diagnostics", {}).get("Q_led_driver_loss_W", 0.0)
            ),
            "target_led_room_upper_W": float(
                getattr(self, "physics_diagnostics", {}).get("Q_led_room_upper_W", 0.0)
            ),
            "target_led_heat_total_W": float(
                getattr(self, "physics_diagnostics", {}).get("Q_led_room_W", 0.0)
            ),
            "temp_target_c": float(targets["temp_setpoint_c"]),
            "co2_target_ppm": float(targets["co2_setpoint_ppm"]),
            "vpd_target_kpa": float(targets["vpd_target_kpa"]),
            "light_on": bool(targets["light_on"]),
            "episode_reward": self.episode_reward,
            "total_cost": self.total_cost,
            "hours_continuous_light": self.hours_continuous_light,
            "hours_continuous_dark": self.hours_continuous_dark,
            "daily_DLI_dense": float(self.daily_DLI_dense),
            "daily_DLI_finishing": float(self.daily_DLI_finishing),
            "daily_DLI": float(daily_light_metrics["realized_weighted"]),
            "daily_light_hours": self.daily_light_hours,
            "planned_light_hours_elapsed": float(planned_light_hours),
            "expected_dli_so_far": float(expected_dli_so_far),
            "dli_progress_error": float(dli_progress_error),
            "dli_progress": float(daily_light_metrics["dli_progress"]),
            "dli_dense_cap": float(daily_light_metrics["dense_cap"]),
            "dli_finishing_cap": float(daily_light_metrics["finishing_cap"]),
            "dli_weighted_cap": float(daily_light_metrics["weighted_cap"]),
            "photo_period_hours": float(self.photo_period[0]),
            "light_control_mode": str(self.light_control_mode),
            "light_segments_per_photoperiod": int(self.light_segments_per_photoperiod),
            "lighting_constraint_mode": str(
                getattr(self, "_light_bound_info", {}).get("mode", "hardware_only")
            ),
            "effective_I1_max": float(
                getattr(self, "_light_bound_info", {}).get("effective_I1_max", self.I1_max)
            ),
            "effective_I2_max": float(
                getattr(self, "_light_bound_info", {}).get("effective_I2_max", self.I2_max)
            ),
            "light_update_allowed": bool(light_control_info.get("update_allowed", False)),
            "light_segment_index": int(light_control_info.get("segment_index", -1)),
            "light_hold_initialized": bool(light_control_info.get("initialized", False)),
            "held_I1_target": float(light_control_info.get("held_I1_target", 0.0)),
            "held_I2_target": float(light_control_info.get("held_I2_target", 0.0)),
            "lai_total": lumped.get("lai_total", 0.0),
            "n_harvests": lumped.get("n_harvests", 0),
            "n_transplants": lumped.get("n_transplants", 0),
            "min_dry_mass_per_plant_g": float(
                self.reward_params.get("harvest_min_dry_mass_per_plant", 4.44)
            ),
            "min_fresh_mass_per_plant_g": float(
                self.reward_params.get("harvest_min_dry_mass_per_plant", 4.44) * self.c_fw
            ),
            "target_dry_mass_per_plant_g": float(
                self.reward_params.get(
                    "harvest_target_dry_mass_per_plant",
                    self.reward_params.get("harvest_min_dry_mass_per_plant", 4.44),
                )
            ),
            "target_fresh_mass_per_plant_g": float(
                self.reward_params.get(
                    "harvest_target_dry_mass_per_plant",
                    self.reward_params.get("harvest_min_dry_mass_per_plant", 4.44),
                ) * self.c_fw
            ),
            "oldest_dry_mass_per_plant_g": float(lumped.get("W_oldest_per_plant", 0.0)),
            "oldest_fresh_mass_per_plant_g": float(
                lumped.get("W_oldest_per_plant", 0.0) * self.c_fw
            ),
            "oldest_min_ratio": float(lumped.get("oldest_min_ratio", 0.0)),
            "oldest_target_ratio": float(lumped.get("oldest_target_ratio", 0.0)),
            "oldest_min_gap_g": float(lumped.get("oldest_min_gap_g", 0.0)),
            "oldest_target_gap_g": float(lumped.get("oldest_target_gap_g", 0.0)),
            "oldest_min_surplus_g": float(lumped.get("oldest_min_surplus_g", 0.0)),
            "oldest_target_surplus_g": float(lumped.get("oldest_target_surplus_g", 0.0)),
            "oldest_plant_count": float(lumped.get("oldest_plant_count", 0.0)),
            "reference_harvest_dry_mass_per_plant_g": float(
                reference_growth.get("reference_harvest_dry_mass_per_plant_g", 0.0)
            ),
            "reference_harvest_fresh_mass_per_plant_g": float(
                reference_growth.get("reference_harvest_fresh_mass_per_plant_g", 0.0)
            ),
            "reference_harvest_vs_min_ratio": reference_min_ratio,
            "reference_harvest_vs_target_ratio": reference_target_ratio,
            "schedule_reference_feasibility_class": str(
                self.schedule.get("reference_feasibility_class", "")
            ),
            "schedule_reference_min_feasible": bool(
                self.schedule.get("reference_min_feasible", False)
            ),
            "schedule_reference_target_feasible": bool(
                self.schedule.get("reference_target_feasible", False)
            ),
            "schedule_reference_warning_count": schedule_reference_warning_count,
            "constraint_cost_step": float(
                constraint_info.get("totals", {}).get("overall", 0.0)
            ),
            "constraint_cost_step_climate": float(
                constraint_info.get("totals", {}).get("climate", 0.0)
            ),
            "constraint_cost_step_daily_light": float(
                constraint_info.get("totals", {}).get("daily_light", 0.0)
            ),
            "constraint_cost_step_event": float(
                constraint_info.get("totals", {}).get("event", 0.0)
            ),
            "constraint_cost_step_termination": float(
                constraint_info.get("totals", {}).get("termination", 0.0)
            ),
            "constraint_cost_episode": float(constraint_totals.get("overall", 0.0)),
            "constraint_cost_episode_temp": float(constraint_totals.get("temp", 0.0)),
            "constraint_cost_episode_co2": float(constraint_totals.get("co2", 0.0)),
            "constraint_cost_episode_rh": float(constraint_totals.get("rh", 0.0)),
            "constraint_cost_episode_dli": float(constraint_totals.get("dli", 0.0)),
            "constraint_cost_episode_photoperiod": float(
                constraint_totals.get("photoperiod", 0.0)
            ),
            "constraint_cost_episode_target_progress": float(
                constraint_totals.get("target_progress", 0.0)
            ),
            "constraint_cost_episode_harvest_fail": float(
                constraint_totals.get("harvest_fail", 0.0)
            ),
            "constraint_cost_episode_safety_override": float(
                constraint_totals.get("safety_override", 0.0)
            ),
            "constraint_cost_episode_thermal_meltdown": float(
                constraint_totals.get("thermal_meltdown", 0.0)
            ),
            "constraint_cost_episode_ode_failure": float(
                constraint_totals.get("ode_failure", 0.0)
            ),
            "constraint_cost_episode_climate": float(
                constraint_totals.get("climate", 0.0)
            ),
            "constraint_cost_episode_daily_light": float(
                constraint_totals.get("daily_light", 0.0)
            ),
            "constraint_cost_episode_event": float(
                constraint_totals.get("event", 0.0)
            ),
            "constraint_cost_episode_termination": float(
                constraint_totals.get("termination", 0.0)
            ),
            "constraint_episode_steps": int(constraint_counts.get("steps", actual_steps)),
            "constraint_active_steps": int(constraint_counts.get("any", 0)),
            "constraint_active_ratio": float(
                constraint_counts.get("any", 0) / max(actual_steps, 1)
            ),
            "env_clip_temperature_count": int(env_clip_counts.get("temperature", 0)),
            "env_clip_humidity_count": int(
                env_clip_counts.get("humidity_floor", 0)
                + env_clip_counts.get("humidity_saturation", 0)
            ),
            "env_clip_co2_count": int(env_clip_counts.get("co2", 0)),
            "env_clip_fallback_count": int(env_clip_counts.get("fallback", 0)),
            "env_clip_safe_default_count": int(env_clip_counts.get("safe_default", 0)),
            "n_seedling_batches": int(len(getattr(self.batch_manager, "seedling_batches", []))),
            "n_transplant_batches": int(
                len(getattr(self.batch_manager, "transplant_batches", []))
            ),
            "total_transplants": int(getattr(self.batch_manager, "total_transplants", 0)),
            "total_harvests": int(getattr(self.batch_manager, "total_harvests", 0)),
            "total_harvest_mass_kg": total_harvest_mass_kg,
            "total_harvest_mass_g": float(total_harvest_mass_kg * 1000.0),
        }
        if rich_info:
            info.update(
                {
                    "safety_override_reason_counts": dict(
                        self.episode_safety_override_reason_counts
                    ),
                    "residual_gate_vector": residual_gate_vector.tolist(),
                    "constraint_episode_totals": constraint_totals,
                    "constraint_episode_counts": constraint_counts,
                    "env_step_clip_counts": env_clip_counts,
                    "batch_summary": self.batch_manager.get_state_summary(),
                    "reference_growth_profile": reference_growth,
                    "schedule_reference_warnings": list(schedule_reference_warnings),
                    "physics_diagnostics": dict(
                        getattr(self, "physics_diagnostics", {}) or {}
                    ),
                    "env_step_diagnostics": dict(env_step_diag),
                }
            )
            if self.config_warnings:
                info["config_warnings"] = list(self.config_warnings)
            if self.reward_param_warnings:
                info["reward_param_warnings"] = list(self.reward_param_warnings)
            if self._last_daily_settlement:
                info["daily_settlement"] = dict(self._last_daily_settlement)
        elif self._last_daily_settlement:
            info["daily_settlement"] = dict(self._last_daily_settlement)
        if batch_info:
            info["harvest_mass_g"] = float(batch_info.get("harvest_mass", 0.0))
            info["harvest_fail"] = bool(batch_info.get("harvest_fail", False))
            info["harvest_min_mass_g"] = float(batch_info.get("harvest_min_mass_g", 0.0))
            info["harvest_target_mass_g"] = float(batch_info.get("harvest_target_mass_g", 0.0))
            info["harvest_min_shortfall_g"] = float(
                batch_info.get("harvest_min_shortfall_g", 0.0)
            )
            info["harvest_target_shortfall_g"] = float(
                batch_info.get("harvest_target_shortfall_g", 0.0)
            )
            info["harvest_target_surplus_g"] = float(
                batch_info.get("harvest_target_surplus_g", 0.0)
            )
            info["harvest_mean_dry_mass_per_plant_g"] = float(
                batch_info.get("harvest_mean_dry_mass_per_plant_g", 0.0)
            )
            info["harvest_mean_fresh_mass_per_plant_g"] = float(
                batch_info.get("harvest_mean_fresh_mass_per_plant_g", 0.0)
            )
            info["harvest_mean_target_ratio"] = float(
                batch_info.get("harvest_mean_target_ratio", 0.0)
            )
            info["harvest_fail_n_batches_this_step"] = int(
                batch_info.get("harvest_fail_n_batches_this_step", 0)
            )
        return info

    def _tv_data(self) -> np.ndarray:
        ts = float(self.dt)
        i_max = max(1, int(round(24.0 * 3600.0 / ts)))
        hours = np.arange(i_max, dtype=np.float64) * ts / 3600.0
        dawn = float(self.photo_period[1])
        dusk = min(24.0, dawn + float(self.photo_period[0]))
        x_photo = ((hours >= dawn) & (hours < dusk)).astype(np.float64)
        xT_lb = np.where(x_photo > 0.5, self.temp_range[1][0], self.temp_range[0][0])
        xT_ub = np.where(x_photo > 0.5, self.temp_range[1][1], self.temp_range[0][1])
        T_out = float(self.external[0])
        xH_out = float(self.external[1])
        C_out_ppm = co2_density_to_ppm(float(self.external[2]), T_out)
        xt = np.arange(i_max, dtype=np.float64) / max(i_max - 1, 1)
        return np.column_stack(
            [
                xT_lb,
                xT_ub,
                x_photo,
                np.full(i_max, T_out, dtype=np.float64),
                np.full(i_max, C_out_ppm, dtype=np.float64),
                np.full(i_max, xH_out, dtype=np.float64),
                xt,
            ]
        )

    def scale_inputs(self, u: np.ndarray) -> np.ndarray:
        u_arr = np.asarray(u, dtype=np.float32)
        span = np.maximum(self._act_high - self._act_low, 1e-12)
        return (2.0 * (u_arr - self._act_low) / span - 1.0).astype(np.float32)

    def unscale_inputs(self, u: np.ndarray) -> np.ndarray:
        u_arr = np.asarray(u, dtype=np.float32)
        span = np.maximum(self._act_high - self._act_low, 1e-12)
        return (((u_arr + 1.0) * 0.5) * span + self._act_low).astype(np.float32)

    def scale_states(self, x: np.ndarray) -> np.ndarray:
        return x / self.xscale

    def unscale_states(self, x: np.ndarray) -> np.ndarray:
        return x * self.xscale

    def _update_photoperiod_trackers(self, I1: float, I2: float):
        is_light_on = bool(I1 > 0.0 or I2 > 0.0)
        dt_hours = self.dt / 3600.0
        if is_light_on:
            self.hours_continuous_light += dt_hours
            self.hours_continuous_dark = 0.0
            self.daily_light_hours += dt_hours
            self.daily_DLI_dense += float(I1) * dt_hours * 3600.0 * 1e-6
            self.daily_DLI_finishing += float(I2) * dt_hours * 3600.0 * 1e-6
            self.daily_DLI = self._compute_area_weighted_daily_dli(
                self.daily_DLI_dense,
                self.daily_DLI_finishing,
            )
        else:
            self.hours_continuous_light = 0.0
            self.hours_continuous_dark += dt_hours
        self._was_light_on = is_light_on

    def _update_time_trackers(self):
        total_h = self._initial_hour_of_day + self._episode_hours_elapsed
        self.hour_of_day = int(total_h % 24)
        self.day_of_period = int(total_h // 24)

    def _get_day_night_blend(self, photo_idx: Optional[int] = None) -> float:
        pid_cfg = self.controller_params.get("pid_controller", {})
        ramp_h = max(float(pid_cfg.get("transition_ramp_hours", 1.0)), 1e-6)
        hour = (self._initial_hour_of_day + self._episode_hours_elapsed) % 24.0
        dawn = float(self.photo_period[1])
        dusk = dawn + float(self.photo_period[0])

        if dusk <= 24.0:
            if hour < dawn - ramp_h or hour >= dusk:
                return 0.0
            if dawn - ramp_h <= hour < dawn:
                return float(np.clip((hour - (dawn - ramp_h)) / ramp_h, 0.0, 1.0))
            if dawn <= hour < dusk - ramp_h:
                return 1.0
            return float(np.clip((dusk - hour) / ramp_h, 0.0, 1.0))

        dusk_wrapped = dusk - 24.0
        if hour >= dawn or hour < dusk_wrapped:
            if dawn <= hour < min(24.0, dawn + ramp_h):
                return float(np.clip((hour - dawn) / ramp_h, 0.0, 1.0))
            if max(0.0, dusk_wrapped - ramp_h) <= hour < dusk_wrapped:
                return float(np.clip((dusk_wrapped - hour) / ramp_h, 0.0, 1.0))
            return 1.0
        return 0.0

    def get_climate_targets(
        self,
        photo_idx: Optional[int] = None,
        T_current: Optional[float] = None,
    ) -> Dict[str, float]:
        if self.tvp is None or self.tvp.size == 0:
            raise RuntimeError("tvp is not initialised; call reset() first.")
        if photo_idx is None:
            photo_idx = int(self.time_step) % self._i_max
        if T_current is None:
            T_current = float(self.state[1])

        pid_cfg = self.controller_params.get("pid_controller", {})
        blend = self._get_day_night_blend(photo_idx)
        light_on = bool(self.tvp[photo_idx, 2])

        night_temp_sp = float(
            pid_cfg.get("temp_setpoint_night", 0.5 * sum(self.temp_range[0]))
        )
        day_temp_sp = float(
            pid_cfg.get("temp_setpoint_day", 0.5 * sum(self.temp_range[1]))
        )
        temp_sp = night_temp_sp + blend * (day_temp_sp - night_temp_sp)

        co2_day_ppm = float(
            pid_cfg.get("co2_setpoint_day_ppm", pid_cfg.get("co2_setpoint", 950.0))
        )
        co2_night_ppm = float(pid_cfg.get("co2_setpoint_night_ppm", 400.0))
        co2_sp_ppm = co2_night_ppm + blend * (co2_day_ppm - co2_night_ppm)
        co2_sp_density = co2_ppm_to_density(co2_sp_ppm, T_current)

        vpd_lo = self.vpd_range[0][0] + blend * (
            self.vpd_range[1][0] - self.vpd_range[0][0]
        )
        vpd_hi = self.vpd_range[0][1] + blend * (
            self.vpd_range[1][1] - self.vpd_range[0][1]
        )
        if vpd_hi < vpd_lo:
            vpd_lo, vpd_hi = vpd_hi, vpd_lo
        vpd_sp = 0.5 * (vpd_lo + vpd_hi)
        xH_lo = vpd_to_absolute_humidity(T_current, vpd_hi, self.container_params)
        xH_hi = vpd_to_absolute_humidity(T_current, vpd_lo, self.container_params)
        xH_sp = vpd_to_absolute_humidity(T_current, vpd_sp, self.container_params)

        return {
            "light_on": light_on,
            "blend": float(blend),
            "temp_setpoint_c": float(temp_sp),
            "temp_lo_c": float(self.tvp[photo_idx, 0]),
            "temp_hi_c": float(self.tvp[photo_idx, 1]),
            "co2_setpoint_ppm": float(co2_sp_ppm),
            "co2_setpoint_density": float(co2_sp_density),
            "vpd_lo_kpa": float(vpd_lo),
            "vpd_hi_kpa": float(vpd_hi),
            "vpd_target_kpa": float(vpd_sp),
            "xH_lo": float(min(xH_lo, xH_hi)),
            "xH_hi": float(max(xH_lo, xH_hi)),
            "xH_target": float(xH_sp),
        }

    def _build_schedule_sampler(self) -> ScheduleSampler:
        sp = self.schedule_params
        bounds = {
            "t1_min": int(sp.get("t1_min", 10)),
            "t1_max": int(sp.get("t1_max", 18)),
            "t2_min": int(sp.get("t2_min", 10)),
            "t2_max": int(sp.get("t2_max", 18)),
            "N1_min": int(sp.get("N1_min", 8)),
            "N1_max": int(sp.get("N1_max", 20)),
            "rho2_min": float(sp.get("rho2_min", 20.0)),
            "rho2_max": float(sp.get("rho2_max", 52.0)),
            "PP_fixed": int(resolve_fixed_photoperiod_hours(sp)),
            "rho1_min": float(sp.get("rho1_min", 72.0)),
            "rho1_max": float(sp.get("rho1_max", 144.0)),
            "N_total": int(sp.get("N_total", 80)),
            "er_min": float(sp.get("er_min", 3.0)),
            "er_max": float(sp.get("er_max", 6.0)),
            "total_cycle_min": float(sp.get("total_cycle_min", 24.0)),
            "total_cycle_max": float(sp.get("total_cycle_max", 32.0)),
            "DT_values": sp.get(
                "DT_values",
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
            ),
            "A_board": float(sp.get("A_board", 1.0)),
        }
        return ScheduleSampler(
            bounds,
            container_params=self.container_params,
            crop_params=self.crop_params,
            reward_params=self.reward_params,
            steady_state_params=self.config.get("steady_state_params", {}),
        )

    @staticmethod
    def _clone_reset_option_value(value: Any) -> Any:
        if isinstance(value, dict):
            return dict(value)
        if isinstance(value, np.ndarray):
            return value.copy()
        if isinstance(value, (list, tuple)):
            return list(value)
        return value

    def _update_persistent_reset_options(self, options: Dict[str, Any]) -> None:
        for key, value in (options or {}).items():
            if value is None:
                self._persistent_reset_options.pop(key, None)
            else:
                self._persistent_reset_options[key] = self._clone_reset_option_value(value)

    @staticmethod
    def _schedule_visit_key(schedule: Dict[str, Any]) -> str:
        return (
            f"t1={int(schedule['t1'])}|t2={int(schedule['t2'])}|"
            f"N1={int(schedule['N1'])}|rho2={int(round(float(schedule['rho2'])))}"
        )

    def _record_schedule_visit(
        self,
        schedule: Dict[str, Any],
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        options = options or {}
        self.total_resets += 1
        if options.get("sample_context", False):
            self.contextual_reset_count += 1
        key = self._schedule_visit_key(schedule)
        self.schedule_visit_counts[key] = int(self.schedule_visit_counts.get(key, 0)) + 1
        if key not in self.schedule_visit_meta:
            self.schedule_visit_meta[key] = {
                "t1": int(schedule["t1"]),
                "t2": int(schedule["t2"]),
                "N1": int(schedule["N1"]),
                "rho2": int(round(float(schedule["rho2"]))),
                "reference_feasibility_class": str(
                    schedule.get("reference_feasibility_class", "")
                ),
                "reference_min_feasible": bool(
                    schedule.get("reference_min_feasible", False)
                ),
                "reference_target_feasible": bool(
                    schedule.get("reference_target_feasible", False)
                ),
            }

    def _reset_episode_outcome_tracking(self) -> None:
        planned_steps = int(getattr(self, "episode_length", 0))
        self.last_termination_reason = "running"
        self.last_terminated = False
        self.last_truncated = False
        self.last_episode_actual_steps = 0
        self.last_episode_actual_days = 0.0
        self.last_episode_planned_steps = planned_steps
        self.last_episode_planned_days = float(planned_steps * self.dt / 86400.0)
        self.last_episode_completion_ratio = 0.0
        self.last_episode_ended_early = False
        self.last_ode_status = 0

    def _record_episode_end(
        self,
        termination_reason: str,
        terminated: bool,
        truncated: bool,
        status: int,
    ) -> None:
        actual_steps = int(self.time_step)
        planned_steps = int(max(self.episode_length, 1))
        actual_days = float(actual_steps * self.dt / 86400.0)
        planned_days = float(planned_steps * self.dt / 86400.0)
        completion_ratio = float(actual_steps / planned_steps)
        ended_early = bool((terminated or truncated) and actual_steps < planned_steps)
        reason = str(termination_reason or "unknown")

        self.completed_episode_count += 1
        if ended_early:
            self.early_termination_count += 1
        if terminated:
            self.failure_termination_count += 1
        if truncated:
            self.time_limit_completion_count += 1
        self.episode_completion_ratio_sum += completion_ratio
        self.termination_reason_counts[reason] = int(
            self.termination_reason_counts.get(reason, 0)
        ) + 1

        self.last_termination_reason = reason
        self.last_terminated = bool(terminated)
        self.last_truncated = bool(truncated)
        self.last_episode_actual_steps = actual_steps
        self.last_episode_actual_days = actual_days
        self.last_episode_planned_steps = planned_steps
        self.last_episode_planned_days = planned_days
        self.last_episode_completion_ratio = completion_ratio
        self.last_episode_ended_early = ended_early
        self.last_ode_status = int(status)
        episode_steps = max(int(self._episode_constraint_counts.get("steps", actual_steps)), 1)
        self.last_episode_constraint_cost_totals = dict(self._episode_constraint_cost_totals)
        self.last_episode_constraint_raw_totals = dict(self._episode_constraint_raw_totals)
        self.last_episode_constraint_counts = dict(self._episode_constraint_counts)
        self.last_episode_constraint_active_ratio = float(
            self._episode_constraint_counts.get("any", 0) / episode_steps
        )
        for key, value in self._episode_constraint_cost_totals.items():
            self.completed_constraint_cost_totals[key] = float(
                self.completed_constraint_cost_totals.get(key, 0.0)
            ) + float(value)
        for key, value in self._episode_constraint_raw_totals.items():
            self.completed_constraint_raw_totals[key] = float(
                self.completed_constraint_raw_totals.get(key, 0.0)
            ) + float(value)
        for key, value in self._episode_constraint_counts.items():
            self.completed_constraint_counts[key] = int(
                self.completed_constraint_counts.get(key, 0)
            ) + int(value)
        self.completed_constraint_active_ratio_sum += float(
            self.last_episode_constraint_active_ratio
        )

    def _resolve_episode_length(
        self,
        schedule: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> int:
        schedule = schedule or self.schedule
        options = options or self._persistent_reset_options

        explicit_steps = options.get("episode_length_steps", None)
        if explicit_steps is None:
            explicit_steps = self.config.get("_episode_length", None)

        if explicit_steps is not None:
            steps = max(1, int(round(float(explicit_steps))))
            self.episode_length_mode = "explicit_steps"
            self.episode_length_days = steps * self.dt / 86400.0
            return steps

        mode = str(
            options.get(
                "episode_length_mode",
                self.rl_params.get("episode_length_mode", "schedule_t2"),
            )
        ).lower()
        t1 = float(schedule.get("t1", self.t1))
        t2 = float(schedule.get("t2", self.t2))
        rng = getattr(self, "np_random", None)
        if rng is None:
            rng = np.random.default_rng(int(self.config.get("seed", 42)))

        if mode in {"mixed", "mixed_horizon", "mixed_episode", "curriculum"}:
            sampled = self._sample_episode_length_spec(
                options.get("episode_length_mix", self.rl_params.get("episode_length_mix")),
                rng=rng,
            )
            sampled_mode = str(sampled.get("mode", "schedule_t2")).lower()
            sampled_days = sampled.get("days", None)
            if sampled_days is not None:
                options = dict(options)
                options["episode_days"] = float(sampled_days)
            mode = sampled_mode
            requested_mode = f"mixed:{sampled_mode}"
        else:
            requested_mode = mode

        if mode in {"max_t2", "t2_max"}:
            days = float(self.schedule_params.get("t2_max", t2))
            resolved_mode = "max_t2"
        elif mode in {"fixed_days", "fixed"}:
            default_days = options.get("episode_days", self.rl_params.get("episode_days"))
            if default_days is None:
                default_days = self.schedule_params.get("t2_max", t2)
            days = float(default_days)
            resolved_mode = "fixed_days"
        elif mode in {"total_cycle", "schedule_cycle", "t1_t2"}:
            days = t1 + t2
            resolved_mode = "total_cycle"
        elif mode in {"max_total_cycle", "cycle_max"}:
            days = float(self.schedule_params.get("total_cycle_max", t1 + t2))
            resolved_mode = "max_total_cycle"
        else:
            days = t2
            resolved_mode = "schedule_t2"

        days = max(days, self.dt / 86400.0)
        steps = max(1, int(round(days * 86400.0 / self.dt)))
        self.episode_length_mode = resolved_mode
        self.episode_length_request_mode = str(requested_mode)
        self.episode_length_days = float(days)
        return steps

    @staticmethod
    def _normalise_episode_length_mix_entry(entry: Any) -> Optional[Dict[str, Any]]:
        if isinstance(entry, str):
            return {"mode": str(entry).lower(), "days": None, "weight": 1.0}
        if isinstance(entry, dict):
            mode = str(
                entry.get("mode", entry.get("episode_length_mode", "schedule_t2"))
            ).lower()
            days = entry.get("days", entry.get("episode_days"))
            weight = float(entry.get("weight", 1.0))
            return {"mode": mode, "days": days, "weight": max(weight, 0.0)}
        return None

    def _sample_episode_length_spec(
        self,
        mix_value: Any,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, Any]:
        if rng is None:
            rng = np.random.default_rng(int(self.config.get("seed", 42)))
        default_mix = [
            {"mode": "max_t2", "weight": 0.7},
            {"mode": "max_total_cycle", "weight": 0.3},
        ]
        entries_raw = mix_value if mix_value not in (None, "") else default_mix
        if isinstance(entries_raw, dict):
            entries_raw = [entries_raw]
        elif not isinstance(entries_raw, (list, tuple)):
            entries_raw = [entries_raw]

        parsed: list[Dict[str, Any]] = []
        weights = []
        for entry in entries_raw:
            item = self._normalise_episode_length_mix_entry(entry)
            if item is None:
                continue
            parsed.append(item)
            weights.append(max(float(item.get("weight", 1.0)), 0.0))

        if not parsed:
            parsed = [self._normalise_episode_length_mix_entry("max_t2")]
            weights = [1.0]

        weights_arr = np.asarray(weights, dtype=np.float64)
        if not np.all(np.isfinite(weights_arr)) or float(weights_arr.sum()) <= 0.0:
            weights_arr = np.ones(len(parsed), dtype=np.float64)
        probs = weights_arr / float(weights_arr.sum())
        choice = int(rng.choice(len(parsed), p=probs))
        return parsed[choice]

    @staticmethod
    def _normalise_context_sampling_strategy(strategy: Any) -> str:
        aliases = {
            "random": "random",
            "iid": "random",
            "uniform_random": "random",
            "cycle": "cycle",
            "coverage_cycle": "cycle",
            "round_robin": "cycle",
            "distributed_cycle": "distributed_cycle",
            "distributed_round_robin": "distributed_cycle",
            "coverage_sharded": "distributed_cycle",
        }
        text = str(strategy or "random").strip().lower()
        return aliases.get(text, "random")

    @staticmethod
    def _context_cycle_schedule_key(schedule: Dict[str, Any]) -> Tuple[int, int, int, int]:
        return (
            int(schedule["t1"]),
            int(schedule["t2"]),
            int(schedule["N1"]),
            int(round(float(schedule["rho2"]))),
        )

    def _sample_context_schedule_from_cycle(
        self,
        *,
        curriculum_phase: str,
        fixed_schedule: Optional[Dict[str, Any]],
        narrow_bounds: Optional[Dict[str, Any]],
        sampling_weights: Optional[Dict[str, Any]],
        options: Dict[str, Any],
    ) -> Dict[str, Any]:
        if sampling_weights:
            raise ValueError(
                "Coverage-aware context sampling does not support reference-class "
                "reweighting yet. Set train_context_sampling_reference_weights=null "
                "or switch to random sampling."
            )

        strategy = self._normalise_context_sampling_strategy(
            options.get(
                "context_sampling_strategy",
                self.rl_params.get("train_context_sampling_strategy", "random"),
            )
        )
        cycle_seed_base = int(
            options.get("context_sampling_cycle_seed", self.config.get("seed", 42))
        )
        cycle_rank = max(0, int(options.get("context_sampling_cycle_rank", 0)))
        cycle_world_size = max(1, int(options.get("context_sampling_cycle_world_size", 1)))

        candidates = self._schedule_sampler.get_candidate_pool(
            curriculum_phase=curriculum_phase,
            fixed_schedule=fixed_schedule,
            narrow_bounds=narrow_bounds,
            include_reference=False,
        )
        if not candidates:
            raise RuntimeError(
                f"No feasible contextual schedules available for phase '{curriculum_phase}'."
            )
        candidates = sorted(candidates, key=self._context_cycle_schedule_key)

        global_signature_payload = {
            "phase": str(curriculum_phase),
            "fixed_schedule": fixed_schedule,
            "narrow_bounds": narrow_bounds,
            "cycle_seed_base": int(cycle_seed_base),
            "candidate_count": int(len(candidates)),
        }
        global_signature = json.dumps(
            global_signature_payload,
            sort_keys=True,
            ensure_ascii=True,
        )
        local_signature_payload = {
            "global_signature": global_signature,
            "strategy": str(strategy),
            "cycle_rank": int(cycle_rank),
            "cycle_world_size": int(cycle_world_size),
        }
        local_signature = json.dumps(
            local_signature_payload,
            sort_keys=True,
            ensure_ascii=True,
        )
        state = self._context_sampling_cycle_state.get(local_signature)
        if state is None:
            digest = hashlib.blake2b(global_signature.encode("utf-8"), digest_size=8).digest()
            signature_seed = int.from_bytes(digest, "little", signed=False)
            local_rng = np.random.default_rng(signature_seed)
            permutation = local_rng.permutation(len(candidates)).tolist()
            if strategy == "distributed_cycle":
                local_order = permutation[cycle_rank::cycle_world_size]
                if not local_order:
                    local_order = permutation
            else:
                local_order = permutation
            state = {
                "order": local_order,
                "draw_count": 0,
            }
            self._context_sampling_cycle_state[local_signature] = state

        order = state["order"]
        draw_count = int(state["draw_count"])
        selected_idx = int(order[draw_count % len(order)])
        state["draw_count"] = int(draw_count + 1)
        return dict(candidates[selected_idx])

    def _sample_context_schedule(
        self,
        options: Optional[Dict[str, Any]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, Any]:
        options = options or {}
        curriculum_phase = str(
            options.get(
                "context_sampling_phase",
                options.get(
                    "curriculum_phase",
                    self.rl_params.get("context_sampling_phase", "full"),
                ),
            )
        )
        fixed_schedule = options.get(
            "fixed_schedule",
            self.rl_params.get("context_fixed_schedule"),
        )
        narrow_bounds = options.get(
            "narrow_bounds",
            self.rl_params.get("context_narrow_bounds"),
        )
        sampling_weights = {}
        reference_weights = options.get("context_sampling_reference_weights")
        if reference_weights is not None:
            sampling_weights["reference_class"] = dict(reference_weights)
        if not sampling_weights:
            sampling_weights = None
        strategy = self._normalise_context_sampling_strategy(
            options.get(
                "context_sampling_strategy",
                self.rl_params.get("train_context_sampling_strategy", "random"),
            )
        )
        if strategy in {"cycle", "distributed_cycle"}:
            return self._sample_context_schedule_from_cycle(
                curriculum_phase=curriculum_phase,
                fixed_schedule=fixed_schedule,
                narrow_bounds=narrow_bounds,
                sampling_weights=sampling_weights,
                options=options,
            )
        if rng is None:
            rng = getattr(self, "np_random", None)
        if rng is None:
            rng = np.random.default_rng(int(self.config.get("seed", 42)))
        return self._schedule_sampler.sample(
            curriculum_phase=curriculum_phase,
            fixed_schedule=fixed_schedule,
            narrow_bounds=narrow_bounds,
            sampling_weights=sampling_weights,
            rng=rng,
        )

    def _default_config(self) -> Dict[str, Any]:
        config_dir = str(Path(__file__).resolve().parents[2] / "configs")
        config = load_all_configs(config_dir)
        config.setdefault(
            "schedule",
            {"t1": 14, "t2": 14, "N1": 20, "rho2": 36.0},
        )
        config.setdefault("seed", 42)
        config.setdefault("dt", 600.0)
        return prepare_runtime_config(config)
