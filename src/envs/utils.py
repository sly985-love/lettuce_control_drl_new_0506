# -*- coding: utf-8 -*-
"""Utility helpers for simulator configuration and scaling."""

from __future__ import annotations

import os
from copy import deepcopy
from functools import lru_cache
from typing import Any, Dict, List, Optional

import numpy as np
import yaml


DEFAULT_SCHEDULE: Dict[str, Any] = {
    't1': 14,
    't2': 14,
    'N1': 20,
    'rho2': 36.0,
}

FIXED_PHOTOPERIOD_HOURS = 16

CONTEXTUAL_SCHEDULE_KEYS = ('t1', 't2', 'N1', 'rho2')

PFAL_DEFAULT_ECONOMICS: Dict[str, float] = {
    'electricity_price': 0.74,
    'co2_price': 0.54,
    'lettuce_price_fw': 40.0,
}

DEFAULT_INITIAL_SEEDLING_PRESETS: Dict[str, Dict[str, Any]] = {
    'strict_ref2': {
        'initial_seedling_mass_basis': 'per_plant_dry_mass',
        'initial_seedling_dry_mass_per_plant': 3.0e-5,
        'initial_seedling_mass': 3.0e-5,
    },
    'external_nursery_uniform': {
        'initial_seedling_mass_basis': 'per_plant_dry_mass',
        # Matched to the old 7.2e-3 kg/m^2 proxy at the canonical default
        # schedule x={14,14,20,36,16}, where rho1 = 108 plants/m^2.
        'initial_seedling_dry_mass_per_plant': 6.666666666666667e-5,
        'initial_seedling_mass': 6.666666666666667e-5,
    },
    'external_nursery_proxy': {
        'initial_seedling_mass_basis': 'area_dry_mass_density',
        'initial_seedling_area_dry_mass_density': 7.2e-3,
        'initial_seedling_mass': 7.2e-3,
    },
}

INLET_SEEDLING_METADATA_KEYS = (
    'initial_seedling_mass_preset',
    'initial_seedling_mass_basis',
    'initial_seedling_area_dry_mass_density',
    'initial_seedling_dry_mass_per_plant',
    'initial_seedling_mass',
    'seedling_nonstruct_ratio',
)

REWARD_PARAM_LEGACY_ECON_KEYS = {
    'price_growth',
    'price_energy',
    'price_CO2',
}

REWARD_PARAM_CANONICAL_ECON_KEYS = {
    'electricity_price',
    'co2_price',
    'lettuce_price_fw',
}

KNOWN_REWARD_PARAM_KEYS = {
    'alpha_growth',
    'reward_scale',
    'economic_reward_reference',
    'climate_penalty_reference',
    'daily_penalty_reference',
    'harvest_event_penalty_reference',
    'safety_override_reference',
    'thermal_meltdown_reference',
    'ode_failure_reference',
    'thermal_meltdown_penalty',
    'thermal_meltdown_temp_lo',
    'thermal_meltdown_temp_hi',
    'thermal_meltdown_hold_seconds',
    'thermal_meltdown_temp_lo_hard',
    'thermal_meltdown_temp_hi_hard',
    'temp_light_ideal_lo',
    'temp_light_ideal_hi',
    'temp_light_accept_lo',
    'temp_light_accept_hi',
    'temp_night_ideal_lo',
    'temp_night_ideal_hi',
    'temp_night_accept_lo',
    'temp_night_accept_hi',
    'temp_mild_penalty',
    'temp_severe_penalty',
    'rh_light_ideal_lo',
    'rh_light_ideal_hi',
    'rh_light_accept_lo',
    'rh_light_accept_hi',
    'rh_night_ideal_lo',
    'rh_night_ideal_hi',
    'rh_night_accept_lo',
    'rh_night_accept_hi',
    'rh_penalty',
    'co2_light_ideal_lo',
    'co2_light_ideal_hi',
    'co2_light_accept_lo',
    'co2_light_accept_hi',
    'co2_night_max',
    'co2_mild_penalty',
    'co2_severe_penalty',
    'dli_ideal_min',
    'dli_ideal_max',
    'dli_accept_min',
    'dli_accept_max',
    'dli_low_penalty',
    'dli_high_penalty',
    'daily_dli_penalty_mode',
    'photoperiod_ideal_min',
    'photoperiod_ideal_max',
    'photoperiod_accept_min',
    'photoperiod_accept_max',
    'photoperiod_low_penalty',
    'photoperiod_high_penalty',
    'daily_photoperiod_penalty_mode',
    'photoperiod_guard_tolerance_h',
    'harvest_fail_penalty_mode',
    'harvest_fail_penalty',
    'safety_override_base_penalty',
    'safety_override_q_hvac_penalty',
    'harvest_min_dry_mass_per_plant',
    'harvest_target_dry_mass_per_plant',
    'harvest_target_shaping_window_days',
    'harvest_target_shaping_gain',
    'transplant_loss_rate',
    'weight_growth',
    'weight_cost',
    'weight_penalty',
    'weight_harvest_target',
} | REWARD_PARAM_LEGACY_ECON_KEYS | REWARD_PARAM_CANONICAL_ECON_KEYS

KNOWN_CONTAINER_PARAM_KEYS = {
    'c_Length',
    'c_Width',
    'c_Height',
    'c_surface_area',
    'c_volume',
    'c_total_plant_area',
    'c_cap_q',
    'c_cap_q_v',
    'c_cap_q_p',
    'c_U',
    'c_lat_water',
    'c_v_0',
    'c_v_1',
    'c_v_2',
    'c_v_3',
    'mw_water',
    'c_R',
    'c_T_abs',
    'c_a_pl',
    'c_v_pl_ai',
    'ext_temp_summer',
    'ext_rh_summer',
    'ext_temp_winter',
    'ext_rh_winter',
    'ext_co2_summer',
    'ext_co2_winter',
    'photoperiod_on',
    'photoperiod_off',
    'temp_target_day',
    'temp_target_night',
    'C_day_ppm',
    'C_night_ppm',
    'RH_day',
    'RH_night',
    'I_standard',
    'I_target_seedling',
    'I_target_transplant',
    'T_standard',
    'C_standard',
    'C_standard_ppm',
    'RH_standard',
    'disturb_factor_max',
    'initial_seedling_mass_preset',
    'initial_seedling_mass_presets',
    'initial_seedling_mass_basis',
    'initial_seedling_area_dry_mass_density',
    'initial_seedling_dry_mass_per_plant',
    'seedling_nonstruct_ratio',
    'initial_seedling_mass',
    '_resolved_initial_seedling_mass_preset',
    'I_in_umol',
    'I_standard_umol',
    'dt_steady',
    'default_I1',
    'default_I2',
    'default_Q_HVAC',
    'default_u_CO2',
    'vent_leak_rate',
    'V_vent_fixed',
    'default_m_dehum',
    'default_total_E',
    'default_total_P',
    'default_C',
    'xD_min',
    'xD_max',
    'xDn_min',
    'xDn_max',
    'xDs_min',
    'xDs_max',
    't_state_clip_lo',
    't_state_clip_hi',
    'co2_step_delta_ppm_cap',
    'humidity_clip_saturation_ratio',
    'A1',
    'A2',
    '_A1',
    '_A2',
    '_A_total',
}

KNOWN_CROP_PARAM_KEYS = {
    'c_alpha',
    'c_beta',
    'c_Gamma',
    'c_k',
    'c_optical_eff',
    'c_lar_s',
    'c_par',
    'c_rad_rf',
    'c_eps',
    'c_Q10_resp',
    'c_resp_s',
    'c_resp_r',
    'c_r_gr_max',
    'c_Q10_gr',
    'c_Q10_Gamma',
    'c_tau',
    'c_pl_d',
    'c_rad_phot',
    'c_co2_1',
    'c_co2_2',
    'c_co2_3',
    'c_gamma',
    'c_resp_d',
    'c_resp_c',
    'c_alpha_beta',
    'ideal_density',
    'c_density',
    'c_ppfd_par',
    'c_led_eff',
    'c_bnd',
    'c_stm',
    'c_car_1',
    'c_car_2',
    'c_car_3',
    'c_fw',
    'c_d2f',
}

KNOWN_EQUIPMENT_PARAM_KEYS = {
    'c_led_eff',
    'c_led_ppe',
    'c_optical_eff',
    'lighting_constraint_mode',
    'dli_max_seedling_mol_m2_d',
    'dli_max_transplant_mol_m2_d',
    'led_indoor_heat_fraction',
    'dehum_latent_recovery_fraction',
    'dehum_electric_heat_fraction',
    'photo_chemical_energy_per_mol_co2_kj',
    'led_max_power_density_seedling',
    'led_max_power_density_transplant',
    'c_COP',
    'hvac_max_power_density',
    'hvac_min_power_density',
    'hvac_deadband',
    'co2_supply_max',
    'co2_supply_min',
    'p_CO2',
    'c_vent_fan_cap',
    'fan_eff',
    'vent_co2_deadband',
    'c_dehum_cap',
    'c_dehum_eev',
    'dehum_rh_deadband',
    'I1_min',
    'I1_max',
    'I2_min',
    'I2_max',
    'Q_HVAC_min',
    'Q_HVAC_max',
    'dehum_min',
    'dehum_max',
    'I_max',
    'min_on_off_time',
    'equipment_response_time',
    'p_elec_base',
    'c_elec',
    'p_elec_min',
    'p_elec_max',
    'elec_price_model',
    'p_dry_matter',
    'p_dry_matter_kg',
    'p_lettuce',
    'labor_cost',
    'c_CO2',
    'photo_period_min',
    'photo_period_max',
    'electricity_price',
    'co2_price',
    'lettuce_price_fw',
    'I1_agronomic_max',
    'I2_agronomic_max',
}

KNOWN_CONTROLLER_TOP_LEVEL_KEYS = {
    'rule_controller',
    'pid_controller',
    'action_limits',
}

KNOWN_RULE_CONTROLLER_KEYS = {
    'light_intensity_seedling',
    'light_intensity_transplant',
    'light_on_hour',
    'light_off_hour',
    'temp_setpoint',
    'kp_temp',
    'temp_deadband',
    'co2_setpoint',
    'co2_low_threshold',
    'co2_high_threshold',
    'co2_injection_rate',
    'vent_rate',
    'vent_high_threshold',
    'vent_trigger_co2',
    'vent_trigger_temp',
    'vent_trigger_by_temp',
    'dehum_rate',
    'dehum_high_rh',
    'dehum_mid_rh',
    'dehum_max_rate',
}

KNOWN_PID_CONTROLLER_KEYS = {
    'temp_setpoint_day',
    'temp_setpoint_night',
    'temp_band_day_lo',
    'temp_band_day_hi',
    'temp_band_night_lo',
    'temp_band_night_hi',
    'kp_temp',
    'ki_temp',
    'kd_temp',
    'temp_integral_limit',
    'co2_setpoint',
    'co2_setpoint_day_ppm',
    'co2_setpoint_night_ppm',
    'co2_band_day_lo_ppm',
    'co2_band_day_hi_ppm',
    'co2_band_night_lo_ppm',
    'co2_band_night_hi_ppm',
    'kp_co2',
    'ki_co2',
    'co2_integral_limit_ppm_h',
    'vpd_day_lo_kpa',
    'vpd_day_hi_kpa',
    'vpd_night_lo_kpa',
    'vpd_night_hi_kpa',
    'kp_vpd',
    'ki_vpd',
    'vpd_integral_limit_kpa_h',
    'temp_ff_gain',
    'co2_ff_gain',
    'dehum_ff_gain',
    'rate_limit_Q',
    'rate_limit_CO2',
    'rate_limit_dehum',
    'derivative_filter_alpha',
    'transition_ramp_hours',
}

KNOWN_ACTION_LIMIT_KEYS = {
    'I_max',
    'I_min',
    'Q_HVAC_max',
    'Q_HVAC_min',
    'co2_supply_max',
    'co2_supply_min',
    'vent_max',
    'vent_min',
    'dehum_max',
    'dehum_min',
}

KNOWN_EXPERIMENT_PARAM_KEYS = {
    'weather_data_dir',
    'tmy_data_file',
    'train_weather_dir',
    'weather_sample_interval',
    'weather_columns',
    'price_model_type',
    'tou_tariff_scenario',
    'time_of_use_periods',
    'time_of_use_prices',
    'price_scenarios',
    'constant_price',
    'elec_price_normalization',
    'simulation_step',
    'steps_per_year',
    'steps_per_day',
    'photoperiod',
    'global_seed',
    'n_experiment_repeats',
    'n_weather_repeats',
    'fixed_eval_schedules',
    'fixed_eval_weather',
    'ablation_experiments',
    'sensitivity_params',
    'results_root',
    'model_save_dir',
    'log_save_dir',
    'figure_save_dir',
    'data_save_dir',
    'save_detailed_trajectories',
    'trajectory_format',
    'rule_control',
    'mpc_baseline',
    'smpc_baseline',
}


@lru_cache(maxsize=None)
def _load_all_configs_cached(config_dir: str) -> Dict[str, Any]:
    config: Dict[str, Any] = {}
    config_files = [
        'crop_params.yaml',
        'container_params.yaml',
        'equipment_params.yaml',
        'reward_params.yaml',
        'schedule_params.yaml',
        'rl_params.yaml',
        'experiment_params.yaml',
        'mpc_params.yaml',
        'controller_params.yaml',
    ]

    for fname in config_files:
        fpath = os.path.join(config_dir, fname)
        if os.path.exists(fpath):
            with open(fpath, 'r', encoding='utf-8') as f:
                key = fname.replace('.yaml', '')
                config[key] = yaml.safe_load(f) or {}

    return config


def load_all_configs(config_dir: str) -> Dict[str, Any]:
    """Load YAML configs and return a mutable deep copy."""
    return deepcopy(_load_all_configs_cached(config_dir))


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_float(*values: Any, default: float) -> float:
    for value in values:
        coerced = _coerce_float(value)
        if coerced is not None:
            return coerced
    return float(default)


def _dedupe_preserve_order(values: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _append_unknown_key_warnings(
    warnings: List[str],
    group_name: str,
    params: Optional[Dict[str, Any]],
    known_keys: set[str],
) -> None:
    cfg = dict(params or {})
    unknown_keys = sorted(set(cfg.keys()) - set(known_keys))
    if unknown_keys:
        warnings.append(
            f"Unknown {group_name} keys detected: " + ", ".join(unknown_keys)
        )


def _numbers_conflict(a: Any, b: Any, *, rel_tol: float = 1e-6, abs_tol: float = 1e-9) -> bool:
    a_f = _coerce_float(a)
    b_f = _coerce_float(b)
    if a_f is None or b_f is None:
        return False
    return not bool(np.isclose(a_f, b_f, rtol=rel_tol, atol=abs_tol))


def _co2_density_from_ppm(ppm: float, T_celsius: float = 22.0) -> float:
    p_atm = 101325.0
    m_co2 = 44.01e-3
    r_gas = 8.314
    return float(ppm) * 1e-6 * m_co2 * p_atm / (r_gas * (float(T_celsius) + 273.15))


def _ventilation_rate_to_ach(vent_rate_m3_m2_s: float, grow_area_m2: float, volume_m3: float) -> float:
    grow_area_m2 = max(float(grow_area_m2), 1e-12)
    volume_m3 = max(float(volume_m3), 1e-12)
    return float(vent_rate_m3_m2_s) * grow_area_m2 / volume_m3 * 3600.0


def collect_reward_param_warnings(
    config: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Return human-readable warnings about reward-parameter compatibility."""
    cfg = config or {}
    reward_params = dict(cfg.get('reward_params', {}) or {})
    crop_params = dict(cfg.get('crop_params', {}) or {})
    economics = dict(cfg.get('economics', {}) or {})
    warnings: List[str] = []

    legacy_to_canonical = {
        'price_energy': 'electricity_price',
        'price_CO2': 'co2_price',
        'price_growth': 'lettuce_price_fw',
    }
    legacy_keys_present = sorted(
        key
        for key, canonical in legacy_to_canonical.items()
        if (
            key in reward_params
            and reward_params.get(key) is not None
            and canonical not in reward_params
            and canonical not in economics
        )
    )
    if legacy_keys_present:
        msg = (
            "reward_params legacy price keys detected "
            f"({', '.join(legacy_keys_present)}). "
            "Use reward_params.electricity_price / co2_price / lettuce_price_fw "
            "or config.economics.* instead."
        )
        if economics:
            msg += " Explicit economics.* values take precedence."
        warnings.append(msg)

    unknown_keys = sorted(set(reward_params.keys()) - KNOWN_REWARD_PARAM_KEYS)
    if unknown_keys:
        warnings.append(
            "Unknown reward_params keys detected: "
            + ", ".join(unknown_keys)
        )

    reference_keys = (
        'reward_scale',
        'economic_reward_reference',
        'climate_penalty_reference',
        'daily_penalty_reference',
        'harvest_event_penalty_reference',
        'safety_override_reference',
        'thermal_meltdown_reference',
        'ode_failure_reference',
    )
    non_positive_refs = [
        key for key in reference_keys
        if _coerce_float(reward_params.get(key)) is not None
        and float(reward_params.get(key)) <= 0.0
    ]
    if non_positive_refs:
        warnings.append(
            "reward_params contains non-positive scaling/reference keys: "
            + ", ".join(non_positive_refs)
        )

    negative_weight_keys = [
        key for key in (
            'weight_growth',
            'weight_cost',
            'weight_penalty',
            'weight_harvest_target',
        )
        if _coerce_float(reward_params.get(key)) is not None
        and float(reward_params.get(key)) < 0.0
    ]
    if negative_weight_keys:
        warnings.append(
            "reward_params contains negative top-level weights: "
            + ", ".join(negative_weight_keys)
        )

    thermal_penalty = abs(float(reward_params.get('thermal_meltdown_penalty', -100.0)))
    thermal_reference = _coerce_float(reward_params.get('thermal_meltdown_reference'))
    if (
        thermal_reference is not None
        and thermal_reference > 0.0
        and _numbers_conflict(thermal_reference, thermal_penalty, rel_tol=0.2, abs_tol=1.0)
    ):
        warnings.append(
            "reward_params.thermal_meltdown_reference differs materially from "
            "abs(thermal_meltdown_penalty). Constraint normalization and reward "
            "penalty strength may become inconsistent."
        )

    min_dry_mass = _coerce_float(reward_params.get('harvest_min_dry_mass_per_plant'))
    target_dry_mass = _coerce_float(
        reward_params.get(
            'harvest_target_dry_mass_per_plant',
            reward_params.get('harvest_min_dry_mass_per_plant'),
        )
    )
    c_fw = _coerce_float(crop_params.get('c_fw'))
    if (
        min_dry_mass is not None
        and target_dry_mass is not None
        and target_dry_mass + 1e-12 < min_dry_mass
    ):
        warnings.append(
            "reward_params.harvest_target_dry_mass_per_plant is smaller than "
            "harvest_min_dry_mass_per_plant. The target line should not be below "
            "the hard minimum pass line."
        )

    if min_dry_mass is not None and c_fw is not None and c_fw > 0.0:
        min_fw_g = min_dry_mass * c_fw
        if not np.isclose(min_fw_g, 100.0, rtol=0.12, atol=8.0):
            warnings.append(
                "reward_params.harvest_min_dry_mass_per_plant is inconsistent with "
                f"the default 100 g FW pass line implied by crop_params.c_fw "
                f"(current equivalent: {min_fw_g:.1f} g FW/plant)."
            )

    if target_dry_mass is not None and c_fw is not None and c_fw > 0.0:
        target_fw_g = target_dry_mass * c_fw
        if not np.isclose(target_fw_g, 120.0, rtol=0.1, atol=5.0):
            warnings.append(
                "reward_params.harvest_target_dry_mass_per_plant is inconsistent with "
                f"the 120 g FW target implied by crop_params.c_fw "
                f"(current equivalent: {target_fw_g:.1f} g FW/plant)."
            )

    return warnings


def collect_config_warnings(
    config: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Return cross-config warnings about unknown, legacy, or inconsistent keys."""
    cfg = config or {}
    warnings: List[str] = []

    container_params = _apply_initial_seedling_mass_preset(
        dict(cfg.get('container_params', {}) or {})
    )
    crop_params = dict(cfg.get('crop_params', {}) or {})
    equipment_params = dict(cfg.get('equipment_params', {}) or {})
    controller_params = dict(cfg.get('controller_params', {}) or {})
    experiment_params = dict(cfg.get('experiment_params', {}) or {})
    reward_params = dict(cfg.get('reward_params', {}) or {})
    economics = dict(cfg.get('economics', {}) or {})

    _append_unknown_key_warnings(
        warnings,
        'container_params',
        container_params,
        KNOWN_CONTAINER_PARAM_KEYS,
    )
    _append_unknown_key_warnings(
        warnings,
        'crop_params',
        crop_params,
        KNOWN_CROP_PARAM_KEYS,
    )
    _append_unknown_key_warnings(
        warnings,
        'equipment_params',
        equipment_params,
        KNOWN_EQUIPMENT_PARAM_KEYS,
    )
    _append_unknown_key_warnings(
        warnings,
        'controller_params',
        controller_params,
        KNOWN_CONTROLLER_TOP_LEVEL_KEYS,
    )
    _append_unknown_key_warnings(
        warnings,
        'experiment_params',
        experiment_params,
        KNOWN_EXPERIMENT_PARAM_KEYS,
    )

    rule_controller = dict(controller_params.get('rule_controller', {}) or {})
    pid_controller = dict(controller_params.get('pid_controller', {}) or {})
    action_limits = dict(controller_params.get('action_limits', {}) or {})

    _append_unknown_key_warnings(
        warnings,
        'controller_params.rule_controller',
        rule_controller,
        KNOWN_RULE_CONTROLLER_KEYS,
    )
    _append_unknown_key_warnings(
        warnings,
        'controller_params.pid_controller',
        pid_controller,
        KNOWN_PID_CONTROLLER_KEYS,
    )
    _append_unknown_key_warnings(
        warnings,
        'controller_params.action_limits',
        action_limits,
        KNOWN_ACTION_LIMIT_KEYS,
    )

    c_fw = _coerce_float(crop_params.get('c_fw'))
    c_d2f = _coerce_float(crop_params.get('c_d2f'))
    if c_fw is not None and c_fw > 0.0 and c_d2f is not None and c_d2f > 0.0:
        implied_c_d2f = 1.0 / c_fw
        if _numbers_conflict(c_d2f, implied_c_d2f, rel_tol=1e-3, abs_tol=1e-6):
            warnings.append(
                "crop_params.c_d2f is inconsistent with crop_params.c_fw. "
                f"Expected c_d2f ~= {implied_c_d2f:.12f} from c_fw = {c_fw:.6f}, "
                f"but found {c_d2f:.12f}. Reward economics and fresh-mass reporting "
                "may diverge."
            )

    if (
        'I_in_umol' in container_params
        and 'I_standard_umol' in container_params
        and bool(container_params['I_in_umol']) != bool(container_params['I_standard_umol'])
    ):
        warnings.append(
            "container_params.I_in_umol and container_params.I_standard_umol disagree. "
            "Steady-state initialisation and environment light-unit assumptions should match."
        )

    preset_name_raw = str(container_params.get('initial_seedling_mass_preset', '') or '').strip()
    if preset_name_raw:
        preset_table = {
            str(k): dict(v or {})
            for k, v in DEFAULT_INITIAL_SEEDLING_PRESETS.items()
        }
        preset_table.update(
            {
                str(k): dict(v or {})
                for k, v in dict(container_params.get('initial_seedling_mass_presets', {}) or {}).items()
            }
        )
        if preset_name_raw not in preset_table:
            warnings.append(
                "container_params.initial_seedling_mass_preset is not recognized. "
                "Use one of the built-in presets or define it under "
                "container_params.initial_seedling_mass_presets."
            )

    seedling_mass_basis = str(
        container_params.get('initial_seedling_mass_basis', 'area_dry_mass_density')
    ).strip().lower()
    legacy_xdi = 0.72e-3
    inlet_area_mass = _coerce_float(
        container_params.get(
            'initial_seedling_area_dry_mass_density',
            container_params.get('initial_seedling_mass'),
        )
    )
    inlet_per_plant_mass = _coerce_float(
        container_params.get('initial_seedling_dry_mass_per_plant')
    )
    per_plant_basis_aliases = {'per_plant', 'per_plant_mass', 'per_plant_dry_mass'}
    area_basis_aliases = {
        'area',
        'area_density',
        'area_dry_mass_density',
        'total_dry_mass_density',
        'kg_m2',
    }
    if seedling_mass_basis in per_plant_basis_aliases:
        if inlet_per_plant_mass is not None and np.isclose(
            inlet_per_plant_mass, legacy_xdi, rtol=0.05, atol=1e-6
        ):
            warnings.append(
                "container_params.initial_seedling_dry_mass_per_plant is set near "
                "0.72e-3 kg/plant, but PFAL-DRL's legacy xDi = 0.72e-3 was defined "
                "as an area dry-mass density [kg/m^2]. Recheck the inlet seedling "
                "mass units before using per-plant basis."
            )
    elif seedling_mass_basis in area_basis_aliases:
        if inlet_area_mass is not None and np.isclose(
            inlet_area_mass, legacy_xdi, rtol=0.05, atol=1e-6
        ):
            warnings.append(
                "container_params.initial_seedling_mass is set near the PFAL-DRL "
                "legacy xDi = 0.72e-3 kg/m^2. That value is likely too small for the "
                "current problem setting, where the 14-day nursery stage happens "
                "outside the container and only dense/final stages are modeled inside."
            )
        warnings.append(
            "container_params.initial_seedling_mass_basis uses area_dry_mass_density. "
            "Under the current 'uniform external nursery seedlings' assumption, this "
            "causes inlet dry mass per plant to vary with rho1 across schedules. "
            "Prefer per_plant_dry_mass for the current study."
        )
    else:
        warnings.append(
            "container_params.initial_seedling_mass_basis is not recognized. Use "
            "'area_dry_mass_density' or 'per_plant_dry_mass'."
        )

    if _numbers_conflict(
        crop_params.get('c_optical_eff'),
        equipment_params.get('c_optical_eff'),
    ):
        warnings.append(
            "crop_params.c_optical_eff and equipment_params.c_optical_eff differ. "
            "Light-to-photon conversion should use one shared value."
        )

    c_led_ppe = _coerce_float(equipment_params.get('c_led_ppe'))
    c_optical_eff = _first_float(
        equipment_params.get('c_optical_eff'),
        crop_params.get('c_optical_eff'),
        default=4.6,
    )
    c_led_eff_cfg = _coerce_float(equipment_params.get('c_led_eff'))
    if c_led_ppe is not None:
        if c_led_ppe <= 0.0:
            warnings.append(
                "equipment_params.c_led_ppe must be positive."
            )
        elif c_led_ppe > c_optical_eff:
            warnings.append(
                "equipment_params.c_led_ppe exceeds c_optical_eff, implying >100% "
                "LED radiant efficiency."
            )
        else:
            c_led_eff_derived = c_led_ppe / max(c_optical_eff, 1e-12)
            if c_led_eff_cfg is not None and _numbers_conflict(
                c_led_eff_cfg, c_led_eff_derived, rel_tol=1e-3, abs_tol=1e-4
            ):
                warnings.append(
                    "equipment_params.c_led_eff is inconsistent with "
                    "c_led_ppe / c_optical_eff."
                )

    led_indoor_heat_fraction = _coerce_float(
        equipment_params.get('led_indoor_heat_fraction')
    )
    if led_indoor_heat_fraction is not None:
        if led_indoor_heat_fraction < 0.0 or led_indoor_heat_fraction > 1.0:
            warnings.append(
                "equipment_params.led_indoor_heat_fraction must lie in [0, 1]."
            )

    photo_chemical_energy_per_mol_co2_kj = _coerce_float(
        equipment_params.get('photo_chemical_energy_per_mol_co2_kj')
    )
    if photo_chemical_energy_per_mol_co2_kj is not None:
        if photo_chemical_energy_per_mol_co2_kj <= 0.0:
            warnings.append(
                "equipment_params.photo_chemical_energy_per_mol_co2_kj must be positive."
            )
        elif photo_chemical_energy_per_mol_co2_kj < 300.0 or photo_chemical_energy_per_mol_co2_kj > 700.0:
            warnings.append(
                "equipment_params.photo_chemical_energy_per_mol_co2_kj is outside the "
                "usual stoichiometric range for carbohydrate-equivalent biomass formation."
            )

    vent_rate = _first_float(
        container_params.get('V_vent_fixed'),
        container_params.get('vent_leak_rate'),
        equipment_params.get('c_vent_fan_cap'),
        default=6.36e-5,
    )
    grow_area = _first_float(
        container_params.get('c_total_plant_area'),
        default=80.0,
    )
    volume = _first_float(
        container_params.get('c_volume'),
        default=91.5,
    )
    vent_ach = _ventilation_rate_to_ach(vent_rate, grow_area, volume)
    if _numbers_conflict(
        container_params.get('vent_leak_rate'),
        equipment_params.get('c_vent_fan_cap'),
    ):
        warnings.append(
            "container_params.vent_leak_rate differs from equipment_params.c_vent_fan_cap."
        )
    if vent_ach < 0.05 or vent_ach > 1.0:
        warnings.append(
            f"Fixed ventilation resolves to {vent_ach:.3f} ACH, which is outside the "
            "typical closed-container leakage range [0.05, 1.0] ACH."
        )

    co2_step_delta_ppm_cap = _coerce_float(container_params.get('co2_step_delta_ppm_cap'))
    if co2_step_delta_ppm_cap is not None:
        if co2_step_delta_ppm_cap <= 0.0:
            warnings.append(
                "container_params.co2_step_delta_ppm_cap disables the per-step CO2 rate limiter. "
                "Only use this after confirming the integrator remains numerically stable."
            )
        elif co2_step_delta_ppm_cap < 200.0:
            warnings.append(
                f"container_params.co2_step_delta_ppm_cap={co2_step_delta_ppm_cap:.1f} ppm is very tight "
                "and may suppress physically meaningful CO2 dynamics."
            )

    humidity_clip_ratio = _coerce_float(container_params.get('humidity_clip_saturation_ratio'))
    if humidity_clip_ratio is not None:
        if humidity_clip_ratio <= 0.0 or humidity_clip_ratio > 1.0:
            warnings.append(
                "container_params.humidity_clip_saturation_ratio must lie in (0, 1]."
            )
        elif humidity_clip_ratio < 0.9:
            warnings.append(
                f"container_params.humidity_clip_saturation_ratio={humidity_clip_ratio:.3f} is very low "
                "and may artificially dry the air state during post-step clipping."
            )

    co2_supply_max = _coerce_float(equipment_params.get('co2_supply_max'))
    if co2_supply_max is not None:
        co2_supply_g_m2_h = co2_supply_max * 1000.0 * 3600.0
        if co2_supply_g_m2_h > 10.0:
            warnings.append(
                f"equipment_params.co2_supply_max corresponds to {co2_supply_g_m2_h:.2f} g/m^2/h, "
                "which is very aggressive for a closed PFAL container and can dominate "
                "the indoor CO2 dynamics."
            )
        delta_c_target = max(
            _co2_density_from_ppm(1000.0, 22.0) - _co2_density_from_ppm(400.0, 22.0),
            0.0,
        )
        vent_hold_g_m2_h = vent_rate * delta_c_target * 1000.0 * 3600.0
        if co2_supply_g_m2_h < 0.5 * vent_hold_g_m2_h:
            warnings.append(
                f"equipment_params.co2_supply_max ({co2_supply_g_m2_h:.2f} g/m^2/h) is below "
                f"the ventilation-only hold requirement for 1000 ppm vs 400 ppm outdoors "
                f"({vent_hold_g_m2_h:.2f} g/m^2/h)."
            )

    dehum_latent_recovery_fraction = _coerce_float(
        equipment_params.get('dehum_latent_recovery_fraction')
    )
    if dehum_latent_recovery_fraction is not None:
        if dehum_latent_recovery_fraction < 0.0 or dehum_latent_recovery_fraction > 1.0:
            warnings.append(
                "equipment_params.dehum_latent_recovery_fraction must lie in [0, 1]."
            )

    dehum_electric_heat_fraction = _coerce_float(
        equipment_params.get('dehum_electric_heat_fraction')
    )
    if dehum_electric_heat_fraction is not None:
        if dehum_electric_heat_fraction < 0.0 or dehum_electric_heat_fraction > 1.0:
            warnings.append(
                "equipment_params.dehum_electric_heat_fraction must lie in [0, 1]."
            )

    canonical_elec = economics.get('electricity_price', reward_params.get('electricity_price'))
    canonical_co2 = economics.get('co2_price', reward_params.get('co2_price'))
    canonical_lettuce = economics.get('lettuce_price_fw', reward_params.get('lettuce_price_fw'))

    if _numbers_conflict(equipment_params.get('c_elec'), canonical_elec):
        warnings.append(
            "equipment_params.c_elec differs from canonical electricity price "
            "(economics / reward_params.electricity_price). Runtime reward uses the canonical value."
        )
    if _numbers_conflict(equipment_params.get('c_CO2'), canonical_co2):
        warnings.append(
            "equipment_params.c_CO2 differs from canonical CO2 price "
            "(economics / reward_params.co2_price). Runtime reward uses the canonical value."
        )
    if _numbers_conflict(equipment_params.get('p_lettuce'), canonical_lettuce):
        warnings.append(
            "equipment_params.p_lettuce differs from canonical lettuce price "
            "(economics / reward_params.lettuce_price_fw). Runtime reward uses the canonical value."
        )

    if _numbers_conflict(action_limits.get('I_max'), equipment_params.get('I_max')):
        warnings.append(
            "controller_params.action_limits.I_max differs from equipment_params.I_max."
        )
    if _numbers_conflict(action_limits.get('Q_HVAC_max'), equipment_params.get('hvac_max_power_density')):
        warnings.append(
            "controller_params.action_limits.Q_HVAC_max differs from equipment_params.hvac_max_power_density."
        )
    if _numbers_conflict(action_limits.get('Q_HVAC_min'), equipment_params.get('hvac_min_power_density')):
        warnings.append(
            "controller_params.action_limits.Q_HVAC_min differs from equipment_params.hvac_min_power_density."
        )
    if _numbers_conflict(action_limits.get('co2_supply_max'), equipment_params.get('co2_supply_max')):
        warnings.append(
            "controller_params.action_limits.co2_supply_max differs from equipment_params.co2_supply_max."
        )
    if _numbers_conflict(action_limits.get('dehum_max'), equipment_params.get('c_dehum_cap')):
        warnings.append(
            "controller_params.action_limits.dehum_max differs from equipment_params.c_dehum_cap."
        )

    if 'vent_min' in action_limits or 'vent_max' in action_limits:
        warnings.append(
            "controller_params.action_limits.vent_min / vent_max are legacy and ignored by the current 5D action space."
        )
    if rule_controller:
        warnings.append(
            "controller_params.rule_controller is not used by the current PFALConventionalController / RL path."
        )

    lighting_constraint_mode = str(
        equipment_params.get('lighting_constraint_mode', 'hardware_only') or 'hardware_only'
    ).strip().lower()
    valid_lighting_modes = {'hardware_only', 'static_agronomic', 'dli_aware_agronomic'}
    if lighting_constraint_mode not in valid_lighting_modes:
        warnings.append(
            "equipment_params.lighting_constraint_mode is unknown. "
            "Expected hardware_only, static_agronomic, or dli_aware_agronomic."
        )
    if lighting_constraint_mode == 'static_agronomic':
        if _coerce_float(equipment_params.get('I1_agronomic_max')) is None:
            warnings.append(
                "static_agronomic lighting mode is enabled but equipment_params.I1_agronomic_max is missing."
            )
        if _coerce_float(equipment_params.get('I2_agronomic_max')) is None:
            warnings.append(
                "static_agronomic lighting mode is enabled but equipment_params.I2_agronomic_max is missing."
            )
    if lighting_constraint_mode == 'dli_aware_agronomic':
        if _coerce_float(equipment_params.get('dli_max_seedling_mol_m2_d')) is None:
            warnings.append(
                "dli_aware_agronomic lighting mode is enabled but equipment_params.dli_max_seedling_mol_m2_d is missing."
            )
        if _coerce_float(equipment_params.get('dli_max_transplant_mol_m2_d')) is None:
            warnings.append(
                "dli_aware_agronomic lighting mode is enabled but equipment_params.dli_max_transplant_mol_m2_d is missing."
            )

    simulation_step = _coerce_float(experiment_params.get('simulation_step'))
    runtime_dt = _coerce_float(cfg.get('dt'))
    if _numbers_conflict(simulation_step, runtime_dt):
        warnings.append(
            "experiment_params.simulation_step differs from the runtime dt. "
            "The current simulator / RL stack is not using the same default step size as experiment_params."
        )

    fixed_eval_schedules = experiment_params.get('fixed_eval_schedules', []) or []
    missing_contextual_fields: List[str] = []
    for idx, item in enumerate(fixed_eval_schedules, start=1):
        if not isinstance(item, dict):
            continue
        missing = [key for key in CONTEXTUAL_SCHEDULE_KEYS if key not in item]
        if missing:
            item_name = str(item.get('name', f'index_{idx}'))
            missing_contextual_fields.append(
                f"{item_name}: missing " + ", ".join(missing)
            )
    if missing_contextual_fields:
        warnings.append(
            "experiment_params.fixed_eval_schedules must use the current "
            "contextual schedule fields {t1, t2, N1, rho2}. "
            + "; ".join(missing_contextual_fields)
        )
    if any(
        isinstance(item, dict) and 'A1_A2' in item and 'N1' not in item
        for item in fixed_eval_schedules
    ):
        warnings.append(
            "experiment_params.fixed_eval_schedules uses legacy A1_A2 coordinates. "
            "The current contextual schedule representation is based on {t1, t2, N1, rho2}."
        )

    return _dedupe_preserve_order(warnings)


def resolve_fixed_photoperiod_hours(
    schedule_params: Optional[Dict[str, Any]] = None,
    schedule: Optional[Dict[str, Any]] = None,
) -> int:
    """Resolve the fixed runtime photoperiod used by the 0420 fixed-PP project."""
    sp = dict(schedule_params or {})
    sched = dict(schedule or {})
    if sched.get('PP') is not None:
        return int(sched['PP'])
    if sp.get('PP_fixed') is not None:
        return int(sp['PP_fixed'])
    if sp.get('PP_min') is not None:
        return int(sp['PP_min'])
    if sp.get('PP_max') is not None:
        return int(sp['PP_max'])
    return int(FIXED_PHOTOPERIOD_HOURS)


def complete_schedule(
    schedule: Optional[Dict[str, Any]] = None,
    schedule_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Fill in derived schedule fields used by the runtime simulator."""
    sp = dict(schedule_params or {})
    sched = dict(DEFAULT_SCHEDULE)
    if schedule:
        sched.update(schedule)

    n_total = int(sp.get('N_total', 80))
    a_board = float(sp.get('A_board', 1.0))
    n1 = int(sched.get('N1', DEFAULT_SCHEDULE['N1']))
    n2 = int(sched.get('N2', n_total - n1))
    t1 = int(sched.get('t1', DEFAULT_SCHEDULE['t1']))
    t2 = int(sched.get('t2', DEFAULT_SCHEDULE['t2']))
    rho2 = float(sched.get('rho2', DEFAULT_SCHEDULE['rho2']))
    pp = int(resolve_fixed_photoperiod_hours(sp, sched))
    a1 = float(sched.get('A1', n1 * a_board))
    a2 = float(sched.get('A2', n2 * a_board))
    a_total = float(sched.get('A_total', a1 + a2))

    if n1 > 0 and t2 > 0:
        rho1_default = rho2 * n2 * t1 / (n1 * t2)
    else:
        rho1_default = 0.0

    sched.update({
        't1': t1,
        't2': t2,
        'N1': n1,
        'N2': n2,
        'rho2': rho2,
        'rho1': float(sched.get('rho1', rho1_default)),
        'A1': a1,
        'A2': a2,
        'A_total': a_total,
        'PP': int(pp),
    })
    return sched


def resolve_simulator_economics(
    config: Optional[Dict[str, Any]] = None,
    equipment_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    Resolve the canonical economics used by the simulator runtime.

    Notes:
    - `reward_params.yaml` may carry RL/MPC-specific scaling and optional
      canonical price overrides:
      `electricity_price`, `co2_price`, `lettuce_price_fw`.
    - Runtime simulator statistics and PFAL-style reward should use one
      shared source to avoid script/env inconsistencies.
    """
    cfg = config or {}
    ep = dict(equipment_params or cfg.get('equipment_params', {}) or {})
    rp = dict(cfg.get('reward_params', {}) or {})
    economics = dict(cfg.get('economics', {}) or {})

    electricity_price = _first_float(
        economics.get('electricity_price'),
        rp.get('electricity_price'),
        ep.get('c_elec'),
        ep.get('electricity_price'),
        default=PFAL_DEFAULT_ECONOMICS['electricity_price'],
    )
    co2_price = _first_float(
        economics.get('co2_price'),
        rp.get('co2_price'),
        ep.get('c_CO2'),
        ep.get('co2_price'),
        default=PFAL_DEFAULT_ECONOMICS['co2_price'],
    )

    lettuce_price_fw = _first_float(
        economics.get('lettuce_price_fw'),
        rp.get('lettuce_price_fw'),
        ep.get('p_lettuce'),
        ep.get('lettuce_price_fw'),
        default=PFAL_DEFAULT_ECONOMICS['lettuce_price_fw'],
    )

    return {
        'electricity_price': electricity_price,
        'co2_price': co2_price,
        'lettuce_price_fw': lettuce_price_fw,
    }


def merge_runtime_params(
    container_params: Optional[Dict[str, Any]] = None,
    crop_params: Optional[Dict[str, Any]] = None,
    equipment_params: Optional[Dict[str, Any]] = None,
    schedule: Optional[Dict[str, Any]] = None,
    photo_period_override: Optional[int] = None,
) -> Dict[str, Any]:
    """Build the runtime parameter dict consumed by environment and crops."""
    merged: Dict[str, Any] = dict(container_params or {})
    merged.update(dict(crop_params or {}))
    merged.update(dict(equipment_params or {}))
    merged = _apply_initial_seedling_mass_preset(merged)

    sched = dict(schedule or {})
    a1 = float(sched.get('A1', merged.get('_A1', merged.get('A1', 20.0))))
    a2 = float(sched.get('A2', merged.get('_A2', merged.get('A2', 60.0))))
    a_total = float(sched.get('A_total', a1 + a2))

    merged['_A1'] = a1
    merged['_A2'] = a2
    merged['_A_total'] = a_total
    merged['A1'] = a1
    merged['A2'] = a2
    merged['c_total_plant_area'] = a_total
    merged['V_vent_fixed'] = float(
        merged.get('V_vent_fixed', merged.get('vent_leak_rate', 6.36e-5))
    )
    merged['I_target_seedling'] = float(merged.get('I_target_seedling', 200.0))
    merged['I_target_transplant'] = float(merged.get('I_target_transplant', 300.0))
    i_umol_flag = bool(merged.get('I_standard_umol', merged.get('I_in_umol', True)))
    merged['I_standard_umol'] = i_umol_flag
    merged['I_in_umol'] = i_umol_flag

    if photo_period_override is not None:
        merged['photoperiod_on'] = int(photo_period_override)
        merged['photoperiod_off'] = max(0, 24 - int(photo_period_override))

    return merged


def _apply_initial_seedling_mass_preset(
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    resolved = dict(params or {})
    preset_name = str(resolved.get('initial_seedling_mass_preset', '') or '').strip()
    preset_table: Dict[str, Dict[str, Any]] = {
        str(k): dict(v or {})
        for k, v in DEFAULT_INITIAL_SEEDLING_PRESETS.items()
    }
    preset_table.update(
        {
            str(k): dict(v or {})
            for k, v in dict(resolved.get('initial_seedling_mass_presets', {}) or {}).items()
        }
    )

    if not preset_name:
        preset_name = 'external_nursery_uniform'

    preset = dict(preset_table.get(preset_name, {}) or {})
    if not preset:
        return resolved

    for key, value in preset.items():
        if key == 'description':
            continue
        resolved[key] = value
    resolved['_resolved_initial_seedling_mass_preset'] = preset_name
    return resolved


def build_steady_state_params(
    container_params: Dict[str, Any],
    schedule: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build steady-state initialization parameters from runtime params."""
    cp = dict(container_params or {})
    sched = dict(schedule or {})
    pp = int(cp.get('photoperiod_on', sched.get('PP', 16)))
    i_umol_flag = bool(cp.get('I_standard_umol', cp.get('I_in_umol', True)))
    return {
        'I_standard': cp.get('I_standard', 200.0),
        'T_standard': cp.get('T_standard', 22.0),
        'C_standard_ppm': cp.get('C_standard_ppm', 1000.0),
        'RH_standard': cp.get('RH_standard', 0.75),
        'dt': cp.get('dt_steady', 600.0),
        'disturb_factor_max': cp.get('disturb_factor_max', 0.05),
        'initial_seedling_mass_preset': cp.get(
            'initial_seedling_mass_preset',
            cp.get('_resolved_initial_seedling_mass_preset', 'external_nursery_uniform'),
        ),
        '_resolved_initial_seedling_mass_preset': cp.get(
            '_resolved_initial_seedling_mass_preset',
            cp.get('initial_seedling_mass_preset', 'external_nursery_uniform'),
        ),
        'initial_seedling_mass_basis': cp.get(
            'initial_seedling_mass_basis', 'per_plant_dry_mass'
        ),
        'initial_seedling_area_dry_mass_density': cp.get(
            'initial_seedling_area_dry_mass_density',
            cp.get('initial_seedling_mass', 7.2e-3),
        ),
        'initial_seedling_dry_mass_per_plant': cp.get(
            'initial_seedling_dry_mass_per_plant',
            6.666666666666667e-5,
        ),
        'seedling_nonstruct_ratio': cp.get('seedling_nonstruct_ratio', 0.25),
        'initial_seedling_mass': cp.get('initial_seedling_mass', 6.666666666666667e-5),
        'I_standard_umol': i_umol_flag,
        'photoperiod_on': pp,
        'photoperiod_off': max(0, 24 - pp),
        'temp_target_day': cp.get('temp_target_day', 22.0),
        'temp_target_night': cp.get('temp_target_night', 18.0),
        'C_day_ppm': cp.get('C_day_ppm', cp.get('C_standard_ppm', 1000.0)),
        'C_night_ppm': cp.get('C_night_ppm', 800.0),
    }


def prepare_runtime_config(
    config: Optional[Dict[str, Any]] = None,
    schedule: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    dt: Optional[float] = None,
    photo_period_override: Optional[int] = None,
) -> Dict[str, Any]:
    """Normalize raw config dictionaries into one runtime-ready config."""
    runtime = deepcopy(config or {})
    schedule_params = dict(runtime.get('schedule_params', {}) or {})
    runtime_schedule = complete_schedule(
        schedule or runtime.get('schedule'),
        schedule_params,
    )

    runtime['schedule'] = runtime_schedule
    runtime['seed'] = int(seed if seed is not None else runtime.get('seed', 42))
    runtime['dt'] = float(dt if dt is not None else runtime.get('dt', 600.0))

    runtime['crop_params'] = dict(runtime.get('crop_params', {}) or {})
    runtime['equipment_params'] = dict(runtime.get('equipment_params', {}) or {})
    runtime['reward_params'] = dict(runtime.get('reward_params', {}) or {})
    runtime['controller_params'] = dict(runtime.get('controller_params', {}) or {})
    runtime['container_params'] = _apply_initial_seedling_mass_preset(
        dict(runtime.get('container_params', {}) or {})
    )
    runtime['schedule_params'] = schedule_params
    runtime['config_warnings'] = list(runtime.get('config_warnings', []) or [])
    runtime['config_warnings'] = _dedupe_preserve_order(
        list(runtime['config_warnings']) + collect_config_warnings(runtime)
    )

    runtime['container_params'] = merge_runtime_params(
        runtime['container_params'],
        runtime['crop_params'],
        runtime['equipment_params'],
        runtime_schedule,
        photo_period_override=photo_period_override,
    )
    runtime['steady_state_params'] = build_steady_state_params(
        runtime['container_params'],
        runtime_schedule,
    )
    runtime['reward_param_warnings'] = collect_reward_param_warnings(runtime)
    runtime['economics'] = resolve_simulator_economics(runtime)
    runtime['config_warnings'] = _dedupe_preserve_order(
        list(runtime['config_warnings']) + list(runtime['reward_param_warnings'])
    )

    if photo_period_override is not None:
        runtime['photo_period_override'] = int(photo_period_override)

    return runtime


def has_inlet_seedling_metadata(
    payload: Optional[Dict[str, Any]] = None,
) -> bool:
    """Return True when flat or nested inlet seedling metadata is present."""
    data = dict(payload or {})
    nested = data.get('inlet_seedling_metadata')
    if isinstance(nested, dict):
        for key in INLET_SEEDLING_METADATA_KEYS:
            if nested.get(key) is not None:
                return True
    for key in INLET_SEEDLING_METADATA_KEYS:
        if data.get(key) is not None:
            return True
    return False


def apply_inlet_seedling_metadata(
    config: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Inject saved inlet-seedling metadata back into container params."""
    resolved = deepcopy(config or {})
    if not metadata:
        return resolved

    container_params = dict(resolved.get('container_params', {}) or {})
    candidate_sources: List[Dict[str, Any]] = []
    nested = metadata.get('inlet_seedling_metadata')
    if isinstance(nested, dict):
        candidate_sources.append(dict(nested))
    candidate_sources.append(dict(metadata or {}))

    for source in candidate_sources:
        preset_name = str(
            source.get(
                'initial_seedling_mass_preset',
                source.get(
                    'resolved_initial_seedling_mass_preset',
                    source.get('_resolved_initial_seedling_mass_preset', ''),
                ),
            )
            or ''
        ).strip()
        if preset_name:
            container_params['initial_seedling_mass_preset'] = preset_name
        for key in INLET_SEEDLING_METADATA_KEYS:
            if key == 'initial_seedling_mass_preset':
                continue
            if key in source and source.get(key) is not None:
                container_params[key] = source[key]

    if container_params:
        resolved['container_params'] = container_params
    return resolved


def _filter_inlet_seedling_warnings(
    warnings: Optional[List[str]] = None,
) -> List[str]:
    filtered: List[str] = []
    for warning in list(warnings or []):
        text = str(warning)
        lowered = text.lower()
        if (
            'initial_seedling' in lowered
            or 'uniform external nursery' in lowered
            or 'seedling mass' in lowered
            or 'legacy xdi' in lowered
        ):
            filtered.append(text)
    return _dedupe_preserve_order(filtered)


def extract_inlet_seedling_metadata(
    config: Optional[Dict[str, Any]] = None,
    *,
    assume_runtime: bool = False,
) -> Dict[str, Any]:
    """Extract resolved inlet-seedling metadata from raw or runtime config."""
    runtime = deepcopy(config or {})
    if not assume_runtime:
        runtime = prepare_runtime_config(runtime)

    container_params = _apply_initial_seedling_mass_preset(
        dict(runtime.get('container_params', {}) or {})
    )
    steady_state_params = dict(runtime.get('steady_state_params', {}) or {})

    resolved_preset = str(
        container_params.get(
            '_resolved_initial_seedling_mass_preset',
            steady_state_params.get(
                '_resolved_initial_seedling_mass_preset',
                container_params.get(
                    'initial_seedling_mass_preset',
                    steady_state_params.get('initial_seedling_mass_preset', ''),
                ),
            ),
        )
        or ''
    ).strip()
    basis = str(
        container_params.get(
            'initial_seedling_mass_basis',
            steady_state_params.get('initial_seedling_mass_basis', ''),
        )
        or ''
    ).strip().lower()

    metadata_body = {
        'initial_seedling_mass_preset': resolved_preset,
        'resolved_initial_seedling_mass_preset': resolved_preset,
        'initial_seedling_mass_basis': basis,
        'initial_seedling_area_dry_mass_density': _first_float(
            container_params.get('initial_seedling_area_dry_mass_density'),
            steady_state_params.get('initial_seedling_area_dry_mass_density'),
            default=0.0,
        ),
        'initial_seedling_dry_mass_per_plant': _first_float(
            container_params.get('initial_seedling_dry_mass_per_plant'),
            steady_state_params.get('initial_seedling_dry_mass_per_plant'),
            default=0.0,
        ),
        'initial_seedling_mass': _first_float(
            container_params.get('initial_seedling_mass'),
            steady_state_params.get('initial_seedling_mass'),
            default=0.0,
        ),
        'seedling_nonstruct_ratio': _first_float(
            container_params.get('seedling_nonstruct_ratio'),
            steady_state_params.get('seedling_nonstruct_ratio'),
            default=0.25,
        ),
    }
    return {
        **metadata_body,
        'inlet_seedling_metadata': dict(metadata_body),
        'inlet_seedling_config_warnings': _filter_inlet_seedling_warnings(
            runtime.get('config_warnings', [])
        ),
    }


def create_default_schedule(
    schedule_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create the canonical default schedule clipped to the current bounds."""
    if schedule_params is None:
        schedule_params = {}

    t1_min = int(schedule_params.get('t1_min', 10))
    t1_max = int(schedule_params.get('t1_max', 18))
    t2_min = int(schedule_params.get('t2_min', 10))
    t2_max = int(schedule_params.get('t2_max', 18))
    n1_min = int(schedule_params.get('N1_min', 8))
    n1_max = int(schedule_params.get('N1_max', 20))
    rho2_min = float(schedule_params.get('rho2_min', 20.0))
    rho2_max = float(schedule_params.get('rho2_max', 52.0))

    def _clip_int(value: Any, lo: int, hi: int) -> int:
        return int(min(max(int(value), lo), hi))

    def _clip_float(value: Any, lo: float, hi: float) -> float:
        return float(min(max(float(value), lo), hi))

    return {
        't1': _clip_int(DEFAULT_SCHEDULE['t1'], t1_min, t1_max),
        't2': _clip_int(DEFAULT_SCHEDULE['t2'], t2_min, t2_max),
        'N1': _clip_int(DEFAULT_SCHEDULE['N1'], n1_min, n1_max),
        'rho2': _clip_float(DEFAULT_SCHEDULE['rho2'], rho2_min, rho2_max),
    }


def normalize_observation(
    obs: np.ndarray,
    obs_low: np.ndarray,
    obs_high: np.ndarray,
    method: str = 'linear'
) -> np.ndarray:
    """Normalize observations."""
    if method == 'linear':
        return 2.0 * (obs - obs_low) / (obs_high - obs_low + 1e-8) - 1.0
    if method == 'minmax':
        return (obs - obs_low) / (obs_high - obs_low + 1e-8)
    return obs


def denormalize_action(
    action_norm: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray
) -> np.ndarray:
    """Map normalized actions in [-1, 1] back to physical units."""
    return (action_norm + 1.0) / 2.0 * (action_high - action_low) + action_low


def get_action_bounds(
    equipment_params: Optional[Dict[str, Any]] = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return physical action bounds used across controllers and scripts."""
    if equipment_params is None:
        equipment_params = {}

    i1_min = float(equipment_params.get('I1_min', equipment_params.get('I_min', 0.0)))
    i1_max = float(equipment_params.get('I1_max', equipment_params.get('I_max', 400.0)))
    i2_min = float(equipment_params.get('I2_min', equipment_params.get('I_min', 0.0)))
    i2_max = float(equipment_params.get('I2_max', equipment_params.get('I_max', 400.0)))
    q_hvac_max = float(equipment_params.get('hvac_max_power_density', 212.0))
    q_hvac_min = float(equipment_params.get('hvac_min_power_density', -212.0))
    co2_supply_max = float(equipment_params.get('co2_supply_max', 1.0e-6))
    dehum_max = float(equipment_params.get('c_dehum_cap', 2.083e-5))

    action_low = np.array([
        i1_min,
        i2_min,
        q_hvac_min,
        0.0,
        0.0,
    ], dtype=np.float32)

    action_high = np.array([
        i1_max,
        i2_max,
        q_hvac_max,
        co2_supply_max,
        dehum_max,
    ], dtype=np.float32)

    return action_low, action_high
