# -*- coding: utf-8 -*-
"""
PI/PID baseline controller for the dual-zone PFAL simulator.

This controller is intended to be a credible engineering baseline:
1. Temperature: PID with model-consistent feedforward, dawn/dusk ramping,
   derivative-on-measurement damping, and physical slew-rate limits.
2. CO2: daytime PI with net-canopy feedforward and zero night injection.
3. Humidity: VPD-scheduled dehumidification control. The controller tracks a
   ramped VPD target, converts it to absolute humidity, and applies PI plus a
   mass-balance feedforward term.

All controller outputs are returned in the environment's expected [-1, 1]
normalised action space.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import yaml

from .base_controller import BaseController

# Support both package layouts used in this repo:
# 1) `src.controllers...` with relative imports
# 2) `controllers...` after `src` is added to `sys.path`
try:
    from ..models import (
        absolute_humidity_to_vpd,
        calculate_dehum_heat_rejection,
        calculate_led_room_heat_gain,
        co2_density_to_ppm,
    )
except ImportError:  # pragma: no cover - runtime compatibility fallback
    from models import (
        absolute_humidity_to_vpd,
        calculate_dehum_heat_rejection,
        calculate_led_room_heat_gain,
        co2_density_to_ppm,
    )


@dataclass
class PIDChannel:
    """Simple PID/PI channel with anti-windup and derivative filtering."""

    kp: float
    ki: float
    kd: float
    output_min: float
    output_max: float
    integral_limit: float
    derivative_alpha: float = 0.25

    def __post_init__(self) -> None:
        self.integral = 0.0
        self.filtered_rate = 0.0

    def reset(self) -> None:
        self.integral = 0.0
        self.filtered_rate = 0.0

    def update(
        self,
        error: float,
        dt_hours: float,
        measurement_rate: float = 0.0,
        feedforward: float = 0.0,
    ) -> float:
        """Update the controller and return the saturated physical output."""
        dt_hours = max(float(dt_hours), 1e-9)
        alpha = float(np.clip(self.derivative_alpha, 0.0, 1.0))
        self.filtered_rate = alpha * measurement_rate + (1.0 - alpha) * self.filtered_rate

        candidate_integral = float(np.clip(
            self.integral + error * dt_hours,
            -self.integral_limit,
            self.integral_limit,
        ))

        u_unsat = (
            feedforward
            + self.kp * error
            + self.ki * candidate_integral
            - self.kd * self.filtered_rate
        )
        u_sat = float(np.clip(u_unsat, self.output_min, self.output_max))

        saturated_high = np.isclose(u_sat, self.output_max) and error > 0.0
        saturated_low = np.isclose(u_sat, self.output_min) and error < 0.0
        if not (saturated_high or saturated_low):
            self.integral = candidate_integral

        return u_sat


def _load_controller_params(env) -> Dict:
    """Load controller parameters from the env config, with a YAML fallback."""
    cfg = getattr(env, 'controller_params', None)
    if isinstance(cfg, dict) and cfg:
        return cfg

    cfg_path = Path(__file__).resolve().parents[2] / 'configs' / 'controller_params.yaml'
    if cfg_path.exists():
        with open(cfg_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    return {}


class PFALConventionalController(BaseController):
    """
    PI/PID engineering baseline controller.

    The class name is preserved for compatibility with existing scripts, but
    this implementation is no longer the old proportional band controller.
    """

    def _refresh_env_dependent_state(self) -> None:
        action_low, action_high = self.env._get_action_physical_bounds()
        self._action_low = action_low.astype(np.float64)
        self._action_high = action_high.astype(np.float64)
        self._dt_hours = max(float(self.env.dt) / 3600.0, 1e-9)

        self.schedule_cfg = getattr(self.env, 'schedule_params', {}) or {}
        self._I1_target = float(
            min(
                float(getattr(self.env, 'I_target_seedling', 200.0)),
                float(self._action_high[0]),
            )
        )
        self._I2_target = float(
            min(
                float(getattr(self.env, 'I_target_transplant', 300.0)),
                float(self._action_high[1]),
            )
        )

        self._rate_limit_Q = float(self.pid_cfg.get(
            'rate_limit_Q',
            self.schedule_cfg.get('rate_limit_Q', self._action_high[2]),
        )) * self._dt_hours
        self._rate_limit_CO2 = float(self.pid_cfg.get(
            'rate_limit_CO2',
            self._action_high[3],
        )) * self._dt_hours
        self._rate_limit_dehum = float(self.pid_cfg.get(
            'rate_limit_dehum',
            self._action_high[4],
        )) * self._dt_hours

        if hasattr(self, '_temp_loop'):
            self._temp_loop.output_min = float(self._action_low[2])
            self._temp_loop.output_max = float(self._action_high[2])
        if hasattr(self, '_co2_loop'):
            self._co2_loop.output_min = float(self._action_low[3])
            self._co2_loop.output_max = float(self._action_high[3])
        if hasattr(self, '_dehum_loop'):
            self._dehum_loop.output_min = float(self._action_low[4])
            self._dehum_loop.output_max = float(self._action_high[4])

    def bind_env(self, env) -> None:
        """Rebind the controller to the active simulator env."""
        self.env = env
        self.ctrl_cfg = _load_controller_params(env)
        self.pid_cfg = self.ctrl_cfg.get('pid_controller', {})
        self._refresh_env_dependent_state()

    def __init__(self, env):
        self.env = env
        self.ctrl_cfg = _load_controller_params(env)
        self.pid_cfg = self.ctrl_cfg.get('pid_controller', {})
        self.schedule_cfg = getattr(env, 'schedule_params', {}) or {}

        self._refresh_env_dependent_state()

        self._temp_loop = PIDChannel(
            kp=float(self.pid_cfg.get('kp_temp', 90.0)),
            ki=float(self.pid_cfg.get('ki_temp', 18.0)),
            kd=float(self.pid_cfg.get('kd_temp', 8.0)),
            output_min=float(self._action_low[2]),
            output_max=float(self._action_high[2]),
            integral_limit=float(self.pid_cfg.get('temp_integral_limit', 4.0)),
            derivative_alpha=float(self.pid_cfg.get('derivative_filter_alpha', 0.25)),
        )
        self._co2_loop = PIDChannel(
            kp=float(self.pid_cfg.get('kp_co2', 2.5e-9)),
            ki=float(self.pid_cfg.get('ki_co2', 5.0e-10)),
            kd=0.0,
            output_min=float(self._action_low[3]),
            output_max=float(self._action_high[3]),
            integral_limit=float(self.pid_cfg.get('co2_integral_limit_ppm_h', 800.0)),
            derivative_alpha=0.0,
        )
        self._dehum_loop = PIDChannel(
            kp=float(self.pid_cfg.get('kp_vpd', 6.0e-6)),
            ki=float(self.pid_cfg.get('ki_vpd', 2.0e-6)),
            kd=0.0,
            output_min=float(self._action_low[4]),
            output_max=float(self._action_high[4]),
            integral_limit=float(self.pid_cfg.get('vpd_integral_limit_kpa_h', 0.6)),
            derivative_alpha=0.0,
        )

        self._temp_ff_gain = float(self.pid_cfg.get('temp_ff_gain', 1.0))
        self._co2_ff_gain = float(self.pid_cfg.get('co2_ff_gain', 1.0))
        self._dehum_ff_gain = float(self.pid_cfg.get('dehum_ff_gain', 1.0))
        self._temp_feedback_correction_fraction = float(
            self.pid_cfg.get('temp_feedback_correction_fraction', 0.75)
        )
        self._temp_feedback_max_delta_c_per_step = float(
            self.pid_cfg.get('temp_feedback_max_delta_c_per_step', 4.0)
        )
        self._passive_temp_guard_band_c = float(
            self.pid_cfg.get('passive_temp_guard_band_c', 0.5)
        )

        self._prev_action_phys: Optional[np.ndarray] = None
        self._prev_T: Optional[float] = None
        self._last_light_on: Optional[bool] = None
        super().__init__(self.ctrl_cfg)

    def reset(self) -> None:
        self._temp_loop.reset()
        self._co2_loop.reset()
        self._dehum_loop.reset()
        self._prev_action_phys = None
        self._prev_T = None
        self._last_light_on = None

    def _physical_to_action_norm(self, action_phys: np.ndarray) -> np.ndarray:
        span = np.maximum(self._action_high - self._action_low, 1e-12)
        action_norm = 2.0 * (action_phys - self._action_low) / span - 1.0
        return np.clip(action_norm, -1.0, 1.0).astype(np.float32)

    def _apply_rate_limit(
        self,
        action_phys: np.ndarray,
        light_on: bool,
        transition: bool = False,
    ) -> np.ndarray:
        if getattr(self.env, 'enforce_action_rate_limits', False):
            return action_phys
        if self._prev_action_phys is None:
            self._prev_action_phys = action_phys.copy()
            return action_phys

        out = action_phys.copy()
        out[0] = action_phys[0]  # photoperiod lights switch immediately
        out[1] = action_phys[1]

        if transition:
            out[2:] = action_phys[2:]
            return out

        dq = float(np.clip(
            action_phys[2] - self._prev_action_phys[2],
            -self._rate_limit_Q,
            self._rate_limit_Q,
        ))
        out[2] = self._prev_action_phys[2] + dq

        dco2 = float(np.clip(
            action_phys[3] - self._prev_action_phys[3],
            -self._rate_limit_CO2,
            self._rate_limit_CO2,
        ))
        out[3] = self._prev_action_phys[3] + dco2
        if not light_on:
            out[3] = 0.0

        ddehum = float(np.clip(
            action_phys[4] - self._prev_action_phys[4],
            -self._rate_limit_dehum,
            self._rate_limit_dehum,
        ))
        out[4] = self._prev_action_phys[4] + ddehum

        return out

    def _temperature_feedforward(self, T_target: float, I1: float, I2: float) -> float:
        env = self.env
        if env.external is None:
            return 0.0
        T_out = float(env.external[0])
        A_total = max(float(env.A_total), 1e-12)
        q_led = calculate_led_room_heat_gain(
            I1,
            I2,
            env.A1,
            env.A2,
            env.container_params,
            total_P=float(getattr(env, 'total_P', 0.0)),
            total_R=float(getattr(env, 'total_R', 0.0)),
        )['Q_led_room_W']
        q_wall = env.c_U * env.c_surface_area * (T_out - T_target)
        q_vent = env.c_cap_q_v * env._V_vent_fixed * A_total * (T_out - T_target)
        q_transp = float(getattr(env, 'total_E', 0.0)) * env.c_lat_water * 1000.0
        last_action = getattr(env, '_last_applied_action_phys', None)
        m_dehum_prev = float(last_action[4]) if last_action is not None else 0.0
        q_dehum = calculate_dehum_heat_rejection(
            m_dehum_prev,
            A_total,
            env.container_params,
            c_lat_water_kj_per_kg=env.c_lat_water,
        )['Q_dehum_total_to_room_W']
        q_hvac_total = -(q_led + q_wall + q_vent + q_dehum - q_transp)
        return q_hvac_total / A_total

    def _co2_feedforward(self, C_target_density: float) -> float:
        env = self.env
        if env.external is None:
            return 0.0
        A_total = max(float(env.A_total), 1e-12)
        C_out = float(env.external[2])
        net_canopy = max(0.0, (float(env.total_P) - float(env.total_R)) / A_total)
        vent_loss = env._V_vent_fixed * max(C_target_density - C_out, 0.0)
        return net_canopy + vent_loss

    def _dehum_feedforward(self, xH_target: float) -> float:
        env = self.env
        if env.external is None:
            return 0.0
        A_total = max(float(env.A_total), 1e-12)
        xH_out = float(env.external[1])
        transp = float(getattr(env, 'total_E', 0.0)) / A_total
        vent_term = env._V_vent_fixed * (xH_target - xH_out)
        return max(0.0, transp - vent_term)

    def _apply_passive_temperature_guard(
        self,
        q_hvac: float,
        temp_setpoint_c: float,
        q_hold: float,
    ) -> float:
        """Avoid active heating/cooling when ambient conditions already help."""
        env = self.env
        if env.external is None:
            return q_hvac

        t_out = float(env.external[0])
        guard = float(max(self._passive_temp_guard_band_c, 0.0))
        if q_hvac < 0.0 and t_out <= temp_setpoint_c + guard and q_hold >= 0.0:
            self._temp_loop.integral = min(self._temp_loop.integral, 0.0)
            return 0.0
        if q_hvac > 0.0 and t_out >= temp_setpoint_c - guard and q_hold <= 0.0:
            self._temp_loop.integral = max(self._temp_loop.integral, 0.0)
            return 0.0
        return q_hvac

    def _bound_temperature_feedback(
        self,
        q_feedback: float,
        temp_error_c: float,
    ) -> float:
        """Bound temperature feedback using the room thermal capacitance."""
        c_cap_q = max(float(self.env.container_params.get('c_cap_q', 30000.0)), 1e-9)
        dt_s = max(float(self.env.dt), 1e-9)
        frac = float(max(self._temp_feedback_correction_fraction, 0.0))
        max_delta = float(max(self._temp_feedback_max_delta_c_per_step, 0.0))
        desired_delta = min(abs(float(temp_error_c)), max_delta) * frac
        q_cap = c_cap_q * desired_delta / dt_s
        return float(np.clip(q_feedback, -q_cap, q_cap))

    def _bound_temperature_actuation(
        self,
        q_hvac: float,
        temp_error_c: float,
    ) -> float:
        """Bound the total HVAC command to a safe per-step temperature correction."""
        c_cap_q = max(float(self.env.container_params.get('c_cap_q', 30000.0)), 1e-9)
        dt_s = max(float(self.env.dt), 1e-9)
        frac = float(max(self._temp_feedback_correction_fraction, 0.0))
        max_delta = float(max(self._temp_feedback_max_delta_c_per_step, 0.0))
        desired_delta = min(abs(float(temp_error_c)), max_delta) * frac
        q_cap = c_cap_q * desired_delta / dt_s
        return float(np.clip(q_hvac, -q_cap, q_cap))

    def predict(self, obs: np.ndarray, context=None) -> np.ndarray:
        self._refresh_env_dependent_state()
        env = self.env
        T = float(env.state[1])
        xH = float(env.state[2])
        C_density = float(env.state[0])
        C_ppm = co2_density_to_ppm(C_density, T)

        photo_idx = int(env.time_step) % env._i_max
        targets = env.get_climate_targets(photo_idx=photo_idx, T_current=T)
        light_on = bool(targets['light_on'])
        light_transition = (self._last_light_on is not None and light_on != self._last_light_on)

        if light_transition:
            self._temp_loop.reset()
            self._co2_loop.reset()
            self._dehum_loop.reset()

        I1 = self._I1_target if light_on else 0.0
        I2 = self._I2_target if light_on else 0.0

        temp_rate = 0.0
        if self._prev_T is not None:
            temp_rate = (T - self._prev_T) / self._dt_hours
        self._prev_T = T

        temp_error_c = targets['temp_setpoint_c'] - T
        q_ff = self._temp_ff_gain * self._temperature_feedforward(
            targets['temp_setpoint_c'], I1, I2
        )
        q_fb = self._temp_loop.update(
            error=temp_error_c,
            dt_hours=self._dt_hours,
            measurement_rate=temp_rate,
            feedforward=0.0,
        )
        q_fb = self._bound_temperature_feedback(q_fb, temp_error_c)
        Q_HVAC = self._bound_temperature_actuation(q_ff + q_fb, temp_error_c)
        Q_HVAC = float(np.clip(Q_HVAC, self._action_low[2], self._action_high[2]))
        Q_HVAC = self._apply_passive_temperature_guard(
            Q_HVAC,
            targets['temp_setpoint_c'],
            q_ff,
        )

        if light_on:
            u_ff = self._co2_ff_gain * self._co2_feedforward(targets['co2_setpoint_density'])
            u_CO2 = self._co2_loop.update(
                error=targets['co2_setpoint_ppm'] - C_ppm,
                dt_hours=self._dt_hours,
                feedforward=u_ff,
            )
        else:
            u_CO2 = 0.0

        vpd_kpa = absolute_humidity_to_vpd(T, xH, env.container_params)
        if (xH > targets['xH_target']) or (vpd_kpa < targets['vpd_lo_kpa']):
            m_ff = self._dehum_ff_gain * self._dehum_feedforward(targets['xH_target'])
            m_dehum = self._dehum_loop.update(
                error=targets['vpd_target_kpa'] - vpd_kpa,
                dt_hours=self._dt_hours,
                feedforward=m_ff,
            )
        else:
            self._dehum_loop.reset()
            m_dehum = 0.0

        action_phys = np.array([I1, I2, Q_HVAC, u_CO2, m_dehum], dtype=np.float64)
        action_phys = np.clip(action_phys, self._action_low, self._action_high)
        action_phys = self._apply_rate_limit(
            action_phys,
            light_on=light_on,
            transition=light_transition,
        )
        action_phys = np.clip(action_phys, self._action_low, self._action_high)
        self._prev_action_phys = action_phys.copy()
        self._last_light_on = light_on

        return self._physical_to_action_norm(action_phys)
