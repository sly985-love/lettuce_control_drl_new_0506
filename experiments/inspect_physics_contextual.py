# -*- coding: utf-8 -*-
"""Inspect key physical scales for the contextual PFAL simulator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from envs.plant_factory_env_new import PFALEnvContextual
from envs.utils import (
    apply_inlet_seedling_metadata,
    extract_inlet_seedling_metadata,
    has_inlet_seedling_metadata,
    load_all_configs,
    prepare_runtime_config,
)
from models import absolute_humidity_to_relative


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
        if key in {"t1", "t2", "N1"}:
            schedule[key] = int(float(value))
        else:
            schedule[key] = float(value)
    return schedule


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
        description="Inspect LED / ventilation / CO2 physical scales for the contextual PFAL simulator."
    )
    parser.add_argument(
        "--schedule",
        type=str,
        default=None,
        help='Optional upper-schedule string like "t1=14,t2=14,N1=20,rho2=36".',
    )
    parser.add_argument(
        "--hour_of_day",
        type=float,
        default=10.0,
        help="Hour of day used for the initial reset snapshot.",
    )
    parser.add_argument(
        "--probe_i1",
        type=float,
        default=None,
        help="Override probe PPFD for the seedling zone [umol/m^2/s].",
    )
    parser.add_argument(
        "--probe_i2",
        type=float,
        default=None,
        help="Override probe PPFD for the transplant zone [umol/m^2/s].",
    )
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        help="Optional experiment dir/name or run_config.json path used to restore saved runtime metadata.",
    )
    args = parser.parse_args()

    env, inlet_meta = _build_env(_parse_schedule_arg(args.schedule), args.load)
    _, info = env.reset(options={"hour_of_day": float(args.hour_of_day)})

    diag = dict(
        getattr(env, "physics_diagnostics", {})
        or info.get("physics_diagnostics", {})
        or {}
    )
    print("=== Static Physics Diagnostics ===")
    print(
        "Inlet seedling metadata: "
        f"preset={inlet_meta.get('initial_seedling_mass_preset', '')} "
        f"basis={inlet_meta.get('initial_seedling_mass_basis', '')}"
    )
    print(f"Schedule: t1={env.t1} t2={env.t2} N1={env.N1} rho2={env.rho2:.1f}")
    print(f"Areas: A1={env.A1:.2f} m^2, A2={env.A2:.2f} m^2, total={env.A_total:.2f} m^2")
    print(f"Ventilation: {diag.get('fixed_vent_rate_m3_m2_s', 0.0):.6e} m^3/m^2/s = {diag.get('fixed_vent_ach', 0.0):.3f} ACH")
    print(f"LED PPE: {diag.get('led_ppe_umol_per_j', 0.0):.3f} umol/J")
    print(f"LED radiant efficiency: {diag.get('led_radiant_efficiency', 0.0):.3f}")
    print(f"LED heat fraction: {diag.get('led_heat_fraction', 0.0):.3f}")
    print(f"Target LED electric power: {diag.get('P_led_total_W', 0.0):.1f} W")
    print(f"Target LED driver-loss heat: {diag.get('Q_led_driver_loss_W', 0.0):.1f} W")
    print(f"Target LED room-heat upper bound: {diag.get('Q_led_room_upper_W', 0.0):.1f} W")
    print(f"Target LED net room heat: {diag.get('Q_led_room_W', 0.0):.1f} W")
    print(f"Target biomass chemical storage: {diag.get('Q_biomass_storage_W', 0.0):.1f} W")
    print(f"Dehumidifier max electric power: {diag.get('P_dehum_cap_W', 0.0):.1f} W")
    print(f"Dehumidifier max room heat rejection: {diag.get('Q_dehum_cap_to_room_W', 0.0):.1f} W")
    print(f"CO2 supply max: {diag.get('co2_supply_max_g_m2_h', 0.0):.3f} g/m^2/h")
    print(f"CO2 vent-only hold @1000 ppm: {diag.get('co2_hold_1000ppm_g_m2_h', 0.0):.3f} g/m^2/h")
    print(f"CO2 supply / vent-only hold: {diag.get('co2_supply_to_hold_ratio_1000ppm', 0.0):.3f}")
    print(f"CO2 theoretical ppm rise at max supply: {diag.get('co2_supply_theoretical_ppm_h', 0.0):.1f} ppm/h")
    print(f"Envelope heat exchange per K: wall={diag.get('wall_exchange_W_per_K', 0.0):.2f} W/K, vent={diag.get('vent_exchange_W_per_K', 0.0):.2f} W/K")

    probe_i1 = float(args.probe_i1 if args.probe_i1 is not None else env.I_target_seedling)
    probe_i2 = float(args.probe_i2 if args.probe_i2 is not None else env.I_target_transplant)
    T = float(env.state[1])
    C = float(env.state[0])
    RH = absolute_humidity_to_relative(T, float(env.state[2]), env.container_params)
    batch_info = env.batch_manager.update(env.dt, probe_i1, probe_i2, T, C, RH)
    total_P_g_m2_h = float(batch_info.get("total_P_rate", 0.0)) * 1000.0 * 3600.0 / max(env.A_total, 1e-12)
    total_R_g_m2_h = float(batch_info.get("total_resp_rate", 0.0)) * 1000.0 * 3600.0 / max(env.A_total, 1e-12)
    total_E_g_m2_h = float(batch_info.get("total_E_rate", 0.0)) * 1000.0 * 3600.0 / max(env.A_total, 1e-12)
    co2_headroom_ratio = diag.get("co2_supply_max_g_m2_h", 0.0) / max(
        diag.get("co2_hold_1000ppm_g_m2_h", 0.0) + max(total_P_g_m2_h - total_R_g_m2_h, 0.0),
        1e-12,
    )

    print("")
    print("=== Canopy Probe at Current State ===")
    print(f"Probe PPFD: I1={probe_i1:.1f}, I2={probe_i2:.1f} umol/m^2/s")
    print(f"Current state: T={T:.2f} C, RH={RH * 100.0:.2f} %, C={info.get('C_ppm', 0.0):.1f} ppm")
    print(f"Gross photosynthesis: {total_P_g_m2_h:.3f} g/m^2/h CO2")
    print(f"Respiration: {total_R_g_m2_h:.3f} g/m^2/h CO2")
    print(f"Transpiration: {total_E_g_m2_h:.3f} g/m^2/h H2O")
    print(f"CO2 supply headroom ratio (vent hold + net canopy demand): {co2_headroom_ratio:.3f}")


if __name__ == "__main__":
    main()
