# -*- coding: utf-8 -*-
"""Inspect schedule-level theoretical crop growth under the contextual PFAL model."""

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


def _format_mass_line(label: str, dry_g: float, fw_factor: float, min_fw_g: float, target_fw_g: float) -> str:
    fresh_g = dry_g * fw_factor
    min_ratio = fresh_g / max(min_fw_g, 1e-12)
    target_ratio = fresh_g / max(target_fw_g, 1e-12)
    return (
        f"{label}: dry={dry_g:.3f} g/plant | fresh={fresh_g:.1f} g/plant | "
        f"vs min={min_ratio:.3f}x | vs target={target_ratio:.3f}x"
    )


def _trace_phase_daily(
    bm,
    *,
    phase_name: str,
    xD_init: float,
    age_h: float,
    rho: float,
    I_standard: float,
    T_standard: float,
    C_standard: float,
    RH_standard: float,
    start_day: float,
    fw_factor: float,
) -> list[dict]:
    if age_h <= 0.0:
        return []

    I_seq, T_seq, C_seq, RH_seq = bm._generate_photoperiod_seq(
        age_h, I_standard, T_standard, C_standard, RH_standard
    )
    dt_seconds = float(bm.dt_steady)
    dt_hours = dt_seconds / 3600.0
    steps_per_day = max(int(round(24.0 / max(dt_hours, 1e-12))), 1)
    xD = float(xD_init)

    rows: list[dict] = []
    day_steps = 0
    phase_day = 0
    phi_phot_c_sum = 0.0
    phi_phot_sum = 0.0
    phi_resp_sum = 0.0
    phi_transp_sum = 0.0
    lai_sum = 0.0

    for step_idx, (I, T, C, RH) in enumerate(zip(I_seq, T_seq, C_seq, RH_seq), start=1):
        d_xD_dt, phi_phot, phi_resp, phi_transp, lai_proxy = bm._ref2_kernel(
            xD,
            I,
            T,
            C,
            RH,
            rho,
        )
        xD = bm._clamp_total_density(xD + d_xD_dt * dt_seconds)

        phi_phot_c_sum += float(phi_phot - phi_resp)
        phi_phot_sum += float(phi_phot)
        phi_resp_sum += float(phi_resp)
        phi_transp_sum += float(phi_transp)
        lai_sum += float(lai_proxy)
        day_steps += 1

        if day_steps >= steps_per_day or step_idx == len(I_seq):
            dry_g = xD * 1000.0 / max(rho, 1e-12)
            phase_day += 1
            rows.append(
                {
                    "phase": phase_name,
                    "phase_day": int(phase_day),
                    "day_from_seed": float(start_day + phase_day),
                    "dry_g": float(dry_g),
                    "fresh_g": float(dry_g * fw_factor),
                    "mean_lai_proxy": float(lai_sum / max(day_steps, 1)),
                    "mean_phi_phot_c_g_m2_h": float(
                        (phi_phot_c_sum / max(day_steps, 1)) * 1000.0 * 3600.0
                    ),
                    "mean_phi_phot_g_m2_h": float(
                        (phi_phot_sum / max(day_steps, 1)) * 1000.0 * 3600.0
                    ),
                    "mean_phi_resp_g_m2_h": float(
                        (phi_resp_sum / max(day_steps, 1)) * 1000.0 * 3600.0
                    ),
                    "mean_phi_transp_g_m2_h": float(
                        (phi_transp_sum / max(day_steps, 1)) * 1000.0 * 3600.0
                    ),
                }
            )
            day_steps = 0
            phi_phot_c_sum = 0.0
            phi_phot_sum = 0.0
            phi_resp_sum = 0.0
            phi_transp_sum = 0.0
            lai_sum = 0.0

    return rows


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
        description="Inspect theoretical seedling/transplant growth under contextual PFAL target conditions."
    )
    parser.add_argument(
        "--schedule",
        type=str,
        default=None,
        help='Optional upper-schedule string like "t1=14,t2=14,N1=20,rho2=36".',
    )
    parser.add_argument(
        "--i_seedling",
        type=float,
        default=None,
        help="Override seedling PPFD target [umol/m^2/s].",
    )
    parser.add_argument(
        "--i_transplant",
        type=float,
        default=None,
        help="Override transplant PPFD target [umol/m^2/s].",
    )
    parser.add_argument(
        "--scan_i2",
        type=str,
        default=None,
        help='Optional transplant-PPFD scan "250,300,350,400".',
    )
    parser.add_argument(
        "--scan_t2",
        type=str,
        default=None,
        help='Optional transplant-duration scan in days, for example "15,18,21,24,28".',
    )
    parser.add_argument(
        "--daily_phase",
        type=str,
        default="none",
        choices=("none", "seedling", "transplant", "both"),
        help="Optional day-by-day trajectory dump for the selected phase(s).",
    )
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        help="Optional experiment dir/name or run_config.json path used to restore saved runtime metadata.",
    )
    args = parser.parse_args()

    env, inlet_meta = _build_env(_parse_schedule_arg(args.schedule), args.load)
    bm = env.batch_manager

    if args.i_seedling is not None:
        bm.I_init_seedling = float(args.i_seedling)
    if args.i_transplant is not None:
        bm.I_init_transplant = float(args.i_transplant)

    fw_factor = float(env.c_fw)
    min_dw_g = float(env.reward_params.get("harvest_min_dry_mass_per_plant", 4.44))
    target_dw_g = float(
        env.reward_params.get(
            "harvest_target_dry_mass_per_plant",
            env.reward_params.get("harvest_min_dry_mass_per_plant", 4.44),
        )
    )
    min_fw_g = min_dw_g * fw_factor
    target_fw_g = target_dw_g * fw_factor

    xDn_seed_end, xDs_seed_end = bm._get_seedling_end_state()
    seedling_dry_g = (xDn_seed_end + xDs_seed_end) * 1000.0 / max(bm.rho1, 1e-12)

    xDn_trans_start, xDs_trans_start = bm._convert_seedling_density_to_transplant(
        xDn_seed_end, xDs_seed_end
    )
    transplant_start_dry_g = (xDn_trans_start + xDs_trans_start) * 1000.0 / max(bm.rho2, 1e-12)

    xDn_harvest, xDs_harvest = bm._forward_integrate(
        xDn_trans_start,
        xDs_trans_start,
        bm.t2_hours,
        bm.rho2,
        bm.I_init_transplant,
        bm.T_standard,
        bm.C_standard,
        bm.RH_standard,
    )
    harvest_dry_g = (xDn_harvest + xDs_harvest) * 1000.0 / max(bm.rho2, 1e-12)

    print("=== Growth Inspection ===")
    print(
        "Inlet seedling metadata: "
        f"preset={inlet_meta.get('initial_seedling_mass_preset', '')} "
        f"basis={inlet_meta.get('initial_seedling_mass_basis', '')}"
    )
    print(f"Schedule: t1={env.t1} t2={env.t2} N1={env.N1} rho2={env.rho2:.1f}")
    print(
        f"Targets: I_seedling={bm.I_init_seedling:.1f}, I_transplant={bm.I_init_transplant:.1f} umol/m^2/s | "
        f"T_day={bm.T_day:.1f} C | T_night={bm.T_night:.1f} C | "
        f"C_day={bm.C_day_ppm:.0f} ppm | C_night={bm.C_night_ppm:.0f} ppm"
    )
    print(
        f"Thresholds: min={min_dw_g:.2f} g DW ({min_fw_g:.1f} g FW) | "
        f"target={target_dw_g:.2f} g DW ({target_fw_g:.1f} g FW)"
    )
    print(
        _format_mass_line(
            "Inlet seedling",
            float(
                bm.reference_growth_profile.get(
                    "reference_seedling_initial_dry_mass_per_plant_g",
                    0.0,
                )
            ),
            fw_factor,
            min_fw_g,
            target_fw_g,
        )
    )
    print(_format_mass_line("End of seedling phase", seedling_dry_g, fw_factor, min_fw_g, target_fw_g))
    print(_format_mass_line("Start of transplant phase", transplant_start_dry_g, fw_factor, min_fw_g, target_fw_g))
    print(_format_mass_line("End of transplant phase", harvest_dry_g, fw_factor, min_fw_g, target_fw_g))
    print(
        "Reference class: "
        f"{env.schedule.get('reference_feasibility_class', '')} | "
        f"min_feasible={env.schedule.get('reference_min_feasible', False)} | "
        f"target_feasible={env.schedule.get('reference_target_feasible', False)}"
    )
    schedule_warnings = env._build_schedule_reference_warnings()
    if schedule_warnings:
        print("")
        print("=== Schedule Reference Warnings ===")
        for warning in schedule_warnings:
            print(f"- {warning}")

    if args.scan_i2:
        print("")
        print("=== Finishing PPFD Scan ===")
        for token in args.scan_i2.split(","):
            token = token.strip()
            if not token:
                continue
            i2 = float(token)
            xDn_scan, xDs_scan = bm._forward_integrate(
                xDn_trans_start,
                xDs_trans_start,
                bm.t2_hours,
                bm.rho2,
                i2,
                bm.T_standard,
                bm.C_standard,
                bm.RH_standard,
            )
            dry_g = (xDn_scan + xDs_scan) * 1000.0 / max(bm.rho2, 1e-12)
            print(_format_mass_line(f"I2={i2:.1f}", dry_g, fw_factor, min_fw_g, target_fw_g))

    if args.scan_t2:
        print("")
        print("=== Finishing Duration Scan ===")
        for token in args.scan_t2.split(","):
            token = token.strip()
            if not token:
                continue
            t2_days = float(token)
            xDn_scan, xDs_scan = bm._forward_integrate(
                xDn_trans_start,
                xDs_trans_start,
                t2_days * 24.0,
                bm.rho2,
                bm.I_init_transplant,
                bm.T_standard,
                bm.C_standard,
                bm.RH_standard,
            )
            dry_g = (xDn_scan + xDs_scan) * 1000.0 / max(bm.rho2, 1e-12)
            print(_format_mass_line(f"t2={t2_days:.1f} d", dry_g, fw_factor, min_fw_g, target_fw_g))

    if args.daily_phase != "none":
        daily_rows: list[dict] = []
        if args.daily_phase in {"seedling", "both"}:
            xD_seed_init = bm._resolve_initial_seedling_density(bm.rho1)
            daily_rows.extend(
                _trace_phase_daily(
                    bm,
                    phase_name="seedling",
                    xD_init=xD_seed_init,
                    age_h=bm.t1_hours,
                    rho=bm.rho1,
                    I_standard=bm.I_init_seedling,
                    T_standard=bm.T_standard,
                    C_standard=bm.C_standard,
                    RH_standard=bm.RH_standard,
                    start_day=0.0,
                    fw_factor=fw_factor,
                )
            )
        if args.daily_phase in {"transplant", "both"}:
            daily_rows.extend(
                _trace_phase_daily(
                    bm,
                    phase_name="transplant",
                    xD_init=xDn_trans_start + xDs_trans_start,
                    age_h=bm.t2_hours,
                    rho=bm.rho2,
                    I_standard=bm.I_init_transplant,
                    T_standard=bm.T_standard,
                    C_standard=bm.C_standard,
                    RH_standard=bm.RH_standard,
                    start_day=float(env.t1),
                    fw_factor=fw_factor,
                )
            )

        if daily_rows:
            print("")
            print("=== Daily Trajectory ===")
            for row in daily_rows:
                print(
                    f"{row['phase']:>10s} day={row['phase_day']:2d} "
                    f"(seed+{row['day_from_seed']:4.1f} d) "
                    f"dry={row['dry_g']:6.3f} g/plant fresh={row['fresh_g']:6.1f} g/plant "
                    f"LAI*={row['mean_lai_proxy']:.3f} "
                    f"netC={row['mean_phi_phot_c_g_m2_h']:.3f} gCO2/m^2/h "
                    f"phot={row['mean_phi_phot_g_m2_h']:.3f} resp={row['mean_phi_resp_g_m2_h']:.3f} "
                    f"transp={row['mean_phi_transp_g_m2_h']:.3f} gH2O/m^2/h"
                )


if __name__ == "__main__":
    main()
