# -*- coding: utf-8 -*-
"""Compare initial biomass assumptions under the contextual PFAL crop model."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from pathlib import Path
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from envs.utils import load_all_configs, prepare_runtime_config
from models.batch_manager import BatchManager


def _parse_float_list(text: str) -> list[float]:
    values = []
    for token in str(text).split(","):
        token = token.strip()
        if token:
            values.append(float(token))
    return values


def _parse_schedule(text: str) -> dict:
    schedule = {}
    for token in str(text).split(","):
        token = token.strip()
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


def _configure_bm_climate_targets(bm: BatchManager, cp: dict) -> tuple[float, float, float, float, float]:
    photoperiod_h = int(cp.get("photoperiod_on", 16))
    T_day = float(cp.get("temp_target_day", 22.0))
    T_night = float(cp.get("temp_target_night", 18.0))
    RH_day = float(cp.get("RH_standard", 0.75))
    RH_night = min(RH_day + 0.05, 0.90)
    C_day_ppm = float(cp.get("C_standard_ppm", 1000.0))

    bm.photoperiod_on = photoperiod_h
    bm.photoperiod_off = 24 - photoperiod_h
    bm.T_night = T_night
    bm.C_night_ppm = float(cp.get("C_night_ppm", 800.0))
    bm.RH_night = RH_night
    C_day = bm._ppm_to_density(C_day_ppm, T=T_day)
    return T_day, T_night, RH_day, RH_night, C_day


def _make_bm(
    *,
    runtime: dict,
    schedule: dict,
    initial_total_dry_mass_kg_m2: float,
) -> BatchManager:
    cp = deepcopy(runtime["container_params"])
    cr = deepcopy(runtime["crop_params"])
    rp = deepcopy(runtime["reward_params"])
    ss = deepcopy(runtime["steady_state_params"])
    cp["initial_seedling_mass_basis"] = "area_dry_mass_density"
    cp["initial_seedling_area_dry_mass_density"] = float(initial_total_dry_mass_kg_m2)
    ss["initial_seedling_mass"] = float(initial_total_dry_mass_kg_m2)
    ss["initial_seedling_mass_basis"] = "area_dry_mass_density"
    ss["initial_seedling_area_dry_mass_density"] = float(initial_total_dry_mass_kg_m2)
    return BatchManager(
        schedule,
        cp,
        cr,
        steady_state_params=ss,
        reward_params=rp,
        initialise_batches=False,
    )


def _integrate_constant_density(
    bm: BatchManager,
    *,
    initial_total_dry_mass_kg_m2: float,
    density: float,
    age_days: float,
    I_umol: float,
    T_day: float,
    C_day: float,
    RH_day: float,
) -> tuple[float, float]:
    cp = bm.container_params
    ns_ratio = float(cp.get("seedling_nonstruct_ratio", 0.15))
    xDn0 = ns_ratio * initial_total_dry_mass_kg_m2
    xDs0 = (1.0 - ns_ratio) * initial_total_dry_mass_kg_m2
    return bm._forward_integrate(
        xDn0,
        xDs0,
        age_days * 24.0,
        density,
        I_umol,
        T_day,
        C_day,
        RH_day,
    )


def _scenario_rows_same_density(
    *,
    runtime: dict,
    schedule: dict,
    label: str,
    initial_total_dry_mass_kg_m2: float,
    densities: Iterable[float],
    intensities: Iterable[float],
    age_days: float,
) -> list[dict]:
    rows: list[dict] = []
    c_fw = float(runtime["crop_params"].get("c_fw", 22.5))
    for density in densities:
        for intensity in intensities:
            bm = _make_bm(
                runtime=runtime,
                schedule=schedule,
                initial_total_dry_mass_kg_m2=initial_total_dry_mass_kg_m2,
            )
            T_day, _, RH_day, _, C_day = _configure_bm_climate_targets(bm, bm.container_params)
            xDn, xDs = _integrate_constant_density(
                bm,
                initial_total_dry_mass_kg_m2=initial_total_dry_mass_kg_m2,
                density=float(density),
                age_days=age_days,
                I_umol=float(intensity),
                T_day=T_day,
                C_day=C_day,
                RH_day=RH_day,
            )
            initial_dry_g = initial_total_dry_mass_kg_m2 * 1000.0 / max(float(density), 1e-12)
            final_dry_g = (xDn + xDs) * 1000.0 / max(float(density), 1e-12)
            rows.append(
                {
                    "scenario": label,
                    "mode": "same_density",
                    "age_days": float(age_days),
                    "density_pl_m2": float(density),
                    "I_dense_umol_m2_s": float(intensity),
                    "I_final_umol_m2_s": float(intensity),
                    "rho1_pl_m2": float(density),
                    "rho2_pl_m2": float(density),
                    "initial_total_dry_mass_kg_m2": float(initial_total_dry_mass_kg_m2),
                    "initial_dry_g_per_plant": float(initial_dry_g),
                    "initial_fresh_g_per_plant": float(initial_dry_g * c_fw),
                    "dense_end_dry_g_per_plant": float(final_dry_g),
                    "dense_end_fresh_g_per_plant": float(final_dry_g * c_fw),
                    "final_dry_g_per_plant": float(final_dry_g),
                    "final_fresh_g_per_plant": float(final_dry_g * c_fw),
                }
            )
    return rows


def _scenario_row_two_stage(
    *,
    runtime: dict,
    schedule: dict,
    label: str,
    initial_total_dry_mass_kg_m2: float,
    I_dense_umol: float,
    I_final_umol: float,
) -> dict:
    bm = _make_bm(
        runtime=runtime,
        schedule=schedule,
        initial_total_dry_mass_kg_m2=initial_total_dry_mass_kg_m2,
    )
    T_day, _, RH_day, _, C_day = _configure_bm_climate_targets(bm, bm.container_params)
    c_fw = float(runtime["crop_params"].get("c_fw", 22.5))
    ns_ratio = float(bm.container_params.get("seedling_nonstruct_ratio", 0.15))
    xDn0 = ns_ratio * initial_total_dry_mass_kg_m2
    xDs0 = (1.0 - ns_ratio) * initial_total_dry_mass_kg_m2

    xDn_seed_end, xDs_seed_end = bm._forward_integrate(
        xDn0,
        xDs0,
        bm.t1_hours,
        bm.rho1,
        I_dense_umol,
        T_day,
        C_day,
        RH_day,
    )
    dense_end_dry_g = (xDn_seed_end + xDs_seed_end) * 1000.0 / max(bm.rho1, 1e-12)

    xDn_trans_start, xDs_trans_start = bm._convert_seedling_density_to_transplant(
        xDn_seed_end,
        xDs_seed_end,
    )
    xDn_harvest, xDs_harvest = bm._forward_integrate(
        xDn_trans_start,
        xDs_trans_start,
        bm.t2_hours,
        bm.rho2,
        I_final_umol,
        T_day,
        C_day,
        RH_day,
    )
    final_dry_g = (xDn_harvest + xDs_harvest) * 1000.0 / max(bm.rho2, 1e-12)
    initial_dry_g = initial_total_dry_mass_kg_m2 * 1000.0 / max(bm.rho1, 1e-12)

    return {
        "scenario": label,
        "mode": "two_stage",
        "age_days": float(bm.t1 + bm.t2),
        "density_pl_m2": float(bm.rho2),
        "I_dense_umol_m2_s": float(I_dense_umol),
        "I_final_umol_m2_s": float(I_final_umol),
        "rho1_pl_m2": float(bm.rho1),
        "rho2_pl_m2": float(bm.rho2),
        "initial_total_dry_mass_kg_m2": float(initial_total_dry_mass_kg_m2),
        "initial_dry_g_per_plant": float(initial_dry_g),
        "initial_fresh_g_per_plant": float(initial_dry_g * c_fw),
        "dense_end_dry_g_per_plant": float(dense_end_dry_g),
        "dense_end_fresh_g_per_plant": float(dense_end_dry_g * c_fw),
        "final_dry_g_per_plant": float(final_dry_g),
        "final_fresh_g_per_plant": float(final_dry_g * c_fw),
    }


def _write_csv(rows: list[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare contextual PFAL growth under alternative initial biomass assumptions."
    )
    parser.add_argument(
        "--schedule",
        type=str,
        default="t1=14,t2=14,N1=20,rho2=36",
        help='Two-stage schedule, for example "t1=14,t2=14,N1=20,rho2=36".',
    )
    parser.add_argument(
        "--initial-masses",
        type=str,
        default="0.00072,0.0072",
        help="Comma-separated initial total dry masses [kg/m^2].",
    )
    parser.add_argument(
        "--same-densities",
        type=str,
        default="34,48,50,80,113",
        help="Comma-separated densities [plants/m^2] for same-density 28-day runs.",
    )
    parser.add_argument(
        "--same-intensities",
        type=str,
        default="200,250,300",
        help="Comma-separated PPFD values [umol/m^2/s] for same-density runs.",
    )
    parser.add_argument(
        "--two-stage-dense-intensity",
        type=float,
        default=200.0,
        help="Dense-stage PPFD [umol/m^2/s] for two-stage runs.",
    )
    parser.add_argument(
        "--two-stage-final-intensity",
        type=float,
        default=300.0,
        help="Final-stage PPFD [umol/m^2/s] for two-stage runs.",
    )
    parser.add_argument(
        "--same-age-days",
        type=float,
        default=28.0,
        help="Age in days for same-density runs.",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="results/initial_biomass_compare/initial_biomass_compare.csv",
        help="CSV output path relative to repo root.",
    )
    args = parser.parse_args()

    runtime = prepare_runtime_config(load_all_configs(str(ROOT / "configs")))
    schedule = _parse_schedule(args.schedule)
    initial_masses = _parse_float_list(args.initial_masses)
    same_densities = _parse_float_list(args.same_densities)
    same_intensities = _parse_float_list(args.same_intensities)

    scenario_labels = []
    for mass in initial_masses:
        if abs(mass - 0.00072) < 1e-12:
            scenario_labels.append("legacy_xDi_0p72e_minus3")
        elif abs(mass - 0.0072) < 1e-12:
            scenario_labels.append("inlet_proxy_7p2e_minus3")
        else:
            scenario_labels.append(f"custom_{mass:g}".replace(".", "p"))

    rows: list[dict] = []
    for label, mass in zip(scenario_labels, initial_masses):
        rows.extend(
            _scenario_rows_same_density(
                runtime=runtime,
                schedule=schedule,
                label=label,
                initial_total_dry_mass_kg_m2=mass,
                densities=same_densities,
                intensities=same_intensities,
                age_days=float(args.same_age_days),
            )
        )
        rows.append(
            _scenario_row_two_stage(
                runtime=runtime,
                schedule=schedule,
                label=label,
                initial_total_dry_mass_kg_m2=mass,
                I_dense_umol=float(args.two_stage_dense_intensity),
                I_final_umol=float(args.two_stage_final_intensity),
            )
        )

    out_csv = ROOT / args.out_csv
    _write_csv(rows, out_csv)

    print(f"[SAVE] CSV -> {out_csv}")
    for row in rows:
        if row["mode"] == "two_stage":
            print(
                f"[TWO_STAGE] {row['scenario']}: "
                f"init={row['initial_dry_g_per_plant']:.4f} gDW/plant "
                f"({row['initial_fresh_g_per_plant']:.3f} gFW/plant), "
                f"dense_end={row['dense_end_fresh_g_per_plant']:.2f} gFW/plant, "
                f"final={row['final_fresh_g_per_plant']:.2f} gFW/plant"
            )


if __name__ == "__main__":
    main()
