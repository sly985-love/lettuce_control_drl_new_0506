# -*- coding: utf-8 -*-
"""Run the A2 inlet-seedling-quality sensitivity wrapper."""

from __future__ import annotations

import argparse

from experiment_suite_common import ROOT, build_python_command, run_command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run A2 inlet-seedling-quality sensitivity.")
    parser.add_argument(
        "--schedule",
        type=str,
        default="t1=14,t2=14,N1=20,rho2=36",
    )
    parser.add_argument("--initial-masses", type=str, default="0.00072,0.0072")
    parser.add_argument("--same-densities", type=str, default="34,48,50,80,113")
    parser.add_argument("--same-intensities", type=str, default="200,250,300")
    parser.add_argument("--two-stage-dense-intensity", type=float, default=200.0)
    parser.add_argument("--two-stage-final-intensity", type=float, default=300.0)
    parser.add_argument("--same-age-days", type=float, default=28.0)
    parser.add_argument(
        "--out-csv",
        type=str,
        default=str(ROOT / "results" / "exp07_seedling_quality" / "initial_biomass_compare.csv"),
    )
    parser.add_argument("--python", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cmd = build_python_command(
        "compare_initial_biomass_scenarios.py",
        "--schedule",
        args.schedule,
        "--initial-masses",
        args.initial_masses,
        "--same-densities",
        args.same_densities,
        "--same-intensities",
        args.same_intensities,
        "--two-stage-dense-intensity",
        args.two_stage_dense_intensity,
        "--two-stage-final-intensity",
        args.two_stage_final_intensity,
        "--same-age-days",
        args.same_age_days,
        "--out-csv",
        args.out_csv,
        python_executable=args.python,
    )
    run_command(cmd, dry_run=bool(args.dry_run))


if __name__ == "__main__":
    main()
