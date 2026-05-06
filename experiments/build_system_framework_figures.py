# -*- coding: utf-8 -*-
"""Build paper-ready structural framework diagrams for the PFAL bilevel study.

The diagrams are intentionally generated from code so labels, colors and layout
can be revised consistently during manuscript iteration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper" / "figures_system_framework_20260506"


COLORS = {
    "ink": "#24313A",
    "muted": "#687782",
    "line": "#71828D",
    "bg": "#FAFAF7",
    "panel": "#FFFFFF",
    "upper": "#DCECCB",
    "upper_edge": "#5D8E4F",
    "lower": "#D9EEF2",
    "lower_edge": "#2F7F91",
    "plant": "#E7F2DE",
    "plant_edge": "#6FA15B",
    "energy": "#FFE7B8",
    "energy_edge": "#C6861A",
    "economy": "#FADBD2",
    "economy_edge": "#C45C4A",
    "data": "#E6E4F6",
    "data_edge": "#6C6AA8",
    "accent": "#2F7F91",
    "accent2": "#E26D5A",
    "accent3": "#F2BE4A",
    "water": "#75B7C7",
    "co2": "#78B77A",
    "heat": "#E78A63",
    "light": "#F3C95E",
}


mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.linewidth": 0.8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "figure.facecolor": COLORS["bg"],
        "savefig.facecolor": COLORS["bg"],
    }
)


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    w: float
    h: float
    label: str
    fc: str = "panel"
    ec: str = "line"
    lw: float = 1.15
    radius: float = 0.18
    fontsize: float = 9.5
    weight: str = "regular"
    color: str = "ink"
    align: str = "center"


def _setup(width: float = 14.0, height: float = 8.0):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")
    return fig, ax


def _box(ax, box: Box):
    patch = FancyBboxPatch(
        (box.x, box.y),
        box.w,
        box.h,
        boxstyle=f"round,pad=0.025,rounding_size={box.radius}",
        facecolor=COLORS.get(box.fc, box.fc),
        edgecolor=COLORS.get(box.ec, box.ec),
        linewidth=box.lw,
    )
    ax.add_patch(patch)
    ha = "center" if box.align == "center" else "left"
    x = box.x + box.w / 2 if box.align == "center" else box.x + 0.18
    ax.text(
        x,
        box.y + box.h / 2,
        box.label,
        ha=ha,
        va="center",
        fontsize=box.fontsize,
        fontweight=box.weight,
        color=COLORS.get(box.color, box.color),
        linespacing=1.22,
    )
    return patch


def _label(ax, x, y, s, size=9.0, weight="regular", color="ink", ha="center"):
    ax.text(
        x,
        y,
        s,
        ha=ha,
        va="center",
        fontsize=size,
        fontweight=weight,
        color=COLORS.get(color, color),
        linespacing=1.15,
    )


def _arrow(
    ax,
    start,
    end,
    color="line",
    lw=1.25,
    rad=0.0,
    style="-|>",
    ls="-",
    shrink=4,
    mutation_scale=11,
):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=mutation_scale,
        linewidth=lw,
        color=COLORS.get(color, color),
        linestyle=ls,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=shrink,
        shrinkB=shrink,
    )
    ax.add_patch(arr)
    return arr


def _save(fig, stem: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        path = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(path, dpi=420, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def _badge(ax, x, y, text, fc, ec, w=1.55):
    return _box(
        ax,
        Box(
            x,
            y,
            w,
            0.38,
            text,
            fc=fc,
            ec=ec,
            fontsize=7.8,
            weight="bold",
            radius=0.11,
        ),
    )


def _draw_container(ax, x, y, w, h):
    _box(ax, Box(x, y, w, h, "", fc="plant", ec="plant_edge", radius=0.22, lw=1.25))
    _label(ax, x + w / 2, y + h - 0.35, "Dual-zone multi-batch PFAL simulator", size=10.2, weight="bold")
    ax.plot([x + w * 0.48, x + w * 0.48], [y + 0.34, y + h - 0.72], color=COLORS["plant_edge"], lw=1.0)
    _label(ax, x + w * 0.24, y + h - 0.78, "Dense zone", size=8.8, weight="bold", color="upper_edge")
    _label(ax, x + w * 0.74, y + h - 0.78, "Finishing zone", size=8.8, weight="bold", color="upper_edge")
    for i in range(5):
        bx = x + 0.25 + i * 0.43
        by = y + 0.68 + (i % 2) * 0.30
        ax.add_patch(Rectangle((bx, by), 0.30, 0.95, facecolor="#F9FFF5", edgecolor=COLORS["plant_edge"], lw=0.8))
        for j in range(3):
            ax.scatter(bx + 0.08 + j * 0.07, by + 0.18 + j * 0.20, s=18, c=COLORS["co2"], edgecolors="white", lw=0.35)
    for i in range(4):
        bx = x + w * 0.54 + i * 0.58
        by = y + 0.58 + (i % 2) * 0.26
        ax.add_patch(Rectangle((bx, by), 0.42, 1.12, facecolor="#F9FFF5", edgecolor=COLORS["plant_edge"], lw=0.8))
        for j in range(4):
            ax.scatter(bx + 0.09 + (j % 2) * 0.17, by + 0.20 + (j // 2) * 0.40, s=30, c=COLORS["co2"], edgecolors="white", lw=0.35)
    _label(ax, x + w / 2, y + 0.32, "Batch manager: sowing -> transplanting -> harvesting", size=8.4, color="muted")


def build_bilevel_system():
    fig, ax = _setup(14, 8)
    _label(ax, 0.42, 7.62, "a", size=13, weight="bold", color="ink")

    _box(
        ax,
        Box(
            0.55,
            6.42,
            2.6,
            0.88,
            "External drivers\nweather, tariff, lettuce price",
            fc="data",
            ec="data_edge",
            fontsize=8.6,
            weight="bold",
        ),
    )
    _box(
        ax,
        Box(
            3.55,
            6.42,
            2.75,
            0.88,
            "Feasible schedule set\n368 admissible recipes",
            fc="upper",
            ec="upper_edge",
            fontsize=8.6,
            weight="bold",
        ),
    )
    _box(
        ax,
        Box(
            7.25,
            6.25,
            2.85,
            1.18,
            "Upper layer\nx = (t1, t2, N1, rho2)\nslow structural recipe",
            fc="upper",
            ec="upper_edge",
            fontsize=8.8,
            weight="bold",
        ),
    )
    _box(
        ax,
        Box(
            10.65,
            6.12,
            2.65,
            1.32,
            "Annual objective\nprofit, yield, cost,\nenergy, harvest quality",
            fc="economy",
            ec="economy_edge",
            fontsize=8.6,
            weight="bold",
        ),
    )
    _arrow(ax, (3.15, 6.85), (3.55, 6.85), color="upper_edge")
    _arrow(ax, (6.30, 6.85), (7.25, 6.85), color="upper_edge")
    _arrow(ax, (10.10, 6.85), (10.65, 6.85), color="economy_edge")
    _arrow(ax, (11.95, 6.12), (8.60, 5.33), color="economy_edge", rad=0.14, ls="--")
    _label(ax, 10.25, 5.72, "exact ranking\nselect x*", size=7.8, color="economy_edge")

    _draw_container(ax, 4.35, 2.12, 5.55, 2.65)
    _box(
        ax,
        Box(
            0.65,
            2.18,
            3.05,
            2.55,
            "Lower layer\nResidual-PID SAC\n\nPID anchor + learned residual\nI1, I2, HVAC, CO2, dehum",
            fc="lower",
            ec="lower_edge",
            fontsize=8.9,
            weight="bold",
        ),
    )
    _box(
        ax,
        Box(
            10.65,
            2.25,
            2.75,
            2.35,
            "Closed-loop evidence\n\nfresh weight\nelectricity use\ndehumidification load\ntransplant and harvest events",
            fc="panel",
            ec="line",
            fontsize=8.4,
            weight="bold",
        ),
    )
    _arrow(ax, (8.25, 6.25), (2.18, 4.73), color="upper_edge", rad=0.10)
    _label(ax, 4.90, 5.36, "schedule context x", size=7.8, color="upper_edge")
    _arrow(ax, (8.50, 6.25), (6.90, 4.78), color="upper_edge")
    _label(ax, 7.85, 5.38, "area, density,\ncycle length", size=7.8, color="upper_edge")
    _arrow(ax, (3.70, 3.45), (4.35, 3.45), color="lower_edge", lw=1.45)
    _label(ax, 4.03, 3.75, "actions", size=7.8, color="lower_edge")
    _arrow(ax, (4.35, 2.55), (3.70, 2.70), color="lower_edge", rad=-0.18, ls="--")
    _label(ax, 3.88, 2.30, "state s_t", size=7.8, color="lower_edge")
    _arrow(ax, (9.90, 3.45), (10.65, 3.45), color="line")
    _arrow(ax, (12.00, 4.60), (12.00, 6.12), color="economy_edge")

    _box(ax, Box(4.50, 0.55, 1.55, 0.70, "Light\nLED heat", fc="energy", ec="energy_edge", fontsize=8.2, weight="bold"))
    _box(ax, Box(6.20, 0.55, 1.55, 0.70, "Thermal\nHVAC", fc="energy", ec="energy_edge", fontsize=8.2, weight="bold"))
    _box(ax, Box(7.90, 0.55, 1.55, 0.70, "CO2\nsupply", fc="energy", ec="energy_edge", fontsize=8.2, weight="bold"))
    _box(ax, Box(9.60, 0.55, 1.55, 0.70, "Water vapor\ndehumidifier", fc="energy", ec="energy_edge", fontsize=8.2, weight="bold"))
    for sx in (5.27, 6.97, 8.67, 10.37):
        _arrow(ax, (sx, 1.25), (7.10, 2.12), color="energy_edge", rad=0.04)

    _box(
        ax,
        Box(
            0.65,
            0.62,
            2.95,
            0.86,
            "Shared air volume\nT, RH, CO2, VPD",
            fc="panel",
            ec="line",
            fontsize=8.5,
            weight="bold",
        ),
    )
    _arrow(ax, (3.60, 1.05), (5.40, 2.12), color="line", ls="--")
    _arrow(ax, (0.55, 6.42), (5.05, 4.66), color="data_edge", rad=-0.10)
    _arrow(ax, (0.95, 6.42), (1.50, 4.73), color="data_edge", rad=0.08)
    _label(ax, 1.03, 5.40, "disturbances", size=7.8, color="data_edge")

    _save(fig, "fig01_bilevel_system_framework")


def build_algorithm_workflow():
    fig, ax = _setup(14, 6.4)
    _label(ax, 0.42, 6.05, "b", size=13, weight="bold", color="ink")
    xs = [0.55, 3.10, 5.90, 8.80, 11.35]
    labels = [
        ("Schedule catalog", "enumerate feasible x\nstructural constraints\nreference feasibility"),
        ("Contextual training", "distributed schedule cycle\nSAC updates\nconstraint selection"),
        ("Exact evaluation", "same weather window\nPID and Residual-PID SAC\n368 closed-loop runs"),
        ("System evidence", "rank schedules\nfrontier shift\nlayer attribution"),
        ("Manuscript analysis", "mechanism cases\nprice and TOU extensions\nablation experiments"),
    ]
    fcs = ["upper", "lower", "panel", "economy", "data"]
    ecs = ["upper_edge", "lower_edge", "line", "economy_edge", "data_edge"]
    for i, (x, (head, body)) in enumerate(zip(xs, labels)):
        _box(ax, Box(x, 3.05, 2.05, 1.65, f"{head}\n\n{body}", fc=fcs[i], ec=ecs[i], fontsize=8.4, weight="bold"))
        _badge(ax, x + 0.24, 4.88, f"Step {i + 1}", fcs[i], ecs[i], w=1.05)
        if i < 4:
            _arrow(ax, (x + 2.05, 3.88), (xs[i + 1], 3.88), color="line", lw=1.35)

    _box(
        ax,
        Box(
            0.75,
            0.78,
            3.25,
            1.10,
            "Inputs\nHangzhou weather, prices, crop and equipment parameters,\nfixed 16 h photoperiod",
            fc="data",
            ec="data_edge",
            fontsize=8.2,
            weight="bold",
        ),
    )
    _box(
        ax,
        Box(
            5.10,
            0.78,
            3.25,
            1.10,
            "Reusable artifacts\ntrained policy, exact result tables,\ndetailed trajectories",
            fc="panel",
            ec="line",
            fontsize=8.2,
            weight="bold",
        ),
    )
    _box(
        ax,
        Box(
            9.45,
            0.78,
            3.25,
            1.10,
            "Outputs\noptimal schedule, controller choice,\nmechanistic interpretation",
            fc="economy",
            ec="economy_edge",
            fontsize=8.2,
            weight="bold",
        ),
    )
    _arrow(ax, (2.35, 1.88), (1.55, 3.05), color="data_edge")
    _arrow(ax, (6.75, 1.88), (6.90, 3.05), color="line")
    _arrow(ax, (11.05, 1.88), (12.25, 3.05), color="economy_edge")
    _arrow(ax, (12.35, 3.05), (9.00, 2.04), color="economy_edge", rad=-0.18, ls="--")
    _label(ax, 10.80, 2.42, "design feedback", size=7.8, color="economy_edge")
    _save(fig, "fig02_algorithm_workflow")


def build_residual_controller():
    fig, ax = _setup(14, 6.8)
    _label(ax, 0.42, 6.43, "c", size=13, weight="bold", color="ink")

    _box(ax, Box(0.65, 4.65, 2.55, 1.10, "Observation s_t\nT, RH, CO2, biomass,\ntime, weather", fc="data", ec="data_edge", fontsize=8.4, weight="bold"))
    _box(ax, Box(0.65, 3.15, 2.55, 0.95, "Schedule context x\nt1, t2, N1, rho2", fc="upper", ec="upper_edge", fontsize=8.4, weight="bold"))
    _box(ax, Box(0.65, 1.85, 2.55, 0.82, "Economic context\nprice observation for TOU", fc="economy", ec="economy_edge", fontsize=8.2, weight="bold"))

    _box(ax, Box(4.00, 4.15, 2.38, 1.10, "PID anchor\nu_PID(s_t)\nengineering feedback", fc="panel", ec="line", fontsize=8.7, weight="bold"))
    _box(ax, Box(4.00, 2.15, 2.38, 1.15, "SAC actor\nDelta u_theta(s_t, x)\nlearned residual", fc="lower", ec="lower_edge", fontsize=8.7, weight="bold"))
    _box(ax, Box(7.05, 3.10, 2.28, 1.10, "Residual fusion\nu = clip(u_PID + alpha Delta u_theta)\naction limits", fc="energy", ec="energy_edge", fontsize=8.2, weight="bold"))
    _box(ax, Box(10.05, 3.20, 2.78, 1.00, "Physical actuators\nI1, I2, HVAC, CO2, dehumidifier", fc="plant", ec="plant_edge", fontsize=8.5, weight="bold"))
    _box(ax, Box(10.05, 1.25, 2.78, 0.98, "Reward and constraints\nprofit, harvest quality,\nenergy, safety overrides", fc="economy", ec="economy_edge", fontsize=8.2, weight="bold"))

    for sy in (5.20, 3.62, 2.26):
        _arrow(ax, (3.20, sy), (4.00, 4.65 if sy > 4 else 2.72), color="data_edge" if sy > 4 else "upper_edge" if sy > 3 else "economy_edge", rad=0.0)
    _arrow(ax, (6.38, 4.70), (7.05, 3.83), color="line")
    _arrow(ax, (6.38, 2.72), (7.05, 3.55), color="lower_edge")
    _arrow(ax, (9.33, 3.65), (10.05, 3.70), color="energy_edge", lw=1.45)
    _arrow(ax, (11.45, 3.20), (11.45, 2.23), color="economy_edge")
    _arrow(ax, (10.05, 1.73), (5.30, 2.15), color="economy_edge", rad=-0.18, ls="--")
    _label(ax, 7.55, 1.62, "critic update and policy improvement", size=7.8, color="economy_edge")

    _box(
        ax,
        Box(
            3.55,
            0.55,
            6.35,
            0.72,
            "Key design principle: the policy learns where and how much to deviate from PID, not a fully unconstrained absolute action.",
            fc="panel",
            ec="line",
            fontsize=8.25,
            weight="bold",
        ),
    )
    _save(fig, "fig03_residual_pid_sac_architecture")


def build_physics_mechanism():
    fig, ax = _setup(14, 7.4)
    _label(ax, 0.42, 7.02, "d", size=13, weight="bold", color="ink")

    _box(
        ax,
        Box(
            0.70,
            4.78,
            2.65,
            1.42,
            "Crop physiology\nphotosynthesis\nrespiration\ngrowth and dilution",
            fc="plant",
            ec="plant_edge",
            fontsize=8.7,
            weight="bold",
        ),
    )
    _box(
        ax,
        Box(
            0.70,
            2.72,
            2.65,
            1.42,
            "Canopy water flux\nstomatal response\ntranspiration\nlatent heat",
            fc="plant",
            ec="plant_edge",
            fontsize=8.7,
            weight="bold",
        ),
    )
    _box(
        ax,
        Box(
            4.25,
            4.95,
            2.65,
            1.05,
            "CO2 balance\ndC/dt = supply - uptake - exchange",
            fc="data",
            ec="data_edge",
            fontsize=8.35,
            weight="bold",
        ),
    )
    _box(
        ax,
        Box(
            4.25,
            3.28,
            2.65,
            1.05,
            "Thermal balance\ndT/dt = HVAC + LED + wall\n+ dehum heat - latent sink",
            fc="energy",
            ec="energy_edge",
            fontsize=8.15,
            weight="bold",
        ),
    )
    _box(
        ax,
        Box(
            4.25,
            1.60,
            2.65,
            1.05,
            "Humidity balance\ndH/dt = transpiration\n- ventilation - dehumidification",
            fc="lower",
            ec="lower_edge",
            fontsize=8.15,
            weight="bold",
        ),
    )
    _box(
        ax,
        Box(
            8.00,
            4.85,
            2.55,
            1.12,
            "Equipment model\nLED, HVAC, CO2 dosing,\ndehumidifier",
            fc="energy",
            ec="energy_edge",
            fontsize=8.5,
            weight="bold",
        ),
    )
    _box(
        ax,
        Box(
            8.00,
            2.85,
            2.55,
            1.12,
            "Shared air state\nT, RH, CO2, VPD\nsingle climate volume",
            fc="panel",
            ec="line",
            fontsize=8.5,
            weight="bold",
        ),
    )
    _box(
        ax,
        Box(
            8.00,
            0.95,
            2.55,
            1.12,
            "Economic accounting\nelectricity, CO2 cost,\nfresh-weight revenue",
            fc="economy",
            ec="economy_edge",
            fontsize=8.5,
            weight="bold",
        ),
    )
    _box(
        ax,
        Box(
            11.35,
            2.55,
            2.05,
            1.75,
            "Controller sees\nstate and context,\nthen reallocates\nlight and moisture load",
            fc="lower",
            ec="lower_edge",
            fontsize=8.35,
            weight="bold",
        ),
    )

    _arrow(ax, (3.35, 5.50), (4.25, 5.48), color="co2")
    _arrow(ax, (3.35, 3.44), (4.25, 2.12), color="water", rad=-0.10)
    _arrow(ax, (3.35, 3.44), (4.25, 3.80), color="heat", rad=0.10)
    _arrow(ax, (6.90, 5.48), (8.00, 5.40), color="data_edge")
    _arrow(ax, (6.90, 3.80), (8.00, 3.42), color="energy_edge")
    _arrow(ax, (6.90, 2.12), (8.00, 3.20), color="lower_edge", rad=0.10)
    _arrow(ax, (9.27, 4.85), (9.27, 3.97), color="energy_edge")
    _arrow(ax, (9.27, 2.85), (9.27, 2.07), color="line")
    _arrow(ax, (10.55, 3.42), (11.35, 3.42), color="lower_edge")
    _arrow(ax, (11.35, 2.88), (10.55, 5.25), color="lower_edge", rad=-0.15, ls="--")
    _arrow(ax, (11.35, 3.96), (10.55, 3.35), color="lower_edge", rad=0.08, ls="--")
    _arrow(ax, (11.35, 2.95), (10.55, 1.50), color="lower_edge", rad=0.12, ls="--")

    _box(
        ax,
        Box(
            0.85,
            0.75,
            5.25,
            0.74,
            "Mechanistic pathway: light -> photosynthesis and transpiration -> humidity pressure -> dehumidification electricity -> profit.",
            fc="panel",
            ec="line",
            fontsize=8.15,
            weight="bold",
        ),
    )
    _save(fig, "fig04_crop_environment_equipment_mechanism")


def build_evidence_map():
    fig, ax = _setup(14, 7.0)
    _label(ax, 0.42, 6.65, "e", size=13, weight="bold", color="ink")

    left = [
        ("System validation", "density and light response\nPID default operation", "plant", "plant_edge"),
        ("Upper optimization", "default vs best schedule\nsame lower controller", "upper", "upper_edge"),
        ("Lower optimization", "PID vs Residual-PID SAC\nsame schedule", "lower", "lower_edge"),
        ("Synergy", "best PID vs best Residual-PID SAC\nfrontier migration", "economy", "economy_edge"),
        ("Mechanism", "trajectory cases and humidity-load pathway", "data", "data_edge"),
        ("Extensions and ablations", "price, TOU, daily hold,\nfixed-light climate residual, no-context", "panel", "line"),
    ]
    y0 = 5.55
    for i, (head, body, fc, ec) in enumerate(left):
        y = y0 - i * 0.83
        _box(ax, Box(0.65, y, 3.15, 0.62, f"{head}\n{body}", fc=fc, ec=ec, fontsize=7.85, weight="bold"))
        _arrow(ax, (3.80, y + 0.31), (5.20, 3.28), color=ec, rad=(i - 2.5) * 0.035)

    _box(
        ax,
        Box(
            5.20,
            2.55,
            3.20,
            1.45,
            "Core claim\nbilevel closed-loop co-optimization\nof production schedule and environment control",
            fc="economy",
            ec="economy_edge",
            fontsize=9.0,
            weight="bold",
        ),
    )

    right = [
        ("Why upper layer?", "economic spread across 368 recipes"),
        ("Why lower layer?", "light reallocation and humidity-load reduction"),
        ("Why co-optimization?", "control changes the schedule frontier"),
        ("Why residual design?", "PID prior + learned deviations"),
    ]
    for i, (head, body) in enumerate(right):
        y = 5.15 - i * 1.05
        _box(ax, Box(9.55, y, 3.45, 0.76, f"{head}\n{body}", fc="panel", ec="line", fontsize=8.0, weight="bold"))
        _arrow(ax, (8.40, 3.28), (9.55, y + 0.38), color="line", rad=(1.5 - i) * 0.05)

    _box(
        ax,
        Box(
            5.45,
            0.70,
            2.70,
            0.78,
            "Result figures quantify effects;\nstructural figures define the system.",
            fc="data",
            ec="data_edge",
            fontsize=8.15,
            weight="bold",
        ),
    )
    _arrow(ax, (6.80, 2.55), (6.80, 1.48), color="data_edge", ls="--")
    _save(fig, "fig05_paper_evidence_map")


def main():
    build_bilevel_system()
    build_algorithm_workflow()
    build_residual_controller()
    build_physics_mechanism()
    build_evidence_map()
    print(f"Saved framework figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
