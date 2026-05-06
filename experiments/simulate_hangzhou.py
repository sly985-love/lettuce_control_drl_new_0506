# -*- coding: utf-8 -*-
"""
杭州气象数据驱动的植物工厂多控制器仿真评估脚本

功能:
  1. 基于 weather_hangzhou_2024.csv 气象数据驱动仿真
  2. 支持 PID / RL-Policy / Open-Loop 三种控制器类型
  3. 0420 主线仅开放上层排程参数接口 (t1, t2, N1, rho2)
  4. 开放仿真步长 --dt 和总仿真时长 --duration 设置
  5. 开放手动光强/光周期控制接口 (--I1_manual, --I2_manual, --photo_period_manual)
  6. 输出 CSV 日志 + 4×3 综合轨迹仪表盘 (单图合并展示)
  7. 支持加载已训练的 RL 策略进行评估
  8. 能耗堆叠图（区分加热/制冷）
  9. 批次生长轨迹细化图
  10. 采收月统计数据图
  11. 蒸腾量和呼吸量变化轨迹图

用法示例:
  # PID 控制，dt=10min，28天仿真
  python experiments/simulate_hangzhou.py --controller pid --start-date 2024-01-01 --duration 28 --dt 600

  # 0420 默认上层排程示例（PP 固定为 16 h，不写入 --schedule）
  python experiments/simulate_hangzhou.py --controller pid --schedule "t1=14,t2=14,N1=20,rho2=36" --dt 600 --duration 365

  # 若要做光周期敏感性分析，用手动覆盖而不是把 PP 当上层变量
  python experiments/simulate_hangzhou.py --controller pid --schedule "t1=14,t2=14,N1=20,rho2=36" --photo_period_manual 18 --dt 600 --duration 32

  # 加载 RL 策略评估
  python experiments/simulate_hangzhou.py --controller rl --load log/PFAL-contextual-SAC/sac_contextual/my_run --schedule "t1=14,t2=14,N1=20,rho2=36" --dt 600 --duration 32

  # 仅重新绘图（跳过仿真）
  python experiments/simulate_hangzhou.py --plots-only --controller pid
"""

import argparse
import csv
import datetime as dt
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import yaml

# Avoid Windows console encoding crashes when help text or logs contain Unicode.
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

# ── project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
WORKSPACE = str(ROOT)

from src.envs.plant_factory_env_new import PFALEnvContextual
from src.envs.utils import (
    apply_inlet_seedling_metadata,
    extract_inlet_seedling_metadata,
    has_inlet_seedling_metadata,
    load_all_configs,
    prepare_runtime_config,
)
from src.controllers.pfal_conventional_controller import PFALConventionalController
from src.models import (
    calculate_total_power,
    co2_density_to_ppm,
    absolute_humidity_to_relative,
    relative_humidity_to_vpd,
)
from src.utils import (
    apply_economic_overrides_to_config,
    list_tou_tariff_scenarios,
)

# ── controller config path ───────────────────────────────────────────────────
CTRL_CFG = ROOT / 'configs' / 'controller_params.yaml'

# ── economic constants ───────────────────────────────────────────────────────
_WEATHER_CONTAINER_CFG = None
_PLOT_CTRL_CFG = None
LEGACY_LOAD_INLET_PRESET = "external_nursery_proxy"
RL_RUN_CONFIG_OVERRIDE_KEYS = (
    'observation_semantics',
    'action_semantics',
    'residual_action_scale',
    'light_control_mode',
    'light_segments_per_photoperiod',
    'include_electricity_price_observation',
    'mask_schedule_context_observation',
    'pid_anchor_light',
    'enable_action_safety_projection',
    'safety_temp_guard_band_c',
    'safety_temp_projection_margin_c',
    'safety_temp_kp',
)


def _load_run_config_with_legacy_fallback(experiment_path: str) -> dict:
    """Load run_config.json and backfill legacy inlet metadata when absent."""
    run_cfg_path = Path(experiment_path) / 'run_config.json'
    run_cfg = {}
    if run_cfg_path.exists():
        try:
            with open(run_cfg_path, 'r', encoding='utf-8') as f:
                run_cfg = json.load(f) or {}
        except Exception as exc:
            print(f"[RL][WARN] Failed to parse run_config.json: {exc}")
            run_cfg = {}
    if not isinstance(run_cfg, dict):
        run_cfg = {}
    if not has_inlet_seedling_metadata(run_cfg):
        run_cfg = dict(run_cfg)
        run_cfg['initial_seedling_mass_preset'] = LEGACY_LOAD_INLET_PRESET
        run_cfg['inlet_seedling_metadata_source'] = str(
            run_cfg.get('inlet_seedling_metadata_source') or 'legacy_load_fallback'
        )
        print(
            "[RL][WARN] run_config inlet-seedling metadata missing; "
            f"falling back to legacy preset '{LEGACY_LOAD_INLET_PRESET}'."
        )
    return run_cfg


def _apply_saved_run_config_overrides(cfg: dict, run_cfg: dict | None) -> dict:
    """Apply saved run metadata onto a fresh project config."""
    resolved = apply_inlet_seedling_metadata(dict(cfg or {}), run_cfg or {})
    rl_cfg = dict(resolved.get('rl_params', {}) or {})
    for key in RL_RUN_CONFIG_OVERRIDE_KEYS:
        if isinstance(run_cfg, dict) and key in run_cfg:
            rl_cfg[key] = run_cfg[key]
    if isinstance(run_cfg, dict) and 'light_control_mode' not in run_cfg:
        rl_cfg['light_control_mode'] = 'step'
    if isinstance(run_cfg, dict) and 'light_segments_per_photoperiod' not in run_cfg:
        rl_cfg['light_segments_per_photoperiod'] = 3
    if rl_cfg:
        resolved['rl_params'] = rl_cfg
    return resolved


# ═══════════════════════════════════════════════════════════════════════════════
#  Weather I/O
# ═══════════════════════════════════════════════════════════════════════════════

def load_weather_csv(path: str) -> list[dict]:
    """解析杭州2024年每小时气象数据CSV为字典列表"""
    df = pd.read_csv(path, parse_dates=['DateTime'])
    rows = []
    for _, row in df.iterrows():
        rows.append({
            'dt': row['DateTime'].to_pydatetime(),
            'T_out': float(row['T(C)']),
            'CO2_ppm_out': float(row['CO2(ppm)']),
            'RH_out': float(row['RH(%)']) / 100.0,
        })
    return rows


def slice_weather(rows: list[dict], start: dt.datetime,
                  end: dt.datetime) -> list[dict]:
    """根据时间范围截取气象数据"""
    return [r for r in rows if start <= r['dt'] < end]


def expand_weather_for_dt(rows: list[dict], dt_s: float) -> list[dict]:
    """当dt小于1小时时，复制气象数据以匹配仿真步长"""
    if dt_s >= 3600.0:
        return rows
    factor = max(1, int(round(3600.0 / dt_s)))
    expanded = []
    for r in rows:
        for _ in range(factor):
            expanded.append(r.copy())
    return expanded


# ═══════════════════════════════════════════════════════════════════════════════
#  Schedule helpers
# ═══════════════════════════════════════════════════════════════════════════════

def areas_from_N1(N_total: int, N1: int):
    """根据总板数和密植区板数计算密植区(A1)和定植区(A2)面积"""
    N2 = N_total - N1
    A_board = 1.0
    A1 = N1 * A_board
    A2 = N2 * A_board
    return A1, A2, N2


def rho1_from_schedule(t1, t2, rho2, N1, N_total):
    """
    根据排程参数计算密植区种植密度 rho1

    与 enumerate_feasible_solutions.py 严格一致：
    rho1 = (rho2 * N2 * t1) / (N1 * t2)
    """
    N2 = N_total - N1
    return rho2 * N2 * t1 / (N1 * t2)


def build_schedule(t1, t2, N1, rho2, PP=None):
    """
    构建完整的排程参数字典

    0420 固定光周期框架下，上层排程只包含：
    x = {t1, t2, N1, rho2} in Z^4

    运行时仍保留固定 PP=16，或由手动光周期覆盖注入。
    """
    if PP is None:
        PP = 16
    N_total = 80
    A_board = 1.0
    N2 = N_total - N1
    A1 = N1 * A_board
    A2 = N2 * A_board
    rho1 = rho1_from_schedule(t1, t2, rho2, N1, N_total)
    return {
        't1': t1, 't2': t2, 'N1': N1, 'rho2': rho2, 'PP': PP,
        'rho1': rho1, 'N2': N2, 'A1': A1, 'A2': A2, 'A_total': A1 + A2,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Environment config builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_env_config(schedule: dict, dt_seconds: float, seed: int = 42,
                     photo_period_override: int = None,
                     light_control_mode: str = None,
                     light_segments_per_photoperiod: int = None,
                     action_semantics_override: str = None,
                     run_config_overrides: dict | None = None,
                     price_model_type: str | None = None,
                     tou_tariff_scenario: str | None = None,
                     electricity_price: float | None = None,
                     co2_price: float | None = None,
                     lettuce_price_fw: float | None = None,
                     constant_price: float | None = None):
    """
    构建PFALEnvContextual配置字典

    Parameters
    ----------
    schedule : dict
        排程参数字典，包含t1, t2, rho2, rho1, A1, A2等
    dt_seconds : float
        仿真步长（秒）
    seed : int
        随机种子
    photo_period_override : int, optional
        手动设置的光周期（小时），会覆盖schedule中的默认光周期设置
    """
    cfg = load_all_configs(str(ROOT / 'configs'))
    cfg = _apply_saved_run_config_overrides(cfg, run_config_overrides)
    run_cfg = dict(run_config_overrides or {})
    cfg = apply_economic_overrides_to_config(
        cfg,
        price_model_type=(
            price_model_type if price_model_type is not None else run_cfg.get('price_model_type')
        ),
        tou_tariff_scenario=(
            tou_tariff_scenario
            if tou_tariff_scenario is not None
            else run_cfg.get('tou_tariff_scenario')
        ),
        electricity_price=(
            electricity_price if electricity_price is not None else run_cfg.get('electricity_price')
        ),
        co2_price=co2_price if co2_price is not None else run_cfg.get('co2_price'),
        lettuce_price_fw=(
            lettuce_price_fw
            if lettuce_price_fw is not None
            else run_cfg.get('lettuce_price_fw')
        ),
        constant_price=(
            constant_price if constant_price is not None else run_cfg.get('constant_price')
        ),
    )
    schedule_cfg = dict(cfg.get('schedule_params', {}) or {})

    schedule_cfg.update({
        'rate_limit_I': float(schedule_cfg.get('rate_limit_I', 100.0)),
        'rate_limit_Q': float(schedule_cfg.get('rate_limit_Q', 42.4)),
        'rate_limit_CO2': float(schedule_cfg.get('rate_limit_CO2', 2.778e-5)),
        'rate_limit_m': float(schedule_cfg.get('rate_limit_m', 4.167e-6)),
        'max_continuous_light_hours': float(schedule_cfg.get('max_continuous_light_hours', 18.0)),
        'min_continuous_dark_hours': float(schedule_cfg.get('min_continuous_dark_hours', 6.0)),
    })

    # 如果手动设置了光周期，则覆盖默认配置
    if photo_period_override is not None:
        # 假设schedule中包含photo_period或使用默认的8小时黑夜
        cfg['photo_period_override'] = photo_period_override

    rl_cfg = dict(cfg.get('rl_params', {}) or {})
    if action_semantics_override is not None:
        rl_cfg['action_semantics'] = str(action_semantics_override)
    if light_control_mode is not None:
        rl_cfg['light_control_mode'] = str(light_control_mode)
    if light_segments_per_photoperiod is not None:
        rl_cfg['light_segments_per_photoperiod'] = max(
            1, int(light_segments_per_photoperiod)
        )
    if rl_cfg:
        cfg['rl_params'] = rl_cfg

    cfg['schedule_params'] = schedule_cfg
    cfg['_episode_length'] = None
    return prepare_runtime_config(
        cfg,
        schedule=schedule,
        seed=seed,
        dt=float(dt_seconds),
        photo_period_override=photo_period_override,
    )


def build_price_override_kwargs(
    *,
    price_model_type: str | None = None,
    tou_tariff_scenario: str | None = None,
    electricity_price: float | None = None,
    co2_price: float | None = None,
    lettuce_price_fw: float | None = None,
    constant_price: float | None = None,
) -> dict:
    """Build a reusable kwargs dict for economic scenario overrides."""
    return {
        'price_model_type': price_model_type,
        'tou_tariff_scenario': tou_tariff_scenario,
        'electricity_price': electricity_price,
        'co2_price': co2_price,
        'lettuce_price_fw': lettuce_price_fw,
        'constant_price': constant_price,
    }


def external_from_weather_row(row: dict) -> np.ndarray:
    """从气象数据行构建外部环境数组[T_out, xH_out, CO2_out]

    external 数组被 environment_model.py 的 CO2 动态方程使用：
      - T_out: 外部温度 [°C]
      - xH_out: 外部绝对湿度 [kg/m³]
      - C_out: 外部 CO2 密度 [kg/m³]

    注意：CO2 直接用 co2_ppm_to_density 转换，而非用 1000 ppm 参考值误乘 ppm/1e6。
    """
    from src.envs.utils import load_all_configs
    from src.models import co2_ppm_to_density, relative_humidity_to_absolute
    global _WEATHER_CONTAINER_CFG
    if _WEATHER_CONTAINER_CFG is None:
        _WEATHER_CONTAINER_CFG = load_all_configs(str(ROOT / 'configs')).get('container_params', {})
    T_out = float(row['T_out'])
    RH_out = float(row['RH_out'])
    CO2_ppm_out = float(row['CO2_ppm_out'])
    xH_out = relative_humidity_to_absolute(T_out, RH_out, _WEATHER_CONTAINER_CFG)
    C_out = co2_ppm_to_density(CO2_ppm_out, T_out)
    return np.array([T_out, xH_out, C_out], dtype=np.float32)


def _step_cost_components(env, power_breakdown: dict, u_co2_density: float, dt_s: float) -> tuple[float, float, float]:
    """Compute simulator electricity and CO2 cost from the env's shared economics."""
    dt_h = dt_s / 3600.0
    applied_elec_price = float(
        getattr(env, '_last_applied_elec_price', getattr(env, 'elec_price', env.c_elec))
    )
    elec_cost = power_breakdown.get('P_total', 0.0) / 1000.0 * dt_h * applied_elec_price
    co2_cost = abs(u_co2_density) * float(env.A_total) * dt_s * float(env.c_CO2)
    return elec_cost + co2_cost, elec_cost, co2_cost


# ═══════════════════════════════════════════════════════════════════════════════
#  RL Policy Loader
# ═══════════════════════════════════════════════════════════════════════════════

def load_rl_policy(
    experiment_path: str,
    device: str = "cpu",
    checkpoint: str = "auto",
):
    """
    从实验日志目录加载已训练的SAC策略

    Parameters
    ----------
    experiment_path : str
        实验日志目录路径，例如 "log/PFAL-contextual-SAC/sac_contextual/exp_0401_133814"
    device : str
        "cpu" 或 "cuda"

    Returns
    -------
    policy : tianshou SACPolicy
    action_space : gym.spaces.Box
    """
    # 这里是“推理/评估”而不是在线追踪实验，显式关闭 wandb，
    # 并安装一个轻量 stub，避免 Tianshou 间接导入 wandb 后在 Windows 下
    # 留下临时目录清理告警。
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_SILENT", "true")
    if "wandb" not in sys.modules:
        import importlib.machinery
        import types

        def _noop(*args, **kwargs):
            return None

        class _DummyRun:
            def __init__(self) -> None:
                self.project = "disabled"
                self.name = "disabled"
                self.url = ""

            def log(self, *args, **kwargs):
                return None

            def finish(self):
                return None

        wandb_stub = types.ModuleType("wandb")
        wandb_stub.init = lambda *args, **kwargs: _DummyRun()
        wandb_stub.log = _noop
        wandb_stub.finish = _noop
        wandb_stub.define_metric = _noop
        wandb_stub.Image = lambda *args, **kwargs: None
        wandb_stub.Artifact = lambda *args, **kwargs: None
        wandb_stub.Table = lambda *args, **kwargs: None
        wandb_stub.config = {}
        wandb_stub.run = None
        wandb_stub.__file__ = "<wandb_stub>"
        wandb_stub.__spec__ = importlib.machinery.ModuleSpec(
            name="wandb",
            loader=None,
        )
        sys.modules["wandb"] = wandb_stub

    sys.path.insert(0, WORKSPACE)
    sys.path.insert(0, str(ROOT / 'src'))

    import torch
    from tianshou.policy import SACPolicy
    from tianshou.utils.net.common import Net
    from tianshou.utils.net.continuous import ActorProb, Critic
    from envs.plant_factory_env_new import PFALEnvContextual
    from envs.utils import load_all_configs
    from rl.drl_based_control import resolve_policy_checkpoint_path

    # 构建完整配置字典，使PFALEnvContextual具有正确的观测/动作空间
    cfg_dir = str(ROOT / 'configs')
    cfg = load_all_configs(cfg_dir)
    hidden_sizes = [128, 128]
    run_cfg = _load_run_config_with_legacy_fallback(experiment_path)
    if isinstance(run_cfg, dict):
        cfg = _apply_saved_run_config_overrides(cfg, run_cfg)
        if isinstance(run_cfg.get('hidden_sizes'), list) and run_cfg['hidden_sizes']:
            hidden_sizes = [int(v) for v in run_cfg['hidden_sizes']]
        inlet_meta = extract_inlet_seedling_metadata(cfg)
        print(
            "[RL] Restored inlet seedling metadata: "
            f"preset={inlet_meta['initial_seedling_mass_preset']} "
            f"basis={inlet_meta['initial_seedling_mass_basis']}"
        )
    # Build schedule from schedule_params.yaml bounds (same as _sample_context_schedule uses)
    # Default schedule parameters for env construction
    import yaml as _yaml
    _sp_path = ROOT / 'configs' / 'schedule_params.yaml'
    _sp_cfg = {}
    if _sp_path.exists():
        with open(str(_sp_path), 'r', encoding='utf-8') as _f:
            _sp_cfg = _yaml.safe_load(_f) or {}
    # 默认上层排程语义: x={t1, t2, N1, rho2}={14, 14, 20, 36}; PP 在 0420 中固定为 16 h
    _N_total = 80
    _A_board = 1.0
    _N1_default = 20
    _N2_default = _N_total - _N1_default
    _A1 = _N1_default * _A_board
    _A2 = _N2_default * _A_board
    _rho2_default = 36.0
    _t1_default = 14
    _t2_default = 14
    _PP_default = 16
    # rho1 derived from mass flow conservation: rho1 = (rho2 * N2 * t1) / (N1 * t2)
    _rho1_default = _rho2_default * _N2_default * _t1_default / (_N1_default * _t2_default)
    _default_sched = {
        't1': _t1_default, 't2': _t2_default,
        'N1': _N1_default, 'rho2': _rho2_default, 'PP': _PP_default,
        'rho1': _rho1_default,
        'N2': _N2_default, 'A1': _A1, 'A2': _A2,
        'A_total': _A1 + _A2,
    }
    cfg['schedule'] = _default_sched
    cfg['seed'] = 42
    cfg['dt'] = 3600.0

    env_tmp = PFALEnvContextual(config=cfg)

    state_shape = env_tmp.observation_space.shape
    action_shape = env_tmp.action_space.shape
    max_action = float(env_tmp.action_space.high[0])

    # 构建与训练时相同架构的网络
    net_a = Net(state_shape, hidden_sizes=hidden_sizes, device=device)
    actor = ActorProb(
        net_a,
        action_shape,
        max_action=max_action,
        device=device,
        unbounded=True,
        conditioned_sigma=True,
    ).to(device)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=3e-4)

    net_c1 = Net(state_shape, action_shape, hidden_sizes=hidden_sizes,
                 concat=True, device=device)
    critic1 = Critic(net_c1, device=device).to(device)
    critic1_opt = torch.optim.Adam(critic1.parameters(), lr=3e-4)

    net_c2 = Net(state_shape, action_shape, hidden_sizes=hidden_sizes,
                 concat=True, device=device)
    critic2 = Critic(net_c2, device=device).to(device)
    critic2_opt = torch.optim.Adam(critic2.parameters(), lr=3e-4)

    if device == "cuda":
        alpha = torch.tensor(0.0, device=device)
    else:
        alpha = torch.tensor(0.0)

    policy = SACPolicy(
        actor=actor,
        critic1=critic1,
        critic2=critic2,
        actor_optim=actor_opt,
        critic1_optim=critic1_opt,
        critic2_optim=critic2_opt,
        alpha=alpha,
        action_space=env_tmp.action_space,
        observation_space=env_tmp.observation_space,
    )

    policy_path, resolved_checkpoint = resolve_policy_checkpoint_path(
        experiment_path,
        checkpoint=checkpoint,
        project_root=ROOT,
    )

    try:
        state_dict = torch.load(
            str(policy_path),
            map_location=torch.device(device),
            weights_only=True,
        )
    except TypeError:
        state_dict = torch.load(str(policy_path), map_location=torch.device(device))
    policy.load_state_dict(state_dict)
    policy.eval()
    print(f"[RL] Loaded checkpoint={resolved_checkpoint} from {policy_path}")
    return policy, env_tmp.action_space, run_cfg


# ═══════════════════════════════════════════════════════════════════════════════
#  Simulation core
# ═══════════════════════════════════════════════════════════════════════════════

def _run_pid_simulation(env, weather_rows, schedule, dt_s, seed,
                        I1_manual=None, I2_manual=None, photo_period_manual=None):
    """
    基于PFALConventionalController的PID控制仿真循环

    Parameters
    ----------
    env : PFALEnvContextual
        植物工厂仿真环境
    weather_rows : list[dict]
        气象数据列表
    schedule : dict
        排程参数字典
    dt_s : float
        仿真步长（秒）
    seed : int
        随机种子
    I1_manual : float, optional
        手动设置的密植区光强（μmol/m²/s），覆盖PID输出
    I2_manual : float, optional
        手动设置的定植区光强（μmol/m²/s），覆盖PID输出
    photo_period_manual : float, optional
        手动设置的光周期（小时），覆盖随机采样

    Returns
    -------
    records : list[dict]
        每步仿真记录列表
    batch_trajectory_records : list[dict]
        每步各批次生长轨迹记录列表
    """
    ext0 = external_from_weather_row(weather_rows[0])
    reset_options = {'external': ext0, 'hour_of_day': 0}
    if photo_period_manual is not None:
        reset_options['photo_period'] = photo_period_manual
    env.reset(seed=seed, options=reset_options)

    # 确保episode运行完整气象周期，最少(t1+t2)个循环
    min_needed = int((schedule['t1'] + schedule['t2']) * 24 * 3600.0 / dt_s)
    env.episode_length = max(len(weather_rows), min_needed)

    # PFALConventionalController 直接接收 env 对象（从 env 读取温度带、CO2 带、湿度带等）
    ctrl = PFALConventionalController(env)
    ctrl.reset()

    records = []
    batch_trajectory_records = []
    t_start = weather_rows[0]['dt']
    cum_reward = 0.0
    cum_cost   = 0.0
    cum_transplants = 0
    cum_harvests = 0

    for step_i, wrow in enumerate(weather_rows):
        env.external = external_from_weather_row(wrow)
        obs_phys = env._get_observation()
        action_raw = ctrl.predict(obs_phys, context=None)
        action_5d = np.array(action_raw[:5], dtype=np.float32)

        # PFALConventionalController.predict() returns normalised actions in [-1, 1].
        # When manually overriding PPFD, first recover the physical action vector,
        # then replace the requested light channels, and finally re-normalise.
        if I1_manual is not None or I2_manual is not None:
            al, ah = env._get_action_physical_bounds()
            span = np.maximum(ah - al, 1e-12)
            action_phys = al + 0.5 * (action_5d + 1.0) * span

            if I1_manual is not None:
                action_phys[0] = float(I1_manual)
            if I2_manual is not None:
                action_phys[1] = float(I2_manual)

            action_phys = np.clip(action_phys, al, ah)
            action_5d = ((action_phys - al) / span) * 2.0 - 1.0
            action_5d = np.clip(action_5d, -1.0, 1.0).astype(np.float32)
        
        obs, reward, done, trunc, step_info = env.step(action_5d)

        # 获取实际执行的物理动作（用于能耗计算）
        # 在env.step内部会调用_denormalise_action，应用物理界限和速率限制
        phys_action = env.prev_action_4d
        I1p, I2p = float(phys_action[0]), float(phys_action[1])
        Qp = float(phys_action[2])
        up = float(phys_action[3])
        mp = float(phys_action[4])

        # 计算各分项功率（含加热/制冷分离）
        bp = calculate_total_power(
            np.array([I1p, I2p, Qp, up, mp]),
            env.A1, env.A2, env.equipment_params)
        step_cost, _, _ = _step_cost_components(env, bp, up, dt_s)

        cum_reward += reward
        cum_cost   += step_cost

        # 累计采收/移栽以 batch_manager 为准（info 中无 transplant_events）
        cum_transplants = int(env.batch_manager.total_transplants)
        cum_harvests = int(env.batch_manager.total_harvests)

        # 记录每步的各批次生长数据
        step_batch_records = _extract_batch_trajectory(env, step_i, t_start, dt_s)
        batch_trajectory_records.extend(step_batch_records)

        record = _build_record(env, wrow, phys_action, reward,
                               cum_reward, cum_cost,
                               cum_transplants, cum_harvests,
                               schedule, dt_s, step_i, t_start,
                               bp, step_info)
        records.append(record)

        if done:
            break

    return records, batch_trajectory_records


def _extract_batch_trajectory(env, step_i, t_start, dt_s):
    """
    提取当前步所有批次的生长轨迹数据

    Parameters
    ----------
    env : PFALEnvContextual
    step_i : int
        当前步索引
    t_start : datetime
        仿真开始时间
    dt_s : float
        仿真步长（秒）

    Returns
    -------
    list[dict] : 各批次的生长数据记录
    """
    from src.models import co2_ppm_to_density, absolute_humidity_to_relative

    records = []
    elapsed_h = step_i * dt_s / 3600.0
    elapsed_d = elapsed_h / 24.0
    current_dt = t_start + dt.timedelta(seconds=step_i * dt_s)

    # 获取batch_manager中的所有批次信息
    if not hasattr(env, 'batch_manager') or env.batch_manager is None:
        return records

    bm = env.batch_manager

    # 获取所有批次的干重密度 [kg/m²] -> [g/m²]
    # batch_info包含: {'region': 'seedling'/'transplant', 'pipeline_slot': int,
    #                  'batch_id': int, 'age_h': float, 'xDn': float, 'xDs': float, 'LAI': float}
    records = []
    for region in ['seedling', 'transplant']:
        region_batches = getattr(bm, f'{region}_batches', [])
        for slot in range(len(region_batches)):
            # 尝试从batch_manager获取批次信息
            if slot < len(region_batches):
                batch = region_batches[slot]
                if batch is not None and hasattr(batch, 'xDn') and hasattr(batch, 'xDs'):
                    xDn = float(batch.xDn) if hasattr(batch.xDn, '__float__') else float(batch.xDn)
                    xDs = float(batch.xDs) if hasattr(batch.xDs, '__float__') else float(batch.xDs)
                    xD_total = (xDn + xDs) * 1000.0  # 转换为 g/m²
                    age_h = float(batch.age_h) if hasattr(batch, 'age_h') else 0.0
                    LAI = float(batch.LAI) if hasattr(batch, 'LAI') else 0.0
                    pipeline_slot = int(getattr(batch, 'pipeline_slot', slot))
                    
                    records.append({
                        'datetime': current_dt.isoformat(),
                        'elapsed_d': round(elapsed_d, 4),
                        'step': step_i,
                        'region': region,
                        'region_semantic': 'dense' if region == 'seedling' else 'finishing',
                        'pipeline_slot': pipeline_slot,
                        'batch_id': getattr(batch, 'batch_id', slot),
                        'age_h': round(age_h, 4),
                        'xD_total_g_m2': round(xD_total, 6),
                        'LAI': round(LAI, 4),
                    })

    return records


def _override_action_norm_with_manual_lights(
    env,
    action_norm,
    *,
    I1_manual=None,
    I2_manual=None,
):
    """Project a policy action to the same semantics after forcing light channels.

    This is needed for fair PID-vs-RL evaluation when the RL policy uses
    residual semantics rather than absolute physical actions.
    """
    action_norm = np.asarray(action_norm, dtype=np.float32).ravel()[:5]
    if I1_manual is None and I2_manual is None:
        return np.clip(action_norm, -1.0, 1.0).astype(np.float32)

    desired_phys = np.asarray(env._denormalise_action(action_norm), dtype=np.float32).copy()

    light_on = True
    if env.tvp is not None and getattr(env, "_i_max", 0) > 0:
        photo_idx = int(env.time_step) % env._i_max
        light_on = bool(env.tvp[photo_idx, 2])

    if light_on:
        if I1_manual is not None:
            desired_phys[0] = float(I1_manual)
        if I2_manual is not None:
            desired_phys[1] = float(I2_manual)

    al, ah = env._get_action_physical_bounds()
    span = np.maximum(ah - al, 1e-12)
    desired_phys = np.clip(desired_phys, al, ah).astype(np.float32)
    effective_norm = ((desired_phys - al) / span) * 2.0 - 1.0
    effective_norm = np.clip(effective_norm, -1.0, 1.0).astype(np.float32)

    semantics = str(getattr(env, "action_semantics", "absolute")).lower()
    if semantics in {"residual_pid", "pid_residual", "residual_pid_gated", "pid_residual_gated"}:
        anchor_norm = np.asarray(env._get_action_anchor_norm(), dtype=np.float32)
        if semantics in {"residual_pid", "pid_residual"}:
            gate = np.ones(5, dtype=np.float32)
        else:
            gate = np.asarray(
                env._compute_residual_gate().get("actuator_vector", np.ones(5)),
                dtype=np.float32,
            )
        scale = np.asarray(getattr(env, "_residual_action_scale", np.ones(5)), dtype=np.float32)
        denom = gate * scale
        solved = np.asarray(action_norm, dtype=np.float32).copy()
        mask = np.abs(denom) > 1e-8
        solved[mask] = (effective_norm[mask] - anchor_norm[mask]) / denom[mask]
        action_norm = solved
    else:
        action_norm = effective_norm

    return np.clip(action_norm, -1.0, 1.0).astype(np.float32)


def _dehum_power_kW_from_frame(df: pd.DataFrame, c_dehum_eev: float = 3.0) -> pd.Series:
    """Reconstruct dehumidifier electric power from the physical dehumidification rate."""
    if {'m_dehum', 'A1', 'A2'}.issubset(df.columns):
        a_total = df['A1'].astype(float) + df['A2'].astype(float)
        return df['m_dehum'].astype(float) * a_total * 3600.0 / max(float(c_dehum_eev), 1e-12)
    if 'P_dehum_kW' in df.columns:
        return df['P_dehum_kW'].astype(float)
    return pd.Series(np.zeros(len(df)), index=df.index, dtype=float)


def _ensure_hvac_split_power_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure HVAC heating/cooling electric power columns exist for plotting."""
    out = df.copy()
    zero = pd.Series(np.zeros(len(out)), index=out.index, dtype=float)

    q_hvac = out['Q_HVAC'].astype(float) if 'Q_HVAC' in out.columns else zero
    p_total = out['P_HVAC_kW'].astype(float).clip(lower=0.0) if 'P_HVAC_kW' in out.columns else zero

    if 'P_heating_kW' in out.columns:
        p_heating = out['P_heating_kW'].astype(float).clip(lower=0.0)
    else:
        p_heating = p_total.where(q_hvac > 0.0, 0.0)

    if 'P_cooling_kW' in out.columns:
        p_cooling = out['P_cooling_kW'].astype(float).clip(lower=0.0)
    else:
        p_cooling = p_total.where(q_hvac < 0.0, 0.0)

    out['P_heating_kW'] = p_heating
    out['P_cooling_kW'] = p_cooling
    out['P_HVAC_kW'] = p_heating + p_cooling
    return out


def _run_rl_simulation(env, weather_rows, schedule, dt_s, seed,
                      policy, explore=False, photo_period_manual=None,
                      I1_manual=None, I2_manual=None):
    """
    基于SAC策略的强化学习控制仿真循环

    Parameters
    ----------
    env : PFALEnvContextual
    weather_rows : list[dict]
    schedule : dict
    dt_s : float
    seed : int
    policy : SACPolicy
    explore : bool
        是否对策略输出添加随机采样
    photo_period_manual : int, optional
        手动设置的光周期（小时），会注入到schedule中并通过env.reset生效
    I1_manual : float, optional
        手动设置的密植区光强（μmol/m²/s），覆盖RL输出，用于公平评估
    I2_manual : float, optional
        手动设置的定植区光强（μmol/m²/s），覆盖RL输出，用于公平评估

    Returns
    -------
    records : list[dict]
    batch_trajectory_records : list[dict]
    """
    import torch  # 延迟导入，仅在RL模式下需要
    torch.manual_seed(seed)
    np.random.seed(seed)
    ext0 = external_from_weather_row(weather_rows[0])

    # 如果手动设置了光周期，将其注入schedule
    if photo_period_manual is not None:
        schedule_copy = schedule.copy()
        schedule_copy['PP'] = int(photo_period_manual)
    else:
        schedule_copy = schedule

    reset_options = {'external': ext0, 'hour_of_day': 0, 'schedule': schedule_copy}
    if photo_period_manual is not None:
        reset_options['photo_period'] = int(photo_period_manual)
    env.reset(seed=seed, options=reset_options)

    # 确保episode运行完整气象周期，最少(t1+t2)个循环
    min_needed = int((schedule['t1'] + schedule['t2']) * 24 * 3600.0 / dt_s)
    env.episode_length = max(len(weather_rows), min_needed)

    records = []
    batch_trajectory_records = []
    t_start = weather_rows[0]['dt']
    cum_reward = 0.0
    cum_cost   = 0.0
    cum_transplants = 0
    cum_harvests = 0

    for step_i, wrow in enumerate(weather_rows):
        env.external = external_from_weather_row(wrow)

        # 获取26维归一化观测
        obs_26d = env._get_observation()
        obs_26d = np.asarray(obs_26d, dtype=np.float32)

        # 策略前向传播 - 从学习到的分布中采样
        with torch.no_grad():
            obs_t = torch.from_numpy(obs_26d).float().to(policy.actor.device)
            batch_t = obs_t.unsqueeze(0)
            (mu_t, sigma_t), _state = policy.actor(batch_t)
            if explore:
                action_t = torch.distributions.Normal(mu_t, sigma_t).sample()
            else:
                action_t = mu_t
            action_t = action_t.detach().float()
            action_np = action_t.squeeze(0).cpu().numpy()

        action_5d = np.clip(action_np, -1.0, 1.0).astype(np.float32)
        action_5d = _override_action_norm_with_manual_lights(
            env,
            action_5d,
            I1_manual=I1_manual,
            I2_manual=I2_manual,
        )

        obs, reward, done, trunc, step_info = env.step(action_5d)

        # env.step()内部调用_denormalise_action进行：
        # 1. 物理界限约束 (al/ah from env.equipment_params)
        # 2. 速率限制
        # 3. 光周期遮蔽（在超出连续光照时长限制时强制I1=I2=CO2=0）
        # 使用env.prev_action_4d（step内部设置）进行准确的能耗计算
        phys_action = env.prev_action_4d
        I1p, I2p = float(phys_action[0]), float(phys_action[1])
        Qp = float(phys_action[2])
        up = float(phys_action[3])
        mp = float(phys_action[4])

        bp = calculate_total_power(
            np.array([I1p, I2p, Qp, up, mp]), env.A1, env.A2, env.equipment_params)
        step_cost, _, _ = _step_cost_components(env, bp, up, dt_s)

        cum_reward += reward
        cum_cost   += step_cost

        cum_transplants = int(env.batch_manager.total_transplants)
        cum_harvests = int(env.batch_manager.total_harvests)

        # 记录每步的各批次生长数据
        step_batch_records = _extract_batch_trajectory(env, step_i, t_start, dt_s)
        batch_trajectory_records.extend(step_batch_records)

        record = _build_record(env, wrow, phys_action, reward,
                               cum_reward, cum_cost,
                               cum_transplants, cum_harvests,
                               schedule, dt_s, step_i, t_start,
                               bp, step_info)
        records.append(record)

        if done:
            break

    return records, batch_trajectory_records


def _run_openloop_simulation(env, weather_rows, schedule, dt_s, seed):
    """
    开环固定动作仿真：I1=I2=200, Q_HVAC=0, u_CO2=0, m_dehum=0

    Returns
    -------
    records : list[dict]
    batch_trajectory_records : list[dict]
    """
    ext0 = external_from_weather_row(weather_rows[0])
    env.reset(seed=seed, options={'external': ext0, 'hour_of_day': 0})

    # 确保episode运行完整气象周期，最少(t1+t2)个循环
    min_needed = int((schedule['t1'] + schedule['t2']) * 24 * 3600.0 / dt_s)
    env.episode_length = max(len(weather_rows), min_needed)

    records = []
    batch_trajectory_records = []
    t_start = weather_rows[0]['dt']
    cum_reward = 0.0
    cum_cost   = 0.0
    cum_transplants = 0
    cum_harvests = 0

    # 开环固定动作物理值
    action_phys = np.array([200.0, 200.0, 0.0, 0.0, 0.0], dtype=np.float32)

    for step_i, wrow in enumerate(weather_rows):
        env.external = external_from_weather_row(wrow)

        # 将物理值转换为归一化[-1,1]
        al, ah = env._get_action_physical_bounds()
        action_norm = ((action_phys - al) / (ah - al)) * 2.0 - 1.0
        action_5d = np.clip(action_norm, -1.0, 1.0).astype(np.float32)

        obs, reward, done, trunc, step_info = env.step(action_5d)

        # 获取实际执行的物理动作
        phys_action = env.prev_action_4d

        bp = calculate_total_power(
            np.array([200.0, 200.0, 0.0, 0.0, 0.0]),
            env.A1, env.A2, env.equipment_params,
        )
        step_cost, _, _ = _step_cost_components(env, bp, 0.0, dt_s)

        cum_reward += reward
        cum_cost   += step_cost

        cum_transplants = int(env.batch_manager.total_transplants)
        cum_harvests = int(env.batch_manager.total_harvests)

        # 记录每步的各批次生长数据
        step_batch_records = _extract_batch_trajectory(env, step_i, t_start, dt_s)
        batch_trajectory_records.extend(step_batch_records)

        record = _build_record(env, wrow, phys_action, reward,
                               cum_reward, cum_cost,
                               cum_transplants, cum_harvests,
                               schedule, dt_s, step_i, t_start,
                               bp, step_info)
        records.append(record)

        if done:
            break

    return records, batch_trajectory_records


def _harvest_dry_mass_g_from_info(step_info) -> float:
    """
    Harvest mass exposed by the environment is dry mass [g].

    The legacy CSV field name `harvest_mass_g` is preserved for compatibility,
    but it should be interpreted as harvested dry mass.
    """
    if not step_info:
        return 0.0
    if 'harvest_mass_g' in step_info:
        return float(step_info['harvest_mass_g'])
    return float(step_info.get('harvest_mass', 0.0))


def _dry_to_fresh_multiplier(env) -> float:
    crop_cfg = dict(getattr(env, 'crop_params', {}) or {})
    if 'c_fw' in crop_cfg:
        return float(crop_cfg['c_fw'])
    c_d2f = float(getattr(env, 'c_d2f', 0.0) or 0.0)
    if c_d2f > 1e-12:
        return 1.0 / c_d2f
    return 22.5


def _infer_c_fw_from_frame(df: pd.DataFrame, default: float = 22.5) -> float:
    if (
        'harvest_mean_dry_mass_per_plant_g' in df.columns
        and 'harvest_mean_fresh_mass_per_plant_g' in df.columns
    ):
        dry = df['harvest_mean_dry_mass_per_plant_g'].astype(float)
        fresh = df['harvest_mean_fresh_mass_per_plant_g'].astype(float)
        mask = (dry > 1e-9) & (fresh > 1e-9)
        if mask.any():
            ratio = (fresh[mask] / dry[mask]).replace([np.inf, -np.inf], np.nan).dropna()
            if not ratio.empty:
                return float(ratio.median())
    return float(default)


def _ensure_harvest_metrics_frame(df: pd.DataFrame, default_c_fw: float = 22.5) -> pd.DataFrame:
    out = df.copy()
    if 'harvest_dry_mass_g' not in out.columns:
        out['harvest_dry_mass_g'] = out.get('harvest_mass_g', 0.0)
    c_fw = _infer_c_fw_from_frame(out, default=default_c_fw)
    if 'harvest_fresh_mass_equiv_g' not in out.columns:
        out['harvest_fresh_mass_equiv_g'] = out['harvest_dry_mass_g'].astype(float) * c_fw
    if 'harvest_mean_dry_mass_per_plant_g' not in out.columns:
        out['harvest_mean_dry_mass_per_plant_g'] = 0.0
    if 'harvest_mean_fresh_mass_per_plant_g' not in out.columns:
        out['harvest_mean_fresh_mass_per_plant_g'] = (
            out['harvest_mean_dry_mass_per_plant_g'].astype(float) * c_fw
        )
    return out


def _build_record(env, wrow, action_raw, reward,
                  cum_reward, cum_cost,
                  cum_transplants, cum_harvests,
                  schedule, dt_s, step_i, t_start,
                  power_breakdown, step_info=None):
    """
    提取所有单步指标到字典中

    Parameters
    ----------
    env : PFALEnvContextual
    wrow : dict
        气象数据行
    action_raw : np.ndarray
        物理动作值 [I1, I2, Q_HVAC, u_CO2, m_dehum]
    reward : float
        当前步奖励
    cum_reward : float
        累计奖励
    cum_cost : float
        累计成本
    cum_transplants : int
        累计移栽次数
    cum_harvests : int
        累计采收次数
    schedule : dict
        排程参数字典
    dt_s : float
        仿真步长（秒）
    step_i : int
        当前步索引
    t_start : datetime
        仿真开始时间
    power_breakdown : dict
        calculate_total_power返回的分项功率字典
    step_info : dict, optional
        env.step返回的额外信息

    Returns
    -------
    dict : 单步记录字典
    """
    from src.models import co2_ppm_to_density, absolute_humidity_to_vpd

    # 环境状态
    C  = float(env.state[0])
    T  = float(env.state[1])
    xH = float(env.state[2])
    RH = absolute_humidity_to_relative(T, xH, env.container_params)
    VPD_kPa = absolute_humidity_to_vpd(T, xH, env.container_params)
    CO2_ppm = co2_density_to_ppm(C, T)

    # 物理动作值
    I1_f, I2_f = float(action_raw[0]), float(action_raw[1])
    Q_HVAC_f   = float(action_raw[2])
    u_CO2_f    = float(action_raw[3])
    m_dehum_f  = float(action_raw[4])

    # LED功率 [W/m²]
    led_ppe = float(getattr(env, 'c_led_ppe', 2.5))
    P_led1 = max(0.0, I1_f) / led_ppe
    P_led2 = max(0.0, I2_f) / led_ppe

    # 功率分项 (W -> kW)
    breakdown_kW = {k: v / 1000.0 for k, v in power_breakdown.items()}
    A1 = float(getattr(env, 'A1', 0.0) or 0.0)
    A2 = float(getattr(env, 'A2', 0.0) or 0.0)
    A_tot = max(A1 + A2, 1e-12)
    env_step_diag = dict((step_info or {}).get('env_step_diagnostics', {}) or {})
    applied_elec_price = float(
        (step_info or {}).get(
            'elec_price',
            getattr(env, '_last_applied_elec_price', getattr(env, 'elec_price', env.c_elec)),
        )
    )

    # 加热/制冷功率分离
    P_heating_kW = breakdown_kW.get('P_heating', 0.0)
    P_cooling_kW = breakdown_kW.get('P_cooling', 0.0)

    dt_h = dt_s / 3600.0
    E_kWh = breakdown_kW.get('P_total', 0.0) * dt_h
    _, c_elec, c_co2 = _step_cost_components(env, power_breakdown, u_CO2_f, dt_s)

    # 生物量 / LAI：get_state_summary() 不含密度与 LAI，须用集总特征
    bm_lump = env.batch_manager._extract_lumped_features()
    c_fw = _dry_to_fresh_multiplier(env)
    harvest_dry_mass = _harvest_dry_mass_g_from_info(step_info)
    harvest_mean_dry = float((step_info or {}).get('harvest_mean_dry_mass_per_plant_g', 0.0))
    harvest_mean_fresh = float((step_info or {}).get('harvest_mean_fresh_mass_per_plant_g', 0.0))
    if harvest_mean_fresh <= 0.0 and harvest_mean_dry > 0.0:
        harvest_mean_fresh = harvest_mean_dry * c_fw
    harvest_fresh_mass = harvest_dry_mass * c_fw
    d_tr = float(bm_lump.get('density_transplant', 0.0))
    d_se = float(bm_lump.get('density_seedling', 0.0))
    biomass_trans_kg_m2 = d_tr / 1000.0
    biomass_seed_kg_m2 = d_se / 1000.0
    biomass_tot_kg_m2 = (d_se * A1 + d_tr * A2) / A_tot / 1000.0
    lai_tr = float(bm_lump.get('lai_transplant', 0.0))
    lai_se = float(bm_lump.get('lai_seedling', 0.0))

    # 蒸腾/呼吸：由环境同步的速率 [kg/s] × 步长 → 本步质量（info 中通常不提供）
    E_transp_kg = float(getattr(env, 'total_E', 0.0)) * dt_s
    E_resp_kg = float(getattr(env, 'total_R', 0.0)) * dt_s
    dehum_removed_kg = m_dehum_f * A_tot * dt_s

    elapsed_h = step_i * dt_s / 3600.0
    elapsed_d = elapsed_h / 24.0
    current_dt = t_start + dt.timedelta(seconds=step_i * dt_s)

    return {
        'step': step_i,
        'elapsed_h': round(elapsed_h, 4),
        'elapsed_d': round(elapsed_d, 4),
        'datetime': current_dt.isoformat(),
        'T_in': round(T, 4),
        'RH_pct': round(RH * 100.0, 4),
        'VPD_kPa': round(VPD_kPa, 4),
        'C_ppm': round(CO2_ppm, 2),
        'T_out': round(wrow['T_out'], 4),
        'RH_out_pct': round(wrow['RH_out'] * 100.0, 4),
        'C_out_ppm': round(wrow['CO2_ppm_out'], 2),
        'I1': round(I1_f, 4),
        'I2': round(I2_f, 4),
        'I_dense': round(I1_f, 4),
        'I_finishing': round(I2_f, 4),
        'Q_HVAC': round(Q_HVAC_f, 4),
        'u_CO2': round(u_CO2_f, 10),
        'm_dehum': round(m_dehum_f, 10),
        'P_LED1_W_m2': round(P_led1, 4),
        'P_LED2_W_m2': round(P_led2, 4),
        'P_LED_dense_W_m2': round(P_led1, 4),
        'P_LED_finishing_W_m2': round(P_led2, 4),
        'P_LED_total_kW': round(breakdown_kW.get('P_led_total', 0.0), 4),
        'P_HVAC_kW': round(breakdown_kW.get('P_hvac_total', 0.0), 4),
        'P_heating_kW': round(P_heating_kW, 4),
        'P_cooling_kW': round(P_cooling_kW, 4),
        'P_CO2_kW': 0.0,   # CO₂是气体成本，非电功率
        'P_dehum_kW': round(breakdown_kW.get('P_dehum', 0.0), 4),
        'P_total_kW': round(breakdown_kW.get('P_total', 0.0), 4),
        'elec_price_rmb_kwh': round(applied_elec_price, 6),
        'E_step_kWh': round(E_kWh, 6),
        'cost_CO2_rmb': round(c_co2, 6),
        'cost_elec_rmb': round(c_elec, 6),
        'cost_total_rmb': round(c_elec + c_co2, 6),
        'E_transp_kg': round(E_transp_kg, 6),
        'E_resp_kg': round(E_resp_kg, 6),
        'dehum_removed_kg': round(dehum_removed_kg, 6),
        'env_condensation_removed_kg': round(
            float(env_step_diag.get('condensation_removed_kg', 0.0)), 6
        ),
        'Q_led_driver_loss_W': round(
            float(env_step_diag.get('Q_led_driver_loss_W', 0.0)), 4
        ),
        'Q_led_room_upper_W': round(
            float(env_step_diag.get('Q_led_room_upper_W', 0.0)), 4
        ),
        'Q_biomass_storage_W': round(
            float(env_step_diag.get('Q_biomass_storage_W', 0.0)), 4
        ),
        'Q_led_room_W': round(
            float(env_step_diag.get('Q_led_room_W', 0.0)), 4
        ),
        'Q_dehum_latent_to_room_W': round(
            float(env_step_diag.get('Q_dehum_latent_to_room_W', 0.0)), 4
        ),
        'Q_dehum_electric_to_room_W': round(
            float(env_step_diag.get('Q_dehum_electric_to_room_W', 0.0)), 4
        ),
        'Q_dehum_total_to_room_W': round(
            float(env_step_diag.get('Q_dehum_total_to_room_W', 0.0)), 4
        ),
        'biomass_transplant_kg_m2': round(biomass_trans_kg_m2, 6),
        'biomass_seedling_kg_m2': round(biomass_seed_kg_m2, 6),
        'biomass_finishing_kg_m2': round(biomass_trans_kg_m2, 6),
        'biomass_dense_kg_m2': round(biomass_seed_kg_m2, 6),
        'biomass_total_kg_m2': round(biomass_tot_kg_m2, 6),
        'harvest_event': 1 if harvest_dry_mass > 0 else 0,
        'harvest_mass_g': round(harvest_dry_mass, 4),
        'harvest_dry_mass_g': round(harvest_dry_mass, 4),
        'harvest_fresh_mass_equiv_g': round(harvest_fresh_mass, 4),
        'harvest_mean_dry_mass_per_plant_g': round(harvest_mean_dry, 4),
        'harvest_mean_fresh_mass_per_plant_g': round(harvest_mean_fresh, 4),
        'LAI_transplant': round(lai_tr, 4),
        'LAI_seedling': round(lai_se, 4),
        'LAI_finishing': round(lai_tr, 4),
        'LAI_dense': round(lai_se, 4),
        'hours_light': round(env.hours_continuous_light, 4),
        'hours_dark': round(env.hours_continuous_dark, 4),
        'DLI': round(env.daily_DLI, 4),
        'DLI_weighted': round(env.daily_DLI, 4),
        'DLI_dense': round(float(getattr(env, 'daily_DLI_dense', 0.0)), 4),
        'DLI_finishing': round(float(getattr(env, 'daily_DLI_finishing', 0.0)), 4),
        'rho1': round(schedule.get('rho1', 0.0), 4),
        'rho2': round(schedule.get('rho2', 0.0), 4),
        'rho_dense': round(schedule.get('rho1', 0.0), 4),
        'rho_finishing': round(schedule.get('rho2', 0.0), 4),
        'A1': round(schedule.get('A1', 0.0), 4),
        'A2': round(schedule.get('A2', 0.0), 4),
        'A_dense': round(schedule.get('A1', 0.0), 4),
        'A_finishing': round(schedule.get('A2', 0.0), 4),
        'total_transplants': cum_transplants,
        'total_harvests': cum_harvests,
        'reward_step': round(reward, 6),
        'cum_reward': round(cum_reward, 4),
        'cum_cost': round(cum_cost, 4),
        'step_size_s': dt_s,
    }


# CSV记录字段列表
FIELDS = [
    'step', 'elapsed_h', 'elapsed_d', 'datetime',
    'T_in', 'RH_pct', 'VPD_kPa', 'C_ppm',
    'T_out', 'RH_out_pct', 'C_out_ppm',
    'I1', 'I2', 'Q_HVAC', 'u_CO2', 'm_dehum',
    'I_dense', 'I_finishing',
    'P_LED1_W_m2', 'P_LED2_W_m2',
    'P_LED_dense_W_m2', 'P_LED_finishing_W_m2',
    'P_LED_total_kW', 'P_HVAC_kW',
    'P_heating_kW', 'P_cooling_kW',
    'P_CO2_kW', 'P_dehum_kW', 'P_total_kW', 'elec_price_rmb_kwh',
    'E_step_kWh', 'cost_CO2_rmb', 'cost_elec_rmb', 'cost_total_rmb',
    'E_transp_kg', 'E_resp_kg',
    'dehum_removed_kg', 'env_condensation_removed_kg',
    'Q_led_driver_loss_W', 'Q_led_room_upper_W', 'Q_biomass_storage_W',
    'Q_led_room_W', 'Q_dehum_latent_to_room_W',
    'Q_dehum_electric_to_room_W', 'Q_dehum_total_to_room_W',
    'biomass_transplant_kg_m2', 'biomass_seedling_kg_m2',
    'biomass_finishing_kg_m2', 'biomass_dense_kg_m2',
    'biomass_total_kg_m2', 'harvest_event',
    'harvest_mass_g', 'harvest_dry_mass_g', 'harvest_fresh_mass_equiv_g',
    'harvest_mean_dry_mass_per_plant_g', 'harvest_mean_fresh_mass_per_plant_g',
    'LAI_transplant', 'LAI_seedling',
    'LAI_finishing', 'LAI_dense',
    'hours_light', 'hours_dark', 'DLI', 'DLI_weighted', 'DLI_dense', 'DLI_finishing',
    'rho1', 'rho2', 'A1', 'A2',
    'rho_dense', 'rho_finishing', 'A_dense', 'A_finishing',
    'total_transplants', 'total_harvests',
    'reward_step', 'cum_reward', 'cum_cost',
    'step_size_s',
]

# 批次生长轨迹CSV字段列表
BATCH_TRAJECTORY_FIELDS = [
    'datetime', 'elapsed_d', 'step',
    'region', 'region_semantic', 'pipeline_slot', 'batch_id',
    'age_h', 'xD_total_g_m2', 'LAI'
]


# ═══════════════════════════════════════════════════════════════════════════════
#  Dashboard plotting
# ═══════════════════════════════════════════════════════════════════════════════

def plot_dashboard(csv_path: str, out_png: str,
                   title: str,
                   photoperiod_off: int = 8,
                   x_axis_days: bool = False,
                   use_actual_light_night: bool = True):
    """
    4行×3列综合仪表盘，所有12个轨迹面板合并在一张图中便于对比

    行0: 温度 | 湿度 | CO₂
    行1: 光强(I1/I2) | HVAC功率 | CO₂注入
    行2: 生物量 | DLI | 除湿
    行3: 步能耗 | 累计成本 | 累计奖励

    Parameters
    ----------
    use_actual_light_night : bool
        True: 夜间遮罩基于实际 I1/I2 > 0 判断（数据驱动，最准确）
        False: 基于固定 photoperiod_off 小时数估算
    """
    df = pd.read_csv(csv_path, parse_dates=['datetime'])
    n = len(df)
    if n == 0:
        print(f"[WARN] Empty CSV: {csv_path}")
        return

    df = _ensure_hvac_split_power_columns(df)
    x = df['elapsed_d'].values if x_axis_days else df['elapsed_h'].values
    xlabel = 'Day of Simulation [d]' if x_axis_days else 'Elapsed Time [h]'
    targets = _compute_dashboard_target_arrays(df['elapsed_h'].values, photoperiod_off)

    if use_actual_light_night:
        night_mask = (df['I1'].values <= 0) & (df['I2'].values <= 0)
    else:
        night_mask = _compute_night_mask(df, photoperiod_off)

    fig = plt.figure(figsize=(24, 18))
    fig.suptitle(title, fontsize=17, fontweight='bold', y=0.98)
    gs = GridSpec(4, 3, figure=fig, hspace=0.42, wspace=0.32)

    ax_T    = fig.add_subplot(gs[0, 0])
    ax_RH   = fig.add_subplot(gs[0, 1])
    ax_CO2  = fig.add_subplot(gs[0, 2])
    ax_I    = fig.add_subplot(gs[1, 0])
    ax_Q    = fig.add_subplot(gs[1, 1])
    ax_CO2u = fig.add_subplot(gs[1, 2])
    ax_BM   = fig.add_subplot(gs[2, 0])
    ax_DLI  = fig.add_subplot(gs[2, 1])
    ax_MD   = fig.add_subplot(gs[2, 2])
    ax_E    = fig.add_subplot(gs[3, 0])
    ax_Cost = fig.add_subplot(gs[3, 1])
    ax_Rew  = fig.add_subplot(gs[3, 2])

    # ── 行0 ─────────────────────────────────────────────────────────────────
    ax_T.plot(x, df['T_in'].values, color='#1E88E5', lw=1.5, label='Indoor T')
    ax_T.plot(x, df['T_out'].values, color='#FF9800', lw=1.0, ls='--', label='Outdoor T')
    ax_T.fill_between(
        x, targets['temp_lo'], targets['temp_hi'],
        color='#66BB6A', alpha=0.16, label='Temp target band'
    )
    ax_T.plot(x, targets['temp_sp'], color='#2E7D32', lw=1.0, ls='-.', label='Temp target')
    _shade_night(ax_T, x, night_mask)
    ax_T.set_xlabel(xlabel, fontsize=9)
    ax_T.set_ylabel('Temperature [°C]', fontsize=9)
    ax_T.legend(fontsize=8, loc='upper right')
    ax_T.set_title('Indoor vs Outdoor Temperature + Day/Night Target Band', fontsize=10)
    ax_T.grid(True, alpha=0.3)

    ax_RH.plot(x, df['RH_pct'].values, color='#1E88E5', lw=1.5, label='Indoor RH')
    ax_RH.plot(x, df['RH_out_pct'].values, color='#FF9800', lw=1.0, ls='--', label='Outdoor RH')
    _shade_night(ax_RH, x, night_mask)
    ax_RH.set_xlabel(xlabel, fontsize=9)
    ax_RH.set_ylabel('Relative Humidity [%]', fontsize=9)
    ax_RH.legend(fontsize=8)
    ax_RH.set_title('Indoor vs Outdoor Humidity (VPD-targeted control)', fontsize=10)
    ax_RH.grid(True, alpha=0.3)

    ax_CO2.plot(x, df['C_ppm'].values, color='#2196F3', lw=1.5, label='Indoor CO₂')
    ax_CO2.plot(x, df['C_out_ppm'].values, color='#FF9800', lw=1.0, ls='--', label='Outdoor CO₂')
    ax_CO2.fill_between(
        x, targets['co2_lo'], targets['co2_hi'],
        color='#81C784', alpha=0.16, label='CO₂ target band'
    )
    ax_CO2.plot(x, targets['co2_sp'], color='#2E7D32', lw=1.0, ls='-.', label='CO₂ target')
    _shade_night(ax_CO2, x, night_mask)
    ax_CO2.set_xlabel(xlabel, fontsize=9)
    ax_CO2.set_ylabel('CO₂ [ppm]', fontsize=9)
    ax_CO2.legend(fontsize=8)
    ax_CO2.set_title('CO₂ Concentration + Day/Night Target Band', fontsize=10)
    ax_CO2.grid(True, alpha=0.3)

    # ── 行1 ─────────────────────────────────────────────────────────────────
    # I1 → 密植区 (dense)，I2 → 定植区 (finishing)；内部仍沿用 seedling/transplant 映射
    ax_I.plot(x, df['I1'].values, color='#FF5722', lw=1.5, label='I₁ (Dense)', alpha=0.9)
    ax_I.plot(x, df['I2'].values, color='#4CAF50', lw=1.5, label='I₂ (Finishing)', alpha=0.9)
    _shade_night(ax_I, x, night_mask)
    ax_I.set_xlabel(xlabel, fontsize=9)
    ax_I.set_ylabel('PPFD [μmol/m²/s]', fontsize=9)
    ax_I.legend(fontsize=8)
    ax_I.set_title('LED Light Intensity', fontsize=10)
    ax_I.grid(True, alpha=0.3)

    q_hvac = df['Q_HVAC'].astype(float).values
    q_heating = np.where(q_hvac > 0.0, q_hvac, np.nan)
    q_cooling = np.where(q_hvac < 0.0, q_hvac, np.nan)
    ax_Q.plot(x, q_heating, color='#E53935', lw=1.2, label='Heating')
    ax_Q.plot(x, q_cooling, color='#1E88E5', lw=1.2, label='Cooling')
    ax_Q.fill_between(x, 0.0, q_hvac, where=q_hvac > 0.0,
                      color='#E53935', alpha=0.12, interpolate=True)
    ax_Q.fill_between(x, 0.0, q_hvac, where=q_hvac < 0.0,
                      color='#1E88E5', alpha=0.12, interpolate=True)
    _shade_night(ax_Q, x, night_mask)
    ax_Q.axhline(0, color='k', lw=0.8)
    ax_Q.set_xlabel(xlabel, fontsize=9)
    ax_Q.set_ylabel('Q_HVAC [W/m²]', fontsize=9)
    ax_Q.set_title('HVAC Power Density', fontsize=10)
    ax_Q.legend(fontsize=8)
    ax_Q.grid(True, alpha=0.3)

    # CO2 injection rate: u_CO2 [kg/m2/s]
    co2_rate = df['u_CO2'].values
    ax_CO2u.plot(x, co2_rate, color='#009688', lw=1.2, label='u_CO2')
    _shade_night(ax_CO2u, x, night_mask)
    ax_CO2u.set_xlabel(xlabel, fontsize=9)
    ax_CO2u.set_ylabel('CO2 injection [kg/m2/s]', fontsize=9)
    ax_CO2u.set_title('CO2 Injection Rate', fontsize=10)
    ax_CO2u.legend(fontsize=8)
    ax_CO2u.grid(True, alpha=0.3)

    # ── 行2 ─────────────────────────────────────────────────────────────────
    ax_BM.plot(x, df['biomass_transplant_kg_m2'].values,
               color='#3F51B5', lw=2.0, label='Finishing (Dₛ)')
    ax_BM.plot(x, df['biomass_seedling_kg_m2'].values,
               color='#009688', lw=1.5, ls='--', label='Dense (Dₛ)')
    _shade_night(ax_BM, x, night_mask)
    ax_BM.set_xlabel(xlabel, fontsize=9)
    ax_BM.set_ylabel('Dry Mass [kg/m²]', fontsize=9)
    ax_BM.legend(fontsize=8)
    ax_BM.set_title('Biomass Accumulation', fontsize=10)
    ax_BM.grid(True, alpha=0.3)

    if 'VPD_kPa' in df.columns:
        vpd = df['VPD_kPa'].values
    else:
        from src.envs.utils import load_all_configs
        cfg = _WEATHER_CONTAINER_CFG or load_all_configs(str(ROOT / 'configs')).get('container_params', {})
        vpd = np.array([
            relative_humidity_to_vpd(t, rh / 100.0, cfg)
            for t, rh in zip(df['T_in'].values, df['RH_pct'].values)
        ])
    ax_DLI.plot(x, vpd, color='#6A1B9A', lw=1.5, label='Indoor VPD')
    ax_DLI.fill_between(
        x, targets['vpd_lo'], targets['vpd_hi'],
        color='#CE93D8', alpha=0.18, label='VPD target band'
    )
    ax_DLI.plot(x, targets['vpd_sp'], color='#8E24AA', lw=1.0, ls='-.', label='VPD target')
    _shade_night(ax_DLI, x, night_mask)
    ax_DLI.set_xlabel(xlabel, fontsize=9)
    ax_DLI.set_ylabel('VPD [kPa]', fontsize=9)
    ax_DLI.set_title('VPD Trajectory + Day/Night Target Band', fontsize=10)
    ax_DLI.legend(fontsize=8)
    ax_DLI.grid(True, alpha=0.3)

    # Dehumidification rate: m_dehum [kg/m2/s]
    dehum_rate = df['m_dehum'].values
    ax_MD.plot(x, dehum_rate, color='#795548', lw=1.2)
    _shade_night(ax_MD, x, night_mask)
    ax_MD.set_xlabel(xlabel, fontsize=9)
    ax_MD.set_ylabel('m_dehum [kg/m2/s]', fontsize=9)
    ax_MD.set_title('Dehumidification Rate', fontsize=10)
    ax_MD.grid(True, alpha=0.3)

    # ── 行3 ─────────────────────────────────────────────────────────────────
    w = (x[1] - x[0]) * 0.8 if len(x) > 1 else 0.8
    ax_E.bar(x, df['E_step_kWh'].values,
             width=w, color='#FFC107', alpha=0.7, align='edge')
    _shade_night(ax_E, x, night_mask)
    ax_E.set_xlabel(xlabel, fontsize=9)
    ax_E.set_ylabel('Energy [kWh/step]', fontsize=9)
    ax_E.set_title('Step Energy Consumption', fontsize=10)
    ax_E.grid(True, alpha=0.3)
    ax_E.set_ylim(bottom=0)

    ax_Cost.plot(x, df['cum_cost'].values, color='#E91E63', lw=2.0)
    _shade_night(ax_Cost, x, night_mask)
    ax_Cost.set_xlabel(xlabel, fontsize=9)
    ax_Cost.set_ylabel('Cumulative Cost [RMB]', fontsize=9)
    ax_Cost.set_title('Cumulative Operating Cost', fontsize=10)
    ax_Cost.grid(True, alpha=0.3)

    ax_Rew.plot(x, df['cum_reward'].values, color='#4CAF50', lw=2.0)
    _shade_night(ax_Rew, x, night_mask)
    ax_Rew.set_xlabel(xlabel, fontsize=9)
    ax_Rew.set_ylabel('Cumulative Reward [RMB]', fontsize=9)
    ax_Rew.set_title('Cumulative Reward', fontsize=10)
    ax_Rew.grid(True, alpha=0.3)

    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.05, top=0.94, hspace=0.40, wspace=0.28)
    fig.savefig(out_png, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[SAVE] Dashboard -> {out_png}")


def _compute_night_mask(df: pd.DataFrame, photoperiod_off: int = 8) -> np.ndarray:
    """
    计算夜间遮罩（光周期外的时段视为夜间）。

    Deprecated: 该方法假设固定的光期长度，无法反映 RL 策略动态开关灯。
    推荐在 plot_dashboard 中使用 use_actual_light_night=True（基于 I1/I2 > 0）。

    Parameters
    ----------
    photoperiod_off : int
        固定为夜间的起始小时（默认 8，即 hour 0-8 为夜）。
    """
    hours = df['elapsed_h'].values if 'elapsed_h' in df.columns \
             else np.arange(len(df))
    hour_of_day = hours % 24
    return (hour_of_day < photoperiod_off).astype(bool)


def _shade_night(ax, x, night_mask):
    """在图表上绘制夜间遮罩区域"""
    nx = x[night_mask]
    if len(nx) == 0:
        return
    for i in range(len(nx) - 1):
        ax.axvspan(nx[i], nx[i + 1], alpha=0.12, color='gray', zorder=0)


def _load_dashboard_controller_targets():
    """Load controller/target parameters once for plotting."""
    global _PLOT_CTRL_CFG
    if _PLOT_CTRL_CFG is None:
        if CTRL_CFG.exists():
            with open(CTRL_CFG, 'r', encoding='utf-8') as f:
                _PLOT_CTRL_CFG = yaml.safe_load(f) or {}
        else:
            _PLOT_CTRL_CFG = {}
    return _PLOT_CTRL_CFG.get('pid_controller', {})


def _compute_dashboard_target_arrays(hours: np.ndarray, photoperiod_off: int) -> dict:
    """Compute day/night target bands with the same ramp logic as the controller."""
    pid_cfg = _load_dashboard_controller_targets()
    hour_of_day = np.mod(hours, 24.0)
    pp_off = float(photoperiod_off)
    pp_on = max(0.0, 24.0 - pp_off)
    ramp_h = max(float(pid_cfg.get('transition_ramp_hours', 1.0)), 1e-6)
    dawn = pp_off
    dusk = dawn + pp_on

    blend = np.zeros_like(hour_of_day, dtype=np.float64)
    rise = (hour_of_day >= max(0.0, dawn - ramp_h)) & (hour_of_day < dawn)
    if np.any(rise):
        blend[rise] = np.clip((hour_of_day[rise] - (dawn - ramp_h)) / ramp_h, 0.0, 1.0)
    day = (hour_of_day >= dawn) & (hour_of_day < max(dawn, dusk - ramp_h))
    blend[day] = 1.0
    fall = (hour_of_day >= max(dawn, dusk - ramp_h)) & (hour_of_day < min(24.0, dusk))
    if np.any(fall):
        blend[fall] = np.clip((dusk - hour_of_day[fall]) / ramp_h, 0.0, 1.0)

    temp_day_lo = float(pid_cfg.get('temp_band_day_lo', 22.0))
    temp_day_hi = float(pid_cfg.get('temp_band_day_hi', 25.0))
    temp_night_lo = float(pid_cfg.get('temp_band_night_lo', 18.0))
    temp_night_hi = float(pid_cfg.get('temp_band_night_hi', 20.0))
    temp_day_sp = float(pid_cfg.get('temp_setpoint_day', 23.5))
    temp_night_sp = float(pid_cfg.get('temp_setpoint_night', 19.0))

    co2_day_lo = float(pid_cfg.get('co2_band_day_lo_ppm', 800.0))
    co2_day_hi = float(pid_cfg.get('co2_band_day_hi_ppm', 1200.0))
    co2_night_lo = float(pid_cfg.get('co2_band_night_lo_ppm', 400.0))
    co2_night_hi = float(pid_cfg.get('co2_band_night_hi_ppm', 450.0))
    co2_day_sp = float(pid_cfg.get('co2_setpoint_day_ppm', pid_cfg.get('co2_setpoint', 950.0)))
    co2_night_sp = float(pid_cfg.get('co2_setpoint_night_ppm', 400.0))

    vpd_day_lo = float(pid_cfg.get('vpd_day_lo_kpa', 0.55))
    vpd_day_hi = float(pid_cfg.get('vpd_day_hi_kpa', 0.95))
    vpd_night_lo = float(pid_cfg.get('vpd_night_lo_kpa', 0.25))
    vpd_night_hi = float(pid_cfg.get('vpd_night_hi_kpa', 0.60))

    return {
        'blend': blend,
        'temp_lo': temp_night_lo + blend * (temp_day_lo - temp_night_lo),
        'temp_hi': temp_night_hi + blend * (temp_day_hi - temp_night_hi),
        'temp_sp': temp_night_sp + blend * (temp_day_sp - temp_night_sp),
        'co2_lo': co2_night_lo + blend * (co2_day_lo - co2_night_lo),
        'co2_hi': co2_night_hi + blend * (co2_day_hi - co2_night_hi),
        'co2_sp': co2_night_sp + blend * (co2_day_sp - co2_night_sp),
        'vpd_lo': vpd_night_lo + blend * (vpd_day_lo - vpd_night_lo),
        'vpd_hi': vpd_night_hi + blend * (vpd_day_hi - vpd_night_hi),
        'vpd_sp': 0.5 * (
            (vpd_night_lo + vpd_night_hi)
            + blend * ((vpd_day_lo + vpd_day_hi) - (vpd_night_lo + vpd_night_hi))
        ),
    }


def _plants_per_harvest_event(schedule: dict) -> float:
    """Number of plants harvested per transplant event under the pipeline schedule."""
    t1 = int(schedule.get('t1', 14))
    t2 = int(schedule.get('t2', 21))
    rho2 = float(schedule.get('rho2', 25.0))
    A2 = float(schedule.get('A2', 72.0))
    delta_days = max(1, math.gcd(t1, t2))
    k2 = max(1, t2 // delta_days)
    area_per_event = A2 / k2
    return max(area_per_event * rho2, 1e-12)


# ═══════════════════════════════════════════════════════════════════════════════
#  Monthly energy stacking plot
# ═══════════════════════════════════════════════════════════════════════════════

def plot_energy_stacking_monthly(csv_path: str, out_png: str, title: str):
    """
    月度能耗堆叠图

    行1: kWh/m²堆叠条形图（区分LED/制冷/加热/CO₂/除湿）
    行2: kWh/kg采收干重堆叠条形图（无采收的月份为NaN）
    """
    df = pd.read_csv(csv_path, parse_dates=['datetime'])
    if 'elapsed_d' not in df.columns or len(df) == 0:
        return

    df = _ensure_hvac_split_power_columns(df)
    dt_s = float(df['step_size_s'].iloc[0])
    dt_h = dt_s / 3600.0
    A_total = float((df['A1'].iloc[0] + df['A2'].iloc[0]) if ('A1' in df.columns and 'A2' in df.columns) else 80.0)

    # 计算各分项能耗 [kWh/step]
    # LED electrical energy is already stored as total kW.
    df['E_LED_kWh'] = df['P_LED_total_kW'] * dt_h
    # HVAC制冷能耗
    df['E_cooling_kWh'] = df['P_cooling_kW'] * dt_h
    # HVAC加热能耗
    df['E_heating_kWh'] = df['P_heating_kW'] * dt_h
    # 除湿能耗
    df['E_dehum_kWh'] = _dehum_power_kW_from_frame(df) * dt_h

    df['month'] = df['datetime'].dt.month

    # 按月汇总能耗
    monthly_energy = df.groupby('month')[
        ['E_LED_kWh', 'E_cooling_kWh', 'E_heating_kWh', 'E_dehum_kWh']
    ].sum()

    # 按月汇总采收量（用于计算 kWh/kg DW）
    df = _ensure_harvest_metrics_frame(df)
    df['harvest_kg'] = df['harvest_dry_mass_g'] / 1000.0
    monthly_harvest = df.groupby('month')['harvest_kg'].sum()

    # 创建图表：2行1列
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    months = np.arange(1, 13)

    # ── 行1: kWh/m² 堆叠条形图 ───────────────────────────────────────────────
    # 扩展monthly_energy到12个月（缺失月份补0）
    for m in months:
        if m not in monthly_energy.index:
            monthly_energy.loc[m] = 0.0
    monthly_energy = monthly_energy.sort_index()

    # 计算每月总能耗和单位面积能耗
    E_total_monthly = (
        monthly_energy['E_LED_kWh'] +
        monthly_energy['E_cooling_kWh'] +
        monthly_energy['E_heating_kWh'] +
        monthly_energy['E_dehum_kWh']
    )
    E_per_m2 = E_total_monthly / A_total

    stack_spec = [
        ('E_LED_kWh', 'LED Lighting', '#FF9800'),
        ('E_heating_kWh', 'HVAC Heating', '#E53935'),
        ('E_cooling_kWh', 'HVAC Cooling', '#1E88E5'),
        ('E_dehum_kWh', 'Dehumidification', '#795548'),
    ]
    monthly_energy = monthly_energy.reindex(months, fill_value=0.0)
    monthly_harvest = monthly_harvest.reindex(months, fill_value=0.0)

    bottom = np.zeros(12)
    for col, label, color in stack_spec:
        vals = monthly_energy[col].values
        E_per_layer = vals / A_total
        ax1.bar(months, E_per_layer, bottom=bottom, color=color,
                label=label, alpha=0.85)
        bottom += E_per_layer

    ax1.set_xlabel('Month', fontsize=12)
    ax1.set_ylabel('Electric Energy [kWh/m²]', fontsize=12)
    ax1.set_title('Monthly Electric Energy per Unit Area (kWh/m²)', fontsize=12)
    ax1.set_xticks(months)
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(True, alpha=0.3, axis='y')

    # ── 行2: kWh/kg 采收干重堆叠条形图 ──────────────────────────────────────
    # 计算每月单位采收干重能耗
    E_per_kg = np.full(12, np.nan)
    for m in months:
        if m in monthly_harvest.index and monthly_harvest[m] > 0:
            E_per_kg[m - 1] = E_total_monthly[m] / monthly_harvest[m]
        else:
            E_per_kg[m - 1] = np.nan

    bottom2 = np.zeros(12)
    nan_mask = np.isnan(E_per_kg)
    harvest_vals = monthly_harvest.values

    for col, label, color in stack_spec:
        vals = monthly_energy[col].values
        E_layer_per_kg = np.divide(
            vals,
            harvest_vals,
            out=np.zeros_like(vals, dtype=float),
            where=harvest_vals > 0.0,
        )
        E_layer_per_kg = np.where(nan_mask, 0.0, E_layer_per_kg)
        ax2.bar(months, E_layer_per_kg, bottom=bottom2,
                color=color,
                label=label, alpha=0.85)
        bottom2 += E_layer_per_kg

    # 用×标记表示无采收的月份
    for m in months:
        if np.isnan(E_per_kg[m - 1]):
            ax2.scatter(m, 0, marker='x', s=100, color='gray', zorder=5)

    ax2.set_xlabel('Month', fontsize=12)
    ax2.set_ylabel('Electric Energy [kWh/kg harvest]', fontsize=12)
    ax2.set_title('Monthly Electric Energy per Harvest Dry Mass (kWh/kg)', fontsize=12)
    ax2.set_xticks(months)
    ax2.legend(fontsize=9, loc='upper left')
    ax2.grid(True, alpha=0.3, axis='y')

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[SAVE] Energy stacking -> {out_png}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Batch growth trajectory plot
# ═══════════════════════════════════════════════════════════════════════════════

def plot_growth_trajectory_batch(batch_csv_path: str, out_png: str, title: str):
    """
    各批次生长轨迹细化图

    展示密植区（内部兼容名 seedling）和定植区（内部兼容名 transplant）中每个子种植槽对应批次的
    独立生长曲线。

    Parameters
    ----------
    batch_csv_path : str
        批次轨迹CSV文件路径
    out_png : str
        输出图像路径
    title : str
        图表标题
    """
    try:
        df = pd.read_csv(batch_csv_path, parse_dates=['datetime'])
    except Exception as e:
        print(f"[WARN] Failed to read batch trajectory CSV: {batch_csv_path} | error: {e}")
        return

    if len(df) == 0 or 'xD_total_g_m2' not in df.columns:
        print(f"[WARN] Batch trajectory CSV is empty or missing required columns: {batch_csv_path}")
        return

    fig, ax = plt.subplots(figsize=(14, 8))

    # 为每个(region, pipeline_slot, batch_id)组合绘制一条曲线
    groups = df.groupby(['region', 'pipeline_slot', 'batch_id'])

    # 定义颜色映射
    seedling_colors = plt.cm.Blues(np.linspace(0.3, 0.9, 10))
    transplant_colors = plt.cm.Greens(np.linspace(0.3, 0.9, 10))

    color_idx = {'seedling': 0, 'transplant': 0}
    max_color_idx = 10

    for (region, pipeline_slot, batch_id), group_df in groups:
        # 按时间排序
        group_df = group_df.sort_values('elapsed_d')

        # 选择颜色
        if region == 'seedling':
            c = seedling_colors[color_idx['seedling'] % max_color_idx]
            color_idx['seedling'] += 1
            ls = '--'  # 虚线表示密植区
            label_prefix = f'D{pipeline_slot}'
        else:
            c = transplant_colors[color_idx['transplant'] % max_color_idx]
            color_idx['transplant'] += 1
            ls = '-'   # 实线表示定植区
            label_prefix = f'F{pipeline_slot}'

        region_label = 'dense' if region == 'seedling' else 'finishing'

        ax.plot(group_df['elapsed_d'].values,
                group_df['xD_total_g_m2'].values,
                color=c, linestyle=ls, linewidth=1.5,
                label=f'{label_prefix}-B{batch_id} ({region_label[:4]})',
                marker='o', markersize=2, alpha=0.8)

    ax.set_xlabel('Elapsed Time [days]', fontsize=12)
    ax.set_ylabel('Dry Mass Density [g/m²]', fontsize=12)
    ax.set_title(f'Batch Growth Trajectory — {title}', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 添加图例（放在图外右侧）
    ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.02, 0.5),
              ncol=1, title='Slot-Batch (Stage)')

    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[SAVE] Batch growth trajectory -> {out_png}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Monthly harvest summary plot
# ═══════════════════════════════════════════════════════════════════════════════

def plot_harvest_monthly(
    csv_path: str,
    out_png: str,
    title: str,
    schedule: dict,
    c_fw: float = 22.5,
    harvest_target_dry_g: float = 5.33,
):
    """
    月度采收统计图

    上图:
      左轴: 当月采收总干重 (kg)
      右轴: 当月总采收次数
    下图:
      每次采收平均单株干重/鲜重，并给出达标线

    Parameters
    ----------
    csv_path : str
        仿真结果CSV路径
    out_png : str
        输出图像路径
    title : str
        图表标题
    """
    df = pd.read_csv(csv_path, parse_dates=['datetime'])
    if 'elapsed_d' not in df.columns or len(df) == 0:
        return
    df = _ensure_harvest_metrics_frame(df, default_c_fw=c_fw)

    df['month'] = df['datetime'].dt.month
    df['harvest_dry_kg'] = df['harvest_dry_mass_g'] / 1000.0
    df['harvest_fresh_kg'] = df['harvest_fresh_mass_equiv_g'] / 1000.0

    # 按月汇总采收干重和采收次数
    monthly = df.groupby('month').agg({
        'harvest_dry_kg': 'sum',
        'harvest_fresh_kg': 'sum',
        'harvest_event': 'sum'
    }).reset_index()
    monthly.columns = ['month', 'harvest_dry_kg', 'harvest_fresh_kg', 'harvest_count']

    months = np.arange(1, 13)
    harvest_dry_kg = np.zeros(12)
    harvest_count = np.zeros(12)

    for _, row in monthly.iterrows():
        m = int(row['month'])
        harvest_dry_kg[m - 1] = row['harvest_dry_kg']
        harvest_count[m - 1] = row['harvest_count']

    avg_dry_g = np.zeros(12)
    avg_fresh_g = np.zeros(12)
    for i in range(12):
        month_mask = (df['month'] == (i + 1)) & (df['harvest_event'] > 0)
        if month_mask.any():
            avg_dry_g[i] = float(df.loc[month_mask, 'harvest_mean_dry_mass_per_plant_g'].mean())
            avg_fresh_g[i] = float(df.loc[month_mask, 'harvest_mean_fresh_mass_per_plant_g'].mean())
        else:
            avg_dry_g[i] = np.nan
            avg_fresh_g[i] = np.nan

    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # 左轴: 采收干重条形图
    bar_width = 0.35
    bars = ax1.bar(months - bar_width / 2, harvest_dry_kg,
                   width=bar_width, color='#1E88E5', alpha=0.8,
                   label='Harvest Dry Mass')
    ax1.set_xlabel('Month', fontsize=12)
    ax1.set_ylabel('Harvest Dry Mass [kg]', fontsize=12, color='#1E88E5')
    ax1.tick_params(axis='y', labelcolor='#1E88E5')
    ax1.set_xticks(months)

    # 在条形上方添加数值标签
    for bar, val in zip(bars, harvest_dry_kg):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f'{val:.1f}', ha='center', va='bottom', fontsize=9,
                     color='#1E88E5')

    # 右轴: 采收次数折线图
    ax2 = ax1.twinx()
    ax2.plot(months, harvest_count, color='#E53935',
             marker='o', linewidth=2.5, markersize=8,
             label='Harvest Count')
    ax2.set_ylabel('Harvest Count', fontsize=12, color='#E53935')
    ax2.tick_params(axis='y', labelcolor='#E53935')

    # 在数据点旁边添加数值标签
    for m, count in zip(months, harvest_count):
        if count > 0:
            ax2.annotate(f'{int(count)}', (m, count),
                        textcoords="offset points",
                        xytext=(0, 8), ha='center', fontsize=9,
                        color='#E53935')

    ax1.set_title(f'Monthly Harvest Summary — {title}', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)

    # 下图: 单株干重 / 鲜重
    ax3.plot(months, avg_dry_g, color='#3949AB', marker='o', lw=2.0,
             label='Mean Dry Mass per Harvest Event')
    ax3.axhline(harvest_target_dry_g, color='#1B5E20', ls='--', lw=1.4,
                label=f'Dry Target ({harvest_target_dry_g:.2f} g/plant)')
    ax3.set_ylabel('Dry Mass [g/plant]', fontsize=12, color='#3949AB')
    ax3.tick_params(axis='y', labelcolor='#3949AB')
    ax3.grid(True, alpha=0.3, axis='y')

    ax4 = ax3.twinx()
    ax4.plot(months, avg_fresh_g, color='#FB8C00', marker='s', lw=2.0,
             label='Mean Fresh Mass per Harvest Event')
    ax4.axhline(harvest_target_dry_g * c_fw, color='#E65100', ls='--', lw=1.4,
                label=f'Fresh Target ({harvest_target_dry_g * c_fw:.0f} g/plant)')
    ax4.set_ylabel('Fresh Mass [g/plant]', fontsize=12, color='#FB8C00')
    ax4.tick_params(axis='y', labelcolor='#FB8C00')
    ax3.set_xticks(months)
    ax3.set_xlabel('Month', fontsize=12)
    ax3.set_title('Per-Harvest Mean Plant Mass and Thresholds', fontsize=12)

    for m, dry_val, fresh_val in zip(months, avg_dry_g, avg_fresh_g):
        if np.isfinite(dry_val):
            ax3.annotate(f'{dry_val:.2f}', (m, dry_val),
                         textcoords='offset points', xytext=(0, 8),
                         ha='center', fontsize=9, color='#3949AB')
        if np.isfinite(fresh_val):
            ax4.annotate(f'{fresh_val:.0f}', (m, fresh_val),
                         textcoords='offset points', xytext=(0, -14),
                         ha='center', fontsize=8, color='#FB8C00')

    lines3, labels3 = ax3.get_legend_handles_labels()
    lines4, labels4 = ax4.get_legend_handles_labels()
    ax3.legend(lines3 + lines4, labels3 + labels4, loc='upper left', fontsize=10)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[SAVE] Monthly harvest -> {out_png}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Transpiration and Respiration trajectory plot
# ═══════════════════════════════════════════════════════════════════════════════

def plot_transpiration_respiration(csv_path: str, out_png: str, title: str):
    """
    总蒸腾量和总呼吸量变化轨迹图

    绘制累计蒸腾量（kg水）和累计呼吸量（kg CO₂）随时间的变化曲线

    Parameters
    ----------
    csv_path : str
        仿真结果CSV路径
    out_png : str
        输出图像路径
    title : str
        图表标题
    """
    df = pd.read_csv(csv_path, parse_dates=['datetime'])
    if 'elapsed_d' not in df.columns or len(df) == 0:
        return

    # 检查是否有蒸腾和呼吸数据
    has_transp = 'E_transp_kg' in df.columns
    has_resp = 'E_resp_kg' in df.columns

    if not has_transp and not has_resp:
        print("[WARN] CSV is missing transpiration/respiration data; skipping plot.")
        return

    x = df['elapsed_d'].values
    photo_period_off = int(round(24.0 - float(df['photo_period_hours'].iloc[0]))) \
        if 'photo_period_hours' in df.columns else 8
    targets = _compute_dashboard_target_arrays(df['elapsed_h'].values, photoperiod_off=photo_period_off)
    night_mask = (df['I1'].values <= 0) & (df['I2'].values <= 0)

    fig, (ax1, ax_vpd) = plt.subplots(
        2, 1, figsize=(14, 10), sharex=True,
        gridspec_kw={'height_ratios': [2.0, 1.0]}
    )

    # 累计蒸腾量曲线（左轴）
    if has_transp:
        transp_vals = df['E_transp_kg'].values
        # 如果是单步值，需要累计
        if 'cum_transp_kg' not in df.columns:
            cum_transp = np.cumsum(transp_vals)
        else:
            cum_transp = df['cum_transp_kg'].values

        ax1.plot(x, cum_transp, color='#1565C0',
                 linewidth=2, label='Cumulative Transpiration')
        ax1.fill_between(x, 0, cum_transp, alpha=0.15, color='#1565C0')
        ax1.set_xlabel('Elapsed Time [days]', fontsize=12)
        ax1.set_ylabel('Cumulative Transpiration [kg water]', fontsize=12,
                       color='#1565C0')
        ax1.tick_params(axis='y', labelcolor='#1565C0')
        ax1.grid(True, alpha=0.3)

    # 累计呼吸量曲线（右轴）
    if has_resp:
        ax2 = ax1.twinx() if has_transp else None
        resp_vals = df['E_resp_kg'].values
        # 如果是单步值，需要累计
        if 'cum_resp_kg' not in df.columns:
            cum_resp = np.cumsum(resp_vals)
        else:
            cum_resp = df['cum_resp_kg'].values

        if ax2 is not None:
            ax2.plot(x, cum_resp, color='#C62828',
                     linewidth=2, linestyle='--', label='Cumulative Respiration')
            ax2.set_ylabel('Cumulative Respiration [kg CO₂]', fontsize=12,
                          color='#C62828')
            ax2.tick_params(axis='y', labelcolor='#C62828')
        else:
            ax1.plot(x, cum_resp, color='#C62828',
                     linewidth=2, linestyle='--', label='Cumulative Respiration')
            ax1.set_ylabel('Cumulative Respiration [kg CO₂]', fontsize=12,
                          color='#C62828')
            ax1.tick_params(axis='y', labelcolor='#C62828')

    # 合并图例
    if has_transp and has_resp:
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    elif has_transp:
        ax1.legend(loc='upper left', fontsize=10)
    elif has_resp:
        ax1.legend(loc='upper left', fontsize=10)

    if 'VPD_kPa' in df.columns:
        vpd = df['VPD_kPa'].values
    else:
        from src.envs.utils import load_all_configs
        cfg = _WEATHER_CONTAINER_CFG or load_all_configs(str(ROOT / 'configs')).get('container_params', {})
        vpd = np.array([
            relative_humidity_to_vpd(t, rh / 100.0, cfg)
            for t, rh in zip(df['T_in'].values, df['RH_pct'].values)
        ])

    ax_vpd.plot(x, vpd, color='#6A1B9A', lw=1.5, label='Indoor VPD')
    ax_vpd.fill_between(
        x, targets['vpd_lo'], targets['vpd_hi'],
        color='#CE93D8', alpha=0.18, label='VPD target band'
    )
    ax_vpd.plot(x, targets['vpd_sp'], color='#8E24AA', lw=1.0, ls='-.', label='VPD target')
    _shade_night(ax_vpd, x, night_mask)
    ax_vpd.set_xlabel('Elapsed Time [days]', fontsize=12)
    ax_vpd.set_ylabel('VPD [kPa]', fontsize=12)
    ax_vpd.legend(loc='upper right', fontsize=10)
    ax_vpd.grid(True, alpha=0.3)

    _shade_night(ax1, x, night_mask)
    ax1.set_title(f'Transpiration, Respiration & VPD - {title}', fontsize=13, fontweight='bold')

    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[SAVE] Transpiration & Respiration -> {out_png}")


def plot_thesis_summary(
    csv_path: str,
    out_png: str,
    title: str,
    schedule: dict,
    photoperiod_off: int = 8,
):
    """Compact thesis/report figure with target bands and key KPIs."""
    df = pd.read_csv(csv_path, parse_dates=['datetime'])
    if 'elapsed_d' not in df.columns or len(df) == 0:
        return
    df = _ensure_harvest_metrics_frame(df)
    df = _ensure_hvac_split_power_columns(df)

    x = df['elapsed_d'].values
    night_mask = (df['I1'].values <= 0) & (df['I2'].values <= 0)
    targets = _compute_dashboard_target_arrays(df['elapsed_h'].values, photoperiod_off)
    dt_h = float(df['step_size_s'].iloc[0]) / 3600.0
    smooth_window = max(1, int(round(6.0 / dt_h)))

    def smooth(series_name: str) -> np.ndarray:
        return df[series_name].rolling(smooth_window, min_periods=1, center=True).mean().values

    vpd = df['VPD_kPa'].values if 'VPD_kPa' in df.columns else np.zeros(len(df))
    harvest_df = df[df['harvest_dry_mass_g'] > 0].copy()
    harvest_cum_kg = (df['harvest_dry_mass_g'].cumsum() / 1000.0).values
    mean_fw_per_plant = (
        float(harvest_df['harvest_mean_fresh_mass_per_plant_g'].mean())
        if len(harvest_df) > 0 else 0.0
    )
    df = df.copy()
    df['P_dehum_plot_kW'] = _dehum_power_kW_from_frame(df)
    total_energy = float((
        df['P_LED_total_kW'].astype(float)
        + df['P_HVAC_kW'].astype(float)
        + df['P_dehum_plot_kW'].astype(float)
    ).sum() * dt_h)
    total_cost = float(df['cum_cost'].iloc[-1])
    total_reward = float(df['cum_reward'].iloc[-1])
    total_harvest_dry = float(df['harvest_dry_mass_g'].sum() / 1000.0)
    total_harvest_fresh = float(df['harvest_fresh_mass_equiv_g'].sum() / 1000.0)

    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(f'{title}\nCompact Thesis Summary', fontsize=16, fontweight='bold', y=0.98)
    gs = GridSpec(3, 2, figure=fig, hspace=0.34, wspace=0.24)

    ax_T = fig.add_subplot(gs[0, 0])
    ax_C = fig.add_subplot(gs[0, 1])
    ax_V = fig.add_subplot(gs[1, 0])
    ax_B = fig.add_subplot(gs[1, 1])
    ax_P = fig.add_subplot(gs[2, 0])
    ax_E = fig.add_subplot(gs[2, 1])

    ax_T.fill_between(x, targets['temp_lo'], targets['temp_hi'],
                      color='#66BB6A', alpha=0.16, label='Target band')
    ax_T.plot(x, df['T_in'].values, color='#1E88E5', lw=1.2, label='Indoor T')
    ax_T.plot(x, smooth('T_in'), color='#0D47A1', lw=2.0, label='6h mean')
    ax_T.plot(x, targets['temp_sp'], color='#2E7D32', lw=1.0, ls='-.', label='Target')
    _shade_night(ax_T, x, night_mask)
    ax_T.set_ylabel('Temperature [°C]')
    ax_T.set_title('Temperature Control')
    ax_T.grid(True, alpha=0.3)
    ax_T.legend(fontsize=8)

    ax_C.fill_between(x, targets['co2_lo'], targets['co2_hi'],
                      color='#81C784', alpha=0.16, label='Target band')
    ax_C.plot(x, df['C_ppm'].values, color='#2196F3', lw=1.2, label='Indoor CO₂')
    ax_C.plot(x, smooth('C_ppm'), color='#0D47A1', lw=2.0, label='6h mean')
    ax_C.plot(x, targets['co2_sp'], color='#2E7D32', lw=1.0, ls='-.', label='Target')
    _shade_night(ax_C, x, night_mask)
    ax_C.set_ylabel('CO₂ [ppm]')
    ax_C.set_title('CO₂ Control')
    ax_C.grid(True, alpha=0.3)
    ax_C.legend(fontsize=8)

    ax_V.fill_between(x, targets['vpd_lo'], targets['vpd_hi'],
                      color='#CE93D8', alpha=0.18, label='Target band')
    ax_V.plot(x, vpd, color='#8E24AA', lw=1.2, label='Indoor VPD')
    ax_V.plot(x, pd.Series(vpd).rolling(smooth_window, min_periods=1, center=True).mean().values,
              color='#4A148C', lw=2.0, label='6h mean')
    ax_V.plot(x, targets['vpd_sp'], color='#6A1B9A', lw=1.0, ls='-.', label='Target')
    _shade_night(ax_V, x, night_mask)
    ax_V.set_xlabel('Elapsed Time [days]')
    ax_V.set_ylabel('VPD [kPa]')
    ax_V.set_title('VPD Regulation')
    ax_V.grid(True, alpha=0.3)
    ax_V.legend(fontsize=8)

    ax_B.plot(x, df['biomass_transplant_kg_m2'].values, color='#3F51B5', lw=2.0,
              label='Finishing biomass')
    ax_B.plot(x, df['biomass_seedling_kg_m2'].values, color='#009688', lw=1.5, ls='--',
              label='Dense biomass')
    if len(harvest_df) > 0:
        ax_B.scatter(harvest_df['elapsed_d'].values, harvest_df['biomass_transplant_kg_m2'].values,
                     s=np.clip(harvest_df['harvest_dry_mass_g'].values / 20.0, 20.0, 180.0),
                     color='#FB8C00', alpha=0.65, label='Harvest event')
    _shade_night(ax_B, x, night_mask)
    ax_B.set_xlabel('Elapsed Time [days]')
    ax_B.set_ylabel('Dry Mass [kg/m²]')
    ax_B.set_title('Biomass & Harvest Events')
    ax_B.grid(True, alpha=0.3)
    ax_B.legend(fontsize=8)

    ax_P.plot(x, smooth('P_LED_total_kW'), color='#FF9800', lw=2.0, label='LED')
    ax_P.plot(x, smooth('P_heating_kW'), color='#E53935', lw=1.8, label='HVAC Heating')
    ax_P.plot(x, smooth('P_cooling_kW'), color='#1E88E5', lw=1.8, label='HVAC Cooling')
    ax_P.plot(x, smooth('P_dehum_plot_kW'), color='#795548', lw=1.6, label='Dehumidification')
    _shade_night(ax_P, x, night_mask)
    ax_P.axhline(0.0, color='k', lw=0.8)
    ax_P.set_xlabel('Elapsed Time [days]')
    ax_P.set_ylabel('Power [kW]')
    ax_P.set_title('Electric Power Breakdown (6h mean)')
    ax_P.grid(True, alpha=0.3)
    ax_P.legend(fontsize=8)

    ax_E.plot(x, df['cum_cost'].values, color='#E91E63', lw=2.0, label='Cumulative cost')
    ax_E.plot(x, df['cum_reward'].values, color='#43A047', lw=2.0, label='Cumulative reward')
    ax_E.set_xlabel('Elapsed Time [days]')
    ax_E.set_ylabel('Economic Value [RMB]')
    ax_E.set_title('Economics & Production')
    ax_E.grid(True, alpha=0.3)
    ax_E2 = ax_E.twinx()
    ax_E2.plot(x, harvest_cum_kg, color='#FB8C00', lw=1.8, ls='--', label='Cumulative harvest')
    ax_E2.set_ylabel('Cumulative Harvest [kg DW]')

    lines1, labels1 = ax_E.get_legend_handles_labels()
    lines2, labels2 = ax_E2.get_legend_handles_labels()
    ax_E.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)

    summary_text = (
        f'Total energy: {total_energy:,.0f} kWh\n'
        f'Total cost: RMB {total_cost:,.0f}\n'
        f'Total harvest: {total_harvest_dry:,.1f} kg DW\n'
        f'Fresh equivalent: {total_harvest_fresh:,.1f} kg FW\n'
        f'Mean fresh mass: {mean_fw_per_plant:.1f} g/plant\n'
        f'Final reward: {total_reward:,.1f}'
    )
    fig.text(
        0.985, 0.975, summary_text,
        ha='right', va='top', fontsize=10,
        bbox=dict(facecolor='white', alpha=0.85, edgecolor='#BDBDBD', boxstyle='round,pad=0.4')
    )

    fig.subplots_adjust(left=0.06, right=0.95, bottom=0.06, top=0.90, hspace=0.38, wspace=0.28)
    fig.savefig(out_png, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[SAVE] Thesis summary -> {out_png}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Monthly energy breakdown (legacy)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_energy_breakdown(csv_path: str, out_png: str, title: str):
    """月度电能分项堆叠图（区分 HVAC 加热与制冷）"""
    df = pd.read_csv(csv_path, parse_dates=['datetime'])
    if 'elapsed_d' not in df.columns or len(df) == 0:
        return
    df = _ensure_hvac_split_power_columns(df)

    dt_s = float(df['step_size_s'].iloc[0])
    dt_h = dt_s / 3600.0

    df['E_LED_kWh'] = df['P_LED_total_kW'] * dt_h
    df['E_heating_kWh'] = df['P_heating_kW'] * dt_h
    df['E_cooling_kWh'] = df['P_cooling_kW'] * dt_h
    df['E_dehum_kWh'] = _dehum_power_kW_from_frame(df) * dt_h

    df['month'] = df['datetime'].dt.month
    monthly = df.groupby('month')[
        ['E_LED_kWh', 'E_heating_kWh', 'E_cooling_kWh', 'E_dehum_kWh']].sum()
    monthly = monthly.reindex(range(1, 13), fill_value=0.0)

    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = np.zeros(len(monthly))
    colors  = ['#FF9800', '#E53935', '#1E88E5', '#795548']
    labels  = ['LED Lighting', 'HVAC Heating', 'HVAC Cooling', 'Dehumidification']
    cols    = ['E_LED_kWh', 'E_heating_kWh', 'E_cooling_kWh', 'E_dehum_kWh']
    for col, color, label in zip(cols, colors, labels):
        ax.bar(monthly.index, monthly[col].values,
               bottom=bottom, color=color, label=label, alpha=0.85)
        bottom += monthly[col].values

    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Electric Energy [kWh]', fontsize=12)
    ax.set_title(f'{title} - Monthly Electric Energy Breakdown', fontsize=14, fontweight='bold')
    ax.set_xticks(range(1, 13))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[SAVE] Energy breakdown -> {out_png}")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_schedule_arg(s: str) -> dict:
    """解析排程参数字符串

    0420 推荐的上层排程字符串为：
    x = {t1, t2, N1, rho2} in Z^4

    仍兼容旧格式中附带的 PP 字段，但不再建议这样使用。
    """
    kv = {}
    for part in s.split(','):
        k, v = part.split('=')
        kv[k.strip()] = float(v.strip())
    return build_schedule(
        int(kv.get('t1', 14)),
        int(kv.get('t2', 14)),
        int(kv.get('N1', 20)),
        float(kv.get('rho2', 36.0)),
        int(kv['PP']) if 'PP' in kv else None,
    )


def main():
    parser = argparse.ArgumentParser(
        description='Hangzhou Weather-Driven PFAL Simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
控制器类型 (--controller):
  pid       PFALConventionalController (规则PID baseline)
  rl        加载已训练 SAC 策略 (需 --load)
  openloop  开环固定动作 (I1=I2=200, Q_HVAC=0, u_CO2=0)

排程参数 (--schedule, 默认 x={t1,t2,N1,rho2}={14,14,20,36}):
  --schedule "t1=14,t2=14,N1=20,rho2=36"

手动控制参数:
  --I1_manual         手动设置密植区光强 [μmol/m²/s]，覆盖PID输出
  --I2_manual         手动设置定植区光强 [μmol/m²/s]，覆盖PID输出
  --photo_period_manual 手动设置光周期 [小时]，覆盖默认排程

仿真时间:
  --dt         仿真步长 [秒], 默认 3600 (1小时)
  --duration   总仿真时长 [天], 默认 28
  --start-date 开始日期 (YYYY-MM-DD), 默认 2024-01-01

RL 评估:
  --load  已训练策略路径, e.g. log/PFAL-contextual-SAC/sac_contextual/exp_0401_133814
  --device cpu|cuda (默认 cpu)
  --explore 对策略输出添加随机采样 (默认: 使用均值)
''')
    parser.add_argument('--controller',  default='pid',
                        choices=['pid', 'rl', 'openloop'],
                        help='控制器类型 (默认: pid)')
    parser.add_argument('--schedule',   default=None,
                        help='排程参数 "t1=N,t2=N,N1=N,rho2=N"')
    parser.add_argument('--t1',        type=int, default=None,
                        help='密植期时长 [天] (默认: 14)')
    parser.add_argument('--t2',        type=int, default=None,
                        help='定植期时长 [天] (默认: 14)')
    parser.add_argument('--N1',        type=int, default=None,
                        help='密植区板数 [块] (默认: 20)')
    parser.add_argument('--rho2',      type=float, default=None,
                        help='定植区密度 [株/m2] (默认: 36)')
    parser.add_argument('--PP',        type=int, default=None,
                        help=argparse.SUPPRESS)
    # 手动控制参数
    parser.add_argument('--I1_manual',         type=float, default=None,
                        help='手动设置密植区光强 [μmol/m²/s] (覆盖PID)')
    parser.add_argument('--I2_manual',           type=float, default=None,
                        help='手动设置定植区光强 [μmol/m²/s] (覆盖PID)')
    parser.add_argument('--photo_period_manual', type=int, default=None,
                        help='手动设置光周期 [小时] (覆盖默认排程)')
    parser.add_argument('--light_control_mode', type=str, default=None,
                        choices=['step', 'daily_hold', 'segmented_hold'],
                        help='光强执行模式: step | daily_hold | segmented_hold')
    parser.add_argument('--light_segments_per_photoperiod', type=int, default=None,
                        help='当 segmented_hold 启用时，每个光周期划分的段数')
    parser.add_argument('--price_model_type', type=str, default=None,
                        choices=['constant', 'time_of_use'],
                        help='电价模型: constant | time_of_use')
    parser.add_argument('--tou_tariff_scenario', type=str, default=None,
                        choices=list_tou_tariff_scenarios(),
                        help='内置峰谷电价场景名称')
    parser.add_argument('--electricity_price', type=float, default=None,
                        help='覆盖固定电价 [RMB/kWh]')
    parser.add_argument('--co2_price', type=float, default=None,
                        help='覆盖 CO2 价格 [RMB/kg]')
    parser.add_argument('--lettuce_price_fw', type=float, default=None,
                        help='覆盖生菜鲜重售价 [RMB/kg]')
    parser.add_argument('--constant_price', type=float, default=None,
                        help='覆盖 TOU 模型的平段/回退固定电价 [RMB/kWh]')
    parser.add_argument('--dt',         type=float, default=3600.0,
                        help='仿真步长 [秒] (默认: 3600)')
    parser.add_argument('--duration',   type=float, default=28.0,
                        help='总仿真时长 [天] (默认: 28)')
    parser.add_argument('--start-date', default='2024-01-01',
                        help='仿真开始日期 YYYY-MM-DD (默认: 2024-01-01)')
    parser.add_argument('--seed',       type=int, default=42,
                        help='随机种子 (默认: 42)')
    parser.add_argument('--out-dir',    default=None,
                        help='结果输出目录 (默认: results/hangzhou_sim/)')
    parser.add_argument('--plots-only', action='store_true',
                        help='跳过仿真,仅从已有CSV重新绘制图表')
    parser.add_argument('--load',       default=None,
                        help='RL策略路径 (--controller rl 时必填)')
    parser.add_argument('--load_checkpoint', default='auto',
                        choices=['best', 'final', 'selected', 'auto'],
                        help='RL检查点类型: best | final | selected | auto (默认: auto)')
    parser.add_argument('--device',     default='cpu',
                        help='RL策略设备 cpu|cuda (默认: cpu)')
    parser.add_argument('--explore',    action='store_true',
                        help='RL推理时对策略输出添加随机采样 (默认: 使用均值)')


    parser.description = (
        "Run Hangzhou weather simulation for the contextual PFAL model.\n\n"
        "Default schedule:\n"
        "  x={t1,t2,N1,rho2}={14,14,20,36}\n\n"
        "Examples:\n"
        "  python experiments/simulate_hangzhou.py --controller pid --duration 28 --dt 600\n"
        "  python experiments/simulate_hangzhou.py --controller pid --schedule "
        "\"t1=14,t2=14,N1=20,rho2=36\" --duration 60 --dt 3600\n"
        "  python experiments/simulate_hangzhou.py --controller rl --load "
        "log/PFAL-contextual-SAC/sac_contextual/your_run --schedule "
        "\"t1=14,t2=14,N1=20,rho2=36\" --duration 28 --dt 600"
    )
    parser.epilog = None
    _help_overrides = {
        'controller': 'Controller type: pid, rl, or openloop. Default: pid.',
        'schedule': 'Optional schedule string like "t1=14,t2=14,N1=20,rho2=36".',
        't1': 'Dense-stage duration in days. Default: 14.',
        't2': 'Finishing-stage duration in days. Default: 14.',
        'N1': 'Dense-zone board count. Default: 20.',
        'rho2': 'Finishing-zone density [plants/m^2]. Default: 36.',
        'I1_manual': 'Manual dense-zone PPFD target [umol/m^2/s].',
        'I2_manual': 'Manual finishing-zone PPFD target [umol/m^2/s].',
        'photo_period_manual': 'Manual photoperiod override [h/day].',
        'light_control_mode': 'Light-intensity execution mode: step, daily_hold, or segmented_hold.',
        'light_segments_per_photoperiod': 'Segments per photoperiod when light_control_mode=segmented_hold.',
        'price_model_type': 'Electricity-price model: constant or time_of_use.',
        'tou_tariff_scenario': 'Built-in TOU tariff scenario name for extension experiments.',
        'electricity_price': 'Override fixed electricity price [RMB/kWh].',
        'co2_price': 'Override CO2 price [RMB/kg].',
        'lettuce_price_fw': 'Override lettuce fresh-weight price [RMB/kg].',
        'constant_price': 'Override fallback / flat electricity price [RMB/kWh].',
        'dt': 'Simulation step size [s]. Default: 3600.',
        'duration': 'Simulation duration [days]. Default: 28.',
        'start_date': 'Simulation start date in YYYY-MM-DD. Default: 2024-01-01.',
        'seed': 'Random seed. Default: 42.',
        'out_dir': 'Output directory. Default: results/hangzhou_sim/.',
        'plots_only': 'Skip simulation and render plots from existing CSV outputs.',
        'load': 'Path to a trained RL run directory. Required when --controller rl.',
        'load_checkpoint': 'Which RL checkpoint to load: best, final, selected, or auto.',
        'device': 'RL inference device: cpu or cuda. Default: cpu.',
        'explore': 'Enable stochastic RL action sampling instead of deterministic rollout.',
    }
    for _action in parser._actions:
        if _action.dest in _help_overrides:
            _action.help = _help_overrides[_action.dest]

    args = parser.parse_args()

    # ── 输出目录 ──────────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir) if args.out_dir else \
              ROOT / 'results' / 'hangzhou_sim'
    out_dir.mkdir(parents=True, exist_ok=True)
    run_tag = (f"{args.controller}_{args.start_date}_d{int(args.duration)}"
               f"_dt{int(args.dt)}")

    # 手动设置标识
    if args.I1_manual is not None or args.I2_manual is not None:
        run_tag += "_manual_light"
    if args.photo_period_manual is not None:
        run_tag += f"_pp{args.photo_period_manual}h"
    if args.light_control_mode is not None:
        run_tag += f"_{args.light_control_mode}"
        if args.light_control_mode == 'segmented_hold' and args.light_segments_per_photoperiod is not None:
            run_tag += f"_seg{int(args.light_segments_per_photoperiod)}"
    if args.price_model_type is not None:
        run_tag += f"_{args.price_model_type}"
    if args.tou_tariff_scenario is not None:
        run_tag += f"_{args.tou_tariff_scenario}"

    # ── 排程 ────────────────────────────────────────────────────────────────────
    # 默认上层排程: x={t1, t2, N1, rho2}={14, 14, 20, 36}
    t1_arg = args.t1 if args.t1 is not None else 14
    t2_arg = args.t2 if args.t2 is not None else 14
    N1_arg = args.N1 if args.N1 is not None else 20
    rho2_arg = args.rho2 if args.rho2 is not None else 36.0

    if args.schedule:
        schedule = _parse_schedule_arg(args.schedule)
    else:
        schedule = build_schedule(t1=t1_arg, t2=t2_arg,
                                  N1=N1_arg, rho2=rho2_arg)

    if args.PP is not None:
        schedule['PP'] = int(args.PP)
        if args.photo_period_manual is None:
            args.photo_period_manual = int(args.PP)
        print("[WARN] --PP is deprecated in code_0420; use --photo_period_manual instead.")

    # 如果手动设置了光周期，将其注入schedule
    if args.photo_period_manual is not None:
        schedule['PP'] = int(args.photo_period_manual)
        print(f"[INFO] Manual photoperiod override: {args.photo_period_manual} h")

    # ── 气象数据 ───────────────────────────────────────────────────────────────
    weather_path = ROOT / 'data' / 'weather' / 'weather_hangzhou_2024.csv'
    all_weather  = load_weather_csv(str(weather_path))
    start_dt = dt.datetime.fromisoformat(args.start_date)
    end_dt   = start_dt + dt.timedelta(days=int(args.duration))
    weather_slice = slice_weather(all_weather, start_dt, end_dt)

    if not weather_slice:
        print(f"[ERROR] No weather data for window: {args.start_date} + {args.duration} days")
        return

    weather_rows = expand_weather_for_dt(weather_slice, args.dt)
    n_steps = len(weather_rows)
    print(f"[INFO] dt={args.dt}s -> {n_steps} steps "
          f"({weather_slice[0]['dt'].date()} -> {weather_slice[-1]['dt'].date()})")

    csv_path = out_dir / f'{run_tag}.csv'
    batch_csv_path = out_dir / f'{run_tag}_batch_trajectory.csv'

    # ── RL策略加载（提前加载以便尽早发现错误） ─────────────────────────────────
    policy, action_space = None, None
    rl_run_cfg = None
    if args.controller == 'rl':
        if not args.load:
            print("[ERROR] --controller rl requires --load <policy_path>")
            return
        policy, action_space, rl_run_cfg = load_rl_policy(
            args.load,
            args.device,
            checkpoint=args.load_checkpoint,
        )

    # ── 仿真 ─────────────────────────────────────────────────────────────────────
    if args.plots_only:
        if not csv_path.exists():
            print(f"[ERROR] CSV not found: {csv_path}")
            return
    else:
        print("[INFO] Building PFALEnvContextual ...")
        action_semantics_override = (
            'absolute' if args.controller in {'pid', 'openloop'} else None
        )
        env_cfg = build_env_config(
            schedule,
            args.dt,
            seed=args.seed,
            photo_period_override=args.photo_period_manual,
            light_control_mode=args.light_control_mode,
            light_segments_per_photoperiod=args.light_segments_per_photoperiod,
            action_semantics_override=action_semantics_override,
            run_config_overrides=rl_run_cfg,
            price_model_type=args.price_model_type,
            tou_tariff_scenario=args.tou_tariff_scenario,
            electricity_price=args.electricity_price,
            co2_price=args.co2_price,
            lettuce_price_fw=args.lettuce_price_fw,
            constant_price=args.constant_price,
        )
        env = PFALEnvContextual(env_cfg)
        print(f"[INFO] Running {args.controller} simulation "
              f"({n_steps} steps, dt={args.dt}s) ...")

        # 打印手动设置信息
        if args.I1_manual is not None:
            print(f"[INFO] Manual I1 override: {args.I1_manual} umol/m^2/s")
        if args.I2_manual is not None:
            print(f"[INFO] Manual I2 override: {args.I2_manual} umol/m^2/s")
        print(
            "[INFO] Light control mode: "
            f"{getattr(env, 'light_control_mode', 'step')} "
            f"(segments={getattr(env, 'light_segments_per_photoperiod', 1)})"
        )
        print(
            "[INFO] Electricity pricing: "
            f"{getattr(env, 'electricity_price_model', 'constant')} "
            f"(base={getattr(env, 'constant_electricity_price', env.c_elec):.4f} RMB/kWh)"
        )

        if args.controller == 'pid':
            records, batch_records = _run_pid_simulation(
                env, weather_rows, schedule, args.dt, args.seed,
                I1_manual=args.I1_manual, I2_manual=args.I2_manual,
                photo_period_manual=args.photo_period_manual)
        elif args.controller == 'rl':
            records, batch_records = _run_rl_simulation(
                env, weather_rows, schedule, args.dt,
                args.seed, policy, explore=args.explore,
                photo_period_manual=args.photo_period_manual,
                I1_manual=args.I1_manual, I2_manual=args.I2_manual)
        else:   # openloop
            records, batch_records = _run_openloop_simulation(
                env, weather_rows, schedule, args.dt, args.seed)

        # 写入主CSV
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(records)
        print(f"[SAVE] CSV -> {csv_path}")

        # 写入批次轨迹CSV
        if batch_records:
            with open(batch_csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=BATCH_TRAJECTORY_FIELDS)
                writer.writeheader()
                writer.writerows(batch_records)
            print(f"[SAVE] Batch trajectory CSV -> {batch_csv_path}")

        # ── 仿真汇总 ─────────────────────────────────────────────────────────────
        df_res = _ensure_harvest_metrics_frame(pd.DataFrame(records))
        total_E   = df_res['E_step_kWh'].sum()
        total_cost= df_res['cum_cost'].iloc[-1] if len(df_res) else 0.0
        total_rwd = df_res['cum_reward'].iloc[-1] if len(df_res) else 0.0
        harvest_dry_g = df_res['harvest_dry_mass_g'].sum()
        harvest_fresh_g = df_res['harvest_fresh_mass_equiv_g'].sum()
        harvest_rows = df_res[df_res['harvest_event'] > 0]
        mean_fresh_g_per_plant = (
            float(harvest_rows['harvest_mean_fresh_mass_per_plant_g'].mean())
            if len(harvest_rows) > 0 else 0.0
        )
        total_transplants = int(df_res['total_transplants'].iloc[-1]) if 'total_transplants' in df_res.columns else 0
        total_harvests = int(df_res['total_harvests'].iloc[-1]) if 'total_harvests' in df_res.columns else 0

        print(f"\n{'='*62}")
        print(f"  Simulation Summary")
        print(f"{'='*62}")
        print(f"  Schedule      : t1={schedule['t1']}d  t2={schedule['t2']}d  "
              f"N1={schedule['N1']}  rho2={schedule['rho2']:.1f}")
        if args.photo_period_manual:
            print(f"  Photo Period  : {args.photo_period_manual}h (manual)")
        if args.I1_manual is not None or args.I2_manual is not None:
            print(f"  Manual Light  : I1={args.I1_manual}, I2={args.I2_manual}")
        print(
            f"  Light Control : {getattr(env, 'light_control_mode', 'step')}  "
            f"(segments={getattr(env, 'light_segments_per_photoperiod', 1)})"
        )
        print(f"  Duration      : {args.duration} days  "
              f"({n_steps} steps x {args.dt}s = {n_steps*args.dt/86400:.1f} sim-days)")
        print(f"  Controller    : {args.controller.upper()}")
        print(f"  Total Energy  : {total_E:,.1f} kWh")
        print(f"  Total Cost    : RMB {total_cost:,.2f}")
        print(f"  Cumulative Rwd: {total_rwd:,.2f}")
        print(f"  Harvest Dry Mass : {harvest_dry_g:,.1f} g")
        print(f"  Harvest Fresh Eq.: {harvest_fresh_g:,.1f} g")
        if mean_fresh_g_per_plant > 0.0:
            print(f"  Mean Fresh Mass : {mean_fresh_g_per_plant:,.1f} g/plant")
        print(f"  Total Transplants: {total_transplants}")
        print(f"  Total Harvests: {total_harvests}")
        print(f"{'='*62}\n")

    # ── 仪表盘 ─────────────────────────────────────────────────────────────────
    dash_path = out_dir / f'{run_tag}_dashboard.png'
    title = (f'Hangzhou PFAL Simulation - {args.controller.upper()}  |  '
             f't1={schedule["t1"]}d t2={schedule["t2"]}d '
             f'N1={schedule["N1"]} rho2={schedule["rho2"]:.0f}  '
             f'dt={int(args.dt)}s  {args.start_date}+{int(args.duration)}d')
    plot_dashboard(
        str(csv_path), str(dash_path), title,
        photoperiod_off=max(0, 24 - int(args.photo_period_manual or schedule.get('PP', 16))),
        x_axis_days=False,
    )

    # ── 月度能耗堆叠图 ─────────────────────────────────────────────────────────
    df_check = pd.read_csv(csv_path, parse_dates=['datetime'])
    if df_check['datetime'].dt.month.nunique() > 1:
        energy_stack_path = out_dir / f'{run_tag}_energy_stacking.png'
        plot_energy_stacking_monthly(str(csv_path), str(energy_stack_path), title)

        # 保留原有的能耗分项图（兼容性）
        energy_path = out_dir / f'{run_tag}_energy_monthly.png'
        plot_energy_breakdown(str(csv_path), str(energy_path), title)

    # ── 批次生长轨迹图 ─────────────────────────────────────────────────────────
    if batch_csv_path.exists():
        growth_path = out_dir / f'{run_tag}_growth_trajectory.png'
        plot_growth_trajectory_batch(str(batch_csv_path), str(growth_path), title)
    else:
        print(f"[WARN] Batch trajectory CSV not found; skipping growth plot: {batch_csv_path}")

    # ── 月度采收统计图 ─────────────────────────────────────────────────────────
    harvest_path = out_dir / f'{run_tag}_harvest_monthly.png'
    plot_harvest_monthly(str(csv_path), str(harvest_path), title, schedule)

    # ── 蒸腾量和呼吸量变化轨迹图 ───────────────────────────────────────────────
    transp_resp_path = out_dir / f'{run_tag}_transp_resp.png'
    plot_transpiration_respiration(str(csv_path), str(transp_resp_path), title)

    thesis_path = out_dir / f'{run_tag}_thesis_summary.png'
    plot_thesis_summary(
        str(csv_path), str(thesis_path), title, schedule,
        photoperiod_off=max(0, 24 - int(args.photo_period_manual or schedule.get('PP', 16))),
    )

    print(f"[DONE] Results -> {out_dir}")


if __name__ == '__main__':
    main()
