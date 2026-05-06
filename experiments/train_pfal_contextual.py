# -*- coding: utf-8 -*-
"""
PFALEnvContextual — 上下文SAC训练脚本

基于 PFAL-DRL/codes/drl_based_control.py，使用 Tianshou SAC 训练
双区多批次植物工厂上下文强化学习策略。

使用方法:
    # 标准训练（TensorBoard + WandB）
    python experiments/train_pfal_contextual.py --epoch 50

    python experiments/train_pfal_contextual.py --epoch 100 
    python experiments/train_pfal_contextual.py --epoch 300
    python experiments/train_pfal_contextual.py --epoch 500



    # 无WandB快速测试
    python experiments/train_pfal_contextual.py --epoch 50 --no_wandb

    # 自定义超参数
    python experiments/train_pfal_contextual.py --epoch 100 --nstep 1000 --batch_size 256

    # 评估已保存策略
    python experiments/train_pfal_contextual.py --eval_only --load my_experiment

    python experiments/train_pfal_contextual.py --eval_only --load log/PFAL-contextual-SAC/sac_contextual/exp_0401_133814

    # C:/Users/29341/Desktop/code_0331/plant_factory_optimization_PFAL-DRL/log/PFAL-contextual-SAC/sac_contextual/exp_0401_133814

    # 在多个不同排程上评估泛化能力
    python experiments/train_pfal_contextual.py --eval_only --load my_experiment --n_eval_schedules 20

    # 加载并继续训练
    python experiments/train_pfal_contextual.py --load my_experiment --epoch 50

来源:
  - PFAL-DRL/codes/drl_based_control.py  (SAC训练框架)
  - PFAL-DRL/codes/PFALEnv.py             (环境设计参考)
"""

import os
import sys
import argparse
import datetime
import json
import math
import pprint
import importlib.machinery
import functools
import multiprocessing as mp
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch

if "--no_wandb" in sys.argv or "--help" in sys.argv or "-h" in sys.argv:
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_SILENT", "true")
    os.environ.setdefault("WANDB_DISABLE_CODE", "true")
    os.environ.setdefault("WANDB_DISABLED", "true")

    def _install_wandb_stub() -> None:
        if "wandb" in sys.modules:
            return

        def _noop(*args, **kwargs):
            return None

        class _DummyRun:
            project = "disabled"
            name = "disabled"
            url = ""

            def log(self, *args, **kwargs):
                return None

            def finish(self, *args, **kwargs):
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

    _install_wandb_stub()

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter

from envs.plant_factory_env_new import PFALEnvContextual
from utils import list_tou_tariff_scenarios
from rl.drl_based_control import (
    apply_env_rl_overrides,
    build_context_reset_options,
    compute_constraint_aware_selection_score,
    collect_vector_env_constraint_stats,
    collect_vector_env_schedule_coverage,
    collect_vector_env_termination_stats,
    collect_episode_stats,
    create_env_from_run_params,
    create_policy,
    flatten_constraint_metrics,
    flatten_termination_metrics,
    flatten_schedule_coverage_metrics,
    load_schedule_bounds,
    load_policy,
    load_saved_run_config,
    parse_fixed_schedule_values,
    parse_narrow_bounds_values,
    prepare_reset_options_for_env,
    resolve_experiment_dir,
    save_json_report,
    select_evaluation_schedules,
    sync_inlet_seedling_metadata,
)


# =============================================================================
# 配置加载
# =============================================================================

def load_rl_params(config_path: str = None) -> dict:
    """
    从 YAML 加载 RL 超参数，与 rl_params.yaml 对齐。

    默认值完全匹配 PFAL-DRL 原始实现：
      - epoch=50, nstep=4032, batch_size=512
      - hidden=(128,128), gamma=0.99
      - train_num=8, test_num=20
    """
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "rl_params.yaml"

    defaults = {
        # 训练长度
        "epoch": 50,
        # 每epoch收集步数 = t2_max × 24，确保至少覆盖一个完整episode
        # t2_max ∈ [10,28]，默认值28 → nstep = 28×24 = 672
        # episode_length = t2 × 24h，dt=3600s → 每episode 672 步
        # Legacy value 3025 came from PFAL-DRL with dt=600s and ~21-day tasks; current default is 4032 for the updated schedule space.
        "nstep": 28 * 24 * 6,   # 4032 steps/epoch（匹配 dt=600s 的 t2_max episode）
        "batch_size": 512,   # 与rl_params.yaml PPO batch_size对齐
        "buffer_size": 1_000_000,
        # 网络结构
        "hidden_sizes": [128, 128],
        "gamma": 0.99,
        "actor_lr": 3e-4,
        "critic_lr": 3e-4,
        "alpha_lr": 3e-4,
        "auto_alpha": True,
        "auto_nstep": True,
        "nstep_factor": 1.0,
        # 并行环境
        "train_num": 8,
        "test_num": 20,
        # 探索
        "exploration_noise": True,
        "initial_random_episodes": 4,
        # Collect a slightly larger chunk before each trainer update loop.
        # This reduces Python-side trainer overhead without changing the
        # objective, reward design, or total env-step budget.
        "step_per_collect_env_steps": 8,
        "update_per_step": 1.0,
        "info_detail_level": "compact",
        "vector_env_backend": "auto",
        "test_interval_epochs": 10,
        # 随机种子
        "seed": 42,
        # contextual schedule curriculum
        "context_sampling_phase": "full",
        "train_context_sampling_strategy": "distributed_cycle",
        "context_curriculum": None,
        "context_fixed_schedule": None,
        "context_narrow_bounds": None,
        "episode_length_mode": "max_t2",
        "episode_days": None,
        "episode_length_mix": None,
        "train_episode_length_mode": None,
        "train_episode_days": None,
        "train_episode_length_mix": None,
        "test_episode_length_mode": None,
        "test_episode_days": None,
        "test_episode_length_mix": None,
        "eval_episode_length_mode": None,
        "eval_episode_days": None,
        "eval_episode_length_mix": None,
        "eval_schedule_selection": "reference_stratified",
        "train_hour_of_day_mode": "random",
        "test_hour_of_day_mode": "random",
        "eval_hour_of_day_mode": "fixed",
        "horizon_profile": "config",
        "observation_semantics": "target31_v2",
        "action_semantics": "residual_pid",
        "residual_action_scale": [0.75, 0.75, 0.75, 0.75, 0.75],
        "residual_gate_min": 0.0,
        "light_control_mode": "step",
        "light_segments_per_photoperiod": 3,
        "include_electricity_price_observation": False,
        "mask_schedule_context_observation": False,
        "pid_anchor_light": None,
        "controller_design": "residual_pid_sac",
        "curriculum_profile": "off",
        "runtime_profile": "default",
        "price_model_type": None,
        "tou_tariff_scenario": None,
        "electricity_price": None,
        "co2_price": None,
        "lettuce_price_fw": None,
        "constant_price": None,
        "train_context_sampling_reference_weights": None,
        "constraint_selection_interval_epochs": 10,
        "constraint_selection_start_epoch": 1,
        "constraint_selection_n_schedules": 4,
        "constraint_selection_n_episodes_per_schedule": 3,
        "constraint_selection_reward_weight": 1.0,
        "constraint_selection_constraint_cost_weight": 1.0,
        "constraint_selection_constraint_active_ratio_weight": 0.0,
        "constraint_selection_cost_mode": "overall",
        "constraint_selection_early_termination_weight": 10.0,
        "constraint_selection_harvest_fail_weight": 1.0,
        "constraint_selection_safety_override_weight": 0.1,
    }

    if os.path.exists(str(config_path)):
        import yaml
        with open(str(config_path), "r", encoding="utf-8") as f:
            yaml_cfg = yaml.safe_load(f) or {}
        # 覆盖默认值
        for key in ["epoch", "nstep", "batch_size", "hidden_sizes", "gamma",
                    "actor_lr", "critic_lr", "alpha_lr", "train_num", "test_num",
                    "auto_nstep", "nstep_factor",
                    "initial_random_episodes",
                    "step_per_collect_env_steps", "update_per_step",
                    "info_detail_level", "vector_env_backend", "test_interval_epochs",
                    "buffer_size", "seed", "context_sampling_phase",
                    "train_context_sampling_strategy",
                    "context_curriculum",
                    "context_fixed_schedule", "context_narrow_bounds",
                    "episode_length_mode", "episode_days",
                    "episode_length_mix",
                    "train_episode_length_mode", "train_episode_days", "train_episode_length_mix",
                    "test_episode_length_mode", "test_episode_days", "test_episode_length_mix",
                    "eval_episode_length_mode", "eval_episode_days", "eval_episode_length_mix",
                    "eval_schedule_selection",
                    "train_hour_of_day_mode", "test_hour_of_day_mode",
                    "eval_hour_of_day_mode",
                    "horizon_profile",
                    "observation_semantics",
                    "action_semantics", "residual_action_scale", "residual_gate_min",
                    "light_control_mode", "light_segments_per_photoperiod",
                    "include_electricity_price_observation",
                    "mask_schedule_context_observation",
                    "pid_anchor_light",
                    "controller_design", "curriculum_profile", "runtime_profile",
                    "price_model_type", "tou_tariff_scenario", "electricity_price",
                    "co2_price", "lettuce_price_fw", "constant_price",
                    "train_context_sampling_reference_weights",
                    "constraint_selection_interval_epochs",
                    "constraint_selection_start_epoch",
                    "constraint_selection_n_schedules",
                    "constraint_selection_n_episodes_per_schedule",
                    "constraint_selection_reward_weight",
                    "constraint_selection_constraint_cost_weight",
                    "constraint_selection_constraint_active_ratio_weight",
                    "constraint_selection_cost_mode",
                    "constraint_selection_early_termination_weight",
                    "constraint_selection_harvest_fail_weight",
                    "constraint_selection_safety_override_weight"]:
            if key in yaml_cfg:
                defaults[key] = yaml_cfg[key]
        # 处理 hidden_sizes 可能是列表或字符串
        if "policy_net_arch" in yaml_cfg:
            defaults["hidden_sizes"] = yaml_cfg["policy_net_arch"]

    return defaults


def parse_residual_action_scale(values) -> float | list[float] | None:
    if values in (None, [], ()):
        return None
    parsed = [float(v) for v in values]
    if len(parsed) == 1:
        return float(parsed[0])
    if len(parsed) != 5:
        raise ValueError("residual_action_scale must be a scalar or five values.")
    return parsed


def canonicalize_residual_action_scale(
    scale_value,
    *,
    default_scalar: float = 0.75,
) -> list[float]:
    if scale_value in (None, [], ()):
        return [float(default_scalar)] * 5
    if isinstance(scale_value, (int, float)):
        return [float(scale_value)] * 5
    if isinstance(scale_value, str):
        return [float(scale_value)] * 5
    parsed = [float(v) for v in scale_value]
    if len(parsed) == 1:
        return [float(parsed[0])] * 5
    if len(parsed) != 5:
        raise ValueError("residual_action_scale must be a scalar or five values.")
    return parsed


def is_climate_only_residual_scale(scale_value) -> bool:
    try:
        parsed = canonicalize_residual_action_scale(scale_value)
    except (TypeError, ValueError):
        return False
    tol = 1e-9
    return abs(parsed[0]) <= tol and abs(parsed[1]) <= tol


def enforce_controller_design_consistency(params: dict) -> None:
    design_name = str(params.get("controller_design", "residual_pid_sac")).strip().lower()
    if design_name != "climate_only_residual_pid_sac":
        return
    scale = canonicalize_residual_action_scale(params.get("residual_action_scale", 0.75))
    scale[0] = 0.0
    scale[1] = 0.0
    params["residual_action_scale"] = scale


def normalize_vector_env_backend(value: str | None) -> str:
    backend = str(value or "auto").strip().lower()
    if backend not in {"auto", "dummy", "subproc", "shmem"}:
        raise ValueError(f"Unsupported vector_env_backend: {value}")
    return backend


def build_vector_env(
    env_fns,
    backend: str | None,
    *,
    env_role: str,
):
    from tianshou.env import DummyVectorEnv, ShmemVectorEnv, SubprocVectorEnv

    env_fns = list(env_fns)
    env_num = len(env_fns)
    requested = normalize_vector_env_backend(backend)
    if requested == "auto":
        candidates = ["subproc", "dummy"] if env_num > 1 else ["dummy"]
    else:
        candidates = [requested]

    classes = {
        "dummy": DummyVectorEnv,
        "subproc": SubprocVectorEnv,
        "shmem": ShmemVectorEnv,
    }
    last_exc = None
    for candidate in candidates:
        vector_env = None
        try:
            vector_env = classes[candidate](env_fns)
            print(
                f"[vector-env] role={env_role} backend={candidate} envs={env_num}",
                flush=True,
            )
            return vector_env, candidate
        except Exception as exc:
            last_exc = exc
            if vector_env is not None:
                try:
                    vector_env.close()
                except Exception:
                    pass
            if requested != "auto":
                raise
            print(
                f"[vector-env] role={env_role} backend={candidate} failed: {exc}",
                flush=True,
            )
    raise RuntimeError(
        f"Unable to construct vector env for role={env_role}. Last error: {last_exc}"
    )


def apply_runtime_thread_limits(device: str, params: dict) -> dict:
    """
    Reduce CPU thread oversubscription when GPU training and vector envs run together.

    This does not change the learning objective. It only trims host-side thread
    contention so sub-process env workers and the main learner do not fight over
    the same CPU cores.
    """
    requested_backend = normalize_vector_env_backend(
        params.get("vector_env_backend", "auto")
    )
    train_num = max(1, int(params.get("train_num", 1)))
    is_cuda = str(device or "").strip().lower().startswith("cuda")
    should_limit_threads = is_cuda and train_num > 1 and requested_backend != "dummy"

    summary = {
        "enabled": bool(should_limit_threads),
        "torch_num_threads": None,
        "torch_num_interop_threads": None,
        "env": {},
    }
    if not should_limit_threads:
        return summary

    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        if not os.environ.get(key):
            os.environ[key] = "1"
        summary["env"][key] = os.environ.get(key)
    return summary


def build_context_curriculum_from_profile(
    profile: str | None,
    *,
    epoch_count: int,
) -> list[dict] | None:
    profile_name = str(profile or "off").strip().lower()
    if profile_name in {"off", "none", "direct"}:
        return None

    warmup_epochs = max(1, int(round(max(epoch_count, 1) * 0.4)))

    if profile_name == "legacy_narrow_to_full":
        return [
            {
                "start_epoch": 1,
                "until_epoch": warmup_epochs,
                "context_sampling_phase": "narrow",
                "context_narrow_bounds": {
                    "t1_min": 12,
                    "t1_max": 16,
                    "t2_min": 12,
                    "t2_max": 16,
                    "N1_min": 16,
                    "N1_max": 20,
                    "rho2_min": 28,
                    "rho2_max": 40,
                },
            },
            {
                "start_epoch": warmup_epochs + 1,
                "until_epoch": max(epoch_count, warmup_epochs + 1),
                "context_sampling_phase": "full",
            },
        ]

    if profile_name == "target_to_full":
        return [
            {
                "start_epoch": 1,
                "until_epoch": warmup_epochs,
                "context_sampling_phase": "full_target_feasible",
            },
            {
                "start_epoch": warmup_epochs + 1,
                "until_epoch": max(epoch_count, warmup_epochs + 1),
                "context_sampling_phase": "full",
            },
        ]

    raise ValueError(f"Unsupported curriculum_profile: {profile}")


def apply_curriculum_profile(params: dict, profile: str | None) -> None:
    profile_name = str(profile or params.get("curriculum_profile", "off")).strip().lower()
    params["curriculum_profile"] = profile_name
    if profile_name == "config":
        return
    params["context_sampling_phase"] = "full"
    params["context_curriculum"] = build_context_curriculum_from_profile(
        profile_name,
        epoch_count=int(params.get("epoch", 1)),
    )


def apply_controller_design(params: dict, design: str | None) -> None:
    design_name = str(design or params.get("controller_design", "residual_pid_sac")).strip().lower()
    params["controller_design"] = design_name
    if design_name == "contextual_sac":
        params["action_semantics"] = "absolute"
        return
    if design_name == "residual_pid_sac":
        params["action_semantics"] = "residual_pid"
        return
    if design_name == "gated_residual_pid_sac":
        params["action_semantics"] = "residual_pid_gated"
        return
    if design_name == "climate_only_residual_pid_sac":
        params["action_semantics"] = "residual_pid"
        enforce_controller_design_consistency(params)
        return
    raise ValueError(f"Unsupported controller_design: {design}")


def apply_horizon_profile(params: dict, profile: str | None) -> None:
    profile_name = str(profile or params.get("horizon_profile", "config")).strip().lower()
    params["horizon_profile"] = profile_name
    if profile_name == "config":
        return
    if profile_name == "fast_t2max":
        params.update(
            {
                "episode_length_mode": "max_t2",
                "episode_days": None,
                "episode_length_mix": None,
                "train_episode_length_mode": "max_t2",
                "train_episode_days": None,
                "train_episode_length_mix": None,
                "test_episode_length_mode": "max_t2",
                "test_episode_days": None,
                "test_episode_length_mix": None,
                "eval_episode_length_mode": "max_t2",
                "eval_episode_days": None,
                "eval_episode_length_mix": None,
            }
        )
        return
    if profile_name == "mainline_long_horizon":
        mixed_profile = [
            {"mode": "max_t2", "weight": 0.7},
            {"mode": "max_total_cycle", "weight": 0.3},
        ]
        params.update(
            {
                "episode_length_mode": "max_t2",
                "episode_days": None,
                "episode_length_mix": [dict(item) for item in mixed_profile],
                "train_episode_length_mode": "mixed",
                "train_episode_days": None,
                "train_episode_length_mix": [dict(item) for item in mixed_profile],
                "test_episode_length_mode": "max_t2",
                "test_episode_days": None,
                "test_episode_length_mix": None,
                "eval_episode_length_mode": "max_t2",
                "eval_episode_days": None,
                "eval_episode_length_mix": None,
            }
        )
        return
    raise ValueError(f"Unsupported horizon_profile: {profile}")


def apply_runtime_profile(params: dict, profile: str | None) -> None:
    profile_name = str(profile or params.get("runtime_profile", "default")).strip().lower()
    params["runtime_profile"] = profile_name
    if profile_name == "default":
        return
    if profile_name == "pilot_fast":
        params.update(
            {
                "epoch": 12,
                "batch_size": 256,
                "train_num": 4,
                "test_num": 1,
                "initial_random_episodes": 2,
                "step_per_collect_env_steps": 8,
                "update_per_step": 1.0,
                "test_interval_epochs": 6,
                "vector_env_backend": "auto",
                "train_episode_length_mode": "fixed_days",
                "train_episode_days": 4.0,
                "test_episode_length_mode": "fixed_days",
                "test_episode_days": 4.0,
                "eval_episode_length_mode": "max_total_cycle",
                "eval_episode_days": None,
                "constraint_selection_interval_epochs": 0,
                "constraint_selection_start_epoch": 999999,
            }
        )
        return
    if profile_name == "pilot_ultrafast":
        params.update(
            {
                "epoch": 8,
                "batch_size": 256,
                "train_num": 1,
                "test_num": 1,
                "initial_random_episodes": 2,
                "step_per_collect_env_steps": 1,
                "update_per_step": 0.5,
                "test_interval_epochs": 8,
                "vector_env_backend": "dummy",
                "nstep_factor": 0.5,
                "train_episode_length_mode": "fixed_days",
                "train_episode_days": 2.0,
                "test_episode_length_mode": "fixed_days",
                "test_episode_days": 2.0,
                "eval_episode_length_mode": "fixed_days",
                "eval_episode_days": 2.0,
                "constraint_selection_interval_epochs": 0,
                "constraint_selection_start_epoch": 999999,
            }
        )
        return
    raise ValueError(f"Unsupported runtime_profile: {profile}")


def _steps_from_days(days: float, dt_seconds: float = 600.0) -> int:
    return max(1, int(round(float(days) * 24.0 * 3600.0 / float(dt_seconds))))


def resolve_episode_length_steps(
    mode: str | None,
    *,
    episode_days: float | None,
    episode_length_mix,
    schedule_bounds: dict,
    dt_seconds: float = 600.0,
) -> int:
    resolved_mode = str(mode or "max_t2").strip().lower()
    t2_max_days = float(schedule_bounds.get("t2_max", 14))
    total_cycle_max_days = float(
        schedule_bounds.get(
            "total_cycle_max",
            float(schedule_bounds.get("t1_max", 14)) + float(schedule_bounds.get("t2_max", 14)),
        )
    )

    if resolved_mode in {"schedule_t2", "max_t2"}:
        return _steps_from_days(t2_max_days, dt_seconds=dt_seconds)
    if resolved_mode in {"total_cycle", "max_total_cycle"}:
        return _steps_from_days(total_cycle_max_days, dt_seconds=dt_seconds)
    if resolved_mode == "fixed_days":
        reference_days = episode_days if episode_days is not None else t2_max_days
        return _steps_from_days(reference_days, dt_seconds=dt_seconds)
    if resolved_mode in {"mixed", "mixed_horizon", "mixed_episode", "curriculum"}:
        entries = episode_length_mix or []
        candidate_steps = []
        for entry in entries:
            if isinstance(entry, dict):
                candidate_steps.append(
                    resolve_episode_length_steps(
                        entry.get("mode"),
                        episode_days=entry.get("days", entry.get("episode_days")),
                        episode_length_mix=entry.get("episode_length_mix"),
                        schedule_bounds=schedule_bounds,
                        dt_seconds=dt_seconds,
                    )
                )
            elif entry is not None:
                candidate_steps.append(
                    resolve_episode_length_steps(
                        str(entry),
                        episode_days=None,
                        episode_length_mix=None,
                        schedule_bounds=schedule_bounds,
                        dt_seconds=dt_seconds,
                    )
                )
        if candidate_steps:
            return max(candidate_steps)
        return max(
            _steps_from_days(t2_max_days, dt_seconds=dt_seconds),
            _steps_from_days(total_cycle_max_days, dt_seconds=dt_seconds),
        )
    return _steps_from_days(t2_max_days, dt_seconds=dt_seconds)


def maybe_resolve_auto_nstep(
    params: dict,
    *,
    schedule_bounds: dict,
    dt_seconds: float = 600.0,
) -> int:
    train_mode, train_days, train_mix = resolve_phase_episode_settings_from_params(params, "train")
    train_episode_steps = resolve_episode_length_steps(
        train_mode,
        episode_days=train_days,
        episode_length_mix=train_mix,
        schedule_bounds=schedule_bounds,
        dt_seconds=dt_seconds,
    )
    params["resolved_train_episode_steps"] = int(train_episode_steps)
    params["resolved_train_episode_days"] = float(
        train_episode_steps * dt_seconds / 3600.0 / 24.0
    )
    if not bool(params.get("auto_nstep", True)):
        return int(params["nstep"])
    factor = max(float(params.get("nstep_factor", 1.0)), 0.1)
    resolved_nstep = max(1, int(math.ceil(train_episode_steps * factor)))
    params["nstep"] = int(resolved_nstep)
    return int(resolved_nstep)


def resolve_total_epoch_steps(params: dict, train_num: int) -> int:
    """
    Resolve the trainer-facing step_per_epoch in total env steps.

    Tianshou's offpolicy_trainer counts vector-env steps in aggregate, while our
    auto nstep logic is defined from a single-environment episode horizon.
    Therefore we scale by train_num here so that "one epoch ≈ one per-env
    horizon" remains semantically true when train_num > 1.
    """
    per_env_nstep = max(1, int(params.get("nstep", 1)))
    return int(per_env_nstep * max(int(train_num), 1))


def attach_inlet_seedling_metadata(payload: dict, params: dict) -> dict:
    """Attach resolved inlet-seedling metadata to saved reports."""
    if payload is None:
        return payload
    payload["inlet_seedling_metadata"] = dict(
        params.get("inlet_seedling_metadata", {}) or {}
    )
    payload["inlet_seedling_metadata_source"] = str(
        params.get("inlet_seedling_metadata_source", "") or ""
    )
    return payload


def resolve_phase_episode_settings_from_params(params: dict, prefix: str):
    """Resolve per-phase episode horizon settings with global fallback."""
    mode = params.get(f"{prefix}_episode_length_mode")
    if mode is None:
        mode = params.get("episode_length_mode")
    days = params.get(f"{prefix}_episode_days")
    if days is None:
        days = params.get("episode_days")
    mix = params.get(f"{prefix}_episode_length_mix")
    if mix is None:
        mix = params.get("episode_length_mix")
    return mode, days, mix


def normalise_context_curriculum(params: dict) -> list[dict]:
    """Normalise epoch-wise contextual curriculum stages from params."""
    raw_curriculum = params.get("context_curriculum")
    if raw_curriculum in (None, "", []):
        return []
    if isinstance(raw_curriculum, dict):
        raw_curriculum = [raw_curriculum]

    stages: list[dict] = []
    prev_until = 0
    final_epoch = int(params.get("epoch", 0) or 0)
    default_phase = str(params.get("context_sampling_phase", "full"))

    for index, raw_stage in enumerate(raw_curriculum):
        if not isinstance(raw_stage, dict):
            continue
        stage = dict(raw_stage)
        start_epoch = int(stage.get("start_epoch", prev_until + 1))
        until_epoch = stage.get("until_epoch", stage.get("end_epoch", stage.get("epoch")))
        if until_epoch is None:
            until_epoch = final_epoch if final_epoch > 0 else start_epoch
        until_epoch = int(until_epoch)
        if until_epoch < start_epoch:
            raise ValueError(
                f"context_curriculum stage {index} has until_epoch < start_epoch."
            )
        stage["start_epoch"] = int(start_epoch)
        stage["until_epoch"] = int(until_epoch)
        stage["context_sampling_phase"] = str(
            stage.get("context_sampling_phase", stage.get("phase", default_phase))
        )
        stages.append(stage)
        prev_until = until_epoch

    if not stages:
        return []

    if final_epoch > 0 and stages[-1]["until_epoch"] < final_epoch:
        tail_stage = dict(stages[-1])
        tail_stage["start_epoch"] = int(stages[-1]["until_epoch"] + 1)
        tail_stage["until_epoch"] = int(final_epoch)
        stages.append(tail_stage)
    return stages


def resolve_context_curriculum_stage(
    curriculum: list[dict],
    epoch: int,
) -> dict | None:
    """Resolve the active curriculum stage for the given epoch."""
    if not curriculum:
        return None
    epoch_int = max(1, int(epoch))
    for index, stage in enumerate(curriculum):
        if int(stage["start_epoch"]) <= epoch_int <= int(stage["until_epoch"]):
            resolved = dict(stage)
            resolved["stage_index"] = int(index)
            return resolved
    resolved = dict(curriculum[-1])
    resolved["stage_index"] = int(len(curriculum) - 1)
    return resolved


def build_train_reset_options_from_stage(
    params: dict,
    sample_context: bool,
    train_episode_mode,
    train_episode_days,
    train_episode_mix,
    stage: dict | None = None,
) -> dict:
    """Build train reset options with optional curriculum-stage overrides."""
    stage = dict(stage or {})
    return build_context_reset_options(
        sample_context=sample_context,
        context_sampling_phase=stage.get(
            "context_sampling_phase",
            params.get("context_sampling_phase"),
        ),
        context_sampling_strategy=stage.get(
            "train_context_sampling_strategy",
            stage.get(
                "context_sampling_strategy",
                params.get("train_context_sampling_strategy", "distributed_cycle"),
            ),
        ),
        fixed_schedule=stage.get(
            "context_fixed_schedule",
            stage.get("fixed_schedule", params.get("context_fixed_schedule")),
        ),
        narrow_bounds=stage.get(
            "context_narrow_bounds",
            stage.get("narrow_bounds", params.get("context_narrow_bounds")),
        ),
        context_sampling_reference_weights=stage.get(
            "train_context_sampling_reference_weights",
            stage.get(
                "context_sampling_reference_weights",
                params.get("train_context_sampling_reference_weights"),
            ),
        ),
        episode_length_mode=stage.get(
            "train_episode_length_mode",
            stage.get("episode_length_mode", train_episode_mode),
        ),
        episode_days=stage.get(
            "train_episode_days",
            stage.get("episode_days", train_episode_days),
        ),
        episode_length_mix=stage.get(
            "train_episode_length_mix",
            stage.get("episode_length_mix", train_episode_mix),
        ),
        hour_of_day_mode=stage.get(
            "train_hour_of_day_mode",
            stage.get("hour_of_day_mode", params.get("train_hour_of_day_mode", "random")),
        ),
    )


def apply_vector_env_reset_options(
    vector_env,
    reset_options: dict,
    *,
    cycle_seed: int | None = None,
) -> None:
    """Update persistent reset options on every sub-env in a vector env."""
    env_count = len(vector_env.get_env_attr("_persistent_reset_options"))
    for env_id in range(env_count):
        vector_env.set_env_attr(
            "_persistent_reset_options",
            prepare_reset_options_for_env(
                reset_options,
                env_rank=env_id,
                env_count=env_count,
                cycle_seed=cycle_seed,
            ),
            id=env_id,
        )


# =============================================================================
# 环境工厂（顶层函数供 multiprocessing pickle）
# =============================================================================

def _make_env(
    seed: int = None,
    sample_context: bool = True,
    reset_options: dict = None,
    rl_param_overrides: dict = None,
):
    """向量化工环境的工厂函数（必须在顶层以支持pickle序列化）。"""
    def _fn():
        env = create_env_from_run_params(
            rl_param_overrides,
            project_root=PROJECT_ROOT,
        )
        opts = dict(reset_options or {})
        opts["sample_context"] = bool(sample_context)
        if seed is not None:
            env.reset(seed=seed, options=opts)
        else:
            env.reset(options=opts)
        return env
    return _fn


def _create_initialized_env(
    seed: int = None,
    sample_context: bool = True,
    reset_options: dict = None,
    rl_param_overrides: dict = None,
):
    """Top-level env constructor so Windows subproc workers can pickle it cleanly."""
    env = create_env_from_run_params(
        rl_param_overrides,
        project_root=PROJECT_ROOT,
    )
    opts = dict(reset_options or {})
    opts["sample_context"] = bool(sample_context)
    if seed is not None:
        env.reset(seed=seed, options=opts)
    else:
        env.reset(options=opts)
    return env


def _make_env_picklable(
    seed: int = None,
    sample_context: bool = True,
    reset_options: dict = None,
    rl_param_overrides: dict = None,
):
    """Return a partial around a top-level constructor for subproc/shmem vector envs."""
    return functools.partial(
        _create_initialized_env,
        seed=seed,
        sample_context=sample_context,
        reset_options=dict(reset_options or {}),
        rl_param_overrides=dict(rl_param_overrides or {}),
    )


# =============================================================================
# WandB / TensorBoard 集成
# =============================================================================

class SafeSummaryWriter:
    """SummaryWriter wrapper that degrades gracefully on Windows file-lock issues."""

    def __init__(self, *args, **kwargs):
        self._writer = SummaryWriter(*args, **kwargs)
        self._enabled = True
        self._warned = False

    def _disable(self, op_name: str, exc: Exception) -> None:
        if not self._warned:
            print(
                "[TensorBoard] 写入失败，已自动关闭后续 TensorBoard 日志。"
                f" op={op_name} err={type(exc).__name__}: {exc}"
            )
            self._warned = True
        self._enabled = False
        try:
            self._writer.close()
        except Exception:
            pass

    def __getattr__(self, name):
        attr = getattr(self._writer, name)
        if not callable(attr):
            return attr

        def _wrapped(*args, **kwargs):
            if not self._enabled:
                return None
            try:
                return attr(*args, **kwargs)
            except Exception as exc:
                self._disable(name, exc)
                return None

        return _wrapped


def setup_logging(log_dir: str, experiment: str, use_wandb: bool = True,
                  wandb_project: str = "PFAL-contextual-SAC") -> object:
    """初始化日志记录器。"""
    writer = SafeSummaryWriter(log_dir=log_dir)
    logger = TensorboardLogger(writer)

    wandb_run = None
    if use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=wandb_project,
                name=experiment,
                dir=log_dir,
                sync_tensorboard=False,
                save_code=True,
                notes="Contextual SAC for dual-zone PFAL optimal control",
            )
            print(f"[WandB] 初始化: {wandb_run.project}/{wandb_run.name}")
            print(f"[WandB] URL: {wandb_run.url}")
        except ImportError:
            print("[WandB] 未安装，跳过（仅使用TensorBoard）")
        except Exception as e:
            print(f"[WandB] 初始化失败: {e}，仅使用TensorBoard")

    return logger, wandb_run


def wandb_log_metrics(wandb_run, metrics: dict, step: int = None):
    """将指标记录到WandB（安全处理未初始化情况）。"""
    if wandb_run is not None:
        try:
            wandb_run.log(metrics, step=step)
        except Exception:
            pass


# =============================================================================
# 多排程评估
# =============================================================================

def evaluate_on_schedules(
    policy,
    n_schedules: int = 10,
    n_episodes_per_schedule: int = 1,
    seed: int = 42,
    context_sampling_phase: str = "full",
    fixed_schedule: dict = None,
    narrow_bounds: dict = None,
    eval_schedule_selection: str = "coverage",
    episode_length_mode: str = "max_t2",
    episode_days: float = None,
    episode_length_mix = None,
    hour_of_day_mode: str = "fixed",
    hour_of_day: float = None,
    rl_param_overrides: dict = None,
    verbose: bool = True,
) -> dict:
    """
    在多个随机采样的排程上评估策略泛化能力。

    这是上下文RL的核心评估方式：训练时策略见过各种排程，
    测试时在全新排程上验证泛化性能。
    """
    if int(n_schedules) <= 0 or int(n_episodes_per_schedule) <= 0:
        return {
            "mean_reward": 0.0,
            "std_reward": 0.0,
            "min_reward": 0.0,
            "max_reward": 0.0,
            "mean_length": 0.0,
            "std_length": 0.0,
            "mean_sim_days": 0.0,
            "std_sim_days": 0.0,
            "mean_harvest_g": 0.0,
            "mean_cost": 0.0,
            "mean_reward_per_day": 0.0,
            "mean_harvest_g_per_day": 0.0,
            "mean_cost_per_day": 0.0,
            "mean_constraint_cost": 0.0,
            "mean_constraint_cost_per_day": 0.0,
            "mean_constraint_active_ratio": 0.0,
            "mean_constraint_climate_cost": 0.0,
            "mean_constraint_daily_light_cost": 0.0,
            "mean_constraint_target_progress_cost": 0.0,
            "mean_constraint_event_cost": 0.0,
            "mean_constraint_termination_cost": 0.0,
            "mean_constraint_temp_cost": 0.0,
            "mean_constraint_co2_cost": 0.0,
            "mean_constraint_rh_cost": 0.0,
            "mean_constraint_dli_cost": 0.0,
            "mean_constraint_photoperiod_cost": 0.0,
            "mean_constraint_harvest_fail_cost": 0.0,
            "mean_constraint_safety_override_cost": 0.0,
            "mean_constraint_thermal_meltdown_cost": 0.0,
            "mean_constraint_ode_failure_cost": 0.0,
            "mean_completion_ratio": 0.0,
            "mean_harvest_target_ratio": 0.0,
            "mean_harvest_target_shortfall_g": 0.0,
            "mean_harvest_target_surplus_g": 0.0,
            "harvest_fail_episode_count": 0,
            "harvest_fail_episode_ratio": 0.0,
            "harvest_fail_batch_count": 0,
            "early_termination_count": 0,
            "early_termination_ratio": 0.0,
            "terminated_episode_count": 0,
            "truncated_episode_count": 0,
            "safety_override_episode_count": 0,
            "safety_override_episode_ratio": 0.0,
            "mean_safety_overrides_per_episode": 0.0,
            "termination_reason_counts": {},
            "termination_reason_shares": {},
            "eval_schedule_selection": str(eval_schedule_selection),
            "episode_length_mode": str(episode_length_mode),
            "hour_of_day_mode": str(hour_of_day_mode),
            "schedules": [],
            "per_schedule_outcomes": [],
            "rewards_per_schedule": [],
            "lengths_per_schedule": [],
            "n_schedules": int(n_schedules),
            "n_episodes_per_schedule": int(n_episodes_per_schedule),
            "eval_skipped": True,
        }

    schedules = select_evaluation_schedules(
        n_schedules=n_schedules,
        seed=seed,
        context_sampling_phase=context_sampling_phase,
        fixed_schedule=fixed_schedule,
        narrow_bounds=narrow_bounds,
        selection_strategy=eval_schedule_selection,
        bounds=load_schedule_bounds(str(PROJECT_ROOT / "configs" / "schedule_params.yaml")),
    )

    policy.eval()
    all_rewards = []
    all_lengths = []
    harvest_list = []
    cost_list = []
    sim_days_list = []
    reward_per_day_list = []
    harvest_per_day_list = []
    cost_per_day_list = []
    completion_ratio_list = []
    constraint_cost_list = []
    constraint_cost_per_day_list = []
    constraint_active_ratio_list = []
    constraint_climate_cost_list = []
    constraint_daily_light_cost_list = []
    constraint_target_progress_cost_list = []
    constraint_event_cost_list = []
    constraint_termination_cost_list = []
    constraint_temp_cost_list = []
    constraint_co2_cost_list = []
    constraint_rh_cost_list = []
    constraint_dli_cost_list = []
    constraint_photoperiod_cost_list = []
    constraint_harvest_fail_cost_list = []
    constraint_safety_override_cost_list = []
    constraint_thermal_meltdown_cost_list = []
    constraint_ode_failure_cost_list = []
    termination_reason_counts = {}
    early_termination_count = 0
    terminated_episode_count = 0
    truncated_episode_count = 0
    safety_override_episode_count = 0
    safety_override_count_list = []
    harvest_target_ratio_list = []
    harvest_target_shortfall_list = []
    harvest_target_surplus_list = []
    harvest_fail_episode_count = 0
    harvest_fail_batch_count = 0
    per_schedule_outcomes = []

    for i, sched in enumerate(schedules):
        sched_rewards = []
        sched_lengths = []
        sched_harvests = []
        sched_costs = []
        sched_days = []
        sched_reward_per_day = []
        sched_harvest_per_day = []
        sched_cost_per_day = []
        sched_completion_ratios = []
        sched_constraint_costs = []
        sched_constraint_costs_per_day = []
        sched_constraint_active_ratios = []
        sched_constraint_climate_costs = []
        sched_constraint_daily_light_costs = []
        sched_constraint_target_progress_costs = []
        sched_constraint_event_costs = []
        sched_constraint_termination_costs = []
        sched_constraint_temp_costs = []
        sched_constraint_co2_costs = []
        sched_constraint_rh_costs = []
        sched_constraint_dli_costs = []
        sched_constraint_photoperiod_costs = []
        sched_constraint_harvest_fail_costs = []
        sched_constraint_safety_override_costs = []
        sched_constraint_thermal_meltdown_costs = []
        sched_constraint_ode_failure_costs = []
        sched_termination_reason_counts = {}
        sched_early_termination_count = 0
        sched_terminated_count = 0
        sched_truncated_count = 0
        sched_safety_override_counts = []
        sched_harvest_target_ratios = []
        sched_harvest_target_shortfalls = []
        sched_harvest_target_surpluses = []
        sched_harvest_fail_episodes = 0
        sched_harvest_fail_batches = 0

        for rep in range(n_episodes_per_schedule):
            env = create_env_from_run_params(
                rl_param_overrides,
                project_root=PROJECT_ROOT,
            )
            reset_options = {
                "schedule": sched,
                "episode_length_mode": episode_length_mode,
                "hour_of_day_mode": hour_of_day_mode,
            }
            if episode_days is not None:
                reset_options["episode_days"] = float(episode_days)
            if episode_length_mix is not None:
                reset_options["episode_length_mix"] = episode_length_mix
            if hour_of_day is not None:
                reset_options["hour_of_day"] = float(hour_of_day)
            obs, _ = env.reset(seed=seed + i * 100 + rep, options=reset_options)
            done = False
            ep_reward = 0.0
            steps = 0
            ep_harvest_target_mass_g = 0.0
            ep_harvest_target_shortfall_g = 0.0
            ep_harvest_target_surplus_g = 0.0
            ep_harvest_fail_batches = 0

            while not done:
                o = np.asarray(obs, dtype=np.float32).reshape(1, -1)
                with torch.inference_mode():
                    action, _ = policy.actor(o)
                action = action[0].cpu().numpy()
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                ep_reward += reward
                steps += 1
                ep_harvest_target_mass_g += float(info.get("harvest_target_mass_g", 0.0))
                ep_harvest_target_shortfall_g += float(
                    info.get("harvest_target_shortfall_g", 0.0)
                )
                ep_harvest_target_surplus_g += float(
                    info.get("harvest_target_surplus_g", 0.0)
                )
                ep_harvest_fail_batches += int(
                    info.get("harvest_fail_n_batches_this_step", 0)
                )

            sim_days = max(
                float(info.get("actual_episode_days", steps * env.dt / 86400.0)),
                1e-9,
            )
            episode_harvest_g = float(
                info.get(
                    "total_harvest_mass_g",
                    float(info.get("batch_summary", {}).get("total_harvest_mass_kg", 0.0))
                    * 1000.0,
                )
            )
            episode_cost = float(info.get("total_cost", 0.0))
            completion_ratio = float(info.get("episode_completion_ratio", 0.0))
            termination_reason = str(info.get("termination_reason", "unknown"))
            ended_early = bool(info.get("ended_early", False))
            safety_override_count = int(info.get("safety_override_count", 0))
            episode_constraint_cost = float(
                info.get(
                    "constraint_cost_episode",
                    float(
                        (info.get("constraint_episode_totals", {}) or {}).get(
                            "overall", 0.0
                        )
                    ),
                )
            )
            episode_constraint_active_ratio = float(
                info.get("constraint_active_ratio", 0.0)
            )
            episode_constraint_totals = dict(info.get("constraint_episode_totals", {}) or {})
            if not episode_constraint_totals:
                episode_constraint_totals = {
                    "climate": float(info.get("constraint_cost_episode_climate", 0.0)),
                    "daily_light": float(
                        info.get("constraint_cost_episode_daily_light", 0.0)
                    ),
                    "target_progress": float(
                        info.get("constraint_cost_episode_target_progress", 0.0)
                    ),
                    "event": float(info.get("constraint_cost_episode_event", 0.0)),
                    "termination": float(
                        info.get("constraint_cost_episode_termination", 0.0)
                    ),
                    "temp": float(info.get("constraint_cost_episode_temp", 0.0)),
                    "co2": float(info.get("constraint_cost_episode_co2", 0.0)),
                }
            episode_target_ratio = (
                episode_harvest_g / max(ep_harvest_target_mass_g, 1e-12)
                if ep_harvest_target_mass_g > 0.0
                else 0.0
            )
            sched_rewards.append(float(ep_reward))
            sched_lengths.append(int(steps))
            sched_harvests.append(episode_harvest_g)
            sched_costs.append(episode_cost)
            sched_days.append(sim_days)
            sched_reward_per_day.append(float(ep_reward / sim_days))
            sched_harvest_per_day.append(float(episode_harvest_g / sim_days))
            sched_cost_per_day.append(float(episode_cost / sim_days))
            sched_completion_ratios.append(completion_ratio)
            sched_constraint_costs.append(episode_constraint_cost)
            sched_constraint_costs_per_day.append(float(episode_constraint_cost / sim_days))
            sched_constraint_active_ratios.append(episode_constraint_active_ratio)
            sched_constraint_climate_costs.append(
                float(episode_constraint_totals.get("climate", 0.0))
            )
            sched_constraint_daily_light_costs.append(
                float(episode_constraint_totals.get("daily_light", 0.0))
            )
            sched_constraint_target_progress_costs.append(
                float(episode_constraint_totals.get("target_progress", 0.0))
            )
            sched_constraint_event_costs.append(
                float(episode_constraint_totals.get("event", 0.0))
            )
            sched_constraint_termination_costs.append(
                float(episode_constraint_totals.get("termination", 0.0))
            )
            sched_constraint_temp_costs.append(
                float(episode_constraint_totals.get("temp", 0.0))
            )
            sched_constraint_co2_costs.append(
                float(episode_constraint_totals.get("co2", 0.0))
            )
            sched_constraint_rh_costs.append(
                float(episode_constraint_totals.get("rh", 0.0))
            )
            sched_constraint_dli_costs.append(
                float(episode_constraint_totals.get("dli", 0.0))
            )
            sched_constraint_photoperiod_costs.append(
                float(episode_constraint_totals.get("photoperiod", 0.0))
            )
            sched_constraint_harvest_fail_costs.append(
                float(episode_constraint_totals.get("harvest_fail", 0.0))
            )
            sched_constraint_safety_override_costs.append(
                float(episode_constraint_totals.get("safety_override", 0.0))
            )
            sched_constraint_thermal_meltdown_costs.append(
                float(episode_constraint_totals.get("thermal_meltdown", 0.0))
            )
            sched_constraint_ode_failure_costs.append(
                float(episode_constraint_totals.get("ode_failure", 0.0))
            )
            sched_safety_override_counts.append(safety_override_count)
            if ep_harvest_target_mass_g > 0.0:
                sched_harvest_target_ratios.append(float(episode_target_ratio))
            sched_harvest_target_shortfalls.append(float(ep_harvest_target_shortfall_g))
            sched_harvest_target_surpluses.append(float(ep_harvest_target_surplus_g))
            sched_harvest_fail_episodes += int(ep_harvest_fail_batches > 0)
            sched_harvest_fail_batches += int(ep_harvest_fail_batches)
            sched_termination_reason_counts[termination_reason] = int(
                sched_termination_reason_counts.get(termination_reason, 0)
            ) + 1
            sched_early_termination_count += int(ended_early)
            sched_terminated_count += int(bool(info.get("terminated", False)))
            sched_truncated_count += int(bool(info.get("truncated", False)))

        mean_r = np.mean(sched_rewards)
        mean_l = np.mean(sched_lengths)
        mean_h = np.mean(sched_harvests)
        mean_c = np.mean(sched_costs)
        mean_d = np.mean(sched_days)
        mean_completion_ratio = (
            float(np.mean(sched_completion_ratios)) if sched_completion_ratios else 0.0
        )
        mean_constraint_cost = (
            float(np.mean(sched_constraint_costs)) if sched_constraint_costs else 0.0
        )
        mean_constraint_cost_per_day = (
            float(np.mean(sched_constraint_costs_per_day))
            if sched_constraint_costs_per_day
            else 0.0
        )
        mean_constraint_active_ratio = (
            float(np.mean(sched_constraint_active_ratios))
            if sched_constraint_active_ratios
            else 0.0
        )
        mean_safety_override_count = (
            float(np.mean(sched_safety_override_counts)) if sched_safety_override_counts else 0.0
        )
        mean_harvest_target_ratio = (
            float(np.mean(sched_harvest_target_ratios)) if sched_harvest_target_ratios else 0.0
        )
        mean_harvest_target_shortfall = (
            float(np.mean(sched_harvest_target_shortfalls))
            if sched_harvest_target_shortfalls else 0.0
        )
        mean_harvest_target_surplus = (
            float(np.mean(sched_harvest_target_surpluses))
            if sched_harvest_target_surpluses else 0.0
        )
        dominant_reason = (
            max(sched_termination_reason_counts.items(), key=lambda kv: kv[1])[0]
            if sched_termination_reason_counts
            else "unknown"
        )
        all_rewards.append(mean_r)
        all_lengths.append(mean_l)
        harvest_list.append(mean_h)
        cost_list.append(mean_c)
        sim_days_list.append(mean_d)
        reward_per_day_list.append(float(np.mean(sched_reward_per_day)))
        harvest_per_day_list.append(float(np.mean(sched_harvest_per_day)))
        cost_per_day_list.append(float(np.mean(sched_cost_per_day)))
        constraint_cost_list.append(mean_constraint_cost)
        constraint_cost_per_day_list.append(mean_constraint_cost_per_day)
        constraint_active_ratio_list.append(mean_constraint_active_ratio)
        constraint_climate_cost_list.append(
            float(np.mean(sched_constraint_climate_costs))
            if sched_constraint_climate_costs
            else 0.0
        )
        constraint_daily_light_cost_list.append(
            float(np.mean(sched_constraint_daily_light_costs))
            if sched_constraint_daily_light_costs
            else 0.0
        )
        constraint_target_progress_cost_list.append(
            float(np.mean(sched_constraint_target_progress_costs))
            if sched_constraint_target_progress_costs
            else 0.0
        )
        constraint_event_cost_list.append(
            float(np.mean(sched_constraint_event_costs))
            if sched_constraint_event_costs
            else 0.0
        )
        constraint_termination_cost_list.append(
            float(np.mean(sched_constraint_termination_costs))
            if sched_constraint_termination_costs
            else 0.0
        )
        constraint_temp_cost_list.append(
            float(np.mean(sched_constraint_temp_costs))
            if sched_constraint_temp_costs
            else 0.0
        )
        constraint_co2_cost_list.append(
            float(np.mean(sched_constraint_co2_costs))
            if sched_constraint_co2_costs
            else 0.0
        )
        constraint_rh_cost_list.append(
            float(np.mean(sched_constraint_rh_costs))
            if sched_constraint_rh_costs
            else 0.0
        )
        constraint_dli_cost_list.append(
            float(np.mean(sched_constraint_dli_costs))
            if sched_constraint_dli_costs
            else 0.0
        )
        constraint_photoperiod_cost_list.append(
            float(np.mean(sched_constraint_photoperiod_costs))
            if sched_constraint_photoperiod_costs
            else 0.0
        )
        constraint_harvest_fail_cost_list.append(
            float(np.mean(sched_constraint_harvest_fail_costs))
            if sched_constraint_harvest_fail_costs
            else 0.0
        )
        constraint_safety_override_cost_list.append(
            float(np.mean(sched_constraint_safety_override_costs))
            if sched_constraint_safety_override_costs
            else 0.0
        )
        constraint_thermal_meltdown_cost_list.append(
            float(np.mean(sched_constraint_thermal_meltdown_costs))
            if sched_constraint_thermal_meltdown_costs
            else 0.0
        )
        constraint_ode_failure_cost_list.append(
            float(np.mean(sched_constraint_ode_failure_costs))
            if sched_constraint_ode_failure_costs
            else 0.0
        )
        completion_ratio_list.extend(sched_completion_ratios)
        for reason, count in sched_termination_reason_counts.items():
            termination_reason_counts[reason] = int(
                termination_reason_counts.get(reason, 0)
            ) + int(count)
        early_termination_count += int(sched_early_termination_count)
        terminated_episode_count += int(sched_terminated_count)
        truncated_episode_count += int(sched_truncated_count)
        safety_override_episode_count += int(sum(1 for v in sched_safety_override_counts if v > 0))
        safety_override_count_list.extend(sched_safety_override_counts)
        harvest_target_ratio_list.extend(sched_harvest_target_ratios)
        harvest_target_shortfall_list.extend(sched_harvest_target_shortfalls)
        harvest_target_surplus_list.extend(sched_harvest_target_surpluses)
        harvest_fail_episode_count += int(sched_harvest_fail_episodes)
        harvest_fail_batch_count += int(sched_harvest_fail_batches)
        per_schedule_outcomes.append(
            {
                "schedule": dict(sched),
                "mean_sim_days": float(mean_d),
                "mean_completion_ratio": mean_completion_ratio,
                "mean_constraint_cost": mean_constraint_cost,
                "mean_constraint_cost_per_day": mean_constraint_cost_per_day,
                "mean_constraint_active_ratio": mean_constraint_active_ratio,
                "mean_constraint_climate_cost": (
                    float(np.mean(sched_constraint_climate_costs))
                    if sched_constraint_climate_costs
                    else 0.0
                ),
                "mean_constraint_daily_light_cost": (
                    float(np.mean(sched_constraint_daily_light_costs))
                    if sched_constraint_daily_light_costs
                    else 0.0
                ),
                "mean_constraint_target_progress_cost": (
                    float(np.mean(sched_constraint_target_progress_costs))
                    if sched_constraint_target_progress_costs
                    else 0.0
                ),
                "mean_constraint_event_cost": (
                    float(np.mean(sched_constraint_event_costs))
                    if sched_constraint_event_costs
                    else 0.0
                ),
                "mean_constraint_termination_cost": (
                    float(np.mean(sched_constraint_termination_costs))
                    if sched_constraint_termination_costs
                    else 0.0
                ),
                "mean_safety_override_count": mean_safety_override_count,
                "mean_harvest_target_ratio": mean_harvest_target_ratio,
                "mean_harvest_target_shortfall_g": mean_harvest_target_shortfall,
                "mean_harvest_target_surplus_g": mean_harvest_target_surplus,
                "harvest_fail_episode_ratio": float(
                    sched_harvest_fail_episodes / max(n_episodes_per_schedule, 1)
                ),
                "harvest_fail_batch_count": int(sched_harvest_fail_batches),
                "early_termination_ratio": float(
                    sched_early_termination_count / max(n_episodes_per_schedule, 1)
                ),
                "dominant_termination_reason": str(dominant_reason),
                "termination_reason_counts": dict(sched_termination_reason_counts),
            }
        )

        if verbose:
            print(
                f"  [{i+1:02d}/{n_schedules}] "
                f"sched: t1={sched['t1']:2d} t2={sched['t2']:2d} "
                f"N1={sched['N1']:2d} rho2={sched['rho2']:4.0f}  "
                f"reward={mean_r:8.2f}  r/day={reward_per_day_list[-1]:8.2f}  "
                f"days={mean_d:5.1f}  len={mean_l:5.0f}  "
                f"harvest={harvest_list[-1]:8.1f}g  "
                f"cons={mean_constraint_cost:6.2f}  "
                f"end={dominant_reason:>16s}  "
                f"early={sched_early_termination_count / max(n_episodes_per_schedule, 1):5.2f}  "
                f"safety={mean_safety_override_count:5.1f}"
            )

    total_episodes = max(n_schedules * n_episodes_per_schedule, 1)
    results = {
        "mean_reward": float(np.mean(all_rewards)),
        "std_reward": float(np.std(all_rewards)),
        "min_reward": float(np.min(all_rewards)),
        "max_reward": float(np.max(all_rewards)),
        "mean_length": float(np.mean(all_lengths)),
        "std_length": float(np.std(all_lengths)),
        "mean_sim_days": float(np.mean(sim_days_list)),
        "std_sim_days": float(np.std(sim_days_list)),
        "mean_harvest_g": float(np.mean(harvest_list)),
        "mean_cost": float(np.mean(cost_list)),
        "mean_reward_per_day": float(np.mean(reward_per_day_list)),
        "mean_harvest_g_per_day": float(np.mean(harvest_per_day_list)),
        "mean_cost_per_day": float(np.mean(cost_per_day_list)),
        "mean_constraint_cost": float(np.mean(constraint_cost_list))
        if constraint_cost_list
        else 0.0,
        "mean_constraint_cost_per_day": float(np.mean(constraint_cost_per_day_list))
        if constraint_cost_per_day_list
        else 0.0,
        "mean_constraint_active_ratio": float(np.mean(constraint_active_ratio_list))
        if constraint_active_ratio_list
        else 0.0,
        "mean_constraint_climate_cost": float(np.mean(constraint_climate_cost_list))
        if constraint_climate_cost_list
        else 0.0,
        "mean_constraint_daily_light_cost": float(
            np.mean(constraint_daily_light_cost_list)
        )
        if constraint_daily_light_cost_list
        else 0.0,
        "mean_constraint_target_progress_cost": float(
            np.mean(constraint_target_progress_cost_list)
        )
        if constraint_target_progress_cost_list
        else 0.0,
        "mean_constraint_event_cost": float(np.mean(constraint_event_cost_list))
        if constraint_event_cost_list
        else 0.0,
        "mean_constraint_termination_cost": float(
            np.mean(constraint_termination_cost_list)
        )
        if constraint_termination_cost_list
        else 0.0,
        "mean_constraint_temp_cost": float(np.mean(constraint_temp_cost_list))
        if constraint_temp_cost_list
        else 0.0,
        "mean_constraint_co2_cost": float(np.mean(constraint_co2_cost_list))
        if constraint_co2_cost_list
        else 0.0,
        "mean_constraint_rh_cost": float(np.mean(constraint_rh_cost_list))
        if constraint_rh_cost_list
        else 0.0,
        "mean_constraint_dli_cost": float(np.mean(constraint_dli_cost_list))
        if constraint_dli_cost_list
        else 0.0,
        "mean_constraint_photoperiod_cost": float(
            np.mean(constraint_photoperiod_cost_list)
        )
        if constraint_photoperiod_cost_list
        else 0.0,
        "mean_constraint_harvest_fail_cost": float(
            np.mean(constraint_harvest_fail_cost_list)
        )
        if constraint_harvest_fail_cost_list
        else 0.0,
        "mean_constraint_safety_override_cost": float(
            np.mean(constraint_safety_override_cost_list)
        )
        if constraint_safety_override_cost_list
        else 0.0,
        "mean_constraint_thermal_meltdown_cost": float(
            np.mean(constraint_thermal_meltdown_cost_list)
        )
        if constraint_thermal_meltdown_cost_list
        else 0.0,
        "mean_constraint_ode_failure_cost": float(
            np.mean(constraint_ode_failure_cost_list)
        )
        if constraint_ode_failure_cost_list
        else 0.0,
        "mean_completion_ratio": float(np.mean(completion_ratio_list))
        if completion_ratio_list
        else 0.0,
        "mean_harvest_target_ratio": float(np.mean(harvest_target_ratio_list))
        if harvest_target_ratio_list
        else 0.0,
        "mean_harvest_target_shortfall_g": float(np.mean(harvest_target_shortfall_list))
        if harvest_target_shortfall_list
        else 0.0,
        "mean_harvest_target_surplus_g": float(np.mean(harvest_target_surplus_list))
        if harvest_target_surplus_list
        else 0.0,
        "harvest_fail_episode_count": int(harvest_fail_episode_count),
        "harvest_fail_episode_ratio": float(
            harvest_fail_episode_count / max(total_episodes, 1)
        ),
        "harvest_fail_batch_count": int(harvest_fail_batch_count),
        "early_termination_count": int(early_termination_count),
        "early_termination_ratio": float(early_termination_count / total_episodes),
        "terminated_episode_count": int(terminated_episode_count),
        "truncated_episode_count": int(truncated_episode_count),
        "safety_override_episode_count": int(safety_override_episode_count),
        "safety_override_episode_ratio": float(safety_override_episode_count / total_episodes),
        "mean_safety_overrides_per_episode": float(np.mean(safety_override_count_list))
        if safety_override_count_list
        else 0.0,
        "termination_reason_counts": dict(termination_reason_counts),
        "termination_reason_shares": {
            key: float(count / total_episodes)
            for key, count in sorted(termination_reason_counts.items())
        },
        "eval_schedule_selection": str(eval_schedule_selection),
        "episode_length_mode": str(episode_length_mode),
        "hour_of_day_mode": str(hour_of_day_mode),
        "schedules": schedules,
        "per_schedule_outcomes": per_schedule_outcomes,
        "rewards_per_schedule": all_rewards,
        "lengths_per_schedule": all_lengths,
        "n_schedules": n_schedules,
        "n_episodes_per_schedule": n_episodes_per_schedule,
    }
    return results


# =============================================================================
# 核心训练循环
# =============================================================================

def train(
    experiment: str,
    params: dict,
    use_wandb: bool = True,
    device: str = None,
    policy=None,
) -> tuple:
    """
    使用 Tianshou SAC 训练上下文强化学习策略。

    参数
    ----
    experiment : str
        实验名称（用于日志目录）
    params : dict
        超参数字典（来自 load_rl_params）
    use_wandb : bool
        是否启用 WandB
    device : str
        计算设备（"cuda" 或 "cpu"）

    返回
    ----
    mean_reward, std_reward : tuple
        最终评估的平均/标准差奖励
    """
    from tianshou.trainer import OffpolicyTrainer
    from tianshou.data import Collector

    class SparseEvalOffpolicyTrainer(OffpolicyTrainer):
        def __init__(self, *args, test_interval_epochs: int = 1, **kwargs):
            self.test_interval_epochs = max(1, int(test_interval_epochs))
            super().__init__(*args, **kwargs)

        def test_step(self):
            should_test = (
                self.test_interval_epochs <= 1
                or self.epoch <= 1
                or self.epoch >= self.max_epoch
                or (self.epoch % self.test_interval_epochs == 0)
            )
            if should_test:
                return super().test_step()
            if self.test_fn:
                self.test_fn(self.epoch, self.env_step)
            if self.verbose:
                print(
                    f"Epoch #{self.epoch}: skip test "
                    f"(interval={self.test_interval_epochs}), "
                    f"best_reward: {self.best_reward:.6f} ± "
                    f"{self.best_reward_std:.6f} in #{self.best_epoch}",
                    flush=True,
                )
            return {}, False

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_num = params["train_num"]
    test_num = params["test_num"]
    sample_context = True  # 始终使用上下文RL
    env_rl_overrides = {
        key: params.get(key)
        for key in [
            "observation_semantics",
            "action_semantics",
            "residual_action_scale",
            "light_control_mode",
            "light_segments_per_photoperiod",
            "include_electricity_price_observation",
            "mask_schedule_context_observation",
            "info_detail_level",
            "enable_action_safety_projection",
            "safety_temp_guard_band_c",
            "safety_temp_projection_margin_c",
            "safety_temp_kp",
        ]
        if key in params
    }
    pid_anchor_light = params.get("pid_anchor_light")
    if pid_anchor_light is not None:
        anchor = [float(v) for v in pid_anchor_light]
        if len(anchor) != 2:
            raise ValueError("pid_anchor_light must contain exactly two values: I1 I2.")
        env_rl_overrides["pid_anchor_light"] = anchor
    train_episode_mode, train_episode_days, train_episode_mix = (
        resolve_phase_episode_settings_from_params(params, "train")
    )
    test_episode_mode, test_episode_days, test_episode_mix = (
        resolve_phase_episode_settings_from_params(params, "test")
    )
    eval_episode_mode, eval_episode_days, eval_episode_mix = (
        resolve_phase_episode_settings_from_params(params, "eval")
    )
    context_curriculum = normalise_context_curriculum(params)
    initial_curriculum_stage = resolve_context_curriculum_stage(context_curriculum, 1)
    train_reset_options = build_train_reset_options_from_stage(
        params,
        sample_context=sample_context,
        train_episode_mode=train_episode_mode,
        train_episode_days=train_episode_days,
        train_episode_mix=train_episode_mix,
        stage=initial_curriculum_stage,
    )
    test_reset_options = build_context_reset_options(
        sample_context=sample_context,
        context_sampling_phase=params.get("context_sampling_phase"),
        fixed_schedule=params.get("context_fixed_schedule"),
        narrow_bounds=params.get("context_narrow_bounds"),
        episode_length_mode=test_episode_mode,
        episode_days=test_episode_days,
        episode_length_mix=test_episode_mix,
        hour_of_day_mode=params.get("test_hour_of_day_mode", "random"),
    )

    # ---- 环境 ----
    vector_env_backend = normalize_vector_env_backend(
        params.get("vector_env_backend", "auto")
    )
    env_factory = (
        _make_env_picklable
        if vector_env_backend in {"auto", "subproc", "shmem"}
        else _make_env
    )
    train_envs, resolved_train_vector_env_backend = build_vector_env([
        env_factory(
            seed=i,
            sample_context=sample_context,
            reset_options=prepare_reset_options_for_env(
                train_reset_options,
                env_rank=i,
                env_count=train_num,
                cycle_seed=int(params.get("seed", 42)),
            ),
            rl_param_overrides=env_rl_overrides,
        )
        for i in range(train_num)
    ], vector_env_backend, env_role="train")
    test_backend = "dummy" if test_num <= 1 else vector_env_backend
    test_envs, resolved_test_vector_env_backend = build_vector_env([
        env_factory(
            seed=1000 + i,
            sample_context=sample_context,
            reset_options=test_reset_options,
            rl_param_overrides=env_rl_overrides,
        )
        for i in range(test_num)
    ], test_backend, env_role="test")
    params["resolved_train_vector_env_backend"] = str(
        resolved_train_vector_env_backend
    )
    params["resolved_test_vector_env_backend"] = str(
        resolved_test_vector_env_backend
    )
    current_train_context = {
        "context_sampling_phase": str(
            train_reset_options.get(
                "context_sampling_phase",
                params.get("context_sampling_phase", "full"),
            )
        ),
        "context_sampling_strategy": str(
            train_reset_options.get(
                "context_sampling_strategy",
                params.get("train_context_sampling_strategy", "distributed_cycle"),
            )
        ),
        "context_fixed_schedule": train_reset_options.get("fixed_schedule"),
        "context_narrow_bounds": train_reset_options.get("narrow_bounds"),
        "context_sampling_reference_weights": train_reset_options.get(
            "context_sampling_reference_weights"
        ),
        "curriculum_stage": (
            dict(initial_curriculum_stage) if initial_curriculum_stage is not None else None
        ),
    }
    current_train_curriculum_signature = {"value": None}

    # ---- 策略 ----
    if policy is None:
        policy = create_policy(
            hidden_sizes=tuple(params["hidden_sizes"]),
            gamma=params["gamma"],
            actor_lr=params["actor_lr"],
            critic_lr=params["critic_lr"],
            alpha_lr=params["alpha_lr"],
            auto_alpha=params["auto_alpha"],
            device=device,
            run_params=params,
            project_root=PROJECT_ROOT,
        )

    # ---- 经验回放 ----
    buffer = VectorReplayBuffer(total_size=params["buffer_size"], buffer_num=train_num)

    # ---- 收集器 ----
    train_collector = Collector(
        policy, train_envs, buffer, exploration_noise=params["exploration_noise"]
    )
    test_collector = Collector(policy, test_envs)

    # ---- 初始探索 ----
    print("执行初始随机探索...")
    initial_random_episodes = max(
        1,
        int(params.get("initial_random_episodes", max(4, train_num // 2))),
    )
    train_collector.collect(n_episode=initial_random_episodes, random=True)

    # ---- 日志目录 ----
    algo_name = "sac_contextual"
    log_name = os.path.join("PFAL-contextual-SAC", algo_name, str(experiment))
    log_path = os.path.join(str(PROJECT_ROOT / "log"), log_name)
    os.makedirs(log_path, exist_ok=True)
    print(f"日志目录: {log_path}")

    # ---- TensorBoard + WandB ----
    writer = SafeSummaryWriter(log_dir=log_path)
    logger = TensorboardLogger(writer)
    schedule_bounds = load_schedule_bounds(str(PROJECT_ROOT / "configs" / "schedule_params.yaml"))

    wandb_run = None
    if use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project="PFAL-contextual-SAC",
                name=experiment,
                dir=log_path,
                sync_tensorboard=False,
                save_code=True,
                config={
                    "algorithm": "SAC",
                    "hidden_sizes": params["hidden_sizes"],
                    "gamma": params["gamma"],
                    "batch_size": params["batch_size"],
                    "nstep": params["nstep"],
                    "train_num": train_num,
                    "contextual": True,
                    "context_phase": params.get("context_sampling_phase", "full"),
                    "context_curriculum": context_curriculum,
                    "episode_length_mode": params.get("episode_length_mode", "max_t2"),
                    "episode_days": params.get("episode_days"),
                },
            )
            print(f"[WandB] {wandb_run.project}/{wandb_run.name}")
        except ImportError:
            print("[WandB] 未安装")
        except Exception as e:
            print(f"[WandB] 初始化失败: {e}")

    selection_interval = max(
        int(params.get("constraint_selection_interval_epochs", 10)),
        0,
    )
    selection_start_epoch = max(
        int(params.get("constraint_selection_start_epoch", 1)),
        1,
    )
    selection_n_schedules = max(
        int(params.get("constraint_selection_n_schedules", 4)),
        1,
    )
    selection_n_episodes = max(
        int(params.get("constraint_selection_n_episodes_per_schedule", 1)),
        1,
    )
    selection_enabled = selection_interval > 0
    selection_weights = {
        "reward_weight": float(params.get("constraint_selection_reward_weight", 1.0)),
        "constraint_cost_weight": float(
            params.get("constraint_selection_constraint_cost_weight", 1.0)
        ),
        "constraint_active_ratio_weight": float(
            params.get("constraint_selection_constraint_active_ratio_weight", 0.0)
        ),
        "constraint_cost_mode": str(
            params.get("constraint_selection_cost_mode", "overall")
        ),
        "early_termination_weight": float(
            params.get("constraint_selection_early_termination_weight", 10.0)
        ),
        "harvest_fail_weight": float(
            params.get("constraint_selection_harvest_fail_weight", 1.0)
        ),
        "safety_override_weight": float(
            params.get("constraint_selection_safety_override_weight", 0.1)
        ),
    }

    def maybe_apply_train_curriculum(epoch: int, env_step: int, force: bool = False) -> None:
        stage = resolve_context_curriculum_stage(context_curriculum, epoch)
        train_options = build_train_reset_options_from_stage(
            params,
            sample_context=sample_context,
            train_episode_mode=train_episode_mode,
            train_episode_days=train_episode_days,
            train_episode_mix=train_episode_mix,
            stage=stage,
        )
        signature = json.dumps(
            {
                "context_sampling_phase": train_options.get("context_sampling_phase"),
                "context_sampling_strategy": train_options.get("context_sampling_strategy"),
                "fixed_schedule": train_options.get("fixed_schedule"),
                "narrow_bounds": train_options.get("narrow_bounds"),
                "context_sampling_reference_weights": train_options.get(
                    "context_sampling_reference_weights"
                ),
                "episode_length_mode": train_options.get("episode_length_mode"),
                "episode_days": train_options.get("episode_days"),
                "episode_length_mix": train_options.get("episode_length_mix"),
                "hour_of_day_mode": train_options.get("hour_of_day_mode"),
            },
            sort_keys=True,
            ensure_ascii=True,
        )
        if (not force) and signature == current_train_curriculum_signature["value"]:
            return

        apply_vector_env_reset_options(
            train_envs,
            train_options,
            cycle_seed=int(params.get("seed", 42)),
        )
        current_train_curriculum_signature["value"] = signature
        current_train_context["context_sampling_phase"] = str(
            train_options.get(
                "context_sampling_phase",
                params.get("context_sampling_phase", "full"),
            )
        )
        current_train_context["context_sampling_strategy"] = str(
            train_options.get(
                "context_sampling_strategy",
                params.get("train_context_sampling_strategy", "distributed_cycle"),
            )
        )
        current_train_context["context_fixed_schedule"] = train_options.get("fixed_schedule")
        current_train_context["context_narrow_bounds"] = train_options.get("narrow_bounds")
        current_train_context["context_sampling_reference_weights"] = train_options.get(
            "context_sampling_reference_weights"
        )
        current_train_context["curriculum_stage"] = dict(stage) if stage is not None else None

        metrics = {
            "curriculum/train_stage_index": float(stage.get("stage_index", -1))
            if stage is not None
            else -1.0,
            "curriculum/train_stage_start_epoch": float(stage.get("start_epoch", 1))
            if stage is not None
            else 1.0,
            "curriculum/train_stage_until_epoch": float(stage.get("until_epoch", params["epoch"]))
            if stage is not None
            else float(params["epoch"]),
            "curriculum/train_phase_is_narrow": float(
                current_train_context["context_sampling_phase"] == "narrow"
            ),
            "curriculum/train_phase_is_full": float(
                current_train_context["context_sampling_phase"] == "full"
            ),
            "curriculum/train_phase_is_full_target_feasible": float(
                current_train_context["context_sampling_phase"] == "full_target_feasible"
            ),
        }
        logger.write("scalar", int(env_step), metrics)
        wandb_log_metrics(wandb_run, metrics, step=int(env_step))
        print(
            f"[curriculum] epoch={int(epoch):03d} "
            f"phase={current_train_context['context_sampling_phase']} "
            f"strategy={current_train_context['context_sampling_strategy']} "
            f"stage={metrics['curriculum/train_stage_index']:.0f}"
        )

    def summarize_schedule_coverage(
        prefix: str,
        envs,
        env_step: int,
        *,
        context_sampling_phase: str | None = None,
        fixed_schedule: dict | None = None,
        narrow_bounds: dict | None = None,
    ) -> dict:
        summary = collect_vector_env_schedule_coverage(
            envs,
            context_sampling_phase=context_sampling_phase
            or params.get("context_sampling_phase", "full"),
            fixed_schedule=(
                params.get("context_fixed_schedule")
                if fixed_schedule is None
                else fixed_schedule
            ),
            narrow_bounds=(
                params.get("context_narrow_bounds")
                if narrow_bounds is None
                else narrow_bounds
            ),
            bounds=schedule_bounds,
        )
        metrics = flatten_schedule_coverage_metrics(summary, prefix)
        for name, value in metrics.items():
            writer.add_scalar(name, value, env_step)
        wandb_log_metrics(wandb_run, metrics, step=env_step)
        return summary

    def summarize_termination(prefix: str, envs, env_step: int) -> dict:
        summary = collect_vector_env_termination_stats(envs)
        metrics = flatten_termination_metrics(summary, prefix)
        for name, value in metrics.items():
            writer.add_scalar(name, value, env_step)
        wandb_log_metrics(wandb_run, metrics, step=env_step)
        return summary

    def summarize_constraint(prefix: str, envs, env_step: int) -> dict:
        summary = collect_vector_env_constraint_stats(envs)
        metrics = flatten_constraint_metrics(summary, prefix)
        for name, value in metrics.items():
            writer.add_scalar(name, value, env_step)
        wandb_log_metrics(wandb_run, metrics, step=env_step)
        return summary

    # ---- 最佳模型保存 ----
    best_reward = -float("inf")
    best_selection_score = -float("inf")
    best_selection_summary = None
    selection_history = []
    selection_seed = int(params["seed"]) + 10_000
    selection_history_path = os.path.join(log_path, "constraint_selection_history.json")
    selected_policy_path = os.path.join(log_path, "policy_selected.pth")
    selected_policy_summary_path = os.path.join(log_path, "policy_selected_summary.json")
    latest_policy_path = os.path.join(log_path, "policy_latest.pth")
    latest_policy_summary_path = os.path.join(log_path, "policy_latest_summary.json")
    last_selection_signature = None

    def save_best_fn(pol):
        nonlocal best_reward
        torch.save(pol.state_dict(), os.path.join(log_path, "policy.pth"))

    def save_latest_fn(epoch: int, env_step: int) -> None:
        payload = {
            "latest_epoch": int(epoch),
            "latest_env_step": int(env_step),
            "requested_load_checkpoint": str(getattr(policy, "_loaded_checkpoint_kind", "")),
            "loaded_checkpoint_path": str(getattr(policy, "_loaded_checkpoint_path", "")),
        }
        try:
            torch.save(policy.state_dict(), latest_policy_path)
            save_json_report(latest_policy_summary_path, payload)
        except Exception as exc:
            print(
                "[latest-checkpoint] 保存失败，但不会中断训练。"
                f" err={type(exc).__name__}: {exc}"
            )

    def maybe_run_constraint_selection(
        epoch: int,
        env_step: int,
        force: bool = False,
    ) -> dict | None:
        nonlocal best_selection_score
        nonlocal best_selection_summary
        nonlocal last_selection_signature

        signature = (int(epoch), int(env_step))
        if force:
            if signature == last_selection_signature and best_selection_summary is not None:
                return best_selection_summary
        else:
            if not selection_enabled:
                return None
            if int(epoch) < selection_start_epoch:
                return None
            if (int(epoch) - selection_start_epoch) % selection_interval != 0:
                return None

        selection_eval = evaluate_on_schedules(
            policy,
            n_schedules=selection_n_schedules,
            n_episodes_per_schedule=selection_n_episodes,
            seed=selection_seed,
            context_sampling_phase=params.get("context_sampling_phase", "full"),
            fixed_schedule=params.get("context_fixed_schedule"),
            narrow_bounds=params.get("context_narrow_bounds"),
            eval_schedule_selection=params.get("eval_schedule_selection", "coverage"),
            episode_length_mode=eval_episode_mode,
            episode_days=eval_episode_days,
            episode_length_mix=eval_episode_mix,
            hour_of_day_mode=params.get("eval_hour_of_day_mode", "fixed"),
            rl_param_overrides=params,
            verbose=False,
        )
        selection_score = compute_constraint_aware_selection_score(
            selection_eval,
            **selection_weights,
        )
        selection_summary = dict(selection_eval)
        selection_summary["selection_epoch"] = int(epoch)
        selection_summary["selection_env_step"] = int(env_step)
        selection_summary["selection_score"] = float(selection_score["score"])
        selection_summary["selection_components"] = dict(selection_score["components"])
        selection_summary["selection_weights"] = dict(selection_score["weights"])
        selection_summary["selection_seed"] = int(selection_seed)
        selection_summary["selection_n_schedules"] = int(selection_n_schedules)
        selection_summary["selection_n_episodes_per_schedule"] = int(selection_n_episodes)

        is_best = float(selection_score["score"]) > float(best_selection_score)
        selection_summary["is_best_selection"] = bool(is_best)
        selection_history.append(
            {
                "epoch": int(epoch),
                "env_step": int(env_step),
                "selection_score": float(selection_score["score"]),
                "is_best_selection": bool(is_best),
                "mean_reward": float(selection_eval.get("mean_reward", 0.0)),
                "mean_constraint_cost": float(
                    selection_eval.get("mean_constraint_cost", 0.0)
                ),
                "early_termination_ratio": float(
                    selection_eval.get("early_termination_ratio", 0.0)
                ),
                "harvest_fail_episode_ratio": float(
                    selection_eval.get("harvest_fail_episode_ratio", 0.0)
                ),
                "mean_safety_overrides_per_episode": float(
                    selection_eval.get("mean_safety_overrides_per_episode", 0.0)
                ),
                "selection_components": dict(selection_score["components"]),
                "selection_weights": dict(selection_score["weights"]),
            }
        )

        metrics = {
            "selection/score": float(selection_score["score"]),
            "selection/best_score": float(
                max(best_selection_score, float(selection_score["score"]))
            ),
            "selection/is_best": 1.0 if is_best else 0.0,
            "selection/mean_reward": float(selection_eval.get("mean_reward", 0.0)),
            "selection/mean_constraint_cost": float(
                selection_eval.get("mean_constraint_cost", 0.0)
            ),
            "selection/early_termination_ratio": float(
                selection_eval.get("early_termination_ratio", 0.0)
            ),
            "selection/harvest_fail_episode_ratio": float(
                selection_eval.get("harvest_fail_episode_ratio", 0.0)
            ),
            "selection/mean_safety_overrides_per_episode": float(
                selection_eval.get("mean_safety_overrides_per_episode", 0.0)
            ),
            "selection/safety_override_episode_ratio": float(
                selection_eval.get("safety_override_episode_ratio", 0.0)
            ),
        }
        for name, value in selection_score["components"].items():
            try:
                metrics[f"selection/components/{name}"] = float(value)
            except (TypeError, ValueError):
                continue
        for name, value in metrics.items():
            writer.add_scalar(name, value, int(env_step))
        wandb_log_metrics(wandb_run, metrics, step=int(env_step))

        if is_best:
            best_selection_score = float(selection_score["score"])
            best_selection_summary = dict(selection_summary)
            torch.save(policy.state_dict(), selected_policy_path)
            save_json_report(selected_policy_summary_path, best_selection_summary)
            print(
                f"[selection] epoch={epoch:03d} "
                f"score={best_selection_score:.4f} "
                f"reward={selection_eval.get('mean_reward', 0.0):.4f} "
                f"constraint={selection_eval.get('mean_constraint_cost', 0.0):.4f} "
                f"-> policy_selected.pth"
            )
        else:
            print(
                f"[selection] epoch={epoch:03d} "
                f"score={selection_score['score']:.4f} "
                f"(best={best_selection_score:.4f})"
            )

        save_json_report(
            selection_history_path,
            {
                "selection_seed": int(selection_seed),
                "selection_interval_epochs": int(selection_interval),
                "selection_start_epoch": int(selection_start_epoch),
                "selection_n_schedules": int(selection_n_schedules),
                "selection_n_episodes_per_schedule": int(selection_n_episodes),
                "selection_weights": dict(selection_weights),
                "best_selection_score": float(best_selection_score)
                if best_selection_summary is not None
                else None,
                "best_selection_epoch": (
                    int(best_selection_summary["selection_epoch"])
                    if best_selection_summary is not None
                    else None
                ),
                "history": selection_history,
            },
        )
        last_selection_signature = signature
        return selection_summary

    maybe_apply_train_curriculum(1, 0, force=True)
    save_latest_fn(0, 0)
    summarize_schedule_coverage(
        "coverage/train",
        train_envs,
        0,
        context_sampling_phase=current_train_context["context_sampling_phase"],
        fixed_schedule=current_train_context["context_fixed_schedule"],
        narrow_bounds=current_train_context["context_narrow_bounds"],
    )
    summarize_schedule_coverage("coverage/test", test_envs, 0)
    summarize_termination("termination/train", train_envs, 0)
    summarize_termination("termination/test", test_envs, 0)
    summarize_constraint("constraint/train", train_envs, 0)
    summarize_constraint("constraint/test", test_envs, 0)

    def train_fn(epoch: int, env_step: int) -> None:
        maybe_apply_train_curriculum(epoch, env_step)
        save_latest_fn(epoch, env_step)
        summarize_schedule_coverage(
            "coverage/train",
            train_envs,
            int(env_step),
            context_sampling_phase=current_train_context["context_sampling_phase"],
            fixed_schedule=current_train_context["context_fixed_schedule"],
            narrow_bounds=current_train_context["context_narrow_bounds"],
        )
        summarize_termination("termination/train", train_envs, int(env_step))
        summarize_constraint("constraint/train", train_envs, int(env_step))

    def test_fn(epoch: int, env_step: int) -> None:
        summarize_schedule_coverage(
            "coverage/train",
            train_envs,
            int(env_step),
            context_sampling_phase=current_train_context["context_sampling_phase"],
            fixed_schedule=current_train_context["context_fixed_schedule"],
            narrow_bounds=current_train_context["context_narrow_bounds"],
        )
        summarize_schedule_coverage("coverage/test", test_envs, int(env_step))
        summarize_termination("termination/train", train_envs, int(env_step))
        summarize_termination("termination/test", test_envs, int(env_step))
        summarize_constraint("constraint/train", train_envs, int(env_step))
        summarize_constraint("constraint/test", test_envs, int(env_step))
        maybe_run_constraint_selection(epoch, env_step)

    # ---- 训练 ----
    print("\n" + "=" * 60)
    print("开始训练 — 上下文SAC (PFALEnvContextual)")
    print("=" * 60)
    pprint.pprint({
        k: v for k, v in params.items()
        if k not in ["hidden_sizes"]
    })
    print(f"设备     : {device}")
    print(f"日志路径 : {log_path}")
    print("=" * 60)

    start = datetime.datetime.now()
    print(f"开始时间 : {start.strftime('%Y-%m-%d %H:%M:%S')}")

    step_per_collect_env_steps = max(
        1,
        int(params.get("step_per_collect_env_steps", 1)),
    )
    resolved_step_per_collect = int(step_per_collect_env_steps * max(train_num, 1))
    resolved_step_per_epoch_total = int(resolve_total_epoch_steps(params, train_num))
    resolved_update_per_step = float(params.get("update_per_step", 1.0))
    resolved_test_interval_epochs = max(
        1,
        int(params.get("test_interval_epochs", 1)),
    )
    params["resolved_step_per_collect"] = int(resolved_step_per_collect)
    params["resolved_step_per_epoch_total"] = int(resolved_step_per_epoch_total)
    params["resolved_update_per_step"] = float(resolved_update_per_step)
    params["resolved_test_interval_epochs"] = int(resolved_test_interval_epochs)
    params["resolved_updates_per_epoch"] = float(
        resolved_step_per_epoch_total * max(resolved_update_per_step, 0.0)
    )

    def _run_offpolicy_trainer(*args, **kwargs):
        kwargs["step_per_collect"] = resolved_step_per_collect
        kwargs["update_per_step"] = resolved_update_per_step
        kwargs["test_interval_epochs"] = resolved_test_interval_epochs
        return SparseEvalOffpolicyTrainer(*args, **kwargs).run()

    save_json_report(os.path.join(log_path, "run_config.json"), params)
    result = _run_offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        max_epoch=params["epoch"],
        step_per_epoch=resolved_step_per_epoch_total,
        step_per_collect=1,   # 收集每个step后立即更新（SAC标准做法）
        episode_per_test=test_num,
        batch_size=params["batch_size"],
        train_fn=train_fn,
        test_fn=test_fn,
        save_best_fn=save_best_fn,
        logger=logger,
        update_per_step=1,
        test_in_train=False,
        verbose=True,
    )

    end = datetime.datetime.now()
    elapsed_s = (end - start).total_seconds()
    params["elapsed_seconds"] = float(elapsed_s)
    print(f"\n训练完成: {end.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时  : {elapsed_s / 60:.1f} 分钟")
    pprint.pprint(result)
    final_env_step = int(params["epoch"] * resolved_step_per_epoch_total)
    maybe_apply_train_curriculum(int(params["epoch"]), final_env_step, force=True)
    train_coverage = summarize_schedule_coverage(
        "coverage/train_final",
        train_envs,
        final_env_step,
        context_sampling_phase=current_train_context["context_sampling_phase"],
        fixed_schedule=current_train_context["context_fixed_schedule"],
        narrow_bounds=current_train_context["context_narrow_bounds"],
    )
    test_coverage = summarize_schedule_coverage("coverage/test_final", test_envs, final_env_step)
    train_termination = summarize_termination("termination/train_final", train_envs, final_env_step)
    test_termination = summarize_termination("termination/test_final", test_envs, final_env_step)
    train_constraint = summarize_constraint("constraint/train_final", train_envs, final_env_step)
    test_constraint = summarize_constraint("constraint/test_final", test_envs, final_env_step)
    save_json_report(os.path.join(log_path, "train_schedule_coverage.json"), train_coverage)
    save_json_report(os.path.join(log_path, "test_schedule_coverage.json"), test_coverage)
    save_json_report(os.path.join(log_path, "train_termination_summary.json"), train_termination)
    save_json_report(os.path.join(log_path, "test_termination_summary.json"), test_termination)
    save_json_report(os.path.join(log_path, "train_constraint_summary.json"), train_constraint)
    save_json_report(os.path.join(log_path, "test_constraint_summary.json"), test_constraint)
    selection_summary = maybe_run_constraint_selection(
        int(params["epoch"]),
        final_env_step,
        force=True,
    )
    print(
        f"[episode-end/train] early={train_termination['early_termination_ratio']:.3f}  "
        f"failure={train_termination['failure_termination_ratio']:.3f}  "
        f"time_limit={train_termination['time_limit_completion_ratio']:.3f}"
    )
    print(
        f"[episode-end/test ] early={test_termination['early_termination_ratio']:.3f}  "
        f"failure={test_termination['failure_termination_ratio']:.3f}  "
        f"time_limit={test_termination['time_limit_completion_ratio']:.3f}"
    )
    print(
        f"[constraint/train] mean={train_constraint['mean_constraint_cost']:.3f}  "
        f"active={train_constraint['mean_constraint_active_ratio']:.3f}  "
        f"climate={train_constraint['mean_constraint_climate_cost']:.3f}"
    )
    print(
        f"[constraint/test ] mean={test_constraint['mean_constraint_cost']:.3f}  "
        f"active={test_constraint['mean_constraint_active_ratio']:.3f}  "
        f"climate={test_constraint['mean_constraint_climate_cost']:.3f}"
    )

    # ---- 最佳策略最终评估 ----
    if selection_summary is not None:
        print(
            f"[selection/final] score={selection_summary['selection_score']:.3f}  "
            f"reward={selection_summary['mean_reward']:.3f}  "
            f"constraint={selection_summary['mean_constraint_cost']:.3f}"
        )

    policy.eval()
    test_envs.seed(0)
    test_collector.reset()
    eval_result = test_collector.collect(n_episode=test_num)
    mean_r = float(eval_result["rews"].mean())
    std_r = float(eval_result["rews"].std())
    print(f"\n最终评估 ({test_num} episodes): "
          f"reward={mean_r:.4f} ± {std_r:.4f}, "
          f"len={eval_result['lens'].mean():.1f}")

    wandb_log_metrics(wandb_run, {
        "final/mean_reward": mean_r,
        "final/std_reward": std_r,
        "final/mean_length": float(eval_result["lens"].mean()),
        "final/test_mean_constraint_cost": float(
            test_constraint.get("mean_constraint_cost", 0.0)
        ),
        "final/test_constraint_active_ratio": float(
            test_constraint.get("mean_constraint_active_ratio", 0.0)
        ),
        "training/elapsed_minutes": elapsed_s / 60,
    })

    # ---- 保存最终策略 ----
    torch.save(policy.state_dict(), os.path.join(log_path, "policy_final.pth"))
    print(f"策略已保存: {log_path}/policy_final.pth")

    if wandb_run:
        wandb_run.finish()

    return mean_r, std_r, log_path, result


# =============================================================================
# 命令行入口
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="上下文SAC训练 — PFALEnvContextual（双区多批次植物工厂）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 标准训练（50 epochs，TensorBoard + WandB）
  python experiments/train_pfal_contextual.py --epoch 50

  # 无WandB，CPU训练
  python experiments/train_pfal_contextual.py --epoch 50 --no_wandb --device cpu

  # 自定义超参数（用于超参数搜索）
  python experiments/train_pfal_contextual.py --epoch 100 --nstep 5000 \\
      --batch_size 256 --hidden 256 256 --actor_lr 1e-4

  # 在多个排程上评估泛化性能
  python experiments/train_pfal_contextual.py --eval_only \\
      --load PFAL-contextual-SAC/sac_contextual/my_exp

  # 加载已保存策略并继续训练
  python experiments/train_pfal_contextual.py --load my_exp --epoch 100

  # 评估时在20个不同排程上测试泛化
  python experiments/train_pfal_contextual.py --eval_only --load my_exp \\
      --n_eval_schedules 20 --eval_seed 42
        """
    )
    # 训练参数
    g_train = parser.add_argument_group("训练参数")
    g_train.add_argument("--epoch", type=int, default=None,
                         help="训练轮数（默认: rl_params.yaml 或 50）")
    g_train.add_argument("--nstep", type=int, default=None,
                         help="手动指定每个 epoch 的环境步数；若启用 auto_nstep 则会被自动重算。")
    g_train.add_argument("--batch_size", type=int, default=None,
                         help="批大小（默认: 512）")
    g_train.add_argument("--hidden", type=int, nargs=2, default=None,
                         help="隐藏层大小，如: --hidden 128 128")
    g_train.add_argument("--gamma", type=float, default=None,
                         help="折扣因子（默认: 0.99）")
    g_train.add_argument("--actor_lr", type=float, default=None,
                         help="Actor学习率（默认: 3e-4）")
    g_train.add_argument("--critic_lr", type=float, default=None,
                         help="Critic学习率（默认: 3e-4）")
    g_train.add_argument("--train_num", type=int, default=None,
                         help="并行训练环境数（默认取 rl_params.yaml）")
    g_train.add_argument("--test_num", type=int, default=None,
                         help="并行测试环境数（默认取 rl_params.yaml）")
    g_train.add_argument("--buffer_size", type=int, default=None,
                         help="经验回放缓冲区大小（默认: 1_000_000）")
    g_train.add_argument("--controller_design", type=str, default=None,
                         choices=[
                             "contextual_sac",
                             "residual_pid_sac",
                             "gated_residual_pid_sac",
                             "climate_only_residual_pid_sac",
                         ],
                         help="控制器设计：直接 Contextual SAC 或 PID 残差 SAC")
    g_train.add_argument("--curriculum_profile", type=str, default=None,
                         choices=["config", "off", "legacy_narrow_to_full", "target_to_full"],
                         help="课程学习方案：关闭/沿用配置/窄域到全域/目标可行到全域")
    g_train.add_argument("--runtime_profile", type=str, default=None,
                         choices=["default", "pilot_fast", "pilot_ultrafast"],
                         help="运行时预设：default / pilot_fast / pilot_ultrafast（方法筛选）")
    g_train.add_argument("--horizon_profile", type=str, default=None,
                         choices=["config", "fast_t2max", "mainline_long_horizon"],
                         help="episode-horizon 预设：快筛版 fast_t2max / 主线长时域版 mainline_long_horizon")
    g_train.add_argument("--context_phase", type=str, default=None,
                         help="上下文排程采样阶段，例如 full / narrow / fixed / full_min_feasible / full_target_feasible / full_infeasible")
    g_train.add_argument("--train_context_sampling_strategy", type=str, default=None,
                         choices=["random", "cycle", "distributed_cycle"],
                         help="train schedule sampling strategy")
    g_train.add_argument("--fixed_schedule", type=float, nargs=4, default=None,
                         metavar=("T1", "T2", "N1", "RHO2"),
                         help="固定上层排程: t1 t2 N1 rho2")
    g_train.add_argument("--narrow_bounds", type=float, nargs=8, default=None,
                         metavar=("T1_MIN", "T1_MAX", "T2_MIN", "T2_MAX",
                                  "N1_MIN", "N1_MAX", "RHO2_MIN", "RHO2_MAX"),
                         help="窄域上层排程边界: t1/t2/N1/rho2 的上下界")
    g_train.add_argument("--initial_random_episodes", type=int, default=None,
                         help="initial random exploration episodes")
    g_train.add_argument("--step_per_collect_env_steps", type=int, default=None,
                         help="per-env rollout steps for each collect")
    g_train.add_argument("--update_per_step", type=float, default=None,
                         help="gradient updates per sampled environment step")
    g_train.add_argument("--info_detail_level", type=str, default=None,
                         choices=["compact", "rich"],
                         help="训练时 info 返回粒度：compact 更稳，rich 保留完整诊断")
    g_train.add_argument("--vector_env_backend", type=str, default=None,
                         choices=["auto", "dummy", "subproc", "shmem"],
                         help="向量环境后端：auto 会优先尝试子进程并行，失败则回退 dummy")
    g_train.add_argument("--test_interval_epochs", type=int, default=None,
                         help="训练中完整测试的 epoch 间隔；最终评估仍然会完整执行")
    g_train.add_argument("--residual_action_scale", type=float, nargs="+", default=None,
                         help="残差动作缩放，支持 1 个标量或 5 个动作维度的值")
    g_train.add_argument("--light_control_mode", type=str, default=None,
                         choices=["step", "daily_hold", "segmented_hold"],
                         help="光强执行模式：逐步更新 / 每日一次持有 / 日内少段持有")
    g_train.add_argument("--light_segments_per_photoperiod", type=int, default=None,
                         help="当 light_control_mode=segmented_hold 时，每个光周期分成多少段")
    g_train.add_argument("--include_electricity_price_observation", action="store_true",
                         help="在 RL 观测中显式追加电价信号，用于 TOU 扩展实验")
    g_train.add_argument("--mask_schedule_context_observation", action="store_true",
                         help="Ablation: set schedule-context observation entries (t1,t2,N1,rho2) to zero.")
    g_train.add_argument("--pid_anchor_light", type=float, nargs=2, default=None,
                         metavar=("I1", "I2"),
                         help="Override PID-anchor light targets for residual controllers, e.g. --pid_anchor_light 250 300.")
    g_train.add_argument("--price_model_type", type=str, default=None,
                         choices=["constant", "time_of_use"],
                         help="电价模型：constant 或 time_of_use")
    g_train.add_argument("--tou_tariff_scenario", type=str, default=None,
                         choices=list_tou_tariff_scenarios(),
                         help="内置 TOU 电价场景名称")
    g_train.add_argument("--electricity_price", type=float, default=None,
                         help="固定电价覆盖值 [RMB/kWh]")
    g_train.add_argument("--co2_price", type=float, default=None,
                         help="CO2 价格覆盖值 [RMB/kg]")
    g_train.add_argument("--lettuce_price_fw", type=float, default=None,
                         help="生菜鲜重售价覆盖值 [RMB/kg]")
    g_train.add_argument("--constant_price", type=float, default=None,
                         help="TOU 模型中的平段/回退电价 [RMB/kWh]")
    g_train.add_argument("--auto_nstep", action="store_true",
                         help="根据当前排程空间与训练 horizon 自动重算 nstep")
    g_train.add_argument("--no_auto_nstep", action="store_true",
                         help="关闭自动 nstep，使用手动/配置文件中的 nstep")
    g_train.add_argument("--nstep_factor", type=float, default=None,
                         help="自动 nstep 的放缩系数，默认 1.0")
    g_train.add_argument("--constraint_selection_interval_epochs", type=int, default=None,
                         help="约束感知策略筛选间隔；设为 0 可关闭")
    g_train.add_argument("--constraint_selection_start_epoch", type=int, default=None,
                         help="约束感知策略筛选起始 epoch")
    g_train.add_argument("--constraint_selection_n_schedules", type=int, default=None,
                         help="每次筛选评估的 schedule 数")
    g_train.add_argument("--constraint_selection_n_episodes_per_schedule", type=int, default=None,
                         help="每个筛选 schedule 的 episode 数")
    g_train.add_argument("--constraint_selection_reward_weight", type=float, default=None,
                         help="筛选评分中 reward 项权重")
    g_train.add_argument("--constraint_selection_constraint_cost_weight", type=float, default=None,
                         help="筛选评分中约束成本项权重")
    g_train.add_argument("--constraint_selection_constraint_active_ratio_weight", type=float, default=None,
                         help="筛选评分中约束激活比例项权重")
    g_train.add_argument("--constraint_selection_cost_mode", type=str, default=None,
                         choices=["overall", "climate_plus_daily_light"],
                         help="筛选时约束成本统计方式")
    g_train.add_argument("--constraint_selection_early_termination_weight", type=float, default=None,
                         help="筛选评分中提前终止惩罚权重")
    g_train.add_argument("--constraint_selection_harvest_fail_weight", type=float, default=None,
                         help="筛选评分中采收失败惩罚权重")
    g_train.add_argument("--constraint_selection_safety_override_weight", type=float, default=None,
                         help="筛选评分中 safety override 惩罚权重")
    episode_length_mode_choices = [
        "schedule_t2", "max_t2", "fixed_days", "total_cycle", "max_total_cycle",
        "mixed", "mixed_horizon", "mixed_episode", "curriculum",
    ]
    g_train.add_argument("--episode_length_mode", type=str, default=None,
                         choices=episode_length_mode_choices,
                         help="episode 长度模式")
    g_train.add_argument("--episode_days", type=float, default=None,
                         help="当 episode_length_mode=fixed_days 时的仿真天数")

    # 实验配置
    g_exp = parser.add_argument_group("实验配置")
    g_train.add_argument("--train_episode_length_mode", type=str, default=None,
                         choices=episode_length_mode_choices,
                         help="override train episode horizon mode")
    g_train.add_argument("--train_episode_days", type=float, default=None,
                         help="override train episode days when train mode=fixed_days")
    g_train.add_argument("--test_episode_length_mode", type=str, default=None,
                         choices=episode_length_mode_choices,
                         help="override test episode horizon mode")
    g_train.add_argument("--test_episode_days", type=float, default=None,
                         help="override test episode days when test mode=fixed_days")
    g_train.add_argument("--eval_episode_length_mode", type=str, default=None,
                         choices=episode_length_mode_choices,
                         help="override eval episode horizon mode")
    g_train.add_argument("--eval_episode_days", type=float, default=None,
                         help="override eval episode days when eval mode=fixed_days")
    g_exp.add_argument("--experiment", type=str, default=None,
                       help="实验名称（默认: 自动时间戳）")
    g_exp.add_argument("--seed", type=int, default=None,
                       help="随机种子（默认: 42）")
    g_exp.add_argument("--device", type=str, default=None,
                       choices=["cpu", "cuda"],
                       help="计算设备（默认: cuda if available else cpu）")

    # WandB
    g_wandb = parser.add_argument_group("日志")
    g_wandb.add_argument("--no_wandb", action="store_true",
                         help="禁用WandB日志")
    g_wandb.add_argument("--wandb_project", type=str, default="PFAL-contextual-SAC",
                         help="WandB项目名")

    # 加载/保存
    g_io = parser.add_argument_group("模型加载/保存")
    g_io.add_argument("--load", type=str, default=None,
                      help="从指定实验加载策略权重")
    g_io.add_argument("--save_path", type=str, default=None,
                      help="策略保存路径（默认: log/.../policy_final.pth）")
    g_io.add_argument("--load_checkpoint", type=str, default=None,
                      choices=["best", "final", "selected", "auto"],
                      help="加载哪类检查点；eval_only 默认 auto，继续训练默认 best")

    # 评估模式
    g_eval = parser.add_argument_group("评估")
    g_eval.add_argument("--eval_only", action="store_true",
                        help="仅运行评估（需要 --load）")
    g_eval.add_argument("--n_eval_schedules", type=int, default=10,
                        help="评估用排程数量（默认: 10）")
    g_eval.add_argument("--n_eval_episodes_per_schedule", type=int, default=5,
                        help="每个排程评估多少个完整episode（默认: 1）")
    g_eval.add_argument("--eval_seed", type=int, default=42,
                        help="评估随机种子")
    g_eval.add_argument("--eval_selection", type=str, default=None,
                        choices=["coverage", "random", "reference_stratified"],
                        help="评估排程选择方式")
    g_eval.add_argument("--final_eval_schedules", type=int, default=None,
                        help="训练结束后泛化评估的排程数量")
    g_eval.add_argument("--final_eval_episodes_per_schedule", type=int, default=None,
                        help="训练结束后每个排程评估的 episode 数")

    g_eval.add_argument("--skip_selected_policy_eval", action="store_true",
                        help="skip the duplicate selected-policy generalization eval")

    return parser.parse_args()


def main():
    mp.freeze_support()
    args = parse_args()

    if args.no_wandb:
        os.environ.setdefault("WANDB_MODE", "disabled")
        os.environ.setdefault("WANDB_SILENT", "true")
        os.environ.setdefault("WANDB_DISABLE_CODE", "true")
        os.environ.setdefault("WANDB_DISABLED", "true")

    # ---- 加载默认超参数 ----
    params = load_rl_params()
    if args.load:
        phase_episode_keys = [
            "train_episode_length_mode", "train_episode_days", "train_episode_length_mix",
            "test_episode_length_mode", "test_episode_days", "test_episode_length_mix",
            "eval_episode_length_mode", "eval_episode_days", "eval_episode_length_mix",
        ]
        saved_run_cfg = load_saved_run_config(args.load, PROJECT_ROOT)
        if saved_run_cfg:
            params.update(saved_run_cfg)
            missing_phase_keys = [key for key in phase_episode_keys if key not in saved_run_cfg]
            if missing_phase_keys:
                for key in missing_phase_keys:
                    params[key] = None
                print("[run_config] run_config.json 缂哄皯 phase episode 閰嶇疆锛屾寜鍏ㄥ眬 episode_length_* 鍏煎鍔犺浇")
            if "observation_semantics" not in saved_run_cfg:
                params["observation_semantics"] = "legacy31"
                print("[run_config] run_config.json 缺少 observation_semantics，按旧版 legacy31 观测兼容加载")
            if "auto_nstep" not in saved_run_cfg:
                params["auto_nstep"] = False
                print("[run_config] 老实验未记录 auto_nstep，继续训练时默认保持手动 nstep 兼容")
            if "controller_design" not in saved_run_cfg:
                action_semantics = str(params.get("action_semantics", "absolute")).lower()
                if action_semantics == "absolute":
                    params["controller_design"] = "contextual_sac"
                elif action_semantics in {"residual_pid_gated", "pid_residual_gated"}:
                    params["controller_design"] = "gated_residual_pid_sac"
                elif is_climate_only_residual_scale(params.get("residual_action_scale")):
                    params["controller_design"] = "climate_only_residual_pid_sac"
                else:
                    params["controller_design"] = "residual_pid_sac"
            if "curriculum_profile" not in saved_run_cfg:
                params["curriculum_profile"] = (
                    "config" if params.get("context_curriculum") not in (None, "", []) else "off"
                )
            if "runtime_profile" not in saved_run_cfg:
                params["runtime_profile"] = "default"
            if "horizon_profile" not in saved_run_cfg:
                params["horizon_profile"] = "config"
            if "light_control_mode" not in saved_run_cfg:
                params["light_control_mode"] = "step"
            if "light_segments_per_photoperiod" not in saved_run_cfg:
                params["light_segments_per_photoperiod"] = 3
            if "train_context_sampling_strategy" not in saved_run_cfg:
                params["train_context_sampling_strategy"] = "random"
            print(f"[run_config] 从已保存实验配置加载: {args.load}")
        else:
            for key in phase_episode_keys:
                params[key] = None
            params["action_semantics"] = "absolute"
            params["observation_semantics"] = "legacy31"
            params["controller_design"] = "contextual_sac"
            params["curriculum_profile"] = "off"
            params["runtime_profile"] = "default"
            params["horizon_profile"] = "config"
            params["auto_nstep"] = False
            params["light_control_mode"] = "step"
            params["light_segments_per_photoperiod"] = 3
            params["train_context_sampling_strategy"] = "random"
            print("[run_config] 未找到 run_config.json，按旧版 absolute 动作 + legacy31 观测兼容加载")

    load_checkpoint = args.load_checkpoint or ("auto" if args.eval_only else "best")

    apply_horizon_profile(params, params.get("horizon_profile", "config"))
    apply_runtime_profile(params, args.runtime_profile or params.get("runtime_profile", "default"))
    if args.horizon_profile is not None:
        apply_horizon_profile(params, args.horizon_profile)
    apply_controller_design(params, args.controller_design or params.get("controller_design"))
    apply_curriculum_profile(params, args.curriculum_profile or params.get("curriculum_profile"))

    # ---- 命令行覆盖 ----
    for key in ["epoch", "nstep", "batch_size", "gamma",
                "actor_lr", "critic_lr", "train_num", "test_num",
                "initial_random_episodes", "step_per_collect_env_steps",
                "update_per_step", "nstep_factor", "info_detail_level",
                "vector_env_backend", "test_interval_epochs",
                "buffer_size", "seed"]:
        val = getattr(args, key, None)
        if val is not None:
            params[key] = val
    for key in [
        "constraint_selection_interval_epochs",
        "constraint_selection_start_epoch",
        "constraint_selection_n_schedules",
        "constraint_selection_n_episodes_per_schedule",
        "constraint_selection_reward_weight",
        "constraint_selection_constraint_cost_weight",
        "constraint_selection_constraint_active_ratio_weight",
        "constraint_selection_cost_mode",
        "constraint_selection_early_termination_weight",
        "constraint_selection_harvest_fail_weight",
        "constraint_selection_safety_override_weight",
    ]:
        val = getattr(args, key, None)
        if val is not None:
            params[key] = val

    if args.nstep is not None:
        params["auto_nstep"] = False
    if args.auto_nstep:
        params["auto_nstep"] = True
    if args.no_auto_nstep:
        params["auto_nstep"] = False

    if args.hidden is not None:
        params["hidden_sizes"] = list(args.hidden)
    if args.residual_action_scale is not None:
        params["residual_action_scale"] = parse_residual_action_scale(args.residual_action_scale)
    if args.light_control_mode is not None:
        params["light_control_mode"] = str(args.light_control_mode)
    if args.light_segments_per_photoperiod is not None:
        params["light_segments_per_photoperiod"] = max(
            1, int(args.light_segments_per_photoperiod)
        )
    if args.include_electricity_price_observation:
        params["include_electricity_price_observation"] = True
    if args.mask_schedule_context_observation:
        params["mask_schedule_context_observation"] = True
    if args.pid_anchor_light is not None:
        params["pid_anchor_light"] = [float(v) for v in args.pid_anchor_light]
    for key in [
        "price_model_type",
        "tou_tariff_scenario",
        "electricity_price",
        "co2_price",
        "lettuce_price_fw",
        "constant_price",
    ]:
        val = getattr(args, key, None)
        if val is not None:
            params[key] = val

    if args.context_phase is not None:
        params["context_sampling_phase"] = args.context_phase
    if args.train_context_sampling_strategy is not None:
        params["train_context_sampling_strategy"] = args.train_context_sampling_strategy
    if args.fixed_schedule is not None:
        params["context_fixed_schedule"] = parse_fixed_schedule_values(args.fixed_schedule)
        if args.context_phase is None:
            params["context_sampling_phase"] = "fixed"
    if args.narrow_bounds is not None:
        params["context_narrow_bounds"] = parse_narrow_bounds_values(args.narrow_bounds)
        if args.context_phase is None and args.fixed_schedule is None:
            params["context_sampling_phase"] = "narrow"
    if args.episode_length_mode is not None:
        params["episode_length_mode"] = args.episode_length_mode
    if args.episode_days is not None:
        params["episode_days"] = args.episode_days
    if args.episode_length_mode is not None:
        for prefix in ("train", "test", "eval"):
            if getattr(args, f"{prefix}_episode_length_mode") is None:
                params[f"{prefix}_episode_length_mode"] = args.episode_length_mode
    if args.episode_days is not None:
        for prefix in ("train", "test", "eval"):
            if getattr(args, f"{prefix}_episode_days") is None:
                params[f"{prefix}_episode_days"] = args.episode_days
    for prefix in ("train", "test", "eval"):
        phase_mode = getattr(args, f"{prefix}_episode_length_mode")
        phase_days = getattr(args, f"{prefix}_episode_days")
        if phase_mode is not None:
            params[f"{prefix}_episode_length_mode"] = phase_mode
        if phase_days is not None:
            params[f"{prefix}_episode_days"] = phase_days
    if args.eval_selection is not None:
        params["eval_schedule_selection"] = args.eval_selection

    enforce_controller_design_consistency(params)

    if str(params.get("curriculum_profile", "off")).strip().lower() != "config":
        apply_curriculum_profile(params, params.get("curriculum_profile"))

    schedule_bounds = load_schedule_bounds(str(PROJECT_ROOT / "configs" / "schedule_params.yaml"))
    maybe_resolve_auto_nstep(params, schedule_bounds=schedule_bounds, dt_seconds=600.0)
    final_eval_schedules = (
        10 if args.final_eval_schedules is None else max(0, int(args.final_eval_schedules))
    )
    final_eval_episodes = (
        1
        if args.final_eval_episodes_per_schedule is None
        else max(0, int(args.final_eval_episodes_per_schedule))
    )
    params["final_eval_schedules"] = final_eval_schedules
    params["final_eval_episodes_per_schedule"] = final_eval_episodes

    inlet_seedling_meta = sync_inlet_seedling_metadata(
        params,
        project_root=PROJECT_ROOT,
        fallback_preset="external_nursery_proxy" if args.load else None,
    )
    print(
        "[inlet_seedling] "
        f"preset={inlet_seedling_meta['initial_seedling_mass_preset']} "
        f"basis={inlet_seedling_meta['initial_seedling_mass_basis']} "
        f"source={inlet_seedling_meta['inlet_seedling_metadata_source']}"
    )
    print(
        "[rl-design] "
        f"controller_design={params.get('controller_design')} "
        f"action_semantics={params.get('action_semantics')} "
        f"residual_action_scale={params.get('residual_action_scale')} "
        f"light_control_mode={params.get('light_control_mode')} "
        f"light_segments={params.get('light_segments_per_photoperiod')} "
        f"price_obs={bool(params.get('include_electricity_price_observation', False))} "
        f"mask_schedule_context={bool(params.get('mask_schedule_context_observation', False))} "
        f"pid_anchor_light={params.get('pid_anchor_light')} "
        f"curriculum_profile={params.get('curriculum_profile')} "
        f"runtime_profile={params.get('runtime_profile')} "
        f"horizon_profile={params.get('horizon_profile')}"
    )
    print(
        "[economics] "
        f"price_model_type={params.get('price_model_type') or 'config'} "
        f"tou_tariff_scenario={params.get('tou_tariff_scenario') or 'none'} "
        f"electricity_price={params.get('electricity_price')} "
        f"co2_price={params.get('co2_price')} "
        f"lettuce_price_fw={params.get('lettuce_price_fw')}"
    )
    print(
        "[rl-runtime] "
        f"auto_nstep={bool(params.get('auto_nstep', True))} "
        f"nstep={int(params.get('nstep', 0))} "
        f"train_episode_steps={int(params.get('resolved_train_episode_steps', 0))} "
        f"step_per_epoch_total={int(resolve_total_epoch_steps(params, params.get('train_num', 1)))} "
        f"train_num={int(params.get('train_num', 0))} "
        f"test_num={int(params.get('test_num', 0))}"
    )

    eval_episode_mode, eval_episode_days, eval_episode_mix = (
        resolve_phase_episode_settings_from_params(params, "eval")
    )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    runtime_thread_limits = apply_runtime_thread_limits(device, params)
    params["runtime_thread_limits"] = dict(runtime_thread_limits)
    if runtime_thread_limits.get("enabled"):
        thread_bits = []
        if runtime_thread_limits.get("torch_num_threads") is not None:
            thread_bits.append(f"torch={runtime_thread_limits.get('torch_num_threads')}")
        if runtime_thread_limits.get("torch_num_interop_threads") is not None:
            thread_bits.append(
                f"interop={runtime_thread_limits.get('torch_num_interop_threads')}"
            )
        thread_bits.append(f"env={runtime_thread_limits.get('env')}")
        print(
            "[runtime-threads] "
            + " ".join(thread_bits)
        )

    # ---- 随机种子 ----
    seed = params.get("seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # ---- 实验名称 ----
    if args.experiment:
        experiment = args.experiment
    else:
        ts = datetime.datetime.now().strftime("%m%d_%H%M%S")
        experiment = f"exp_{ts}"

    policy = None

    # ---- 评估模式 ----
    if args.eval_only:
        print("=" * 60)
        print("评估模式")
        print("=" * 60)

        # 创建并加载策略
        policy = create_policy(
            hidden_sizes=tuple(params["hidden_sizes"]),
            gamma=params["gamma"],
            actor_lr=params["actor_lr"],
            critic_lr=params["critic_lr"],
            alpha_lr=params["alpha_lr"],
            auto_alpha=params["auto_alpha"],
            device=device,
            run_params=params,
            project_root=PROJECT_ROOT,
        )

        if args.load:
            policy = load_policy(
                policy,
                args.load,
                device=device,
                checkpoint=load_checkpoint,
                project_root=PROJECT_ROOT,
            )
        else:
            print("错误: 评估模式需要 --load 指定实验名称")
            print("  例: --load PFAL-contextual-SAC/sac_contextual/my_exp")
            return

        # 多排程泛化评估
        print(f"\n在 {args.n_eval_schedules} 个随机排程上评估...\n")
        eval_results = evaluate_on_schedules(
            policy,
            n_schedules=args.n_eval_schedules,
            n_episodes_per_schedule=args.n_eval_episodes_per_schedule,
            seed=args.eval_seed,
            context_sampling_phase=params.get("context_sampling_phase", "full"),
            fixed_schedule=params.get("context_fixed_schedule"),
            narrow_bounds=params.get("context_narrow_bounds"),
            eval_schedule_selection=params.get("eval_schedule_selection", "coverage"),
            episode_length_mode=eval_episode_mode,
            episode_days=eval_episode_days,
            episode_length_mix=eval_episode_mix,
            hour_of_day_mode=params.get("eval_hour_of_day_mode", "fixed"),
            rl_param_overrides=params,
            verbose=True,
        )

        print("\n" + "=" * 60)
        print("泛化评估汇总")
        print("=" * 60)
        print(f"排程数量       : {eval_results['n_schedules']}")
        print(f"每排程episodes : {eval_results['n_episodes_per_schedule']}")
        print(f"平均奖励       : {eval_results['mean_reward']:.4f} ± {eval_results['std_reward']:.4f}")
        print(f"奖励范围       : [{eval_results['min_reward']:.4f}, {eval_results['max_reward']:.4f}]")
        print(f"平均回合长度   : {eval_results['mean_length']:.1f} ± {eval_results['std_length']:.1f}")
        print(f"平均仿真天数   : {eval_results['mean_sim_days']:.2f} ± {eval_results['std_sim_days']:.2f}")
        print(f"平均奖励/天    : {eval_results['mean_reward_per_day']:.4f}")
        print(f"平均采收量     : {eval_results['mean_harvest_g']:.1f} g/episode")
        print(f"平均采收/天    : {eval_results['mean_harvest_g_per_day']:.1f} g/day")
        print(f"平均控制成本   : {eval_results['mean_cost']:.4f} RMB/episode")
        print(f"平均成本/天    : {eval_results['mean_cost_per_day']:.4f} RMB/day")
        print(f"平均约束代价   : {eval_results['mean_constraint_cost']:.4f} /episode")
        print(f"平均约束/天    : {eval_results['mean_constraint_cost_per_day']:.4f} /day")
        print(f"约束活跃比例   : {eval_results['mean_constraint_active_ratio']:.4f}")
        print(
            f"气候/日光/目标 : "
            f"{eval_results['mean_constraint_climate_cost']:.4f} / "
            f"{eval_results['mean_constraint_daily_light_cost']:.4f} / "
            f"{eval_results['mean_constraint_target_progress_cost']:.4f}"
        )
        print(
            f"事件/终止代价 : "
            f"{eval_results['mean_constraint_event_cost']:.4f} / "
            f"{eval_results['mean_constraint_termination_cost']:.4f}"
        )
        print(f"评估选择方式   : {eval_results['eval_schedule_selection']}")
        print(f"episode 模式   : {eval_results['episode_length_mode']}")
        print(
            f"加载检查点     : "
            f"{getattr(policy, '_loaded_checkpoint_kind', load_checkpoint)}"
        )
        print(
            f"检查点路径     : "
            f"{getattr(policy, '_loaded_checkpoint_path', 'N/A')}"
        )
        print(f"completion 比例 : {eval_results['mean_completion_ratio']:.4f}")
        print(f"early-stop 比例 : {eval_results['early_termination_ratio']:.4f}")
        print(f"shield介入比例 : {eval_results['safety_override_episode_ratio']:.4f}")
        print(f"平均shield次数: {eval_results['mean_safety_overrides_per_episode']:.2f}")
        print(f"time-limit 占比 : {eval_results['termination_reason_shares'].get('time_limit', 0.0):.4f}")
        print(f"ODE-failure 占比: {eval_results['termination_reason_shares'].get('ode_failure', 0.0):.4f}")
        print(f"meltdown 占比  : {eval_results['termination_reason_shares'].get('thermal_meltdown', 0.0):.4f}")
        print(f"termination 统计: {eval_results['termination_reason_counts']}")
        eval_output_dir = resolve_experiment_dir(args.load, PROJECT_ROOT)
        eval_results["requested_load_checkpoint"] = str(load_checkpoint)
        eval_results["loaded_checkpoint_kind"] = str(
            getattr(policy, "_loaded_checkpoint_kind", load_checkpoint)
        )
        eval_results["loaded_checkpoint_path"] = str(
            getattr(policy, "_loaded_checkpoint_path", "")
        )
        attach_inlet_seedling_metadata(eval_results, params)
        checkpoint_tag = eval_results["loaded_checkpoint_kind"]
        save_json_report(
            str(eval_output_dir / f"generalization_eval_eval_only_{checkpoint_tag}.json"),
            eval_results,
        )
        save_json_report(str(eval_output_dir / "generalization_eval_eval_only.json"), eval_results)
        print("=" * 60)
        return

    # ---- 加载已有策略继续训练 ----
    if args.load:
        print(f"[加载策略] 从实验: {args.load}")
        policy = create_policy(
            hidden_sizes=tuple(params["hidden_sizes"]),
            gamma=params["gamma"],
            actor_lr=params["actor_lr"],
            critic_lr=params["critic_lr"],
            alpha_lr=params["alpha_lr"],
            auto_alpha=params["auto_alpha"],
            device=device,
            run_params=params,
            project_root=PROJECT_ROOT,
        )
        policy = load_policy(
            policy,
            args.load,
            device=device,
            checkpoint=load_checkpoint,
            project_root=PROJECT_ROOT,
        )
        print(f"[继续训练] epoch={params['epoch']}, nstep={params['nstep']}")

    # ---- 训练 ----
    mean_r, std_r, log_path, trainer_result = train(
        experiment=experiment,
        params=params,
        use_wandb=not args.no_wandb,
        device=device,
        policy=policy,
    )

    # ---- 训练后多排程泛化评估 ----
    print("\n" + "=" * 60)
    print(f"训练后泛化评估（{final_eval_schedules}个随机排程）")
    print("=" * 60)

    # 重新加载最佳策略
    best_policy = create_policy(
        hidden_sizes=tuple(params["hidden_sizes"]),
        gamma=params["gamma"],
        actor_lr=params["actor_lr"],
        critic_lr=params["critic_lr"],
        alpha_lr=params["alpha_lr"],
        auto_alpha=params["auto_alpha"],
        device=device,
        run_params=params,
        project_root=PROJECT_ROOT,
    )
    best_policy = load_policy(
        best_policy,
        experiment,
        device=device,
        checkpoint="best",
        project_root=PROJECT_ROOT,
    )

    eval_results = evaluate_on_schedules(
        best_policy,
        n_schedules=final_eval_schedules,
        n_episodes_per_schedule=final_eval_episodes,
        seed=params["seed"],
        context_sampling_phase=params.get("context_sampling_phase", "full"),
        fixed_schedule=params.get("context_fixed_schedule"),
        narrow_bounds=params.get("context_narrow_bounds"),
        eval_schedule_selection=params.get("eval_schedule_selection", "coverage"),
        episode_length_mode=eval_episode_mode,
        episode_days=eval_episode_days,
        episode_length_mix=eval_episode_mix,
        hour_of_day_mode=params.get("eval_hour_of_day_mode", "fixed"),
        rl_param_overrides=params,
        verbose=True,
    )
    eval_results["loaded_checkpoint_kind"] = str(
        getattr(best_policy, "_loaded_checkpoint_kind", "best")
    )
    eval_results["loaded_checkpoint_path"] = str(
        getattr(best_policy, "_loaded_checkpoint_path", "")
    )
    attach_inlet_seedling_metadata(eval_results, params)
    save_json_report(os.path.join(log_path, "generalization_eval.json"), eval_results)
    selected_eval_results = None
    selected_policy_path = os.path.join(log_path, "policy_selected.pth")
    if (not args.skip_selected_policy_eval) and os.path.exists(selected_policy_path):
        selected_policy = create_policy(
            hidden_sizes=tuple(params["hidden_sizes"]),
            gamma=params["gamma"],
            actor_lr=params["actor_lr"],
            critic_lr=params["critic_lr"],
            alpha_lr=params["alpha_lr"],
            auto_alpha=params["auto_alpha"],
            device=device,
            run_params=params,
            project_root=PROJECT_ROOT,
        )
        selected_policy = load_policy(
            selected_policy,
            log_path,
            device=device,
            checkpoint="selected",
            project_root=PROJECT_ROOT,
        )
        selected_eval_results = evaluate_on_schedules(
            selected_policy,
            n_schedules=final_eval_schedules,
            n_episodes_per_schedule=final_eval_episodes,
            seed=params["seed"],
            context_sampling_phase=params.get("context_sampling_phase", "full"),
            fixed_schedule=params.get("context_fixed_schedule"),
            narrow_bounds=params.get("context_narrow_bounds"),
            eval_schedule_selection=params.get("eval_schedule_selection", "coverage"),
            episode_length_mode=eval_episode_mode,
            episode_days=eval_episode_days,
            episode_length_mix=eval_episode_mix,
            hour_of_day_mode=params.get("eval_hour_of_day_mode", "fixed"),
            rl_param_overrides=params,
            verbose=True,
        )
        selected_eval_results["loaded_checkpoint_kind"] = str(
            getattr(selected_policy, "_loaded_checkpoint_kind", "selected")
        )
        selected_eval_results["loaded_checkpoint_path"] = str(
            getattr(selected_policy, "_loaded_checkpoint_path", selected_policy_path)
        )
        attach_inlet_seedling_metadata(selected_eval_results, params)
        save_json_report(
            os.path.join(log_path, "generalization_eval_selected.json"),
            selected_eval_results,
        )
        print(
            f"[selected-eval] reward={selected_eval_results['mean_reward']:.4f} "
            f"constraint={selected_eval_results['mean_constraint_cost']:.4f}"
        )

    report_eval_results = selected_eval_results if selected_eval_results is not None else eval_results
    report_eval_source = "selected" if selected_eval_results is not None else "final"

    training_summary = {
        "experiment": str(experiment),
        "log_path": str(log_path),
        "device": str(device),
        "controller_design": str(params.get("controller_design", "")),
        "action_semantics": str(params.get("action_semantics", "")),
        "curriculum_profile": str(params.get("curriculum_profile", "")),
        "runtime_profile": str(params.get("runtime_profile", "")),
        "horizon_profile": str(params.get("horizon_profile", "")),
        "auto_nstep": bool(params.get("auto_nstep", True)),
        "nstep": int(params.get("nstep", 0)),
        "resolved_train_episode_steps": int(params.get("resolved_train_episode_steps", 0)),
        "resolved_train_episode_days": float(params.get("resolved_train_episode_days", 0.0)),
        "resolved_step_per_epoch_total": int(params.get("resolved_step_per_epoch_total", 0)),
        "resolved_step_per_collect_total": int(params.get("resolved_step_per_collect", 0)),
        "resolved_update_per_step": float(params.get("resolved_update_per_step", 0.0)),
        "resolved_test_interval_epochs": int(
            params.get("resolved_test_interval_epochs", 0)
        ),
        "resolved_updates_per_epoch": float(params.get("resolved_updates_per_epoch", 0.0)),
        "batch_size": int(params.get("batch_size", 0)),
        "step_per_collect_env_steps": int(params.get("step_per_collect_env_steps", 0)),
        "train_num": int(params.get("train_num", 0)),
        "test_num": int(params.get("test_num", 0)),
        "vector_env_backend": str(params.get("vector_env_backend", "")),
        "resolved_train_vector_env_backend": str(
            params.get("resolved_train_vector_env_backend", "")
        ),
        "resolved_test_vector_env_backend": str(
            params.get("resolved_test_vector_env_backend", "")
        ),
        "runtime_thread_limits": dict(params.get("runtime_thread_limits", {}) or {}),
        "elapsed_seconds": float(params.get("elapsed_seconds", 0.0)),
        "trainer_result": (
            dict(trainer_result) if isinstance(trainer_result, dict) else trainer_result
        ),
        "final_eval_schedules": int(final_eval_schedules),
        "final_eval_episodes_per_schedule": int(final_eval_episodes),
        "skip_selected_policy_eval": bool(args.skip_selected_policy_eval),
        "final_test_mean_reward": float(mean_r),
        "final_test_std_reward": float(std_r),
        "generalization_eval": dict(eval_results),
        "selected_generalization_eval": (
            dict(selected_eval_results) if selected_eval_results is not None else None
        ),
        "report_eval_source": str(report_eval_source),
        "report_generalization_eval": dict(report_eval_results),
    }
    save_json_report(os.path.join(log_path, "training_summary.json"), training_summary)

    print("\n" + "=" * 60)
    print("最终汇总")
    print("=" * 60)
    print(f"实验名称       : {experiment}")
    print(f"日志路径       : {log_path}")
    print(f"最终测试奖励   : {mean_r:.4f} ± {std_r:.4f}")
    print(
        f"泛化奖励({report_eval_source}) : "
        f"{report_eval_results['mean_reward']:.4f} ± {report_eval_results['std_reward']:.4f}"
    )
    print(
        f"泛化奖励范围   : "
        f"[{report_eval_results['min_reward']:.4f}, {report_eval_results['max_reward']:.4f}]"
    )
    print(f"泛化约束代价   : {report_eval_results['mean_constraint_cost']:.4f}")
    print(f"约束活跃比例   : {report_eval_results['mean_constraint_active_ratio']:.4f}")
    print("=" * 60)
    print("\n训练完成！")
    print(f"TensorBoard:  tensorboard --logdir {PROJECT_ROOT / 'log'}")
    print(f"Policy saved: {log_path}/policy.pth")
    if selected_eval_results is not None:
        print(f"Selected policy: {log_path}/policy_selected.pth")


if __name__ == "__main__":
    main()
