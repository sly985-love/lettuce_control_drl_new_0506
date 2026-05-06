# 期刊主实验推荐矩阵与启动命令

日期：2026-04-22  
项目：`lettuce_control_drl_new_0422`

## 1. 适用前提

这套主实验方案默认采用我们当前已经统一后的设定：

- 上层排程决策变量只有 4 个：`t1, t2, N1, rho2`
- `PP=16 h` 不是上层排程变量，而是下层环境控制中的固定参数
- 可行排程全集为 `368` 个 4 维可行组合
- RL 主训练采用 `residual_pid_sac`
- 训练排程采样采用 `distributed_cycle`

如果后续又改了上层排程定义，这份矩阵也必须同步更新。

## 2. 期刊主实验的推荐设计

### 2.1 总体建议

面向高水平期刊，建议把实验分成两层：

1. `训练复现实验层`
   目标是证明 RL 训练结果不是单个 seed 偶然得到的。
   推荐 `3` 个主 seed，优先用 `42 / 52 / 62`。

2. `精确主结果层`
   目标是做全文最核心的 `PID vs RL` 全排程精确对比。
   这一层不建议对每个 seed 都跑 368 个排程的全年 exact baseline，而是：
   先用主 seed 训练结果选出一个“代表性 policy”，再对该 policy 做精确穷举。

### 2.2 为什么主训练推荐 500 epoch

当前项目在修正后采用：

- `train_num = 8`
- `train_context_sampling_strategy = distributed_cycle`
- `nstep = 4032`
- `epoch = 500`

按我们前面的覆盖分析：

- `epoch = 300` 可以作为“最低可接受主线”
- `epoch = 500` 更适合作为期刊主结果默认设置

因此建议：

- `300 epoch` 只用于 pilot / 预筛 / 消融初筛
- `500 epoch` 用于期刊主结果

### 2.3 主 seed 数建议

推荐分两档：

- 标准主结果：`3 seeds = 42, 52, 62`
- 强化稳健性版本：`5 seeds = 42, 52, 62, 72, 82`

如果 GPU 资源有限，正文主表先用 `3 seeds`，补充材料再扩到 `5 seeds`。

### 2.4 代表性 policy 的选择原则

精确 RL baseline 不建议直接挑“最好 seed”，那样容易带来乐观偏差。更合理的做法是：

1. 对每个 seed 的 run，比较 `best / selected / final / auto` checkpoint
2. 在统一评估集上选出该 run 内部的最佳 checkpoint
3. 再在多个 seed 之间选择“中位表现 seed”作为代表性 policy

建议优先比较的指标：

- `mean_reward_per_day`
- `mean_constraint_cost_per_day`
- `early_termination_ratio`
- `harvest_fail_episode_ratio`
- `coverage metrics`

如果论文主叙事强调“保守可信”，代表性 policy 优先选“中位 seed + selected checkpoint”，而不是全局最优 seed。

## 3. 推荐实验矩阵

| ID | 目的 | 推荐设置 | 重复数 | 主要输出 |
| --- | --- | --- | ---: | --- |
| `P0` | 训练烟雾测试 | `train_pfal_contextual_128_500.py`, `epoch=2~5` | 1 | 训练是否正常、coverage 指标是否出现 |
| `T1` | 主训练复现 | `residual_pid_sac`, `epoch=500`, `train_num=8`, `distributed_cycle` | 3 seeds | 训练日志、`training_summary.json`、checkpoint |
| `T2` | seed 与 checkpoint 选择 | 对 `T1` 的各 run 做统一评估 | 3 runs | 代表性 run / checkpoint 选择依据 |
| `M1` | 系统验证 | 默认排程下 `PID + RL` 代表轨迹 | 1 RL run | 默认排程动态轨迹图 |
| `M2` | 精确 PID 基线 | 全部 `368` 可行排程，全年 exact 仿真 | 1 | `pid_exact_schedule_results.csv` |
| `M3` | 精确 RL 基线 | 代表性 RL policy，全部 `368` 可行排程，全年 exact 仿真 | 1 | `rl_exact_schedule_results.csv` |
| `M4` | PID vs RL 总对比 | 基于 `M2 + M3` 汇总 | 1 | 胜率、利润差、能耗差、散点图 |
| `M5` | 上下层协同矩阵 | `default/optimized schedule × PID/RL` | 1 | synergy matrix 主图 |
| `M6` | 机制解释轨迹 | 默认排程 + 优化排程的 PID/RL 代表轨迹 | 2 schedules | 分项能耗与动作机制解释 |

## 4. 推荐执行顺序

推荐严格按下面顺序执行：

1. `P0` 先做 2 到 5 epoch 烟雾测试
2. `T1` 跑 3 个 500 epoch 主 seed
3. `T2` 用统一评估协议选代表性 run 与 checkpoint
4. `M1` 用代表性 run 做默认排程系统验证
5. `M2` 跑 exact PID baseline
6. `M3` 跑 exact RL baseline
7. `M4` 生成 PID vs RL 主比较结果
8. `M5` 生成 synergy matrix
9. `M6` 只对代表性排程保存详细轨迹，做机制解释

## 5. 统一命名规范

建议训练实验统一命名成：

- `journal_main_respid_sac_e500_s42_v1`
- `journal_main_respid_sac_e500_s52_v1`
- `journal_main_respid_sac_e500_s62_v1`

这样后续：

- `analyze_rl_runs.py`
- `compare_rl_checkpoints.py`
- exact RL baseline

都更好串联。

## 6. 推荐启动命令

以下命令默认在仓库根目录执行：

```powershell
cd C:\Users\29341\Desktop\code_0420\lettuce_control_drl_new_0422
```

Python 环境默认：

```powershell
$PY = "C:\Users\29341\.conda\envs\control_env\python.exe"
```

### 6.1 P0 训练烟雾测试

```powershell
& $PY experiments/train_pfal_contextual_128_500.py `
  --experiment journal_smoke_respid_sac_e005_s42_v1 `
  --seed 42 `
  --epoch 5 `
  --device cuda `
  --no_wandb
```

### 6.2 T1 主训练复现

```powershell
& $PY experiments/train_pfal_contextual_128_500.py `
  --experiment journal_main_respid_sac_e500_s42_v1 `
  --seed 42 `
  --device cuda `
  --no_wandb

& $PY experiments/train_pfal_contextual_128_500.py `
  --experiment journal_main_respid_sac_e500_s52_v1 `
  --seed 52 `
  --device cuda `
  --no_wandb

& $PY experiments/train_pfal_contextual_128_500.py `
  --experiment journal_main_respid_sac_e500_s62_v1 `
  --seed 62 `
  --device cuda `
  --no_wandb
```

### 6.3 T2 多 seed 汇总分析

```powershell
& $PY experiments/analyze_rl_runs.py `
  --experiments `
    journal_main_respid_sac_e500_s42_v1 `
    journal_main_respid_sac_e500_s52_v1 `
    journal_main_respid_sac_e500_s62_v1 `
  --out_dir results/journal_main/seed_analysis
```

### 6.4 T2 每个 run 的 checkpoint 比较

```powershell
& $PY experiments/compare_rl_checkpoints.py `
  --load journal_main_respid_sac_e500_s42_v1 `
  --device cuda `
  --checkpoints best selected final auto `
  --n_eval_schedules 32 `
  --n_eval_episodes_per_schedule 5 `
  --eval_selection reference_stratified `
  --out_dir results/journal_main/checkpoint_compare_s42

& $PY experiments/compare_rl_checkpoints.py `
  --load journal_main_respid_sac_e500_s52_v1 `
  --device cuda `
  --checkpoints best selected final auto `
  --n_eval_schedules 32 `
  --n_eval_episodes_per_schedule 5 `
  --eval_selection reference_stratified `
  --out_dir results/journal_main/checkpoint_compare_s52

& $PY experiments/compare_rl_checkpoints.py `
  --load journal_main_respid_sac_e500_s62_v1 `
  --device cuda `
  --checkpoints best selected final auto `
  --n_eval_schedules 32 `
  --n_eval_episodes_per_schedule 5 `
  --eval_selection reference_stratified `
  --out_dir results/journal_main/checkpoint_compare_s62
```

### 6.5 M1 系统验证

先假定你最终选出的代表性 run 是：

```powershell
$REP_RUN = "journal_main_respid_sac_e500_s52_v1"
```

然后执行：

```powershell
& $PY experiments/1_mainline_system_validation.py `
  --controllers pid rl `
  --load $REP_RUN `
  --device cuda `
  --schedule "t1=14,t2=14,N1=20,rho2=36" `
  --out-dir results/journal_main/exp01_system_validation
```

### 6.6 M2 精确 PID baseline

推荐 `8` 个 shard：

```powershell
& $PY experiments/2_mainline_exact_pid_baseline.py `
  --num-shards 8 `
  --shard-id 0 `
  --out-dir results/journal_main/exp02_exact_pid_baseline
```

将 `--shard-id` 从 `0` 跑到 `7` 后，再执行合并：

```powershell
& $PY experiments/2_mainline_exact_pid_baseline.py `
  --num-shards 8 `
  --merge-shards-only `
  --out-dir results/journal_main/exp02_exact_pid_baseline
```

### 6.7 M3 精确 RL baseline

```powershell
& $PY experiments/3_mainline_exact_rl_baseline.py `
  --load $REP_RUN `
  --load-checkpoint selected `
  --device cuda `
  --num-shards 8 `
  --shard-id 0 `
  --out-dir results/journal_main/exp03_exact_rl_baseline
```

同样将 `--shard-id` 从 `0` 跑到 `7`，再执行合并：

```powershell
& $PY experiments/3_mainline_exact_rl_baseline.py `
  --load $REP_RUN `
  --load-checkpoint selected `
  --device cuda `
  --num-shards 8 `
  --merge-shards-only `
  --out-dir results/journal_main/exp03_exact_rl_baseline
```

### 6.8 M4 PID vs RL 主比较

```powershell
& $PY experiments/4_mainline_pid_rl_comparison.py `
  --pid-csv results/journal_main/exp02_exact_pid_baseline/pid_exact_schedule_results.csv `
  --rl-csv results/journal_main/exp03_exact_rl_baseline/rl_exact_schedule_results.csv `
  --out-dir results/journal_main/exp04_pid_rl_comparison
```

### 6.9 M5 synergy matrix

```powershell
& $PY experiments/5_mainline_synergy_matrix.py `
  --pid-csv results/journal_main/exp02_exact_pid_baseline/pid_exact_schedule_results.csv `
  --rl-csv results/journal_main/exp03_exact_rl_baseline/rl_exact_schedule_results.csv `
  --out-dir results/journal_main/exp05_synergy_matrix
```

### 6.10 M6 机制解释轨迹

默认排程：

```powershell
$DEFAULT_KEY = "t1=14|t2=14|N1=20|rho2=36"
```

优化排程建议从下面文件读取：

- `results/journal_main/exp05_synergy_matrix/synergy_matrix_summary.json`

然后分别对默认排程和优化排程保存 PID / RL 详细轨迹。

PID 默认排程：

```powershell
& $PY experiments/2_mainline_exact_pid_baseline.py `
  --schedule-key $DEFAULT_KEY `
  --save-detailed-traces `
  --save-batch-trajectories `
  --out-dir results/journal_main/mechanism/pid_default
```

RL 默认排程：

```powershell
& $PY experiments/3_mainline_exact_rl_baseline.py `
  --load $REP_RUN `
  --load-checkpoint selected `
  --device cuda `
  --schedule-key $DEFAULT_KEY `
  --save-detailed-traces `
  --save-batch-trajectories `
  --out-dir results/journal_main/mechanism/rl_default
```

优化排程：

```powershell
$OPT_KEY = (Get-Content results/journal_main/exp05_synergy_matrix/synergy_matrix_summary.json -Raw | ConvertFrom-Json).optimized_schedule_key
```

PID 优化排程：

```powershell
& $PY experiments/2_mainline_exact_pid_baseline.py `
  --schedule-key $OPT_KEY `
  --save-detailed-traces `
  --save-batch-trajectories `
  --out-dir results/journal_main/mechanism/pid_optimized
```

RL 优化排程：

```powershell
& $PY experiments/3_mainline_exact_rl_baseline.py `
  --load $REP_RUN `
  --load-checkpoint selected `
  --device cuda `
  --schedule-key $OPT_KEY `
  --save-detailed-traces `
  --save-batch-trajectories `
  --out-dir results/journal_main/mechanism/rl_optimized
```

### 6.11 机制增益分析汇总

默认排程的 detailed trace 文件名会按下面这种 slug 生成：

- `t1-14__t2-14__N1-20__rho2-36.csv`

因此可以直接做一版默认排程机制分析：

```powershell
& $PY experiments/analyze_rl_gain_mechanisms.py `
  --pid-csv results/journal_main/exp02_exact_pid_baseline/pid_exact_schedule_results.csv `
  --rl-csv results/journal_main/exp03_exact_rl_baseline/rl_exact_schedule_results.csv `
  --pid-trace-csv results/journal_main/mechanism/pid_default/detailed_traces/t1-14__t2-14__N1-20__rho2-36.csv `
  --rl-trace-csv results/journal_main/mechanism/rl_default/detailed_traces/t1-14__t2-14__N1-20__rho2-36.csv `
  --out-dir results/journal_main/exp06_gain_mechanism_default
```

如果要分析优化排程，只需要把 trace 路径替换成对应 `OPT_KEY` 生成出来的 slug 文件。

## 7. 论文主文推荐采用的最小配置

如果资源有限，但还想维持“期刊主实验”质量底线，建议至少做到：

- RL 训练：`3 seeds × 500 epochs`
- seed 选择：`32 schedules × 5 episodes`
- exact PID：全 `368` schedule
- exact RL：代表性 policy 全 `368` schedule
- 系统验证：默认排程代表轨迹
- 主对比：`M4 + M5`

## 8. 如果资源足够，建议再增强的部分

- 将主训练从 `3 seeds` 增到 `5 seeds`
- 将 checkpoint 比较评估从 `32 schedules` 增到 `48 schedules`
- 将 `M6` 机制解释从“默认排程 + 最优排程”扩展到“默认 + 最优 + 一个高密度极端排程”
- 在补充材料中增加 `300 epoch vs 500 epoch` 收敛比较，证明主实验超参数不是随意挑的

## 9. 最终建议

对当前这个项目，最稳妥的高水平期刊主线不是“所有东西都跑到最大”，而是：

- 用 `3~5` 个 seed 证明 RL 训练结论可复现
- 用一个经统一规则选出的代表性 policy 做 exact 全排程对比
- 用 `M4 + M5 + M6` 形成“性能提升 + 协同来源 + 机制解释”三段式证据链

这会比“只汇报一个最好 run”更容易说服审稿人。
