# RL 训练执行手册（2026-04-18）

## 1. 适用范围

本手册面向当前 `Applied Energy` 主线论文所需的 RL 训练与评估工作，默认对应如下系统设定：

- 仓库：`lettuce_control_drl_new`
- 问题主线：`two-time-scale hierarchical co-optimization of production scheduling and closed-loop climate control under uncertainty`
- 主线控制器：`Contextual Residual SAC over PID`
- 公平对比对象：`PID baseline`
- 控制步长：`dt = 600 s`
- 换气设定：仅 `0.2 ACH` 被动漏风，无主动通风
- 默认人工排程：`x = {14, 14, 20, 36, 16}`

本手册的目标不是罗列所有源码细节，而是给出一套你可以直接复制执行、同时又符合论文主线口径的训练命令与参数解释。

## 2. 当前推荐实验矩阵

| 实验 | 目的 | 推荐程度 | 论文角色 |
| --- | --- | --- | --- |
| `residual_pid_sac + direct` | 主线正式训练 | 最高 | 主结果候选 |
| `residual_pid_sac + curriculum` | 与 direct 公平比较 | 最高 | 训练策略消融 |
| `contextual_sac + direct` | 检验是否必须保留 PID 骨架 | 高 | 结构消融 |
| `eval_only` | 对已训练模型做统一评估 | 必需 | 训练后泛化检查 |

如果你现在只打算先跑一个完整 RL，我建议先跑：

`residual_pid_sac + direct + default + mainline_long_horizon + 300 epoch`

原因是这组设定最贴近当前论文主线，也最适合作为之后 exact RL baseline 的候选模型。

## 3. 四套可直接复制的命令

以下命令默认在 `lettuce_control_drl_new` 仓库中执行，从而避免和旧仓库里正在运行的 exact PID baseline 结果目录混淆。

### 3.1 主线正式训练：Residual SAC over PID，直接训练

```powershell
cd c:\Users\29341\Desktop\code_0413\lettuce_control_drl_new
C:\Users\29341\.conda\envs\control_env\python.exe experiments/train_pfal_contextual.py --device cuda --no_wandb --controller_design residual_pid_sac --curriculum_profile off --runtime_profile default --horizon_profile mainline_long_horizon --epoch 300 --experiment rl_mainline_residual_direct_20260418_v1
```

适用目的：

- 先拿到一条最符合论文主线的完整 RL
- 作为后续 exact RL feasible-set baseline 的候选模型
- 作为 PID 公平比较的主方法候选

### 3.2 主线对比训练：Residual SAC over PID，课程学习

```powershell
cd c:\Users\29341\Desktop\code_0413\lettuce_control_drl_new
C:\Users\29341\.conda\envs\control_env\python.exe experiments/train_pfal_contextual.py --device cuda --no_wandb --controller_design residual_pid_sac --curriculum_profile target_to_full --runtime_profile default --horizon_profile mainline_long_horizon --epoch 300 --experiment rl_mainline_residual_curriculum_20260418_v1
```

适用目的：

- 与 direct 训练做公平比较
- 判断课程学习是否真的带来更好的稳定性或最终性能
- 为论文 `E2 RL design ablation` 提供证据

### 3.3 结构消融训练：Plain Contextual SAC

```powershell
cd c:\Users\29341\Desktop\code_0413\lettuce_control_drl_new
C:\Users\29341\.conda\envs\control_env\python.exe experiments/train_pfal_contextual.py --device cuda --no_wandb --controller_design contextual_sac --curriculum_profile off --runtime_profile default --horizon_profile mainline_long_horizon --epoch 300 --experiment rl_ablation_plain_contextual_direct_20260418_v1
```

适用目的：

- 检验 `plain contextual SAC` 与 `residual_pid_sac` 的差异
- 回答“保留 PID 骨架是否必要”
- 不建议把它作为最终主线方案

### 3.4 训练后统一评估：Eval-only

```powershell
cd c:\Users\29341\Desktop\code_0413\lettuce_control_drl_new
C:\Users\29341\.conda\envs\control_env\python.exe experiments/train_pfal_contextual.py --eval_only --device cuda --no_wandb --load rl_mainline_residual_direct_20260418_v1 --n_eval_schedules 20 --n_eval_episodes_per_schedule 1 --eval_selection reference_stratified --eval_seed 42
```

适用目的：

- 对某个已训练实验做统一的排程泛化评估
- 快速检查模型是否值得进入 exact RL baseline
- 先做小规模结构性检查，再决定是否投入年尺度 exact evaluation

如果你要评估课程学习版，只需要把 `--load` 改成：

`rl_mainline_residual_curriculum_20260418_v1`

## 4. 这四套命令中每个参数的值、含义与目的

### 4.1 `--device`

可选值：

- `cpu`
- `cuda`

含义：

- 指定训练/评估所使用的计算设备

目的：

- `cuda` 用于正式训练，通常显著更快
- `cpu` 只建议在调试、冒烟测试或无 GPU 时使用

当前建议：

- 正式训练和 eval-only 都优先用 `cuda`

### 4.2 `--no_wandb`

可选值：

- 写上该参数：关闭 WandB
- 不写：按配置启用 WandB

含义：

- 是否关闭 WandB 在线日志

目的：

- 降低外部依赖
- 避免联网或账号配置问题
- 更适合本地稳定执行

当前建议：

- 当前阶段统一使用 `--no_wandb`

### 4.3 `--controller_design`

可选值：

- `contextual_sac`
- `residual_pid_sac`
- `gated_residual_pid_sac`

含义：

- 指定 RL 控制器结构

目的：

- `contextual_sac`：纯 RL，直接输出绝对动作
- `residual_pid_sac`：在 PID 基础上输出残差动作，是当前论文主线
- `gated_residual_pid_sac`：在残差基础上加入更保守的门控机制

当前建议：

- 主线正式训练：`residual_pid_sac`
- 结构消融：`contextual_sac`
- `gated_residual_pid_sac` 暂不作为第一批完整训练主线，只在后续确认需要更保守残差时再补

### 4.4 `--curriculum_profile`

可选值：

- `config`
- `off`
- `legacy_narrow_to_full`
- `target_to_full`

含义：

- 指定训练时的课程学习策略

目的：

- `off`：直接在完整目标分布上训练
- `legacy_narrow_to_full`：前 `40%` epoch 在窄排程窗口训练，后续切到全空间
- `target_to_full`：前 `40%` epoch 在 `full_target_feasible` 子集训练，后续切到全空间
- `config`：跟随 `configs/rl_params.yaml`

当前建议：

- 主线基准：`off`
- 课程学习对照：`target_to_full`
- 不建议把 `legacy_narrow_to_full` 当成第一优先对照，因为它更像旧版经验式课程，而不是当前更科学的 target-feasible warm start

### 4.5 `--runtime_profile`

可选值：

- `default`
- `pilot_fast`
- `pilot_ultrafast`

含义：

- 指定训练预算和 episode 长度模板

目的：

- `default`：正式训练，不主动压缩训练预算
- `pilot_fast`：较快筛查，自动把训练改成更短的 4 天 episode 和较小预算
- `pilot_ultrafast`：超快冒烟测试，自动把训练改成 2 天 episode、更少并行环境和更小更新强度

当前建议：

- 论文主线完整训练：`default`
- 快速试跑或排查脚本是否能跑：`pilot_fast` 或 `pilot_ultrafast`

### 4.6 `--horizon_profile`

可选值：

- `config`
- `fast_t2max`
- `mainline_long_horizon`

含义：

- 指定 train/test/eval 的 episode horizon 设计

目的：

- `fast_t2max`：以 `t2_max` 为主，对当前空间相当于 `18 d = 2592 steps`
- `mainline_long_horizon`：训练时混合 `max_t2` 与 `max_total_cycle`，更接近完整生产周期压力
- `config`：跟随 `rl_params.yaml`

当前建议：

- 主线完整训练：`mainline_long_horizon`
- 快速筛查：`fast_t2max`

补充判断：

- 如果你只追求更快训练，`fast_t2max` 会更快
- 如果你希望 RL 在训练时见到部分更长的完整周期压力，`mainline_long_horizon` 更科学
- 对当前论文主线，我更建议优先保留 `mainline_long_horizon`

### 4.7 `--epoch`

可选值：

- 任意正整数，例如 `8`、`12`、`50`、`100`、`300`、`500`

含义：

- 总训练轮数

目的：

- 决定训练预算大小

当前建议：

- pilot：`8` 或 `12`
- 主线完整训练：`300`
- 若后续发现 300 epoch 仍未收敛，再考虑 `500`

### 4.8 `--experiment`

可选值：

- 任意不含路径歧义的字符串

含义：

- 实验名，同时决定日志目录名

目的：

- 区分不同训练任务
- 便于后续 `--load`

当前建议：

- 在名字中明确写出结构、训练策略、日期和版本号
- 例如：`rl_mainline_residual_direct_20260418_v1`

训练结果默认存放位置：

`log/PFAL-contextual-SAC/sac_contextual/<experiment>/`

### 4.9 `--eval_only`

可选值：

- 写上该参数：仅评估
- 不写：训练或继续训练

含义：

- 是否只运行评估流程

目的：

- 对已有模型做快速泛化检查
- 不再进行训练更新

当前建议：

- 在完整训练结束后，用它先筛查模型是否值得进入 exact RL baseline

### 4.10 `--load`

可选值：

- 实验名，例如 `rl_mainline_residual_direct_20260418_v1`
- 实验目录路径

含义：

- 加载已有实验

目的：

- 用于 `eval_only`
- 用于继续训练
- 用于读取保存的 `run_config.json` 和策略权重

当前建议：

- 如果你在本仓库标准日志目录下训练，优先直接写实验名即可

### 4.11 `--n_eval_schedules`

可选值：

- 任意正整数，默认脚本值为 `10`

含义：

- 评估多少个排程

目的：

- 快速做排程泛化筛查

当前建议：

- 快速检查：`10`
- 论文前期筛查：`20`
- 更严谨的中期筛查：可以继续提高

### 4.12 `--n_eval_episodes_per_schedule`

可选值：

- 任意正整数

含义：

- 每个排程评估多少次 episode

目的：

- 用于控制评估稳定性和耗时

当前建议：

- 第一轮筛查：`1`
- 如果后续引入更多随机性，可考虑增加

### 4.13 `--eval_selection`

可选值：

- `coverage`
- `random`
- `reference_stratified`

含义：

- 选择哪些排程进入评估

目的：

- `coverage`：尽量覆盖整个排程空间
- `random`：纯随机
- `reference_stratified`：优先覆盖 `target_feasible / marginal / infeasible` 三类参考排程，再做 coverage 补充

当前建议：

- 论文主线评估优先用 `reference_stratified`

### 4.14 `--eval_seed`

可选值：

- 任意整数

含义：

- 评估采样随机种子

目的：

- 保证评估可复现

当前建议：

- 固定为 `42`

## 5. 高频附加参数字典

以下参数没有全部出现在上面四套主命令里，但你后面很可能会调。

### 5.1 `--nstep`

可选值：

- 任意正整数

含义：

- 每个 epoch 的单环境参考环境步数

目的：

- 控制每个 epoch 的采样长度

注意：

- 在当前项目中，如果 `auto_nstep=true`，你手写的 `--nstep` 会被自动重算覆盖

### 5.2 `--auto_nstep` 与 `--no_auto_nstep`

可选值：

- `--auto_nstep`
- `--no_auto_nstep`

含义：

- 是否根据当前 horizon 自动计算 `nstep`

目的：

- 保证 “每个 epoch 大致覆盖一个合理 episode horizon”

当前建议：

- 主线训练保持自动，即不要主动关闭

### 5.3 `--nstep_factor`

可选值：

- 任意正浮点，常用如 `0.5`、`0.75`、`1.0`

含义：

- 自动 `nstep` 的缩放系数

目的：

- 通过缩短或保持 `nstep` 来调节训练速度

当前建议：

- 主线训练：`1.0`
- 只为提速的 pilot：`0.5`

### 5.4 `--batch_size`

可选值：

- 任意正整数，常见如 `256`、`512`

含义：

- SAC 更新的 mini-batch 大小

目的：

- 平衡显存占用、更新稳定性和速度

当前建议：

- 主线：`512`
- 显存紧张或 pilot：`256`

### 5.5 `--train_num`

可选值：

- 任意正整数

含义：

- 并行训练环境数量

目的：

- 增加采样并行度

当前建议：

- 当前默认是 `4`
- 并不是越大越快，因为环境本身较重

### 5.6 `--test_num`

可选值：

- 任意正整数

含义：

- 并行测试环境数量

目的：

- 控制测试阶段的开销

当前建议：

- 当前默认是 `1`

### 5.7 `--hidden`

可选值：

- 两个整数，例如 `--hidden 128 128`
- 例如也可以写 `--hidden 256 256`

含义：

- Actor/Critic 隐层宽度

目的：

- 控制网络容量

当前建议：

- 先保留 `128 128`
- 不建议在第一轮主线训练里同时改结构宽度和训练策略

### 5.8 `--gamma`

可选值：

- `0~1` 之间的浮点数

含义：

- 奖励折扣因子

目的：

- 控制长期收益权重

当前建议：

- 保持当前默认 `0.99`

### 5.9 `--actor_lr` 与 `--critic_lr`

可选值：

- 任意正浮点

含义：

- Actor 与 Critic 的学习率

目的：

- 控制参数更新步长

当前建议：

- 当前默认都是 `3e-4`

### 5.10 `--residual_action_scale`

可选值：

- 1 个数，例如 `0.5`
- 或 5 个数，例如 `0.50 0.50 0.35 0.35 0.35`

含义：

- 残差动作幅度缩放

目的：

- 控制 RL 对基线 PID 动作的最大扰动强度

5 个位置分别对应：

- `I1`
- `I2`
- `Q_HVAC`
- `u_CO2`
- `m_dehum`

当前建议：

- 保持当前默认 `[0.50, 0.50, 0.35, 0.35, 0.35]`

### 5.11 `--context_phase`

可选值：

- `full`
- `narrow`
- `fixed`
- `full_min_feasible`
- `full_target_feasible`
- `full_infeasible`

含义：

- 指定训练或评估时从哪个排程子空间采样

目的：

- 做课程学习
- 做局部问题训练
- 做固定排程分析

当前建议：

- 主线训练不要手动改，直接交给 `curriculum_profile`

### 5.12 `--fixed_schedule`

可选值：

- 五个数：`t1 t2 N1 rho2 PP`

含义：

- 固定在某个排程上训练或评估

目的：

- 做单排程 case study
- 做局部问题调试

当前人工默认排程：

`--fixed_schedule 14 14 20 36 16`

变量范围：

- `t1: 10~18`
- `t2: 10~18`
- `N1: 8~20`
- `rho2: 20~52`
- `PP: 14~18`

注意：

- 即使单变量落在范围内，也仍需满足派生可行性约束

### 5.13 `--narrow_bounds`

可选值：

- 十个数：
  - `T1_MIN T1_MAX T2_MIN T2_MAX N1_MIN N1_MAX RHO2_MIN RHO2_MAX PP_MIN PP_MAX`

含义：

- 自定义窄排程窗口

目的：

- 用于局部训练或局部分析

### 5.14 `--episode_length_mode`

可选值：

- `schedule_t2`
- `max_t2`
- `fixed_days`
- `total_cycle`
- `max_total_cycle`
- `mixed`
- `mixed_horizon`
- `mixed_episode`
- `curriculum`

含义：

- 控制 episode 长度的模式

目的：

- 控制训练看到的时间跨度

当前论文主线判断：

- 如果只想更快：偏向 `max_t2`
- 如果想更贴近完整 recipe 压力：偏向 `max_total_cycle` 或 mixed
- 当前主线不建议手动覆盖，直接通过 `--horizon_profile mainline_long_horizon` 控制

## 6. 为什么当前主线建议这样设

### 6.1 为什么主线选 `residual_pid_sac`

- 它最符合“在工业基线 PID 上做结构保留式增强”的论文叙事
- 它更利于和 PID 做公平比较
- 它通常比 plain RL 更稳定，也更容易解释

### 6.2 为什么保留 `contextual_sac`

- 它是很重要的结构消融
- 它可以回答“PID 骨架是否必要”
- 但它不宜作为第一主线候选

### 6.3 为什么同时比较 `direct` 和 `curriculum`

- 这是当前你明确提出的研究问题之一
- 课程学习可能更稳，但不一定最终更优
- 只有在同一主结构、同一 horizon、同一 runtime 下比较，结论才可信

### 6.4 为什么主线更偏向 `mainline_long_horizon`

- 你的真实问题不是只优化 `t2`
- 排程-控制协同本质上与完整阶段组合压力有关
- `mainline_long_horizon` 会让训练过程偶尔看到 `max_total_cycle` 级别的长时域压力，更符合论文主线

## 7. 训练输出在哪里看

默认日志目录：

`log/PFAL-contextual-SAC/sac_contextual/<experiment>/`

常见输出包括：

- `policy.pth`
- `policy_final.pth`
- `run_config.json`
- `generalization_eval.json`
- `generalization_eval_eval_only.json`
- 训练过程日志与 coverage 统计

## 8. 你现在最建议先跑哪一条

如果你要立刻先启动一个完整 RL，我建议先跑：

```powershell
cd c:\Users\29341\Desktop\code_0413\lettuce_control_drl_new
C:\Users\29341\.conda\envs\control_env\python.exe experiments/train_pfal_contextual.py --device cuda --no_wandb --controller_design residual_pid_sac --curriculum_profile off --runtime_profile default --horizon_profile mainline_long_horizon --epoch 300 --experiment rl_mainline_residual_direct_20260418_v1
```

然后第二条再跑：

```powershell
cd c:\Users\29341\Desktop\code_0413\lettuce_control_drl_new
C:\Users\29341\.conda\envs\control_env\python.exe experiments/train_pfal_contextual.py --device cuda --no_wandb --controller_design residual_pid_sac --curriculum_profile target_to_full --runtime_profile default --horizon_profile mainline_long_horizon --epoch 300 --experiment rl_mainline_residual_curriculum_20260418_v1
```

这样你就能先回答一个非常核心的问题：

`在当前主线结构 residual_pid_sac 下，curriculum 是否真的优于 direct？`

## 9. 备注

- 本手册默认优先服务于论文主线，而不是最快训练速度。
- 如果你只想确认脚本能否跑通，可以另开一条 `pilot_fast` 或 `pilot_ultrafast`，但不要把它们的结果直接写进论文主文。
- exact RL feasible-set baseline 与 exact PID-vs-RL paired comparison 应该建立在完整主线训练完成之后。
