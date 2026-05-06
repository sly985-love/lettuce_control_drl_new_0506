# Project Structure

## 1. 当前判断

原始项目目录确实有些混乱，主要体现在：

- 研究文档、代码、日志、图表都混在根目录附近
- `results/` 同时承担“正式结果”和“临时分析缓存”
- `log/` 很大，不适合作为代码副本的一部分
- 一些脚本默认依赖仓库外部的 `results/feasibility/`

因此，创建 `lettuce_control_drl_new` 作为干净工作副本是合理的。

## 2. 当前新副本的原则

这个新副本现在遵循：

- 保留代码、配置、输入数据、研究文档
- 不复制旧的运行产物
- 把可行解目录纳入 `data/feasibility/`
- 让 `log/` 和 `results/` 作为运行时自动生成目录

## 3. 推荐目录语义

### `configs/`

放：

- 作物参数
- 环控参数
- 奖励参数
- RL 参数
- 排程空间定义

### `data/`

放：

- 天气数据
- `feasible_solutions.csv`
- 其他静态输入数据

### `src/`

放：

- 环境
- 作物模型
- 控制器
- RL 核心实现
- 通用工具

### `experiments/`

只放“可直接执行的入口脚本”：

- 仿真
- 训练
- exact baseline
- approximate search
- 结果分析

### `docs/`

放：

- 研究框架说明
- 实验设计
- 论文图像规划
- 项目结构说明

### `log/`

放：

- 训练日志
- checkpoint
- tensorboard
- run_config / summary json

### `results/`

放：

- exact baseline 结果
- 近似搜索结果
- 对比表格
- 论文图表

## 4. 现在不建议做的大动作

暂时不建议：

- 大规模移动现有 `src/` 和 `experiments/` 文件
- 重命名已有主入口脚本
- 让后台正在运行的旧实验切换到新路径

原因：

- 当前最重要的是保证研究主线推进
- 过度重构容易打断路径依赖和已有命令

## 5. 当前已经做的结构性修复

已经完成：

- 新建干净工作副本 `lettuce_control_drl_new`
- 新增 `.gitignore`
- 把 `feasible_solutions.csv` 纳入项目 `data/feasibility/`
- exact / approx / analyze 脚本默认优先读取本地可行解目录
- 新增 exact RL baseline 脚本
- 新增 exact PID vs RL comparison 脚本

## 6. 后续如果还要再整理

下一步更深度的整理建议是：

1. 把历史 `CODEX_*.md` 文档逐步收进 `docs/history/`
2. 把 `experiments/` 细分为：
   - `experiments/training/`
   - `experiments/baselines/`
   - `experiments/analysis/`
3. 把共用绘图函数抽到 `src/utils/plotting.py`
4. 把 exact PID / exact RL 的共用逻辑抽到 `src/utils/exact_baseline.py`

但这些都属于“第二阶段重构”，不应先于主实验推进。
