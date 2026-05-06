# Experiment Design And Figures

## 1. 研究主问题

当前最科学的主问题应该写成：

`container PFAL lettuce production scheduling and closed-loop climate control under uncertainty`

进一步可写为：

`two-time-scale hierarchical co-optimization of production scheduling and closed-loop climate control for a multi-batch two-zone container PFAL`

## 2. 实验设计主线

### E0. 参考目录与排程语义确认

目的：

- 固化当前离散排程空间
- 固化入箱初始苗质量、参考可行性标签、默认排程

输出：

- `feasible_solutions.csv`
- 参考可行性分类统计图

### E1. Exact PID feasible-set baseline

目的：

- 先回答“排程本身能带来多大差距”
- 给出公平的上层 benchmark

核心输出：

- 全 1840 个可行解的 profit / harvest / cost / energy
- default schedule 与 best-valid schedule 的差距

建议图：

- Fig. 1: PID exact paper summary
- Fig. 2: PID design-space heatmaps
- Fig. 3: Top-10 valid schedules

### E2. Approximate PID schedule search

目的：

- 解决 exact baseline 太慢的问题
- 建立接近最优、但更便宜的 schedule screening 方法

需要回答：

- 近似搜索离 exact best 差多少
- 节省了多少时间

建议图：

- Fig. 4: approximate-vs-exact ranking consistency
- Fig. 5: computation budget vs best-found profit

### E3. RL design ablation

必须做的对比：

- `contextual_sac` vs `residual_pid_sac`
- `direct` vs `curriculum`

建议结论口径：

- 先证明哪种 RL 架构更稳定
- 再讨论是否优于 PID，而不是反过来

建议图：

- Fig. 6: reward vs constraint tradeoff
- Fig. 7: risk profile
- Fig. 8: training time breakdown

### E4. Exact RL feasible-set baseline

目的：

- 训练完成后，不只看少量 sampled schedules
- 要在整个可行集上逐排程评估 RL

这是非常关键的一步，因为它才能真正回答：

- RL 的最好排程是什么
- RL 在全排程空间上到底赢多少、输多少

建议图：

- Fig. 9: exact RL paper summary
- Fig. 10: exact RL design-space heatmaps

### E5. Exact PID vs Exact RL fair comparison

这是高水平论文的关键公平性实验。

比较原则：

- 同一可行排程集合
- 同一天气窗口
- 同一仿真时长
- 同一 `dt`
- 同一经济口径

建议图：

- Fig. 11: PID vs RL profit scatter
- Fig. 12: RL-PID profit-gap heatmap
- Fig. 13: gap distribution histograms

建议表：

- Table 1: RL win rate over all schedules
- Table 2: RL win rate over target-feasible schedules
- Table 3: top RL win / top RL loss schedules

### E6. 轨迹级机制解释

目的：

- 审稿人不会满足于“分数更高”
- 需要解释 RL 为什么赢/输

建议 case：

- 默认排程
- PID 最优排程
- RL 最优排程
- 一个 RL 明显输给 PID 的排程

建议图：

- Fig. 14: 温度/RH/CO2/VPD 时序
- Fig. 15: 光照/HVAC/CO2/除湿动作时序
- Fig. 16: biomass / harvest trajectory
- Fig. 17: constraint activation / safety override timeline

### E7. 鲁棒性实验

建议至少做一个：

- 跨天气窗口
- 跨随机种子
- 跨入箱苗质量扰动
- 跨参考可行性子空间

建议图：

- Fig. 18: robustness boxplots

## 3. 审稿人最关心的点

### Q1. 为什么不是直接 joint optimization？

回答：

- 调度是慢时间尺度、离散变量
- 控制是快时间尺度、连续变量
- 全耦合会成为高代价 hybrid stochastic dynamic optimization

### Q2. 为什么 RL 要和 PID 比？

回答：

- PID 是工程上可部署、可信的基线
- RL 必须在相同 schedule 上公平比较

### Q3. 为什么要做 exact feasible-set evaluation？

回答：

- 否则无法区分“排程差”与“控制差”
- 也无法判断 RL 训练应该覆盖哪些 schedule 区域

### Q4. 为什么 residual RL 更合理？

回答：

- 更容易保证基础稳定性
- 对工业控制更有可部署性
- 更适合做“超过 PID 的增益控制器”而非“完全替代”

## 4. 当前优先级建议

建议执行顺序：

1. 跑完 exact PID baseline
2. 训练主线 `residual_pid_sac`
3. 跑 exact RL baseline
4. 跑 PID vs RL exact comparison
5. 选代表排程做轨迹图

如果资源紧张：

1. approximate PID search
2. 主线 RL 训练
3. RL exact baseline on top-k schedules first
4. full exact RL baseline later
