# GAC-MAC Project Development Plan
#
# NOTE: This file intentionally starts with ASCII-only lines to avoid a known
# Windows sandbox Unicode truncation issue in the patching tool. Chinese content
# begins below.

---

## 0. 目标与验收

### 0.1 总目标
- 新开一个独立目录实现（不依赖现有 `3.0/`，也不要求 Part 1/5 等内容对齐）。
- 搭建 LAWN 分布式 TDMA 时隙分配的仿真环境：动态拓扑、物理层干扰与 SINR、业务队列与时延。
- 实现 GAC-MAC：GATv2（空间）+ GRU（时间）+ MAPPO（CTDE）训练与评测。
- 提供基线策略（Random / Greedy / CSMA）与可视化（曲线 + 拓扑着色快照）。

### 0.2 验收标准
1) **跑通**：训练脚本可启动并持续运行，日志/曲线可产出，无崩溃。  
2) **趋势正确**：训练后 GAC-MAC **碰撞率明显低于 Random**，吞吐量 **高于 CSMA**。  
3) **可视化**：可生成“拓扑着色图”（边=干扰边；节点颜色=选定时隙；静默=灰色）。  

---

## 1. 关键需求约束（实现时必须遵守）

### 1.1 信道模型（简化但严谨）
- **单位统一：** 路损公式中 `d` 用米（m），`f_c` 用 GHz 直接代入：
  - `PL_LoS = 28.0 + 22*log10(d) + 20*log10(f_c_GHz)`
  - `PL_NLoS = -17.5 + (46 - 7*log10(z_avg)) * log10(d) + 20*log10(f_c_GHz)`（或按需求中的第二套不同公式实现）
- **必须落实：**
  - LoS/NLoS 概率（与距离 + 高度相关）；
  - 两套不同路损公式；
  - 每条链路的随机 LoS/NLoS 判定逻辑。
- **Shadowing：** LoS `σ=4 dB`，NLoS `σ=6 dB`（固定）。
- **快衰落：** Nakagami-m；LoS `m=3`，NLoS `m=1`（瑞利）。
- `f_c` 需要转 Hz 的地方（如噪声功率计算等）用 `f_c_Hz = f_c_GHz * 1e9`。

### 1.2 通信与干扰拓扑
- **链路：** UAV-UAV 成对链路；UAV `i` 的接收端为 `(i+1) % N`（逻辑环）。
- **干扰图：** **有向图**，不对称化（允许 `A_ij != A_ji`）。
- **构边规则：** 若 `P_rx(j->i) > cs_threshold` 则添加有向边 `j -> i`。

### 1.3 业务/队列/时延
- **到达过程：** 泊松到达（参数 `lambda_arrival`，例如 `0.2 packet/slot`）。
- **成功判定：** 仅看 SINR 阈值；成功则理想 ACK（不模拟 ACK 包过程）。

### 1.4 能量维度
- 特征中保留 `E_i`，但第一版固定为常量（如 1.0），并保留后续扩展接口。

### 1.5 奖励与协同项
- `r_coop_i = lambda_coop * mean_{j in N_i}(r_perf_j)`，严格按邻居集合计算。
- 若 `|N_i|=0`：`r_coop_i = 0`（或等于 `r_perf_i`，实现前需固定一种并保持一致）。
- `lambda_coop` 可配置（默认 0.5）。

### 1.6 MAPPO + GRU
- **Critic：** 输出全局 `V`，再广播给每个 agent。
- **截断 BPTT：** 每固定步长（建议 40~64）对 GRU hidden state `detach`。
- **Buffer 对齐：** Buffer 语义按 `[Time, N, ...]` 组织；喂给网络/Batch 前再按一致顺序 flatten 为 `[Time*N, ...]`。

### 1.7 基线与对比
- 仅实现：Random / Greedy Coloring / CSMA/CA。

---

## 2. 技术栈与依赖（建议）
- Python 3.10+
- `numpy`, `torch`, `torch_geometric`, `matplotlib`, `networkx`, `tqdm`
- （可选）`tensorboard` 用于训练日志
- 环境接口：`gymnasium`（推荐）或自定义最小 reset/step 接口

---

## 3. 目录结构（新建目录）

在仓库根目录新建：`gac_mac/`

建议结构：
- `gac_mac/`
  - `README.md`：运行与复现说明
  - `configs/`：默认超参（yaml 或 python dataclass）
  - `env/`：`channel.py` / `mobility.py` / `traffic.py` / `graph_builder.py` / `lawn_env.py`
  - `models/`：`encoder.py` / `agent.py`
  - `algo/`：`buffer.py` / `mappo.py`
  - `baselines/`：`random_agent.py` / `greedy_coloring.py` / `csma.py`
  - `viz/`：`plots.py` / `topology_snapshot.py`
  - `scripts/`：`train.py` / `evaluate.py`
  - `results/`：输出（曲线、快照、模型权重、日志）

---

## 4. 里程碑与任务拆解（推荐顺序）

### M1：工程骨架 + 最小可运行闭环
- 建立 `gac_mac/` 目录与脚本入口（`scripts/train.py`）。
- 实现配置管理（Args/dataclass/yaml）、随机种子、结果输出目录。
- 先用“极简 reward + 简化 env”跑通训练循环，验证数据流与维度。

### M2：环境侧（物理层 + 业务 + 观测）
- `ChannelModel`：LoS/NLoS 概率 + 双路损 + shadowing + Nakagami；输出链路增益矩阵与可用于 SINR 的量。
- `MobilityModel`：Gauss-Markov + 边界裁剪/回弹，保证不出界。
- `TrafficModel`：泊松到达、队列更新、包级时延统计（用于 Access Delay）。
- `GraphBuilder`：由 carrier sensing 生成有向 `edge_index`；构建节点特征 `x`（位置/队列/能量常量/历史动作/历史状态/度数）。
- `LAWNEnv`：`reset/step`；基于同 slot 干扰计算 SINR；更新 success/collision；计算 reward 与 metrics（sum_rate、collision_rate、avg_delay、fairness 等）。

### M3：模型与训练（GATv2+GRU + MAPPO）
- `ST_GNN_Encoder`：2 层 `GATv2Conv` + `LayerNorm` + `ELU` + residual + `GRUCell`。
- `Agent`：Actor 输出 `K+1` 分布；Critic 用全局 pooling 得 `V_global` 并广播。
- `GraphReplayBuffer`：存 `Data`、actions、log_probs、rewards、dones、h_states；实现 GAE 与 mini-batch 生成（PyG Batch）。
- `MAPPOTrainer`：PPO clip、value loss、entropy、grad clip；实现截断 BPTT 的 hidden-state 管理策略。

### M4：基线与评测脚本
- Random / Greedy Coloring / CSMA/CA 策略实现。
- `scripts/evaluate.py`：统一环境与种子跑对比，输出汇总与曲线数据。

### M5：可视化与结果产出
- 收敛曲线：reward / sum_rate / collision（滑动平均）。
- 拓扑着色快照：节点位置 + 有向边（可视化时可画成无向线段）+ 按时隙上色 + 冲突提示。
- 输出 `results/result_comparison.png` 与若干快照 `topology_ep_*.png`。

---

## 5. 风险点与规避
- **PyG Batch + RNN 对齐**最易出错：先固定 time-major 展平顺序（t0 的 N 个节点、t1 的 N 个节点…），并做 1-2 个最小断言测试。
- **cs_threshold** 过高/过低导致图无边/全连接：训练前打印 `avg_degree`，把平均度控制在合理范围（如 4-8）。
- **奖励尺度**导致“全静默”：默认奖励保持在小范围（如成功 +1、碰撞 -2、空闲按队列区分），并监控静默比例。

---

## 6. 最终输出清单
- 训练：`python -m gac_mac.scripts.train`（或 `python gac_mac/scripts/train.py`）
- 评测：`python -m gac_mac.scripts.evaluate`
- 结果：训练曲线、baseline 对比图、拓扑着色快照、（可选）tensorboard 日志与模型权重

