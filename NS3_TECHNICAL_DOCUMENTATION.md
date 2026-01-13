# GAT\_MAPPO\_MAC（GAC-MAC）当前实现技术文档（用于 ns-3 复现）

更新时间：2026-01-11  
仓库根目录：`C:\Users\admin\Desktop\intelligent_aj\GAT_MAPPO_MAC`  
Python实现目录：`gac_mac/`（standalone，不依赖旧 `3.0/`）  

> 本文目标：把“目前代码里已经实现的所有内容”按**可复现、可移植到 ns-3**的方式做一次“全量、详实”的技术说明（包含：系统模型、环境、动作/观测/奖励、训练算法、baseline、评测与绘图、当前最佳版本与产物、以及 ns-3 映射建议）。
>
> 本仓库的实验过程与中间结论以 `DEVLOG.md` 为准；本文把它整理成一份面向工程落地（ns-3复现）的说明书。

---

## 1. 总览：我们做的是什么

我们实现了一个“低空无中心网络（LAWN）分布式时隙/资源调度”的仿真环境，并在其上训练一个多智能体强化学习（MARL）MAC策略：

- **网络形态**：N 个 UAV（节点），在 3D 空间移动（Gauss-Markov），每帧进行一次 MAC 决策。
- **MAC 资源**：每帧有 `K` 个时隙，支持 `C` 个正交信道（多信道），一个资源定义为 `(channel, slot)`。
- **多资源占用（L）**：每个节点每帧允许最多占用 `L = max_tx_per_frame` 个资源（可用于“空闲资源多时占用多个时隙/信道”）。
- **链路配对**：UAV `i` 的接收端固定为 `(i+1) % N`（逻辑环）。
- **物理层**：简化 3GPP 风格的 LoS/NLoS + 阴影衰落 + Nakagami-m 快衰落；基于 SINR 判断每次发送是否成功。
- **观测（分布式）**：每个节点看到一个**有向干扰图**（src->dst），节点特征包含位置/队列/上次动作/上次状态/入度等；边特征包含归一化 SIR proxy。
- **学习策略（GAC-MAC）**：GATv2（空间聚合）+ GRUCell（时间记忆）+ MAPPO（CTDE，全局 critic）训练。
- **目标**：提高系统吞吐量、优化资源分配效果；同时在高密度下尽可能压低碰撞（collision）。

---

## 2. 代码结构（Repo Layout）

核心实现都在 `gac_mac/` 下：

- `gac_mac/config.py`：统一配置（环境/PHY/奖励/模型/训练超参）。
- `gac_mac/env/`：
  - `lawn_env.py`：主环境 LAWNEnv（按帧 step，构图、计算 SINR、队列更新、输出指标）。
  - `channel.py`：信道模型（LoS/NLoS 概率 + 路损 + 阴影 + Nakagami-m），输出全矩阵增益。
  - `mobility.py`：Gauss-Markov 移动（反射边界）。
  - `traffic.py`：泊松到达 + 队列 + per-packet delay 跟踪。
  - `graph_builder.py`：基于 carrier sensing / conflict 规则构建有向图观测（含节点/边特征）。
  - `toy_env.py`：早期 smoke test 环境（仍保留用于最小跑通）。
- `gac_mac/models/`：
  - `encoder.py`：`STGNNEncoder`（2-layer GATv2 + GRUCell）。
  - `agent.py`：`GACMACAgent`（actor + 全局 critic，动作多选、邻居约束、tie-break）。
- `gac_mac/algo/`：
  - `buffer.py`：rollout buffer（存图、动作、logp、value、reward、done、hidden等）。
  - `mappo.py`：MAPPOTrainer（PPO clipping、GAE、value clip、target KL early-stop、conflict regularizer 等）。
- `gac_mac/baselines/`：
  - `random_agent.py`：随机选择资源（队列空则 NoTx）。
  - `aloha.py`：Slotted ALOHA（p_tx=0.2）。
  - `csma.py`：CSMA/CA 风格（CW 自适应，frame-level 近似）。
  - `fixed_tdma.py`：固定 TDMA（按节点 id 分配资源）。
  - `satmac.py`：SATMAC 风格（半持久资源，碰撞重选）。
  - `hsatmac.py`：H-SATMAC 风格近似实现（半持久 BS + SlotGroup 内 CSMA/CA），带实现近似说明。
  - `greedy_coloring.py`：集中式 Greedy（基于 large-scale gain 的 pairwise conflict 图做 greedy coloring）。
- `gac_mac/scripts/`：
  - `train.py`：训练入口（toy/lawn，支持并行 env，周期性 eval，best ckpt，coll 约束等）。
  - `evaluate.py`：单点对比评测（GAC-MAC + baselines）。
  - `benchmark_density.py`：随节点密度（N）扫参评测，输出 JSON + 论文风格图。
  - 其它脚本：训练曲线绘图、拓扑快照等（见目录）。
- `gac_mac/viz/plots.py`：绘图（含 paper-style density plot）。

---

## 3. 环境 LAWNEnv：离散帧、动作-执行-转移

代码：`gac_mac/env/lawn_env.py`

### 3.1 时间结构（Frame）

一次 `env.step()` 表示一个 MAC 帧（frame）：

1) 用“当前状态”生成观测图 `obs`（包含 `obs.adj` 干扰边）。  
2) 智能体输出本帧要占用的资源（(channel,slot) 的离散索引）。  
3) 环境在本帧内计算每个发送尝试的 SINR -> 成功/碰撞；更新队列并记录 delay。  
4) 帧末更新：移动一步 + 新到达（泊松）；重新采样下一帧信道；进入下一帧。

### 3.2 资源定义与动作空间（C、K、L）

在代码里：

- `K = num_slots`：每帧时隙数。
- `C = num_channels`：正交信道数。
- 每个可用资源 index：`a = ch*K + slot`，其中：
  - `ch = a // K`，`slot = a % K`。
- `A = C*K + 1`：动作维度，最后一个为 `NoTx`（不发）。
- `L = max_tx_per_frame`：每节点每帧最大可选资源数（多资源占用）。

动作张量形状：

- 允许 `(N,)`：等价于每节点只选 1 个动作（L=1）。
- 允许 `(N, L)`：每节点一个动作序列，按顺序选资源；出现 `NoTx` 则“早停”。

#### 多资源选择：不放回 + 早停

策略网络输出的是 `A` 维 logits。若 `L>1`，则按序多次采样/argmax：

- 第 1 次从所有资源+NoTx 里选一个；
- 若不是 NoTx，则把该资源在 logits 上 mask 掉（-1e9）后继续选；
- 若选到 NoTx，则该节点后续保持 NoTx（早停）。

实现位置：`gac_mac/models/agent.py` 中 `_select_multi_actions()`。

### 3.3 “多资源”碰撞控制：secondary_lbt / primary_lbt

代码：`gac_mac/env/lawn_env.py` 中“Transmission decisions”部分。

环境对 `(N,L)` 动作进行“去重、截断到队列长度”后得到 `chosen[i]`（每节点实际尝试的资源列表）。

为了降低 “L>1” 带来的 secondary 冲突，加入了两级 LBT 机制：

- `secondary_lbt=True`：只对“第 2..L 次”资源请求做 gate（推荐使用）
  - 定义 primary 为每节点序列的第一个有效资源；
  - secondary 请求不能占用任何被 primary 使用的资源；
  - 同一资源最多只给一个 secondary（用全局随机顺序做 LBT grant），避免 secondary-secondary 冲突。
- `primary_lbt=True`：连 primary 都做 contention resolution（会把 primary 层面的同资源冲突也裁掉）
  - 这会让 collision 指标被“机制性压成 0”，不利于“真实基线对齐”，因此目前**不建议**作为主要实验设置。

> ns-3复现时，建议把 secondary_lbt 视为一种“协议规则”：当允许 L>1 时，secondary 发送需要先感知/退避，避免占用已有 primary 资源；而 primary 是否允许仲裁由你们的论文定位决定。

### 3.4 PHY：LoS/NLoS + 路损 + 阴影 + Nakagami-m

代码：`gac_mac/env/channel.py`

输入：所有 UAV 的 3D 位置 `positions_m`（N×3）。  
输出：全矩阵的“有向链路”信道增益（src->dst）。

关键点（与 `plan.md` 的约束一致）：

1) 距离单位：
   - `d` 以 **米**（m）计。
   - `fc_ghz` 以 **GHz** 直接代入路损公式。

2) LoS 概率（简化 3GPP 风格）：

`P_LoS(d2d, z_avg) = min(d1/d2d, 1) * (1 - exp(-z_avg/p1)) + exp(-z_avg/p1)`

参数：`los_d1_m`, `los_p1_m`（配置在 `Config`）。

3) 路损（dB）：

- `PL_LoS = 28.0 + 22*log10(d3d) + 20*log10(fc_ghz)`
- `PL_NLoS = -17.5 + (46 - 7*log10(z_avg)) * log10(d3d) + 20*log10(fc_ghz)`

4) 阴影衰落（dB，高斯）：

- LoS：`sigma=4 dB`
- NLoS：`sigma=6 dB`

5) 快衰落：Nakagami-m（单位均值的功率增益）

- LoS：`m=3`
- NLoS：`m=1`（退化到瑞利）

最终功率增益：

`gain = fading_power * large_scale_gain`，其中 `large_scale_gain = 1 / PL_linear`。

环境计算接收功率（瓦）：

`rx_power_w = p_tx_w * gain`

### 3.5 干扰与成功判定（按资源分组，算 SINR）

代码：`gac_mac/env/lawn_env.py`，核心逻辑：

- 每个节点 i 的接收端：`rx_i = (i+1)%N`。
- 对每个资源 `(ch,slot)`，收集所有选择该资源的发送节点集合 `g`：
  - `signal_i = P_tx * gain(i -> rx_i)`
  - `inter_i = sum_{j in g, j!=i} P_tx * gain(j -> rx_i)`
  - `SINR_i = signal_i / (inter_i + noise_w)`
  - 成功条件：`SINR_i >= sinr_threshold_lin`（配置 `sinr_threshold_db`）
- 成功则给该次尝试累计速率：`log2(1+SINR)`；失败计为 collision。

噪声：

`noise_w = dbm_to_watt(noise_psd_dbm_per_hz) * (bandwidth_hz / C)`

（每个信道均分总带宽 `bandwidth_hz`）

### 3.6 业务模型：泊松到达 + 队列 + delay

代码：`gac_mac/env/traffic.py`

- 每帧到达量：对每个节点 i，`arrivals_i ~ Poisson(lambda_arrival_per_slot * K)`。
- 队列长度 cap：`max_queue_len`（满则丢包）。
- 成功服务：若节点 i 本帧成功次数为 `succ_counts[i]`，则出队对应数量的 packet，并用 “当前帧序号 t - arrival_t” 记录 delay。

### 3.7 指标定义（info 中输出）

代码：`gac_mac/env/lawn_env.py`，每帧输出（聚合）：

- `tx_attempts`：所有节点的发送尝试总数（包含 L>1 多次）。
- `tx_collision`：所有尝试中失败次数总和（每次失败算 1）。
- `tx_success`：成功次数总和。
- `collision_rate = tx_collision / max(1, tx_attempts)`（这是当前“coll 指标”的定义）。
- `sum_rate_mbps`：
  - 先在成功尝试上累计 `rate_per_node[i] += log2(1+SINR)`；
  - 再做：`sum_rate_mbps = sum_i (bandwidth_per_channel_hz * rate_per_node[i]) / 1e6`。
- `jain`：Jain fairness index（基于 `rate_per_node`）。
- `avg_delay`：本帧所有成功服务 packet 的平均 delay（若无成功则 0）。
- 诊断项：
  - `max_group_size`：任意一个资源上同时发送的最大节点数（用于验证 coll=0 是否真实）。
  - `num_multi_tx_resources`：本帧上“被>=2节点同时占用”的资源数量。
  - `per_node_tx_attempts/successes/collisions`：每节点向量（训练不直接用，仅日志/分析）。

---

## 4. 图观测 GraphObs：节点特征、边、edge_attr

代码：`gac_mac/env/graph_builder.py`

`GraphObs` 结构：

- `x`: (N, F) 节点特征矩阵
- `edge_index`: (2, E) 有向边 src->dst
- `edge_attr`: (E, 1) 边特征（归一化 SIR proxy）
- `adj`: (N, N) uint8 邻接矩阵（src->dst）

### 4.1 有向边构建（两种模式）

输入 `rx_power_dbm` 是一个 (N,N) 的矩阵：src->“link receiver”。

在 `LAWNEnv` 中它被构造为：

- `receivers[i] = (i+1)%N`
- `rx_power_dbm_link = rx_power_dbm_large[:, receivers]`
  - 解释：列索引 i 对应“链路 i 的接收端 rx_i”，所以 `rx_power_dbm_link[src, i]` 表示 “src 的功率在 rx_i 的接收功率（dBm）”。

两种图模式：

1) `graph_mode="cs"`（carrier sensing）
   - `adj[src,dst] = 1` 当 `rx_power_dbm[src,dst] >= cs_threshold_dbm`

2) `graph_mode="conflict"`（冲突边，更利于确定性策略）
   - 先要求 `cs_mask` 成立；
   - 再用 large-scale 的 SIR proxy 判断是否会把 dst 的 SINR 拉到阈值以下：
     - `sig_dbm_all = diag(rx_power_dbm)`（每条链路自身信号功率）
     - `sir_db_mat = sig_dbm_all[dst] - rx_power_dbm[src,dst]`
     - `conflict_mask = sir_db_mat < sinr_threshold_db`
   - `adj = cs_mask & conflict_mask`

### 4.2 边特征 edge_attr（1 维）

对每条边 (src, dst)，定义：

`sir_db = P_sig(dst->rx_dst)[dBm] - P_int(src->rx_dst)[dBm]`

然后 clip 到 [-40, 20] dB，再线性归一化到 [0,1]。

### 4.3 节点特征 x 的拼接（obs_version）

公共部分：

- `pos_norm`: 位置归一化：
  - `x,y` 除以 `map_size_m`
  - `z` 除以 `height_m`
- `q_norm`: `queue / max_queue_len`（在 x 的索引 3）
- `energy`: 目前固定为 1（保留扩展接口）
- `last_actions`: one-hot 或 multi-hot（维度 A）
  - 环境里维护 `last_action_mh`，可记录 L>1 的多资源占用
- `last_status_oh`: (3) one-hot（success/collision/idle）
- `in_deg_norm`: 入度 / N（本节点可感知的潜在干扰者数量）

额外特征（由 obs_version 控制）：

- `v2` / `v3`：追加 `sig_norm`（本链路自身信号强度归一化）
- `v3`：追加 `agent_id`（i/N 的归一化 id，用于打破对称性）

> ns-3复现时，如果你们也要做 GNN/RL 的输入对齐，需要保证“图的语义一致”：边表示“谁会干扰谁”，edge_attr 表示“干扰强弱”，节点特征包含“队列、历史动作、历史反馈”等。

---

## 5. 学习算法：GAC-MAC（GATv2 + GRU + MAPPO）

### 5.1 网络结构（Actor-Critic）

代码：`gac_mac/models/encoder.py`, `gac_mac/models/agent.py`

1) 编码器 `STGNNEncoder`

- 2 层 `GATv2Conv`：
  - 第 1 层：`hidden_dim`，`heads=gat_heads`，concat
  - 第 2 层：`hidden_dim`，`heads=1`，不 concat
- 残差：把第一层输出投影到 hidden_dim 后与第二层输出相加
- 激活：ELU；归一化：LayerNorm
- 时间记忆：`GRUCell(hidden_dim, hidden_dim)`

2) Actor

- MLP：`hidden_dim -> 64 -> action_dim(A)`

3) Critic（CTDE：全局 value）

- `AttentionalAggregation` 对所有节点 hidden 做 attention pooling 得到 global embedding
- 对每个节点拼接 `[h_i, h_global]` -> value MLP 输出 `V_i`
  - 实现里会广播全局信息，使得 critic 在训练阶段更稳定（符合 CTDE）

### 5.2 执行期“策略约束/偏好”机制

代码：`gac_mac/models/agent.py`

1) `agent_id_tiebreak_eps`（仅 v3）

- 给每个节点一个 “偏好 slot” 的微小 logit bias，打破完全对称，帮助收敛到稳定 coloring。

2) `neighbor_last_action_mask / penalty`

- 根据 `edge_index` 与节点特征中的 `last_action_mh`，统计每个节点的 in-neighbors 上一帧用过的资源；
- 对这些资源施加：
  - hard mask：logits=-1e9（不允许）
  - soft penalty：logits -= penalty

> 这是“只用局部/邻居历史信息”的执行期约束，属于可在 ns-3 协议中实现的分布式规则。

### 5.3 MAPPO（PPO for Multi-Agent）

代码：`gac_mac/algo/mappo.py`, `gac_mac/algo/buffer.py`, `gac_mac/scripts/train.py`

要点：

- Rollout 收集：存 `GraphObs`（x, edge_index, edge_attr）、actions、logp、values、rewards、done、hidden state。
- GAE：`gamma=0.99`, `gae_lambda=0.95`
- PPO 目标：
  - ratio = exp(new_logp - old_logp)
  - actor_loss = -mean(min(ratio*A, clip(ratio)*A))
- value_loss：
  - 支持 value clip：`vf_clip_eps`（类似 PPO2）
- entropy bonus：`entropy_coef`
- KL early-stop：`target_kl`（每次 update 内若 approx_kl 超过阈值就提前停 epoch，防止崩）
- truncated BPTT：按 `bptt_len` 片段训练；片段起点 hidden detach；done 处清零 hidden
- conflict regularizer：`conflict_loss_coef`（对邻居 slot 概率点积做惩罚，鼓励差异化）

### 5.4 训练脚本 train.py 的关键能力

代码：`gac_mac/scripts/train.py`

支持项（与参数一致）：

- `--env toy|lawn`
- 并行采样：
  - `--num-envs N` 向量化 env
  - `--parallel-env` 用子进程并行（提升速度）
- 周期性评估 + best checkpoint：
  - `--eval-every`, `--eval-episodes`, `checkpoint_best_argmax.pth`
- greedy 模仿预训练：
  - `--pretrain-greedy-steps`, `--pretrain-only`, `checkpoint_pretrain.pth`
  - 注意：实现限制：`--pretrain-greedy-steps > 0` 与 `--parallel-env` 不兼容（teacher 需要 in-process env 状态）。
- 碰撞约束（Lagrange）与发送惩罚（用于探索“碰撞-吞吐权衡”）：
  - `--target-coll`, `--coll-lagrange-*`, `--attempt-penalty`
  - 这些在 dense-80 场景上需要谨慎，否则容易学到 degenerate 的 “NoTx”。

---

## 6. Baselines（用于对比/复现对齐）

统一接口：返回动作数组 `(N,)` 或 `(N,L)`；队列为空时应返回 NoTx。

> 重要：为了避免“胜之不武”，评测脚本支持 `--baseline-max-tx`（默认 1），使 baseline 的“每帧发送次数”按论文常见设置对齐；否则在 `L=2` 环境下让 baseline 也能每帧发 2 次，会改变其行为与指标。

### 6.1 Random

代码：`gac_mac/baselines/random_agent.py`

队列非空：均匀随机从 `[0..A-1]`（含 NoTx）选择；若 `max_tx>1` 则随机选多个不同资源（不含 NoTx），不实现早停。

### 6.2 Slotted ALOHA

代码：`gac_mac/baselines/aloha.py`

队列非空：

- 以概率 `p_tx` 尝试（默认 0.2）
- 若尝试：随机选一个资源（或多资源）
- 否则 NoTx

### 6.3 CSMA/CA（frame-level 近似）

代码：`gac_mac/baselines/csma.py`

- 每节点维护 CW（初始 `cw_min=4`，最大 `cw_max=64`）
- 每帧尝试概率 `1/CW`
- collision -> CW*2；success -> CW=cw_min；idle -> 不变

这是“协议行为近似”，并非 ns-3 802.11 的逐slot退避仿真。

### 6.4 Fixed TDMA

代码：`gac_mac/baselines/fixed_tdma.py`

确定性映射：

- `slot = i % K`
- `channel = (i // K) % C`

队列空则 NoTx。

### 6.5 SATMAC（简化版）

代码：`gac_mac/baselines/satmac.py`

- 每节点维护一个 `pref`（半持久资源）
- collision -> 以概率 `p_reselect` 重选（默认 1.0）
- success/idle（队列非空）-> 继续用 pref

### 6.6 H-SATMAC（论文风格“尽量对齐”的近似实现）

代码：`gac_mac/baselines/hsatmac.py`

实现意图：

- BS（Basic Slot）：半持久占用；collision 重选
- SG（Slot Group）：区域内共享一个 group，持续 `Tvalid` 帧；在 group 内形成连续空闲 slots 作为 contention window，做 CSMA/CA 选择

关键近似（在代码 docstring 中也写了）：

- 不显式模拟 FI/SGI 控制包；用 `env.last_actions` + `obs.adj` 近似 2-hop 感知
- “区域 geohash”用地图网格近似（默认 `R=10`）
- CSMA/CA 用“在 cw_win 内按 backoff 选 slot”的 frame-level 近似

默认参数（在 `benchmark_density.py` 里实例化时固定）：

- `LSG=4, Lmin=2, Tvalid=4, num_regions=10, cw_min=4, cw_max=64`
- `burst_q_norm=0.6`：队列归一化超过该阈值才启用 SG（额外占用资源）

### 6.7 Greedy Coloring（集中式上界参考）

代码：`gac_mac/baselines/greedy_coloring.py`

这是一个**集中式**的 greedy：

- 用环境 `_ch.large_scale_gain` 构造 pairwise conflict 图（两两同时发会使任一方 SINR<阈值则 conflict）
- 按度排序 greedy coloring 分配资源
- 支持 L>1：多轮分配（每轮再给节点多分一个资源）

它可以显著高于分布式策略（也是你之前看到 greedy 吞吐 135 Mbps 的根本原因之一：它利用了全局信息与集中式决策）。

---

## 7. 评测与绘图（用于论文风格输出）

### 7.1 单点评测 evaluate.py

代码：`gac_mac/scripts/evaluate.py`

输入：

- `--checkpoint <path>`
- `--episodes`, `--seed`, `--device`
- 可 override：`--num-uavs`, `--map-size-m`, `--num-channels`, `--max-tx-per-frame`, `--cs-threshold-dbm`
- `--secondary-lbt/--primary-lbt`
- `--baseline-max-tx`：baseline 每帧最多发送次数（默认 1）
- `--policy argmax|sample|grouped`：
  - `argmax`：确定性
  - `sample`：按 softmax 采样（可 temperature）
  - `grouped`：先比较 P(Tx)=sum(slots) vs P(NoTx) 再决定（用于减少 NoTx 崩溃）

输出：

控制台打印各算法的平均：

- `thr`（Mbps）
- `coll`
- `jain`
- `attempt/succ/coll_ct`
- `max_grp`（诊断）
- `multi_res`（诊断）
- `delay`

### 7.2 密度扫参 benchmark_density.py

代码：`gac_mac/scripts/benchmark_density.py`

- 输入 `--ns "20,30,40,50,60,80"`：扫 N
- 密度定义（固定方形区域）：
  - area_km2 = (map_size_m/1000)^2
  - density = N / area_km2
- 输出：
  - `density.json`：所有算法的曲线数据
  - `density_paper.png`：论文风格单图三子图（Throughput/Collision/Jain）

论文风格绘图实现：`gac_mac/viz/plots.py::plot_density_paper()`

特点：

- 同一张图 3 个子图；减少 legend，用曲线末端 end-label 标注算法名；
- 突出 `GAC-MAC` 与 `Greedy`，其它算法线条淡化；
- 尽量避免“折线交叉导致难读”（但当曲线真实交叉时仍可能交叉，图上以 end-label 提升可读性）。

---

## 8. 当前保留版本、最佳 checkpoint 与关键产物（可直接复现）

### 8.1 保留的有效 runs（已清理冗余）

当前 `runs_collfocus/` 仅保留两个有效版本：

1) 基础（N=30）训练得到的 best：

- `runs_collfocus\mc6_L2_nlap02_conf05_argmax-20260109-183430\checkpoints\checkpoint_best_argmax.pth`

2) dense-80 微调后 best（当前高密度最优）：

- `runs_collfocus\mc6L2_dense80_finetune-20260110-173923\checkpoints\checkpoint_best_argmax.pth`

### 8.2 dense-80 最佳评估口径（你要的“拿结果说话”）

命令（与你现在环境一致：conda env 为 `intelligent_AJ`）：

`conda run -n intelligent_AJ python -m gac_mac.scripts.evaluate --checkpoint runs_collfocus\mc6L2_dense80_finetune-20260110-173923\checkpoints\checkpoint_best_argmax.pth --episodes 200 --seed 123 --policy argmax --device cuda --num-uavs 80 --secondary-lbt --baseline-max-tx 1`

对应输出（已在本地跑过并写入 `DEVLOG.md`）：

- `GAC-MAC thr 174.344 Mbps | coll 0.260 | jain 0.578`

### 8.3 论文风格密度图（最新生成）

以 dense-80 best ckpt 做 density sweep（N=20..80）：

- `runs_density\mc6L2_dense80best_paper_20260110-233449-20260110-233740\density_paper.png`
- `runs_density\mc6L2_dense80best_paper_20260110-233449-20260110-233740\density.json`

> 注意（非常重要，涉及“对齐/公平”口径）：该图里 baseline 使用 `--baseline-max-tx 1`（单次发送对齐论文习惯），而 GAC-MAC checkpoint 是在 `L=2` 的动作空间上训练出来的（策略具备“每帧最多 2 次发送”的自由度）。如果你要在 ns-3 做“严格对齐公平对比”，需要选择其一：
>
> 1) 让 baselines 也具备 L>1 的行为（并明确这是扩展版 baseline）；或  
> 2) 把 GAC-MAC 执行期限制到 L=1（等价于只取动作序列第一个资源）；或  
> 3) 在论文中明确声明：提出方法允许“多资源占用”，baseline 为单次发送版本（这是协议设计差异，不是评测作弊）。

### 8.4 备份（训练前版本锁定）

为了 ns-3 复现建议直接引用这个备份目录（里面是“可追溯的关键文件集合”）：

- `backup\keep_before_retrain_20260110-202431\`
  - `ckpt_best_argmax_mc6L2_Ntrain30.pth`
  - `ckpt_best_argmax_mc6L2_dense80_finetune.pth`
  - `density_paper_mc6L2_20260110.png`
  - `density_mc6L2_20260110.json`
  - `DEVLOG_snapshot.md`

---

## 9. 为什么 greedy 能到 135 Mbps，而 RL 只能更低（复现时必须理解）

这不是“实现错了”这么简单，核心原因是信息与决策能力不对称：

1) GreedyColoring 是**集中式**：它直接访问 `env._ch.large_scale_gain`（全局大尺度信道矩阵），构造全局冲突图后做全局分配。
2) RL 策略是**分布式可观测**：每个节点只通过图特征看到局部邻居与局部信号强度 proxy；它并不知道全局最优的 coloring。
3) 训练目标与评估目标可能不一致：
   - `reward_mode="binary"` 时奖励是“成功次数”（非 Mbps），吞吐最优不一定等价于成功次数最优；
   - 加上协同项、idle 惩罚等，可能产生折中解。
4) 高密度 primary 冲突是主要瓶颈：
   - `secondary_lbt` 只能抑制 “第二次发送”带来的额外冲突；
   - primary 仍可能出现多人选同一资源导致大量 collision。

这也是你观察到“信道数多了 coll 没降下来”的根因之一：如果策略没有学会在 primary 层面分散（或缺少足够信息/训练稳定性），仅增加 C 并不会自动降低 primary 冲突。

---

## 10. ns-3 复现映射（把当前 Python 仿真一比一搬到 ns-3）

下面给出“尽量 1:1”复现所需的 ns-3 组件清单与建议实现方式。

### 10.1 仿真粒度：建议做“帧级离散事件”

当前 Python 环境是“每帧一步”的离散仿真（每帧内部不模拟更细粒度的退避/控制包时序）。

在 ns-3 复现有两条路：

**A) 在 ns-3 内实现一个离散帧驱动的 MAC/PHY 近似（推荐 1:1 对齐）**

- 用 `Simulator::Schedule(Seconds(T_frame), ...)` 每帧触发一次决策与统一结算。
- 不使用 Wi-Fi 复杂 PHY/MAC 的逐包退避机制，改为自定义“资源占用 + SINR 成功判定”。
- 优点：与当前 Python 模型对齐最���格���易复现结果与指标。

**B) 基于 ns-3 Spectrum/Wi-Fi 逐包模型重建**

- 需要把“时隙/信道资源占用”映射到真实 PHY 带宽/调制/包时长，且要实现 control phase、backoff 等。
- 这会偏离当前模型（因为当前模型本身不是 802.11），对齐难度很高。

本文默认你要做 A)。

### 10.2 实体与状态

对每个 UAV 节点 i，ns-3 需要维护以下状态（对应 Python）：

- 位置 `p_i(t)=(x,y,z)`、速度 `v_i(t)`（Gauss-Markov）。
- 队列长度 `Q_i(t)` 与每个 packet arrival time（用于 delay）。
- 上一帧动作 `last_actions`：
  - Python 里是 multi-hot（记录 L>1），同时保留一个“第一个动作 index”的 legacy 标量版本。
- 上一帧反馈 `last_status`：
  - 0 success / 1 collision / 2 idle。

### 10.3 Mobility：Gauss-Markov + 反射边界

直接照搬 `gac_mac/env/mobility.py`：

- `v(t+1) = a*v(t) + (1-a)*v_mean + sqrt(1-a^2)*w(t)`，`w~N(0,sigma_v^2)`
- 速度范数 clip 到 `v_max`
- `p(t+1) = p(t) + v(t+1)*dt`
- 边界反射（<0 取反，>bound 做 2*bound-pos 并翻转速度分量）

### 10.4 Channel：LoS 概率 + 路损 + 阴影 + Nakagami

照搬 `gac_mac/env/channel.py`（vectorized 版在 ns-3 可用双循环实现）：

对每对有向链路 j->i：

1) 计算 d2d/d3d（m）；z_avg。
2) 采样 is_los ~ Bernoulli(P_LoS(d2d,z_avg))。
3) 根据 LoS/NLoS 选择路损公式；再加阴影（高斯 dB）。
4) 转 linear：`large_scale_gain = 1/PL_linear`。
5) 采样 Nakagami-m 功率增益（Gamma(shape=m, scale=1/m)）。
6) `gain = fading * large_scale_gain`。

### 10.5 链路配对：逻辑环

固定：`rx_i = (i+1)%N`。

ns-3 中可以直接用节点索引来映射“本节点的接收端”，不需要真正创建独立 receiver 对象（如果做帧级结算）。

### 10.6 资源与动作（C,K,L）

每帧每节点选择一个资源序列：

- `A = C*K + 1`（NoTx 为 A-1）
- 行为同 Python：
  - 资源不放回；
  - 选到 NoTx 早停；
  - 实际尝试次数 cap 到队列长度：`min(L, Q_i)`；
  - 去重。

### 10.7 secondary_lbt/primary_lbt（协议规则层）

若实现 `secondary_lbt`：

- 先收集所有节点 primary 资源；
- secondary 请求：
  - 不能与任何 primary 资源冲突；
  - 同一资源最多给一个节点（按随机顺序 grant）。

若实现 `primary_lbt`：

- 对每个 primary 资源组随机选 1 个胜出者，其余节点 primary 直接被裁掉（相当于“争用仲裁”）。

### 10.8 SINR 与成功判定（按资源分组）

对每个资源 r=(ch,slot)，令使用该资源的节点集合为 g：

对每个 i∈g：

- `S = P_tx * gain(i->rx_i)`
- `I = sum_{j∈g, j!=i} P_tx * gain(j->rx_i)`
- `N = noise_psd_w_per_hz * (bandwidth_hz/C)`
- `SINR = S/(I+N)`
- success iff `SINR >= 10^(sinr_threshold_db/10)`

速率累计（成功才算）：

- `rate_i += log2(1+SINR)`
- 帧吞吐（Mbps/frame）：`sum_i (bandwidth_hz/C * rate_i) / 1e6`

### 10.9 collision 指标（必须一致）

当前 Python 的 collision 定义是：

- 每次发送尝试（每个节点、每个资源）都算一次 attempt；
- 若该次尝试 SINR < threshold，则该次算一次 collision；
- `collision_rate = total_collisions / total_attempts`（分母至少 1）。

在 ns-3 复现时请严格按这个口径，不要把 collision 改成“冲突资源比例”或“冲突节点比例”，否则对不上。

### 10.10 队列与 delay（per-packet）

- 每帧 arrivals：`Poisson(lambda_arrival_per_slot*K)`；队列 cap=20（默认）。
- 若本帧成功次数为 s，则出队 s 个 packet（先到先服务），delay = 当前帧号 - arrival帧号。

### 10.11 图观测（若你们在 ns-3 也要跑 GNN/RL）

如果 ns-3 只复现协议而不跑 RL，可跳过此节。

若要 RL 输入对齐，需要在 ns-3 侧构造与 Python 相同的 `GraphObs`：

- `adj` 需要用 `graph_mode=conflict` 或 `cs` 的同款规则（注意 Python 的 rx_power_dbm_link 的含义）。
- 节点特征 `x` 需要包含：
  - 归一化位置、q_norm、energy、sig_norm（v2/v3）、agent_id（v3）、last_action_mh（A维）、last_status_oh（3维）、in_deg_norm。
- 边特征 `edge_attr`：归一化 SIR proxy（[-40,20] -> [0,1]）。

### 10.12 与 RL 交互（ns-3 <-> Python）

当前训练是在 Python 内完成的。若要在 ns-3 环境上训练/推理，有三种常见路线：

1) **ns3-ai**（共享内存/消息队列与 Python 交互）
   - ns-3 负责 step、计算奖励与 obs；Python 负责策略推理与训练。
2) **自定义 socket/IPC**（每帧发送 obs -> 收动作 -> ns-3 执行）
   - 易实现但性能可能受限。
3) **离线推理**（只在 ns-3 上跑已训练好的策略）
   - Python 先导出模型（TorchScript/ONNX），ns-3 C++ 侧加载推理（工程量大）。

若你的目标是“先复现协议效果”，建议先做离线推理或直接复现 baseline + greedy，再逐步接入 RL。

---

## 11. “实现日志摘要”（从 DEVLOG.md 抽取的关键节点）

完整过程请看 `DEVLOG.md`。这里列出与 ns-3 复现强相关的关键里程碑：

- 引入 Mbps 指标（`sum_rate_mbps`）与 Jain、公平性统计；评测脚本统一输出（`gac_mac/env/lawn_env.py`, `gac_mac/utils/metrics.py`, `gac_mac/scripts/evaluate.py`）。
- 加入多信道 `C` 与多资源 `L` 的动作空间：`A=C*K+1`，动作可为 `(N,L)` 序列。
- 加入 `secondary_lbt`（只 gate secondary）与 `primary_lbt`（连 primary 也仲裁；不建议用于公平对比）。
- 训练稳定性增强：PPO 诊断、target-KL early stop、value clipping、BPTT segment shuffle、best-argmax checkpoint。
- baseline 对齐：加入 `--baseline-max-tx` 默认 1，避免 baselines 在 L=2 环境被“强行允许多次发送”导致对齐争议。
- 加入多种 baseline：ALOHA、Fixed TDMA、SATMAC、H-SATMAC（近似实现，含参数说明）。
- 图形输出优化：`benchmark_density.py --plot-style paper` 输出论文风格“同图三子图+末端标注”，减少交叉难读。
- dense-80 高密度微调：得到当前高密度最优 ckpt（吞吐 ~174 Mbps，collision ~0.26）。

---

## 12. 你接下来在 ns-3 复现时建议的“最小闭环 checklist”

1) 先只复现环境与指标（不接 RL）：
   - Mobility、Channel、PoissonTraffic、ring pairing、SINR 成功判定、collision 口径、throughput Mbps。
2) 复现 baselines（Random/ALOHA/CSMA/FixedTDMA/SATMAC/Greedy）：
   - 确保 `--baseline-max-tx=1` 对齐版本能跑出趋势一致的曲线。
3) 再引入多资源 L=2 与 `secondary_lbt`：
   - 验证 `num_multi_tx_resources`、`max_group_size` 诊断与碰撞趋势一致。
4) 最后接入 RL（推理或训练）：
   - 先离线推理（把 `checkpoint_best_argmax.pth` 放进去复现 N=80 指标）
   - 再做在线训练（ns3-ai/IPC）。

---

## 13. 参考命令（当前仓库直接复现）

（以下命令在 repo root 执行）

1) dense-80 best checkpoint 单点评测（200 eps）：

`conda run -n intelligent_AJ python -m gac_mac.scripts.evaluate --checkpoint runs_collfocus\mc6L2_dense80_finetune-20260110-173923\checkpoints\checkpoint_best_argmax.pth --episodes 200 --seed 123 --policy argmax --device cuda --num-uavs 80 --secondary-lbt --baseline-max-tx 1`

2) 生成论文风格密度图（N sweep）：

`conda run -n intelligent_AJ python -m gac_mac.scripts.benchmark_density --checkpoint runs_collfocus\mc6L2_dense80_finetune-20260110-173923\checkpoints\checkpoint_best_argmax.pth --ns "20,30,40,50,60,80" --episodes 20 --seed 123 --policy argmax --device cuda --baseline-max-tx 1 --secondary-lbt --plot-style paper --run-name mc6L2_dense80best_paper --results-dir runs_density`

