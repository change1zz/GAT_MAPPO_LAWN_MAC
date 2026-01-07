# GAC-MAC Code Walkthrough (Beginner-Friendly)
#
# NOTE: This file starts with ASCII-only lines to avoid a known Windows patching
# issue. Chinese content begins below.

---

## 0. 你应该先知道的 3 件事（非常重要）

1) 这个项目里 **有两套环境**：
- **Toy 环境**：`gac_mac/env/toy_env.py`（用于快速冒烟测试，逻辑简单）
- **完整 LAWN 环境**：`gac_mac/env/lawn_env.py`（包含信道/移动/队列/有向干扰图/SINR）

训练脚本默认跑 toy；要跑完整环境必须加：`--env lawn`。  

2) 这是一个 **多智能体**、**离散动作** 的强化学习项目：
- 每个 UAV 每一帧要在 `K` 个时隙里选一个，或者选 “不发”(No-Tx)
- 训练使用 MAPPO（PPO 的多智能体版本），策略网络是 GATv2+GRU

3) 评测时 **不要用 argmax** 直接选动作（会导致“永远不发”这种极端行为），推荐 `sample`：
- `evaluate.py` 默认 `--policy sample`

---

## 1. 一键运行（你只需要复制命令）

### 1.1 激活环境
```powershell
conda activate intelligent_AJ
```

### 1.2 训练（完整 LAWN 环境）
```powershell
python -m gac_mac.scripts.train --env lawn --run-name lawn
```

训练结束后，会在 `results/` 下生成一个新目录，例如：
`results/lawn-YYYYMMDD-HHMMSS/`

### 1.3 TensorBoard 实时监控（推荐）
训练开始后，会看到类似提示：
`TensorBoard: tensorboard --logdir results\<run>\tb`

你也可以直接监控整个 results：
```powershell
tensorboard --logdir results --port 6006
```
浏览器打开：`http://localhost:6006`

### 1.4 评测（GAC-MAC vs Random/CSMA/Greedy）
```powershell
python -m gac_mac.scripts.evaluate --checkpoint <checkpoint.pth> --episodes 50 --seed 123 --policy sample --out eval.png
```

### 1.5 生成拓扑着色快照（论文风格图）
```powershell
python -m gac_mac.scripts.snapshot_topology --checkpoint <checkpoint.pth> --seed 123 --step 5 --policy sample --out topology.png
```

---

## 2. 项目结构（从“人话”理解每个文件）

你可以把整个项目理解为 5 大块：

### 2.1 `gac_mac/scripts/`：可运行入口（你平时只用这里）
- `train.py`：训练入口（支持 toy / lawn），保存 checkpoint、写 TensorBoard、保存曲线图
- `evaluate.py`：载入 checkpoint，在固定随机种子下对比 Baselines
- `plot_training.py`：从 checkpoint 画训练曲线
- `snapshot_topology.py`：从 checkpoint + 环境生成拓扑着色图

### 2.2 `gac_mac/env/`：环境（仿真器）
- `toy_env.py`：ToyLawnEnv（简单版，只用于调通训练流程）
- `lawn_env.py`：LAWNEnv（完整环境：信道 + 移动 + 业务队列 + SINR）
- `channel.py`：ChannelModel（LoS 概率 / LoS&NLoS 路损 / shadowing / Nakagami-m）
- `mobility.py`：GaussMarkovMobility（移动模型，保证不出界）
- `traffic.py`：PoissonTraffic（泊松到达、队列、时延统计）
- `graph_builder.py`：GraphBuilder（把物理量转成 GNN 输入的图：`x` 和 `edge_index`）

### 2.3 `gac_mac/models/`：神经网络
- `encoder.py`：STGNNEncoder（两层 GATv2 + GRUCell）
- `agent.py`：GACMACAgent（Actor 输出动作分布；Value 使用全局注意力池化+按节点估值）

### 2.4 `gac_mac/algo/`：RL 算法实现（PPO/MAPPO）
- `buffer.py`：RolloutBuffer（按时间顺序存图、动作、回报、hidden state）
- `mappo.py`：MAPPOTrainer（GAE + PPO 更新 + 截断 BPTT）

### 2.5 `gac_mac/baselines/`：对比算法
- `random_agent.py`：随机
- `csma.py`：近似 CSMA/CA
- `greedy_coloring.py`：集中式贪婪“图着色式”上界近似（用链路冲突判定）

---

## 3. 训练流程全链路（从 train.py 一步步走）

训练入口：`gac_mac/scripts/train.py`

你可以把训练理解为一个大循环：

1) **初始化环境**（toy 或 lawn）
2) **初始化 agent 网络**（GATv2+GRU + Actor/Value）
3) 重复 `total_updates` 次：
   - 和环境交互 `steps_per_update` 步，存进 buffer
   - 用 MAPPOTrainer 做一次（或多次 epoch）更新
   - 写日志 / 保存 checkpoint / 写 TensorBoard

核心函数调用链（建议你在这些地方打断点）：

- `train.py`：
  - `env.reset()`
  - 循环里 `agent.act(...)`
  - `env.step(actions)`
  - `buffer.compute_gae(...)`
  - `trainer.update(...)`

---

## 4. 环境（LAWNEnv）到底模拟了什么？

### 4.1 一帧（one step）是什么意思？
在 `LAWNEnv.step()` 中，“一步”代表一个 **MAC 帧**：
- 每个 UAV 选择动作：`0..K-1`（选某个时隙发）或 `K`（No-Tx）
- 同一时隙内会互相干扰，成功与否由 SINR 决定
- step 结束后：移动一次 + 新到达一次（泊松） -> 得到下一帧状态

### 4.2 链路定义（必须看懂）
本项目使用固定配对：UAV `i` 的接收端为 `(i+1) % N`（逻辑环）。  
因此 **每个 i 都对应一条“链路 i”**。

### 4.3 信道模型（ChannelModel）在做什么？
文件：`gac_mac/env/channel.py`

它输出的是矩阵（都按“有向”链接处理）：
- `large_scale_gain[src, dst] = 1 / PL_linear`（只包含路损+阴影，不含快衰落）
- `gain[src, dst] = large_scale_gain * fading_power`（再乘 Nakagami 快衰落）

实现细节（你不用背公式，只要知道做了这些事情）：
1) 计算距离矩阵 `d2d` / `d3d`
2) 计算 LoS 概率（随距离&高度变化），随机判定 is_los
3) 分别算 LoS 路损 / NLoS 路损（`f_c` 用 GHz，`d` 用 m）
4) 加 shadowing（LoS 4dB，NLoS 6dB）
5) 快衰落用 Nakagami-m：LoS m=3，NLoS m=1（瑞利）

### 4.4 有向干扰图怎么构建？
文件：`gac_mac/env/graph_builder.py` + `gac_mac/env/lawn_env.py`

关键点：我们的图节点是 **UAV 节点**，但干扰判定是 **“谁会干扰链路 i 的接收端”**。  
因此在 LAWNEnv 中，我们把“接收端”映射成链路索引：

- `receivers[i] = (i+1)%N`（链路 i 的接收端）
- `rx_power_dbm_link[src, i] = src 发射到 receivers[i] 的接收功率`
- 构边规则：
  - 如果 `rx_power_dbm_link[src, i] >= cs_threshold_dbm`，则在图里加边 `src -> i`

这意味着：图的“dst=i”可以理解为“干扰到链路 i 的接收端（也就是 i 的下一个节点）”。

### 4.5 观测特征 x 是什么？（最常见的 debug 点）
文件：`gac_mac/env/graph_builder.py`

每个 UAV 的特征向量 `x[i]` 是拼出来的，包含：
- 归一化位置：`x,y,z`（除以 map_size / height）
- 队列：`queue/max_queue_len`
- 能量：目前固定为 1（保留扩展位）
- 上一帧动作 onehot：维度 `K+1`
- 上一帧状态 onehot：Success/Collision/Idle（3维）
- 入度（in-degree）归一化：`deg_in/N`

调试建议：你可以在 `LAWNEnv.reset()` 后打印：
- `obs.x.shape`（应该是 `(N, K+10)`）
- `obs.edge_index.shape`（(2,E)，E 随时变）
- `obs.adj.sum(axis=0).mean()`（平均入度，太高/太低都不利于学习）

### 4.6 reward 是怎么计算的？
文件：`gac_mac/env/lawn_env.py`（toy 版在 `toy_env.py`）

我们分三块：

1) `r_perf`：性能项  
- 成功：`+1`
- 空闲且队列空：`+0.1`

2) `r_pen`：惩罚项  
- 碰撞（SINR 不够）：`-2`
- 空闲但队列不空：`-0.5`

3) `r_coop`：协同项（严格按邻居集合）  
邻居集合使用 **入邻居**：`N_i = { j | j -> i }`  
公式：
`r_coop[i] = lambda_coop * mean_{j in N_i}(r_perf[j])`

如果 `N_i` 为空，协同项为 0。

---

## 5. 模型（GATv2+GRU）在做什么？

### 5.1 Encoder：STGNNEncoder
文件：`gac_mac/models/encoder.py`

输入：
- `x`：节点特征 `(N, F)`
- `edge_index`：图边 `(2, E)`
- `h_in`：GRU 的上一次隐藏状态 `(N, H)`

输出：
- `h_out`：新的隐藏状态 `(N, H)`（也是每个 UAV 的 embedding）

结构：
1) `GATv2Conv` 第1层（多头）
2) LayerNorm + ELU
3) `GATv2Conv` 第2层（聚合）
4) 残差（把第1层投影后加到第2层输出）
5) GRUCell（把“当前图 embedding”与“历史记忆”融合）

### 5.2 Actor：输出动作 logits
文件：`gac_mac/models/agent.py`

用一个小 MLP 把 `h_out` 映射到 `(N, K+1)` 的 logits，代表每个动作的偏好。

### 5.3 Value（Critic）：为什么是“全局 + 按节点”？
文件：`gac_mac/models/agent.py`

做法：
1) 用 `AttentionalAggregation` 把全图节点 embedding 聚合成 `global_feat`
2) 对每个节点 i，把 `[h_i, global_feat]` 拼起来，输出 `V_i`

好处：
- 仍然是 CTDE（训练时可利用全局信息）
- 但 Value 不是“所有节点一样的一个标量”，对优化更稳定

---

## 6. MAPPO（训练算法）怎么更新？

### 6.1 RolloutBuffer 为什么这样存？
文件：`gac_mac/algo/buffer.py`

为了避免 “PyG Batch + RNN 对齐” 的经典坑，我们按 **时间顺序** 存：
- 第 t 步：存 `x_t, edge_index_t, action_t, logp_t, value_t, reward_t, done_t, h_in_t`
这样永远清楚 `h_in_t` 对应的是哪一帧、哪一批节点。

### 6.2 GAE（优势函数）怎么来的？
`compute_gae()` 做的事：
- 用 `rewards + gamma * V_next - V` 得到 TD error
- 递推累计成 advantage
- returns = advantage + V

### 6.3 截断 BPTT（非常重要）
文件：`gac_mac/algo/mappo.py`

训练时我们把时间序列切成一段一段（长度 `bptt_len`）：
- 段的开头 hidden state 做 `detach()`，切断梯度
- 段内按时间顺序跑 encoder（保证 GRU 的时序一致）
- 遇到 done（episode 结束）会把 hidden 清零，避免跨 episode 串味

### 6.4 PPO loss（你只要知道“在变好”）
PPO 核心：
- `ratio = exp(new_logp - old_logp)`
- clip ratio 在 `[1-eps, 1+eps]`
- actor loss 是 “让 advantage 大的动作概率上升，但不要一下子变太猛”
- value loss 是 MSE
- entropy 是鼓励探索

---

## 7. 评测（evaluate.py）怎么保证公平？

文件：`gac_mac/scripts/evaluate.py`

关键点：
- 对所有算法固定环境 seed：每个 episode 用 `seed = base_seed + ep`
- 这样 Random / CSMA / Greedy / GAC-MAC 在同一批“地图初始位置+移动+到达”上对比

GAC-MAC 动作策略选择：
- `--policy sample`：按策略分布采样（推荐）
- `--policy argmax`：直接取最大概率（容易变成全 No-Tx）
- `--policy grouped`：比较 P(Tx) 与 P(NoTx) 决定是否发（更保守）

---

## 8. 调试手册（照着做就能定位问题）

### 8.1 先确认你跑的是 LAWN 还是 toy
- toy：`python -m gac_mac.scripts.train`（默认）
- lawn：`python -m gac_mac.scripts.train --env lawn`

### 8.2 用最小参数快速复现问题（强烈推荐）
```powershell
python -m gac_mac.scripts.train --env lawn --total-updates 1 --steps-per-update 8 --log-interval 1
```

### 8.3 最常见问题 1：图太密/太稀，学不动
观察 TensorBoard 的 `env/avg_degree_in`：
- 太大（接近 N）-> 全连接图，噪声大，学不动
- 太小（接近 0）-> 没边，GNN 退化

你可以调 `cs_threshold_dbm`：
```powershell
python -m gac_mac.scripts.train --env lawn --cs-threshold-dbm -50
```
经验：平均入度 4~12 更容易学（不是绝对值，主要看稳定性）。

### 8.4 最常见问题 2：模型“永远不发”（No-Tx）
看 TensorBoard：
- `policy/no_tx_frac_given_queue` 如果接近 1，说明只要有包也不发

常见原因：
- 奖励尺度不合理（碰撞惩罚太大导致“装死”）
- 评测用 argmax（策略会变得极端）

解决：
- 评测用 `--policy sample`
- 训练时观察 reward/entropy 是否快速掉到很低

### 8.5 打断点应该打在哪里？
如果你用 VSCode/PyCharm：

1) 环境是否正确：在 `LAWNEnv.step()` 里看 `sinr/success/collision` 是否正常变化  
文件：`gac_mac/env/lawn_env.py`

2) 观测是否正常：在 `GraphBuilder.build()` 看 `x` 的每一段是否在 [0,1] 左右  
文件：`gac_mac/env/graph_builder.py`

3) 网络是否输出正常：在 `agent.actor(h_out)` 后看 logits 是否全一样  
文件：`gac_mac/models/agent.py`

4) PPO 是否在更新：在 `trainer.update()` 里看 loss 是否 NaN、ratio 是否爆炸  
文件：`gac_mac/algo/mappo.py`

### 8.6 最小化打印（不会刷屏）
你可以在这些位置加一两行 print（调完记得删）：
- `LAWNEnv.step()` 末尾：打印 `attempts/success/collision/sinr.mean`
- `train.py` 每 N 个 update：打印 `no_tx_frac_given_queue`

---

## 9. 我建议你用 TensorBoard 重点看哪些曲线？

训练是否在“收敛/变好”，最直观的指标：
- `train/reward_mean`
- `train/sum_rate_mean`
- `train/collision_rate_mean`
- `loss/value`（通常会逐渐下降到一个稳定范围）
- `loss/entropy`（会缓慢下降，但不应立刻变成 0）

如果你看到：
- reward 一直极低、entropy 很快归零、NoTx 比例很高  
=> 大概率策略陷入“装死”，优先检查奖励与评测策略。

---

## 10. FAQ（你可能会遇到）

### Q1：为什么 `evaluate --policy argmax` 会出现 sum_rate=0？
因为 `argmax` 会把概率最大动作当作硬决策；如果模型学到“稳妥=不发”，argmax 就会全部 No-Tx。  
用 `--policy sample` 能更真实体现“策略分布”的效果。

### Q2：Greedy 为什么看起来像上界？
Greedy 是集中式近似，能利用全局冲突信息来“贪婪着色”，并不受分布式局部观测限制。

---

如果你希望我再补一个“从 0 到能跑”的 VSCode 调试配置（launch.json）示例，我也可以直接加到 `.vscode/`（并说明每个字段是什么意思）。  
