# Dev Log

## 2026-01-07

- Backup before changes: `backup/pre_scale_units_20260107-104422/`
- Goal: add units to evaluation/benchmark plots; add scaling experiments for both fixed-area (varying density) and fixed-density (expanding area) settings while keeping CS power-threshold graph.
- Changes:
  - Added plot units for throughput/collision/delay in `gac_mac/viz/plots.py`.
  - Updated scaling benchmark to run `fixed_area` and `fixed_density` modes and save separate figures + `scaling_both.json` in the run directory (`gac_mac/scripts/benchmark_scaling.py`).
- Generated artifacts (example checkpoint):
  - `results/lawn-big-v2-fixlast-20260106-220033/scaling_fixed_area.png`
  - `results/lawn-big-v2-fixlast-20260106-220033/scaling_fixed_density.png`
  - `results/lawn-big-v2-fixlast-20260106-220033/scaling_both.json`

## 2026-01-07 (Algorithm Refinement / Compatibility)

- Backups:
  - `backup/pre_algo_refine_20260107-113037/` (global critic + action masking work)
  - `backup/pre_hier_actor_20260107-120807/` (hierarchical actor experiments)
- Changes (kept backward compatible with existing best checkpoints):
  - Added `Config.critic_mode` (`node|global_mean|global`) and `Config.action_mask_empty_queue` (enforce `No-Tx` when queue is empty).
  - Updated trainer to apply the same empty-queue action mask during PPO updates (`gac_mac/algo/mappo.py`).
  - Added `Config.policy_mode` (`flat|hierarchical`) as an optional experiment, but default remains `flat`.
  - Preserved old checkpoint keys by keeping the flat actor named `agent.actor`; hierarchical heads are extra modules (`tx_head`, `slot_head`).
  - Switched evaluation/benchmark to use `agent.action_logits()` so flat/hierarchical share one interface.
- Quick validation (existing best checkpoint still loads & matches prior performance):
  - `results/lawn-big-v2-fixlast-20260106-220033/checkpoints/checkpoint_3000.pth`
    - `--policy sample --temperature 0.5`: `sum_rate 90.262 | coll 0.349 | delay 23.425`
    - `--policy argmax`: `sum_rate 60.833 | coll 0.493 | delay 24.888`
- Experiments (hierarchical actor) did not outperform the current best within limited runs:
  - `results/lawn-v2-hier-20260107-121525/` and `results/lawn-v2-hier-tune-20260107-130018/` (kept for reference).

## 2026-01-07 (Git + Mbps Metrics)

- Git:
  - Initial repo commit: `c442934` (tag: `pre-mbps-metric`)
  - Added `.gitignore` to keep `results/`, `backup/`, `tmp_papers/` and large artifacts out of git.
- Metrics:
  - Added `throughput_mbps` to `LAWNEnv` step info, derived from `sum_rate * (bandwidth_hz/num_slots) / 1e6` (`gac_mac/utils/metrics.py`).
  - Updated `train/evaluate/benchmark_scaling` to report and plot Mbps when available.
  - Smoke check: `python -m gac_mac.scripts.train --env lawn --run-name smoke-mbps ...` shows logs like `thr X.XX Mbps`.

## 2026-01-07 (Phase A: Evaluation + Fairness)

- Git backup tag before changes: `pre-phase-a-metrics` (at `490bcac`)
- Added per-node evaluation signals to env step info (does not change observation):
  - `per_node_throughput_mbps`, `per_node_tx_attempt`, `per_node_tx_success`, `per_node_tx_collision`, `per_node_delay_served` (`gac_mac/env/lawn_env.py`, `gac_mac/env/traffic.py`)
- Added Jain fairness metric (Mbps-based): `jain_index()` (`gac_mac/utils/metrics.py`)
- Evaluation improvements:
  - Default fixed eval seeds: `Config.eval_seeds = [123..132]` (`gac_mac/config.py`)
  - `evaluate.py` now reports throughput Mbps + Jain, and saves a detailed JSON including per-seed per-node vectors when `--out` is used.
  - Example outputs:
    - `results/lawn-big-v2-fixlast-20260106-220033/eval_phaseA.png`
    - `results/lawn-big-v2-fixlast-20260106-220033/eval_phaseA.json`
- Scaling benchmark improvements:
  - Adds Jain fairness trend to scaling plots as a 4th subplot when present (`gac_mac/scripts/benchmark_scaling.py`, `gac_mac/viz/plots.py`)

## 2026-01-07 (Phase B: PPO Stability + Anti-Collapse)

- Note: From this point forward, version backups are done via git commits/tags (no more directory copies).
- PPO diagnostics + stability:
  - Added PPO training diagnostics: `approx_kl`, `clip_frac`, `explained_variance`, `grad_norm` (`gac_mac/algo/mappo.py`) and logs them to TensorBoard (`gac_mac/scripts/train.py`).
  - Added PPO KL-based early stopping within each update via `Config.target_kl` / `--target-kl` (`gac_mac/algo/mappo.py`, `gac_mac/scripts/train.py`).
  - Added per-update action distribution logging (Tx/No-Tx and per-slot fractions conditioned on queue>0) to TensorBoard (`gac_mac/scripts/train.py`).
- Reward refinements (to avoid “silence”/degenerate equilibria while remaining local-observation compatible):
  - Cooperative reward now uses the full per-agent outcome reward `r_perf` (including collision/idle penalties), and total reward is `r_perf + r_coop` (`gac_mac/env/lawn_env.py`, `gac_mac/env/toy_env.py`).
  - Added semi-persistent shaping: repeat the same slot after a previous success gets a bonus; repeating after a previous collision gets a penalty (`Config.reward_repeat_success`, `Config.reward_repeat_collision`) (`gac_mac/env/lawn_env.py`, `gac_mac/env/toy_env.py`).
- Training robustness:
  - Added automatic best-checkpoint tracking based on MA(10) of `throughput_mbps` and saving to `checkpoints/checkpoint_best.pth` (`gac_mac/scripts/train.py`).
- Quick validation run (RTX 4070 Ti SUPER, `--num-envs 8 --parallel-env`, LAWN env):
  - Run: `results/stable_best_track-20260107-174250/`
  - Best checkpoint: `results/stable_best_track-20260107-174250/checkpoints/checkpoint_best.pth`
    - `evaluate --policy sample --temperature 0.7` (10 seeds): `thr 8.415 Mbps | coll 0.609 | jain 0.854`
    - `evaluate --policy argmax`: collapses to No-Tx (0 throughput); use sampling policy for deployment/eval.

## 2026-01-07 (Branch exp/distill-greedy)

- Goal: use a centralized GreedyColoring teacher to distill slot decisions into the same POMDP-local policy (no new observation features at inference).
- Changes:
  - Fix hierarchical entropy to use expected entropy: `H = H(tx) + p_tx * H(slot)` (`gac_mac/models/agent.py`).
  - Add optional distillation loss (cross-entropy on greedy actions) controlled by `Config.distill_coef` / `--distill-coef` (`gac_mac/algo/buffer.py`, `gac_mac/algo/mappo.py`, `gac_mac/scripts/train.py`).
  - Distillation uses SyncVectorEnv (in-process) so the trainer can access per-env teacher labels (it auto-disables `--parallel-env` when distill is enabled).
- Reference run:
  - `results/B_distill_greedy-20260107-233255/checkpoints/checkpoint_0200.pth`
  - `evaluate --policy sample --temperature 0.7` (10 seeds): `thr 8.435 Mbps | coll 0.416 | jain 0.863`

## 2026-01-08 (exp/distill-greedy: reward aligned to throughput)

- Observation: binary success/collision reward is misaligned with Mbps throughput; sampling temperature heavily impacts Tx activity.
- Change: switch training to `reward_mode=rate` (success reward uses `log2(1+SINR)`), keep distillation, and add a small static Tx-attempt bonus.
- Best checkpoint found so far (10 seeds eval, `--policy sample --temperature 0.7`):
  - `results/B_distill_rate_reward-20260108-125732/checkpoints/checkpoint_0050.pth`
  - `thr 8.539 Mbps | coll 0.514 | jain 0.860`

## 2026-01-08 (Baseline Lock + Armax Debug)

- Baseline frozen from backup directory:
  - Branch: `baseline/pre_scale_units_20260107-104422`
  - Tag: `baseline-pre_scale_units_20260107-104422`
- New work branch from that tag: `exp/argmax-flat-opt`
- Eval alignment:
  - `LAWNEnv` now reports `sum_rate_mbps` + `jain` in step `info` (used by `scripts/evaluate.py` and plots).
- Training knobs added to reduce argmax collapse (still within flat `K+1` action space):
  - Optional greedy warm-start: `--pretrain-greedy-steps`
  - Optional conflict regularizer: `--conflict-loss-coef`
  - Optional agent-id tie-break (v3 only): `--agent-id-tiebreak-eps`
  - Optional argmax rollout collection: `--rollout-policy argmax`
  - Periodic argmax eval + best checkpoint: `--eval-every/--eval-episodes` saves `checkpoints/checkpoint_best_argmax.pth`

## 2026-01-08 (Argmax Throughput Optimization: conflict graph + bigger model)

- Key finding: switching `GraphBuilder` to `--graph-mode conflict` is the most impactful change for deterministic argmax evaluation.
- Best so far (argmax policy):
  - Run: `results/opt4_argmax_conflict_hd128h4_v3_lr1e4_tie01_idle07-20260108-200453/`
  - Config highlights: `--graph-mode conflict --hidden-dim 128 --gat-heads 4 --lr 1e-4 --agent-id-tiebreak-eps 0.1 --reward-idle-nonempty -0.7`
  - Best checkpoint: `results/opt4_argmax_conflict_hd128h4_v3_lr1e4_tie01_idle07-20260108-200453/checkpoints/checkpoint_best_argmax.pth`
  - `evaluate --policy argmax --episodes 30`: `thr 54.754 Mbps | coll 0.644 | jain 0.103` (close to CSMA baseline in this setting)

## 2026-01-08 (Argmax >= CSMA)

- Add periodic argmax evaluation inside training and select best via `checkpoint_best_argmax.pth` (reduce selection noise with larger `--eval-episodes`).
- Run: `results/opt6_argmax_conflict_hd128h4_v3_lr1e4_tie01_idle07_eval20-20260108-201637/`
  - Config highlights: `--graph-mode conflict --obs-version v3 --hidden-dim 128 --gat-heads 4 --lr 1e-4 --agent-id-tiebreak-eps 0.1 --reward-idle-nonempty -0.7 --eval-every 20 --eval-episodes 20`
  - Best checkpoint: `results/opt6_argmax_conflict_hd128h4_v3_lr1e4_tie01_idle07_eval20-20260108-201637/checkpoints/checkpoint_best_argmax.pth`
  - `evaluate --policy argmax --episodes 50`: `thr 59.747 Mbps | coll 0.645 | jain 0.109` (beats CSMA `57.453 Mbps` on same eval setting)

## 2026-01-08 (Local Anti-Collision Mask)

- Added an optional policy-side constraint using only local/neighbor observations: avoid slots used by in-neighbors in the previous frame.
  - Flags: `--neighbor-last-action-mask` (hard mask) / `--neighbor-last-action-penalty <p>` (soft logit penalty)
- Quick check:
  - Run: `results/opt7_masklast_conflict_hd128h4_v3_lr1e4_tie01_idle07-20260108-203340/`
  - Best checkpoint: `results/opt7_masklast_conflict_hd128h4_v3_lr1e4_tie01_idle07-20260108-203340/checkpoints/checkpoint_best_argmax.pth`
  - `evaluate --policy argmax --episodes 50`: `thr 57.523 Mbps | coll 0.802 | jain 0.108` (about CSMA-level but collision ratio increases under this metric)

## 2026-01-08 (Argmax Sweep: New Best)

- Sweep runner added: `python -m gac_mac.scripts.sweep_argmax` (auto-trains and evaluates candidates).
- New best (argmax policy, 100 episodes eval):
  - Run: `results/sweepA_idle070_tie010_lr1e4-20260108-204929/`
  - Best checkpoint: `results/sweepA_idle070_tie010_lr1e4-20260108-204929/checkpoints/checkpoint_best_argmax.pth`
  - `evaluate --policy argmax --episodes 100`: `thr 70.825 Mbps | coll 0.385 | jain 0.125`
  - Baselines on same eval: `CSMA thr 58.041 Mbps`, `Greedy thr 135.713 Mbps`
- Convergence check: later checkpoints in the same run stay around ~`8.39–8.50 Mbps` (no consistent improvement beyond `checkpoint_0050.pth`).

## 2026-01-08 (Argmax New Best: Tune cs_threshold + tie-break)

- Stability note: to avoid losing run artifacts, newer experiments are saved under `runs_argmax/` via `--results-dir runs_argmax`.
- Key finding: `cs_threshold_dbm` strongly affects the *conflict graph* density; lowering from `-50` to `-60` improved deterministic argmax performance.
- New best (argmax policy, averaged over 500 eval episodes):
  - Run: `runs_argmax/sweepC_seed42_cs60_idle070_tie015_lr7e5-20260108-223303/`
  - Config highlights: `--graph-mode conflict --obs-version v3 --hidden-dim 128 --gat-heads 4 --lr 7e-5 --reward-idle-nonempty -0.7 --agent-id-tiebreak-eps 0.15 --cs-threshold-dbm -60`
  - Best checkpoint: `runs_argmax/sweepC_seed42_cs60_idle070_tie015_lr7e5-20260108-223303/checkpoints/checkpoint_best_argmax.pth`
  - `evaluate --policy argmax --episodes 500 --seed 123`:
    - `GAC-MAC thr 71.994 Mbps | jain 0.127 | coll 0.472 | delay 19.311`
    - `CSMA thr 58.332 Mbps | coll 0.288`
    - `Greedy thr 135.505 Mbps | coll 0.027`
  - Repro:
    - `conda run -n intelligent_AJ python -m gac_mac.scripts.evaluate --checkpoint runs_argmax\\sweepC_seed42_cs60_idle070_tie015_lr7e5-20260108-223303\\checkpoints\\checkpoint_best_argmax.pth --episodes 500 --seed 123 --policy argmax --device cuda`

## 2026-01-08 (Multi-Channel MAC: C*K flat action)

- Backup:
  - Tag: `pre-multichannel-20260108-2250` (before multi-channel changes)
  - Branch: `exp/multichannel-mac`
- Design (kept backward compatible with `num_channels=1`):
  - Add `Config.num_channels` (C). Action becomes flat `a ∈ [0..C*K-1] ∪ {C*K(NoTx)}` (still argmax-friendly).
  - LAWNEnv interference only happens when two links pick the same `(channel, slot)`; total bandwidth is fixed and split across channels (`B_c = B/C`), noise scales with `B_c`.
  - GraphBuilder last-action one-hot resized to `C*K+1` so the policy can learn per-channel reuse patterns from local history.
- Smoke result (C=2):
  - Train:
    - `conda run -n intelligent_AJ python -m gac_mac.scripts.train --env lawn --device cuda --seed 42 --num-channels 2 --reward-mode rate --graph-mode conflict --obs-version v3 --cs-threshold-dbm -60 --reward-idle-nonempty -0.7 --agent-id-tiebreak-eps 0.15 --hidden-dim 128 --gat-heads 4 --lr 7e-5 --entropy-coef 0.01 --num-envs 8 --steps-per-update 64 --total-updates 140 --eval-every 20 --eval-episodes 20 --results-dir runs_multich --run-name mc2_cs60_idle070_tie015_lr7e5`
    - Best ckpt: `runs_multich/mc2_cs60_idle070_tie015_lr7e5-20260108-233210/checkpoints/checkpoint_best_argmax.pth`
  - Eval (argmax, 300 episodes):
    - `GAC-MAC thr 77.121 Mbps | jain 0.254 | coll 0.711 | delay 13.346`
    - `Greedy thr 71.774 Mbps | coll 0.707` (pairwise-conflict coloring is not enough under aggregate multi-interferer collisions)

## 2026-01-09 (Multi-Resource Scheduling: allow multi-(channel,slot) per frame)

- Goal: improve throughput by allowing nodes to occupy multiple idle resources when available (still fixed total bandwidth; multi-channel is for collision-avoidance freedom).
- Design:
  - Per-node action becomes a sequence of up to `L=max_tx_per_frame` picks from `C*K` resources, with `NoTx` as an early-stop token.
  - Environment computes SINR/collisions per resource (same `(channel,slot)` interferes) and serves up to `success_counts[i]` packets per frame.
  - Reward/metrics aggregate over per-resource attempts; collision penalty scales with collision count (keeps "use more only if beneficial").
- Long run (C=2, L=2):
  - Train:
    - `conda run -n intelligent_AJ python -m gac_mac.scripts.train --env lawn --device cuda --seed 42 --num-channels 2 --max-tx-per-frame 2 --reward-mode rate --graph-mode conflict --obs-version v3 --cs-threshold-dbm -60 --reward-idle-nonempty -0.7 --reward-collision -2.0 --agent-id-tiebreak-eps 0.15 --hidden-dim 128 --gat-heads 4 --lr 7e-5 --entropy-coef 0.01 --conflict-loss-coef 0.2 --num-envs 8 --steps-per-update 64 --total-updates 600 --eval-every 25 --eval-episodes 30 --results-dir runs_multich_long --run-name mc2_L2_conf02_cs60_lr7e5`
    - Best ckpt: `runs_multich_long/mc2_L2_conf02_cs60_lr7e5-20260108-235857/checkpoints/checkpoint_best_argmax.pth`
  - Eval (argmax, 500 episodes):
    - `conda run -n intelligent_AJ python -m gac_mac.scripts.evaluate --checkpoint runs_multich_long\\mc2_L2_conf02_cs60_lr7e5-20260108-235857\\checkpoints\\checkpoint_best_argmax.pth --episodes 500 --seed 123 --policy argmax --device cuda`
    - `GAC-MAC thr 88.743 Mbps | jain 0.292 | coll 0.332 | delay 13.732`
    - Baselines: `Greedy thr 139.900 Mbps | coll 0.105`, `CSMA thr 54.971 Mbps | coll 0.228`

## 2026-01-09 (Sweep C=6..8: more channels, fixed total bandwidth)

- Notes:
  - Total bandwidth stays fixed; per-channel bandwidth is `B/C` (no "extra bandwidth").
  - Increasing `C` reduces same-resource collisions, but also reduces per-link throughput per success; net effect depends on how many additional successful transmissions are unlocked.

- C=6, L=2 (best in this sweep):
  - Train: `conda run -n intelligent_AJ python -m gac_mac.scripts.train --env lawn --device cuda --seed 42 --num-channels 6 --max-tx-per-frame 2 --reward-mode rate --graph-mode conflict --obs-version v3 --cs-threshold-dbm -60 --reward-idle-nonempty -0.7 --reward-collision -2.0 --agent-id-tiebreak-eps 0.15 --hidden-dim 128 --gat-heads 4 --lr 7e-5 --entropy-coef 0.01 --conflict-loss-coef 0.2 --num-envs 8 --steps-per-update 64 --total-updates 350 --eval-every 25 --eval-episodes 30 --results-dir runs_multich_Csweep --run-name mc6_L2_conf02_cs60_lr7e5`
  - Best ckpt: `runs_multich_Csweep/mc6_L2_conf02_cs60_lr7e5-20260109-090851/checkpoints/checkpoint_best_argmax.pth`
  - Eval (argmax, 500 episodes):
    - `conda run -n intelligent_AJ python -m gac_mac.scripts.evaluate --checkpoint runs_multich_Csweep\\mc6_L2_conf02_cs60_lr7e5-20260109-090851\\checkpoints\\checkpoint_best_argmax.pth --episodes 500 --seed 123 --policy argmax --device cuda`
    - `GAC-MAC thr 88.046 Mbps | jain 0.772 | coll 0.591 | delay 11.184`

- C=8, L=2 (degrades; even below Random baseline):
  - Train: `conda run -n intelligent_AJ python -m gac_mac.scripts.train --env lawn --device cuda --seed 42 --num-channels 8 --max-tx-per-frame 2 --reward-mode rate --graph-mode conflict --obs-version v3 --cs-threshold-dbm -60 --reward-idle-nonempty -0.7 --reward-collision -2.0 --agent-id-tiebreak-eps 0.15 --hidden-dim 128 --gat-heads 4 --lr 7e-5 --entropy-coef 0.01 --conflict-loss-coef 0.2 --num-envs 8 --steps-per-update 64 --total-updates 350 --eval-every 25 --eval-episodes 30 --results-dir runs_multich_Csweep --run-name mc8_L2_conf02_cs60_lr7e5`
  - Best ckpt: `runs_multich_Csweep/mc8_L2_conf02_cs60_lr7e5-20260109-093154/checkpoints/checkpoint_best_argmax.pth`
  - Eval (argmax, 500 episodes):
    - `conda run -n intelligent_AJ python -m gac_mac.scripts.evaluate --checkpoint runs_multich_Csweep\\mc8_L2_conf02_cs60_lr7e5-20260109-093154\\checkpoints\\checkpoint_best_argmax.pth --episodes 500 --seed 123 --policy argmax --device cuda`
    - `GAC-MAC thr 44.424 Mbps | jain 0.510 | coll 0.733 | delay 11.426`

- Extra attempts (not improvements):
  - C=6, L=3 with heavier regularization collapsed under argmax (thr ~10–18 Mbps): `runs_multich_Csweep/mc6_L3_conf05_nlap05_rs025_rc10_cs60-20260109-100757/`
  - C=6, L=2 with `neighbor_last_action_penalty=0.5` + `conflict_loss_coef=0.5` also suffered argmax collapse: `runs_multich_Csweep/mc6_L2_conf05_nlap05_temp08_cs60-20260109-105651/`

## 2026-01-09 (Constrained Training: improve thr while nudging coll down)

- Motivation: current best `C=2,L=2` has high throughput but coll ~0.33; want to push both (Pareto improvement) without adding bandwidth.
- Implementation:
  - Env now returns per-node per-frame counts in `info` (does not affect observation):
    - `per_node_tx_attempts`, `per_node_tx_collisions`, `per_node_tx_successes`
  - Train supports optional collision constraint (Lagrange) with warmup + cap:
    - `--target-coll`, `--coll-lagrange-lr`, `--coll-lagrange-max`, `--coll-warmup-updates`
    - extra `--attempt-penalty` to discourage spamming (set to 0.0 in the best run below)
- Best constrained run found (C=2, L=2):
  - Train:
    - `conda run -n intelligent_AJ python -m gac_mac.scripts.train --env lawn --device cuda --seed 42 --num-channels 2 --max-tx-per-frame 2 --reward-mode rate --graph-mode conflict --obs-version v3 --cs-threshold-dbm -60 --reward-idle-nonempty -0.7 --reward-collision -2.0 --agent-id-tiebreak-eps 0.15 --hidden-dim 128 --gat-heads 4 --lr 7e-5 --entropy-coef 0.01 --conflict-loss-coef 0.2 --num-envs 8 --steps-per-update 64 --total-updates 350 --eval-every 25 --eval-episodes 30 --eval-seed 123 --rollout-temperature 1.0 --target-coll 0.30 --coll-lagrange-lr 0.05 --coll-lagrange-max 6.0 --coll-warmup-updates 75 --attempt-penalty 0.0 --no-tensorboard --results-dir runs_constrained2 --run-name mc2_L2_constrained_tc030_lr005_cap6_wu75`
  - Best ckpt: `runs_constrained2/mc2_L2_constrained_tc030_lr005_cap6_wu75-20260109-150406/checkpoints/checkpoint_best_argmax.pth`
  - Eval (argmax, 500 episodes):
    - `conda run -n intelligent_AJ python -m gac_mac.scripts.evaluate --checkpoint runs_constrained2\\mc2_L2_constrained_tc030_lr005_cap6_wu75-20260109-150406\\checkpoints\\checkpoint_best_argmax.pth --episodes 500 --seed 123 --policy argmax --device cuda`
    - `GAC-MAC thr 91.396 Mbps | jain 0.299 | coll 0.326 | delay 13.066`

## 2026-01-09 (Fix C↑ but coll not ↓: align rollout to argmax + local anti-repeat)

- Root cause (confirmed in experiments): with larger `C`, action space grows and training with stochastic rollouts can look "fine" in on-policy returns but fails to learn a stable *deterministic* resource dispersion; collisions remain high because many nodes still concentrate on overlapping resources.
- Effective fix: train using deterministic rollout collection (`--rollout-policy argmax`) and add a local soft penalty against reusing in-neighbors' last-frame resources (`--neighbor-last-action-penalty`).
- Best C=6 run so far (C=6, L=2) with meaningful coll reduction:
  - Train:
    - `conda run -n intelligent_AJ python -m gac_mac.scripts.train --env lawn --device cuda --seed 42 --num-channels 6 --max-tx-per-frame 2 --reward-mode rate --graph-mode conflict --obs-version v3 --cs-threshold-dbm -60 --reward-idle-nonempty -0.7 --reward-collision -2.0 --agent-id-tiebreak-eps 0.15 --neighbor-last-action-penalty 0.2 --hidden-dim 128 --gat-heads 4 --lr 7e-5 --entropy-coef 0.005 --conflict-loss-coef 0.5 --num-envs 8 --steps-per-update 64 --total-updates 350 --eval-every 25 --eval-episodes 40 --eval-seed 123 --rollout-policy argmax --no-tensorboard --results-dir runs_collfocus --run-name mc6_L2_nlap02_conf05_argmax`
  - Best ckpt: `runs_collfocus/mc6_L2_nlap02_conf05_argmax-20260109-183430/checkpoints/checkpoint_best_argmax.pth`
  - Eval (argmax, 500 episodes):
    - `conda run -n intelligent_AJ python -m gac_mac.scripts.evaluate --checkpoint runs_collfocus\\mc6_L2_nlap02_conf05_argmax-20260109-183430\\checkpoints\\checkpoint_best_argmax.pth --episodes 500 --seed 123 --policy argmax --device cuda`
    - `GAC-MAC thr 110.039 Mbps | jain 0.950 | coll 0.490 | delay 10.935`
    - Baselines (same eval): `Random thr 75.761 Mbps | coll 0.556`, `Greedy thr 117.748 Mbps | coll 0.104`

## 2026-01-09 (Longer Training + Parallel Env)

- Goal: push `C=6,L=2` further by training longer and speeding up rollouts with subprocess vector envs.
- Command (resume + parallel):
  - `conda run -n intelligent_AJ python -m gac_mac.scripts.train --env lawn --device cuda --resume runs_collfocus\\mc6_L2_nlap02_conf05_argmax-20260109-183430\\checkpoints\\checkpoint_best_argmax.pth --parallel-env --num-envs 16 --steps-per-update 64 --total-updates 900 --checkpoint-interval 50 --log-interval 25 --eval-every 25 --eval-episodes 60 --eval-seed 123 --rollout-policy argmax --no-tensorboard`
- Outcome:
  - A slightly better best checkpoint was found and saved during training.
  - Long run showed instability after ~upd 550 (throughput collapsed); keep `checkpoint_best_argmax.pth` as the stable artifact.
- Updated best (argmax, 500 episodes):
  - `conda run -n intelligent_AJ python -m gac_mac.scripts.evaluate --checkpoint runs_collfocus\\mc6_L2_nlap02_conf05_argmax-20260109-183430\\checkpoints\\checkpoint_best_argmax.pth --episodes 500 --seed 123 --policy argmax --device cuda`
  - `GAC-MAC thr 110.374 Mbps | jain 0.953 | coll 0.489 | delay 10.893`

## 2026-01-09 (PPO 稳定性护栏 + 继续训练崩溃复现)

- 现象：`checkpoint_best_argmax.pth` 在评估时仍然稳定（~110 Mbps），但从该 ckpt 继续训练很容易在几十个 update 内把策略“训坏”，吞吐显著下降、碰撞率上升。
- 初步判断：优化过程不稳定（value/encoder 共享导致 critic 梯度主导、长训后出现灾难性遗忘），同时缺少 KL/clip 诊断与护栏让“坏更新”难以及时止损。
- 代码改动（用于诊断 + 护栏 + 可复现日志）：
  - PPO 更新加入 `approx_kl / clip_frac / grad_norm` 统计、`target_kl` early-stop、value clipping、BPTT 段落随机化：`gac_mac/algo/mappo.py`
  - 训练脚本加入 `train_log.jsonl`（每个 update + eval 写入），并支持 `--lr-anneal/--min-lr/--target-kl/--vf-clip-eps`：`gac_mac/scripts/train.py`
  - 新增可选 `--critic-detach-encoder` 与 `--value-loss-coef`（尝试缓解 critic 梯度破坏 actor）：`gac_mac/config.py`, `gac_mac/models/agent.py`, `gac_mac/scripts/train.py`
- 继续训练（失败示例，吞吐显著下降）：
  - Run（resume + 仅用于观察是否会再次崩溃）：
    - `conda run -n intelligent_AJ python -m gac_mac.scripts.train --env lawn --device cuda --resume runs_collfocus\\mc6_L2_nlap02_conf05_argmax-20260109-183430\\checkpoints\\checkpoint_best_argmax.pth --parallel-env --num-envs 16 --steps-per-update 64 --total-updates 525 --eval-every 25 --eval-episodes 30 --eval-seed 123 --rollout-policy argmax --no-tensorboard --lr 5e-5 --lr-anneal --min-lr 2e-5 --target-kl 0.02 --vf-clip-eps 0.2`
  - Eval during training (argmax, 30 eps):
    - `upd 450: thr 51.294 Mbps | jain 0.453 | coll 0.758`
    - `upd 475: thr 57.458 Mbps | jain 0.509 | coll 0.728`
    - `upd 500: thr 66.507 Mbps | jain 0.585 | coll 0.689`
    - `upd 525: thr 73.587 Mbps | jain 0.642 | coll 0.658`
- 结论：当前继续训练仍然会把已学到的好策略“训坏”；需要把“坏更新”机制性隔离（例如 actor/critic 分离、value loss 降权、或在吞吐退化时自动回滚到 best）。

## 2026-01-09 (关键发现：L=2 的高 coll 主要是“第二次占用”导致，直接把 L 限制为 1 即可接近零碰撞且吞吐几乎不变)

- 观察：`C=6,L=2` 的 best policy 在每步几乎打满尝试次数（~59 attempts/step），资源数为 `C*K=48`，必然出现大量资源重叠；碰撞计数几乎全部来自“第二次占用”，但对吞吐贡献很小。
- 对同一个 checkpoint 仅在评估时把 `--max-tx-per-frame` 从 2 改为 1（不改模型权重）：
  - `conda run -n intelligent_AJ python -m gac_mac.scripts.evaluate --checkpoint runs_collfocus\\mc6_L2_nlap02_conf05_argmax-20260109-183430\\checkpoints\\checkpoint_best_argmax.pth --episodes 200 --seed 123 --policy argmax --device cuda --max-tx-per-frame 1`
  - `GAC-MAC thr 109.518 Mbps | jain 0.962 | coll 0.000 | attempt 29.75/step | succ 29.75/step | coll_ct 0.00/step | delay 11.081`
- 对比原设置（同 checkpoint, L=2）：
  - `GAC-MAC thr 110.127 Mbps | jain 0.954 | coll 0.489 | attempt 58.89/step | succ 30.02/step | coll_ct 28.88/step | delay 10.910`
- 结论：当前训练出来的策略几乎不具备“占用多个资源但不碰撞”的能力；如果目标是同时最大化吞吐并显著降低碰撞，优先把 `L` 限制为 1（或设计“机会式第二次占用”机制，仅在存在空闲资源且不会与他人主占用冲突时才允许第二次占用）。

## 2026-01-09 (落地：机会式第二次占用 secondary-LBT，L=2 下 coll=0 且吞吐接近 Greedy 上界)

- 环境改动：新增 `secondary_lbt`（listen-before-talk gate）机制用于 `L>1` 时的 secondary picks：
  - primary（第 1 个动作）照常尝试；
  - secondary（第 2..L 个动作）只有在对应资源未被任何 primary 占用、且该资源未被其他 secondary 抢占时才允许发送（全局随机 LBT 顺序），从机制上避免 secondary-secondary 与 secondary-primary 的碰撞。
  - 相关代码：`gac_mac/env/lawn_env.py`, `gac_mac/config.py`, `gac_mac/env/vector_env.py`, `gac_mac/scripts/train.py`, `gac_mac/scripts/evaluate.py`
- 同一 checkpoint（不改权重）在 `C=6,L=2` 下开启 `secondary_lbt` 的评估结果（argmax, 200 eps）：
  - `conda run -n intelligent_AJ python -m gac_mac.scripts.evaluate --checkpoint runs_collfocus\\mc6_L2_nlap02_conf05_argmax-20260109-183430\\checkpoints\\checkpoint_best_argmax.pth --episodes 200 --seed 123 --policy argmax --device cuda --secondary-lbt`
  - `GAC-MAC thr 128.520 Mbps | jain 0.877 | coll 0.000 | attempt 34.99/step | succ 34.99/step | coll_ct 0.00/step`
  - 同口径 baselines（由 evaluate 输出）：`Greedy thr 128.657 Mbps | coll 0.065`, `Random thr 108.116 Mbps | coll 0.243`, `CSMA thr 42.464 Mbps | coll 0.046`
- 结论：在不增加总带宽的前提下，secondary-LBT 让“多资源占用”真正变成可用的防碰撞自由度；当前策略在吞吐上已逼近 Greedy 上界，同时把碰撞压到 0。

## 2026-01-10 (验证 coll=0 是否“测试有问题”：加入资源占用诊断 + 加大节点密度)

- 为排除“coll=0 是统计口径/代码 bug”的可能，新增诊断指标：
  - `max_group_size`：单个 (channel,slot) 资源上同时发送的最大节点数
  - `num_multi_tx_resources`：每步发生并发(>1)发送的资源数量
  - 相关代码：`gac_mac/env/lawn_env.py`, `gac_mac/scripts/evaluate.py`
- 低密度（N=30）下 coll=0 是“真实无并发”导致，而不是统计错误（`max_group_size=1`，并发资源数为 0）：
  - `conda run -n intelligent_AJ python -m gac_mac.scripts.evaluate --checkpoint runs_collfocus\\mc6_L2_nlap02_conf05_argmax-20260109-183430\\checkpoints\\checkpoint_best_argmax.pth --episodes 100 --seed 123 --policy argmax --device cuda --secondary-lbt`
  - `GAC-MAC thr 128.558 Mbps | coll 0.000 | max_grp 1.00 | multi_res 0.00/step`
- 提高密度（N=60）后，coll 不再为 0（secondary-LBT 只保护 secondary，占用冲突主要来自 primary 同资源并发）：
  - `conda run -n intelligent_AJ python -m gac_mac.scripts.evaluate --checkpoint runs_collfocus\\mc6_L2_nlap02_conf05_argmax-20260109-183430\\checkpoints\\checkpoint_best_argmax.pth --episodes 100 --seed 123 --policy argmax --device cuda --secondary-lbt --num-uavs 60`
  - `GAC-MAC thr 166.994 Mbps | coll 0.212 | max_grp 7.93 | multi_res 6.69/step`
  - `--map-size-m 700`（更高密度）结果同量级（coll≈0.212，max_grp 更大）。
- 进一步“把 primary 也做 LBT 竞争解析”（`primary_lbt`）可以把 coll 归零，但这会让所有策略都变成“无并发同资源发送”（coll 对比不再有区分度）：
  - `conda run -n intelligent_AJ python -m gac_mac.scripts.evaluate --checkpoint runs_collfocus\\mc6_L2_nlap02_conf05_argmax-20260109-183430\\checkpoints\\checkpoint_best_argmax.pth --episodes 100 --seed 123 --policy argmax --device cuda --secondary-lbt --primary-lbt --num-uavs 60`
  - 现象：所有策略 `max_grp=1.00`，`coll=0.000`（由 MAC 执行阶段的仲裁机制保证）。

## 2026-01-10 (Baseline 折线对比：性能 vs 节点密度相关性，无 primary_lbt)

- 需求：对比 GAC(secondary_lbt) / Random / Greedy / CSMA 在不同节点密度下的吞吐、碰撞、Jain 折线关系（不启用 primary_lbt）。
- 实现：
  - 新增脚本 `gac_mac/scripts/benchmark_density.py`：扫 `N` 并计算 density（nodes/km^2），输出 `density_lines.png` + `density_lines.json`
  - 新增绘图函数 `plot_density_lines()`：`gac_mac/viz/plots.py`
- 运行命令（C=6,L=2，secondary_lbt=ON，argmax）：
  - `conda run -n intelligent_AJ python -m gac_mac.scripts.benchmark_density --checkpoint runs_collfocus\\mc6_L2_nlap02_conf05_argmax-20260109-183430\\checkpoints\\checkpoint_best_argmax.pth --ns "20,30,40,50,60,80" --episodes 50 --seed 123 --policy argmax --device cuda --secondary-lbt --run-name mc6L2_secondaryLBT --results-dir runs_density`
- 输出：
  - 图：`runs_density\\mc6L2_secondaryLBT-20260110-155625\\density_lines.png`
  - 数据：`runs_density\\mc6L2_secondaryLBT-20260110-155625\\density_lines.json`
## 2026-01-09 (Greedy 模仿学习预训练：C=6,L=2 失败尝试)

- 目的：用 Greedy teacher 先把确定性策略拉到可用区，再用 PPO 微调（避免从随机策略开始的塌缩）。
- 修复：Greedy pretrain 支持 `L>1` 的顺序多资源动作（NoTx 早停 + 去重 mask），并确保 teacher 调用传入 `max_tx`：`gac_mac/scripts/train.py`
- Pretrain-only：
  - `conda run -n intelligent_AJ python -m gac_mac.scripts.train --env lawn --device cuda --results-dir runs_il --run-name mc6_L2_greedy_pretrain5k --num-channels 6 --max-tx-per-frame 2 --num-envs 8 --steps-per-update 64 --pretrain-greedy-steps 5000 --pretrain-only --pretrain-no-tx-weight 0.1 --no-tensorboard`
  - ckpt: `runs_il\\mc6_L2_greedy_pretrain5k-20260109-223710\\checkpoints\\checkpoint_pretrain.pth`
- Eval（argmax, 200 eps）：
  - `conda run -n intelligent_AJ python -m gac_mac.scripts.evaluate --checkpoint runs_il\\mc6_L2_greedy_pretrain5k-20260109-223710\\checkpoints\\checkpoint_pretrain.pth --episodes 200 --seed 123 --policy argmax --device cuda`
  - `GAC-MAC thr 1.533 Mbps | jain 0.020 | coll 0.988 | delay 14.137`
  - 同口径 baselines（由 evaluate 输出）：`Greedy thr 117.909 Mbps | coll 0.104`, `Random thr 75.491 Mbps | coll 0.558`, `CSMA thr 33.731 Mbps | coll 0.123`
- 结论：纯 CE 模仿在当前“分布式可观测”条件下很难拟合集中式 Greedy（teacher 依赖全局冲突信息），需要更强的监督信号（例如 teacher 的局部可见版本、或把 imitation 改为 ranking/energy-based loss）。
