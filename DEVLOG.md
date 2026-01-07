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

## 2026-01-07 (Branch exp/static-shaping-strong)

- Goal: (1) fix hierarchical entropy regularization to avoid No-Tx collapse; (2) add a static “try-transmit” shaping term (no curriculum).
- Changes:
  - Fix hierarchical entropy to use expected entropy: `H = H(tx) + p_tx * H(slot)` (instead of conditioning on sampled/selected Tx) (`gac_mac/models/agent.py`).
  - Add `Config.reward_tx_attempt` and apply it to any transmission attempt with backlog (`gac_mac/env/lawn_env.py`, `gac_mac/env/toy_env.py` + wiring).
- Reference run:
  - `results/A_static_shaping_strong-20260107-230434/checkpoints/checkpoint_best.pth`
  - `evaluate --policy sample --temperature 0.7` (10 seeds): `thr 8.310 Mbps | coll 0.619 | jain 0.850`
