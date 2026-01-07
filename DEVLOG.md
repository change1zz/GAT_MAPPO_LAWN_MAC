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
