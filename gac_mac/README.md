# GAC-MAC (Standalone Implementation)

This folder is a clean, standalone implementation of the project described in `requirement.md`.
It does **not** depend on the legacy code under `3.0/`.

## M1 (current): Toy environment smoke test

`gac_mac/env/toy_env.py` is a minimal multi-agent environment used to validate:
- data/shape plumbing (N agents, K slots),
- directed interference graph,
- Poisson arrivals + queues,
- reward shaping with neighbor cooperation,
- GATv2 + GRU policy and MAPPO training loop,
- checkpointing / resume.

## Run

From repo root:

- Toy smoke test: `python -m gac_mac.scripts.train --env toy`
- Full LAWN env (M2+): `python -m gac_mac.scripts.train --env lawn`
- Resume: `python -m gac_mac.scripts.train --resume <path-to-checkpoint.pth>`

## TensorBoard (实时监视训练)

训练会默认写入 TensorBoard 日志到每个 run 目录下的 `tb/`：
- 启动训练后，在另一个终端运行：`tensorboard --logdir results --port 6006`
- 浏览器打开：`http://localhost:6006`
- 若要关闭 TB 记录：训练命令加 `--no-tensorboard`

## Evaluate (LAWN env)

- `python -m gac_mac.scripts.evaluate --checkpoint <checkpoint.pth> --episodes 20 --policy sample --out eval.png`

## Visualization

- Plot curves from checkpoint: `python -m gac_mac.scripts.plot_training --checkpoint <checkpoint.pth>`
- Topology snapshot: `python -m gac_mac.scripts.snapshot_topology --checkpoint <checkpoint.pth> --seed 123 --step 0 --policy sample --out topology.png`

## Dependencies

You will need (at minimum):
- Python 3.10+
- `numpy`
- `torch`
- `torch_geometric` (PyTorch Geometric)

Later milestones add plotting (`matplotlib`, `networkx`) and comparison baselines.
