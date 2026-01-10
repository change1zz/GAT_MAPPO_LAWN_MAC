from __future__ import annotations

import numpy as np


class GreedyColoringAgent:
    def __init__(self, num_slots: int, *, num_channels: int = 1) -> None:
        self.K = int(num_slots)
        self.C = int(max(1, num_channels))
        self.A = int(self.C * self.K + 1)  # flattened (ch,slot) + NoTx

    def select_actions(self, *, env, obs_x: np.ndarray, max_tx: int = 1) -> np.ndarray:
        n = int(obs_x.shape[0])
        q_norm = obs_x[:, 3]
        active = q_norm > 0.0

        # Centralized greedy coloring on a link-conflict graph derived from the current large-scale channel.
        #
        # Two links (i and j) conflict if *either* link's SINR would fall below threshold under
        # pairwise simultaneous transmission (using large-scale gain as a proxy).
        receivers = (np.arange(n, dtype=np.int64) + 1) % n
        gain_ls = env._ch.large_scale_gain  # (N,N) src->node (already sampled for current state)
        p_tx = float(env.p_tx_w)
        noise = float(env.noise_w)
        thr = float(env.sinr_threshold_lin)

        signal = p_tx * gain_ls[np.arange(n), receivers]  # (N,)
        inter = p_tx * gain_ls[:, receivers]  # (N,N) row=j, col=i : power from j at rx(i)
        sinr_with_j = signal.reshape(1, n) / (inter + noise)  # (N,N)

        conflict = (sinr_with_j < thr) | (sinr_with_j.T < thr)
        np.fill_diagonal(conflict, False)

        degrees = conflict.sum(axis=1)
        order = np.argsort(-degrees)  # high degree first

        num_tx = self.A - 1
        l = int(max(1, max_tx))
        no_tx = self.A - 1

        # Multi-allocation heuristic: repeat coloring rounds, each node can grab up to L resources.
        alloc = np.full((n, l), fill_value=no_tx, dtype=np.int64)
        alloc_count = np.zeros((n,), dtype=np.int32)
        used_by_node: list[set[int]] = [set() for _ in range(n)]

        for _round in range(l):
            for i0 in order:
                i = int(i0)
                if not active[i]:
                    continue
                if alloc_count[i] >= l:
                    continue

                neigh = np.where(conflict[i])[0]
                forbidden: set[int] = set()
                for j0 in neigh:
                    j = int(j0)
                    forbidden |= used_by_node[j]

                chosen = None
                for s in range(num_tx):
                    if s in forbidden or s in used_by_node[i]:
                        continue
                    chosen = s
                    break
                if chosen is None:
                    continue
                used_by_node[i].add(int(chosen))
                alloc[i, alloc_count[i]] = int(chosen)
                alloc_count[i] += 1

        return alloc[:, 0] if l == 1 else alloc
