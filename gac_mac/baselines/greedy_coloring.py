from __future__ import annotations

import numpy as np


class GreedyColoringAgent:
    def __init__(self, num_slots: int) -> None:
        self.K = int(num_slots)

    def select_actions(self, *, env, obs_x: np.ndarray) -> np.ndarray:
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

        actions = np.full((n,), fill_value=self.K, dtype=np.int64)
        for i in order:
            i = int(i)
            if not active[i]:
                continue
            # +1 to tolerate any accidental No-Tx (==K) bookkeeping without crashing.
            used = np.zeros((self.K + 1,), dtype=np.bool_)
            neigh = np.where(conflict[i])[0]
            for j in neigh:
                j = int(j)
                a = int(actions[j])
                if 0 <= a <= self.K:
                    used[a] = True
            # pick smallest available slot
            chosen = None
            for s in range(self.K):
                if not bool(used[s]):
                    chosen = s
                    break
            actions[i] = self.K if chosen is None else int(chosen)
        return actions
