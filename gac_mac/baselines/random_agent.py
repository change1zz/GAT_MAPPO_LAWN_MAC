from __future__ import annotations

import numpy as np


class RandomAgent:
    def __init__(self, num_slots: int, *, num_channels: int = 1) -> None:
        self.K = int(num_slots)
        self.C = int(max(1, num_channels))
        self.A = int(self.C * self.K + 1)  # flattened (ch,slot) + NoTx

    def select_actions(self, obs_x: np.ndarray, rng: np.random.Generator, *, max_tx: int = 1) -> np.ndarray:
        n = int(obs_x.shape[0])
        # If queue is empty -> No-Tx. Queue feature is at index 3 (pos[3] + q[1]).
        q_norm = obs_x[:, 3]
        has_pkt = q_norm > 0.0

        no_tx = self.A - 1
        l = int(max(1, max_tx))
        actions = np.full((n, l), fill_value=no_tx, dtype=np.int64)
        idx = np.where(has_pkt)[0]
        if idx.size == 0:
            return actions[:, 0] if l == 1 else actions
        for i in idx.astype(int):
            if l == 1:
                actions[i, 0] = int(rng.integers(low=0, high=self.A, dtype=np.int64))
            else:
                # Sample unique tx resources, allow early stop with NoTx.
                k = int(min(l, self.A - 1))
                picks = rng.choice(self.A - 1, size=k, replace=False).astype(np.int64)
                actions[i, :k] = picks
        return actions[:, 0] if l == 1 else actions
