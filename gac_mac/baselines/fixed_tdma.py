from __future__ import annotations

import numpy as np


class FixedTDMAAgent:
    """Fixed TDMA schedule baseline.

    Deterministically assigns each node a resource based on its index:
    resource = (channel, slot) where:
    - slot = i % K
    - channel = (i // K) % C

    If queue empty -> NoTx. For L>1, extra picks are left as NoTx.
    """

    def __init__(self, num_slots: int, num_nodes: int, *, num_channels: int = 1) -> None:
        self.K = int(num_slots)
        self.C = int(max(1, num_channels))
        self.A = int(self.C * self.K + 1)
        self.N = int(num_nodes)

    def select_actions(self, obs_x: np.ndarray, rng: np.random.Generator, *, max_tx: int = 1) -> np.ndarray:
        _ = rng  # deterministic
        q_norm = obs_x[:, 3]
        has_pkt = q_norm > 0.0

        no_tx = self.A - 1
        l = int(max(1, max_tx))
        actions = np.full((self.N, l), fill_value=no_tx, dtype=np.int64)

        idx = np.where(has_pkt)[0].astype(int)
        if idx.size == 0:
            return actions[:, 0] if l == 1 else actions

        for i in idx:
            slot = int(i % self.K)
            ch = int((i // self.K) % self.C)
            actions[i, 0] = int(ch * self.K + slot)

        return actions[:, 0] if l == 1 else actions

