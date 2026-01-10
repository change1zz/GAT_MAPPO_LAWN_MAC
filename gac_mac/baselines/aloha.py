from __future__ import annotations

import numpy as np


class AlohaAgent:
    """Slotted ALOHA baseline (frame-level).

    If queue non-empty:
    - transmit with probability p_tx
    - if transmit, pick a random (channel,slot) resource (or up to L unique resources)
    - else NoTx
    """

    def __init__(self, num_slots: int, *, num_channels: int = 1, p_tx: float = 0.2) -> None:
        self.K = int(num_slots)
        self.C = int(max(1, num_channels))
        self.A = int(self.C * self.K + 1)  # flattened (ch,slot) + NoTx
        self.p_tx = float(p_tx)

    def select_actions(self, obs_x: np.ndarray, rng: np.random.Generator, *, max_tx: int = 1) -> np.ndarray:
        n = int(obs_x.shape[0])
        q_norm = obs_x[:, 3]
        has_pkt = q_norm > 0.0

        no_tx = self.A - 1
        l = int(max(1, max_tx))
        actions = np.full((n, l), fill_value=no_tx, dtype=np.int64)

        idx = np.where(has_pkt)[0]
        if idx.size == 0:
            return actions[:, 0] if l == 1 else actions

        attempt = rng.random(idx.size) < float(np.clip(self.p_tx, 0.0, 1.0))
        tx_idx = idx[attempt].astype(int)
        if tx_idx.size == 0:
            return actions[:, 0] if l == 1 else actions

        if l == 1:
            actions[tx_idx, 0] = rng.integers(low=0, high=(self.A - 1), size=tx_idx.size, dtype=np.int64)
            return actions[:, 0]

        k = int(min(l, self.A - 1))
        for i in tx_idx:
            picks = rng.choice(self.A - 1, size=k, replace=False).astype(np.int64)
            actions[i, :k] = picks
        return actions

