from __future__ import annotations

import numpy as np


class CSMAAgent:
    """A lightweight CSMA/CA-like baseline (frame-level approximation).

    Each agent maintains a contention window (CW). Per frame:
    - if queue empty -> No-Tx
    - else attempt with probability 1/CW:
        - attempt: pick a random slot in [0, K-1]
        - no attempt: No-Tx
    CW update:
    - collision: CW = min(CW*2, cw_max)
    - success: CW = cw_min
    - idle: CW unchanged
    """

    STATUS_SUCCESS = 0
    STATUS_COLLISION = 1
    STATUS_IDLE = 2

    def __init__(self, num_slots: int, num_nodes: int, *, cw_min: int = 4, cw_max: int = 64) -> None:
        self.K = int(num_slots)
        self.N = int(num_nodes)
        self.cw_min = int(cw_min)
        self.cw_max = int(cw_max)
        self.cw = np.full((self.N,), fill_value=self.cw_min, dtype=np.int32)

    def reset(self) -> None:
        self.cw.fill(self.cw_min)

    def select_actions(self, obs_x: np.ndarray, last_status: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        q_norm = obs_x[:, 3]
        has_pkt = q_norm > 0.0

        # Update CW from last status
        collided = last_status == self.STATUS_COLLISION
        succeeded = last_status == self.STATUS_SUCCESS
        self.cw[collided] = np.minimum(self.cw[collided] * 2, self.cw_max)
        self.cw[succeeded] = self.cw_min

        actions = np.full((self.N,), fill_value=self.K, dtype=np.int64)
        probs = 1.0 / np.maximum(self.cw.astype(np.float32), 1.0)
        attempt = (rng.random(self.N) < probs) & has_pkt
        n_attempt = int(attempt.sum())
        if n_attempt > 0:
            actions[attempt] = rng.integers(low=0, high=self.K, size=n_attempt, dtype=np.int64)
        return actions

