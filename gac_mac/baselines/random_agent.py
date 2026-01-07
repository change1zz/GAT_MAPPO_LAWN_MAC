from __future__ import annotations

import numpy as np


class RandomAgent:
    def __init__(self, num_slots: int) -> None:
        self.K = int(num_slots)

    def select_actions(self, obs_x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n = int(obs_x.shape[0])
        # If queue is empty -> No-Tx. Queue feature is at index 3 (pos[3] + q[1]).
        q_norm = obs_x[:, 3]
        has_pkt = q_norm > 0.0

        actions = np.full((n,), fill_value=self.K, dtype=np.int64)
        actions[has_pkt] = rng.integers(low=0, high=self.K + 1, size=int(has_pkt.sum()), dtype=np.int64)
        return actions

