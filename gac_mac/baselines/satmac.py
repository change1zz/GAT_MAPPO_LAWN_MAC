from __future__ import annotations

import numpy as np


class SATMACAgent:
    """A simple self-adaptive TDMA baseline inspired by SATMAC-style behavior.

    Each node keeps a preferred resource (channel,slot):
    - If last frame was SUCCESS: keep the resource.
    - If last frame was COLLISION: reselect a new random resource with prob p_reselect (else keep).
    - If last frame was IDLE and queue non-empty: try the preferred resource.
    - If queue empty: NoTx.

    This is a lightweight distributed heuristic using only ACK feedback.
    """

    STATUS_SUCCESS = 0
    STATUS_COLLISION = 1
    STATUS_IDLE = 2

    def __init__(
        self,
        num_slots: int,
        num_nodes: int,
        *,
        num_channels: int = 1,
        p_reselect: float = 1.0,
    ) -> None:
        self.K = int(num_slots)
        self.C = int(max(1, num_channels))
        self.A = int(self.C * self.K + 1)
        self.N = int(num_nodes)
        self.p_reselect = float(p_reselect)
        self.pref = np.zeros((self.N,), dtype=np.int64)

    def reset(self, rng: np.random.Generator | None = None) -> None:
        if rng is None:
            rng = np.random.default_rng()
        self.pref = rng.integers(low=0, high=(self.A - 1), size=self.N, dtype=np.int64)

    def select_actions(self, obs_x: np.ndarray, last_status: np.ndarray, rng: np.random.Generator, *, max_tx: int = 1) -> np.ndarray:
        q_norm = obs_x[:, 3]
        has_pkt = q_norm > 0.0

        # lazy init
        if self.pref.shape[0] != int(obs_x.shape[0]):
            self.N = int(obs_x.shape[0])
            self.pref = rng.integers(low=0, high=(self.A - 1), size=self.N, dtype=np.int64)

        collided = last_status == self.STATUS_COLLISION
        if np.any(collided):
            reseed = rng.random(self.N) < float(np.clip(self.p_reselect, 0.0, 1.0))
            upd = collided & reseed
            if np.any(upd):
                self.pref[upd] = rng.integers(low=0, high=(self.A - 1), size=int(upd.sum()), dtype=np.int64)

        no_tx = self.A - 1
        l = int(max(1, max_tx))
        actions = np.full((self.N, l), fill_value=no_tx, dtype=np.int64)

        idx = np.where(has_pkt)[0].astype(int)
        if idx.size == 0:
            return actions[:, 0] if l == 1 else actions

        actions[idx, 0] = self.pref[idx]

        if l > 1:
            # Optional extra picks: sample unique additional resources (excluding the primary pick).
            k = int(min(l - 1, self.A - 2))
            if k > 0:
                for i in idx:
                    # avoid repeating primary
                    candidates = np.arange(self.A - 1, dtype=np.int64)
                    candidates = candidates[candidates != self.pref[i]]
                    picks = rng.choice(candidates, size=k, replace=False).astype(np.int64)
                    actions[i, 1 : 1 + k] = picks

        return actions[:, 0] if l == 1 else actions

