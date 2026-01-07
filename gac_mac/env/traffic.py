from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class TrafficStepInfo:
    arrivals: np.ndarray  # (N,) int
    accepted: np.ndarray  # (N,) int
    dropped: np.ndarray  # (N,) int


class PoissonTraffic:
    """Poisson arrivals with per-packet delay tracking via per-node deques."""

    def __init__(self, *, num_nodes: int, num_slots: int, lambda_arrival_per_slot: float, max_queue_len: int) -> None:
        self.N = int(num_nodes)
        self.K = int(num_slots)
        self.lambda_arrival_per_slot = float(lambda_arrival_per_slot)
        self.max_queue_len = int(max_queue_len)

        self.queues = np.zeros((self.N,), dtype=np.int32)
        self._arrival_times: list[deque[int]] = [deque() for _ in range(self.N)]

    def reset(self) -> None:
        self.queues.fill(0)
        self._arrival_times = [deque() for _ in range(self.N)]

    def step(self, *, t: int, rng: np.random.Generator) -> TrafficStepInfo:
        lam_per_frame = self.lambda_arrival_per_slot * self.K
        arrivals = rng.poisson(lam=lam_per_frame, size=self.N).astype(np.int32)

        accepted = np.zeros_like(arrivals)
        dropped = np.zeros_like(arrivals)

        if self.max_queue_len <= 0:
            # Unlimited queue (not recommended); still track arrivals.
            accepted = arrivals.copy()
            self.queues += accepted
            for i in range(self.N):
                if accepted[i] > 0:
                    self._arrival_times[i].extend([t] * int(accepted[i]))
            return TrafficStepInfo(arrivals=arrivals, accepted=accepted, dropped=dropped)

        for i in range(self.N):
            room = self.max_queue_len - int(self.queues[i])
            add = int(min(room, int(arrivals[i])))
            drop = int(arrivals[i]) - add
            accepted[i] = add
            dropped[i] = drop
            if add > 0:
                self._arrival_times[i].extend([t] * add)
                self.queues[i] += add

        return TrafficStepInfo(arrivals=arrivals, accepted=accepted, dropped=dropped)

    def serve_successes(self, *, success_mask: np.ndarray, t: int) -> list[int]:
        """Serve at most 1 packet per successful transmission, return list of delays."""
        delays: list[int] = []
        idxs = np.where(success_mask)[0]
        for i in idxs:
            if self.queues[i] <= 0:
                continue
            if not self._arrival_times[i]:
                # Shouldn't happen, but keep robust.
                self.queues[i] -= 1
                continue
            arrival_t = self._arrival_times[i].popleft()
            delays.append(int(t - arrival_t))
            self.queues[i] -= 1
        return delays

    def serve_successes_with_per_node_delay(self, *, success_mask: np.ndarray, t: int) -> tuple[list[int], np.ndarray]:
        """Like serve_successes(), but also returns per-node served delay (NaN if not served)."""
        delays: list[int] = []
        per_node = np.full((self.N,), np.nan, dtype=np.float32)
        idxs = np.where(success_mask)[0]
        for i in idxs:
            if self.queues[i] <= 0:
                continue
            if not self._arrival_times[i]:
                self.queues[i] -= 1
                continue
            arrival_t = self._arrival_times[i].popleft()
            d = int(t - arrival_t)
            delays.append(d)
            per_node[i] = float(d)
            self.queues[i] -= 1
        return delays, per_node
