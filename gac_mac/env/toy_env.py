from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GraphObs:
    x: "np.ndarray"  # (N, F)
    edge_index: "np.ndarray"  # (2, E), int64, [src, dst]
    edge_attr: "np.ndarray"  # (E, 1), float32


class ToyLawnEnv:
    """A minimal multi-agent environment to validate the training pipeline (M1).

    - Directed interference graph from distance threshold (carrier-sensing proxy)
    - Poisson arrivals -> per-node queues -> per-packet delay tracking
    - Success/collision uses same-slot interference at receiver (ring pairing)
    - Reward includes strict neighbor cooperation term over OUT-neighbors
    """

    STATUS_SUCCESS = 0
    STATUS_COLLISION = 1
    STATUS_IDLE = 2

    def __init__(
        self,
        *,
        num_uavs: int,
        num_slots: int,
        map_size_m: float,
        height_m: float,
        neighbor_radius_m: float,
        max_queue_len: int,
        lambda_arrival_per_slot: float,
        episode_len: int,
        reward_success: float,
        reward_collision: float,
        reward_idle_empty: float,
        reward_idle_nonempty: float,
        lambda_coop: float,
    ) -> None:
        self.N = int(num_uavs)
        self.K = int(num_slots)
        self.map_size_m = float(map_size_m)
        self.height_m = float(height_m)
        self.neighbor_radius_m = float(neighbor_radius_m)
        self.max_queue_len = int(max_queue_len)
        self.lambda_arrival_per_slot = float(lambda_arrival_per_slot)
        self.episode_len = int(episode_len)

        self.reward_success = float(reward_success)
        self.reward_collision = float(reward_collision)
        self.reward_idle_empty = float(reward_idle_empty)
        self.reward_idle_nonempty = float(reward_idle_nonempty)
        self.lambda_coop = float(lambda_coop)

        self._rng = np.random.default_rng()
        self._t = 0

        # State
        self.positions = np.zeros((self.N, 3), dtype=np.float32)
        self.queues = np.zeros((self.N,), dtype=np.int32)
        self.last_actions = np.full((self.N,), fill_value=self.K, dtype=np.int64)
        self.last_status = np.full((self.N,), fill_value=self.STATUS_IDLE, dtype=np.int64)

        # Packet age tracking for delay: list-of-lists with small caps (M1 OK)
        self._packet_ages: list[list[int]] = [[] for _ in range(self.N)]

    def reset(self, *, seed: int | None = None) -> GraphObs:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0

        # Random 3D positions
        xy = self._rng.uniform(0.0, self.map_size_m, size=(self.N, 2))
        z = self._rng.uniform(0.25 * self.height_m, self.height_m, size=(self.N, 1))
        self.positions = np.concatenate([xy, z], axis=1).astype(np.float32)

        self.queues.fill(0)
        self.last_actions.fill(self.K)
        self.last_status.fill(self.STATUS_IDLE)
        self._packet_ages = [[] for _ in range(self.N)]

        return self._get_obs()

    def step(self, actions: np.ndarray) -> tuple[GraphObs, np.ndarray, bool, dict[str, Any]]:
        self._t += 1
        actions = np.asarray(actions, dtype=np.int64).reshape(self.N)
        actions = np.clip(actions, 0, self.K)  # K means No-Tx

        # --- Traffic: Poisson arrivals per frame (lambda per slot * K slots) ---
        lam_per_frame = self.lambda_arrival_per_slot * self.K
        arrivals = self._rng.poisson(lam=lam_per_frame, size=self.N).astype(np.int32)
        if self.max_queue_len > 0:
            for i in range(self.N):
                room = self.max_queue_len - self.queues[i]
                add = int(min(room, arrivals[i]))
                if add > 0:
                    self._packet_ages[i].extend([0] * add)
                    self.queues[i] += add

        # Age all queued packets by 1 frame
        for i in range(self.N):
            if self._packet_ages[i]:
                self._packet_ages[i] = [age + 1 for age in self._packet_ages[i]]

        # --- Build directed interference graph A (N,N) with A[j,i]=1 meaning j->i ---
        adj = self._build_adj_matrix()

        # Determine which nodes actually transmit (must have packet and choose slot)
        has_pkt = self.queues > 0
        tx_mask = has_pkt & (actions < self.K)

        # Receiver for transmitter i is (i+1) % N
        receivers = (np.arange(self.N, dtype=np.int64) + 1) % self.N

        # Collision rule: at receiver r, if any other tx j in same slot with edge j->r, it's collision
        collision = np.zeros((self.N,), dtype=bool)
        for i in range(self.N):
            if not tx_mask[i]:
                continue
            r = int(receivers[i])
            slot_i = int(actions[i])
            interferers = np.where(adj[:, r] == 1)[0]
            if interferers.size == 0:
                continue
            same_slot = (actions[interferers] == slot_i) & tx_mask[interferers]
            # Exclude self if present
            if i in interferers:
                self_idx = np.where(interferers == i)[0]
                if self_idx.size > 0:
                    same_slot[self_idx[0]] = False
            if np.any(same_slot):
                collision[i] = True

        success = tx_mask & (~collision)

        # Serve one packet on success and record delay
        per_step_delays: list[int] = []
        for i in np.where(success)[0]:
            if self._packet_ages[i]:
                per_step_delays.append(self._packet_ages[i].pop(0))
            self.queues[i] -= 1

        # --- Reward decomposition ---
        r_perf = np.zeros((self.N,), dtype=np.float32)
        r_pen = np.zeros((self.N,), dtype=np.float32)

        r_perf[success] = self.reward_success

        idle = ~tx_mask
        idle_empty = idle & (~has_pkt)
        idle_nonempty = idle & has_pkt

        r_perf[idle_empty] = self.reward_idle_empty
        r_pen[collision] = self.reward_collision
        r_pen[idle_nonempty] = self.reward_idle_nonempty

        # Cooperative term over IN-neighbors: N_i = { j | j -> i }
        in_deg = adj.sum(axis=0).astype(np.float32)
        coop_raw = adj.T @ r_perf  # (N,)
        r_coop = np.zeros((self.N,), dtype=np.float32)
        nonzero = in_deg > 0
        r_coop[nonzero] = self.lambda_coop * (coop_raw[nonzero] / in_deg[nonzero])

        rewards = r_perf + r_pen + r_coop

        # Update last-action/status: only record actual transmission attempt (or No-Tx).
        effective_actions = actions.copy()
        effective_actions[~tx_mask] = self.K
        self.last_actions = effective_actions
        self.last_status = np.full((self.N,), self.STATUS_IDLE, dtype=np.int64)
        self.last_status[success] = self.STATUS_SUCCESS
        self.last_status[collision] = self.STATUS_COLLISION

        done = self._t >= self.episode_len
        obs = self._get_obs()

        attempts = int(tx_mask.sum())
        collisions = int(collision.sum())
        info: dict[str, Any] = {
            "t": self._t,
            "sum_rate": float(success.sum()),  # toy proxy: #successful links per frame
            "collision_rate": float(collisions / max(1, attempts)),
            "avg_delay": float(np.mean(per_step_delays)) if per_step_delays else 0.0,
            "tx_attempts": attempts,
            "tx_success": int(success.sum()),
            "tx_collision": collisions,
            "avg_degree_in": float(adj.sum(axis=0).mean()),
            "avg_degree_out": float(adj.sum(axis=1).mean()),
        }
        return obs, rewards, done, info

    # --- Internals ---
    def _build_adj_matrix(self) -> np.ndarray:
        # Distance-based directed edges (symmetric by construction in toy env, but kept directed)
        diffs = self.positions[:, None, :] - self.positions[None, :, :]
        dist = np.linalg.norm(diffs, axis=2)  # (N,N)
        adj = (dist < self.neighbor_radius_m).astype(np.int8)
        np.fill_diagonal(adj, 0)
        # Interpret as j->i by using rows as sources, cols as targets (already so)
        return adj

    def _get_obs(self) -> GraphObs:
        adj = self._build_adj_matrix()

        # Feature normalization
        pos = self.positions.copy()
        pos[:, 0:2] = pos[:, 0:2] / max(1e-6, self.map_size_m)
        pos[:, 2] = pos[:, 2] / max(1e-6, self.height_m)

        q = (self.queues.astype(np.float32) / max(1, self.max_queue_len)).reshape(self.N, 1)
        energy = np.ones((self.N, 1), dtype=np.float32)  # constant placeholder

        last_act_oh = np.eye(self.K + 1, dtype=np.float32)[self.last_actions]  # (N, K+1)
        last_status_oh = np.eye(3, dtype=np.float32)[self.last_status]  # (N, 3)

        in_deg = adj.sum(axis=0, keepdims=False).astype(np.float32).reshape(self.N, 1)
        in_deg_norm = in_deg / float(self.N)

        x = np.concatenate(
            [pos.astype(np.float32), q, energy, last_act_oh, last_status_oh, in_deg_norm], axis=1
        )
        rows, cols = np.where(adj == 1)
        edge_index = np.stack([rows, cols], axis=0).astype(np.int64)
        # Simple edge weight: normalized inverse distance (local), clipped to [0,1].
        if rows.size == 0:
            edge_attr = np.zeros((0, 1), dtype=np.float32)
        else:
            diffs = self.positions[rows] - self.positions[cols]
            dist = np.linalg.norm(diffs, axis=1).astype(np.float32)
            dist = np.maximum(dist, 1e-3)
            inv = 1.0 / dist
            inv = inv / np.max(inv)
            edge_attr = inv.reshape(-1, 1).astype(np.float32)
        return GraphObs(x=x, edge_index=edge_index, edge_attr=edge_attr)
