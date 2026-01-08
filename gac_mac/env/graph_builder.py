from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GraphObs:
    x: np.ndarray  # (N, F)
    edge_index: np.ndarray  # (2, E) int64, source->target
    edge_attr: np.ndarray  # (E, D) float32
    adj: np.ndarray  # (N, N) uint8, adj[src, dst]


class GraphBuilder:
    def __init__(
        self,
        *,
        num_nodes: int,
        num_slots: int,
        map_size_m: float,
        height_m: float,
        max_queue_len: int,
        cs_threshold_dbm: float,
        graph_mode: str = "cs",
        sinr_threshold_db: float = 10.0,
        obs_version: str = "v2",
    ) -> None:
        self.N = int(num_nodes)
        self.K = int(num_slots)
        self.map_size_m = float(map_size_m)
        self.height_m = float(height_m)
        self.max_queue_len = int(max_queue_len)
        self.cs_threshold_dbm = float(cs_threshold_dbm)
        self.graph_mode = str(graph_mode)
        self.sinr_threshold_db = float(sinr_threshold_db)
        self.obs_version = str(obs_version)

    def build(
        self,
        *,
        positions_m: np.ndarray,  # (N,3)
        queues: np.ndarray,  # (N,)
        energy_norm: np.ndarray | None,  # (N,) or None
        last_actions: np.ndarray,  # (N,) in [0..K] (K=no-tx)
        last_status: np.ndarray,  # (N,) 0/1/2 success/collision/idle
        rx_power_dbm: np.ndarray,  # (N,N) power at "effective dst" from src (see below)
    ) -> GraphObs:
        # Directed adjacency: src->dst if rx_power_dbm[src,dst] >= threshold.
        #
        # In LAWNEnv we pass a link-interference matrix where columns correspond to each
        # transmitter's intended receiver (ring pairing). That makes dst index == "link index".
        rx_power_dbm = np.asarray(rx_power_dbm, dtype=np.float32)
        cs_mask = rx_power_dbm >= self.cs_threshold_dbm
        if self.graph_mode == "conflict":
            sig_dbm_all = np.diag(rx_power_dbm).astype(np.float32)  # (N,)
            sir_db_mat = sig_dbm_all.reshape(1, self.N) - rx_power_dbm
            conflict_mask = sir_db_mat < self.sinr_threshold_db
            adj = (cs_mask & conflict_mask).astype(np.uint8)
        else:
            adj = cs_mask.astype(np.uint8)
        np.fill_diagonal(adj, 0)

        rows, cols = np.where(adj == 1)
        edge_index = np.stack([rows, cols], axis=0).astype(np.int64)

        # Edge attributes (local): normalized pairwise SIR proxy at receiver of link `dst`,
        # computed from large-scale received powers only.
        #
        # sir_db(j->i) = P_sig(i->rx_i)[dBm] - P_int(j->rx_i)[dBm]
        # Higher is better (weaker interferer). This stays within local sensing.
        if rows.size == 0:
            edge_attr = np.zeros((0, 1), dtype=np.float32)
        else:
            sig_dbm_all = np.diag(rx_power_dbm).astype(np.float32)  # (N,)
            inter_dbm = rx_power_dbm[rows, cols].astype(np.float32)  # (E,)
            sir_db = sig_dbm_all[cols] - inter_dbm
            sir_clip = np.clip(sir_db, -40.0, 20.0)
            sir_norm = (sir_clip + 40.0) / 60.0
            edge_attr = sir_norm.reshape(-1, 1).astype(np.float32)

        pos = np.asarray(positions_m, dtype=np.float32).copy()
        pos[:, 0:2] = pos[:, 0:2] / max(1e-6, self.map_size_m)
        pos[:, 2] = pos[:, 2] / max(1e-6, self.height_m)

        q = (np.asarray(queues, dtype=np.float32) / max(1, self.max_queue_len)).reshape(self.N, 1)
        if energy_norm is None:
            energy = np.ones((self.N, 1), dtype=np.float32)
        else:
            energy = np.asarray(energy_norm, dtype=np.float32).reshape(self.N, 1)

        extra: list[np.ndarray] = []
        if self.obs_version in {"v2", "v3"}:
            # Node-local desired-signal strength at its own receiver (normalized).
            sig_dbm = np.diag(rx_power_dbm).astype(np.float32)
            sig_min = float(self.cs_threshold_dbm - 40.0)
            sig_max = float(self.cs_threshold_dbm + 20.0)
            sig_clip = np.clip(sig_dbm, sig_min, sig_max)
            sig_norm = ((sig_clip - sig_min) / max(1e-6, sig_max - sig_min)).reshape(self.N, 1)
            extra.append(sig_norm.astype(np.float32))
        if self.obs_version == "v3":
            # Agent identity (local, symmetry-breaking). Helps converge to stable colorings.
            idx = (np.arange(self.N, dtype=np.float32) / max(1.0, float(self.N))).reshape(self.N, 1)
            extra.append(idx.astype(np.float32))

        last_act_oh = np.eye(self.K + 1, dtype=np.float32)[np.asarray(last_actions, dtype=np.int64)]
        last_status_oh = np.eye(3, dtype=np.float32)[np.asarray(last_status, dtype=np.int64)]

        # Degree uses in-degree: how many potential interferers the node can sense.
        in_deg = adj.sum(axis=0, keepdims=False).astype(np.float32).reshape(self.N, 1)
        in_deg_norm = in_deg / float(self.N)

        x = np.concatenate([pos, q, energy, *extra, last_act_oh, last_status_oh, in_deg_norm], axis=1).astype(np.float32)
        return GraphObs(x=x, edge_index=edge_index, edge_attr=edge_attr, adj=adj)
