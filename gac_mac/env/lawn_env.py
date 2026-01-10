from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from gac_mac.env.channel import ChannelModel, ChannelSample, dbm_to_watt, watt_to_dbm
from gac_mac.env.graph_builder import GraphBuilder, GraphObs
from gac_mac.env.mobility import GaussMarkovMobility
from gac_mac.env.traffic import PoissonTraffic


@dataclass(frozen=True)
class StepInfo:
    sum_rate: float
    sum_rate_mbps: float
    jain: float
    collision_rate: float
    avg_delay: float
    tx_attempts: int
    tx_success: int
    tx_collision: int
    avg_degree_in: float
    # Note: per-node arrays are returned in the info dict (not in this dataclass) to keep logs lightweight.


class LAWNEnv:
    """LAWN environment (frame-based) with a directed observation graph.

    Step represents one MAC frame:
    - agents choose a slot in {0..K-1} or No-Tx (K)
    - transmissions happen in that frame using the CURRENT state
    - then mobility + new traffic arrivals are applied to form NEXT state
    """

    STATUS_SUCCESS = 0
    STATUS_COLLISION = 1
    STATUS_IDLE = 2

    def __init__(
        self,
        *,
        num_uavs: int,
        num_slots: int,
        num_channels: int = 1,
        secondary_lbt: bool = False,
        primary_lbt: bool = False,
        map_size_m: float,
        height_m: float,
        max_queue_len: int,
        lambda_arrival_per_slot: float,
        episode_len: int,
        obs_version: str = "v2",
        graph_mode: str = "cs",
        # PHY
        p_tx_dbm: float,
        cs_threshold_dbm: float,
        sinr_threshold_db: float,
        noise_psd_dbm_per_hz: float,
        bandwidth_hz: float,
        channel: ChannelModel,
        # Mobility
        mobility: GaussMarkovMobility,
        # Reward
        reward_mode: str = "binary",
        reward_success: float,
        reward_collision: float,
        reward_idle_empty: float,
        reward_idle_nonempty: float,
        lambda_coop: float,
    ) -> None:
        self.N = int(num_uavs)
        self.K = int(num_slots)
        self.C = int(max(1, num_channels))
        self.A = int(self.C * self.K + 1)  # (channel,slot) flattened + NoTx
        self.secondary_lbt = bool(secondary_lbt)
        self.primary_lbt = bool(primary_lbt)
        self.map_size_m = float(map_size_m)
        self.height_m = float(height_m)
        self.max_queue_len = int(max_queue_len)
        self.lambda_arrival_per_slot = float(lambda_arrival_per_slot)
        self.episode_len = int(episode_len)
        self.obs_version = str(obs_version)
        self.graph_mode = str(graph_mode)

        self.p_tx_dbm = float(p_tx_dbm)
        self.p_tx_w = float(dbm_to_watt(self.p_tx_dbm))
        self.cs_threshold_dbm = float(cs_threshold_dbm)
        self.sinr_threshold_db = float(sinr_threshold_db)
        self.sinr_threshold_lin = float(10.0 ** (self.sinr_threshold_db / 10.0))
        self.bandwidth_hz = float(bandwidth_hz)  # total bandwidth across channels
        self.bandwidth_per_channel_hz = float(self.bandwidth_hz / float(self.C))
        self.noise_w = float(dbm_to_watt(noise_psd_dbm_per_hz) * self.bandwidth_per_channel_hz)

        self.reward_mode = str(reward_mode)
        self.reward_success = float(reward_success)
        self.reward_collision = float(reward_collision)
        self.reward_idle_empty = float(reward_idle_empty)
        self.reward_idle_nonempty = float(reward_idle_nonempty)
        self.lambda_coop = float(lambda_coop)

        self._rng = np.random.default_rng()
        self._t = 0

        self.channel = channel
        self.mobility = mobility
        self.traffic = PoissonTraffic(
            num_nodes=self.N,
            num_slots=self.K,
            lambda_arrival_per_slot=self.lambda_arrival_per_slot,
            max_queue_len=self.max_queue_len,
        )
        self.graph_builder = GraphBuilder(
            num_nodes=self.N,
            num_slots=self.K,
            num_channels=self.C,
            map_size_m=self.map_size_m,
            height_m=self.height_m,
            max_queue_len=self.max_queue_len,
            cs_threshold_dbm=self.cs_threshold_dbm,
            graph_mode=self.graph_mode,
            sinr_threshold_db=self.sinr_threshold_db,
            obs_version=self.obs_version,
        )

        # State for observation
        self.positions_m = np.zeros((self.N, 3), dtype=np.float32)
        self.last_actions = np.full((self.N,), fill_value=(self.A - 1), dtype=np.int64)
        self.last_action_mh = np.zeros((self.N, self.A), dtype=np.float32)
        self.last_action_mh[:, self.A - 1] = 1.0
        self.last_status = np.full((self.N,), fill_value=self.STATUS_IDLE, dtype=np.int64)

        # Per-frame sampled channel (large-scale + fading) for THIS state.
        self._ch: ChannelSample | None = None

    def reset(self, *, seed: int | None = None) -> GraphObs:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0

        mob_state = self.mobility.reset(self._rng)
        self.positions_m = mob_state.positions_m.copy()
        self.last_actions.fill(self.A - 1)
        self.last_action_mh.fill(0.0)
        self.last_action_mh[:, self.A - 1] = 1.0
        self.last_status.fill(self.STATUS_IDLE)

        self.traffic.reset()
        # Arrivals for the first decision epoch (t=0)
        self.traffic.step(t=0, rng=self._rng)

        self._ch = self.channel.sample(self.positions_m, self._rng)
        return self._get_obs()

    def step(self, actions: np.ndarray) -> tuple[GraphObs, np.ndarray, bool, dict[str, Any]]:
        frame_t = self._t
        actions = np.asarray(actions, dtype=np.int64)
        if actions.ndim == 1:
            actions = actions.reshape(self.N, 1)
        elif actions.ndim == 2:
            if actions.shape[0] != self.N:
                raise ValueError(f"actions must have shape (N,L) with N={self.N}, got {actions.shape}")
        else:
            raise ValueError(f"actions must have shape (N,) or (N,L), got {actions.shape}")

        if self._ch is None:
            self._ch = self.channel.sample(self.positions_m, self._rng)

        receivers = (np.arange(self.N, dtype=np.int64) + 1) % self.N

        # --- Build adjacency for THIS frame from large-scale received power at each link receiver ---
        rx_power_w_large = self.p_tx_w * self._ch.large_scale_gain  # (N,N) src->node
        rx_power_dbm_large = watt_to_dbm(rx_power_w_large)  # (N,N)
        # Link-interference: src -> (receiver of link i)
        rx_power_dbm_link = rx_power_dbm_large[:, receivers]  # (N,N) src->rx(i)
        obs = self.graph_builder.build(
            positions_m=self.positions_m,
            queues=self.traffic.queues,
            energy_norm=None,
            last_actions=self.last_action_mh,
            last_status=self.last_status,
            rx_power_dbm=rx_power_dbm_link,
        )
        adj = obs.adj  # uint8 (N,N)

        # --- Transmission decisions ---
        no_tx = self.A - 1
        has_pkt = self.traffic.queues > 0

        # Per-node chosen resources (deduped), capped by queue length.
        #
        # If max_tx_per_frame > 1 and secondary_lbt is enabled, treat the first action as "primary"
        # and gate secondary picks with a listen-before-talk rule:
        # - do not allow a secondary pick on a resource used by ANY primary transmission
        # - grant at most one secondary transmission per resource (global LBT order) to avoid secondary-secondary collisions
        max_tx = int(actions.shape[1])
        chosen: list[list[int]] = [[] for _ in range(self.N)]
        # Primary picks (t=0). Optionally apply LBT contention resolution per resource.
        primary_resources: set[int] = set()
        primary_groups: dict[int, list[int]] = {}
        for i in range(self.N):
            if not bool(has_pkt[i]):
                continue
            a0 = int(actions[i, 0])
            if 0 <= a0 < no_tx and int(self.traffic.queues[i]) > 0:
                primary_groups.setdefault(a0, []).append(i)

        if self.primary_lbt and primary_groups:
            for a0, nodes in primary_groups.items():
                if not nodes:
                    continue
                if len(nodes) == 1:
                    i = int(nodes[0])
                else:
                    i = int(nodes[int(self._rng.integers(0, len(nodes)))])
                chosen[i].append(int(a0))
                primary_resources.add(int(a0))
        else:
            for a0, nodes in primary_groups.items():
                for i in nodes:
                    chosen[int(i)].append(int(a0))
                if nodes:
                    primary_resources.add(int(a0))

        if max_tx > 1:
            requests: list[tuple[int, int]] = []
            for i in range(self.N):
                if not bool(has_pkt[i]):
                    continue
                cap = int(min(max_tx, int(self.traffic.queues[i])))
                if len(chosen[i]) >= cap:
                    continue
                seen = set(chosen[i])
                for t in range(1, max_tx):
                    a = int(actions[i, t])
                    if a == no_tx:
                        break
                    if 0 <= a < no_tx and a not in seen:
                        requests.append((i, a))
                        seen.add(a)
                        if len(seen) >= cap:
                            break

            if self.secondary_lbt and requests:
                granted_resources: set[int] = set()
                order = self._rng.permutation(len(requests))
                for idx in order.tolist():
                    i, a = requests[int(idx)]
                    if not bool(has_pkt[i]):
                        continue
                    cap = int(min(max_tx, int(self.traffic.queues[i])))
                    if len(chosen[i]) >= cap:
                        continue
                    if a in primary_resources:
                        continue
                    if a in granted_resources:
                        continue
                    if a in chosen[i]:
                        continue
                    chosen[i].append(int(a))
                    granted_resources.add(int(a))
            else:
                for i, a in requests:
                    cap = int(min(max_tx, int(self.traffic.queues[i])))
                    if len(chosen[i]) >= cap:
                        continue
                    if a not in chosen[i]:
                        chosen[i].append(int(a))

        attempt_counts = np.array([len(chosen[i]) for i in range(self.N)], dtype=np.int32)
        tx_mask_any = attempt_counts > 0

        # Full received power matrix with fading
        rx_power_w = self.p_tx_w * self._ch.gain  # (N,N) src->dst

        # Power from each transmitter j to receiver of each i
        power_to_rx = rx_power_w[:, receivers]  # (N, N): row=j (tx), col=i (target receiver of i)
        signal = power_to_rx[np.arange(self.N), np.arange(self.N)]  # (N,) tx i -> rx_i

        # Group transmissions by (channel, slot) resource and compute SINR per attempt.
        succ_counts = np.zeros((self.N,), dtype=np.int32)
        coll_counts = np.zeros((self.N,), dtype=np.int32)
        rate_per_node = np.zeros((self.N,), dtype=np.float32)  # sum log2(1+sinr) over successful tx attempts

        groups: dict[tuple[int, int], list[int]] = {}
        for i in range(self.N):
            for a in chosen[i]:
                ch_i = int(a // self.K)
                sl_i = int(a % self.K)
                groups.setdefault((ch_i, sl_i), []).append(i)

        for (_ch, _sl), nodes in groups.items():
            if not nodes:
                continue
            g = np.array(nodes, dtype=np.int64)
            tot = power_to_rx[np.ix_(g, g)].sum(axis=0).astype(np.float64)  # (|g|,)
            sig = signal[g].astype(np.float64)
            inter = tot - sig
            denom = inter + float(self.noise_w)
            denom = np.maximum(denom, 1e-30)
            sinr_g = (sig / denom).astype(np.float32)
            ok = sinr_g >= self.sinr_threshold_lin
            if np.any(ok):
                rate_per_node[g[ok]] += np.log2(1.0 + sinr_g[ok]).astype(np.float32)
                succ_counts[g[ok]] += 1
            if np.any(~ok):
                coll_counts[g[~ok]] += 1

        # Debug/diagnostics: resource occupancy (helps validate collision metrics).
        max_group_size = 0
        num_multi_tx_resources = 0
        for nodes in groups.values():
            m = int(len(nodes))
            if m > max_group_size:
                max_group_size = m
            if m > 1:
                num_multi_tx_resources += 1

        delays = self.traffic.serve_successes(success_counts=succ_counts, t=frame_t)

        # --- Rewards ---
        r_pen = np.zeros((self.N,), dtype=np.float32)

        if self.reward_mode == "mbps":
            # Reward aligned with throughput (Mbps). Total bandwidth is fixed; each channel gets B/C.
            r_perf = (self.reward_success * (self.bandwidth_per_channel_hz / 1e6) * rate_per_node).astype(np.float32)
        elif self.reward_mode == "rate":
            # Reward aligned with spectral efficiency (sum log2(1+SINR)), independent of bandwidth.
            r_perf = (self.reward_success * rate_per_node).astype(np.float32)
        else:
            r_perf = (self.reward_success * succ_counts.astype(np.float32)).astype(np.float32)

        idle = ~tx_mask_any
        idle_empty = idle & (~has_pkt)
        idle_nonempty = idle & has_pkt

        r_perf[idle_empty] = self.reward_idle_empty
        if np.any(coll_counts > 0):
            r_pen[coll_counts > 0] = self.reward_collision * coll_counts[coll_counts > 0].astype(np.float32)
        r_pen[idle_nonempty] = self.reward_idle_nonempty

        # Cooperative term over IN-neighbors: N_i = { j | j -> i }
        in_deg = adj.sum(axis=0).astype(np.float32)  # (N,)
        coop_sum = adj.T.astype(np.float32) @ r_perf  # (N,)
        r_coop = np.zeros((self.N,), dtype=np.float32)
        mask = in_deg > 0
        r_coop[mask] = self.lambda_coop * (coop_sum[mask] / in_deg[mask])

        rewards = r_perf + r_pen + r_coop

        # --- Update last-action/status for next observation ---
        # Record multi-hot last action over resources; if no attempt, mark NoTx.
        self.last_action_mh.fill(0.0)
        for i in range(self.N):
            if chosen[i]:
                for a in chosen[i]:
                    self.last_action_mh[i, int(a)] = 1.0
            else:
                self.last_action_mh[i, no_tx] = 1.0
        # Keep a legacy "last action index": first chosen or NoTx.
        for i in range(self.N):
            self.last_actions[i] = int(chosen[i][0]) if chosen[i] else int(no_tx)
        self.last_status = np.full((self.N,), self.STATUS_IDLE, dtype=np.int64)
        any_succ = succ_counts > 0
        any_coll = coll_counts > 0
        self.last_status[any_succ & (~any_coll)] = self.STATUS_SUCCESS
        self.last_status[any_coll] = self.STATUS_COLLISION

        # --- Transition: mobility + new arrivals for next frame ---
        mob_state = self.mobility.step(self._rng)
        self.positions_m = mob_state.positions_m.copy()
        self.traffic.step(t=frame_t + 1, rng=self._rng)

        # Sample next-frame channel based on new positions
        self._ch = self.channel.sample(self.positions_m, self._rng)

        self._t = frame_t + 1
        done = self._t >= self.episode_len

        # Metrics (aggregated over all per-resource attempts)
        attempts = int(attempt_counts.sum())
        n_coll = int(coll_counts.sum())
        sum_rate = float(rate_per_node.sum())
        sum_rate_mbps = float((self.bandwidth_per_channel_hz * rate_per_node).sum() / 1e6)
        denom = float((rate_per_node ** 2).sum())
        jain = float((sum_rate * sum_rate) / (self.N * denom)) if denom > 0.0 else 0.0
        info = StepInfo(
            sum_rate=sum_rate,
            sum_rate_mbps=sum_rate_mbps,
            jain=jain,
            collision_rate=float(n_coll / max(1, attempts)),
            avg_delay=float(np.mean(delays)) if delays else 0.0,
            tx_attempts=attempts,
            tx_success=int(succ_counts.sum()),
            tx_collision=n_coll,
            avg_degree_in=float(in_deg.mean()),
        )
        return self._get_obs(), rewards, done, {
            "t": self._t,
            **info.__dict__,
            "max_group_size": int(max_group_size),
            "num_multi_tx_resources": int(num_multi_tx_resources),
            # Training-only auxiliaries (do not affect observation):
            "per_node_tx_attempts": attempt_counts.astype(np.int32),
            "per_node_tx_successes": succ_counts.astype(np.int32),
            "per_node_tx_collisions": coll_counts.astype(np.int32),
        }

    def _get_obs(self) -> GraphObs:
        if self._ch is None:
            self._ch = self.channel.sample(self.positions_m, self._rng)
        receivers = (np.arange(self.N, dtype=np.int64) + 1) % self.N
        rx_power_w_large = self.p_tx_w * self._ch.large_scale_gain
        rx_power_dbm_large = watt_to_dbm(rx_power_w_large)
        rx_power_dbm_link = rx_power_dbm_large[:, receivers]
        return self.graph_builder.build(
            positions_m=self.positions_m,
            queues=self.traffic.queues,
            energy_norm=None,
            last_actions=self.last_action_mh,
            last_status=self.last_status,
            rx_power_dbm=rx_power_dbm_link,
        )
