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
    throughput_mbps: float
    collision_rate: float
    avg_delay: float
    tx_attempts: int
    tx_success: int
    tx_collision: int
    avg_degree_in: float


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
        reward_tx_attempt: float,
        reward_repeat_success: float,
        reward_repeat_collision: float,
        lambda_coop: float,
    ) -> None:
        self.N = int(num_uavs)
        self.K = int(num_slots)
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
        self.bandwidth_hz = float(bandwidth_hz)
        self.noise_w = float(dbm_to_watt(noise_psd_dbm_per_hz) * self.bandwidth_hz)

        self.reward_mode = str(reward_mode)
        self.reward_success = float(reward_success)
        self.reward_collision = float(reward_collision)
        self.reward_idle_empty = float(reward_idle_empty)
        self.reward_idle_nonempty = float(reward_idle_nonempty)
        self.reward_tx_attempt = float(reward_tx_attempt)
        self.reward_repeat_success = float(reward_repeat_success)
        self.reward_repeat_collision = float(reward_repeat_collision)
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
        self.last_actions = np.full((self.N,), fill_value=self.K, dtype=np.int64)
        self.last_status = np.full((self.N,), fill_value=self.STATUS_IDLE, dtype=np.int64)

        # Per-frame sampled channel (large-scale + fading) for THIS state.
        self._ch: ChannelSample | None = None

        # Optional teacher (for training-time distillation only).
        self.enable_greedy_teacher: bool = False
        self._greedy_teacher = None

    def _get_greedy_teacher(self):
        from gac_mac.baselines.greedy_coloring import GreedyColoringAgent

        if self._greedy_teacher is None:
            self._greedy_teacher = GreedyColoringAgent(self.K)
        return self._greedy_teacher

    def reset(self, *, seed: int | None = None) -> GraphObs:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0

        mob_state = self.mobility.reset(self._rng)
        self.positions_m = mob_state.positions_m.copy()
        self.last_actions.fill(self.K)
        self.last_status.fill(self.STATUS_IDLE)

        self.traffic.reset()
        # Arrivals for the first decision epoch (t=0)
        self.traffic.step(t=0, rng=self._rng)

        self._ch = self.channel.sample(self.positions_m, self._rng)
        return self._get_obs()

    def step(self, actions: np.ndarray) -> tuple[GraphObs, np.ndarray, bool, dict[str, Any]]:
        frame_t = self._t
        actions = np.asarray(actions, dtype=np.int64).reshape(self.N)
        actions = np.clip(actions, 0, self.K)  # K means No-Tx

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
            last_actions=self.last_actions,
            last_status=self.last_status,
            rx_power_dbm=rx_power_dbm_link,
        )
        adj = obs.adj  # uint8 (N,N)

        teacher_actions = None
        if self.enable_greedy_teacher:
            # Teacher is centralized and may use env internal PHY snapshot (CTDE).
            teacher_actions = self._get_greedy_teacher().select_actions(env=self, obs_x=obs.x).astype(np.int64)

        # --- Transmission attempt mask ---
        has_pkt = self.traffic.queues > 0
        tx_mask = has_pkt & (actions < self.K)

        # Full received power matrix with fading
        rx_power_w = self.p_tx_w * self._ch.gain  # (N,N) src->dst

        # Power from each transmitter j to receiver of each i
        power_to_rx = rx_power_w[:, receivers]  # (N, N): row=j (tx), col=i (target receiver of i)
        signal = power_to_rx[np.arange(self.N), np.arange(self.N)]  # (N,) tx i -> rx_i

        # Same-slot activity matrix (j,i): both active and same slot
        same_slot = (actions[:, None] == actions[None, :]) & (actions[:, None] < self.K)
        active_pair = same_slot & tx_mask[:, None] & tx_mask[None, :]

        total_same_slot = (active_pair.astype(np.float32) * power_to_rx.astype(np.float32)).sum(axis=0)  # (N,)
        interference = total_same_slot - signal

        sinr = np.zeros((self.N,), dtype=np.float32)
        denom = interference + self.noise_w
        denom = np.maximum(denom, 1e-30)
        sinr[tx_mask] = (signal[tx_mask] / denom[tx_mask]).astype(np.float32)

        success = tx_mask & (sinr >= self.sinr_threshold_lin)
        collision = tx_mask & (~success)

        delays, per_node_delay = self.traffic.serve_successes_with_per_node_delay(success_mask=success, t=frame_t)

        # --- Rewards ---
        r_perf = np.zeros((self.N,), dtype=np.float32)

        if self.reward_mode == "rate":
            r_perf[success] = (self.reward_success * np.log2(1.0 + sinr[success])).astype(np.float32)
        else:
            r_perf[success] = self.reward_success

        idle = ~tx_mask
        idle_empty = idle & (~has_pkt)
        idle_nonempty = idle & has_pkt

        r_perf[idle_empty] = self.reward_idle_empty
        r_perf[collision] = self.reward_collision
        r_perf[idle_nonempty] = self.reward_idle_nonempty

        # Encourage "dare to transmit" when having backlog (static, local).
        if float(self.reward_tx_attempt) != 0.0:
            r_perf[tx_mask] += self.reward_tx_attempt

        # Semi-persistent shaping (local): keep slot after success; avoid stubborn repeats after collision.
        if float(self.reward_repeat_success) != 0.0 or float(self.reward_repeat_collision) != 0.0:
            prev_tx = self.last_actions < self.K
            repeat_slot = tx_mask & prev_tx & (actions == self.last_actions)
            prev_success = self.last_status == self.STATUS_SUCCESS
            prev_collision = self.last_status == self.STATUS_COLLISION
            r_perf[repeat_slot & prev_success] += self.reward_repeat_success
            r_perf[repeat_slot & prev_collision] += self.reward_repeat_collision

        # Cooperative term over IN-neighbors: N_i = { j | j -> i }
        in_deg = adj.sum(axis=0).astype(np.float32)  # (N,)
        coop_sum = adj.T.astype(np.float32) @ r_perf  # (N,)
        r_coop = np.zeros((self.N,), dtype=np.float32)
        mask = in_deg > 0
        r_coop[mask] = self.lambda_coop * (coop_sum[mask] / in_deg[mask])

        rewards = r_perf + r_coop

        # --- Update last-action/status for next observation ---
        # Only record actual transmission attempt (or No-Tx). If no packet, treat as No-Tx.
        effective_actions = actions.copy()
        effective_actions[~tx_mask] = self.K
        self.last_actions = effective_actions
        self.last_status = np.full((self.N,), self.STATUS_IDLE, dtype=np.int64)
        self.last_status[success] = self.STATUS_SUCCESS
        self.last_status[collision] = self.STATUS_COLLISION

        # --- Transition: mobility + new arrivals for next frame ---
        mob_state = self.mobility.step(self._rng)
        self.positions_m = mob_state.positions_m.copy()
        self.traffic.step(t=frame_t + 1, rng=self._rng)

        # Sample next-frame channel based on new positions
        self._ch = self.channel.sample(self.positions_m, self._rng)

        self._t = frame_t + 1
        done = self._t >= self.episode_len

        # Metrics
        attempts = int(tx_mask.sum())
        n_coll = int(collision.sum())
        per_node_se = np.zeros((self.N,), dtype=np.float32)
        if np.any(success):
            per_node_se[success] = np.log2(1.0 + sinr[success]).astype(np.float32)
        sum_rate = float(per_node_se.sum())
        from gac_mac.utils.metrics import spectral_eff_sum_to_mbps_per_frame

        throughput_mbps = spectral_eff_sum_to_mbps_per_frame(
            sum_rate, bandwidth_hz=self.bandwidth_hz, num_slots=self.K
        )
        per_node_thr_mbps = (per_node_se * (self.bandwidth_hz / float(self.K)) / 1e6).astype(np.float32)
        info = StepInfo(
            sum_rate=sum_rate,
            throughput_mbps=throughput_mbps,
            collision_rate=float(n_coll / max(1, attempts)),
            avg_delay=float(np.mean(delays)) if delays else 0.0,
            tx_attempts=attempts,
            tx_success=int(success.sum()),
            tx_collision=n_coll,
            avg_degree_in=float(in_deg.mean()),
        )
        return self._get_obs(), rewards, done, {
            "t": self._t,
            **info.__dict__,
            **({"teacher_actions": teacher_actions} if teacher_actions is not None else {}),
            # Per-node episode accounting (for evaluation only; policy doesn't need this).
            "per_node_spectral_eff": per_node_se,
            "per_node_throughput_mbps": per_node_thr_mbps,
            "per_node_tx_attempt": tx_mask.astype(np.uint8),
            "per_node_tx_success": success.astype(np.uint8),
            "per_node_tx_collision": collision.astype(np.uint8),
            "per_node_delay_served": per_node_delay,
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
            last_actions=self.last_actions,
            last_status=self.last_status,
            rx_power_dbm=rx_power_dbm_link,
        )
