from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # --- Experiment ---
    seed: int = 42
    run_name: str = "gac-mac"
    results_dir: str = "results"
    device: str = "auto"  # "auto" | "cpu" | "cuda"

    # Observation / model versioning (for checkpoint compatibility)
    obs_version: str = "v3"  # "v1" (legacy) | "v2" (+link-signal) | "v3" (+link-signal +agent-id)
    use_edge_attr: bool = True  # whether GAT uses edge_attr

    # --- Environment (toy in M1; real in later milestones) ---
    num_uavs: int = 30
    num_slots: int = 8
    map_size_m: float = 1000.0
    height_m: float = 200.0
    neighbor_radius_m: float = 250.0  # used by toy env to build a directed graph
    max_queue_len: int = 20
    lambda_arrival_per_slot: float = 0.2
    episode_len: int = 64

    # Carrier sensing (kept as dBm to match later PHY model; toy env may not use)
    p_tx_dbm: float = 23.0
    cs_threshold_dbm: float = -50.0
    graph_mode: str = "cs"  # "cs" (rx power threshold) | "conflict" (pairwise SINR conflict edges)

    # --- PHY (M2+) ---
    fc_ghz: float = 2.4
    bandwidth_hz: float = 1_000_000.0
    noise_psd_dbm_per_hz: float = -174.0
    sinr_threshold_db: float = 10.0

    # LoS probability parameters (simplified 3GPP-style; distance in m, height in m)
    los_d1_m: float = 18.0
    los_p1_m: float = 36.0

    # Shadowing (dB)
    shadow_sigma_los_db: float = 4.0
    shadow_sigma_nlos_db: float = 6.0

    # Nakagami-m (power gain) parameters
    nakagami_m_los: float = 3.0
    nakagami_m_nlos: float = 1.0

    # Numerics
    min_distance_m: float = 1.0

    # --- Mobility (M2+) ---
    mobility_alpha: float = 0.8
    mobility_sigma_v: float = 3.0
    mobility_v_max_mps: float = 20.0
    mobility_dt_s: float = 0.2

    # Reward shaping
    reward_mode: str = "binary"  # "binary" | "rate" (use log2(1+SINR) on success)
    reward_success: float = 1.0
    reward_collision: float = -2.0
    reward_idle_empty: float = 0.1
    reward_idle_nonempty: float = -0.5
    lambda_coop: float = 0.5

    # --- Model ---
    hidden_dim: int = 64
    gat_heads: int = 2
    agent_id_tiebreak_eps: float = 0.0

    # --- PPO / MAPPO ---
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ppo_epochs: int = 4
    bptt_len: int = 32  # truncated BPTT segment length
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    conflict_loss_coef: float = 0.0

    # --- Runtime / checkpointing ---
    num_envs: int = 1  # parallel rollout environments (vectorized)
    total_updates: int = 200
    steps_per_update: int = 128  # environment steps collected per update
    log_interval: int = 1
    checkpoint_interval: int = 50
    pretrain_greedy_steps: int = 0
