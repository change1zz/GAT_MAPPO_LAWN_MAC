from __future__ import annotations

import argparse
import os
from dataclasses import asdict

import numpy as np

from gac_mac.config import Config
from gac_mac.env.channel import ChannelModel
from gac_mac.env.lawn_env import LAWNEnv
from gac_mac.env.mobility import GaussMarkovMobility
from gac_mac.models.agent import GACMACAgent
from gac_mac.utils.checkpoint import load_checkpoint
from gac_mac.viz.topology import plot_topology_snapshot


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Save a topology snapshot (node colors=actions).")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--step", type=int, default=0, help="Frame index to snapshot (0-based)")
    p.add_argument("--policy", type=str, default="sample", choices=["sample", "argmax", "grouped"])
    p.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature for --policy sample.")
    p.add_argument("--out", type=str, default="topology_snapshot.png")
    return p.parse_args()


def make_env(cfg: Config) -> LAWNEnv:
    channel = ChannelModel(
        fc_ghz=cfg.fc_ghz,
        los_d1_m=cfg.los_d1_m,
        los_p1_m=cfg.los_p1_m,
        shadow_sigma_los_db=cfg.shadow_sigma_los_db,
        shadow_sigma_nlos_db=cfg.shadow_sigma_nlos_db,
        nakagami_m_los=cfg.nakagami_m_los,
        nakagami_m_nlos=cfg.nakagami_m_nlos,
        min_distance_m=cfg.min_distance_m,
    )
    mobility = GaussMarkovMobility(
        num_nodes=cfg.num_uavs,
        bounds_xyz_m=(cfg.map_size_m, cfg.map_size_m, cfg.height_m),
        alpha=cfg.mobility_alpha,
        sigma_v=cfg.mobility_sigma_v,
        v_max_mps=cfg.mobility_v_max_mps,
        dt_s=cfg.mobility_dt_s,
    )
    return LAWNEnv(
        num_uavs=cfg.num_uavs,
        num_slots=cfg.num_slots,
        num_channels=int(getattr(cfg, "num_channels", 1)),
        map_size_m=cfg.map_size_m,
        height_m=cfg.height_m,
        max_queue_len=cfg.max_queue_len,
        lambda_arrival_per_slot=cfg.lambda_arrival_per_slot,
        episode_len=cfg.episode_len,
        obs_version=cfg.obs_version,
        graph_mode=cfg.graph_mode,
        p_tx_dbm=cfg.p_tx_dbm,
        cs_threshold_dbm=cfg.cs_threshold_dbm,
        sinr_threshold_db=cfg.sinr_threshold_db,
        noise_psd_dbm_per_hz=cfg.noise_psd_dbm_per_hz,
        bandwidth_hz=cfg.bandwidth_hz,
        channel=channel,
        mobility=mobility,
        reward_mode=cfg.reward_mode,
        reward_success=cfg.reward_success,
        reward_collision=cfg.reward_collision,
        reward_idle_empty=cfg.reward_idle_empty,
        reward_idle_nonempty=cfg.reward_idle_nonempty,
        lambda_coop=cfg.lambda_coop,
    )


def main() -> None:
    args = parse_args()
    ckpt = load_checkpoint(args.checkpoint)
    cfg_dict = ckpt.get("config", asdict(Config()))
    # Backward-compat: older checkpoints (v1) did not include obs/model versioning fields.
    if "obs_version" not in cfg_dict:
        cfg_dict = {**cfg_dict, "obs_version": "v1", "use_edge_attr": False}
    cfg = Config(**cfg_dict)

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    env = make_env(cfg)
    obs = env.reset(seed=args.seed)

    input_dim = int(obs.x.shape[1])
    agent = GACMACAgent(
        input_dim=input_dim,
        hidden_dim=cfg.hidden_dim,
        action_dim=int(cfg.num_slots) * int(getattr(cfg, "num_channels", 1)) + 1,
        gat_heads=cfg.gat_heads,
        use_edge_attr=cfg.use_edge_attr,
    ).to(device)
    agent.load_state_dict(ckpt["agent_state_dict"])
    agent.eval()

    h = agent.initial_hidden(cfg.num_uavs, device)

    rng = np.random.default_rng(args.seed)

    snapshot_obs = obs
    snapshot_positions = env.positions_m.copy()
    no_tx = int(cfg.num_slots) * int(getattr(cfg, "num_channels", 1))
    snapshot_actions = np.full((cfg.num_uavs,), fill_value=no_tx, dtype=np.int64)

    for t in range(cfg.episode_len):
        x = torch.tensor(obs.x, dtype=torch.float32, device=device)
        ei = torch.tensor(obs.edge_index, dtype=torch.long, device=device)
        ea = torch.tensor(obs.edge_attr, dtype=torch.float32, device=device)
        with torch.no_grad():
            if args.policy == "grouped":
                h = agent.encoder(x, ei, ea, h)
                logits = agent.actor(h)
                probs = torch.softmax(logits, dim=-1)
                p_no = probs[:, no_tx]
                p_tx = probs[:, :no_tx].sum(dim=-1)
                best_tx = torch.argmax(probs[:, :no_tx], dim=-1)
                act0 = torch.where(p_tx > p_no, best_tx, torch.full_like(best_tx, no_tx))
                act_env = act0.view(-1, 1)
                actions = act0.cpu().numpy()
            else:
                act_t, _logp, _v, _ent, h = agent.act(
                    x,
                    ei,
                    ea,
                    h,
                    deterministic=(args.policy == "argmax"),
                    temperature=float(args.temperature),
                    max_tx=int(getattr(cfg, "max_tx_per_frame", 1)),
                )
                act_env = act_t
                actions = act_t[:, 0].cpu().numpy()

        if t == args.step:
            snapshot_obs = obs
            snapshot_positions = env.positions_m.copy()
            snapshot_actions = actions.copy()
            break

        obs, _r, done, _info = env.step(act_env.cpu().numpy())
        if done:
            break

        # Add a tiny RNG advance to keep deterministic behavior consistent if needed.
        rng.random()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plot_topology_snapshot(
        positions_m=snapshot_positions,
        edge_index=snapshot_obs.edge_index,
        actions=snapshot_actions,
        num_slots=cfg.num_slots,
        num_channels=int(getattr(cfg, "num_channels", 1)),
        save_path=args.out,
        title=f"Topology Snapshot (t={args.step})",
    )
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
