from __future__ import annotations

import argparse
from dataclasses import asdict

import numpy as np

from gac_mac.baselines.csma import CSMAAgent
from gac_mac.baselines.aloha import AlohaAgent
from gac_mac.baselines.fixed_tdma import FixedTDMAAgent
from gac_mac.baselines.greedy_coloring import GreedyColoringAgent
from gac_mac.baselines.random_agent import RandomAgent
from gac_mac.baselines.hsatmac import HSATMACAgent
from gac_mac.baselines.satmac import SATMACAgent
from gac_mac.config import Config
from gac_mac.env.channel import ChannelModel
from gac_mac.env.lawn_env import LAWNEnv
from gac_mac.env.mobility import GaussMarkovMobility
from gac_mac.models.agent import GACMACAgent
from gac_mac.utils.checkpoint import load_checkpoint
from gac_mac.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate trained GAC-MAC vs baselines on LAWNEnv.")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint_XXXX.pth")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--cs-threshold-dbm", type=float, default=None, help="Override cs_threshold for evaluation env.")
    p.add_argument("--num-uavs", type=int, default=None, help="Override number of UAVs (density).")
    p.add_argument("--map-size-m", type=float, default=None, help="Override map size in meters (density).")
    p.add_argument("--num-channels", type=int, default=None, help="Override num_channels (C) for evaluation env.")
    p.add_argument("--max-tx-per-frame", type=int, default=None, help="Override max_tx_per_frame (L) for evaluation.")
    p.add_argument("--baseline-max-tx", type=int, default=1, help="Max transmissions per frame for baselines (paper-aligned default=1).")
    p.add_argument("--secondary-lbt", action="store_true", help="If L>1, gate secondary picks via listen-before-talk (reduce collisions).")
    p.add_argument("--primary-lbt", action="store_true", help="Apply listen-before-talk contention resolution for primary pick too.")
    p.add_argument("--agent-id-tiebreak-eps", type=float, default=None, help="Override agent-id tiebreak epsilon.")
    p.add_argument("--neighbor-last-action-mask", action="store_true", help="Mask resources used by in-neighbors in last frame.")
    p.add_argument("--neighbor-last-action-penalty", type=float, default=None, help="Logit penalty for resources used by in-neighbors in last frame.")
    p.add_argument(
        "--policy",
        type=str,
        default="sample",
        choices=["grouped", "argmax", "sample"],
        help="Action selection for GAC-MAC. 'grouped' compares P(Tx)=sum(slots) vs P(NoTx).",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for GAC-MAC when --policy sample (smaller => more deterministic).",
    )
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="(deprecated) Equivalent to --policy argmax.",
    )
    p.add_argument("--out", type=str, default=None, help="Optional path to save a comparison png.")
    return p.parse_args()


def _resolve_device(device: str) -> "object":
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


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
        secondary_lbt=bool(getattr(cfg, "secondary_lbt", False)),
        primary_lbt=bool(getattr(cfg, "primary_lbt", False)),
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


def run_policy(name: str, policy_fn, *, cfg: Config, base_seed: int, episodes: int) -> dict[str, float]:
    env = make_env(cfg)
    rng = np.random.default_rng(base_seed)

    ep_rates = []
    ep_jains = []
    ep_colls = []
    ep_delays = []
    ep_attempts = []
    ep_success = []
    ep_collisions = []
    ep_max_group = []
    ep_multi_res = []

    for ep in range(episodes):
        seed = int(base_seed + ep)
        obs = env.reset(seed=seed)
        if hasattr(policy_fn, "reset"):
            policy_fn.reset()  # type: ignore[attr-defined]

        rate_sum = 0.0
        jain_sum = 0.0
        coll_sum = 0.0
        delay_sum = 0.0
        attempts_sum = 0.0
        success_sum = 0.0
        collisions_sum = 0.0
        max_group_sum = 0.0
        multi_res_sum = 0.0

        for _t in range(cfg.episode_len):
            actions = policy_fn(env, obs, rng)
            obs, _rew, done, info = env.step(actions)
            rate_sum += float(info.get("sum_rate_mbps", info["sum_rate"]))
            jain_sum += float(info.get("jain", 0.0))
            coll_sum += float(info["collision_rate"])
            delay_sum += float(info["avg_delay"])
            attempts_sum += float(info.get("tx_attempts", 0.0))
            success_sum += float(info.get("tx_success", 0.0))
            collisions_sum += float(info.get("tx_collision", 0.0))
            max_group_sum += float(info.get("max_group_size", 0.0))
            multi_res_sum += float(info.get("num_multi_tx_resources", 0.0))
            if done:
                break

        ep_rates.append(rate_sum / cfg.episode_len)
        ep_jains.append(jain_sum / cfg.episode_len)
        ep_colls.append(coll_sum / cfg.episode_len)
        ep_delays.append(delay_sum / cfg.episode_len)
        ep_attempts.append(attempts_sum / cfg.episode_len)
        ep_success.append(success_sum / cfg.episode_len)
        ep_collisions.append(collisions_sum / cfg.episode_len)
        ep_max_group.append(max_group_sum / cfg.episode_len)
        ep_multi_res.append(multi_res_sum / cfg.episode_len)

    return {
        "name": name,
        "sum_rate_mbps": float(np.mean(ep_rates)),
        "jain": float(np.mean(ep_jains)),
        "collision_rate": float(np.mean(ep_colls)),
        "avg_delay": float(np.mean(ep_delays)),
        "tx_attempts": float(np.mean(ep_attempts)),
        "tx_success": float(np.mean(ep_success)),
        "tx_collision": float(np.mean(ep_collisions)),
        "max_group_size": float(np.mean(ep_max_group)),
        "num_multi_tx_resources": float(np.mean(ep_multi_res)),
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    ckpt = load_checkpoint(args.checkpoint)
    cfg_dict = ckpt.get("config", asdict(Config()))
    # Backward-compat: older checkpoints (v1) did not include obs/model versioning fields.
    if "obs_version" not in cfg_dict:
        cfg_dict = {**cfg_dict, "obs_version": "v1", "use_edge_attr": False}
    cfg = Config(**cfg_dict)
    if args.cs_threshold_dbm is not None:
        cfg = cfg.__class__(**{**asdict(cfg), "cs_threshold_dbm": float(args.cs_threshold_dbm)})
    if args.num_uavs is not None:
        cfg = cfg.__class__(**{**asdict(cfg), "num_uavs": int(args.num_uavs)})
    if args.map_size_m is not None:
        cfg = cfg.__class__(**{**asdict(cfg), "map_size_m": float(args.map_size_m)})
    if args.num_channels is not None:
        cfg = cfg.__class__(**{**asdict(cfg), "num_channels": int(args.num_channels)})
    if args.max_tx_per_frame is not None:
        cfg = cfg.__class__(**{**asdict(cfg), "max_tx_per_frame": int(args.max_tx_per_frame)})
    if args.secondary_lbt:
        cfg = cfg.__class__(**{**asdict(cfg), "secondary_lbt": True})
    if args.primary_lbt:
        cfg = cfg.__class__(**{**asdict(cfg), "primary_lbt": True})
    if args.agent_id_tiebreak_eps is not None:
        cfg = cfg.__class__(**{**asdict(cfg), "agent_id_tiebreak_eps": float(args.agent_id_tiebreak_eps)})
    if args.neighbor_last_action_mask:
        cfg = cfg.__class__(**{**asdict(cfg), "neighbor_last_action_mask": True})
    if args.neighbor_last_action_penalty is not None:
        cfg = cfg.__class__(**{**asdict(cfg), "neighbor_last_action_penalty": float(args.neighbor_last_action_penalty)})

    device = _resolve_device(args.device)
    env_for_shape = make_env(cfg)
    obs0 = env_for_shape.reset(seed=args.seed)
    input_dim = int(obs0.x.shape[1])

    import torch

    agent = GACMACAgent(
        input_dim=input_dim,
        hidden_dim=cfg.hidden_dim,
        action_dim=int(cfg.num_slots) * int(getattr(cfg, "num_channels", 1)) + 1,
        gat_heads=cfg.gat_heads,
        use_edge_attr=cfg.use_edge_attr,
        use_agent_id_tiebreak=(cfg.obs_version == "v3" and getattr(cfg, "agent_id_tiebreak_eps", 0.0) > 0.0),
        agent_id_tiebreak_eps=getattr(cfg, "agent_id_tiebreak_eps", 0.0),
        neighbor_last_action_mask=getattr(cfg, "neighbor_last_action_mask", False),
        neighbor_last_action_penalty=getattr(cfg, "neighbor_last_action_penalty", 0.0),
    ).to(device)
    agent.load_state_dict(ckpt["agent_state_dict"])
    agent.eval()

    if args.deterministic and args.policy == "sample":
        args.policy = "argmax"

    def gac_policy(env: LAWNEnv, obs, rng: np.random.Generator) -> np.ndarray:
        x = torch.tensor(obs.x, dtype=torch.float32, device=device)
        ei = torch.tensor(obs.edge_index, dtype=torch.long, device=device)
        ea = torch.tensor(obs.edge_attr, dtype=torch.float32, device=device)
        if not hasattr(gac_policy, "h"):
            gac_policy.h = agent.initial_hidden(cfg.num_uavs, device)
        h_in = gac_policy.h
        with torch.no_grad():
            max_tx = int(getattr(cfg, "max_tx_per_frame", 1))
            if args.policy == "grouped":
                logits, h_out = agent.forward_logits(x, ei, ea, h_in)
                probs = torch.softmax(logits, dim=-1)
                no_tx = int(cfg.num_slots) * int(getattr(cfg, "num_channels", 1))
                p_no = probs[:, no_tx]
                p_tx = probs[:, :no_tx].sum(dim=-1)
                best_tx = torch.argmax(probs[:, :no_tx], dim=-1)
                first = torch.where(p_tx > p_no, best_tx, torch.full_like(best_tx, no_tx))
                act = torch.full((cfg.num_uavs, max_tx), no_tx, dtype=torch.long, device=device)
                act[:, 0] = first
            else:
                act, _logp, _v, _ent, h_out = agent.act(
                    x,
                    ei,
                    ea,
                    h_in,
                    deterministic=(args.policy == "argmax"),
                    temperature=float(args.temperature),
                    max_tx=max_tx,
                )
        gac_policy.h = h_out.detach()
        return act.cpu().numpy()

    def _gac_reset() -> None:
        gac_policy.h = agent.initial_hidden(cfg.num_uavs, device)

    gac_policy.reset = _gac_reset  # type: ignore[attr-defined]

    random_agent = RandomAgent(cfg.num_slots, num_channels=int(getattr(cfg, "num_channels", 1)))

    def random_policy(_env: LAWNEnv, obs, rng: np.random.Generator) -> np.ndarray:
        return random_agent.select_actions(obs.x, rng, max_tx=int(max(1, args.baseline_max_tx)))

    aloha_agent = AlohaAgent(cfg.num_slots, num_channels=int(getattr(cfg, "num_channels", 1)), p_tx=0.2)

    def aloha_policy(_env: LAWNEnv, obs, rng: np.random.Generator) -> np.ndarray:
        return aloha_agent.select_actions(obs.x, rng, max_tx=int(max(1, args.baseline_max_tx)))

    greedy_agent = GreedyColoringAgent(cfg.num_slots, num_channels=int(getattr(cfg, "num_channels", 1)))

    def greedy_policy(_env: LAWNEnv, obs, _rng: np.random.Generator) -> np.ndarray:
        return greedy_agent.select_actions(env=_env, obs_x=obs.x, max_tx=int(max(1, args.baseline_max_tx)))

    tdma_agent = FixedTDMAAgent(cfg.num_slots, cfg.num_uavs, num_channels=int(getattr(cfg, "num_channels", 1)))

    def tdma_policy(_env: LAWNEnv, obs, rng: np.random.Generator) -> np.ndarray:
        return tdma_agent.select_actions(obs.x, rng, max_tx=int(max(1, args.baseline_max_tx)))

    csma_agent = CSMAAgent(cfg.num_slots, cfg.num_uavs, num_channels=int(getattr(cfg, "num_channels", 1)))

    def csma_policy(env: LAWNEnv, obs, rng: np.random.Generator) -> np.ndarray:
        # Use env.last_status as ACK feedback.
        return csma_agent.select_actions(obs.x, env.last_status, rng, max_tx=int(max(1, args.baseline_max_tx)))

    satmac_agent = SATMACAgent(cfg.num_slots, cfg.num_uavs, num_channels=int(getattr(cfg, "num_channels", 1)), p_reselect=1.0)

    def satmac_policy(env: LAWNEnv, obs, rng: np.random.Generator) -> np.ndarray:
        return satmac_agent.select_actions(obs.x, env.last_status, rng, max_tx=int(max(1, args.baseline_max_tx)))

    hsatmac_agent = HSATMACAgent(
        cfg.num_slots,
        cfg.num_uavs,
        num_channels=int(getattr(cfg, "num_channels", 1)),
        lsg=4,
        lmin=2,
        tvalid=4,
        num_regions=10,
        cw_min=4,
        cw_max=64,
        burst_q_norm=0.6,
    )

    def hsatmac_policy(env: LAWNEnv, obs, rng: np.random.Generator) -> np.ndarray:
        # H-SATMAC uses BS + optional slot-group CSMA, so allow up to 2 transmissions.
        return hsatmac_agent.select_actions(env=env, obs=obs, rng=rng, max_tx=int(min(2, max(1, getattr(cfg, "max_tx_per_frame", 1)))))

    results = []
    results.append(run_policy("GAC-MAC", gac_policy, cfg=cfg, base_seed=args.seed, episodes=args.episodes))
    results.append(run_policy("Random", random_policy, cfg=cfg, base_seed=args.seed, episodes=args.episodes))
    results.append(run_policy("ALOHA", aloha_policy, cfg=cfg, base_seed=args.seed, episodes=args.episodes))
    results.append(run_policy("Greedy", greedy_policy, cfg=cfg, base_seed=args.seed, episodes=args.episodes))
    results.append(run_policy("TDMA", tdma_policy, cfg=cfg, base_seed=args.seed, episodes=args.episodes))
    csma_policy.reset = csma_agent.reset  # type: ignore[attr-defined]
    results.append(run_policy("CSMA", csma_policy, cfg=cfg, base_seed=args.seed, episodes=args.episodes))
    satmac_policy.reset = satmac_agent.reset  # type: ignore[attr-defined]
    results.append(run_policy("SATMAC", satmac_policy, cfg=cfg, base_seed=args.seed, episodes=args.episodes))
    hsatmac_policy.reset = hsatmac_agent.reset  # type: ignore[attr-defined]
    results.append(run_policy("H-SAT", hsatmac_policy, cfg=cfg, base_seed=args.seed, episodes=args.episodes))

    print("=== Evaluation (mean over episodes) ===")
    for r in results:
        print(
            f"{r['name']:7s} | thr {r['sum_rate_mbps']:.3f} Mbps | jain {r['jain']:.3f} | coll {r['collision_rate']:.3f} | "
            f"attempt {r['tx_attempts']:.2f}/step | succ {r['tx_success']:.2f}/step | coll_ct {r['tx_collision']:.2f}/step | "
            f"max_grp {r['max_group_size']:.2f} | multi_res {r['num_multi_tx_resources']:.2f}/step | delay {r['avg_delay']:.3f}"
        )

    if args.out:
        from gac_mac.viz.plots import plot_eval_summary

        plot_eval_summary(results, args.out)
        print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
