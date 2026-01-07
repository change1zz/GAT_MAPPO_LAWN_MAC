from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict

import numpy as np

from gac_mac.baselines.csma import CSMAAgent
from gac_mac.baselines.greedy_coloring import GreedyColoringAgent
from gac_mac.baselines.random_agent import RandomAgent
from gac_mac.config import Config, config_from_dict
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
    p.add_argument("--no-action-mask", action="store_true", help="Disable queue-based action masking.")
    p.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated eval seeds (default: Config.eval_seeds).",
    )
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="(deprecated) Equivalent to --policy argmax.",
    )
    p.add_argument("--out", type=str, default=None, help="Optional path to save a comparison png.")
    p.add_argument("--json-out", type=str, default=None, help="Optional path to save detailed metrics json.")
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
        reward_repeat_success=getattr(cfg, "reward_repeat_success", 0.0),
        reward_repeat_collision=getattr(cfg, "reward_repeat_collision", 0.0),
        lambda_coop=cfg.lambda_coop,
    )


def _percentiles(x: np.ndarray, ps=(10, 50, 90)) -> dict[str, float]:
    if x.size == 0:
        return {f"p{p}": 0.0 for p in ps}
    return {f"p{p}": float(np.percentile(x, p)) for p in ps}


def run_policy(name: str, policy_fn, *, cfg: Config, seeds: list[int]) -> dict[str, object]:
    env = make_env(cfg)

    ep_sum_rate = []
    ep_thr_mbps = []
    ep_colls = []
    ep_delays = []
    ep_jain = []

    per_seed_per_node_thr: dict[int, list[float]] = {}
    per_seed_per_node_success: dict[int, list[int]] = {}
    per_seed_per_node_attempt: dict[int, list[int]] = {}
    per_seed_per_node_collision: dict[int, list[int]] = {}
    per_seed_per_node_delay_means: dict[int, list[float]] = {}

    from gac_mac.utils.metrics import jain_index

    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        obs = env.reset(seed=seed)
        if hasattr(policy_fn, "reset"):
            policy_fn.reset()  # type: ignore[attr-defined]

        rate_sum = 0.0
        thr_sum = 0.0
        coll_sum = 0.0
        delay_sum = 0.0

        per_node_thr = np.zeros((cfg.num_uavs,), dtype=np.float64)
        per_node_attempt = np.zeros((cfg.num_uavs,), dtype=np.int64)
        per_node_success = np.zeros((cfg.num_uavs,), dtype=np.int64)
        per_node_collision = np.zeros((cfg.num_uavs,), dtype=np.int64)
        per_node_delays: list[list[float]] = [[] for _ in range(cfg.num_uavs)]

        for _t in range(cfg.episode_len):
            actions = policy_fn(env, obs, rng)
            obs, _rew, done, info = env.step(actions)
            rate_sum += float(info["sum_rate"])
            thr_sum += float(info.get("throughput_mbps", 0.0))
            coll_sum += float(info["collision_rate"])
            delay_sum += float(info["avg_delay"])

            if "per_node_throughput_mbps" in info:
                per_node_thr += np.asarray(info["per_node_throughput_mbps"], dtype=np.float64)
            if "per_node_tx_attempt" in info:
                per_node_attempt += np.asarray(info["per_node_tx_attempt"], dtype=np.int64)
            if "per_node_tx_success" in info:
                per_node_success += np.asarray(info["per_node_tx_success"], dtype=np.int64)
            if "per_node_tx_collision" in info:
                per_node_collision += np.asarray(info["per_node_tx_collision"], dtype=np.int64)
            if "per_node_delay_served" in info:
                d = np.asarray(info["per_node_delay_served"], dtype=np.float32)
                idx = np.where(~np.isnan(d))[0]
                for i in idx:
                    per_node_delays[int(i)].append(float(d[int(i)]))

            if done:
                break

        per_node_thr_mean = per_node_thr / float(cfg.episode_len)
        jain = jain_index(per_node_thr_mean)

        per_seed_per_node_thr[int(seed)] = per_node_thr_mean.astype(np.float64).tolist()
        per_seed_per_node_attempt[int(seed)] = per_node_attempt.astype(np.int64).tolist()
        per_seed_per_node_success[int(seed)] = per_node_success.astype(np.int64).tolist()
        per_seed_per_node_collision[int(seed)] = per_node_collision.astype(np.int64).tolist()

        per_node_delay_mean = np.array(
            [float(np.mean(v)) if v else float("nan") for v in per_node_delays], dtype=np.float64
        )
        per_seed_per_node_delay_means[int(seed)] = per_node_delay_mean.tolist()

        ep_sum_rate.append(rate_sum / cfg.episode_len)
        ep_thr_mbps.append(thr_sum / cfg.episode_len)
        ep_colls.append(coll_sum / cfg.episode_len)
        ep_delays.append(delay_sum / cfg.episode_len)
        ep_jain.append(float(jain))

    return {
        "name": name,
        "seeds": [int(s) for s in seeds],
        "throughput_mbps": float(np.mean(ep_thr_mbps)),
        "sum_rate": float(np.mean(ep_sum_rate)),
        "collision_rate": float(np.mean(ep_colls)),
        "avg_delay": float(np.mean(ep_delays)),
        "jain_throughput": float(np.mean(ep_jain)),
        "throughput_mbps_percentiles": _percentiles(np.asarray(ep_thr_mbps, dtype=np.float32)),
        "jain_percentiles": _percentiles(np.asarray(ep_jain, dtype=np.float32)),
        "per_seed_per_node_throughput_mbps": per_seed_per_node_thr,
        "per_seed_per_node_tx_attempts": per_seed_per_node_attempt,
        "per_seed_per_node_tx_success": per_seed_per_node_success,
        "per_seed_per_node_tx_collision": per_seed_per_node_collision,
        "per_seed_per_node_delay_mean": per_seed_per_node_delay_means,
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    ckpt = load_checkpoint(args.checkpoint)
    cfg_dict = ckpt.get("config", asdict(Config()))
    # Backward-compat: older checkpoints (v1) did not include obs/model versioning fields.
    if "obs_version" not in cfg_dict:
        cfg_dict = {**cfg_dict, "obs_version": "v1", "use_edge_attr": False}
    cfg = config_from_dict(cfg_dict)
    if args.cs_threshold_dbm is not None:
        cfg = cfg.__class__(**{**asdict(cfg), "cs_threshold_dbm": float(args.cs_threshold_dbm)})

    if args.seeds:
        seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    else:
        seeds = list(getattr(cfg, "eval_seeds", [args.seed + i for i in range(args.episodes)]))
    if args.episodes is not None and args.seeds is None:
        # Keep backward compatibility: episodes controls how many seeds to take from the list.
        seeds = seeds[: int(args.episodes)]

    device = _resolve_device(args.device)
    env_for_shape = make_env(cfg)
    obs0 = env_for_shape.reset(seed=args.seed)
    input_dim = int(obs0.x.shape[1])

    import torch

    agent = GACMACAgent(
        input_dim=input_dim,
        hidden_dim=cfg.hidden_dim,
        action_dim=cfg.num_slots + 1,
        gat_heads=cfg.gat_heads,
        use_edge_attr=cfg.use_edge_attr,
        critic_mode=getattr(cfg, "critic_mode", "global_mean"),
        policy_mode=getattr(cfg, "policy_mode", "hierarchical"),
    ).to(device)
    agent.load_state_dict(ckpt["agent_state_dict"], strict=False)
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
            h_out = agent.encoder(x, ei, ea, h_in)
            logits = agent.action_logits(h_out)
            if (not args.no_action_mask) and bool((x[:, 3] <= 0.0).any().item()):
                has_pkt = x[:, 3] > 0.0
                logits = logits.clone()
                logits[~has_pkt, : cfg.num_slots] = -1e9
            if args.policy == "sample":
                temp = float(max(1e-6, args.temperature))
                dist = torch.distributions.Categorical(logits=logits / temp)
                act = dist.sample()
            else:
                probs = torch.softmax(logits, dim=-1)
                if args.policy == "argmax":
                    act = torch.argmax(probs, dim=-1)
                else:
                    # grouped: compare transmit mass vs no-tx
                    p_no = probs[:, cfg.num_slots]
                    p_tx = probs[:, : cfg.num_slots].sum(dim=-1)
                    best_slot = torch.argmax(probs[:, : cfg.num_slots], dim=-1)
                    act = torch.where(p_tx > p_no, best_slot, torch.full_like(best_slot, cfg.num_slots))
        gac_policy.h = h_out.detach()
        return act.cpu().numpy()

    def _gac_reset() -> None:
        gac_policy.h = agent.initial_hidden(cfg.num_uavs, device)

    gac_policy.reset = _gac_reset  # type: ignore[attr-defined]

    random_agent = RandomAgent(cfg.num_slots)

    def random_policy(_env: LAWNEnv, obs, rng: np.random.Generator) -> np.ndarray:
        return random_agent.select_actions(obs.x, rng)

    greedy_agent = GreedyColoringAgent(cfg.num_slots)

    def greedy_policy(_env: LAWNEnv, obs, _rng: np.random.Generator) -> np.ndarray:
        return greedy_agent.select_actions(env=_env, obs_x=obs.x)

    csma_agent = CSMAAgent(cfg.num_slots, cfg.num_uavs)

    def csma_policy(env: LAWNEnv, obs, rng: np.random.Generator) -> np.ndarray:
        # Use env.last_status as ACK feedback.
        return csma_agent.select_actions(obs.x, env.last_status, rng)

    results: list[dict[str, object]] = []
    results.append(run_policy("GAC-MAC", gac_policy, cfg=cfg, seeds=seeds))
    results.append(run_policy("Random", random_policy, cfg=cfg, seeds=seeds))
    results.append(run_policy("Greedy", greedy_policy, cfg=cfg, seeds=seeds))
    csma_policy.reset = csma_agent.reset  # type: ignore[attr-defined]
    results.append(run_policy("CSMA", csma_policy, cfg=cfg, seeds=seeds))

    print("=== Evaluation (mean over episodes) ===")
    for r in results:
        print(
            f"{str(r['name']):7s} | thr {float(r.get('throughput_mbps', 0.0)):.3f} Mbps | "
            f"jain {float(r.get('jain_throughput', 0.0)):.3f} | "
            f"sum_rate {float(r['sum_rate']):.3f} | coll {float(r['collision_rate']):.3f} | delay {float(r['avg_delay']):.3f}"
        )

    if args.out:
        from gac_mac.viz.plots import plot_eval_summary

        # plot_eval_summary expects float dicts; pass a compact view.
        compact = [
            {
                "name": str(r["name"]),
                "throughput_mbps": float(r.get("throughput_mbps", 0.0)),
                "sum_rate": float(r.get("sum_rate", 0.0)),
                "collision_rate": float(r.get("collision_rate", 0.0)),
                "avg_delay": float(r.get("avg_delay", 0.0)),
                "jain_throughput": float(r.get("jain_throughput", 0.0)),
            }
            for r in results
        ]
        plot_eval_summary(compact, args.out)
        print(f"saved: {args.out}")

    json_out = args.json_out
    if json_out is None and args.out:
        json_out = os.path.splitext(args.out)[0] + ".json"
    if json_out:
        os.makedirs(os.path.dirname(os.path.abspath(json_out)), exist_ok=True)
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "checkpoint": os.path.abspath(args.checkpoint),
                    "seeds": seeds,
                    "config": asdict(cfg),
                    "results": results,
                },
                f,
                indent=2,
            )
        print(f"saved: {json_out}")


if __name__ == "__main__":
    main()
