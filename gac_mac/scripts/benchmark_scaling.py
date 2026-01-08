from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict

import numpy as np

from gac_mac.baselines.greedy_coloring import GreedyColoringAgent
from gac_mac.config import Config
from gac_mac.env.channel import ChannelModel
from gac_mac.env.lawn_env import LAWNEnv
from gac_mac.env.mobility import GaussMarkovMobility
from gac_mac.models.agent import GACMACAgent
from gac_mac.utils.checkpoint import load_checkpoint
from gac_mac.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scaling benchmark (line plots over N).")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint_XXXX.pth")
    p.add_argument("--ns", type=str, default="10,20,30,40,50", help="Comma-separated N values.")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--policy", type=str, default="sample", choices=["argmax", "sample"])
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--out", type=str, default=None, help="Output png path (default: alongside checkpoint run dir).")
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
        lambda_coop=cfg.lambda_coop,
    )


def _run_episode(env: LAWNEnv, policy_fn, *, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    obs = env.reset(seed=seed)
    if hasattr(policy_fn, "reset"):
        policy_fn.reset()  # type: ignore[attr-defined]

    rate_sum = 0.0
    coll_sum = 0.0
    delay_sum = 0.0

    for _t in range(env.episode_len):
        actions = policy_fn(env, obs, rng)
        obs, _rew, done, info = env.step(actions)
        rate_sum += float(info["sum_rate"])
        coll_sum += float(info["collision_rate"])
        delay_sum += float(info["avg_delay"])
        if done:
            break
    denom = float(env.episode_len)
    return rate_sum / denom, coll_sum / denom, delay_sum / denom


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    ckpt_path = os.path.abspath(args.checkpoint)
    ckpt = load_checkpoint(ckpt_path)
    cfg_dict = ckpt.get("config", asdict(Config()))
    if "obs_version" not in cfg_dict:
        cfg_dict = {**cfg_dict, "obs_version": "v1", "use_edge_attr": False}
    base_cfg = Config(**cfg_dict)

    ns = [int(x.strip()) for x in args.ns.split(",") if x.strip()]
    device = _resolve_device(args.device)

    # Build agent once; evaluate across N by changing hidden size.
    env0 = make_env(base_cfg)
    obs0 = env0.reset(seed=args.seed)
    input_dim = int(obs0.x.shape[1])

    import torch

    agent = GACMACAgent(
        input_dim=input_dim,
        hidden_dim=base_cfg.hidden_dim,
        action_dim=base_cfg.num_slots + 1,
        gat_heads=base_cfg.gat_heads,
        use_edge_attr=base_cfg.use_edge_attr,
    ).to(device)
    agent.load_state_dict(ckpt["agent_state_dict"])
    agent.eval()

    results_by_algo: dict[str, dict[str, list[float]]] = {
        "GAC-MAC": {"sum_rate": [], "collision_rate": [], "avg_delay": []},
        "Greedy": {"sum_rate": [], "collision_rate": [], "avg_delay": []},
    }

    greedy = GreedyColoringAgent(base_cfg.num_slots)

    for n in ns:
        cfg = base_cfg.__class__(**{**asdict(base_cfg), "num_uavs": int(n)})
        env = make_env(cfg)

        def gac_policy(env: LAWNEnv, obs, _rng: np.random.Generator) -> np.ndarray:
            x = torch.tensor(obs.x, dtype=torch.float32, device=device)
            ei = torch.tensor(obs.edge_index, dtype=torch.long, device=device)
            ea = torch.tensor(obs.edge_attr, dtype=torch.float32, device=device)
            if not hasattr(gac_policy, "h"):
                gac_policy.h = agent.initial_hidden(cfg.num_uavs, device)
            h_in = gac_policy.h
            with torch.no_grad():
                h_out = agent.encoder(x, ei, ea, h_in)
                logits = agent.actor(h_out)
                if args.policy == "sample":
                    temp = float(max(1e-6, args.temperature))
                    dist = torch.distributions.Categorical(logits=logits / temp)
                    act = dist.sample()
                else:
                    act = torch.argmax(logits, dim=-1)
            gac_policy.h = h_out.detach()
            return act.cpu().numpy()

        def _gac_reset() -> None:
            gac_policy.h = agent.initial_hidden(cfg.num_uavs, device)

        gac_policy.reset = _gac_reset  # type: ignore[attr-defined]

        def greedy_policy(_env: LAWNEnv, obs, _rng: np.random.Generator) -> np.ndarray:
            return greedy.select_actions(env=_env, obs_x=obs.x)

        rates = []
        colls = []
        delays = []
        for ep in range(args.episodes):
            r, c, d = _run_episode(env, gac_policy, seed=int(args.seed + ep))
            rates.append(r)
            colls.append(c)
            delays.append(d)
        results_by_algo["GAC-MAC"]["sum_rate"].append(float(np.mean(rates)))
        results_by_algo["GAC-MAC"]["collision_rate"].append(float(np.mean(colls)))
        results_by_algo["GAC-MAC"]["avg_delay"].append(float(np.mean(delays)))

        rates = []
        colls = []
        delays = []
        for ep in range(args.episodes):
            r, c, d = _run_episode(env, greedy_policy, seed=int(args.seed + ep))
            rates.append(r)
            colls.append(c)
            delays.append(d)
        results_by_algo["Greedy"]["sum_rate"].append(float(np.mean(rates)))
        results_by_algo["Greedy"]["collision_rate"].append(float(np.mean(colls)))
        results_by_algo["Greedy"]["avg_delay"].append(float(np.mean(delays)))

        print(
            f"N={n:3d} | "
            f"GAC rate {results_by_algo['GAC-MAC']['sum_rate'][-1]:.3f} coll {results_by_algo['GAC-MAC']['collision_rate'][-1]:.3f} delay {results_by_algo['GAC-MAC']['avg_delay'][-1]:.3f} || "
            f"Greedy rate {results_by_algo['Greedy']['sum_rate'][-1]:.3f} coll {results_by_algo['Greedy']['collision_rate'][-1]:.3f} delay {results_by_algo['Greedy']['avg_delay'][-1]:.3f}"
        )

    run_dir = os.path.dirname(os.path.dirname(ckpt_path))
    out_png = args.out or os.path.join(run_dir, "scaling.png")
    out_json = os.path.splitext(out_png)[0] + ".json"

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"ns": ns, "results_by_algo": results_by_algo}, f, indent=2)

    from gac_mac.viz.plots import plot_scaling_lines

    plot_scaling_lines(ns=ns, results_by_algo=results_by_algo, save_path=out_png)
    print(f"saved: {out_png}")
    print(f"saved: {out_json}")


if __name__ == "__main__":
    main()

