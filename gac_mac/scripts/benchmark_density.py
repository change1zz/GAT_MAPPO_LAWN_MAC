from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict

import numpy as np

from gac_mac.baselines.aloha import AlohaAgent
from gac_mac.baselines.csma import CSMAAgent
from gac_mac.baselines.fixed_tdma import FixedTDMAAgent
from gac_mac.baselines.greedy_coloring import GreedyColoringAgent
from gac_mac.baselines.hsatmac import HSATMACAgent
from gac_mac.baselines.random_agent import RandomAgent
from gac_mac.baselines.satmac import SATMACAgent
from gac_mac.config import Config
from gac_mac.env.channel import ChannelModel
from gac_mac.env.lawn_env import LAWNEnv
from gac_mac.env.mobility import GaussMarkovMobility
from gac_mac.models.agent import GACMACAgent
from gac_mac.utils.checkpoint import load_checkpoint
from gac_mac.utils.paths import make_run_dir
from gac_mac.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Density benchmark (line plots over node density).")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint_XXXX.pth")
    p.add_argument("--ns", type=str, default="20,30,40,50,60", help="Comma-separated N values.")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--policy", type=str, default="argmax", choices=["argmax", "sample"])
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--map-size-m", type=float, default=None, help="Override map size (controls density).")
    p.add_argument("--secondary-lbt", action="store_true", help="Enable secondary-LBT gate (recommended when L>1).")
    p.add_argument("--baseline-max-tx", type=int, default=1, help="Max transmissions per frame for baselines (paper-aligned default=1).")
    p.add_argument("--gac-neighbor-last-action-mask", action="store_true", help="Enable GAC execution-time neighbor last-action mask.")
    p.add_argument("--gac-neighbor-last-action-penalty", type=float, default=None, help="Enable GAC execution-time neighbor last-action penalty.")
    p.add_argument("--gac-agent-id-tiebreak-eps", type=float, default=None, help="Override GAC agent-id tiebreak epsilon at execution.")
    p.add_argument("--plot-style", type=str, default="facets", choices=["facets", "lines"], help="Plot style for density correlation.")
    p.add_argument("--run-name", type=str, default="density", help="Run name used for output dir.")
    p.add_argument("--results-dir", type=str, default="runs_density", help="Output root dir.")
    p.add_argument("--out", type=str, default=None, help="Optional output png path (overrides run dir).")
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


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    ckpt_path = os.path.abspath(args.checkpoint)
    ckpt = load_checkpoint(ckpt_path)
    cfg_dict = ckpt.get("config", asdict(Config()))
    if "obs_version" not in cfg_dict:
        cfg_dict = {**cfg_dict, "obs_version": "v1", "use_edge_attr": False}
    base_cfg = Config(**cfg_dict)

    if args.map_size_m is not None:
        base_cfg = base_cfg.__class__(**{**asdict(base_cfg), "map_size_m": float(args.map_size_m)})
    if args.secondary_lbt:
        base_cfg = base_cfg.__class__(**{**asdict(base_cfg), "secondary_lbt": True})
    # Explicitly keep primary_lbt off for this benchmark.
    base_cfg = base_cfg.__class__(**{**asdict(base_cfg), "primary_lbt": False})

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
        action_dim=int(base_cfg.num_slots) * int(getattr(base_cfg, "num_channels", 1)) + 1,
        gat_heads=base_cfg.gat_heads,
        use_edge_attr=base_cfg.use_edge_attr,
        use_agent_id_tiebreak=(base_cfg.obs_version == "v3" and getattr(base_cfg, "agent_id_tiebreak_eps", 0.0) > 0.0),
        agent_id_tiebreak_eps=getattr(base_cfg, "agent_id_tiebreak_eps", 0.0),
        neighbor_last_action_mask=getattr(base_cfg, "neighbor_last_action_mask", False),
        neighbor_last_action_penalty=getattr(base_cfg, "neighbor_last_action_penalty", 0.0),
    ).to(device)
    agent.load_state_dict(ckpt["agent_state_dict"])
    agent.eval()

    results_by_algo: dict[str, dict[str, list[float]]] = {}
    algos = ["GAC-MAC", "Random", "ALOHA", "TDMA", "SATMAC", "H-SAT", "Greedy", "CSMA"]
    keys = [
        "sum_rate_mbps",
        "collision_rate",
        "jain",
        "avg_delay",
        "tx_attempts",
        "tx_success",
        "tx_collision",
        "max_group_size",
        "num_multi_tx_resources",
    ]
    for a in algos:
        results_by_algo[a] = {k: [] for k in keys}

    xs_density_km2: list[float] = []
    xs_n: list[int] = []

    for n in ns:
        cfg = base_cfg.__class__(**{**asdict(base_cfg), "num_uavs": int(n)})
        env = make_env(cfg)

        density_km2 = float(n) / max(1e-9, (cfg.map_size_m / 1000.0) ** 2)
        xs_density_km2.append(density_km2)
        xs_n.append(int(n))

        # Build a per-N execution agent to allow override of neighbor constraints without touching training weights.
        exec_agent = GACMACAgent(
            input_dim=input_dim,
            hidden_dim=base_cfg.hidden_dim,
            action_dim=int(base_cfg.num_slots) * int(getattr(base_cfg, "num_channels", 1)) + 1,
            gat_heads=base_cfg.gat_heads,
            use_edge_attr=base_cfg.use_edge_attr,
            use_agent_id_tiebreak=(base_cfg.obs_version == "v3" and (args.gac_agent_id_tiebreak_eps or getattr(base_cfg, "agent_id_tiebreak_eps", 0.0)) > 0.0),
            agent_id_tiebreak_eps=float(args.gac_agent_id_tiebreak_eps) if args.gac_agent_id_tiebreak_eps is not None else getattr(base_cfg, "agent_id_tiebreak_eps", 0.0),
            neighbor_last_action_mask=bool(args.gac_neighbor_last_action_mask),
            neighbor_last_action_penalty=float(args.gac_neighbor_last_action_penalty) if args.gac_neighbor_last_action_penalty is not None else 0.0,
        ).to(device)
        exec_agent.load_state_dict(agent.state_dict())
        exec_agent.eval()

        h = exec_agent.initial_hidden(cfg.num_uavs, device)

        random_agent = RandomAgent(cfg.num_slots, num_channels=int(getattr(cfg, "num_channels", 1)))
        aloha_agent = AlohaAgent(cfg.num_slots, num_channels=int(getattr(cfg, "num_channels", 1)), p_tx=0.2)
        greedy_agent = GreedyColoringAgent(cfg.num_slots, num_channels=int(getattr(cfg, "num_channels", 1)))
        tdma_agent = FixedTDMAAgent(cfg.num_slots, cfg.num_uavs, num_channels=int(getattr(cfg, "num_channels", 1)))
        csma_agent = CSMAAgent(cfg.num_slots, cfg.num_uavs, num_channels=int(getattr(cfg, "num_channels", 1)))
        satmac_agent = SATMACAgent(cfg.num_slots, cfg.num_uavs, num_channels=int(getattr(cfg, "num_channels", 1)), p_reselect=1.0)
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

        def _reset_hidden() -> None:
            nonlocal h
            h = exec_agent.initial_hidden(cfg.num_uavs, device)

        def gac_policy(env: LAWNEnv, obs, _rng: np.random.Generator) -> np.ndarray:
            nonlocal h
            x = torch.tensor(obs.x, dtype=torch.float32, device=device)
            ei = torch.tensor(obs.edge_index, dtype=torch.long, device=device)
            ea = torch.tensor(obs.edge_attr, dtype=torch.float32, device=device)
            with torch.no_grad():
                act, _logp, _v, _ent, h_out = exec_agent.act(
                    x,
                    ei,
                    ea,
                    h,
                    deterministic=(args.policy == "argmax"),
                    temperature=float(args.temperature),
                    max_tx=int(getattr(cfg, "max_tx_per_frame", 1)),
                )
            h = h_out.detach()
            return act.cpu().numpy()

        gac_policy.reset = _reset_hidden  # type: ignore[attr-defined]

        def random_policy(_env: LAWNEnv, obs, rng: np.random.Generator) -> np.ndarray:
            return random_agent.select_actions(obs.x, rng, max_tx=int(max(1, args.baseline_max_tx)))

        def aloha_policy(_env: LAWNEnv, obs, rng: np.random.Generator) -> np.ndarray:
            return aloha_agent.select_actions(obs.x, rng, max_tx=int(max(1, args.baseline_max_tx)))

        def greedy_policy(_env: LAWNEnv, obs, _rng: np.random.Generator) -> np.ndarray:
            return greedy_agent.select_actions(env=_env, obs_x=obs.x, max_tx=int(max(1, args.baseline_max_tx)))

        def tdma_policy(_env: LAWNEnv, obs, rng: np.random.Generator) -> np.ndarray:
            return tdma_agent.select_actions(obs.x, rng, max_tx=int(max(1, args.baseline_max_tx)))

        def csma_policy(env: LAWNEnv, obs, rng: np.random.Generator) -> np.ndarray:
            return csma_agent.select_actions(obs.x, env.last_status, rng, max_tx=int(max(1, args.baseline_max_tx)))

        def satmac_policy(env: LAWNEnv, obs, rng: np.random.Generator) -> np.ndarray:
            return satmac_agent.select_actions(obs.x, env.last_status, rng, max_tx=int(max(1, args.baseline_max_tx)))

        def hsatmac_policy(env: LAWNEnv, obs, rng: np.random.Generator) -> np.ndarray:
            return hsatmac_agent.select_actions(env=env, obs=obs, rng=rng, max_tx=int(min(2, max(1, getattr(cfg, "max_tx_per_frame", 1)))))

        csma_policy.reset = csma_agent.reset  # type: ignore[attr-defined]
        satmac_policy.reset = satmac_agent.reset  # type: ignore[attr-defined]
        hsatmac_policy.reset = hsatmac_agent.reset  # type: ignore[attr-defined]

        policies = {
            "GAC-MAC": gac_policy,
            "Random": random_policy,
            "ALOHA": aloha_policy,
            "Greedy": greedy_policy,
            "TDMA": tdma_policy,
            "CSMA": csma_policy,
            "SATMAC": satmac_policy,
            "H-SAT": hsatmac_policy,
        }

        for name, policy_fn in policies.items():
            ep = {k: [] for k in keys}
            rng = np.random.default_rng(int(args.seed))
            for e in range(int(args.episodes)):
                seed = int(args.seed + e)
                obs = env.reset(seed=seed)
                if hasattr(policy_fn, "reset"):
                    policy_fn.reset()  # type: ignore[attr-defined]
                thr_sum = 0.0
                coll_sum = 0.0
                jain_sum = 0.0
                delay_sum = 0.0
                attempts_sum = 0.0
                success_sum = 0.0
                collisions_sum = 0.0
                max_group_sum = 0.0
                multi_res_sum = 0.0
                for _t in range(cfg.episode_len):
                    act = policy_fn(env, obs, rng)
                    obs, _rew, done, info = env.step(act)
                    thr_sum += float(info.get("sum_rate_mbps", info.get("sum_rate", 0.0)))
                    coll_sum += float(info.get("collision_rate", 0.0))
                    jain_sum += float(info.get("jain", 0.0))
                    delay_sum += float(info.get("avg_delay", 0.0))
                    attempts_sum += float(info.get("tx_attempts", 0.0))
                    success_sum += float(info.get("tx_success", 0.0))
                    collisions_sum += float(info.get("tx_collision", 0.0))
                    max_group_sum += float(info.get("max_group_size", 0.0))
                    multi_res_sum += float(info.get("num_multi_tx_resources", 0.0))
                    if done:
                        break
                denom = float(cfg.episode_len)
                ep["sum_rate_mbps"].append(thr_sum / denom)
                ep["collision_rate"].append(coll_sum / denom)
                ep["jain"].append(jain_sum / denom)
                ep["avg_delay"].append(delay_sum / denom)
                ep["tx_attempts"].append(attempts_sum / denom)
                ep["tx_success"].append(success_sum / denom)
                ep["tx_collision"].append(collisions_sum / denom)
                ep["max_group_size"].append(max_group_sum / denom)
                ep["num_multi_tx_resources"].append(multi_res_sum / denom)

            for k in keys:
                results_by_algo[name][k].append(float(np.mean(ep[k])))

        print(
            f"N={n:3d} dens={density_km2:7.2f}/km^2 | "
            f"GAC thr {results_by_algo['GAC-MAC']['sum_rate_mbps'][-1]:.1f} coll {results_by_algo['GAC-MAC']['collision_rate'][-1]:.3f} | "
            f"Greedy thr {results_by_algo['Greedy']['sum_rate_mbps'][-1]:.1f} coll {results_by_algo['Greedy']['collision_rate'][-1]:.3f}",
            flush=True,
        )

    run_dir = make_run_dir(args.results_dir, args.run_name)
    out_base = os.path.abspath(args.out) if args.out else os.path.join(run_dir, "density")
    # If args.out is a png path, strip extension to form a base path.
    if out_base.lower().endswith(".png"):
        out_base = os.path.splitext(out_base)[0]
    out_json = out_base + ".json"
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": ckpt_path,
                "config": asdict(base_cfg),
                "xs_n": xs_n,
                "xs_density_per_km2": xs_density_km2,
                "results_by_algo": results_by_algo,
            },
            f,
            indent=2,
        )

    out_files: list[str] = []
    if args.plot_style == "lines":
        from gac_mac.viz.plots import plot_density_lines

        out_png = out_base + "_lines.png"
        plot_density_lines(xs=xs_density_km2, results_by_algo=results_by_algo, save_path=out_png)
        out_files.append(out_png)
    else:
        from gac_mac.viz.plots import plot_density_facets

        plot_density_facets(
            xs=xs_density_km2,
            results_by_algo=results_by_algo,
            metric="sum_rate_mbps",
            y_label="Throughput (Mbps/frame)",
            save_path=out_base + "_thr.png",
            sort_by_last=True,
        )
        out_files.append(out_base + "_thr.png")
        plot_density_facets(
            xs=xs_density_km2,
            results_by_algo=results_by_algo,
            metric="collision_rate",
            y_label="Collision rate",
            y_lim=(0.0, 1.0),
            save_path=out_base + "_coll.png",
            sort_by_last=False,
        )
        out_files.append(out_base + "_coll.png")
        plot_density_facets(
            xs=xs_density_km2,
            results_by_algo=results_by_algo,
            metric="jain",
            y_label="Jain",
            y_lim=(0.0, 1.0),
            save_path=out_base + "_jain.png",
            sort_by_last=False,
        )
        out_files.append(out_base + "_jain.png")
    for p in out_files:
        print(f"saved: {p}")
    print(f"saved: {out_json}")


if __name__ == "__main__":
    main()
