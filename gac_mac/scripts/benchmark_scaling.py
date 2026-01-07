from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Any

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
    p.add_argument("--no-action-mask", action="store_true", help="Disable queue-based action masking.")
    p.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["fixed_area", "fixed_density", "both"],
        help="Scaling mode: fixed_area keeps map_size constant; fixed_density expands map_size with N; both runs both.",
    )
    p.add_argument(
        "--density-per-km2",
        type=float,
        default=None,
        help="Target density for fixed_density mode (nodes/km^2). Default uses checkpoint's N and map_size.",
    )
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


def _run_episode(env: LAWNEnv, policy_fn, *, seed: int) -> tuple[float, float, float, float]:
    rng = np.random.default_rng(seed)
    obs = env.reset(seed=seed)
    if hasattr(policy_fn, "reset"):
        policy_fn.reset()  # type: ignore[attr-defined]

    rate_sum = 0.0
    thr_sum = 0.0
    coll_sum = 0.0
    delay_sum = 0.0

    for _t in range(env.episode_len):
        actions = policy_fn(env, obs, rng)
        obs, _rew, done, info = env.step(actions)
        rate_sum += float(info["sum_rate"])
        thr_sum += float(info.get("throughput_mbps", 0.0))
        coll_sum += float(info["collision_rate"])
        delay_sum += float(info["avg_delay"])
        if done:
            break
    denom = float(env.episode_len)
    return rate_sum / denom, thr_sum / denom, coll_sum / denom, delay_sum / denom


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
        critic_mode=getattr(base_cfg, "critic_mode", "global_mean"),
        policy_mode=getattr(base_cfg, "policy_mode", "hierarchical"),
    ).to(device)
    agent.load_state_dict(ckpt["agent_state_dict"], strict=False)
    agent.eval()

    greedy = GreedyColoringAgent(base_cfg.num_slots)

    def run_mode(
        mode_name: str, *, cfg_for_n, x_values: list[float], x_label: str, out_png: str
    ) -> dict[str, dict[str, list[float]]]:
        results_by_algo: dict[str, dict[str, list[float]]] = {
            "GAC-MAC": {"sum_rate": [], "throughput_mbps": [], "collision_rate": [], "avg_delay": []},
            "Greedy": {"sum_rate": [], "throughput_mbps": [], "collision_rate": [], "avg_delay": []},
        }

        for idx, n in enumerate(ns):
            cfg = cfg_for_n(int(n))
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
                        act = torch.argmax(logits, dim=-1)
                gac_policy.h = h_out.detach()
                return act.cpu().numpy()

            def _gac_reset() -> None:
                gac_policy.h = agent.initial_hidden(cfg.num_uavs, device)

            gac_policy.reset = _gac_reset  # type: ignore[attr-defined]

            def greedy_policy(_env: LAWNEnv, obs, _rng: np.random.Generator) -> np.ndarray:
                return greedy.select_actions(env=_env, obs_x=obs.x)

            rates = []
            thrs = []
            colls = []
            delays = []
            for ep in range(args.episodes):
                r, thr, c, d = _run_episode(env, gac_policy, seed=int(args.seed + ep))
                rates.append(r)
                thrs.append(thr)
                colls.append(c)
                delays.append(d)
            results_by_algo["GAC-MAC"]["sum_rate"].append(float(np.mean(rates)))
            results_by_algo["GAC-MAC"]["throughput_mbps"].append(float(np.mean(thrs)))
            results_by_algo["GAC-MAC"]["collision_rate"].append(float(np.mean(colls)))
            results_by_algo["GAC-MAC"]["avg_delay"].append(float(np.mean(delays)))

            rates = []
            thrs = []
            colls = []
            delays = []
            for ep in range(args.episodes):
                r, thr, c, d = _run_episode(env, greedy_policy, seed=int(args.seed + ep))
                rates.append(r)
                thrs.append(thr)
                colls.append(c)
                delays.append(d)
            results_by_algo["Greedy"]["sum_rate"].append(float(np.mean(rates)))
            results_by_algo["Greedy"]["throughput_mbps"].append(float(np.mean(thrs)))
            results_by_algo["Greedy"]["collision_rate"].append(float(np.mean(colls)))
            results_by_algo["Greedy"]["avg_delay"].append(float(np.mean(delays)))

            print(
                f"[{mode_name}] x={x_values[idx]:.3f} N={n:3d} | "
                f"GAC {results_by_algo['GAC-MAC']['throughput_mbps'][-1]:.3f} Mbps "
                f"(sum_rate {results_by_algo['GAC-MAC']['sum_rate'][-1]:.3f}) "
                f"coll {results_by_algo['GAC-MAC']['collision_rate'][-1]:.3f} delay {results_by_algo['GAC-MAC']['avg_delay'][-1]:.3f} || "
                f"Greedy {results_by_algo['Greedy']['throughput_mbps'][-1]:.3f} Mbps "
                f"(sum_rate {results_by_algo['Greedy']['sum_rate'][-1]:.3f}) "
                f"coll {results_by_algo['Greedy']['collision_rate'][-1]:.3f} delay {results_by_algo['Greedy']['avg_delay'][-1]:.3f}"
            )

        from gac_mac.viz.plots import plot_scaling_lines

        plot_scaling_lines(xs=x_values, results_by_algo=results_by_algo, save_path=out_png, x_label=x_label)
        return results_by_algo

    run_dir = os.path.dirname(os.path.dirname(ckpt_path))
    os.makedirs(run_dir, exist_ok=True)

    # Fixed-area mode: map_size is constant, so density increases with N.
    results_out: dict[str, Any] = {"ns": ns, "modes": {}}
    if args.mode in {"fixed_area", "both"}:
        area_m2 = float(base_cfg.map_size_m) * float(base_cfg.map_size_m)
        densities_per_km2 = [float(n) / (area_m2 / 1e6) for n in ns]

        def cfg_fixed_area(n: int) -> Config:
            return base_cfg.__class__(**{**asdict(base_cfg), "num_uavs": int(n)})

        out_png = args.out or os.path.join(run_dir, "scaling_fixed_area.png")
        res = run_mode(
            "fixed_area",
            cfg_for_n=cfg_fixed_area,
            x_values=densities_per_km2,
            x_label="density (nodes/km^2)",
            out_png=out_png,
        )
        results_out["modes"]["fixed_area"] = {
            "x_label": "density (nodes/km^2)",
            "x": densities_per_km2,
            "results_by_algo": res,
            "map_size_m": base_cfg.map_size_m,
        }
        print(f"saved: {out_png}")

    # Fixed-density mode: expand map_size with N to keep density constant.
    if args.mode in {"fixed_density", "both"}:
        if args.density_per_km2 is None:
            base_area_km2 = (float(base_cfg.map_size_m) * float(base_cfg.map_size_m)) / 1e6
            density_per_km2 = float(base_cfg.num_uavs) / max(1e-12, base_area_km2)
        else:
            density_per_km2 = float(args.density_per_km2)
        rho_m2 = density_per_km2 / 1e6

        map_sizes = [float(np.sqrt(float(n) / max(1e-12, rho_m2))) for n in ns]

        def cfg_fixed_density(n: int) -> Config:
            L = float(np.sqrt(float(n) / max(1e-12, rho_m2)))
            return base_cfg.__class__(**{**asdict(base_cfg), "num_uavs": int(n), "map_size_m": L})

        out_png = args.out or os.path.join(run_dir, "scaling_fixed_density.png")
        res = run_mode(
            "fixed_density",
            cfg_for_n=cfg_fixed_density,
            x_values=[float(n) for n in ns],
            x_label="N (UAVs)",
            out_png=out_png,
        )
        results_out["modes"]["fixed_density"] = {
            "x_label": "N (UAVs)",
            "x": [float(n) for n in ns],
            "results_by_algo": res,
            "density_per_km2": density_per_km2,
            "map_size_m": map_sizes,
        }
        print(f"saved: {out_png}")

    out_json = os.path.join(run_dir, "scaling_both.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results_out, f, indent=2)
    print(f"saved: {out_json}")


if __name__ == "__main__":
    main()
