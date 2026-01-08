from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict

import numpy as np

from gac_mac.algo.buffer import RolloutBuffer
from gac_mac.algo.mappo import MAPPOTrainer
from gac_mac.config import Config, config_from_dict
from gac_mac.env.channel import ChannelModel
from gac_mac.env.lawn_env import LAWNEnv
from gac_mac.env.mobility import GaussMarkovMobility
from gac_mac.env.toy_env import ToyLawnEnv
from gac_mac.models.agent import GACMACAgent
from gac_mac.utils.checkpoint import load_checkpoint, save_checkpoint
from gac_mac.utils.paths import make_run_dir
from gac_mac.utils.seed import seed_everything


def _resolve_device(device: str) -> "object":
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train GAC-MAC (toy or LAWN env).")
    p.add_argument("--env", type=str, default="lawn", choices=["toy", "lawn"])
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None, choices=["auto", "cpu", "cuda"])
    p.add_argument("--results-dir", type=str, default=None)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--cs-threshold-dbm", type=float, default=None)
    p.add_argument("--graph-mode", type=str, default=None, choices=["cs", "conflict"])
    p.add_argument("--obs-version", type=str, default=None, choices=["v1", "v2", "v3"])
    p.add_argument("--no-edge-attr", action="store_true", help="Disable edge_attr usage in GAT (v1-style).")
    p.add_argument("--total-updates", type=int, default=None)
    p.add_argument("--steps-per-update", type=int, default=None)
    p.add_argument("--log-interval", type=int, default=None)
    p.add_argument("--checkpoint-interval", type=int, default=None)
    p.add_argument("--hidden-dim", type=int, default=None)
    p.add_argument("--gat-heads", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--entropy-coef", type=float, default=None)
    p.add_argument("--target-kl", type=float, default=None, help="Early stop PPO epochs if approx_kl exceeds this.")
    p.add_argument("--distill-coef", type=float, default=None, help="Greedy-teacher distillation coefficient (0=off).")
    p.add_argument("--reward-mode", type=str, default=None, choices=["binary", "rate"])
    p.add_argument("--reward-collision", type=float, default=None)
    p.add_argument("--reward-idle-nonempty", type=float, default=None)
    p.add_argument("--reward-tx-attempt", type=float, default=None)
    p.add_argument("--lambda-coop", type=float, default=None)
    p.add_argument("--num-envs", type=int, default=None, help="Vectorized rollout envs (>=1).")
    p.add_argument("--parallel-env", action="store_true", help="Use subprocess envs when --num-envs>1.")
    p.add_argument("--critic-mode", type=str, default=None, choices=["node", "global_mean", "global"])
    p.add_argument("--no-action-mask", action="store_true", help="Disable queue-based action masking.")
    p.add_argument("--policy-mode", type=str, default=None, choices=["flat", "hierarchical"])
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pth")
    p.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard logging.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    base_cfg = Config()
    cfg = base_cfg

    resume_path = None
    resume_ckpt = None
    if args.resume:
        resume_path = os.path.abspath(args.resume)
        resume_ckpt = load_checkpoint(resume_path)
        cfg_dict = resume_ckpt.get("config", asdict(base_cfg))
        # Backward-compat: older checkpoints (v1) did not include obs/model versioning fields.
        if "obs_version" not in cfg_dict:
            cfg_dict = {**cfg_dict, "obs_version": "v1", "use_edge_attr": False}
        cfg = config_from_dict(cfg_dict)

    if args.seed is not None:
        cfg = cfg.__class__(**{**asdict(cfg), "seed": args.seed})
    if args.device is not None:
        cfg = cfg.__class__(**{**asdict(cfg), "device": args.device})
    if args.results_dir is not None:
        cfg = cfg.__class__(**{**asdict(cfg), "results_dir": args.results_dir})
    if args.run_name is not None:
        cfg = cfg.__class__(**{**asdict(cfg), "run_name": args.run_name})
    if args.cs_threshold_dbm is not None:
        cfg = cfg.__class__(**{**asdict(cfg), "cs_threshold_dbm": float(args.cs_threshold_dbm)})
    if args.graph_mode is not None:
        cfg = cfg.__class__(**{**asdict(cfg), "graph_mode": str(args.graph_mode)})
    if args.obs_version is not None:
        cfg = cfg.__class__(**{**asdict(cfg), "obs_version": str(args.obs_version)})
    if args.no_edge_attr:
        cfg = cfg.__class__(**{**asdict(cfg), "use_edge_attr": False})
    if args.total_updates is not None:
        cfg = cfg.__class__(**{**asdict(cfg), "total_updates": args.total_updates})
    if args.steps_per_update is not None:
        cfg = cfg.__class__(**{**asdict(cfg), "steps_per_update": args.steps_per_update})
    if args.log_interval is not None:
        cfg = cfg.__class__(**{**asdict(cfg), "log_interval": args.log_interval})
    if args.checkpoint_interval is not None:
        cfg = cfg.__class__(**{**asdict(cfg), "checkpoint_interval": args.checkpoint_interval})
    if args.hidden_dim is not None:
        cfg = cfg.__class__(**{**asdict(cfg), "hidden_dim": int(args.hidden_dim)})
    if args.gat_heads is not None:
        cfg = cfg.__class__(**{**asdict(cfg), "gat_heads": int(args.gat_heads)})
    if args.lr is not None:
        cfg = cfg.__class__(**{**asdict(cfg), "lr": float(args.lr)})
    if args.entropy_coef is not None:
        cfg = cfg.__class__(**{**asdict(cfg), "entropy_coef": float(args.entropy_coef)})
    if args.target_kl is not None:
        cfg = cfg.__class__(**{**asdict(cfg), "target_kl": float(args.target_kl)})
    if args.distill_coef is not None:
        cfg = cfg.__class__(**{**asdict(cfg), "distill_coef": float(args.distill_coef)})
    if args.reward_mode is not None:
        cfg = cfg.__class__(**{**asdict(cfg), "reward_mode": str(args.reward_mode)})
    if args.reward_collision is not None:
        cfg = cfg.__class__(**{**asdict(cfg), "reward_collision": float(args.reward_collision)})
    if args.reward_idle_nonempty is not None:
        cfg = cfg.__class__(**{**asdict(cfg), "reward_idle_nonempty": float(args.reward_idle_nonempty)})
    if args.reward_tx_attempt is not None:
        cfg = cfg.__class__(**{**asdict(cfg), "reward_tx_attempt": float(args.reward_tx_attempt)})
    if args.lambda_coop is not None:
        cfg = cfg.__class__(**{**asdict(cfg), "lambda_coop": float(args.lambda_coop)})
    if args.num_envs is not None:
        cfg = cfg.__class__(**{**asdict(cfg), "num_envs": int(args.num_envs)})
    if args.critic_mode is not None:
        cfg = cfg.__class__(**{**asdict(cfg), "critic_mode": str(args.critic_mode)})
    if args.no_action_mask:
        cfg = cfg.__class__(**{**asdict(cfg), "action_mask_empty_queue": False})
    if args.policy_mode is not None:
        cfg = cfg.__class__(**{**asdict(cfg), "policy_mode": str(args.policy_mode)})

    seed_everything(cfg.seed)
    device = _resolve_device(cfg.device)

    if resume_path:
        run_dir = os.path.dirname(os.path.dirname(resume_path))
    else:
        run_dir = make_run_dir(cfg.results_dir, cfg.run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Run dir: {run_dir} | device: {device} | seed: {cfg.seed}", flush=True)
    print(
        f"updates={cfg.total_updates} steps/update={cfg.steps_per_update} "
        f"log_interval={cfg.log_interval} ckpt_interval={cfg.checkpoint_interval}",
        flush=True,
    )

    writer = None
    if not args.no_tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        tb_dir = os.path.join(run_dir, "tb")
        writer = SummaryWriter(log_dir=tb_dir)
        writer.add_text("config/json", json.dumps(asdict(cfg), indent=2, sort_keys=True))
        print(f"TensorBoard: tensorboard --logdir {tb_dir}", flush=True)

    if args.env == "toy":
        env = ToyLawnEnv(
            num_uavs=cfg.num_uavs,
            num_slots=cfg.num_slots,
            map_size_m=cfg.map_size_m,
            height_m=cfg.height_m,
            neighbor_radius_m=cfg.neighbor_radius_m,
            max_queue_len=cfg.max_queue_len,
            lambda_arrival_per_slot=cfg.lambda_arrival_per_slot,
            episode_len=cfg.episode_len,
            reward_success=cfg.reward_success,
            reward_collision=cfg.reward_collision,
            reward_idle_empty=cfg.reward_idle_empty,
            reward_idle_nonempty=cfg.reward_idle_nonempty,
            reward_tx_attempt=getattr(cfg, "reward_tx_attempt", 0.0),
            reward_repeat_success=cfg.reward_repeat_success,
            reward_repeat_collision=cfg.reward_repeat_collision,
            lambda_coop=cfg.lambda_coop,
        )
    else:
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
        env = LAWNEnv(
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
            reward_tx_attempt=getattr(cfg, "reward_tx_attempt", 0.0),
            reward_repeat_success=cfg.reward_repeat_success,
            reward_repeat_collision=cfg.reward_repeat_collision,
            lambda_coop=cfg.lambda_coop,
        )

    num_envs = int(max(1, cfg.num_envs))
    if getattr(cfg, "distill_coef", 0.0) > 0.0 and args.env != "lawn":
        raise ValueError("--distill-coef currently supports only --env lawn (greedy teacher uses PHY state).")

    vec_env = None
    if num_envs > 1:
        from gac_mac.env.vector_env import SubprocVectorEnv, SyncVectorEnv

        vec_env = (
            SubprocVectorEnv(cfg, env=args.env, num_envs=num_envs)
            if args.parallel_env
            else SyncVectorEnv(cfg, env=args.env, num_envs=num_envs)
        )
        obs_list = vec_env.reset(seed=cfg.seed)
        input_dim = int(obs_list[0].x.shape[1])
    else:
        obs = env.reset(seed=cfg.seed)
        input_dim = int(obs.x.shape[1])
    action_dim = cfg.num_slots + 1

    import torch

    if getattr(cfg, "distill_coef", 0.0) > 0.0 and num_envs == 1 and args.env == "lawn":
        # Single env case: enable teacher labels in env.step() info.
        env.enable_greedy_teacher = True  # type: ignore[attr-defined]

    agent = GACMACAgent(
        input_dim=input_dim,
        hidden_dim=cfg.hidden_dim,
        action_dim=action_dim,
        gat_heads=cfg.gat_heads,
        use_edge_attr=cfg.use_edge_attr,
        critic_mode=cfg.critic_mode,
        policy_mode=cfg.policy_mode,
    ).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.lr)

    trainer = MAPPOTrainer(
        agent=agent,
        optimizer=optimizer,
        clip_eps=cfg.clip_eps,
        ppo_epochs=cfg.ppo_epochs,
        bptt_len=cfg.bptt_len,
        value_loss_coef=cfg.value_loss_coef,
        entropy_coef=cfg.entropy_coef,
        target_kl=getattr(cfg, "target_kl", None),
        distill_coef=getattr(cfg, "distill_coef", 0.0),
        max_grad_norm=cfg.max_grad_norm,
        device=device,
    )

    start_update = 0
    global_step = 0
    rolling = {"reward": [], "sum_rate": [], "collision_rate": [], "avg_delay": []}
    best = {"throughput_mbps_ma10": float("-inf"), "update": -1}

    if resume_ckpt is not None:
        missing, unexpected = agent.load_state_dict(resume_ckpt["agent_state_dict"], strict=False)
        if missing or unexpected:
            print(f"[resume] missing keys: {len(missing)} | unexpected keys: {len(unexpected)}", flush=True)
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        # Respect any CLI override to lr after loading optimizer state.
        for pg in optimizer.param_groups:
            pg["lr"] = cfg.lr
        start_update = int(resume_ckpt.get("update", 0)) + 1
        global_step = int(resume_ckpt.get("global_step", 0))
        if isinstance(resume_ckpt.get("rolling"), dict):
            rolling = resume_ckpt["rolling"]
        if isinstance(resume_ckpt.get("best"), dict):
            best = resume_ckpt["best"]

    h = agent.initial_hidden(cfg.num_uavs * num_envs, device)
    buffer = RolloutBuffer()

    def _stack_obs(obs_batch):
        xs = []
        eis = []
        eas = []
        offset = 0
        for ob in obs_batch:
            x_t = torch.tensor(ob.x, dtype=torch.float32, device=device)
            ei_t = torch.tensor(ob.edge_index, dtype=torch.long, device=device)
            ea_t = torch.tensor(ob.edge_attr, dtype=torch.float32, device=device)
            if ei_t.numel() > 0:
                ei_t = ei_t + offset
            xs.append(x_t)
            eis.append(ei_t)
            eas.append(ea_t)
            offset += int(x_t.size(0))
        x = torch.cat(xs, dim=0) if xs else torch.zeros((0, 0), dtype=torch.float32, device=device)
        edge_index = torch.cat(eis, dim=1) if eis else torch.zeros((2, 0), dtype=torch.long, device=device)
        edge_attr = torch.cat(eas, dim=0) if eas else torch.zeros((0, 1), dtype=torch.float32, device=device)
        return x, edge_index, edge_attr

    for update in range(start_update, cfg.total_updates):
        buffer.reset()

        # Collect on-policy rollout
        rollout_reward = []
        rollout_sum_rate = []
        rollout_thr_mbps = []
        rollout_collision = []
        rollout_delay = []
        rollout_degree = []
        rollout_attempts = []
        rollout_success = []
        rollout_collisions = []
        rollout_no_tx_frac_has_pkt = []
        rollout_has_pkt_total = 0
        rollout_action_counts_has_pkt = np.zeros((cfg.num_slots + 1,), dtype=np.int64)

        for _ in range(cfg.steps_per_update):
            if num_envs > 1:
                x, edge_index, edge_attr = _stack_obs(obs_list)
            else:
                x = torch.tensor(obs.x, dtype=torch.float32, device=device)
                edge_index = torch.tensor(obs.edge_index, dtype=torch.long, device=device)
                edge_attr = torch.tensor(obs.edge_attr, dtype=torch.float32, device=device)

            teacher_actions = None

            h_in = h.detach()
            with torch.no_grad():
                action_mask = None
                if cfg.action_mask_empty_queue and x.numel() > 0:
                    has_pkt = x[:, 3] > 0.0
                    if bool((~has_pkt).any().item()):
                        action_mask = torch.ones((x.size(0), action_dim), dtype=torch.bool, device=device)
                        action_mask[~has_pkt, : cfg.num_slots] = False
                action, logp, values, _entropy, h_out = agent.act(x, edge_index, edge_attr, h_in, action_mask=action_mask)

            if num_envs > 1:
                act_np = action.view(num_envs, cfg.num_uavs).cpu().numpy()
                next_obs_list, rewards_np, dones_env, infos = vec_env.step(act_np)  # type: ignore[union-attr]
                rewards = torch.tensor(rewards_np.reshape(-1), dtype=torch.float32, device=device)
                done_mask_np = np.repeat(dones_env.reshape(-1, 1), cfg.num_uavs, axis=1).reshape(-1).astype(
                    np.float32
                )
                done_mask = torch.tensor(done_mask_np, dtype=torch.float32, device=device)
                info_rate = float(np.mean([i.get("sum_rate", 0.0) for i in infos]))
                info_thr = float(np.mean([i.get("throughput_mbps", 0.0) for i in infos]))
                info_coll = float(np.mean([i.get("collision_rate", 0.0) for i in infos]))
                info_delay = float(np.mean([i.get("avg_delay", 0.0) for i in infos]))
                info_deg = float(np.mean([i.get("avg_degree_in", 0.0) for i in infos]))
                info_attempts = float(np.mean([i.get("tx_attempts", 0.0) for i in infos]))
                info_success = float(np.mean([i.get("tx_success", 0.0) for i in infos]))
                info_collisions = float(np.mean([i.get("tx_collision", 0.0) for i in infos]))
                if getattr(cfg, "distill_coef", 0.0) > 0.0:
                    ta_list = [i.get("teacher_actions", None) for i in infos]
                    if any(t is None for t in ta_list):
                        raise RuntimeError("distill enabled but teacher_actions missing from vector env info.")
                    teacher_actions = torch.tensor(np.stack(ta_list, axis=0).reshape(-1), dtype=torch.long, device=device)
            else:
                next_obs, rewards_np, done, info = env.step(action.cpu().numpy())
                rewards = torch.tensor(rewards_np, dtype=torch.float32, device=device)
                done_mask = torch.full_like(rewards, 1.0 if done else 0.0)
                info_rate = float(info.get("sum_rate", 0.0))
                info_thr = float(info.get("throughput_mbps", 0.0))
                info_coll = float(info.get("collision_rate", 0.0))
                info_delay = float(info.get("avg_delay", 0.0))
                info_deg = float(info.get("avg_degree_in", 0.0))
                info_attempts = float(info.get("tx_attempts", 0.0))
                info_success = float(info.get("tx_success", 0.0))
                info_collisions = float(info.get("tx_collision", 0.0))
                if getattr(cfg, "distill_coef", 0.0) > 0.0:
                    ta = info.get("teacher_actions", None)
                    if ta is None:
                        raise RuntimeError("distill enabled but teacher_actions missing from env info.")
                    teacher_actions = torch.tensor(np.asarray(ta, dtype=np.int64).reshape(-1), dtype=torch.long, device=device)

            with torch.no_grad():
                has_pkt = x[:, 3] > 0.0
                denom = float(max(1, int(has_pkt.sum().item())))
                if denom > 0:
                    no_tx = (action == cfg.num_slots) & has_pkt
                    rollout_no_tx_frac_has_pkt.append(float(no_tx.sum().item() / denom))
                # Accumulate action histogram conditioned on having a packet.
                n_has = int(has_pkt.sum().item())
                if n_has > 0:
                    rollout_has_pkt_total += n_has
                    for a in range(cfg.num_slots + 1):
                        rollout_action_counts_has_pkt[a] += int(((action == a) & has_pkt).sum().item())

            buffer.add(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                actions=action,
                logp=logp,
                values=values,
                rewards=rewards,
                done=done_mask,
                h_in=h_in,
                teacher_actions=teacher_actions,
            )

            rollout_reward.append(float(rewards.mean().item()))
            rollout_sum_rate.append(info_rate)
            rollout_thr_mbps.append(info_thr)
            rollout_collision.append(info_coll)
            rollout_delay.append(info_delay)
            rollout_degree.append(info_deg)
            rollout_attempts.append(info_attempts)
            rollout_success.append(info_success)
            rollout_collisions.append(info_collisions)

            h = h_out.detach()
            h = h * (1.0 - done_mask.view(-1, 1))
            global_step += num_envs

            if num_envs > 1:
                obs_list = next_obs_list
            else:
                obs = next_obs
                if done:
                    obs = env.reset()
                    h = agent.initial_hidden(cfg.num_uavs, device)

        # Bootstrap value at final state
        with torch.no_grad():
            if num_envs > 1:
                x_last, ei_last, ea_last = _stack_obs(obs_list)
            else:
                x_last = torch.tensor(obs.x, dtype=torch.float32, device=device)
                ei_last = torch.tensor(obs.edge_index, dtype=torch.long, device=device)
                ea_last = torch.tensor(obs.edge_attr, dtype=torch.float32, device=device)
            last_values, _ = agent.value_only(x_last, ei_last, ea_last, h.detach())

        buffer.compute_gae(last_values=last_values, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda, normalize=True)
        stats = trainer.update(buffer.as_batch())

        rolling["reward"].append(float(np.mean(rollout_reward)))
        rolling["sum_rate"].append(float(np.mean(rollout_sum_rate)))
        rolling.setdefault("throughput_mbps", []).append(float(np.mean(rollout_thr_mbps)) if rollout_thr_mbps else 0.0)
        rolling["collision_rate"].append(float(np.mean(rollout_collision)))
        rolling["avg_delay"].append(float(np.mean(rollout_delay)))

        # Track best checkpoint by smoothed throughput (moving average over last 10 updates).
        thr_series = rolling.get("throughput_mbps", [])
        if thr_series:
            ma10 = float(np.mean(thr_series[-10:]))
            if ma10 > float(best.get("throughput_mbps_ma10", float("-inf"))):
                best = {"throughput_mbps_ma10": ma10, "update": int(update + 1)}
                best_path = os.path.join(ckpt_dir, "checkpoint_best.pth")
                save_checkpoint(
                    best_path,
                    {
                        "config": asdict(cfg),
                        "update": update,
                        "global_step": global_step,
                        "agent_state_dict": agent.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "rolling": rolling,
                        "best": best,
                    },
                )

        if writer is not None:
            writer.add_scalar("train/reward_mean", rolling["reward"][-1], update + 1)
            writer.add_scalar("train/sum_rate_mean", rolling["sum_rate"][-1], update + 1)
            if "throughput_mbps" in rolling:
                writer.add_scalar("train/throughput_mbps_mean", rolling["throughput_mbps"][-1], update + 1)
            writer.add_scalar("train/collision_rate_mean", rolling["collision_rate"][-1], update + 1)
            writer.add_scalar("train/avg_delay_mean", rolling["avg_delay"][-1], update + 1)

            if rollout_degree:
                writer.add_scalar("env/avg_degree_in", float(np.mean(rollout_degree)), update + 1)

            if rollout_attempts:
                writer.add_scalar("mac/tx_attempts", float(np.mean(rollout_attempts)), update + 1)
                writer.add_scalar("mac/tx_success", float(np.mean(rollout_success)), update + 1)
                writer.add_scalar("mac/tx_collision", float(np.mean(rollout_collisions)), update + 1)

            if rollout_no_tx_frac_has_pkt:
                writer.add_scalar(
                    "policy/no_tx_frac_given_queue",
                    float(np.mean(rollout_no_tx_frac_has_pkt)),
                    update + 1,
                )

            if rollout_has_pkt_total > 0:
                denom = float(rollout_has_pkt_total)
                writer.add_scalar(
                    "policy/tx_frac_given_queue",
                    float(rollout_action_counts_has_pkt[: cfg.num_slots].sum() / denom),
                    update + 1,
                )
                writer.add_scalar(
                    "policy/no_tx_frac_given_queue_hist",
                    float(rollout_action_counts_has_pkt[cfg.num_slots] / denom),
                    update + 1,
                )
                for s in range(cfg.num_slots):
                    writer.add_scalar(
                        f"policy/slot_frac_given_queue/{s}",
                        float(rollout_action_counts_has_pkt[s] / denom),
                        update + 1,
                    )

            writer.add_scalar("ppo/approx_kl", stats.approx_kl, update + 1)
            writer.add_scalar("ppo/clip_frac", stats.clip_frac, update + 1)
            writer.add_scalar("value/explained_variance", stats.explained_variance, update + 1)
            writer.add_scalar("optim/grad_norm", stats.grad_norm, update + 1)

            writer.add_scalar("loss/total", stats.loss, update + 1)
            writer.add_scalar("loss/actor", stats.actor_loss, update + 1)
            writer.add_scalar("loss/value", stats.value_loss, update + 1)
            writer.add_scalar("loss/entropy", stats.entropy, update + 1)

        if (update + 1) % cfg.log_interval == 0:
            print(
                f"upd {update+1:04d} | step {global_step:07d} | "
                f"R {rolling['reward'][-1]:+.3f} | "
                f"rate {rolling['sum_rate'][-1]:.2f} (thr {rolling.get('throughput_mbps',[0])[-1]:.2f} Mbps) | "
                f"coll {rolling['collision_rate'][-1]:.3f} | "
                f"best_ma10 {float(best.get('throughput_mbps_ma10', 0.0)):.2f} Mbps@{int(best.get('update', -1))} | "
                f"loss {stats.loss:.3f} (pi {stats.actor_loss:.3f}, v {stats.value_loss:.3f}, ent {stats.entropy:.3f})",
                flush=True,
            )

        if (update + 1) % cfg.checkpoint_interval == 0 or (update + 1) == cfg.total_updates:
            ckpt_path = os.path.join(ckpt_dir, f"checkpoint_{update+1:04d}.pth")
            save_checkpoint(
                ckpt_path,
                {
                    "config": asdict(cfg),
                    "update": update,
                    "global_step": global_step,
                    "agent_state_dict": agent.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "rolling": rolling,
                    "best": best,
                },
            )

    if vec_env is not None:
        vec_env.close()

    # Save a quick training-curve figure for convenience.
    try:
        from gac_mac.viz.plots import plot_training_curves

        plot_training_curves(rolling, os.path.join(run_dir, "training_curve.png"))
    except Exception as exc:
        print(f"[warn] skip plotting: {exc}", flush=True)

    if writer is not None:
        writer.flush()
        writer.close()

    print(f"Done. Outputs in: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
