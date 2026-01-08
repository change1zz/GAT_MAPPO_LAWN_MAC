from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict

import numpy as np

from gac_mac.algo.buffer import RolloutBuffer
from gac_mac.algo.mappo import MAPPOTrainer
from gac_mac.baselines.greedy_coloring import GreedyColoringAgent
from gac_mac.config import Config
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
    p = argparse.ArgumentParser(description="Train GAC-MAC (M1 toy env).")
    p.add_argument("--env", type=str, default="toy", choices=["toy", "lawn"])
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
    p.add_argument("--conflict-loss-coef", type=float, default=None)
    p.add_argument("--agent-id-tiebreak-eps", type=float, default=None)
    p.add_argument("--reward-mode", type=str, default=None, choices=["binary", "rate"])
    p.add_argument("--reward-collision", type=float, default=None)
    p.add_argument("--reward-idle-nonempty", type=float, default=None)
    p.add_argument("--lambda-coop", type=float, default=None)
    p.add_argument("--num-envs", type=int, default=None, help="Vectorized rollout envs (>=1).")
    p.add_argument("--parallel-env", action="store_true", help="Use subprocess envs when --num-envs>1.")
    p.add_argument("--pretrain-greedy-steps", type=int, default=None, help="Supervised warm-start steps using Greedy teacher (0=off).")
    p.add_argument("--rollout-policy", type=str, default="sample", choices=["sample", "argmax"], help="Action selection during rollout collection.")
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
        cfg = Config(**cfg_dict)

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
    if args.conflict_loss_coef is not None:
        cfg = cfg.__class__(**{**asdict(cfg), "conflict_loss_coef": float(args.conflict_loss_coef)})
    if args.agent_id_tiebreak_eps is not None:
        cfg = cfg.__class__(**{**asdict(cfg), "agent_id_tiebreak_eps": float(args.agent_id_tiebreak_eps)})
    if args.reward_mode is not None:
        cfg = cfg.__class__(**{**asdict(cfg), "reward_mode": str(args.reward_mode)})
    if args.reward_collision is not None:
        cfg = cfg.__class__(**{**asdict(cfg), "reward_collision": float(args.reward_collision)})
    if args.reward_idle_nonempty is not None:
        cfg = cfg.__class__(**{**asdict(cfg), "reward_idle_nonempty": float(args.reward_idle_nonempty)})
    if args.lambda_coop is not None:
        cfg = cfg.__class__(**{**asdict(cfg), "lambda_coop": float(args.lambda_coop)})
    if args.num_envs is not None:
        cfg = cfg.__class__(**{**asdict(cfg), "num_envs": int(args.num_envs)})
    if args.pretrain_greedy_steps is not None:
        cfg = cfg.__class__(**{**asdict(cfg), "pretrain_greedy_steps": int(args.pretrain_greedy_steps)})

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
            lambda_coop=cfg.lambda_coop,
        )

    num_envs = int(max(1, cfg.num_envs))
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

    agent = GACMACAgent(
        input_dim=input_dim,
        hidden_dim=cfg.hidden_dim,
        action_dim=action_dim,
        gat_heads=cfg.gat_heads,
        use_edge_attr=cfg.use_edge_attr,
        use_agent_id_tiebreak=(cfg.obs_version == "v3" and cfg.agent_id_tiebreak_eps > 0.0),
        agent_id_tiebreak_eps=cfg.agent_id_tiebreak_eps,
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
        conflict_loss_coef=cfg.conflict_loss_coef,
        max_grad_norm=cfg.max_grad_norm,
        device=device,
    )

    start_update = 0
    global_step = 0
    rolling = {"reward": [], "sum_rate": [], "collision_rate": [], "avg_delay": []}

    if resume_ckpt is not None:
        agent.load_state_dict(resume_ckpt["agent_state_dict"])
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        # Respect any CLI override to lr after loading optimizer state.
        for pg in optimizer.param_groups:
            pg["lr"] = cfg.lr
        start_update = int(resume_ckpt.get("update", 0)) + 1
        global_step = int(resume_ckpt.get("global_step", 0))
        if isinstance(resume_ckpt.get("rolling"), dict):
            rolling = resume_ckpt["rolling"]

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

    if cfg.pretrain_greedy_steps > 0:
        if vec_env is not None and args.parallel_env:
            raise RuntimeError(
                "--pretrain-greedy-steps is not supported with --parallel-env (teacher needs in-process env state)."
            )

        greedy = GreedyColoringAgent(cfg.num_slots)
        import torch.nn.functional as F

        print(f"[pretrain] greedy_steps={cfg.pretrain_greedy_steps}", flush=True)
        pre_h = agent.initial_hidden(cfg.num_uavs * num_envs, device)
        pre_losses: list[float] = []
        for s in range(int(cfg.pretrain_greedy_steps)):
            if num_envs > 1:
                x, edge_index, edge_attr = _stack_obs(obs_list)
                # Teacher per-env actions (needs env internals, so only SyncVectorEnv supported here).
                act_np = np.stack(
                    [greedy.select_actions(env=e, obs_x=ob.x) for e, ob in zip(vec_env.envs, obs_list)], axis=0  # type: ignore[union-attr]
                )
                teacher_actions = torch.tensor(act_np.reshape(-1), dtype=torch.long, device=device)
            else:
                x = torch.tensor(obs.x, dtype=torch.float32, device=device)
                edge_index = torch.tensor(obs.edge_index, dtype=torch.long, device=device)
                edge_attr = torch.tensor(obs.edge_attr, dtype=torch.float32, device=device)
                act_np = greedy.select_actions(env=env, obs_x=obs.x)
                teacher_actions = torch.tensor(act_np, dtype=torch.long, device=device)

            logits, h_out = agent.forward_logits(x, edge_index, edge_attr, pre_h.detach())
            q_norm = x[:, 3] if x.numel() > 0 and x.size(-1) > 3 else torch.zeros((logits.size(0),), device=device)
            active = q_norm > 0.0
            loss_active = (
                F.cross_entropy(logits[active], teacher_actions[active]) if bool(active.any().item()) else logits.sum() * 0.0
            )
            loss_inactive = (
                F.cross_entropy(logits[~active], teacher_actions[~active])
                if bool((~active).any().item())
                else logits.sum() * 0.0
            )
            loss = loss_active + 0.1 * loss_inactive

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
            optimizer.step()

            pre_losses.append(float(loss.item()))
            pre_h = h_out.detach()

            if num_envs > 1:
                next_obs_list, _rewards_np, dones_env, _infos = vec_env.step(act_np)  # type: ignore[union-attr]
                done_mask_np = np.repeat(dones_env.reshape(-1, 1), cfg.num_uavs, axis=1).reshape(-1).astype(np.float32)
                pre_h = pre_h * (
                    1.0 - torch.tensor(done_mask_np, dtype=torch.float32, device=device).view(-1, 1)
                )
                obs_list = next_obs_list
            else:
                next_obs, _rewards_np, done, _info = env.step(act_np)
                if done:
                    next_obs = env.reset()
                    pre_h = agent.initial_hidden(cfg.num_uavs, device)
                obs = next_obs

            if (s + 1) % 200 == 0:
                print(f"[pretrain] step {s+1:05d} | ce {float(np.mean(pre_losses[-200:])):.4f}", flush=True)

    for update in range(start_update, cfg.total_updates):
        buffer.reset()

        # Collect on-policy rollout
        rollout_reward = []
        rollout_sum_rate = []
        rollout_collision = []
        rollout_delay = []
        rollout_degree = []
        rollout_attempts = []
        rollout_success = []
        rollout_collisions = []
        rollout_no_tx_frac_has_pkt = []

        for _ in range(cfg.steps_per_update):
            if num_envs > 1:
                x, edge_index, edge_attr = _stack_obs(obs_list)
            else:
                x = torch.tensor(obs.x, dtype=torch.float32, device=device)
                edge_index = torch.tensor(obs.edge_index, dtype=torch.long, device=device)
                edge_attr = torch.tensor(obs.edge_attr, dtype=torch.float32, device=device)

            h_in = h.detach()
            with torch.no_grad():
                action, logp, values, _entropy, h_out = agent.act(
                    x, edge_index, edge_attr, h_in, deterministic=(args.rollout_policy == "argmax")
                )

            if num_envs > 1:
                act_np = action.view(num_envs, cfg.num_uavs).cpu().numpy()
                next_obs_list, rewards_np, dones_env, infos = vec_env.step(act_np)  # type: ignore[union-attr]
                rewards = torch.tensor(rewards_np.reshape(-1), dtype=torch.float32, device=device)
                done_mask_np = np.repeat(dones_env.reshape(-1, 1), cfg.num_uavs, axis=1).reshape(-1).astype(
                    np.float32
                )
                done_mask = torch.tensor(done_mask_np, dtype=torch.float32, device=device)
                info_rate = float(np.mean([i.get("sum_rate_mbps", i.get("sum_rate", 0.0)) for i in infos]))
                info_coll = float(np.mean([i.get("collision_rate", 0.0) for i in infos]))
                info_delay = float(np.mean([i.get("avg_delay", 0.0) for i in infos]))
                info_deg = float(np.mean([i.get("avg_degree_in", 0.0) for i in infos]))
                info_attempts = float(np.mean([i.get("tx_attempts", 0.0) for i in infos]))
                info_success = float(np.mean([i.get("tx_success", 0.0) for i in infos]))
                info_collisions = float(np.mean([i.get("tx_collision", 0.0) for i in infos]))
            else:
                next_obs, rewards_np, done, info = env.step(action.cpu().numpy())
                rewards = torch.tensor(rewards_np, dtype=torch.float32, device=device)
                done_mask = torch.full_like(rewards, 1.0 if done else 0.0)
                info_rate = float(info.get("sum_rate_mbps", info.get("sum_rate", 0.0)))
                info_coll = float(info.get("collision_rate", 0.0))
                info_delay = float(info.get("avg_delay", 0.0))
                info_deg = float(info.get("avg_degree_in", 0.0))
                info_attempts = float(info.get("tx_attempts", 0.0))
                info_success = float(info.get("tx_success", 0.0))
                info_collisions = float(info.get("tx_collision", 0.0))

            with torch.no_grad():
                has_pkt = x[:, 3] > 0.0
                denom = float(max(1, int(has_pkt.sum().item())))
                if denom > 0:
                    no_tx = (action == cfg.num_slots) & has_pkt
                    rollout_no_tx_frac_has_pkt.append(float(no_tx.sum().item() / denom))

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
            )

            rollout_reward.append(float(rewards.mean().item()))
            rollout_sum_rate.append(info_rate)
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
        rolling["collision_rate"].append(float(np.mean(rollout_collision)))
        rolling["avg_delay"].append(float(np.mean(rollout_delay)))

        if writer is not None:
            writer.add_scalar("train/reward_mean", rolling["reward"][-1], update + 1)
            writer.add_scalar("train/sum_rate_mean", rolling["sum_rate"][-1], update + 1)
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

            writer.add_scalar("loss/total", stats.loss, update + 1)
            writer.add_scalar("loss/actor", stats.actor_loss, update + 1)
            writer.add_scalar("loss/value", stats.value_loss, update + 1)
            writer.add_scalar("loss/entropy", stats.entropy, update + 1)
            writer.add_scalar("loss/conflict", stats.conflict_loss, update + 1)

        if (update + 1) % cfg.log_interval == 0:
            print(
                f"upd {update+1:04d} | step {global_step:07d} | "
                f"R {rolling['reward'][-1]:+.3f} | "
                f"rate {rolling['sum_rate'][-1]:.2f} | "
                f"coll {rolling['collision_rate'][-1]:.3f} | "
                f"loss {stats.loss:.3f} (pi {stats.actor_loss:.3f}, v {stats.value_loss:.3f}, ent {stats.entropy:.3f}, conf {stats.conflict_loss:.3f})",
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
