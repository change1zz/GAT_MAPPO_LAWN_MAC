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
    p.add_argument("--num-uavs", type=int, default=None, help="Override number of UAVs for training env (density).")
    p.add_argument("--map-size-m", type=float, default=None, help="Override map size in meters for training env (density).")
    p.add_argument("--cs-threshold-dbm", type=float, default=None)
    p.add_argument("--graph-mode", type=str, default=None, choices=["cs", "conflict"])
    p.add_argument("--obs-version", type=str, default=None, choices=["v1", "v2", "v3"])
    p.add_argument("--num-channels", type=int, default=None, help="Orthogonal channels (C). Total actions = C*K + NoTx.")
    p.add_argument("--max-tx-per-frame", type=int, default=None, help="Allow each node to pick up to L resources per frame.")
    p.add_argument("--secondary-lbt", action="store_true", help="If L>1, gate secondary picks via listen-before-talk (reduce collisions).")
    p.add_argument("--primary-lbt", action="store_true", help="Apply listen-before-talk contention resolution for primary pick too.")
    p.add_argument("--no-edge-attr", action="store_true", help="Disable edge_attr usage in GAT (v1-style).")
    p.add_argument("--total-updates", type=int, default=None)
    p.add_argument("--steps-per-update", type=int, default=None)
    p.add_argument("--log-interval", type=int, default=None)
    p.add_argument("--checkpoint-interval", type=int, default=None)
    p.add_argument("--hidden-dim", type=int, default=None)
    p.add_argument("--gat-heads", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--lr-anneal", action="store_true", help="Linearly anneal lr to --min-lr over total_updates.")
    p.add_argument("--min-lr", type=float, default=None, help="Minimum lr when using --lr-anneal.")
    p.add_argument("--entropy-coef", type=float, default=None)
    p.add_argument("--target-kl", type=float, default=None, help="Early-stop PPO epoch when approx_kl exceeds this (0 disables).")
    p.add_argument("--vf-clip-eps", type=float, default=None, help="Value function clipping epsilon (0 disables).")
    p.add_argument("--value-loss-coef", type=float, default=None, help="Coefficient for value loss term.")
    p.add_argument("--ppo-epochs", type=int, default=None, help="Number of PPO epochs per update.")
    p.add_argument("--clip-eps", type=float, default=None, help="PPO clipping epsilon.")
    p.add_argument("--conflict-loss-coef", type=float, default=None)
    p.add_argument("--agent-id-tiebreak-eps", type=float, default=None)
    p.add_argument("--neighbor-last-action-mask", action="store_true", help="Mask slots used by in-neighbors in last frame (local, uses GNN neighbor features).")
    p.add_argument("--neighbor-last-action-penalty", type=float, default=None, help="Logit penalty for slots used by in-neighbors in last frame (0=off).")
    p.add_argument("--critic-detach-encoder", action="store_true", help="Stop critic gradients from updating encoder/GRU (stability).")
    p.add_argument("--reward-mode", type=str, default=None, choices=["binary", "rate", "mbps"])
    p.add_argument("--reward-success", type=float, default=None)
    p.add_argument("--reward-collision", type=float, default=None)
    p.add_argument("--reward-idle-nonempty", type=float, default=None)
    p.add_argument("--lambda-coop", type=float, default=None)
    p.add_argument("--num-envs", type=int, default=None, help="Vectorized rollout envs (>=1).")
    p.add_argument("--parallel-env", action="store_true", help="Use subprocess envs when --num-envs>1.")
    p.add_argument("--pretrain-greedy-steps", type=int, default=None, help="Supervised warm-start steps using Greedy teacher (0=off).")
    p.add_argument("--pretrain-only", action="store_true", help="Run greedy warm-start then save checkpoint and exit.")
    p.add_argument("--pretrain-no-tx-weight", type=float, default=0.2, help="Down-weight NoTx class in greedy pretrain CE (smaller => less collapse).")
    p.add_argument("--rollout-policy", type=str, default="sample", choices=["sample", "argmax"], help="Action selection during rollout collection.")
    p.add_argument("--rollout-temperature", type=float, default=1.0, help="Sampling temperature for rollout-policy=sample (smaller => more deterministic).")
    p.add_argument("--eval-every", type=int, default=0, help="If >0, run periodic argmax evaluation every N updates.")
    p.add_argument("--eval-episodes", type=int, default=5, help="Episodes per periodic argmax evaluation.")
    p.add_argument("--eval-seed", type=int, default=123, help="Base seed for periodic evaluation.")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pth")
    p.add_argument("--init-from", type=str, default=None, help="Initialize from checkpoint weights but write to a new run dir.")
    p.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard logging.")
    p.add_argument("--target-coll", type=float, default=None, help="Target collision rate (collisions/attempts) for constrained training.")
    p.add_argument("--coll-lagrange-lr", type=float, default=0.05, help="Lagrange multiplier step size for collision constraint.")
    p.add_argument("--coll-lagrange-init", type=float, default=0.0, help="Initial collision Lagrange multiplier (lambda).")
    p.add_argument("--coll-lagrange-max", type=float, default=10.0, help="Max collision Lagrange multiplier (lambda cap).")
    p.add_argument("--coll-warmup-updates", type=int, default=50, help="Do not update lambda during initial warmup updates.")
    p.add_argument("--attempt-penalty", type=float, default=0.0, help="Extra penalty per attempted resource (discourage spamming).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    base_cfg = Config()
    cfg = base_cfg

    if args.resume and args.init_from:
        raise SystemExit("Use only one of --resume or --init-from.")

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

    init_from_path = None
    init_from_ckpt = None
    if args.init_from:
        init_from_path = os.path.abspath(args.init_from)
        init_from_ckpt = load_checkpoint(init_from_path)
        cfg_dict = init_from_ckpt.get("config", asdict(base_cfg))
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
    if args.num_uavs is not None:
        cfg = cfg.__class__(**{**asdict(cfg), "num_uavs": int(args.num_uavs)})
    if args.map_size_m is not None:
        cfg = cfg.__class__(**{**asdict(cfg), "map_size_m": float(args.map_size_m)})
    if args.cs_threshold_dbm is not None:
        cfg = cfg.__class__(**{**asdict(cfg), "cs_threshold_dbm": float(args.cs_threshold_dbm)})
    if args.graph_mode is not None:
        cfg = cfg.__class__(**{**asdict(cfg), "graph_mode": str(args.graph_mode)})
    if args.obs_version is not None:
        cfg = cfg.__class__(**{**asdict(cfg), "obs_version": str(args.obs_version)})
    if args.num_channels is not None:
        cfg = cfg.__class__(**{**asdict(cfg), "num_channels": int(args.num_channels)})
    if args.max_tx_per_frame is not None:
        cfg = cfg.__class__(**{**asdict(cfg), "max_tx_per_frame": int(args.max_tx_per_frame)})
    if args.secondary_lbt:
        cfg = cfg.__class__(**{**asdict(cfg), "secondary_lbt": True})
    if args.primary_lbt:
        cfg = cfg.__class__(**{**asdict(cfg), "primary_lbt": True})
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
    if args.lr_anneal:
        cfg = cfg.__class__(**{**asdict(cfg), "lr_anneal": True})
    if args.min_lr is not None:
        cfg = cfg.__class__(**{**asdict(cfg), "min_lr": float(args.min_lr)})
    if args.entropy_coef is not None:
        cfg = cfg.__class__(**{**asdict(cfg), "entropy_coef": float(args.entropy_coef)})
    if args.target_kl is not None:
        cfg = cfg.__class__(**{**asdict(cfg), "target_kl": float(args.target_kl)})
    if args.vf_clip_eps is not None:
        cfg = cfg.__class__(**{**asdict(cfg), "vf_clip_eps": float(args.vf_clip_eps)})
    if args.value_loss_coef is not None:
        cfg = cfg.__class__(**{**asdict(cfg), "value_loss_coef": float(args.value_loss_coef)})
    if args.ppo_epochs is not None:
        cfg = cfg.__class__(**{**asdict(cfg), "ppo_epochs": int(args.ppo_epochs)})
    if args.clip_eps is not None:
        cfg = cfg.__class__(**{**asdict(cfg), "clip_eps": float(args.clip_eps)})
    if args.conflict_loss_coef is not None:
        cfg = cfg.__class__(**{**asdict(cfg), "conflict_loss_coef": float(args.conflict_loss_coef)})
    if args.agent_id_tiebreak_eps is not None:
        cfg = cfg.__class__(**{**asdict(cfg), "agent_id_tiebreak_eps": float(args.agent_id_tiebreak_eps)})
    if args.neighbor_last_action_mask:
        cfg = cfg.__class__(**{**asdict(cfg), "neighbor_last_action_mask": True})
    if args.neighbor_last_action_penalty is not None:
        cfg = cfg.__class__(**{**asdict(cfg), "neighbor_last_action_penalty": float(args.neighbor_last_action_penalty)})
    if args.critic_detach_encoder:
        cfg = cfg.__class__(**{**asdict(cfg), "critic_detach_encoder": True})
    if args.reward_mode is not None:
        cfg = cfg.__class__(**{**asdict(cfg), "reward_mode": str(args.reward_mode)})
    if args.reward_success is not None:
        cfg = cfg.__class__(**{**asdict(cfg), "reward_success": float(args.reward_success)})
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

    target_coll = None if args.target_coll is None else float(args.target_coll)
    coll_lagrange_lr = float(args.coll_lagrange_lr)
    coll_lambda = float(max(0.0, args.coll_lagrange_init))
    coll_lambda_max = float(max(0.0, args.coll_lagrange_max))
    coll_warmup_updates = int(max(0, args.coll_warmup_updates))
    attempt_penalty = float(max(0.0, args.attempt_penalty))

    def _make_single_env() -> ToyLawnEnv | LAWNEnv:
        if args.env == "toy":
            return ToyLawnEnv(
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

    def _eval_argmax(agent: GACMACAgent, *, episodes: int, base_seed: int) -> dict[str, float]:
        import torch

        env_eval = _make_single_env()
        ep_thr = []
        ep_coll = []
        ep_jain = []
        for ep in range(int(episodes)):
            obs_eval = env_eval.reset(seed=int(base_seed + ep))
            h_eval = agent.initial_hidden(cfg.num_uavs, device)
            thr_sum = 0.0
            coll_sum = 0.0
            jain_sum = 0.0
            for _t in range(cfg.episode_len):
                x = torch.tensor(obs_eval.x, dtype=torch.float32, device=device)
                ei = torch.tensor(obs_eval.edge_index, dtype=torch.long, device=device)
                ea = torch.tensor(obs_eval.edge_attr, dtype=torch.float32, device=device)
                with torch.no_grad():
                    act, _logp, _v, _ent, h_out = agent.act(
                        x,
                        ei,
                        ea,
                        h_eval,
                        deterministic=True,
                        temperature=1.0,
                        max_tx=int(getattr(cfg, "max_tx_per_frame", 1)),
                    )
                obs_eval, _rew, done, info = env_eval.step(act.cpu().numpy())
                thr_sum += float(info.get("sum_rate_mbps", info.get("sum_rate", 0.0)))
                coll_sum += float(info.get("collision_rate", 0.0))
                jain_sum += float(info.get("jain", 0.0))
                h_eval = h_out.detach()
                if done:
                    break
            denom = float(cfg.episode_len)
            ep_thr.append(thr_sum / denom)
            ep_coll.append(coll_sum / denom)
            ep_jain.append(jain_sum / denom)
        return {
            "thr_mbps": float(np.mean(ep_thr)) if ep_thr else 0.0,
            "coll": float(np.mean(ep_coll)) if ep_coll else 0.0,
            "jain": float(np.mean(ep_jain)) if ep_jain else 0.0,
        }

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

    log_path = os.path.join(run_dir, "train_log.jsonl")
    log_f = open(log_path, "a", encoding="utf-8")
    if log_f.tell() == 0:
        log_f.write(json.dumps({"event": "config", "config": asdict(cfg)}, ensure_ascii=False) + "\n")
    if resume_path:
        log_f.write(json.dumps({"event": "resume", "path": resume_path}, ensure_ascii=False) + "\n")
    log_f.flush()

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
    action_dim = int(cfg.num_slots) * int(getattr(cfg, "num_channels", 1)) + 1

    import torch

    agent = GACMACAgent(
        input_dim=input_dim,
        hidden_dim=cfg.hidden_dim,
        action_dim=action_dim,
        gat_heads=cfg.gat_heads,
        use_edge_attr=cfg.use_edge_attr,
        use_agent_id_tiebreak=(cfg.obs_version == "v3" and cfg.agent_id_tiebreak_eps > 0.0),
        agent_id_tiebreak_eps=cfg.agent_id_tiebreak_eps,
        neighbor_last_action_mask=cfg.neighbor_last_action_mask,
        neighbor_last_action_penalty=cfg.neighbor_last_action_penalty,
        critic_detach_encoder=cfg.critic_detach_encoder,
    ).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.lr)

    trainer = MAPPOTrainer(
        agent=agent,
        optimizer=optimizer,
        clip_eps=cfg.clip_eps,
        target_kl=cfg.target_kl,
        ppo_epochs=cfg.ppo_epochs,
        bptt_len=cfg.bptt_len,
        value_loss_coef=cfg.value_loss_coef,
        vf_clip_eps=cfg.vf_clip_eps,
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
    elif init_from_ckpt is not None:
        agent.load_state_dict(init_from_ckpt["agent_state_dict"])

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

        greedy = GreedyColoringAgent(cfg.num_slots, num_channels=int(getattr(cfg, "num_channels", 1)))
        import torch.nn.functional as F

        print(f"[pretrain] greedy_steps={cfg.pretrain_greedy_steps}", flush=True)
        pre_h = agent.initial_hidden(cfg.num_uavs * num_envs, device)
        pre_losses: list[float] = []
        for s in range(int(cfg.pretrain_greedy_steps)):
            max_tx = int(getattr(cfg, "max_tx_per_frame", 1))
            if num_envs > 1:
                x, edge_index, edge_attr = _stack_obs(obs_list)
                # Teacher per-env actions (needs env internals, so only SyncVectorEnv supported here).
                act_np = np.stack(
                    [
                        greedy.select_actions(env=e, obs_x=ob.x, max_tx=max_tx)
                        for e, ob in zip(vec_env.envs, obs_list)  # type: ignore[union-attr]
                    ],
                    axis=0,
                )
                teacher_actions = torch.tensor(act_np.reshape(-1, max_tx), dtype=torch.long, device=device)
            else:
                x = torch.tensor(obs.x, dtype=torch.float32, device=device)
                edge_index = torch.tensor(obs.edge_index, dtype=torch.long, device=device)
                edge_attr = torch.tensor(obs.edge_attr, dtype=torch.float32, device=device)
                act_np = greedy.select_actions(env=env, obs_x=obs.x, max_tx=max_tx)
                teacher_actions = torch.tensor(np.asarray(act_np).reshape(-1, max_tx), dtype=torch.long, device=device)

            logits, h_out = agent.forward_logits(x, edge_index, edge_attr, pre_h.detach())
            q_norm = x[:, 3] if x.numel() > 0 and x.size(-1) > 3 else torch.zeros((logits.size(0),), device=device)
            active = q_norm > 0.0
            w_no = float(max(0.0, args.pretrain_no_tx_weight))
            no_tx_idx = int(cfg.num_slots) * int(getattr(cfg, "num_channels", 1))

            def _weighted_ce(logits_sub: torch.Tensor, labels_sub: torch.Tensor) -> torch.Tensor:
                if logits_sub.numel() == 0:
                    return logits.sum() * 0.0
                per = F.cross_entropy(logits_sub, labels_sub, reduction="none")  # (M,)
                is_no = labels_sub == no_tx_idx  # NoTx index
                weights = torch.where(is_no, torch.full_like(per, w_no), torch.ones_like(per))
                return (per * weights).mean()

            # Sequential multi-resource imitation (L>1): NoTx acts as early-stop token.
            max_tx = int(getattr(cfg, "max_tx_per_frame", 1))
            ended = torch.zeros((logits.size(0),), dtype=torch.bool, device=device)
            masked = logits
            loss_active_steps: list[torch.Tensor] = []
            for j in range(max_tx):
                labels = teacher_actions[:, j]
                mask = active & (~ended)
                if bool(mask.any().item()):
                    loss_active_steps.append(_weighted_ce(masked[mask], labels[mask]))
                ended = ended | (labels == no_tx_idx)
                # Avoid duplicating the same resource within a frame.
                not_stop = labels != no_tx_idx
                if bool(not_stop.any().item()):
                    idx = torch.arange(logits.size(0), device=device)[not_stop]
                    sel = labels[not_stop]
                    masked = masked.clone()
                    masked[idx, sel] = -1e9

            loss_active = torch.stack(loss_active_steps).mean() if loss_active_steps else logits.sum() * 0.0
            # Inactive nodes don't matter at inference (env masks), but keep a tiny signal to avoid garbage logits.
            if bool((~active).any().item()):
                loss_inactive = _weighted_ce(logits[~active], teacher_actions[:, 0][~active])
            else:
                loss_inactive = logits.sum() * 0.0
            loss = loss_active + 0.05 * loss_inactive

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

        pre_ckpt_path = os.path.join(ckpt_dir, "checkpoint_pretrain.pth")
        save_checkpoint(
            pre_ckpt_path,
            {
                "config": asdict(cfg),
                "update": -1,
                "global_step": global_step,
                "agent_state_dict": agent.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "rolling": rolling,
                "pretrain": {"steps": int(cfg.pretrain_greedy_steps), "ce_last200": float(np.mean(pre_losses[-200:])) if pre_losses else None},
            },
        )
        print(f"[pretrain] saved: {pre_ckpt_path}", flush=True)
        if args.pretrain_only:
            if vec_env is not None:
                vec_env.close()
            if writer is not None:
                writer.flush()
                writer.close()
            log_f.close()
            print(f"Done (pretrain-only). Outputs in: {run_dir}", flush=True)
            return

    for update in range(start_update, cfg.total_updates):
        buffer.reset()

        if cfg.lr_anneal and cfg.total_updates > 1:
            frac = 1.0 - float(update) / float(cfg.total_updates - 1)
            lr_now = max(float(cfg.min_lr), float(cfg.lr) * frac)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

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
        rollout_total_attempts = 0.0
        rollout_total_collisions = 0.0

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
                    x,
                    edge_index,
                    edge_attr,
                    h_in,
                    deterministic=(args.rollout_policy == "argmax"),
                    temperature=float(args.rollout_temperature),
                    max_tx=int(getattr(cfg, "max_tx_per_frame", 1)),
                )

            if num_envs > 1:
                max_tx = int(action.size(1)) if action.dim() == 2 else 1
                act_np = action.view(num_envs, cfg.num_uavs, max_tx).cpu().numpy()
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

                if target_coll is not None or attempt_penalty > 0.0:
                    a_np = np.stack([i.get("per_node_tx_attempts") for i in infos], axis=0)  # (E,N)
                    c_np = np.stack([i.get("per_node_tx_collisions") for i in infos], axis=0)  # (E,N)
                    a_flat = torch.tensor(a_np.reshape(-1), dtype=torch.float32, device=device)
                    c_flat = torch.tensor(c_np.reshape(-1), dtype=torch.float32, device=device)
                    if attempt_penalty > 0.0:
                        rewards = rewards - attempt_penalty * a_flat
                    if target_coll is not None and coll_lambda > 0.0:
                        g = c_flat - float(target_coll) * a_flat
                        rewards = rewards - float(coll_lambda) * torch.clamp(g, min=0.0)
                    rollout_total_attempts += float(a_flat.sum().item())
                    rollout_total_collisions += float(c_flat.sum().item())
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

                if target_coll is not None or attempt_penalty > 0.0:
                    a_np = np.asarray(info.get("per_node_tx_attempts"), dtype=np.float32).reshape(-1)
                    c_np = np.asarray(info.get("per_node_tx_collisions"), dtype=np.float32).reshape(-1)
                    a_t = torch.tensor(a_np, dtype=torch.float32, device=device)
                    c_t = torch.tensor(c_np, dtype=torch.float32, device=device)
                    if attempt_penalty > 0.0:
                        rewards = rewards - attempt_penalty * a_t
                    if target_coll is not None and coll_lambda > 0.0:
                        g = c_t - float(target_coll) * a_t
                        rewards = rewards - float(coll_lambda) * torch.clamp(g, min=0.0)
                    rollout_total_attempts += float(a_t.sum().item())
                    rollout_total_collisions += float(c_t.sum().item())

            with torch.no_grad():
                has_pkt = x[:, 3] > 0.0
                denom = float(max(1, int(has_pkt.sum().item())))
                if denom > 0:
                    no_tx_idx = int(cfg.num_slots) * int(getattr(cfg, "num_channels", 1))
                    if action.dim() == 1:
                        no_tx = (action == no_tx_idx) & has_pkt
                    else:
                        no_tx = (action == no_tx_idx).all(dim=1) & has_pkt
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

        lr_report = float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else float(cfg.lr)
        log_f.write(
            json.dumps(
                {
                    "event": "update",
                    "update": int(update + 1),
                    "global_step": int(global_step),
                    "lr": lr_report,
                    "reward_mean": rolling["reward"][-1],
                    "sum_rate_mbps_mean": rolling["sum_rate"][-1],
                    "collision_rate_mean": rolling["collision_rate"][-1],
                    "avg_delay_mean": rolling["avg_delay"][-1],
                    "tx_attempts": float(np.mean(rollout_attempts)) if rollout_attempts else None,
                    "tx_success": float(np.mean(rollout_success)) if rollout_success else None,
                    "tx_collision": float(np.mean(rollout_collisions)) if rollout_collisions else None,
                    "no_tx_frac_given_queue": float(np.mean(rollout_no_tx_frac_has_pkt)) if rollout_no_tx_frac_has_pkt else None,
                    "loss_total": float(stats.loss),
                    "loss_actor": float(stats.actor_loss),
                    "loss_value": float(stats.value_loss),
                    "entropy": float(stats.entropy),
                    "loss_conflict": float(stats.conflict_loss),
                    "approx_kl": float(stats.approx_kl),
                    "clip_frac": float(stats.clip_frac),
                    "grad_norm": float(stats.grad_norm),
                    "ppo_epochs_run": int(stats.epochs_run),
                    "coll_lambda": float(coll_lambda) if target_coll is not None else None,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        log_f.flush()

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
            writer.add_scalar("diag/approx_kl", stats.approx_kl, update + 1)
            writer.add_scalar("diag/clip_frac", stats.clip_frac, update + 1)
            writer.add_scalar("diag/grad_norm", stats.grad_norm, update + 1)
            writer.add_scalar("diag/ppo_epochs_run", stats.epochs_run, update + 1)
            writer.add_scalar("train/lr", lr_report, update + 1)

        if (update + 1) % cfg.log_interval == 0:
            lambda_str = f"lambda {coll_lambda:.3f} | " if target_coll is not None else ""
            print(
                f"upd {update+1:04d} | step {global_step:07d} | "
                f"R {rolling['reward'][-1]:+.3f} | "
                f"rate {rolling['sum_rate'][-1]:.2f} | "
                f"coll {rolling['collision_rate'][-1]:.3f} | "
                + lambda_str
                + f"loss {stats.loss:.3f} (pi {stats.actor_loss:.3f}, v {stats.value_loss:.3f}, ent {stats.entropy:.3f}, conf {stats.conflict_loss:.3f}) | "
                + f"kl {stats.approx_kl:.4f} clip {stats.clip_frac:.3f} gnorm {stats.grad_norm:.3f} ep {stats.epochs_run} lr {lr_report:.2e}",
                flush=True,
            )

        if not np.isfinite([stats.loss, stats.actor_loss, stats.value_loss, stats.entropy, stats.approx_kl, stats.grad_norm]).all():
            bad_path = os.path.join(ckpt_dir, "checkpoint_bad_numeric.pth")
            save_checkpoint(
                bad_path,
                {
                    "config": asdict(cfg),
                    "update": update,
                    "global_step": global_step,
                    "agent_state_dict": agent.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "rolling": rolling,
                    "bad_numeric": True,
                    "stats": asdict(stats),
                },
            )
            print(f"[fatal] non-finite stats detected; saved: {bad_path}", flush=True)
            break

        # Update collision constraint multiplier once per update using rollout statistics.
        if (
            target_coll is not None
            and (update + 1) > coll_warmup_updates
            and rollout_total_attempts > 0.0
        ):
            coll_rate_rollout = float(rollout_total_collisions / max(1e-9, rollout_total_attempts))
            coll_lambda = float(
                np.clip(
                    coll_lambda + coll_lagrange_lr * (coll_rate_rollout - float(target_coll)),
                    0.0,
                    coll_lambda_max,
                )
            )
            if writer is not None:
                writer.add_scalar("constraint/coll_lambda", coll_lambda, update + 1)
                writer.add_scalar("constraint/coll_rollout", coll_rate_rollout, update + 1)

        if args.eval_every and (update + 1) % int(args.eval_every) == 0:
            agent.eval()
            metrics = _eval_argmax(agent, episodes=int(args.eval_episodes), base_seed=int(args.eval_seed))
            if writer is not None:
                writer.add_scalar("eval_argmax/thr_mbps", metrics["thr_mbps"], update + 1)
                writer.add_scalar("eval_argmax/coll", metrics["coll"], update + 1)
                writer.add_scalar("eval_argmax/jain", metrics["jain"], update + 1)
            print(
                f"[eval_argmax] upd {update+1:04d} | thr {metrics['thr_mbps']:.3f} Mbps | "
                f"jain {metrics['jain']:.3f} | coll {metrics['coll']:.3f}",
                flush=True,
            )
            log_f.write(
                json.dumps(
                    {
                        "event": "eval_argmax",
                        "update": int(update + 1),
                        "global_step": int(global_step),
                        "thr_mbps": float(metrics["thr_mbps"]),
                        "jain": float(metrics["jain"]),
                        "coll": float(metrics["coll"]),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            log_f.flush()

            best_path = os.path.join(ckpt_dir, "checkpoint_best_argmax.pth")
            best_thr = float("-inf")
            if os.path.exists(best_path):
                try:
                    best_ckpt = load_checkpoint(best_path)
                    best_thr = float(best_ckpt.get("best_argmax_thr_mbps", float("-inf")))
                except Exception:
                    best_thr = float("-inf")
            if metrics["thr_mbps"] > best_thr:
                save_checkpoint(
                    best_path,
                    {
                        "config": asdict(cfg),
                        "update": update,
                        "global_step": global_step,
                        "best_argmax_thr_mbps": metrics["thr_mbps"],
                        "best_argmax_jain": metrics["jain"],
                        "best_argmax_coll": metrics["coll"],
                        "agent_state_dict": agent.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "rolling": rolling,
                    },
                )
                print(f"[eval_argmax] new best saved: {best_path}", flush=True)
            agent.train()

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
    log_f.close()

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
