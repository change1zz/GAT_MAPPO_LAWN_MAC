from __future__ import annotations

import multiprocessing as mp
from dataclasses import asdict
from typing import Any, Literal

import numpy as np

from gac_mac.config import Config
from gac_mac.env.channel import ChannelModel
from gac_mac.env.lawn_env import LAWNEnv
from gac_mac.env.mobility import GaussMarkovMobility
from gac_mac.env.toy_env import ToyLawnEnv


def make_env_from_config(cfg: Config, *, env: Literal["toy", "lawn"]) -> Any:
    if env == "toy":
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
            reward_tx_attempt=getattr(cfg, "reward_tx_attempt", 0.0),
            reward_repeat_success=getattr(cfg, "reward_repeat_success", 0.0),
            reward_repeat_collision=getattr(cfg, "reward_repeat_collision", 0.0),
            reward_neighbor_slot_conflict=getattr(cfg, "reward_neighbor_slot_conflict", 0.0),
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
    e = LAWNEnv(
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
        reward_repeat_success=getattr(cfg, "reward_repeat_success", 0.0),
        reward_repeat_collision=getattr(cfg, "reward_repeat_collision", 0.0),
        reward_neighbor_slot_conflict=getattr(cfg, "reward_neighbor_slot_conflict", 0.0),
        lambda_coop=cfg.lambda_coop,
    )
    if getattr(cfg, "distill_coef", 0.0) > 0.0:
        e.enable_greedy_teacher = True
    return e


class SyncVectorEnv:
    """Synchronous vectorized wrapper (single process, multiple env instances)."""

    def __init__(self, cfg: Config, *, env: Literal["toy", "lawn"], num_envs: int) -> None:
        self.cfg = cfg
        self.env_name = env
        self.num_envs = int(num_envs)
        self.envs = [make_env_from_config(cfg, env=env) for _ in range(self.num_envs)]

    def reset(self, *, seed: int | None = None) -> list[Any]:
        obs = []
        for i, e in enumerate(self.envs):
            s = None if seed is None else int(seed + 10_000 * i)
            obs.append(e.reset(seed=s))
        return obs

    def step(self, actions: np.ndarray) -> tuple[list[Any], np.ndarray, np.ndarray, list[dict[str, Any]]]:
        actions = np.asarray(actions)
        if actions.shape[0] != self.num_envs:
            raise ValueError(f"actions must have shape (num_envs, N), got {actions.shape}")

        next_obs = []
        rewards = []
        dones = np.zeros((self.num_envs,), dtype=np.bool_)
        infos: list[dict[str, Any]] = []

        for i, e in enumerate(self.envs):
            ob, rew, done, info = e.step(actions[i])
            if done:
                ob = e.reset()
            next_obs.append(ob)
            rewards.append(rew)
            dones[i] = bool(done)
            infos.append(info)

        return next_obs, np.asarray(rewards), dones, infos

    def close(self) -> None:
        return


def _worker(remote, cfg_dict: dict[str, Any], env_name: str) -> None:
    cfg = Config(**cfg_dict)
    env = make_env_from_config(cfg, env=env_name)  # type: ignore[arg-type]
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                seed = data.get("seed", None) if isinstance(data, dict) else None
                obs = env.reset(seed=seed)
                remote.send(obs)
            elif cmd == "step":
                actions = data
                obs, rew, done, info = env.step(actions)
                if done:
                    obs = env.reset()
                remote.send((obs, rew, bool(done), info))
            elif cmd == "close":
                remote.close()
                break
            else:
                raise RuntimeError(f"Unknown cmd: {cmd}")
    except KeyboardInterrupt:
        pass


class SubprocVectorEnv:
    """Multiprocess vector env (Windows-friendly spawn)."""

    def __init__(self, cfg: Config, *, env: Literal["toy", "lawn"], num_envs: int) -> None:
        self.cfg = cfg
        self.env_name = env
        self.num_envs = int(num_envs)
        ctx = mp.get_context("spawn")
        self.remotes, self.work_remotes = zip(*[ctx.Pipe(duplex=True) for _ in range(self.num_envs)])
        self.ps = []
        cfg_dict = asdict(cfg)
        for wr in self.work_remotes:
            p = ctx.Process(target=_worker, args=(wr, cfg_dict, env))
            p.daemon = True
            p.start()
            self.ps.append(p)
        for wr in self.work_remotes:
            wr.close()

    def reset(self, *, seed: int | None = None) -> list[Any]:
        for i, r in enumerate(self.remotes):
            s = None if seed is None else int(seed + 10_000 * i)
            r.send(("reset", {"seed": s}))
        return [r.recv() for r in self.remotes]

    def step(self, actions: np.ndarray) -> tuple[list[Any], np.ndarray, np.ndarray, list[dict[str, Any]]]:
        actions = np.asarray(actions)
        if actions.shape[0] != self.num_envs:
            raise ValueError(f"actions must have shape (num_envs, N), got {actions.shape}")
        for r, a in zip(self.remotes, actions):
            r.send(("step", a))
        results = [r.recv() for r in self.remotes]

        next_obs = []
        rewards = []
        dones = np.zeros((self.num_envs,), dtype=np.bool_)
        infos: list[dict[str, Any]] = []
        for i, (ob, rew, done, info) in enumerate(results):
            next_obs.append(ob)
            rewards.append(rew)
            dones[i] = bool(done)
            infos.append(info)

        return next_obs, np.asarray(rewards), dones, infos

    def close(self) -> None:
        for r in self.remotes:
            try:
                r.send(("close", None))
            except Exception:
                pass
        for p in self.ps:
            p.join(timeout=1.0)
