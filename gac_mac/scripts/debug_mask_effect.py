import argparse
from dataclasses import asdict

import numpy as np

from gac_mac.config import Config
from gac_mac.scripts.evaluate import make_env
from gac_mac.models.agent import GACMACAgent
from gac_mac.utils.checkpoint import load_checkpoint
from gac_mac.utils.seed import seed_everything


def _resolve_device(device: str) -> "object":
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--num-channels", type=int, default=None)
    p.add_argument("--max-tx-per-frame", type=int, default=None)
    args = p.parse_args()

    seed_everything(args.seed)
    ckpt = load_checkpoint(args.checkpoint)
    cfg_dict = ckpt.get("config", asdict(Config()))
    if "obs_version" not in cfg_dict:
        cfg_dict = {**cfg_dict, "obs_version": "v1", "use_edge_attr": False}
    cfg = Config(**cfg_dict)
    if args.num_channels is not None:
        cfg = cfg.__class__(**{**asdict(cfg), "num_channels": int(args.num_channels)})
    if args.max_tx_per_frame is not None:
        cfg = cfg.__class__(**{**asdict(cfg), "max_tx_per_frame": int(args.max_tx_per_frame)})

    device = _resolve_device(args.device)
    env = make_env(cfg)
    obs = env.reset(seed=args.seed)
    input_dim = int(obs.x.shape[1])
    action_dim = int(cfg.num_slots) * int(getattr(cfg, "num_channels", 1)) + 1

    import torch

    agent_plain = GACMACAgent(
        input_dim=input_dim,
        hidden_dim=cfg.hidden_dim,
        action_dim=action_dim,
        gat_heads=cfg.gat_heads,
        use_edge_attr=cfg.use_edge_attr,
        use_agent_id_tiebreak=(cfg.obs_version == "v3" and getattr(cfg, "agent_id_tiebreak_eps", 0.0) > 0.0),
        agent_id_tiebreak_eps=getattr(cfg, "agent_id_tiebreak_eps", 0.0),
        neighbor_last_action_mask=False,
        neighbor_last_action_penalty=0.0,
    ).to(device)
    agent_plain.load_state_dict(ckpt["agent_state_dict"])
    agent_plain.eval()

    agent_mask = GACMACAgent(
        input_dim=input_dim,
        hidden_dim=cfg.hidden_dim,
        action_dim=action_dim,
        gat_heads=cfg.gat_heads,
        use_edge_attr=cfg.use_edge_attr,
        use_agent_id_tiebreak=(cfg.obs_version == "v3" and getattr(cfg, "agent_id_tiebreak_eps", 0.0) > 0.0),
        agent_id_tiebreak_eps=getattr(cfg, "agent_id_tiebreak_eps", 0.0),
        neighbor_last_action_mask=True,
        neighbor_last_action_penalty=0.0,
    ).to(device)
    agent_mask.load_state_dict(ckpt["agent_state_dict"])
    agent_mask.eval()

    h_plain = agent_plain.initial_hidden(cfg.num_uavs, device)
    h_mask = agent_mask.initial_hidden(cfg.num_uavs, device)

    for t in range(int(args.steps)):
        x = torch.tensor(obs.x, dtype=torch.float32, device=device)
        ei = torch.tensor(obs.edge_index, dtype=torch.long, device=device)
        ea = torch.tensor(obs.edge_attr, dtype=torch.float32, device=device)
        with torch.no_grad():
            a_plain, _lp, _v, _e, h_plain = agent_plain.act(
                x, ei, ea, h_plain, deterministic=True, temperature=1.0, max_tx=int(getattr(cfg, "max_tx_per_frame", 1))
            )
            a_mask, _lp, _v, _e, h_mask = agent_mask.act(
                x, ei, ea, h_mask, deterministic=True, temperature=1.0, max_tx=int(getattr(cfg, "max_tx_per_frame", 1))
            )
        ap = a_plain.cpu().numpy()
        am = a_mask.cpu().numpy()
        diff = float(np.mean(ap != am))
        print(f"t={t:02d} diff_frac={diff:.3f} ap0={ap[0].tolist()} am0={am[0].tolist()}")

        # advance env with plain actions to update last_action features deterministically
        obs, _r, done, info = env.step(ap)
        if done:
            break

    env.close() if hasattr(env, "close") else None


if __name__ == "__main__":
    main()

