from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Small hyperparameter sweep targeting argmax throughput.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--episodes", type=int, default=100, help="Episodes for final evaluation per run.")
    p.add_argument("--out", type=str, default="results/sweep_argmax.json")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _find_latest_run_dir(results_dir: Path, run_name: str) -> Path:
    cands = [p for p in results_dir.glob(f"{run_name}-*") if p.is_dir()]
    if not cands:
        raise FileNotFoundError(f"no run dirs found for run_name={run_name} in {results_dir}")
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def _load_best_ckpt_metrics(ckpt_path: Path) -> dict[str, float]:
    import torch

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    return {
        "best_argmax_thr_mbps": float(ckpt.get("best_argmax_thr_mbps", float("-inf"))),
        "best_argmax_coll": float(ckpt.get("best_argmax_coll", 0.0)),
        "best_argmax_jain": float(ckpt.get("best_argmax_jain", 0.0)),
    }


_EVAL_RE = re.compile(
    r"^GAC-MAC\s+\|\s+thr\s+(?P<thr>[0-9.]+)\s+Mbps\s+\|\s+jain\s+(?P<jain>[0-9.]+)\s+\|\s+coll\s+(?P<coll>[0-9.]+)\s+\|"
)


def _evaluate(checkpoint: Path, *, device: str, episodes: int) -> dict[str, float]:
    cmd = [
        sys.executable,
        "-m",
        "gac_mac.scripts.evaluate",
        "--checkpoint",
        str(checkpoint),
        "--episodes",
        str(int(episodes)),
        "--policy",
        "argmax",
        "--device",
        device,
    ]
    out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    thr = jain = coll = None
    for line in out.splitlines():
        m = _EVAL_RE.match(line.strip())
        if m:
            thr = float(m.group("thr"))
            jain = float(m.group("jain"))
            coll = float(m.group("coll"))
            break
    if thr is None:
        raise RuntimeError(f"failed to parse evaluate output:\n{out}")
    return {"thr_mbps": thr, "jain": float(jain), "coll": float(coll)}


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    results_dir = repo_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Focus sweep around the current best family:
    # conflict graph + v3 + (hd=128, heads=4) + lr=1e-4 + tie-break + idle penalty.
    runs = [
        {
            "name": "sweepA_idle070_tie010_lr1e4",
            "train": [
                "--env",
                "lawn",
                "--device",
                args.device,
                "--reward-mode",
                "rate",
                "--graph-mode",
                "conflict",
                "--obs-version",
                "v3",
                "--reward-idle-nonempty",
                "-0.7",
                "--agent-id-tiebreak-eps",
                "0.1",
                "--hidden-dim",
                "128",
                "--gat-heads",
                "4",
                "--lr",
                "1e-4",
                "--num-envs",
                "8",
                "--steps-per-update",
                "64",
                "--total-updates",
                "160",
                "--checkpoint-interval",
                "40",
                "--log-interval",
                "20",
                "--eval-every",
                "20",
                "--eval-episodes",
                "20",
                "--eval-seed",
                "123",
                "--no-tensorboard",
            ],
        },
        {
            "name": "sweepB_idle075_tie010_lr1e4",
            "train": [
                "--env",
                "lawn",
                "--device",
                args.device,
                "--reward-mode",
                "rate",
                "--graph-mode",
                "conflict",
                "--obs-version",
                "v3",
                "--reward-idle-nonempty",
                "-0.75",
                "--agent-id-tiebreak-eps",
                "0.1",
                "--hidden-dim",
                "128",
                "--gat-heads",
                "4",
                "--lr",
                "1e-4",
                "--num-envs",
                "8",
                "--steps-per-update",
                "64",
                "--total-updates",
                "160",
                "--checkpoint-interval",
                "40",
                "--log-interval",
                "20",
                "--eval-every",
                "20",
                "--eval-episodes",
                "20",
                "--eval-seed",
                "123",
                "--no-tensorboard",
            ],
        },
        {
            "name": "sweepC_idle070_tie012_lr7e5",
            "train": [
                "--env",
                "lawn",
                "--device",
                args.device,
                "--reward-mode",
                "rate",
                "--graph-mode",
                "conflict",
                "--obs-version",
                "v3",
                "--reward-idle-nonempty",
                "-0.7",
                "--agent-id-tiebreak-eps",
                "0.12",
                "--hidden-dim",
                "128",
                "--gat-heads",
                "4",
                "--lr",
                "7e-5",
                "--num-envs",
                "8",
                "--steps-per-update",
                "64",
                "--total-updates",
                "160",
                "--checkpoint-interval",
                "40",
                "--log-interval",
                "20",
                "--eval-every",
                "20",
                "--eval-episodes",
                "20",
                "--eval-seed",
                "123",
                "--no-tensorboard",
            ],
        },
    ]

    summary: dict[str, object] = {"runs": []}

    for r in runs:
        run_name = str(r["name"])
        cmd = [sys.executable, "-m", "gac_mac.scripts.train", "--run-name", run_name, *r["train"]]
        print("\n=== RUN", run_name, "===\n", " ".join(cmd), flush=True)
        if not args.dry_run:
            subprocess.check_call(cmd)

        run_dir = _find_latest_run_dir(results_dir, run_name)
        best_ckpt = run_dir / "checkpoints" / "checkpoint_best_argmax.pth"
        metrics = _load_best_ckpt_metrics(best_ckpt) if best_ckpt.exists() else {}
        eval_metrics = _evaluate(best_ckpt, device=args.device, episodes=args.episodes) if best_ckpt.exists() else {}
        summary["runs"].append(
            {
                "run_name": run_name,
                "run_dir": str(run_dir),
                "best_ckpt": str(best_ckpt),
                "best_ckpt_metrics": metrics,
                "eval_argmax": eval_metrics,
                "train_args": r["train"],
            }
        )

        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"saved sweep summary: {args.out}", flush=True)


if __name__ == "__main__":
    main()

