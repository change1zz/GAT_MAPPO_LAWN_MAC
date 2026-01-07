from __future__ import annotations

import argparse
import os

from gac_mac.utils.checkpoint import load_checkpoint
from gac_mac.viz.plots import plot_training_curves


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot training curves from a checkpoint.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--out", type=str, default=None, help="Output png path (default рядом放一个 training_curve.png)")
    p.add_argument("--window", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = load_checkpoint(args.checkpoint)
    rolling = ckpt.get("rolling")
    if not isinstance(rolling, dict):
        raise SystemExit("checkpoint does not contain `rolling` metrics.")

    if args.out is None:
        ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))
        out = os.path.join(ckpt_dir, "training_curve.png")
    else:
        out = args.out

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    plot_training_curves(rolling, out, window=args.window)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()

