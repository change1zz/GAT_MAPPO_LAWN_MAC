from __future__ import annotations

from typing import Any

import numpy as np


def moving_average(x: list[float], window: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.size == 0:
        return arr
    if window <= 1:
        return arr
    if arr.size < window:
        return np.full_like(arr, arr.mean())
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(arr, kernel, mode="valid")


def plot_training_curves(rolling: dict[str, Any], save_path: str, window: int = 20) -> None:
    import matplotlib.pyplot as plt

    reward = rolling.get("reward", [])
    sum_rate = rolling.get("sum_rate", [])
    collision = rolling.get("collision_rate", [])

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(moving_average(reward, window), label="reward")
    plt.title("Reward")
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(moving_average(sum_rate, window), label="sum_rate")
    plt.title("Sum-Rate")
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(moving_average(collision, window), label="collision_rate")
    plt.title("Collision")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_eval_summary(results: list[dict[str, float]], save_path: str) -> None:
    import matplotlib.pyplot as plt

    names = [r["name"] for r in results]
    # Prefer Mbps if present; fall back to sum spectral efficiency.
    has_mbps = all("throughput_mbps" in r for r in results)
    sum_rates = [r.get("throughput_mbps", r["sum_rate"]) for r in results]
    colls = [r["collision_rate"] for r in results]
    jains = [r.get("jain_throughput", 0.0) for r in results]

    plt.figure(figsize=(14, 4))

    plt.subplot(1, 3, 1)
    plt.bar(names, sum_rates, color="steelblue")
    plt.title("Mean Throughput")
    plt.ylabel("Mbps" if has_mbps else "bits/s/Hz per frame")
    plt.grid(True, axis="y", alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.bar(names, colls, color="salmon")
    plt.title("Mean Collision Rate")
    plt.ylabel("fraction")
    plt.ylim(0.0, 1.0)
    plt.grid(True, axis="y", alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.bar(names, jains, color="seagreen")
    plt.title("Jain Fairness (Throughput)")
    plt.ylabel("index")
    plt.ylim(0.0, 1.0)
    plt.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_scaling_lines(
    *,
    xs: list[float],
    results_by_algo: dict[str, dict[str, list[float]]],
    save_path: str,
    x_label: str = "N (UAVs)",
) -> None:
    import matplotlib.pyplot as plt

    has_jain = any("jain_throughput" in m for m in results_by_algo.values())
    fig = plt.figure(figsize=(18, 4) if has_jain else (14, 4))

    cols = 4 if has_jain else 3
    ax1 = fig.add_subplot(1, cols, 1)
    ax2 = fig.add_subplot(1, cols, 2)
    ax3 = fig.add_subplot(1, cols, 3)
    ax4 = fig.add_subplot(1, cols, 4) if has_jain else None

    use_mbps = any("throughput_mbps" in m for m in results_by_algo.values())
    for name, metrics in results_by_algo.items():
        y = metrics.get("throughput_mbps", metrics.get("sum_rate", []))
        ax1.plot(xs, y, marker="o", label=name)
        ax2.plot(xs, metrics.get("avg_delay", []), marker="o", label=name)
        ax3.plot(xs, metrics.get("collision_rate", []), marker="o", label=name)
        if ax4 is not None:
            ax4.plot(xs, metrics.get("jain_throughput", []), marker="o", label=name)

    ax1.set_title("Throughput")
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Mbps" if use_mbps else "bits/s/Hz per frame")
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Delay")
    ax2.set_xlabel(x_label)
    ax2.set_ylabel("frames")
    ax2.grid(True, alpha=0.3)

    ax3.set_title("Collision Rate")
    ax3.set_xlabel(x_label)
    ax3.set_ylabel("fraction")
    ax3.set_ylim(0.0, 1.0)
    ax3.grid(True, alpha=0.3)

    if ax4 is not None:
        ax4.set_title("Jain Fairness")
        ax4.set_xlabel(x_label)
        ax4.set_ylabel("index")
        ax4.set_ylim(0.0, 1.0)
        ax4.grid(True, alpha=0.3)

    ax1.legend(loc="best")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
