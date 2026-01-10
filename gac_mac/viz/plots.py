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
    rate_key = "sum_rate_mbps" if results and "sum_rate_mbps" in results[0] else "sum_rate"
    sum_rates = [r[rate_key] for r in results]
    colls = [r["collision_rate"] for r in results]

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.bar(names, sum_rates, color="steelblue")
    plt.title("Mean Throughput (Mbps)" if rate_key == "sum_rate_mbps" else "Mean Sum-Rate")
    plt.grid(True, axis="y", alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.bar(names, colls, color="salmon")
    plt.title("Mean Collision Rate")
    plt.ylim(0.0, 1.0)
    plt.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_scaling_lines(
    *,
    ns: list[int],
    results_by_algo: dict[str, dict[str, list[float]]],
    save_path: str,
) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 4))

    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    for name, metrics in results_by_algo.items():
        ys = metrics.get("sum_rate_mbps", None)
        if ys is None:
            ys = metrics.get("sum_rate", [])
        ax1.plot(ns, ys, marker="o", label=name)
        ax2.plot(ns, metrics.get("avg_delay", []), marker="o", label=name)
        ax3.plot(ns, metrics.get("collision_rate", []), marker="o", label=name)

    ax1.set_title("Throughput (Mbps/frame)" if any("sum_rate_mbps" in m for m in results_by_algo.values()) else "Throughput (sum_rate/frame)")
    ax1.set_xlabel("N (UAVs)")
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Delay (avg_delay/frame)")
    ax2.set_xlabel("N (UAVs)")
    ax2.grid(True, alpha=0.3)

    ax3.set_title("Collision (collision_rate/frame)")
    ax3.set_xlabel("N (UAVs)")
    ax3.set_ylim(0.0, 1.0)
    ax3.grid(True, alpha=0.3)

    ax1.legend(loc="best")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_density_lines(
    *,
    xs: list[float],
    results_by_algo: dict[str, dict[str, list[float]]],
    save_path: str,
) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 4))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    for name, metrics in results_by_algo.items():
        ax1.plot(xs, metrics.get("sum_rate_mbps", []), marker="o", label=name)
        ax2.plot(xs, metrics.get("collision_rate", []), marker="o", label=name)
        ax3.plot(xs, metrics.get("jain", []), marker="o", label=name)

    ax1.set_title("Throughput vs Density")
    ax1.set_xlabel("Density (nodes/km^2)")
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Collision vs Density")
    ax2.set_xlabel("Density (nodes/km^2)")
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(True, alpha=0.3)

    ax3.set_title("Jain vs Density")
    ax3.set_xlabel("Density (nodes/km^2)")
    ax3.set_ylim(0.0, 1.0)
    ax3.grid(True, alpha=0.3)

    ax1.legend(loc="best")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_density_paper(
    *,
    xs: list[float],
    results_by_algo: dict[str, dict[str, list[float]]],
    save_path: str,
    title: str | None = None,
) -> None:
    """Paper-style plot: 3 subplots (thr/coll/jain) with consistent styling and end labels."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
        }
    )

    def _end_label(ax, x: float, y: float, text: str, color: str, dy: float = 0.0) -> None:
        ax.text(x, y + dy, text, color=color, fontsize=9, va="center")

    # Stable ordering to keep labels consistent.
    names = list(results_by_algo.keys())
    if "GAC-MAC" in names:
        names.remove("GAC-MAC")
        names = ["GAC-MAC"] + sorted(names)
    else:
        names = sorted(names)

    # Color/style map: emphasize proposed + key baselines, fade others.
    palette = {
        "GAC-MAC": ("#d62728", "-"),  # red
        "Greedy": ("#000000", "--"),  # black dashed
        "CSMA": ("#1f77b4", "-."),  # blue dash-dot
        "TDMA": ("#2ca02c", ":"),  # green dotted
        "ALOHA": ("#9467bd", ":"),  # purple dotted
        "SATMAC": ("#ff7f0e", "--"),  # orange dashed
        "H-SAT": ("#8c564b", "--"),  # brown dashed
        "Random": ("#7f7f7f", ":"),  # gray dotted
    }

    def _style(name: str) -> tuple[str, str, float]:
        color, ls = palette.get(name, ("#7f7f7f", ":"))
        alpha = 1.0 if name in {"GAC-MAC", "Greedy"} else 0.75
        return color, ls, alpha

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.0))
    ax_thr, ax_coll, ax_jain = axes

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("Density (nodes/km$^2$)")

    ax_thr.set_title("Throughput")
    ax_thr.set_ylabel("Mbps / frame")
    ax_coll.set_title("Collision Rate")
    ax_coll.set_ylabel("collisions / attempts")
    ax_coll.set_ylim(0.0, 1.0)
    ax_jain.set_title("Fairness")
    ax_jain.set_ylabel("Jain's Index")
    ax_jain.set_ylim(0.0, 1.0)

    x_end = float(xs[-1]) if xs else 0.0
    for name in names:
        m = results_by_algo.get(name, {})
        thr = m.get("sum_rate_mbps", [])
        coll = m.get("collision_rate", [])
        jain = m.get("jain", [])
        color, ls, alpha = _style(name)

        lw = 2.8 if name == "GAC-MAC" else (2.3 if name == "Greedy" else 1.8)
        ax_thr.plot(xs, thr, linestyle=ls, color=color, alpha=alpha, linewidth=lw)
        ax_coll.plot(xs, coll, linestyle=ls, color=color, alpha=alpha, linewidth=lw)
        ax_jain.plot(xs, jain, linestyle=ls, color=color, alpha=alpha, linewidth=lw)

        # End labels (remove need for a cluttered legend).
        if xs and thr:
            _end_label(ax_thr, x_end, float(thr[-1]), name, color, dy=0.0)
        if xs and coll:
            _end_label(ax_coll, x_end, float(coll[-1]), name, color, dy=0.0)
        if xs and jain:
            _end_label(ax_jain, x_end, float(jain[-1]), name, color, dy=0.0)

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
