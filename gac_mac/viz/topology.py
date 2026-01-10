from __future__ import annotations

import numpy as np


def plot_topology_snapshot(
    *,
    positions_m: np.ndarray,  # (N,3)
    edge_index: np.ndarray,  # (2,E)
    actions: np.ndarray,  # (N,)
    num_slots: int,
    num_channels: int = 1,
    save_path: str,
    title: str | None = None,
) -> None:
    import matplotlib.pyplot as plt
    import networkx as nx

    pos = np.asarray(positions_m, dtype=np.float32)
    x = pos[:, 0]
    y = pos[:, 1]

    G = nx.Graph()
    for i in range(pos.shape[0]):
        G.add_node(int(i), pos=(float(x[i]), float(y[i])))

    if edge_index.size > 0:
        src = edge_index[0].astype(int)
        dst = edge_index[1].astype(int)
        edges = list({(min(u, v), max(u, v)) for u, v in zip(src, dst)})
        G.add_edges_from(edges)

    pos_dict = nx.get_node_attributes(G, "pos")

    cmap = plt.get_cmap("tab10")
    c = int(max(1, num_channels))
    no_tx = int(c * int(num_slots))
    colors = []
    for a in actions.astype(int):
        if a == no_tx:
            colors.append("lightgray")
        else:
            # Color by slot index, ignore channel for readability.
            slot = int(a % int(num_slots)) if num_slots > 0 else 0
            colors.append(cmap(slot % 10))

    plt.figure(figsize=(10, 10))
    nx.draw(
        G,
        pos_dict,
        node_color=colors,
        with_labels=True,
        node_size=300,
        edge_color="gray",
        width=0.5,
        alpha=0.8,
    )
    if title:
        plt.title(title)
    plt.savefig(save_path, dpi=200)
    plt.close()
