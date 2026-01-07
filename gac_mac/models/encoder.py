from __future__ import annotations

import torch
import torch.nn as nn


class STGNNEncoder(nn.Module):
    """2-layer GATv2 encoder + GRUCell memory."""

    def __init__(self, input_dim: int, hidden_dim: int, heads: int = 2, edge_dim: int | None = None) -> None:
        super().__init__()

        try:
            from torch_geometric.nn import GATv2Conv
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "torch_geometric is required. Install PyTorch Geometric before running."
            ) from exc

        self.use_edge_attr = edge_dim is not None

        self.gat1 = GATv2Conv(input_dim, hidden_dim, heads=heads, concat=True, edge_dim=edge_dim)
        self.norm1 = nn.LayerNorm(hidden_dim * heads)

        self.gat2 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=1, concat=False, edge_dim=edge_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.res_proj = nn.Linear(hidden_dim * heads, hidden_dim, bias=False)
        self.act = nn.ELU()

        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
        h_in: torch.Tensor,
    ) -> torch.Tensor:
        # Spatial
        ea = edge_attr if self.use_edge_attr else None
        out1 = self.gat1(x, edge_index, edge_attr=ea)
        out1 = self.act(self.norm1(out1))

        out2 = self.gat2(out1, edge_index, edge_attr=ea)
        out2 = self.norm2(out2)
        out2 = self.act(out2 + self.res_proj(out1))

        # Temporal
        h_out = self.gru(out2, h_in)
        return h_out
