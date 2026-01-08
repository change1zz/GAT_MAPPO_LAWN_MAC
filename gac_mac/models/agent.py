from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from gac_mac.models.encoder import STGNNEncoder


@dataclass(frozen=True)
class StepOutputs:
    action_logp: torch.Tensor  # (N,)
    values: torch.Tensor  # (N,)
    entropy: torch.Tensor  # (N,)
    h_out: torch.Tensor  # (N, H)


class GACMACAgent(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        action_dim: int,
        gat_heads: int = 2,
        *,
        use_edge_attr: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.action_dim = int(action_dim)

        edge_dim = 1 if use_edge_attr else None
        self.encoder = STGNNEncoder(input_dim=input_dim, hidden_dim=hidden_dim, heads=gat_heads, edge_dim=edge_dim)

        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

        # Global value with attention pooling (CTDE). We broadcast V_global to each node.
        # Use the non-deprecated aggregation API in recent PyG.
        try:
            from torch_geometric.nn.aggr import AttentionalAggregation
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "torch_geometric is required. Install PyTorch Geometric before running."
            ) from exc

        gate_nn = nn.Sequential(nn.Linear(hidden_dim, 1))
        self.global_pool = AttentionalAggregation(gate_nn=gate_nn)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def initial_hidden(self, num_nodes: int, device: torch.device) -> torch.Tensor:
        return torch.zeros((num_nodes, self.hidden_dim), dtype=torch.float32, device=device)

    def act(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor | None, h_in: torch.Tensor
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Sample actions for rollout (no gradients expected)."""
        h_out = self.encoder(x, edge_index, edge_attr, h_in)
        logits = self.actor(h_out)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)
        entropy = dist.entropy()
        values = self._values_from_hidden(h_out, batch=None)
        return action, logp, values, entropy, h_out

    def evaluate_actions(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
        h_in: torch.Tensor,
        actions: torch.Tensor,
    ) -> StepOutputs:
        h_out = self.encoder(x, edge_index, edge_attr, h_in)
        logits = self.actor(h_out)
        dist = Categorical(logits=logits)
        action_logp = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self._values_from_hidden(h_out, batch=None)
        return StepOutputs(action_logp=action_logp, values=values, entropy=entropy, h_out=h_out)

    def value_only(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor | None, h_in: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_out = self.encoder(x, edge_index, edge_attr, h_in)
        values = self._values_from_hidden(h_out, batch=None)
        return values, h_out

    def _values_from_hidden(self, h: torch.Tensor, batch: torch.Tensor | None) -> torch.Tensor:
        if batch is None:
            batch = torch.zeros((h.size(0),), dtype=torch.long, device=h.device)
        global_feat = self.global_pool(h, index=batch)  # (B, H)
        global_per_node = global_feat[batch]  # (N, H)
        v = self.value_head(torch.cat([h, global_per_node], dim=-1)).squeeze(-1)  # (N,)
        return v
