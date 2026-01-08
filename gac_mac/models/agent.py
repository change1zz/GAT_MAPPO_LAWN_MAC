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
    logits: torch.Tensor  # (N, A)
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
        use_agent_id_tiebreak: bool = False,
        agent_id_tiebreak_eps: float = 0.0,
        neighbor_last_action_mask: bool = False,
        neighbor_last_action_penalty: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.action_dim = int(action_dim)
        self.use_agent_id_tiebreak = bool(use_agent_id_tiebreak)
        self.agent_id_tiebreak_eps = float(agent_id_tiebreak_eps)
        self.neighbor_last_action_mask = bool(neighbor_last_action_mask)
        self.neighbor_last_action_penalty = float(neighbor_last_action_penalty)

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

    def _apply_agent_id_tiebreak(self, logits: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if not self.use_agent_id_tiebreak:
            return logits
        eps = self.agent_id_tiebreak_eps
        if eps <= 0.0:
            return logits
        # GraphBuilder v3 layout places agent-id at feature index 6:
        # pos(3) + q(1) + energy(1) + sig_norm(1) + agent_id(1) => agent_id at 6 (0-based).
        if x.size(-1) <= 6 or logits.size(-1) < 2:
            return logits
        agent_id = x[:, 6].to(dtype=logits.dtype)  # normalized ~ [0,1)
        k = int(logits.size(-1) - 1)  # slots, excluding NoTx
        if k <= 0:
            return logits
        pref = torch.clamp((agent_id * float(k)).to(dtype=torch.long), 0, k - 1)  # (N,)
        bias = torch.zeros_like(logits)
        bias[torch.arange(logits.size(0), device=logits.device), pref] = eps
        return logits + bias

    def _apply_neighbor_last_action_constraints(self, logits: torch.Tensor, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if (not self.neighbor_last_action_mask) and self.neighbor_last_action_penalty <= 0.0:
            return logits
        if edge_index.numel() == 0:
            return logits
        a = int(logits.size(-1))
        if a < 2:
            return logits
        # x layout ends with: last_act_oh (A), last_status_oh (3), in_deg_norm (1)
        f = int(x.size(-1))
        last_act_start = f - (a + 3 + 1)
        if last_act_start < 0:
            return logits
        last_act_oh = x[:, last_act_start : last_act_start + a]
        if last_act_oh.numel() == 0:
            return logits
        last_act = torch.argmax(last_act_oh, dim=-1)  # (N,)

        k = a - 1  # slots
        src = edge_index[0].long()
        dst = edge_index[1].long()
        last_src = last_act[src]
        valid = (last_src >= 0) & (last_src < k)
        if not bool(valid.any().item()):
            return logits

        forbidden = torch.zeros((x.size(0), k), dtype=torch.bool, device=logits.device)
        forbidden[dst[valid], last_src[valid]] = True

        out = logits
        if self.neighbor_last_action_penalty > 0.0:
            out = out.clone()
            out[:, :k] = out[:, :k] - self.neighbor_last_action_penalty * forbidden.to(dtype=out.dtype)
        if self.neighbor_last_action_mask:
            out = out.clone()
            out[:, :k] = out[:, :k].masked_fill(forbidden, -1e9)
        return out

    def forward_logits(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor | None, h_in: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_out = self.encoder(x, edge_index, edge_attr, h_in)
        logits = self.actor(h_out)
        logits = self._apply_agent_id_tiebreak(logits, x)
        logits = self._apply_neighbor_last_action_constraints(logits, x, edge_index)
        return logits, h_out

    def initial_hidden(self, num_nodes: int, device: torch.device) -> torch.Tensor:
        return torch.zeros((num_nodes, self.hidden_dim), dtype=torch.float32, device=device)

    def act(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
        h_in: torch.Tensor,
        *,
        deterministic: bool = False,
        temperature: float = 1.0,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Sample actions for rollout (no gradients expected)."""
        logits, h_out = self.forward_logits(x, edge_index, edge_attr, h_in)
        temp = float(max(1e-6, temperature))
        dist = Categorical(logits=logits / temp)
        action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
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
        logits, h_out = self.forward_logits(x, edge_index, edge_attr, h_in)
        dist = Categorical(logits=logits)
        action_logp = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self._values_from_hidden(h_out, batch=None)
        return StepOutputs(action_logp=action_logp, values=values, entropy=entropy, logits=logits, h_out=h_out)

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
