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
        critic_detach_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.action_dim = int(action_dim)
        self.use_agent_id_tiebreak = bool(use_agent_id_tiebreak)
        self.agent_id_tiebreak_eps = float(agent_id_tiebreak_eps)
        self.neighbor_last_action_mask = bool(neighbor_last_action_mask)
        self.neighbor_last_action_penalty = float(neighbor_last_action_penalty)
        self.critic_detach_encoder = bool(critic_detach_encoder)

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

    def _apply_neighbor_last_action_constraints(
        self, logits: torch.Tensor, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        if (not self.neighbor_last_action_mask) and self.neighbor_last_action_penalty <= 0.0:
            return logits
        if edge_index.numel() == 0:
            return logits
        a = int(logits.size(-1))
        if a < 2:
            return logits
        # x layout ends with: last_act (A) (one-hot or multi-hot), last_status_oh (3), in_deg_norm (1)
        f = int(x.size(-1))
        last_act_start = f - (a + 3 + 1)
        if last_act_start < 0:
            return logits
        last_act_mh = x[:, last_act_start : last_act_start + a]
        if last_act_mh.numel() == 0:
            return logits
        k = a - 1  # tx resources (exclude NoTx)
        src = edge_index[0].long()
        dst = edge_index[1].long()
        used_src = (last_act_mh[src, :k] > 0.5).to(dtype=torch.float32)  # (E, k)
        if used_src.numel() == 0:
            return logits
        accum = torch.zeros((x.size(0), k), dtype=torch.float32, device=logits.device)
        accum.index_add_(0, dst, used_src)
        forbidden = accum > 0.0

        out = logits
        if self.neighbor_last_action_penalty > 0.0:
            out = out.clone()
            out[:, :k] = out[:, :k] - self.neighbor_last_action_penalty * forbidden.to(dtype=out.dtype)
        if self.neighbor_last_action_mask:
            out = out.clone()
            out[:, :k] = out[:, :k].masked_fill(forbidden, -1e9)
        return out

    def _select_multi_actions(
        self,
        logits: torch.Tensor,
        *,
        max_tx: int,
        deterministic: bool,
        temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select up to max_tx resources via sequential masked sampling.

        Returns:
            actions: (N, max_tx) with NoTx padded at the end
            logp_sum: (N,)
            entropy_sum: (N,)
        """
        n = int(logits.size(0))
        a = int(logits.size(-1))
        no_tx = a - 1
        l = int(max(1, max_tx))
        temp = float(max(1e-6, temperature))

        masked = logits.clone()
        actions = torch.full((n, l), no_tx, dtype=torch.long, device=logits.device)
        logp_sum = torch.zeros((n,), dtype=logits.dtype, device=logits.device)
        entropy_sum = torch.zeros((n,), dtype=logits.dtype, device=logits.device)
        ended = torch.zeros((n,), dtype=torch.bool, device=logits.device)

        for t in range(l):
            active = ~ended
            if not bool(active.any().item()):
                break
            idx = torch.nonzero(active, as_tuple=False).view(-1)
            dist = Categorical(logits=masked[idx] / temp)
            sel = torch.argmax(masked[idx], dim=-1) if deterministic else dist.sample()
            actions[idx, t] = sel
            logp_sum[idx] = logp_sum[idx] + dist.log_prob(sel)
            entropy_sum[idx] = entropy_sum[idx] + dist.entropy()

            stop = sel == no_tx
            if bool(stop.any().item()):
                ended[idx[stop]] = True

            not_stop = ~stop
            if bool(not_stop.any().item()):
                idx2 = idx[not_stop]
                sel2 = sel[not_stop]
                masked[idx2, sel2] = -1e9

        return actions, logp_sum, entropy_sum

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
        max_tx: int = 1,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Sample actions for rollout (no gradients expected)."""
        logits, h_out = self.forward_logits(x, edge_index, edge_attr, h_in)
        action, logp, entropy = self._select_multi_actions(
            logits, max_tx=int(max_tx), deterministic=bool(deterministic), temperature=float(temperature)
        )
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
        if actions.dim() == 1:
            actions = actions.view(-1, 1)
        n = int(logits.size(0))
        a = int(logits.size(-1))
        no_tx = a - 1
        l = int(actions.size(1))

        masked = logits.clone()
        action_logp = torch.zeros((n,), dtype=logits.dtype, device=logits.device)
        entropy = torch.zeros((n,), dtype=logits.dtype, device=logits.device)
        ended = torch.zeros((n,), dtype=torch.bool, device=logits.device)

        for t in range(l):
            active = ~ended
            if not bool(active.any().item()):
                break
            idx = torch.nonzero(active, as_tuple=False).view(-1)
            dist = Categorical(logits=masked[idx])
            sel = actions[idx, t]
            action_logp[idx] = action_logp[idx] + dist.log_prob(sel)
            entropy[idx] = entropy[idx] + dist.entropy()

            stop = sel == no_tx
            if bool(stop.any().item()):
                ended[idx[stop]] = True

            not_stop = ~stop
            if bool(not_stop.any().item()):
                idx2 = idx[not_stop]
                sel2 = sel[not_stop]
                masked[idx2, sel2] = -1e9

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
        h_v = h.detach() if self.critic_detach_encoder else h
        global_feat = self.global_pool(h_v, index=batch)  # (B, H)
        global_per_node = global_feat[batch]  # (N, H)
        v = self.value_head(torch.cat([h_v, global_per_node], dim=-1)).squeeze(-1)  # (N,)
        return v
