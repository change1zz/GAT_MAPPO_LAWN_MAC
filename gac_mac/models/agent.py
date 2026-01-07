from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Categorical

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
        critic_mode: str = "global_mean",
        policy_mode: str = "hierarchical",
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.action_dim = int(action_dim)
        self.critic_mode = str(critic_mode)
        self.policy_mode = str(policy_mode)

        edge_dim = 1 if use_edge_attr else None
        self.encoder = STGNNEncoder(input_dim=input_dim, hidden_dim=hidden_dim, heads=gat_heads, edge_dim=edge_dim)

        # Actor:
        # - flat: categorical over (K slots + No-Tx)
        # - hierarchical: Bernoulli(Tx) + categorical(slot|Tx), reducing "No-Tx dominates when K grows" effect
        # Keep the legacy name `actor` to preserve checkpoint compatibility.
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )
        self.tx_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # logit for Tx (Bernoulli)
        )
        self.slot_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim - 1),  # logits for K slots
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
        # Legacy per-node head (kept for checkpoint compatibility).
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        # Global head (state-value) for strict "broadcast V" critic mode.
        self.value_head_global = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def initial_hidden(self, num_nodes: int, device: torch.device) -> torch.Tensor:
        return torch.zeros((num_nodes, self.hidden_dim), dtype=torch.float32, device=device)

    def act(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
        h_in: torch.Tensor,
        *,
        action_mask: torch.Tensor | None = None,  # (N, A) bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample actions for rollout (no gradients expected)."""
        h_out = self.encoder(x, edge_index, edge_attr, h_in)

        if self.policy_mode == "flat":
            logits = self.actor(h_out)
            if action_mask is not None:
                logits = logits.masked_fill(~action_mask, -1e9)
            dist = Categorical(logits=logits)
            action = dist.sample()
            logp = dist.log_prob(action)
            entropy = dist.entropy()
        else:
            # Hierarchical policy: first decide Tx, then choose a slot if Tx.
            # Action encoding stays compatible: slot in [0..K-1], No-Tx is K.
            slot_logits = self.slot_head(h_out)  # (N, K)
            tx_logit = self.tx_head(h_out).squeeze(-1)  # (N,)

            if action_mask is not None:
                # If No-Tx is the only valid action, force tx_logit to -inf (p_tx=0).
                # We detect that via "all slots invalid".
                slots_valid = action_mask[:, : self.action_dim - 1].any(dim=1)
                tx_logit = torch.where(slots_valid, tx_logit, torch.full_like(tx_logit, -1e9))

            tx_dist = Bernoulli(logits=tx_logit)
            tx = tx_dist.sample()  # (N,) in {0,1}

            slot_dist = Categorical(logits=slot_logits)
            slot = slot_dist.sample()  # (N,)

            no_tx_action = torch.full_like(slot, self.action_dim - 1)
            action = torch.where(tx > 0.5, slot, no_tx_action)

            # log p(a) = log p(tx) + 1[tx] * log p(slot)
            logp = tx_dist.log_prob(tx) + tx * slot_dist.log_prob(slot)
            entropy = tx_dist.entropy() + tx * slot_dist.entropy()

        values = self._values_from_hidden(h_out, batch=None)
        return action, logp, values, entropy, h_out

    def evaluate_actions(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
        h_in: torch.Tensor,
        actions: torch.Tensor,
        *,
        action_mask: torch.Tensor | None = None,  # (N, A) bool
    ) -> StepOutputs:
        h_out = self.encoder(x, edge_index, edge_attr, h_in)

        if self.policy_mode == "flat":
            logits = self.actor(h_out)
            if action_mask is not None:
                logits = logits.masked_fill(~action_mask, -1e9)
            dist = Categorical(logits=logits)
            action_logp = dist.log_prob(actions)
            entropy = dist.entropy()
        else:
            slot_logits = self.slot_head(h_out)  # (N, K)
            tx_logit = self.tx_head(h_out).squeeze(-1)  # (N,)

            if action_mask is not None:
                slots_valid = action_mask[:, : self.action_dim - 1].any(dim=1)
                tx_logit = torch.where(slots_valid, tx_logit, torch.full_like(tx_logit, -1e9))

            tx = (actions != (self.action_dim - 1)).to(dtype=slot_logits.dtype)  # (N,)
            slot = torch.clamp(actions, 0, self.action_dim - 2)  # only meaningful when tx==1

            tx_dist = Bernoulli(logits=tx_logit)
            slot_dist = Categorical(logits=slot_logits)

            action_logp = tx_dist.log_prob(tx) + tx * slot_dist.log_prob(slot)
            entropy = tx_dist.entropy() + tx * slot_dist.entropy()

        values = self._values_from_hidden(h_out, batch=None)
        return StepOutputs(action_logp=action_logp, values=values, entropy=entropy, h_out=h_out)

    def value_only(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor | None, h_in: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_out = self.encoder(x, edge_index, edge_attr, h_in)
        values = self._values_from_hidden(h_out, batch=None)
        return values, h_out

    def action_logits(self, h: torch.Tensor) -> torch.Tensor:
        """Return flat (K+1) logits for evaluation/debugging."""
        if self.policy_mode == "flat":
            return self.actor(h)
        slot_logits = self.slot_head(h)  # (N, K)
        tx_logit = self.tx_head(h).squeeze(-1)  # (N,)
        p_tx = torch.sigmoid(tx_logit).clamp(1e-6, 1.0 - 1e-6)
        log_p_tx = torch.log(p_tx)
        log_p_no = torch.log1p(-p_tx)
        slot_logp = F.log_softmax(slot_logits, dim=-1)
        # Convert hierarchical to comparable joint log-prob logits for visualization (up to a constant).
        joint = torch.cat([log_p_tx.unsqueeze(-1) + slot_logp, log_p_no.unsqueeze(-1)], dim=-1)
        return joint

    def _values_from_hidden(self, h: torch.Tensor, batch: torch.Tensor | None) -> torch.Tensor:
        if batch is None:
            batch = torch.zeros((h.size(0),), dtype=torch.long, device=h.device)
        global_feat = self.global_pool(h, index=batch)  # (B, H)

        if self.critic_mode == "node":
            global_per_node = global_feat[batch]  # (N, H)
            v_node = self.value_head(torch.cat([h, global_per_node], dim=-1)).squeeze(-1)  # (N,)
            return v_node

        if self.critic_mode == "global":
            v_global = self.value_head_global(global_feat).squeeze(-1)  # (B,)
            return v_global[batch]

        # Default: "global_mean" - compute per-node values (legacy head) then reduce to a single
        # global V per graph, broadcast back to each node.
        global_per_node = global_feat[batch]  # (N, H)
        v_node = self.value_head(torch.cat([h, global_per_node], dim=-1)).squeeze(-1)  # (N,)
        bsz = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        sum_v = torch.zeros((bsz,), dtype=v_node.dtype, device=v_node.device)
        cnt = torch.zeros((bsz,), dtype=v_node.dtype, device=v_node.device)
        ones = torch.ones_like(v_node)
        sum_v.index_add_(0, batch, v_node)
        cnt.index_add_(0, batch, ones)
        v_global = sum_v / torch.clamp(cnt, min=1.0)
        return v_global[batch]
