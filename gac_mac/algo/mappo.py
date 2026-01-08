from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from gac_mac.algo.buffer import RolloutBatch
from gac_mac.models.agent import GACMACAgent


@dataclass(frozen=True)
class UpdateStats:
    loss: float
    actor_loss: float
    value_loss: float
    entropy: float
    conflict_loss: float


class MAPPOTrainer:
    def __init__(
        self,
        *,
        agent: GACMACAgent,
        optimizer: torch.optim.Optimizer,
        clip_eps: float,
        ppo_epochs: int,
        bptt_len: int,
        value_loss_coef: float,
        entropy_coef: float,
        conflict_loss_coef: float,
        max_grad_norm: float,
        device: torch.device,
    ) -> None:
        self.agent = agent
        self.optimizer = optimizer
        self.clip_eps = float(clip_eps)
        self.ppo_epochs = int(ppo_epochs)
        self.bptt_len = int(bptt_len)
        self.value_loss_coef = float(value_loss_coef)
        self.entropy_coef = float(entropy_coef)
        self.conflict_loss_coef = float(conflict_loss_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.device = device

    def _conflict_loss(self, probs: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Encourage neighbors to use different slot preferences.
        # Use conditional slot distribution softmax(logits_slots) (i.e., exclude NoTx),
        # so this can't be trivially minimized by pushing probability mass into NoTx.
        if edge_index.numel() == 0:
            return torch.zeros((), device=probs.device, dtype=probs.dtype)
        if probs.size(-1) < 2:
            return torch.zeros((), device=probs.device, dtype=probs.dtype)
        slot_probs = probs[:, :-1]  # (N, K), already normalized over slots
        src = edge_index[0].long()
        dst = edge_index[1].long()
        dot = (slot_probs[src] * slot_probs[dst]).sum(dim=-1)  # (E,)
        return dot.mean()

    def update(self, batch: RolloutBatch) -> UpdateStats:
        T = len(batch.rewards)
        assert T > 0

        # Pre-stack targets on CPU then move per-segment to keep memory predictable.
        old_logp = torch.stack(batch.logp, dim=0)  # (T,N)
        advantages = torch.stack(batch.advantages or [], dim=0)  # (T,N)
        returns = torch.stack(batch.returns or [], dim=0)  # (T,N)

        total_loss = 0.0
        total_actor = 0.0
        total_value = 0.0
        total_entropy = 0.0
        total_conflict = 0.0
        steps = 0

        for _epoch in range(self.ppo_epochs):
            seg_start = 0
            while seg_start < T:
                seg_end = min(T, seg_start + self.bptt_len)

                # Truncated BPTT boundary: start hidden is treated as constant.
                h = batch.h_in[seg_start].to(self.device).detach()

                new_logps: list[torch.Tensor] = []
                values: list[torch.Tensor] = []
                entropies: list[torch.Tensor] = []
                conflicts: list[torch.Tensor] = []

                for t in range(seg_start, seg_end):
                    x_t = batch.x[t].to(self.device)
                    ei_t = batch.edge_index[t].to(self.device)
                    ea_t = batch.edge_attr[t].to(self.device)
                    a_t = batch.actions[t].to(self.device)
                    done_t = batch.dones[t].to(self.device)

                    out = self.agent.evaluate_actions(x_t, ei_t, ea_t, h, a_t)
                    h = out.h_out
                    # Reset hidden state across episode boundaries to match rollout.
                    # `done_t` is per-agent (0/1) and may differ across parallel envs.
                    if done_t.numel() > 0:
                        keep = (1.0 - done_t).view(-1, 1)
                        h = h * keep
                    new_logps.append(out.action_logp)
                    values.append(out.values)
                    entropies.append(out.entropy)
                    if self.conflict_loss_coef > 0.0:
                        slot_probs = torch.softmax(out.logits[:, :-1], dim=-1)
                        conflicts.append(self._conflict_loss(slot_probs, ei_t))

                new_logp_seg = torch.stack(new_logps, dim=0)  # (L,N)
                values_seg = torch.stack(values, dim=0)  # (L,N)
                entropy_seg = torch.stack(entropies, dim=0)  # (L,N)
                conflict_seg = torch.stack(conflicts, dim=0).mean() if conflicts else torch.zeros((), device=self.device)

                old_logp_seg = old_logp[seg_start:seg_end].to(self.device)
                adv_seg = advantages[seg_start:seg_end].to(self.device)
                ret_seg = returns[seg_start:seg_end].to(self.device)

                ratio = torch.exp(new_logp_seg - old_logp_seg)
                surr1 = ratio * adv_seg
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_seg
                actor_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values_seg, ret_seg)
                entropy = entropy_seg.mean()

                loss = (
                    actor_loss
                    + self.value_loss_coef * value_loss
                    - self.entropy_coef * entropy
                    + self.conflict_loss_coef * conflict_seg
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += float(loss.item())
                total_actor += float(actor_loss.item())
                total_value += float(value_loss.item())
                total_entropy += float(entropy.item())
                total_conflict += float(conflict_seg.item())
                steps += 1

                seg_start = seg_end

        denom = max(1, steps)
        return UpdateStats(
            loss=total_loss / denom,
            actor_loss=total_actor / denom,
            value_loss=total_value / denom,
            entropy=total_entropy / denom,
            conflict_loss=total_conflict / denom,
        )
