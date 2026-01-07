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
        self.max_grad_norm = float(max_grad_norm)
        self.device = device

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

                for t in range(seg_start, seg_end):
                    x_t = batch.x[t].to(self.device)
                    ei_t = batch.edge_index[t].to(self.device)
                    ea_t = batch.edge_attr[t].to(self.device)
                    a_t = batch.actions[t].to(self.device)
                    done_t = batch.dones[t].to(self.device)

                    # Action mask: queue feature is always at index 3 by GraphBuilder convention.
                    # If queue==0, only allow No-Tx.
                    action_mask = None
                    if x_t.numel() > 0:
                        has_pkt = x_t[:, 3] > 0.0
                        if bool((~has_pkt).any().item()):
                            action_mask = torch.ones((x_t.size(0), self.agent.action_dim), dtype=torch.bool, device=x_t.device)
                            action_mask[~has_pkt, : self.agent.action_dim - 1] = False
                    out = self.agent.evaluate_actions(x_t, ei_t, ea_t, h, a_t, action_mask=action_mask)
                    h = out.h_out
                    # Reset hidden state across episode boundaries to match rollout.
                    # `done_t` is per-agent (0/1) and may differ across parallel envs.
                    if done_t.numel() > 0:
                        keep = (1.0 - done_t).view(-1, 1)
                        h = h * keep
                    new_logps.append(out.action_logp)
                    values.append(out.values)
                    entropies.append(out.entropy)

                new_logp_seg = torch.stack(new_logps, dim=0)  # (L,N)
                values_seg = torch.stack(values, dim=0)  # (L,N)
                entropy_seg = torch.stack(entropies, dim=0)  # (L,N)

                old_logp_seg = old_logp[seg_start:seg_end].to(self.device)
                adv_seg = advantages[seg_start:seg_end].to(self.device)
                ret_seg = returns[seg_start:seg_end].to(self.device)

                ratio = torch.exp(new_logp_seg - old_logp_seg)
                surr1 = ratio * adv_seg
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_seg
                actor_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values_seg, ret_seg)
                entropy = entropy_seg.mean()

                loss = actor_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += float(loss.item())
                total_actor += float(actor_loss.item())
                total_value += float(value_loss.item())
                total_entropy += float(entropy.item())
                steps += 1

                seg_start = seg_end

        denom = max(1, steps)
        return UpdateStats(
            loss=total_loss / denom,
            actor_loss=total_actor / denom,
            value_loss=total_value / denom,
            entropy=total_entropy / denom,
        )
