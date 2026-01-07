from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RolloutBatch:
    # Time-major: (T, N, ...)
    x: list[torch.Tensor]  # each (N, F)
    edge_index: list[torch.Tensor]  # each (2, E)
    edge_attr: list[torch.Tensor]  # each (E, D)
    actions: list[torch.Tensor]  # each (N,)
    logp: list[torch.Tensor]  # each (N,)
    values: list[torch.Tensor]  # each (N,)
    rewards: list[torch.Tensor]  # each (N,)
    dones: list[torch.Tensor]  # each (N,) float 0/1
    h_in: list[torch.Tensor]  # each (N, H)

    advantages: list[torch.Tensor] | None = None  # each (N,)
    returns: list[torch.Tensor] | None = None  # each (N,)

    def __len__(self) -> int:
        return len(self.rewards)


class RolloutBuffer:
    def __init__(self) -> None:
        self.reset()

    def __len__(self) -> int:
        return len(self.rewards)

    def reset(self) -> None:
        self.x: list[torch.Tensor] = []
        self.edge_index: list[torch.Tensor] = []
        self.edge_attr: list[torch.Tensor] = []
        self.actions: list[torch.Tensor] = []
        self.logp: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []
        self.rewards: list[torch.Tensor] = []
        self.dones: list[torch.Tensor] = []
        self.h_in: list[torch.Tensor] = []
        self.advantages: list[torch.Tensor] | None = None
        self.returns: list[torch.Tensor] | None = None

    def add(
        self,
        *,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        actions: torch.Tensor,
        logp: torch.Tensor,
        values: torch.Tensor,
        rewards: torch.Tensor,
        done: bool | torch.Tensor,
        h_in: torch.Tensor,
    ) -> None:
        self.x.append(x.detach().cpu())
        self.edge_index.append(edge_index.detach().cpu())
        self.edge_attr.append(edge_attr.detach().cpu())
        self.actions.append(actions.detach().cpu())
        self.logp.append(logp.detach().cpu())
        self.values.append(values.detach().cpu())
        self.rewards.append(rewards.detach().cpu())
        if isinstance(done, torch.Tensor):
            done_f = done.detach().to(dtype=rewards.dtype, device=rewards.device).view_as(rewards)
        else:
            done_f = torch.full_like(rewards, 1.0 if done else 0.0)
        self.dones.append(done_f.detach().cpu())
        self.h_in.append(h_in.detach().cpu())

    def compute_gae(
        self,
        *,
        last_values: torch.Tensor,
        gamma: float,
        gae_lambda: float,
        normalize: bool = True,
        eps: float = 1e-8,
    ) -> None:
        T = len(self)
        assert T > 0
        last_values = last_values.detach().cpu()

        advantages: list[torch.Tensor] = [torch.zeros_like(self.rewards[0]) for _ in range(T)]
        returns: list[torch.Tensor] = [torch.zeros_like(self.rewards[0]) for _ in range(T)]

        gae = torch.zeros_like(self.rewards[0])
        for t in reversed(range(T)):
            next_values = last_values if (t == T - 1) else self.values[t + 1]
            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            advantages[t] = gae
            returns[t] = gae + self.values[t]

        if normalize:
            flat = torch.cat([a.view(-1) for a in advantages], dim=0)
            mean = flat.mean()
            std = flat.std(unbiased=False) + eps
            advantages = [(a - mean) / std for a in advantages]

        self.advantages = advantages
        self.returns = returns

    def as_batch(self) -> RolloutBatch:
        if self.advantages is None or self.returns is None:
            raise RuntimeError("Call compute_gae() before requesting a batch.")
        return RolloutBatch(
            x=self.x,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            actions=self.actions,
            logp=self.logp,
            values=self.values,
            rewards=self.rewards,
            dones=self.dones,
            h_in=self.h_in,
            advantages=self.advantages,
            returns=self.returns,
        )
