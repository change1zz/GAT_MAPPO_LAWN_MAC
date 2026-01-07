from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MobilityState:
    positions_m: np.ndarray  # (N,3)
    velocities_mps: np.ndarray  # (N,3)


class GaussMarkovMobility:
    """Vectorized Gauss-Markov mobility with reflecting boundaries."""

    def __init__(
        self,
        *,
        num_nodes: int,
        bounds_xyz_m: tuple[float, float, float],
        alpha: float = 0.8,
        sigma_v: float = 1.0,
        v_max_mps: float = 20.0,
        dt_s: float = 0.1,
        v_mean_mps: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        self.N = int(num_nodes)
        self.bounds = tuple(float(x) for x in bounds_xyz_m)
        self.alpha = float(alpha)
        self.sigma_v = float(sigma_v)
        self.v_max = float(v_max_mps)
        self.dt = float(dt_s)
        self.v_mean = np.asarray(v_mean_mps, dtype=np.float32).reshape(1, 3)

        self.state = MobilityState(
            positions_m=np.zeros((self.N, 3), dtype=np.float32),
            velocities_mps=np.zeros((self.N, 3), dtype=np.float32),
        )

    def reset(self, rng: np.random.Generator) -> MobilityState:
        L, W, H = self.bounds
        xy = rng.uniform(0.0, [L, W], size=(self.N, 2))
        z = rng.uniform(0.25 * H, H, size=(self.N, 1))
        self.state.positions_m = np.concatenate([xy, z], axis=1).astype(np.float32)

        self.state.velocities_mps = rng.normal(loc=0.0, scale=self.sigma_v, size=(self.N, 3)).astype(np.float32)
        self._clip_speed()
        return self.state

    def step(self, rng: np.random.Generator) -> MobilityState:
        a = self.alpha
        w = rng.normal(loc=0.0, scale=self.sigma_v, size=(self.N, 3)).astype(np.float32)
        self.state.velocities_mps = (
            a * self.state.velocities_mps + (1.0 - a) * self.v_mean + np.sqrt(max(1e-6, 1.0 - a * a)) * w
        )
        self._clip_speed()

        self.state.positions_m = self.state.positions_m + self.state.velocities_mps * self.dt
        self._reflect_bounds()
        return self.state

    def _clip_speed(self) -> None:
        v = self.state.velocities_mps
        speed = np.linalg.norm(v, axis=1, keepdims=True)
        scale = np.minimum(1.0, self.v_max / np.maximum(speed, 1e-6))
        self.state.velocities_mps = v * scale.astype(np.float32)

    def _reflect_bounds(self) -> None:
        pos = self.state.positions_m
        vel = self.state.velocities_mps
        bounds = np.asarray(self.bounds, dtype=np.float32).reshape(1, 3)

        # Reflect at 0
        low = pos < 0.0
        pos = np.where(low, -pos, pos)
        vel = np.where(low, -vel, vel)

        # Reflect at upper bound
        high = pos > bounds
        pos = np.where(high, 2.0 * bounds - pos, pos)
        vel = np.where(high, -vel, vel)

        # Final clamp (numerical)
        self.state.positions_m = np.clip(pos, 0.0, bounds).astype(np.float32)
        self.state.velocities_mps = vel.astype(np.float32)
