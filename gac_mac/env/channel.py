from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def dbm_to_watt(p_dbm: float | np.ndarray) -> float | np.ndarray:
    return 10.0 ** ((np.asarray(p_dbm) - 30.0) / 10.0)


def watt_to_dbm(p_watt: float | np.ndarray, *, eps: float = 1e-30) -> float | np.ndarray:
    p = np.maximum(np.asarray(p_watt), eps)
    return 10.0 * np.log10(p) + 30.0


@dataclass(frozen=True)
class ChannelSample:
    gain: np.ndarray  # (N,N) full channel power gain (small-scale * large-scale), j->i
    large_scale_gain: np.ndarray  # (N,N) 1 / PL_linear (incl. shadowing), j->i
    is_los: np.ndarray  # (N,N) bool
    d2d_m: np.ndarray  # (N,N)
    d3d_m: np.ndarray  # (N,N)


class ChannelModel:
    """Simplified 3GPP-style aerial channel model (vectorized).

    Conventions:
    - Distances in meters (m).
    - `fc_ghz` in GHz for path-loss formulas (as requested).
    - Returns channel power gain matrices for all directed links j->i.
    """

    def __init__(
        self,
        *,
        fc_ghz: float,
        los_d1_m: float,
        los_p1_m: float,
        shadow_sigma_los_db: float,
        shadow_sigma_nlos_db: float,
        nakagami_m_los: float,
        nakagami_m_nlos: float,
        min_distance_m: float = 1.0,
    ) -> None:
        self.fc_ghz = float(fc_ghz)
        self.los_d1_m = float(los_d1_m)
        self.los_p1_m = float(los_p1_m)
        self.shadow_sigma_los_db = float(shadow_sigma_los_db)
        self.shadow_sigma_nlos_db = float(shadow_sigma_nlos_db)
        self.nakagami_m_los = float(nakagami_m_los)
        self.nakagami_m_nlos = float(nakagami_m_nlos)
        self.min_distance_m = float(min_distance_m)

    def sample(self, positions_m: np.ndarray, rng: np.random.Generator) -> ChannelSample:
        pos = np.asarray(positions_m, dtype=np.float32)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError("positions_m must have shape (N, 3)")

        # Distances (vectorized)
        diffs = pos[:, None, :] - pos[None, :, :]  # (N,N,3)
        d2d = np.linalg.norm(diffs[:, :, :2], axis=2)  # (N,N)
        d3d = np.linalg.norm(diffs, axis=2)  # (N,N)
        d2d = np.maximum(d2d, self.min_distance_m)
        d3d = np.maximum(d3d, self.min_distance_m)

        z = pos[:, 2].astype(np.float32)
        z_avg = (z[:, None] + z[None, :]) * 0.5

        p_los = self._los_probability(d2d_m=d2d, z_avg_m=z_avg)
        is_los = rng.random(size=p_los.shape) < p_los

        # Path loss (dB), using fc in GHz and d in meters
        pl_los_db = 28.0 + 22.0 * np.log10(d3d) + 20.0 * np.log10(self.fc_ghz)
        pl_nlos_db = (
            -17.5
            + (46.0 - 7.0 * np.log10(np.maximum(z_avg, 1.0))) * np.log10(d3d)
            + 20.0 * np.log10(self.fc_ghz)
        )

        # Shadowing (dB)
        shadow_los = rng.normal(loc=0.0, scale=self.shadow_sigma_los_db, size=pl_los_db.shape)
        shadow_nlos = rng.normal(loc=0.0, scale=self.shadow_sigma_nlos_db, size=pl_nlos_db.shape)
        pl_db = np.where(is_los, pl_los_db + shadow_los, pl_nlos_db + shadow_nlos)

        # Large-scale gain
        pl_linear = 10.0 ** (pl_db / 10.0)
        large_scale_gain = 1.0 / pl_linear

        # Small-scale fading power gain (Nakagami-m), unit mean
        m = np.where(is_los, self.nakagami_m_los, self.nakagami_m_nlos).astype(np.float32)
        m = np.maximum(m, 1e-3)
        fading_power = rng.gamma(shape=m, scale=1.0 / m).astype(np.float32)

        gain = fading_power * large_scale_gain

        np.fill_diagonal(gain, 0.0)
        np.fill_diagonal(large_scale_gain, 0.0)

        return ChannelSample(
            gain=gain.astype(np.float32),
            large_scale_gain=large_scale_gain.astype(np.float32),
            is_los=is_los,
            d2d_m=d2d.astype(np.float32),
            d3d_m=d3d.astype(np.float32),
        )

    def _los_probability(self, *, d2d_m: np.ndarray, z_avg_m: np.ndarray) -> np.ndarray:
        # P_LoS(d2d, z_avg) = min(d1/d2d, 1) * (1 - exp(-z_avg/p1)) + exp(-z_avg/p1)
        d2d = np.maximum(d2d_m, self.min_distance_m)
        z_avg = np.maximum(z_avg_m, 0.0)
        exp_term = np.exp(-z_avg / max(self.los_p1_m, 1e-6))
        ratio = np.minimum(self.los_d1_m / d2d, 1.0)
        p_los = ratio * (1.0 - exp_term) + exp_term
        return np.clip(p_los, 0.0, 1.0).astype(np.float32)

