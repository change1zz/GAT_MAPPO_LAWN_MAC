from __future__ import annotations


def spectral_eff_sum_to_mbps_per_frame(
    sum_rate_bits_per_s_hz_per_frame: float, *, bandwidth_hz: float, num_slots: int
) -> float:
    """Convert sum of spectral efficiencies (sum log2(1+SINR) over successful links per frame)
    to Mbps at the frame level.

    Model assumption (matches our env):
    - One environment step = one MAC frame with `num_slots` orthogonal time slots.
    - Each successful transmission occupies exactly one slot within the frame.

    Then:
      bits_per_frame = bandwidth_hz * slot_duration_s * sum_rate
      frame_duration = num_slots * slot_duration_s
      throughput_bps = bits_per_frame / frame_duration = bandwidth_hz/num_slots * sum_rate

    Slot duration cancels, so no explicit timing parameter is required.
    """

    if num_slots <= 0:
        return 0.0
    return float(sum_rate_bits_per_s_hz_per_frame) * float(bandwidth_hz) / float(num_slots) / 1e6


def jain_index(x) -> float:
    """Jain's fairness index for non-negative quantities.

    Returns 0 for empty/all-zero inputs.
    """

    import numpy as np

    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return 0.0
    arr = np.maximum(arr, 0.0)
    s1 = float(arr.sum())
    if s1 <= 0.0:
        return 0.0
    s2 = float((arr * arr).sum())
    if s2 <= 0.0:
        return 0.0
    return float((s1 * s1) / (float(arr.size) * s2))
