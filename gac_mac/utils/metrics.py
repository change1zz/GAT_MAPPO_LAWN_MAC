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

