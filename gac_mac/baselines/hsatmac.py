from __future__ import annotations

import numpy as np


class HSATMACAgent:
    """Paper-aligned(ish) H-SATMAC-style baseline (hybrid semi-persistent TDMA + slot-group CSMA/CA).

    This is a best-effort mapping of the H-SATMAC paper mechanisms onto our simplified frame simulator:
    - Basic Slot (BS): each node semi-persistently occupies one (channel,slot) resource; on collision, reselect.
    - Slot Group (SG): consecutive slots within a channel; nodes in the same "geohash region" share the same SG
      for a limited validity period Tvalid. Within the SG, a contention window (CW) is formed by consecutive free slots.

    Notes/approximations:
    - We do not simulate separate control packets (FI/SGI). Instead, we infer last-frame occupancy via `env.last_actions`
      and the directed adjacency in `obs.adj` to build a 2-hop perception.
    - CSMA/CA is approximated as selecting one slot within the contention window using a per-node contention window size.
    """

    STATUS_SUCCESS = 0
    STATUS_COLLISION = 1
    STATUS_IDLE = 2

    def __init__(
        self,
        num_slots: int,
        num_nodes: int,
        *,
        num_channels: int = 1,
        lsg: int = 4,
        lmin: int = 2,
        tvalid: int = 4,
        num_regions: int = 10,
        cw_min: int = 4,
        cw_max: int = 64,
        burst_q_norm: float = 0.6,
    ) -> None:
        self.K = int(num_slots)
        self.C = int(max(1, num_channels))
        self.A = int(self.C * self.K + 1)
        self.N = int(num_nodes)

        self.lsg = int(max(1, lsg))
        self.lmin = int(max(1, min(lmin, self.lsg)))
        self.tvalid = int(max(1, tvalid))
        self.num_regions = int(max(1, num_regions))
        self.cw_min = int(max(1, cw_min))
        self.cw_max = int(max(self.cw_min, cw_max))
        self.burst_q_norm = float(burst_q_norm)

        self.bs = np.full((self.N,), fill_value=(self.A - 1), dtype=np.int64)
        self.cw = np.full((self.N,), fill_value=self.cw_min, dtype=np.int32)
        # region -> group index, and remaining validity
        self.region_group = np.full((self.num_regions,), fill_value=-1, dtype=np.int32)
        self.region_t = np.zeros((self.num_regions,), dtype=np.int32)

    def reset(self, rng: np.random.Generator | None = None) -> None:
        if rng is None:
            rng = np.random.default_rng()
        self.bs = rng.integers(low=0, high=(self.A - 1), size=self.N, dtype=np.int64)
        self.cw.fill(self.cw_min)
        self.region_group.fill(-1)
        self.region_t.fill(0)

    def _region_id(self, obs_x: np.ndarray) -> np.ndarray:
        # obs_x layout: pos(3) + q(1) + ...
        # Use a 5x2 grid => 10 regions (paper uses R=10 in one setting).
        x = np.clip(obs_x[:, 0], 0.0, 0.999999)
        y = np.clip(obs_x[:, 1], 0.0, 0.999999)
        gx = np.minimum((x * 5).astype(np.int32), 4)
        gy = np.minimum((y * 2).astype(np.int32), 1)
        rid = (gy * 5 + gx).astype(np.int32)
        rid = np.mod(rid, self.num_regions)
        return rid

    def _two_hop_used_bs(self, adj: np.ndarray, last_actions: np.ndarray, *, no_tx: int) -> list[set[int]]:
        # Directed adj (N,N): edge j->i means j can be sensed by i. We treat in-neighbors as 1-hop.
        used: list[set[int]] = [set() for _ in range(self.N)]
        last_actions = np.asarray(last_actions, dtype=np.int64).reshape(-1)
        for i in range(self.N):
            one_hop = np.where(adj[:, i] > 0)[0].astype(np.int64)
            if one_hop.size == 0:
                continue
            two = set(one_hop.tolist())
            for j in one_hop.tolist():
                two.update(np.where(adj[:, int(j)] > 0)[0].astype(np.int64).tolist())
            for j in two:
                a = int(last_actions[int(j)])
                if 0 <= a < no_tx:
                    used[i].add(a)
        return used

    def _groups(self) -> list[tuple[int, int, int]]:
        # Return groups as (channel, start_slot, end_slot_exclusive)
        groups: list[tuple[int, int, int]] = []
        for ch in range(self.C):
            s = 0
            while s < self.K:
                e = min(self.K, s + self.lsg)
                groups.append((ch, s, e))
                s = e
        return groups

    def select_actions(self, *, env, obs, rng: np.random.Generator, max_tx: int = 1) -> np.ndarray:
        obs_x = np.asarray(obs.x, dtype=np.float32)
        adj = np.asarray(obs.adj, dtype=np.uint8)
        last_status = np.asarray(env.last_status, dtype=np.int64)
        last_actions = np.asarray(env.last_actions, dtype=np.int64)

        n = int(obs_x.shape[0])
        if n != self.N:
            self.N = n
            self.bs = np.full((self.N,), fill_value=(self.A - 1), dtype=np.int64)
            self.cw = np.full((self.N,), fill_value=self.cw_min, dtype=np.int32)

        no_tx = self.A - 1
        l = int(max(1, max_tx))
        actions = np.full((self.N, l), fill_value=no_tx, dtype=np.int64)

        q_norm = obs_x[:, 3]
        has_pkt = q_norm > 0.0
        active = np.where(has_pkt)[0].astype(int)
        if active.size == 0:
            return actions[:, 0] if l == 1 else actions

        # Update region validity.
        dec = self.region_t > 0
        self.region_t[dec] -= 1
        self.region_group[self.region_t <= 0] = -1

        used2 = self._two_hop_used_bs(adj, last_actions, no_tx=no_tx)

        # Basic slot selection (semi-persistent; reselect on collision).
        collided = last_status == self.STATUS_COLLISION
        for i in active:
            if collided[i] or not (0 <= int(self.bs[i]) < no_tx):
                free = [a for a in range(no_tx) if a not in used2[i]]
                if free:
                    self.bs[i] = int(rng.choice(np.asarray(free, dtype=np.int64)))
                else:
                    self.bs[i] = int(rng.integers(low=0, high=no_tx, dtype=np.int64))
            actions[i, 0] = int(self.bs[i])

        if l == 1:
            return actions[:, 0]

        # Slot group usage for bursty traffic.
        rid = self._region_id(obs_x)
        groups = self._groups()
        num_groups = len(groups)

        # Update CW based on collisions (CSMA-style).
        self.cw[collided] = np.minimum(self.cw[collided] * 2, self.cw_max)
        self.cw[last_status == self.STATUS_SUCCESS] = self.cw_min

        for i in active:
            if q_norm[i] < self.burst_q_norm:
                continue
            r = int(rid[i])
            if self.region_group[r] < 0:
                # Bind region to a group (dynamic mapping simplified).
                self.region_group[r] = int(r % max(1, num_groups))
                self.region_t[r] = int(self.tvalid)
            g = int(self.region_group[r])
            ch, s0, s1 = groups[g]

            # Determine free slots within the group based on 2-hop BS occupancy.
            used = used2[i]
            free_slots = []
            for sl in range(s0, s1):
                a = int(ch * self.K + sl)
                if a not in used:
                    free_slots.append(sl)

            # Require at least Lmin consecutive free slots to form a contention window.
            cw_slots: list[int] = []
            run: list[int] = []
            for sl in free_slots:
                if not run or sl == run[-1] + 1:
                    run.append(sl)
                else:
                    if len(run) >= self.lmin:
                        cw_slots = run
                        break
                    run = [sl]
            if not cw_slots and len(run) >= self.lmin:
                cw_slots = run
            if not cw_slots:
                continue

            # Use the middle part of the consecutive free slots as contention window (paper uses middle slots as CW).
            # Here we approximate by taking up to lmin slots centered.
            m = int(len(cw_slots))
            take = int(min(m, self.lmin))
            start = max(0, (m - take) // 2)
            cw_win = cw_slots[start : start + take]
            if not cw_win:
                continue

            # CSMA/CA approximation: choose an offset based on current CW.
            backoff = int(rng.integers(low=0, high=int(self.cw[i]), dtype=np.int64))
            sl = int(cw_win[backoff % len(cw_win)])
            actions[i, 1] = int(ch * self.K + sl)

        return actions

