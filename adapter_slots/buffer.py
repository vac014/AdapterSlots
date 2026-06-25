"""
buffer.py -- AlignmentBuffer: per-adapter decode-token queues Q_k.

A queue is dispatched when it holds a full warp or its deadline expires:

    |Q_k| >= W   or   age(Q_k) >= min(T_max^(k), T_SLO)

form_batch() uses one global T_max (threshold mode, or the PI controller's
T_max), form_batch_erlang() the per-adapter Erlang deadlines T*(k),
form_batch_cash() the same deadlines in CASH order, and form_batch_wgkp()
the CASH holdback rule with WGKP promotion flags.

Properties:
    - With full-warp dispatch every tick the buffer holds at most K x W tokens.
    - No token waits longer than T_SLO.
    - Each tick is O(K) in the number of adapters.
    - Reordering happens before tensor-parallel sharding, so every TP rank sees
      the same aligned order.
"""

import math
import time
from collections import deque
from typing import Dict, List, Optional, Set, Tuple


class AlignmentBuffer:
    """Per-adapter token queues with full-warp or deadline dispatch.

    Args:
        adapters:      List of adapter IDs served by this buffer.
        warp_size:     GPU warp width (32 for all NVIDIA hardware).
        tmax_ms:       Global deadline T_max in milliseconds for threshold mode.
        ttft_slo_ms:   T_SLO in milliseconds: no token waits longer than this,
                       whatever the deadline.
    """

    def __init__(
        self,
        adapters: List[str],
        warp_size: int = 32,
        tmax_ms: float = 5.0,
        ttft_slo_ms: float = 200.0,
    ) -> None:
        self.W = warp_size
        self.T_max = tmax_ms / 1000.0       # convert ms -> seconds
        self.ttft_slo = ttft_slo_ms / 1000.0
        # Per-adapter queues: each entry is (seq_id, enqueue_time)
        self.queues: Dict[str, deque] = {k: deque() for k in adapters}
        # Shadow queues Q_k' for preempted tokens (seq_ids), and the enqueue
        # time each held token keeps for when it returns to Q_k.
        self._shadow: Dict[str, deque] = {k: deque() for k in adapters}
        self._shadow_time: Dict[int, float] = {}
        # Wall-clock time of the oldest token currently in each queue
        self.enqueue_time: Dict[str, Optional[float]] = {k: None for k in adapters}
        # Wall-clock time of most recent dispatch for each adapter
        self.last_dispatch: Dict[str, float] = {}
        # Cumulative stats (reset on demand)
        self._n_warps_dispatched: int = 0
        self._n_timeout_dispatches: int = 0
        self._n_tokens_enqueued: int = 0
        self._n_tokens_dispatched: int = 0
        # Per-sequence enqueue time (perf_counter seconds), for buffer-wait logging
        self._seq_enqueue: Dict[int, float] = {}

    # Public API

    def register_adapter(self, adapter_id: str) -> None:
        """Register a new adapter at runtime (idempotent)."""
        if adapter_id not in self.queues:
            self.queues[adapter_id] = deque()
            self._shadow[adapter_id] = deque()
            self.enqueue_time[adapter_id] = None

    def enqueue(self, adapter_id: str, seq_id: int) -> None:
        """O(1) enqueue. Called for every arriving decode token.

        Idempotent: if seq_id is already buffered (deferred from a previous tick),
        the existing enqueue time is preserved and no duplicate entry is added.

        Args:
            adapter_id: String identifier of the adapter this token belongs to.
            seq_id:     Sequence/request identifier (opaque; returned in batch).
        """
        if seq_id in self._seq_enqueue:
            return  # already buffered from a previous tick; keep its wait time
        if adapter_id not in self.queues:
            self.register_adapter(adapter_id)
        t = time.perf_counter()
        self.queues[adapter_id].append((seq_id, t))
        if self.enqueue_time[adapter_id] is None:
            self.enqueue_time[adapter_id] = t
        self._n_tokens_enqueued += 1
        self._seq_enqueue[seq_id] = t

    def is_buffered(self, seq_id: int) -> bool:
        """Return True if seq_id has an enqueue record (waiting, or dispatched
        and not yet released)."""
        return seq_id in self._seq_enqueue

    def release(self, seq_id: int) -> None:
        """Drop the enqueue record of a dispatched sequence."""
        self._seq_enqueue.pop(seq_id, None)

    def form_batch(
        self,
        max_tokens: Optional[int] = None,
    ) -> List[Tuple[str, int]]:
        """Threshold dispatch with the global T_max. O(K) per call.

        Called once per scheduling iteration. Dispatches whole warps where
        possible and flushes a partial queue once min(T_max, T_SLO) expires.

        Args:
            max_tokens: Hard cap on total tokens returned in this batch.
                        Defaults to KxW (full aligned batch from all adapters).

        Returns:
            List of (adapter_id, seq_id) pairs in dispatch order.
            Tokens for the same adapter are contiguous (aligned).

        Invariant:
            Any (adapter_id, seq_id) returned is removed from the queue.
            Sequences NOT returned remain in their queue for the next tick.
        """
        batch: List[Tuple[str, int]] = []
        budget = max_tokens if max_tokens is not None else (self.W * len(self.queues))
        t_now = time.perf_counter()
        effective_deadline = min(self.T_max, self.ttft_slo)

        for adapter_id, q in self.queues.items():
            if len(batch) >= budget:
                break
            if len(q) == 0:
                continue

            enq_t = self.enqueue_time[adapter_id]
            age = (t_now - enq_t) if enq_t is not None else 0.0

            if len(q) >= self.W:
                # Full warps only, to preserve alignment.
                n_full_warps = min(len(q) // self.W, (budget - len(batch)) // self.W)
                n_to_dispatch = n_full_warps * self.W
                if n_to_dispatch <= 0:
                    break
                chunk = [q.popleft() for _ in range(n_to_dispatch)]
                batch.extend((adapter_id, seq_id) for seq_id, _ in chunk)
                self.last_dispatch[adapter_id] = t_now
                self.enqueue_time[adapter_id] = q[0][1] if q else None
                self._n_warps_dispatched += n_full_warps
                self._n_tokens_dispatched += n_to_dispatch

            elif age >= effective_deadline:
                # Deadline expired: flush the partial queue.
                n_to_dispatch = min(len(q), budget - len(batch))
                chunk = [q.popleft() for _ in range(n_to_dispatch)]
                batch.extend((adapter_id, seq_id) for seq_id, _ in chunk)
                self.last_dispatch[adapter_id] = t_now
                self.enqueue_time[adapter_id] = q[0][1] if q else None
                self._n_timeout_dispatches += 1
                self._n_tokens_dispatched += n_to_dispatch

        return batch

    def form_batch_cash(
        self,
        ranked_adapters: List[str],
        tmax_k: Dict[str, float],
        max_tokens: Optional[int] = None,
    ) -> List[Tuple[str, int]]:
        """Erlang deadlines with adapters tried in CASH order.

        Identical to form_batch_erlang() except that adapters are visited in
        decreasing CASH score rather than insertion order.

        Args:
            ranked_adapters: Adapter IDs sorted by CASH score (highest first).
            tmax_k:          {adapter_id: T*(k) in seconds}.
            max_tokens:      Hard cap on total tokens returned.
        """
        batch: List[Tuple[str, int]] = []
        budget = max_tokens if max_tokens is not None else (self.W * len(self.queues))
        t_now = time.perf_counter()

        for adapter_id in ranked_adapters:
            if len(batch) >= budget:
                break
            q = self.queues.get(adapter_id)
            if q is None or len(q) == 0:
                continue

            enq_t = self.enqueue_time.get(adapter_id)
            age = (t_now - enq_t) if enq_t is not None else 0.0

            if len(q) >= self.W:
                n_full_warps = min(len(q) // self.W, (budget - len(batch)) // self.W)
                n_to_dispatch = n_full_warps * self.W
                if n_to_dispatch <= 0:
                    break
                chunk = [q.popleft() for _ in range(n_to_dispatch)]
                batch.extend((adapter_id, seq_id) for seq_id, _ in chunk)
                self.last_dispatch[adapter_id] = t_now
                self.enqueue_time[adapter_id] = q[0][1] if q else None
                self._n_warps_dispatched += n_full_warps
                self._n_tokens_dispatched += n_to_dispatch
            else:
                per_adapter_tmax = tmax_k.get(adapter_id, self.ttft_slo)
                effective_deadline = min(per_adapter_tmax, self.ttft_slo)
                if age >= effective_deadline:
                    n_to_dispatch = min(len(q), budget - len(batch))
                    chunk = [q.popleft() for _ in range(n_to_dispatch)]
                    batch.extend((adapter_id, seq_id) for seq_id, _ in chunk)
                    self.last_dispatch[adapter_id] = t_now
                    self.enqueue_time[adapter_id] = q[0][1] if q else None
                    self._n_timeout_dispatches += 1
                    self._n_tokens_dispatched += n_to_dispatch

        return batch

    def form_batch_wgkp(
        self,
        ranked_adapters: List[str],
        tmax_k: Dict[str, float],
        n_star: int,
        merged_adapter_ids: Set[str],
        max_tokens: Optional[int] = None,
    ) -> List[Tuple[str, int, bool]]:
        """CASH holdback dispatch with WGKP promotion flags.

        Each queue, visited in CASH order, is handled by the first rule that
        applies:
            1. |Q_k| >= W:                     dispatch whole warps
            2. |Q_k| >= n*:                    dispatch the queue
            3. |Q_k| >= ceil(n*/2) and
               age > deadline / 2:             dispatch the queue (early release)
            4. age >= deadline:                dispatch the queue (T_SLO cap)
            5. otherwise:                      hold for later ticks
        where deadline = min(T*(k), T_SLO). Rules 1 and 2 mark the segment
        promoted (L3-eligible) when the adapter is resident in the MWC; rules
        3 and 4 never promote. All tokens dispatched from one queue share the
        same flag.

        Args:
            ranked_adapters:    Adapter IDs in CASH order (highest score first).
            tmax_k:             {adapter_id: T*(k) in seconds}.
            n_star:             Promotion threshold n* (tokens per segment).
            merged_adapter_ids: Adapters with merged weights resident in the MWC.
            max_tokens:         Hard cap on total tokens returned.

        Returns:
            List of (adapter_id, seq_id, is_promoted) triples. All tokens from
            the same adapter run share the same is_promoted value (atomicity).
        """
        batch: List[Tuple[str, int, bool]] = []
        budget = max_tokens if max_tokens is not None else (self.W * len(self.queues))
        t_now = time.perf_counter()

        # The dispatch threshold is capped at one adapter's share of a warp,
        # W / K. At larger K the same arrivals split thinner per adapter and a
        # fixed n* would rarely be reached; for K <= W / n* (K <= 4 at n* = 8)
        # the cap is inactive.
        K = max(len(ranked_adapters), 1)
        eff_n_star = max(1, min(n_star, round(self.W / K)))
        eff_half_n_star = math.ceil(eff_n_star / 2)

        for adapter_id in ranked_adapters:
            if len(batch) >= budget:
                break
            q = self.queues.get(adapter_id)
            if q is None or len(q) == 0:
                continue

            enq_t = self.enqueue_time.get(adapter_id)
            age = (t_now - enq_t) if enq_t is not None else 0.0
            per_adapter_tmax = tmax_k.get(adapter_id, self.T_max)
            effective_deadline = min(per_adapter_tmax, self.ttft_slo)
            q_len = len(q)

            dispatch_n: int = 0
            is_promoted: bool = False

            if q_len >= self.W:
                # Rule 1: full warps.
                n_full_warps = min(q_len // self.W, (budget - len(batch)) // self.W)
                dispatch_n = n_full_warps * self.W
                is_promoted = (adapter_id in merged_adapter_ids) and dispatch_n > 0

            elif q_len >= eff_n_star:
                # Rule 2: the segment reaches n*. MWC residency decides only the
                # promotion flag, not whether the segment is dispatched.
                dispatch_n = min(q_len, budget - len(batch))
                is_promoted = (adapter_id in merged_adapter_ids) and dispatch_n > 0

            elif q_len >= eff_half_n_star and age > effective_deadline / 2.0:
                # Rule 3: half of n* and half the deadline elapsed.
                dispatch_n = min(q_len, budget - len(batch))
                is_promoted = False

            elif age >= effective_deadline:
                # Rule 4: deadline expired.
                dispatch_n = min(q_len, budget - len(batch))
                is_promoted = False
                self._n_timeout_dispatches += 1

            # Rule 5: hold.

            if dispatch_n <= 0:
                continue

            chunk = [q.popleft() for _ in range(dispatch_n)]
            batch.extend((adapter_id, seq_id, is_promoted) for seq_id, _ in chunk)
            self.last_dispatch[adapter_id] = t_now
            self.enqueue_time[adapter_id] = q[0][1] if q else None
            if is_promoted:
                self._n_warps_dispatched += dispatch_n // max(self.W, 1)
            self._n_tokens_dispatched += dispatch_n

        return batch

    # Preemption safety: shadow queues

    def preempt_and_hold(self, adapter_id: str, seq_id: int) -> None:
        """Move a preempted token from Q_k to the shadow queue Q_k'.

        Dropping the token would discard the alignment progress Q_k has
        accumulated (with independent preemptions at rate p, a full warp
        survives with probability (1 - p)^W). Held tokens keep their enqueue
        time and return to the head of Q_k on resumption.

        Idempotent: re-calling for a seq_id already in Q_k' is a no-op.
        """
        q = self.queues.get(adapter_id)
        if q is None:
            return
        shadow = self._shadow.setdefault(adapter_id, deque())
        # Check if seq_id is already in shadow (idempotent).
        if seq_id in {s for s in shadow}:
            return
        new_q: deque = deque()
        removed = False
        for entry in q:
            s, t = entry
            if s == seq_id and not removed:
                shadow.append(seq_id)
                self._shadow_time[seq_id] = t
                removed = True
                self._seq_enqueue.pop(seq_id, None)
            else:
                new_q.append(entry)
        self.queues[adapter_id] = new_q
        if new_q:
            self.enqueue_time[adapter_id] = new_q[0][1]
        else:
            self.enqueue_time[adapter_id] = None

    def resume_from_shadow(self, adapter_id: str, seq_id: int) -> None:
        """Return a held token from Q_k' to the head of Q_k.

        The token keeps its original enqueue time, so the time it already
        waited still counts toward the queue's deadline.
        """
        shadow = self._shadow.get(adapter_id)
        if shadow is None or seq_id not in shadow:
            return
        shadow.remove(seq_id)
        t = self._shadow_time.pop(seq_id, time.perf_counter())
        self.register_adapter(adapter_id)
        self.queues[adapter_id].appendleft((seq_id, t))
        oldest = self.enqueue_time[adapter_id]
        self.enqueue_time[adapter_id] = t if oldest is None else min(oldest, t)
        self._seq_enqueue[seq_id] = t

    def shadow_count(self) -> Dict[str, int]:
        """Return number of tokens currently held in each shadow queue."""
        return {k: len(v) for k, v in self._shadow.items()}

    def pending_count(self) -> Dict[str, int]:
        """Return number of tokens pending in each adapter queue."""
        return {k: len(q) for k, q in self.queues.items()}

    def max_queue_depth(self) -> int:
        """Return the maximum queue depth across all adapters."""
        if not self.queues:
            return 0
        return max(len(q) for q in self.queues.values())

    def oldest_token_age_ms(self) -> float:
        """Return the age in milliseconds of the oldest buffered token."""
        t_now = time.perf_counter()
        ages = [
            (t_now - t) * 1000.0
            for t in self.enqueue_time.values()
            if t is not None
        ]
        return max(ages) if ages else 0.0

    def stats(self) -> dict:
        """Return cumulative dispatch statistics."""
        return {
            "n_warps_dispatched": self._n_warps_dispatched,
            "n_timeout_dispatches": self._n_timeout_dispatches,
            "n_tokens_enqueued": self._n_tokens_enqueued,
            "n_tokens_dispatched": self._n_tokens_dispatched,
            "pending_total": sum(len(q) for q in self.queues.values()),
            "max_queue_depth": self.max_queue_depth(),
        }

    def reset_stats(self) -> None:
        """Reset cumulative stats counters."""
        self._n_warps_dispatched = 0
        self._n_timeout_dispatches = 0
        self._n_tokens_enqueued = 0
        self._n_tokens_dispatched = 0

    def form_batch_erlang(
        self,
        tmax_k: Dict[str, float],
        max_tokens: Optional[int] = None,
    ) -> List[Tuple[str, int]]:
        """Dispatch with the per-adapter Erlang deadlines T*(k) (Theorem 1).

        Identical to form_batch() except that a partial queue is flushed once
        age >= min(T*(k), T_SLO). A fast adapter (large lambda_k, small T*(k))
        flushes partial warps sooner; a slow adapter waits longer for a full
        warp, up to T_SLO.

        Args:
            tmax_k:     {adapter_id: T*(k) in seconds}. Adapters missing from
                        this dict use T_SLO.
            max_tokens: Hard cap on total tokens returned. Defaults to KxW.

        Returns:
            List of (adapter_id, seq_id) pairs in dispatch order.
            Tokens for the same adapter are contiguous (aligned).

        Invariant:
            Any (adapter_id, seq_id) returned is removed from the queue.
            Sequences NOT returned remain for the next tick.
        """
        batch: List[Tuple[str, int]] = []
        budget = max_tokens if max_tokens is not None else (self.W * len(self.queues))
        t_now = time.perf_counter()

        for adapter_id, q in self.queues.items():
            if len(batch) >= budget:
                break
            if len(q) == 0:
                continue

            enq_t = self.enqueue_time[adapter_id]
            age = (t_now - enq_t) if enq_t is not None else 0.0

            if len(q) >= self.W:
                n_full_warps = min(len(q) // self.W, (budget - len(batch)) // self.W)
                n_to_dispatch = n_full_warps * self.W
                if n_to_dispatch <= 0:
                    break
                chunk = [q.popleft() for _ in range(n_to_dispatch)]
                batch.extend((adapter_id, seq_id) for seq_id, _ in chunk)
                self.last_dispatch[adapter_id] = t_now
                self.enqueue_time[adapter_id] = q[0][1] if q else None
                self._n_warps_dispatched += n_full_warps
                self._n_tokens_dispatched += n_to_dispatch

            else:
                # Deadline T*(k), capped at T_SLO.
                per_adapter_tmax = tmax_k.get(adapter_id, self.ttft_slo)
                effective_deadline = min(per_adapter_tmax, self.ttft_slo)

                if age >= effective_deadline:
                    n_to_dispatch = min(len(q), budget - len(batch))
                    chunk = [q.popleft() for _ in range(n_to_dispatch)]
                    batch.extend((adapter_id, seq_id) for seq_id, _ in chunk)
                    self.last_dispatch[adapter_id] = t_now
                    self.enqueue_time[adapter_id] = q[0][1] if q else None
                    self._n_timeout_dispatches += 1
                    self._n_tokens_dispatched += n_to_dispatch

        return batch
