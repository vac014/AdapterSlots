"""
buffer.py -- AlignmentBuffer: per-adapter token accumulator.

This is the alignment_buffer core: fixed global T_max, threshold-based dispatch policy.
Subsequent phases layer on top of this:
    erlang_scheduler: per-adapter Erlang-quantile T_max
    pi_controller: PI controller for T_max
    whittle_scheduler: Whittle index dispatch
    kernel_promotion: WGKP form_batch_wgkp() with CASH holdback policy

Architecture (§3.1 of alignment_buffer.md):

    Incoming requests → AlignmentBuffer (per-adapter queues)
                          → dispatch condition: |Q_k| >= W  OR  age >= T_max
                        → aligned batch → SGMV kernel dispatch

Key invariants:
    - Bounded memory: at most K×W tokens per queue at steady state (Theorem 8.10)
    - No starvation: T_max enforces an upper bound on wait time
    - O(K) form_batch(): linear in number of adapters, not batch size
    - TP-transparent: batch reordering is done before TP sharding
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


class AlignmentBuffer:
    """Core alignment buffer with threshold dispatch policy.

    This is the alignment_buffer version: fixed global T_max, threshold dispatch.
    erlang_scheduler adds per-adapter Erlang T_max.
    pi_controller adds PI controller.
    whittle_scheduler adds Whittle index dispatch.

    Args:
        adapters:      List of adapter IDs served by this buffer.
        warp_size:     GPU warp width (32 for all NVIDIA hardware).
        tmax_ms:       Global timeout in milliseconds before a partial warp is
                       force-dispatched (latency cap).
        ttft_slo_ms:   Hard TTFT SLO -- tokens are never delayed beyond this
                       regardless of T_max setting.
    """

    def __init__(
        self,
        adapters: List[str],
        warp_size: int = 32,
        tmax_ms: float = 5.0,
        ttft_slo_ms: float = 200.0,
    ) -> None:
        self.W = warp_size
        self.T_max = tmax_ms / 1000.0       # convert ms → seconds
        self.ttft_slo = ttft_slo_ms / 1000.0
        # Per-adapter queues: each entry is (seq_id, enqueue_time)
        self.queues: Dict[str, deque] = {k: deque() for k in adapters}
        # Shadow queues for preempt-and-hold (Theorem 8.11): seq_ids only
        self._shadow: Dict[str, deque] = {k: deque() for k in adapters}
        # Wall-clock time of the oldest token currently in each queue
        self.enqueue_time: Dict[str, Optional[float]] = {k: None for k in adapters}
        # Wall-clock time of most recent dispatch for each adapter
        self.last_dispatch: Dict[str, float] = {}
        # Cumulative stats (reset on demand)
        self._n_warps_dispatched: int = 0
        self._n_timeout_dispatches: int = 0
        self._n_tokens_enqueued: int = 0
        self._n_tokens_dispatched: int = 0
        # Per-seq enqueue time for WARτ computation (monotonic, in seconds)
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
            return  # already buffered from a previous tick -- preserve wait time
        if adapter_id not in self.queues:
            self.register_adapter(adapter_id)
        t = time.perf_counter()
        self.queues[adapter_id].append((seq_id, t))
        if self.enqueue_time[adapter_id] is None:
            self.enqueue_time[adapter_id] = t
        self._n_tokens_enqueued += 1
        # Track per-seq enqueue time (perf_counter, seconds) for WARτ computation.
        self._seq_enqueue[seq_id] = t

    def is_buffered(self, seq_id: int) -> bool:
        """Return True if seq_id is currently waiting in the buffer (not yet dispatched)."""
        return seq_id in self._seq_enqueue

    def pop_wartau_ms(self, seq_id: int, t_dispatch: float) -> float:
        """Return WARτ in ms for seq_id and remove its enqueue record.

        WARτ = time from when this request entered the alignment buffer to
        when it was dispatched.  t_dispatch is time.perf_counter() at dispatch.
        Returns 0.0 if seq_id is not tracked (e.g. prefill-only requests).
        """
        t_enq = self._seq_enqueue.pop(seq_id, None)
        if t_enq is None:
            return 0.0
        return max(0.0, (t_dispatch - t_enq) * 1000.0)

    def form_batch(
        self,
        max_tokens: Optional[int] = None,
    ) -> List[Tuple[str, int]]:
        """Main scheduling tick. O(K) per call.

        Called approximately once per vLLM scheduling iteration (~1 kHz).
        Dispatches aligned warp-sized chunks where possible, falling back to
        partial dispatch when T_max or TTFT SLO is exceeded.

        Args:
            max_tokens: Hard cap on total tokens returned in this batch.
                        Defaults to K×W (full aligned batch from all adapters).

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
                # Dispatch condition A: full aligned warp ready.
                # Only dispatch whole warps to preserve alignment.
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
                # Dispatch condition B: T_max expired -- force-flush partial queue.
                n_to_dispatch = min(len(q), budget - len(batch))
                chunk = [q.popleft() for _ in range(n_to_dispatch)]
                batch.extend((adapter_id, seq_id) for seq_id, _ in chunk)
                self.last_dispatch[adapter_id] = t_now
                self.enqueue_time[adapter_id] = q[0][1] if q else None
                self._n_timeout_dispatches += 1
                self._n_tokens_dispatched += n_to_dispatch

        return batch

    def form_batch_whittle(
        self,
        ranked_adapters: List[str],
        tmax_k: Dict[str, float],
        max_tokens: Optional[int] = None,
    ) -> List[Tuple[str, int]]:
        """whittle_scheduler Whittle dispatch: Erlang T_max with Whittle-index adapter ordering.

        Identical to form_batch_erlang() except adapters are tried in
        Whittle-index order (highest index first) rather than insertion order.
        The dispatch conditions are unchanged: full warp OR per-adapter timeout.

        Args:
            ranked_adapters: Adapter IDs sorted by Whittle index (highest first).
            tmax_k:          {adapter_id: T_max^(k)* in seconds}.
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

    # kernel_promotion: WGKP form_batch_wgkp() (Algorithm CASH)

    def form_batch_wgkp(
        self,
        ranked_adapters: List[str],
        tmax_k: Dict[str, float],
        n_star: int,
        merged_adapter_ids: Set[str],
        max_tokens: Optional[int] = None,
    ) -> List[Tuple[str, int, bool]]:
        """kernel_promotion WGKP dispatch: Algorithm CASH with promotion eligibility.

        Implements the five-condition CASH holdback policy from kernel_promotion §2.7:
            1. |Q_k| >= W:              dispatch aligned, is_promoted=True  (full warp)
            2. |Q_k| >= n* AND in MWC:  dispatch promoted, is_promoted=True  (n* threshold)
            3. |Q_k| >= ceil(n*/2)      dispatch partial, is_promoted=False  (half-fill early release)
               AND age > T_max/2:
            4. age >= T_max(k):         dispatch, is_promoted=False  (TTFT hard cap)
            5. else: hold                                             (speculate)

        Promotion atomicity invariant: all tokens dispatched from a single
        adapter queue share the same is_promoted value (True or False).

        Args:
            ranked_adapters:    Adapter IDs in Whittle-index order (highest first).
            tmax_k:             {adapter_id: T_max^(k)* in seconds}.
            n_star:             Promotion threshold (tokens per segment).
            merged_adapter_ids: Set of adapter IDs with valid merged weights in MWC.
                                Only adapters in this set can be promoted.
            max_tokens:         Hard cap on total tokens returned.

        Returns:
            List of (adapter_id, seq_id, is_promoted) triples. All tokens from
            the same adapter run share the same is_promoted value (atomicity).
        """
        batch: List[Tuple[str, int, bool]] = []
        budget = max_tokens if max_tokens is not None else (self.W * len(self.queues))
        t_now = time.perf_counter()

        # K-proportional promotion threshold. n_star (default 8) was tuned
        # against a fixed aggregate arrival burst split across K=4 adapters.
        # At higher K the same burst splits thinner per adapter and a static
        # n_star stops being reachable at all: measured via AS_ADMISSION_LOG
        # diagnostic, K=16 had 0/128 ticks with any n*-triggered dispatch vs
        # 7/159 at K=4 -- CASH silently degenerated into "always force-drain
        # unaligned" at high K. Each adapter's fair share of one full warp
        # (self.W) shrinks as 1/K; capping n_star to that share keeps the
        # threshold reachable as K grows while leaving K <= W/n_star (e.g.
        # K <= 4 at the default n_star=8, W=32) exactly unchanged.
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
                # Condition 1: full warp -- dispatch aligned, promote if in MWC.
                n_full_warps = min(q_len // self.W, (budget - len(batch)) // self.W)
                dispatch_n = n_full_warps * self.W
                is_promoted = (adapter_id in merged_adapter_ids) and dispatch_n > 0

            elif q_len >= eff_n_star:
                # Condition 2: n* threshold (K-scaled). Dispatch must NOT be gated on
                # `adapter_id in merged_adapter_ids`: Level-3 dense-weight-merge
                # promotion is disabled (see model_runner.py), so that set is
                # permanently empty, and gating on it would make this condition dead
                # code for every adapter at every K. is_promoted follows the same
                # pattern as condition 1 -- bookkeeping for WAR/GWAR only, never a
                # dispatch gate.
                dispatch_n = min(q_len, budget - len(batch))
                is_promoted = (adapter_id in merged_adapter_ids) and dispatch_n > 0

            elif q_len >= eff_half_n_star and age > effective_deadline / 2.0:
                # Condition 3: half-fill early release -- dispatch without promotion.
                dispatch_n = min(q_len, budget - len(batch))
                is_promoted = False

            elif age >= effective_deadline:
                # Condition 4: TTFT hard cap -- force-flush without promotion.
                dispatch_n = min(q_len, budget - len(batch))
                is_promoted = False
                self._n_timeout_dispatches += 1

            # else: Condition 5 -- hold (speculate for more tokens).

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

    # multi_gpu_correctness: Preempt-and-Hold (shadow queues, Theorem 8.11)

    def preempt_and_hold(self, adapter_id: str, seq_id: int) -> None:
        """Move a preempted token from Q_k to shadow queue Q_k'.

        Without this, preempting a buffered token destroys its alignment
        contribution -- WAR degrades as (1-p_pre)^W per Theorem 8.11.
        With preempt-and-hold, the token stays in the shadow until resumed,
        so other tokens in Q_k are unaffected.

        Idempotent: re-calling for a seq_id already in shadow is a no-op.
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
                removed = True
                # Also remove from _seq_enqueue so is_buffered() returns False.
                self._seq_enqueue.pop(seq_id, None)
            else:
                new_q.append(entry)
        self.queues[adapter_id] = new_q
        # Recalculate enqueue_time (oldest token in the active queue).
        if new_q:
            self.enqueue_time[adapter_id] = new_q[0][1]
        else:
            self.enqueue_time[adapter_id] = None

    def resume_from_shadow(self, adapter_id: str, seq_id: int) -> None:
        """Re-insert a preempt-and-hold token back into Q_k.

        Called when vLLM resumes a preempted sequence.  The token is placed
        at the back of Q_k (fair re-enqueue -- it waited in shadow, not lost).
        """
        shadow = self._shadow.get(adapter_id)
        if shadow is None or seq_id not in shadow:
            # Not in shadow -- nothing to do.
            return
        shadow.remove(seq_id)
        # Re-enqueue as a fresh token (current time).
        self.enqueue(adapter_id, seq_id)

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

    # erlang_scheduler: per-adapter Erlang T_max dispatch

    def form_batch_erlang(
        self,
        tmax_k: Dict[str, float],
        max_tokens: Optional[int] = None,
    ) -> List[Tuple[str, int]]:
        """erlang_scheduler scheduling tick with per-adapter Erlang T_max^(k)* values.

        Replaces the single global self.T_max with a per-adapter dict tmax_k.
        Called by AlignmentAwareScheduler.schedule_erlang() when AS_MODE=erlang.

        The dispatch conditions are identical to form_batch(), except that
        Dispatch Condition B uses the per-adapter timeout:
            age >= min(tmax_k[adapter_id], self.ttft_slo)

        This allows fast adapters (large λ_k → small T_max^(k)*) to dispatch
        partial warps sooner, while slow adapters wait longer for full warps --
        as prescribed by Theorem 5.3 / Corollary 5.4.

        TP=2 note: tmax_k values are computed by the rank-0 scheduler from
        the EWMA λ_k estimates. The resulting batch is identical regardless
        of TP degree (the per-adapter T_max affects scheduling, not sharding).

        Args:
            tmax_k:     {adapter_id: T_max^(k)* in seconds}. Adapters missing
                        from this dict fall back to self.ttft_slo as the cap.
            max_tokens: Hard cap on total tokens returned. Defaults to K×W.

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
                # Dispatch condition A: full aligned warp ready.
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
                # Dispatch condition B (erlang_scheduler): per-adapter T_max^(k)* timeout.
                # The fairness cap ensures we never exceed the TTFT SLO.
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
