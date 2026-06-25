"""
test_deferred.py -- Unit tests for the deferred-sequence invariant.

The critical invariant (alignment_buffer.md §4.2):
    "Deferred sequences must be re-scheduled in the very next tick, not lost."

These tests verify:
1. Sequences deferred in one tick are available in the next tick.
2. Over N ticks with budget constraints, every enqueued seq_id is eventually
   returned exactly once -- no drops, no duplicates.
3. The timeout pathway eventually dispatches all deferred tokens.
4. A large volume test (10 000 tokens) verifies no request is lost.
"""

import time
import pytest

from adapter_slots.buffer import AlignmentBuffer


def run_until_drained(buf: AlignmentBuffer, max_iters: int = 1000,
                      tick_sleep_ms: float = 0.0) -> list:
    """Drive form_batch() ticks until all queues are empty or max_iters reached.

    Returns the flattened list of (adapter_id, seq_id) dispatched across all ticks.
    """
    all_dispatched = []
    for _ in range(max_iters):
        if buf.max_queue_depth() == 0:
            break
        batch = buf.form_batch()
        all_dispatched.extend(batch)
        if tick_sleep_ms > 0:
            time.sleep(tick_sleep_ms / 1000.0)
    return all_dispatched


# Core invariant: no request loss

def test_all_tokens_dispatched_single_adapter():
    """Every enqueued token is eventually dispatched exactly once."""
    buf = AlignmentBuffer(["A"], warp_size=4, tmax_ms=2.0)
    seq_ids = list(range(13))  # 13 tokens: three full warps + 1 partial
    for s in seq_ids:
        buf.enqueue("A", seq_id=s)

    # Drain with ticks + periodic sleep to trigger T_max
    all_dispatched = []
    for _ in range(100):
        batch = buf.form_batch()
        all_dispatched.extend(batch)
        if buf.max_queue_depth() == 0:
            break
        time.sleep(0.003)  # 3 ms > T_max=2 ms → timeout fires for partial warp

    dispatched_seq_ids = [s for _, s in all_dispatched]
    assert sorted(dispatched_seq_ids) == sorted(seq_ids), (
        f"Missing: {set(seq_ids) - set(dispatched_seq_ids)}, "
        f"Duplicates: {[s for s in dispatched_seq_ids if dispatched_seq_ids.count(s) > 1]}"
    )


def test_all_tokens_dispatched_two_adapters():
    """Multi-adapter: no token from either adapter is lost."""
    buf = AlignmentBuffer(["X", "Y"], warp_size=4, tmax_ms=2.0)
    x_ids = list(range(0, 9))    # 9 tokens for X
    y_ids = list(range(100, 107)) # 7 tokens for Y

    for s in x_ids:
        buf.enqueue("X", seq_id=s)
    for s in y_ids:
        buf.enqueue("Y", seq_id=s)

    all_dispatched = []
    for _ in range(200):
        batch = buf.form_batch()
        all_dispatched.extend(batch)
        if buf.max_queue_depth() == 0:
            break
        time.sleep(0.003)

    dispatched_x = [s for a, s in all_dispatched if a == "X"]
    dispatched_y = [s for a, s in all_dispatched if a == "Y"]
    assert sorted(dispatched_x) == sorted(x_ids)
    assert sorted(dispatched_y) == sorted(y_ids)


def test_no_duplicate_dispatches():
    """A seq_id must never appear twice in the dispatched output."""
    buf = AlignmentBuffer(["A", "B"], warp_size=4, tmax_ms=2.0)
    for i in range(10):
        buf.enqueue("A", seq_id=i)
    for i in range(10, 17):
        buf.enqueue("B", seq_id=i)

    all_dispatched = []
    for _ in range(200):
        batch = buf.form_batch()
        all_dispatched.extend(batch)
        if buf.max_queue_depth() == 0:
            break
        time.sleep(0.003)

    seq_ids = [s for _, s in all_dispatched]
    assert len(seq_ids) == len(set(seq_ids)), (
        f"Duplicate dispatches detected: "
        f"{[s for s in seq_ids if seq_ids.count(s) > 1]}"
    )


# Budget-constrained ticks (simulates deferred sequences)

def test_budget_constrained_ticks_no_loss():
    """With max_tokens=W per tick, tokens are deferred and dispatched over multiple ticks."""
    W = 4
    buf = AlignmentBuffer(["A"], warp_size=W, tmax_ms=500.0)
    n_tokens = 20
    for i in range(n_tokens):
        buf.enqueue("A", seq_id=i)

    all_dispatched = []
    for _ in range(100):
        batch = buf.form_batch(max_tokens=W)  # Only one warp per tick
        all_dispatched.extend(batch)
        if buf.max_queue_depth() == 0:
            break

    dispatched_seq_ids = sorted(s for _, s in all_dispatched)
    assert dispatched_seq_ids == list(range(n_tokens))


def test_deferred_dispatched_on_next_tick():
    """Tokens that couldn't be dispatched in tick 1 should appear in tick 2."""
    W = 4
    buf = AlignmentBuffer(["A", "B"], warp_size=W, tmax_ms=1000.0)
    for i in range(4):
        buf.enqueue("A", seq_id=i)
    for i in range(4, 8):
        buf.enqueue("B", seq_id=i)

    # Tick 1: budget=4, only adapter A dispatched
    batch1 = buf.form_batch(max_tokens=4)
    assert len(batch1) == 4

    # B's tokens should still be pending
    assert buf.pending_count()["B"] == 4

    # Tick 2: B's tokens dispatched
    batch2 = buf.form_batch(max_tokens=4)
    assert len(batch2) == 4
    assert all(a == "B" for a, _ in batch2)


# Large volume test

def test_10000_tokens_no_loss():
    """10 000 tokens across 8 adapters -- all must be dispatched exactly once.

    This is the alignment_buffer exit condition §9.1.6: submit 10 000 requests, verify
    completed count matches submitted count.
    """
    K = 8
    W = 4
    N = 10_000
    adapters = [f"adapter_{k}" for k in range(K)]
    buf = AlignmentBuffer(adapters, warp_size=W, tmax_ms=1.0)

    # Enqueue all tokens with round-robin adapter assignment
    for i in range(N):
        adapter = adapters[i % K]
        buf.enqueue(adapter, seq_id=i)

    all_dispatched = []
    for _ in range(50_000):
        batch = buf.form_batch(max_tokens=512)
        all_dispatched.extend(batch)
        if buf.stats()["pending_total"] == 0:
            break
        if buf.oldest_token_age_ms() >= 1.0:
            pass  # let T_max fire naturally in next call

    total = buf.stats()["n_tokens_dispatched"]
    dispatched_ids = [s for _, s in all_dispatched]

    assert len(dispatched_ids) == N, (
        f"Expected {N} dispatched, got {len(dispatched_ids)}"
    )
    assert len(set(dispatched_ids)) == N, "Duplicate seq_ids in dispatch output"
    assert sorted(dispatched_ids) == list(range(N))
