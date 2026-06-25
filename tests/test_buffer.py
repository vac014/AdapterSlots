"""
test_buffer.py -- Unit tests for AlignmentBuffer (alignment_buffer.md §3.3).

All tests run without GPU or vLLM -- pure Python.

Coverage:
    - enqueue / form_batch basics
    - Dispatch condition A: full warp triggers dispatch
    - Dispatch condition B: T_max expiry triggers partial dispatch
    - Budget enforcement via max_tokens
    - Multiple adapter alignment
    - Tokens from non-dispatched adapters remain in queue (deferred invariant)
    - stats() and pending_count() helpers
    - register_adapter() at runtime
    - TTFT SLO caps the effective deadline
"""

import time
import pytest

from adapter_slots.buffer import AlignmentBuffer


# Fixture helpers

def make_buffer(adapters=None, warp_size=4, tmax_ms=100.0, ttft_slo_ms=200.0):
    """Create a test buffer with a small warp size (W=4) for easier assertions."""
    if adapters is None:
        adapters = ["A", "B"]
    return AlignmentBuffer(adapters, warp_size=warp_size,
                           tmax_ms=tmax_ms, ttft_slo_ms=ttft_slo_ms)


# Basic enqueue / form_batch

def test_empty_buffer_returns_empty_batch():
    buf = make_buffer()
    assert buf.form_batch() == []


def test_enqueue_increments_pending_count():
    buf = make_buffer(adapters=["A"])
    buf.enqueue("A", seq_id=0)
    buf.enqueue("A", seq_id=1)
    assert buf.pending_count() == {"A": 2}


def test_form_batch_below_warp_threshold_no_dispatch():
    """Fewer than W tokens -- should NOT dispatch (no timeout, below threshold)."""
    buf = make_buffer(warp_size=4, tmax_ms=1000.0)
    buf.enqueue("A", seq_id=0)
    buf.enqueue("A", seq_id=1)
    batch = buf.form_batch()
    assert batch == [], "Should not dispatch until warp is full or T_max fires"
    assert buf.pending_count()["A"] == 2, "Tokens should remain in queue"


def test_form_batch_full_warp_dispatches():
    """Exactly W tokens for adapter A → dispatch one warp."""
    buf = make_buffer(warp_size=4, tmax_ms=1000.0)
    for i in range(4):
        buf.enqueue("A", seq_id=i)
    batch = buf.form_batch()
    assert len(batch) == 4
    assert all(adapter == "A" for adapter, _ in batch)
    # Queue should be empty after dispatch
    assert buf.pending_count()["A"] == 0


def test_dispatch_returns_tokens_in_fifo_order():
    """Tokens for the same adapter must come out in enqueue order."""
    buf = make_buffer(warp_size=4, tmax_ms=1000.0)
    seq_ids = [10, 20, 30, 40]
    for s in seq_ids:
        buf.enqueue("A", seq_id=s)
    batch = buf.form_batch()
    dispatched_ids = [seq_id for _, seq_id in batch]
    assert dispatched_ids == seq_ids


def test_dispatch_condition_A_multiple_warps():
    """2×W tokens → two warps dispatched in one tick."""
    buf = make_buffer(warp_size=4, tmax_ms=1000.0)
    for i in range(8):
        buf.enqueue("A", seq_id=i)
    batch = buf.form_batch()
    assert len(batch) == 8
    assert buf.pending_count()["A"] == 0


def test_two_adapters_aligned_separately():
    """W tokens for A and W tokens for B -- both warps dispatched, tokens not mixed."""
    buf = make_buffer(adapters=["A", "B"], warp_size=4, tmax_ms=1000.0)
    for i in range(4):
        buf.enqueue("A", seq_id=i)
    for i in range(4, 8):
        buf.enqueue("B", seq_id=i)
    batch = buf.form_batch()
    assert len(batch) == 8
    # First 4 all from same adapter, last 4 all from same adapter (contiguous)
    adapters_in_batch = [a for a, _ in batch]
    a_block = adapters_in_batch[:4]
    b_block = adapters_in_batch[4:]
    assert len(set(a_block)) == 1
    assert len(set(b_block)) == 1
    assert set(a_block) != set(b_block)


# Dispatch condition B: T_max timeout

def test_tmax_timeout_flushes_partial_warp():
    """Set T_max very small, enqueue fewer than W tokens, wait, expect dispatch."""
    buf = make_buffer(warp_size=4, tmax_ms=1.0)  # 1 ms timeout
    buf.enqueue("A", seq_id=99)
    time.sleep(0.005)  # sleep 5 ms > T_max=1 ms
    batch = buf.form_batch()
    assert len(batch) == 1
    assert batch[0] == ("A", 99)


def test_tmax_timeout_does_not_fire_too_early():
    """Tokens enqueued just now -- T_max has NOT expired yet."""
    buf = make_buffer(warp_size=4, tmax_ms=500.0)  # 500 ms -- won't fire in test
    buf.enqueue("A", seq_id=1)
    buf.enqueue("A", seq_id=2)
    batch = buf.form_batch()
    assert batch == []


def test_ttft_slo_overrides_tmax():
    """TTFT SLO < T_max: effective deadline is TTFT SLO, not T_max."""
    # T_max=100ms but TTFT_SLO=1ms → effective deadline is 1ms
    buf = make_buffer(warp_size=4, tmax_ms=100.0, ttft_slo_ms=1.0)
    buf.enqueue("A", seq_id=5)
    time.sleep(0.005)  # 5 ms > TTFT_SLO=1 ms
    batch = buf.form_batch()
    assert len(batch) == 1, "TTFT SLO should have triggered dispatch"


# Budget enforcement

def test_max_tokens_budget_limits_dispatch():
    """max_tokens=4 with 8 tokens ready → only one warp returned."""
    buf = make_buffer(warp_size=4, tmax_ms=1000.0)
    for i in range(8):
        buf.enqueue("A", seq_id=i)
    batch = buf.form_batch(max_tokens=4)
    assert len(batch) == 4
    # Remaining 4 should still be in queue
    assert buf.pending_count()["A"] == 4


def test_max_tokens_zero_returns_empty():
    buf = make_buffer(warp_size=4, tmax_ms=1000.0)
    for i in range(4):
        buf.enqueue("A", seq_id=i)
    batch = buf.form_batch(max_tokens=0)
    assert batch == []
    assert buf.pending_count()["A"] == 4


# Deferred invariant

def test_deferred_tokens_remain_for_next_tick():
    """Tokens for adapter B (< W) should remain pending when budget is tight."""
    buf = make_buffer(adapters=["A", "B"], warp_size=4, tmax_ms=1000.0)
    for i in range(4):
        buf.enqueue("A", seq_id=i)
    buf.enqueue("B", seq_id=100)  # Only 1 token -- below warp threshold

    batch = buf.form_batch(max_tokens=4)  # Budget for one warp
    dispatched_adapters = [a for a, _ in batch]
    assert "A" in dispatched_adapters
    # B's token should still be pending
    assert buf.pending_count()["B"] == 1


def test_deferred_tokens_dispatched_on_timeout():
    """Deferred tokens eventually dispatched when T_max fires."""
    buf = make_buffer(adapters=["A", "B"], warp_size=4, tmax_ms=2.0)
    buf.enqueue("B", seq_id=77)  # Single token, will defer until timeout

    # First tick: no dispatch (not enough tokens, no timeout yet)
    batch1 = buf.form_batch()
    assert batch1 == []
    assert buf.pending_count()["B"] == 1

    # Wait for T_max
    time.sleep(0.01)  # 10 ms > 2 ms T_max

    # Second tick: timeout fires, token dispatched
    batch2 = buf.form_batch()
    assert len(batch2) == 1
    assert batch2[0] == ("B", 77)
    assert buf.pending_count()["B"] == 0


# Stats and helpers

def test_stats_count_enqueued_dispatched():
    buf = make_buffer(warp_size=4, tmax_ms=1000.0)
    for i in range(4):
        buf.enqueue("A", seq_id=i)
    buf.form_batch()
    s = buf.stats()
    assert s["n_tokens_enqueued"] == 4
    assert s["n_tokens_dispatched"] == 4
    assert s["n_warps_dispatched"] == 1
    assert s["pending_total"] == 0


def test_stats_timeout_dispatch_counted():
    buf = make_buffer(warp_size=4, tmax_ms=1.0)
    buf.enqueue("A", seq_id=0)
    time.sleep(0.005)
    buf.form_batch()
    assert buf.stats()["n_timeout_dispatches"] == 1


def test_reset_stats():
    buf = make_buffer(warp_size=4, tmax_ms=1000.0)
    for i in range(4):
        buf.enqueue("A", seq_id=i)
    buf.form_batch()
    buf.reset_stats()
    s = buf.stats()
    assert s["n_tokens_enqueued"] == 0
    assert s["n_tokens_dispatched"] == 0
    assert s["n_warps_dispatched"] == 0


def test_max_queue_depth():
    buf = make_buffer(adapters=["A", "B"], warp_size=4, tmax_ms=1000.0)
    for i in range(3):
        buf.enqueue("A", seq_id=i)
    buf.enqueue("B", seq_id=99)
    assert buf.max_queue_depth() == 3


def test_oldest_token_age_ms_increases():
    buf = make_buffer(warp_size=4, tmax_ms=1000.0)
    buf.enqueue("A", seq_id=0)
    time.sleep(0.005)
    age = buf.oldest_token_age_ms()
    assert age >= 4.0, f"Expected age >= 4 ms, got {age:.2f} ms"


# Dynamic adapter registration

def test_register_adapter_at_runtime():
    buf = make_buffer(adapters=["A"], warp_size=4, tmax_ms=1000.0)
    buf.register_adapter("C")
    assert "C" in buf.queues
    assert "C" in buf.enqueue_time
    assert buf.pending_count()["C"] == 0


def test_enqueue_auto_registers_unknown_adapter():
    buf = make_buffer(adapters=["A"], warp_size=4, tmax_ms=1000.0)
    buf.enqueue("Z", seq_id=1)
    assert "Z" in buf.queues
    assert buf.pending_count()["Z"] == 1


def test_register_adapter_idempotent():
    buf = make_buffer(adapters=["A"], warp_size=4, tmax_ms=1000.0)
    buf.register_adapter("A")  # Should not raise or reset the queue
    buf.enqueue("A", seq_id=0)
    buf.register_adapter("A")  # Still should not reset
    assert buf.pending_count()["A"] == 1
