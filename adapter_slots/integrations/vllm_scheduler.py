"""
vllm_scheduler.py -- AlignmentAwareScheduler: vLLM scheduler subclass for AS.

Integration point (alignment_buffer.md §4.1):

    vLLM's scheduling flow:
        Scheduler.schedule()
            → _schedule_running()      ← selects which sequences continue
            → SchedulerOutputs         ← contains scheduled_seq_groups
            → ModelRunner.execute_model() ← SGMV dispatch

    AS inserts between _schedule_running() and ModelRunner.execute_model():
        Scheduler.schedule()
            → super().schedule()                    ← standard vLLM selection
            → alignment_buffer.enqueue(decode_seqs) ← buffer decode tokens
            → alignment_buffer.form_batch()         ← get aligned batch
            → reorder SchedulerOutputs              ← replace with aligned order
            → return to ModelRunner

TP=2 integration (§3.1, §4.2):
    The aligned batch is produced at the scheduler level, BEFORE vLLM's TP
    dispatch layer shards the token sequence to each GPU worker. Both GPU workers
    receive the same aligned token order. WAR is therefore TP-invariant.

Usage (--scheduler-class flag, vLLM 0.6+):
    vllm serve <model> \
        --scheduler-class adapter_slots.integrations.vllm_scheduler.AlignmentAwareScheduler \
        ...

Configuration via environment variables:
    AS_TMAX_MS         -- T_max in milliseconds (default: 5.0, used in "threshold" mode)
    AS_PI_KP           -- PI proportional gain (default: 0.01, used in "pi_adaptive" mode)
    AS_PI_KI           -- PI integral gain (default: 0.001, used in "pi_adaptive" mode)
    AS_PI_TMAX_INIT_MS -- Initial T_max for PI controller in ms (default: 18.0)
    AS_PI_UPDATE_TICKS -- Scheduling ticks per PI update (default: 100 ≈ τ_iter on PCIe)
    AS_TTFT_SLO_MS     -- TTFT SLO in milliseconds (default: 200.0, matches AlignmentBuffer's own default)
    AS_WARP_SIZE       -- warp width (default: 32)
    AS_MODE            -- dispatch mode: "threshold" (alignment_buffer), "erlang" (erlang_scheduler), "pi_adaptive" (pi_controller), or "whittle" (whittle_scheduler)
    AS_WHITTLE_DELTA_T -- Whittle scheduling tick interval in seconds; MUST match τ_iter for target hardware
                         (PCIe ≈ 0.100, NVLink ≈ 0.005, single A6000 ≈ 0.030)
    AS_WAR_TARGET      -- WAR* target for Erlang mode (default: 0.8)
    AS_EWMA_ALPHA      -- EWMA smoothing factor for λ_k estimation (default: 0.1)
    AS_LOG_WAR         -- if "1", emit WAR/queue stats to stderr each tick

    adapter_prefetching -- PredictiveLFU cold-start-aware scheduling:
    AS_PREFETCH_POLICY -- "none" (default) | "lru" | "predictive"
                         "predictive": Poisson-scored PredictiveLFU eviction advisory.
                         Deprioritises cold adapters in the dispatch queue so warm-adapter
                         requests fill warp batches first, hiding cold-load latency behind
                         warm serving.  Requires AS_K_WARM > 0.
    AS_TAU_LOAD_MS     -- Measured adapter cold-start load time (ms). Default: 96.3, the
                         measured A6000 PCIe value.
                         Used as τ_load in score(k) = 1 - exp(-λ̂_k × τ_load).
    AS_K_WARM          -- Warm cache capacity (= --max-loras value). Must be set for
                         AS_PREFETCH_POLICY to activate. Default: 0 (disabled).
    AS_COLD_BOOST      -- T_max multiplier for predicted-cold adapters (default: 2.5).
                         Cold adapters wait longer → warm-adapter requests served first.
                         Capped at TTFT SLO to preserve latency guarantees.

    kernel_promotion -- WGKP + compound stack (AS_MODE=wgkp):
    AS_WGKP_THRESHOLD  -- Promotion threshold n* in tokens. Default: 8.
                         Overridden by APT when AS_WGKP_APT=1.
    AS_WGKP_APT        -- 0 (default) | 1. Enable AdaptivePromoThreshold.
    AS_WGKP_HW_PROFILE -- Path to E13.6 JSON profile. Default: hw_profiles/a6000_tp2.json.
    AS_WGKP_LOG        -- 0 (default) | 1. Per-tick WGKP promotion stats to stderr.
    AS_MWC_K_HOT       -- Max adapters in MergedWeightCache. Default: 5.
                         Set to 0 to disable Level-3 GEMM (Fused kernel only).
    AS_MWC_MEMORY_GB   -- GPU VRAM budget for merged weights. Default: 10.0.
    AS_MWC_PROJECTIONS -- Comma-sep projection names. Default: q_proj,k_proj,v_proj,o_proj.
                         Use "all" for all 7 linear layers.
    AS_FUSED_KERNEL    -- 0 (default) | 1. Enable fused Triton kernel for Level-2 path.
    AS_MACRO_N_ACCUM   -- Iterations to accumulate. Default: 1 (disabled).
                         Set to 3 for K=10 T_max=90ms target with τ_iter=30ms.
    AS_APIS_ENABLED    -- 0 (default) | 1. Enable adapter-partitioned routing.
    AS_APIS_N_GPUS     -- Number of independent GPU shards. Default: 2.
    AS_APIS_UPSTREAM_URLS -- Comma-sep vLLM URLs. e.g. "http://gpu0:8000,http://gpu1:8000"
    AS_APIS_REBALANCE_S -- Seconds between APIS rebalancing. Default: 30.
    AS_ADAPTER_RANK    -- LoRA rank for adapter generation scripts. Default: 16.
"""

import json
import os
import pathlib
import time
import logging
from collections import deque
from typing import List, Optional

logger = logging.getLogger(__name__)

# vLLM import guard
# vLLM is an optional dependency (not installed during unit-test runs).
# This module is only imported inside a live vLLM serving process.
try:
    from vllm.core.scheduler import Scheduler, SchedulerOutputs, SchedulerRunningOutputs
    from vllm.sequence import SequenceGroup, SequenceStatus
    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False
    # Provide stub base class so the module can be imported in test environments.
    class Scheduler:  # type: ignore[no-redef]
        def schedule(self):
            raise NotImplementedError

    class SchedulerOutputs:  # type: ignore[no-redef]
        pass

    class SequenceGroup:  # type: ignore[no-redef]
        pass

    class SequenceStatus:  # type: ignore[no-redef]
        RUNNING = "RUNNING"

from adapter_slots.buffer import AlignmentBuffer
from adapter_slots.metrics.war import compute_war_from_ids
from adapter_slots.dispatch.erlang import compute_tmax_erlang_batch
from adapter_slots.dispatch.whittle import WhittleDispatcher
from adapter_slots.control.estimator import ArrivalRateEstimator

# adapter_prefetching: PredictiveLFU -- imported lazily in __init__ when policy is active
_WarmCacheManager = None  # type: ignore[assignment]


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


class AlignmentAwareScheduler(Scheduler):
    """vLLM scheduler subclass that routes decode tokens through AlignmentBuffer.

    On each scheduling tick:
    1. Call super().schedule() to get vLLM's standard token selection.
    2. Extract all decode-phase sequence groups from SchedulerOutputs.
    3. Enqueue their tokens into the AlignmentBuffer (one entry per seq group).
    4. Call form_batch() to get the aligned dispatch order.
    5. Reorder scheduled_seq_groups to match the aligned order.
    6. Sequences not selected by form_batch() are deferred to the next tick.

    Critical invariant: Deferred sequences are re-added to the running queue
    immediately, ensuring they are visible to super().schedule() on the next
    call and are never dropped.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Read configuration from environment variables
        tmax_ms = _env_float("AS_TMAX_MS", 5.0)
        # This default must match AlignmentBuffer's own (buffer.py: ttft_slo_ms=200.0).
        # It is a deadline, not a hint: instrumenting buffer_wait per request
        # (dispatch_time - arrival_time) shows the tail of held requests riding almost
        # exactly to it before force-release (actual_wait ~= T*(k)). A 10x-looser
        # override of 2000.0 therefore does not merely relax a bound, it spends it --
        # it accounts for essentially all of AS's ~2.0-2.1s TTFT gap against vLLM in
        # the K-sweep.
        ttft_slo_ms = _env_float("AS_TTFT_SLO_MS", 200.0)
        warp_size = _env_int("AS_WARP_SIZE", 32)
        self._log_war = os.environ.get("AS_LOG_WAR", "0") == "1"
        self._mode = os.environ.get("AS_MODE", "threshold")

        # erlang_scheduler Erlang mode configuration
        self._war_target = _env_float("AS_WAR_TARGET", 0.8)
        self._ewma_alpha = _env_float("AS_EWMA_ALPHA", 0.1)

        # Hard print: logger.info is silenced until vLLM installs its handlers,
        # which happens after scheduler construction.  This print always reaches
        # stderr (and therefore the per-run server log file).
        import sys as _sys
        # For pi_adaptive the effective buffer T_max is AS_PI_TMAX_INIT_MS, not AS_TMAX_MS.
        # Compute here (before buffer init) so the log is accurate.
        buffer_tmax_ms = (
            _env_float("AS_PI_TMAX_INIT_MS", 18.0)
            if self._mode == "pi_adaptive"
            else tmax_ms
        )
        print(
            f"[AS] AlignmentAwareScheduler.__init__ called: "
            f"mode={self._mode} T_max={buffer_tmax_ms}ms TTFT_SLO={ttft_slo_ms}ms W={warp_size}",
            file=_sys.stderr, flush=True,
        )
        logger.info(
            "AlignmentAwareScheduler: mode=%s T_max=%.1f ms, TTFT_SLO=%.1f ms, "
            "W=%d, WAR*=%.2f, α=%.3f",
            self._mode, buffer_tmax_ms, ttft_slo_ms, warp_size,
            self._war_target, self._ewma_alpha,
        )
        self._schedule_call_count = 0
        self._tick_profiler = None
        if os.environ.get("AS_PROFILE_TICK"):
            import cProfile
            self._tick_profiler = cProfile.Profile()

        # Buffer is initialised with an empty adapter list; adapters are
        # registered lazily as new lora_request names appear in seq groups.
        # buffer_tmax_ms is already computed above (before the log print).
        self._buffer = AlignmentBuffer(
            adapters=[],
            warp_size=warp_size,
            tmax_ms=buffer_tmax_ms,
            ttft_slo_ms=ttft_slo_ms,
        )
        self._tick_times_ms: List[float] = []  # overhead samples for §5.4
        self._ttft_slo_ms = ttft_slo_ms
        self._warp_size = warp_size
        self._batch_id = 0
        self._t0_ms = time.perf_counter() * 1000.0

        # Optional per-batch JSONL output for offline WAR analysis (§2.7.2).
        # Set AS_METRICS_PATH=/path/to/metrics.jsonl to enable.
        metrics_path = os.environ.get("AS_METRICS_PATH", "")
        if metrics_path:
            pathlib.Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
            self._metrics_file = open(metrics_path, "w")  # noqa: WPS515
            logger.info("AlignmentAwareScheduler: writing metrics → %s", metrics_path)
        else:
            self._metrics_file = None

        # erlang_scheduler: EWMA arrival rate estimator (rank-0 scheduler only).
        # enforce_rank0=False because vLLM may set LOCAL_RANK differently
        # depending on version; we rely on the scheduler architecture to ensure
        # only one instance is created in the coordinator process.
        self._estimator = ArrivalRateEstimator(
            alpha=self._ewma_alpha,
            enforce_rank0=False,
        )

        # pi_controller: PI-adaptive mode -- T_max driven by PI controller (Proposition 6.5).
        #
        # vLLM's scheduling tick rate is ~1 kHz (one call per 1 ms decode step).
        # The PI controller is designed for one update per LLM decode ITERATION
        # (τ_iter ≈ 30–100 ms).  Firing every tick gives n_q ≈ 100 for PCIe,
        # which Proposition 6.5 proved UNSTABLE (ρ ≈ 3.49 >> 1).
        #
        # AS_PI_UPDATE_TICKS controls how many scheduling ticks constitute one
        # "iteration boundary" PI update (n_q = 1 in the paper's sense).
        # Default 100 matches τ_iter ≈ 100 ms on Two A6000 PCIe.
        # WAR is accumulated across all ticks in the window and averaged before
        # the single PI update -- identical to IterationBoundaryPIController semantics.
        self._pi_ctrl = None
        self._pi_tick_count = 0
        self._pi_update_interval = 1  # overridden below for pi_adaptive
        if self._mode == "pi_adaptive":
            from adapter_slots.control.pi_controller import (
                PIController, IterationBoundaryPIController,
            )
            kp = _env_float("AS_PI_KP", 0.01)
            ki = _env_float("AS_PI_KI", 0.001)
            tmax_init_s = _env_float("AS_PI_TMAX_INIT_MS", 18.0) / 1000.0
            # Hard cap: T_max must never exceed the TTFT SLO.  Without this the PI
            # default (tmax_max=5.0s) lets the integral wind up far beyond the SLO
            # while the buffer already caps dispatch at min(T_max, ttft_slo=2.0s).
            # Capping at SLO keeps the anti-windup range physically meaningful and
            # prevents the integral from accumulating error against a wall it can
            # never close (key fix for Zipf-load stability on PCIe TP=2).
            tmax_max_s = ttft_slo_ms / 1000.0
            self._pi_update_interval = _env_int("AS_PI_UPDATE_TICKS", 100)
            pi_inner = PIController(
                kp=kp, ki=ki, war_target=self._war_target,
                tmax_init=tmax_init_s,
                tmax_max=tmax_max_s,
            )
            self._pi_ctrl = IterationBoundaryPIController(pi_inner)
            print(
                f"[AS] PI-adaptive: K_p={kp} K_i={ki} "
                f"T_max_init={tmax_init_s*1000:.1f}ms T_max_max={tmax_max_s*1000:.1f}ms "
                f"WAR*={self._war_target} update_every={self._pi_update_interval}_ticks",
                file=_sys.stderr, flush=True,
            )

        # whittle_scheduler: Whittle index dispatcher.
        # Initialised lazily (adapters registered on first request) so __init__
        # doesn't need to know the adapter list up front.
        self._whittle_dispatcher: Optional[WhittleDispatcher] = None
        if self._mode == "whittle":
            delta_t = _env_float("AS_WHITTLE_DELTA_T", 0.030)
            self._whittle_delta_t = delta_t
            print(
                f"[AS] Whittle mode: delta_t={delta_t*1000:.1f}ms "
                f"WAR*={self._war_target} TTFT_SLO={ttft_slo_ms:.0f}ms",
                file=_sys.stderr, flush=True,
            )
        else:
            self._whittle_delta_t = 0.030  # unused placeholder

        # adapter_prefetching: PredictiveLFU cold-start-aware scheduling
        # Integrates with Erlang/Whittle T_max via a cold-boost multiplier:
        # cold adapters (predicted by PredictiveLFU) get T_max boosted so
        # warm-adapter requests accumulate into warp batches first.
        # This pipelines cold-adapter loads behind warm-adapter serving.
        self._prefetch_policy = os.environ.get("AS_PREFETCH_POLICY", "none")
        self._tau_load_ms = _env_float("AS_TAU_LOAD_MS", 96.3)
        self._k_warm = _env_int("AS_K_WARM", 0)
        self._cold_boost = _env_float("AS_COLD_BOOST", 2.5)
        self._cache_mgr = None
        if self._prefetch_policy != "none" and self._k_warm > 0:
            global _WarmCacheManager
            from adapter_slots.prefetch.cache_manager import WarmCacheManager as _WCM
            _WarmCacheManager = _WCM
            pol = self._prefetch_policy if self._prefetch_policy in (
                "lru", "predictive", "topk") else "lru"
            self._cache_mgr = _WCM(
                k_warm_max=self._k_warm,
                tau_load_ms=self._tau_load_ms,
                policy=pol,
            )
            print(
                f"[AS] PredictiveLFU: policy={pol} K_warm={self._k_warm} "
                f"τ_load={self._tau_load_ms:.1f}ms cold_boost={self._cold_boost:.1f}×",
                file=_sys.stderr, flush=True,
            )

        # adapter_prefetching PCIe fix: minimum deferral window for cold adapters.
        # On PCIe hardware τ_load/τ_iter ≈ 1, meaning a cold adapter stalls one
        # full decode iteration if served immediately. S-LoRA §4.3 and CaraServe
        # §4.2 both enforce a hard deferral window of τ_load seconds: the adapter
        # is not eligible for dispatch until t_now - t_first_seen ≥ τ_load, by
        # which time the PCIe DMA has completed. Without this, even cold_boost=2.0
        # can result in dispatching cold adapters mid-load, causing GPU stall.
        # AS_PCIE_MIN_DEFERRAL_S: 0.0 (disabled) or τ_load_s (e.g. 0.0963 on A6000 PCIe).
        self._pcie_min_deferral_s = _env_float("AS_PCIE_MIN_DEFERRAL_S", 0.0)
        # Tracks wall-clock time when each cold adapter was first seen by this scheduler.
        # Cleared when the adapter becomes warm (enters the vLLM GPU LRU).
        self._cold_first_seen: dict = {}

        # Sequences that have received at least one aligned decode token.
        # After the first aligned dispatch (WARτ measured), subsequent tokens
        # bypass the alignment buffer and are dispatched every tick -- otherwise
        # each of 64 decode tokens incurs a fresh T_max wait (192s total for
        # non-dominant adapters vs the 120s timeout, causing all to fail).
        self._post_alignment_seqs: set = set()

        # Buffer-wait instrumentation: dispatch_time - arrival_time per
        # adapter, in ms. arrival_time is AlignmentBuffer._seq_enqueue[seq_id]
        # (set in enqueue()); dispatch_time is captured at the two real exit
        # points from the buffer (CASH-promoted dispatch and force-drain).
        # Uses AlignmentBuffer.pop_wartau_ms(), which existed but was never
        # called anywhere in the codebase before this -- this is what actually
        # answers "how much of the missing throughput is the buffer holding
        # requests, vs. everything else."
        self._buffer_wait_ms: dict = {}  # adapter_id -> List[float]
        self._buffer_wait_log = os.environ.get("AS_BUFFER_WAIT_LOG", "0") == "1"
        self._buffer_wait_dump_path = os.environ.get("AS_BUFFER_WAIT_DUMP")
        self._buffer_wait_sample_count: dict = {}  # adapter_id -> int, caps per-adapter raw log lines
        self._buffer_wait_sample_limit = int(os.environ.get("AS_BUFFER_WAIT_SAMPLE_LIMIT", "20"))

        # kernel_promotion: WGKP components -- initialised lazily when AS_MODE=wgkp.
        self._wgkp_dispatcher = None
        self._mwc = None
        self._apt = None
        # adapter_id (lora_name) -> on-disk adapter dir, set externally by
        # aligned_api_server.run_server() from args.lora_modules right after
        # engine construction (same process, --disable-frontend-multiprocessing).
        self._lora_paths: dict = {}
        self._lora_weights_cache: dict = {}
        self._wgkp_log = os.environ.get("AS_WGKP_LOG", "0") == "1"
        if self._mode == "wgkp":
            self._init_wgkp_components(_sys)

    # Internal helpers

    def _adapter_id_of(self, seq_group: "SequenceGroup") -> str:
        """Extract adapter ID string from a vLLM SequenceGroup.

        vLLM stores the LoRA request as seq_group.lora_request. If the
        sequence group has no LoRA, we use the sentinel "__base__".
        """
        lr = getattr(seq_group, "lora_request", None)
        if lr is None:
            return "__base__"
        # lora_request.lora_name is the adapter ID used in --lora-modules
        return getattr(lr, "lora_name", str(id(lr)))

    def _seq_group_id(self, seq_group: "SequenceGroup") -> int:
        """Return a stable integer ID for a SequenceGroup."""
        seqs = seq_group.get_seqs()
        if seqs:
            return seqs[0].seq_id
        return id(seq_group)

    # Core override

    def _schedule_running(self, budget, curr_loras, enable_chunking=False):
        """Guard against the async-postprocessing race (§13.5).

        AsyncLLMEngine calls schedule() for tick N+1 before tick N's async
        output-processor has called update_num_computed_tokens().  Sequences
        that finished prefill in tick N are therefore still in self.running
        with stage=PREFILL.  When _schedule_prefills() finds nothing new,
        _schedule_running() is called and tries to add each PREFILL seq's
        full prompt token count to the budget → overflow → AssertionError at
        vllm/core/scheduler.py:1035.

        Fix: if any sequence in the running queue is still PREFILL, return an
        empty SchedulerRunningOutputs.  The PREFILL seqs stay untouched in
        self.running; once the async postprocessor fires and transitions them
        to DECODE, the next call processes them normally.  The engine gets an
        empty step and immediately retries -- typically resolves in 1-2 ticks.
        """
        if _VLLM_AVAILABLE and any(sg.is_prefill() for sg in self.running):
            import sys as _sys
            n = sum(1 for sg in self.running if sg.is_prefill())
            print(
                f"[WGKP] _schedule_running: {n} PREFILL seqs in running queue, "
                f"skipping to await async postprocessing",
                file=_sys.stderr, flush=True,
            )
            return SchedulerRunningOutputs.create_empty()
        return super()._schedule_running(budget, curr_loras, enable_chunking)

    def _schedule(self) -> "SchedulerOutputs":
        """Override _schedule() so vLLM builds seq_group_metadata_list from our
        reordered scheduled_seq_groups.  schedule() (which we do NOT override)
        calls _schedule() then iterates over its .scheduled_seq_groups to build
        the metadata list -- lengths therefore always match.

        Dispatches via threshold mode (alignment_buffer) or Erlang mode (erlang_scheduler).
        """
        import sys as _sys
        self._schedule_call_count += 1
        if self._schedule_call_count <= 3:
            print(
                f"[AS] _schedule() tick #{self._schedule_call_count} "
                f"mode={self._mode}",
                file=_sys.stderr, flush=True,
            )
        if self._mode == "wgkp" and self._tick_profiler is not None:
            self._tick_profiler.enable()
            try:
                return self._schedule_wgkp_inner()
            finally:
                self._tick_profiler.disable()
                if self._schedule_call_count % 200 == 0:
                    import pstats
                    path = f"/tmp/as_tick_profile_{os.getpid()}.pstats"
                    self._tick_profiler.dump_stats(path)
                    print(f"[AS_PROFILE_TICK] dumped {self._schedule_call_count} ticks to {path}",
                          file=_sys.stderr, flush=True)
        if self._mode == "pi_adaptive":
            return self._schedule_pi_adaptive_inner()
        if self._mode == "wgkp":
            return self._schedule_wgkp_inner()
        if self._mode == "erlang":
            return self._schedule_erlang_inner()
        if self._mode == "whittle":
            return self._schedule_whittle_inner()
        if self._mode == "sort":
            return self._schedule_sort_inner()
        return self._schedule_threshold_inner()

    # Keep the public schedule() untouched -- vLLM 0.6.3 wraps _schedule().

    def _schedule_threshold_inner(self) -> "SchedulerOutputs":
        """alignment_buffer threshold dispatch: single global T_max for all adapters.

        WARτ is a per-request metric (first-token alignment delay).  After a
        sequence's first token is aligned-dispatched, subsequent decode tokens
        bypass the buffer so they don't each incur a fresh T_max wait -- which
        would make every request from non-dominant adapters take 64×T_max and
        always timeout.
        """
        t_start = time.perf_counter()

        sched_out: "SchedulerOutputs" = super()._schedule()

        if not _VLLM_AVAILABLE:
            return sched_out

        prefill_groups, decode_groups = self._split_groups(sched_out)

        if not decode_groups:
            self._record_overhead(t_start)
            return sched_out

        # Split decode groups: already-aligned sequences bypass the buffer;
        # first-time sequences go through alignment for their first token.
        first_time_groups = []
        post_alignment_groups = []
        for ssg in decode_groups:
            seq_id = self._seq_group_id(ssg.seq_group)
            if seq_id in self._post_alignment_seqs:
                post_alignment_groups.append(ssg)
            else:
                first_time_groups.append(ssg)

        # Alignment buffer only for first-token sequences (WARτ measured here).
        seq_id_to_sg = self._enqueue_decode_groups(first_time_groups)
        aligned_pairs = self._buffer.form_batch(max_tokens=len(decode_groups))

        aligned_first_time = []
        for adapter_id, seq_id in aligned_pairs:
            ssg = seq_id_to_sg.get(seq_id)
            if ssg is not None:
                aligned_first_time.append(ssg)
                self._post_alignment_seqs.add(seq_id)

        # post_alignment always dispatched; first_time only when aligned.
        # Re-sort post_alignment_groups by adapter so the full decode batch sent
        # to the GPU is adapter-contiguous.  This is what gives sustained high WAR
        # across all decode steps, not just the first token of each request.
        # T_max=0 (baseline) dispatches first-time tokens immediately one-by-one
        # in arrival order, so those remain unsorted -- preserving the contrast.
        post_alignment_groups.sort(key=lambda s: self._adapter_id_of(s.seq_group))
        sched_out.scheduled_seq_groups = (
            prefill_groups + post_alignment_groups + aligned_first_time
        )

        if self._log_war and decode_groups:
            self._log_war_stats(aligned_first_time, first_time_groups, aligned_pairs,
                                all_decode=post_alignment_groups + aligned_first_time)

        self._record_overhead(t_start)
        return sched_out

    def _schedule_pi_adaptive_inner(self) -> "SchedulerOutputs":
        """pi_controller PI-adaptive dispatch: global T_max adjusted by PI controller.

        Alignment logic is identical to threshold mode.  WAR is accumulated every tick;
        trigger_iteration_end() fires once per AS_PI_UPDATE_TICKS ticks (default 100),
        matching τ_iter ≈ 100 ms on PCIe.  This keeps n_q = 1 per the paper's definition,
        making K_p/K_i hardware-independent and avoiding the PCIe integral-windup
        instability shown in Proposition 6.5 (ρ ≈ 3.49 for n_q=100 without this guard).
        """
        t_start = time.perf_counter()

        sched_out: "SchedulerOutputs" = super()._schedule()

        if not _VLLM_AVAILABLE:
            return sched_out

        prefill_groups, decode_groups = self._split_groups(sched_out)

        if not decode_groups:
            self._record_overhead(t_start)
            return sched_out

        first_time_groups = []
        post_alignment_groups = []
        for ssg in decode_groups:
            seq_id = self._seq_group_id(ssg.seq_group)
            if seq_id in self._post_alignment_seqs:
                post_alignment_groups.append(ssg)
            else:
                first_time_groups.append(ssg)

        seq_id_to_sg = self._enqueue_decode_groups(first_time_groups)
        aligned_pairs = self._buffer.form_batch(max_tokens=len(decode_groups))

        aligned_first_time = []
        for adapter_id, seq_id in aligned_pairs:
            ssg = seq_id_to_sg.get(seq_id)
            if ssg is not None:
                aligned_first_time.append(ssg)
                self._post_alignment_seqs.add(seq_id)

        post_alignment_groups.sort(key=lambda s: self._adapter_id_of(s.seq_group))
        sched_out.scheduled_seq_groups = (
            prefill_groups + post_alignment_groups + aligned_first_time
        )

        # Observe WAR from the full decode batch and accumulate into PI controller.
        # trigger_iteration_end() is called only every _pi_update_interval ticks so
        # that n_q = 1 in the paper's sense (one PI update per τ_iter, not per 1ms tick).
        all_decode = post_alignment_groups + aligned_first_time
        if all_decode:
            adapter_ids = [self._adapter_id_of(ssg.seq_group) for ssg in all_decode]
            int_ids = [self._adapter_id_to_int(aid) for aid in adapter_ids]
            war_obs = compute_war_from_ids(int_ids, warp_size=self._buffer.W)
            self._pi_ctrl.record_batch_war(war_obs)

        self._pi_tick_count += 1
        if self._pi_tick_count >= self._pi_update_interval:
            new_tmax_s = self._pi_ctrl.trigger_iteration_end()
            # Defensive clamp: PIController already clamps to tmax_max=SLO, but
            # guard against any numerical edge case pushing T_max above the SLO.
            new_tmax_s = max(0.001, min(new_tmax_s, self._ttft_slo_ms / 1000.0))
            self._buffer.T_max = new_tmax_s
            self._pi_tick_count = 0

        # Periodic cleanup: remove completed sequence IDs from _post_alignment_seqs.
        # Without this the set grows monotonically (~45 entries/second at 7 req/s),
        # and set membership checks degrade over a long run.  Every 500 PI iterations
        # (~500 × 100ms = 50s) we intersect against still-running seq_ids.
        if self._pi_tick_count == 0 and self._schedule_call_count % (self._pi_update_interval * 500) == 0:
            try:
                active_seq_ids: set = set()
                for sg in self.running:
                    for seq in sg.get_seqs():
                        active_seq_ids.add(seq.seq_id)
                self._post_alignment_seqs &= active_seq_ids
            except Exception:
                pass  # never let cleanup crash the scheduler

        if self._log_war and decode_groups:
            self._log_war_stats(aligned_first_time, first_time_groups, aligned_pairs,
                                all_decode=all_decode)

        self._record_overhead(t_start)
        return sched_out

    def _schedule_erlang_inner(self) -> "SchedulerOutputs":
        """erlang_scheduler Erlang dispatch: per-adapter T_max^(k)* from Erlang CDF."""
        t_start = time.perf_counter()

        sched_out: "SchedulerOutputs" = super()._schedule()

        if not _VLLM_AVAILABLE:
            return sched_out

        prefill_groups, decode_groups = self._split_groups(sched_out)

        if not decode_groups:
            self._record_overhead(t_start)
            return sched_out

        # Split: already-aligned sequences bypass the buffer (same fix as threshold).
        first_time_groups = []
        post_alignment_groups = []
        for ssg in decode_groups:
            seq_id = self._seq_group_id(ssg.seq_group)
            if seq_id in self._post_alignment_seqs:
                post_alignment_groups.append(ssg)
            else:
                first_time_groups.append(ssg)

        t_now = time.monotonic()
        seq_id_to_sg = {}
        # Deduplicate EWMA updates: only one update per adapter per tick.
        # If ≥2 new requests for the same adapter arrive in the same scheduling
        # tick, calling estimator.update() twice with the same t_now gives
        # IAT=0 → λ̂→∞ → T_max→0 → immediate dispatch regardless of WAR*.
        # Updating once per adapter per tick preserves the true inter-tick IAT.
        adapters_updated_this_tick: set = set()
        for ssg in first_time_groups:
            sg = ssg.seq_group
            adapter_id = self._adapter_id_of(sg)
            seq_id = self._seq_group_id(sg)
            seq_id_to_sg[seq_id] = ssg
            # Gate estimator on true new arrivals. Sequences deferred from a
            # previous tick are still in first_time_groups (not yet in
            # _post_alignment_seqs) and re-enter this loop on every tick until
            # dispatched. Calling estimator.update for them records a fake IAT
            # of ~1.8 ms, inflating λ̂ to ~550 tok/s and collapsing T_max to
            # ~63 ms for all adapters -- eliminating the NoFair/Fair contrast.
            is_new_arrival = not self._buffer.is_buffered(seq_id)
            self._buffer.enqueue(adapter_id, seq_id)  # idempotent for deferred seqs
            if is_new_arrival and adapter_id not in adapters_updated_this_tick:
                self._estimator.update(adapter_id, t_now)
                adapters_updated_this_tick.add(adapter_id)

        lambda_k_dict = self._estimator.get_all_rates()
        tmax_k = compute_tmax_erlang_batch(
            warp_size=self._warp_size,
            lambda_k_dict=lambda_k_dict,
            war_target=self._war_target,
            ttft_slo_ms=self._ttft_slo_ms,
        )
        # adapter_prefetching: update warm-set tracker; T_max boost for erlang only effective
        # when T_max < SLO (e.g., threshold mode). Whittle mode uses fill_frac penalization.
        tmax_k = self._apply_cold_start_boost(
            tmax_k, lambda_k_dict, adapters_updated_this_tick
        )
        aligned_pairs = self._buffer.form_batch_erlang(
            tmax_k=tmax_k,
            max_tokens=len(decode_groups),
        )

        aligned_first_time = []
        for adapter_id, seq_id in aligned_pairs:
            ssg = seq_id_to_sg.get(seq_id)
            if ssg is not None:
                aligned_first_time.append(ssg)
                self._post_alignment_seqs.add(seq_id)

        post_alignment_groups.sort(key=lambda s: self._adapter_id_of(s.seq_group))
        sched_out.scheduled_seq_groups = (
            prefill_groups + post_alignment_groups + aligned_first_time
        )

        if self._log_war and decode_groups:
            self._log_war_stats(aligned_first_time, first_time_groups, aligned_pairs,
                                all_decode=post_alignment_groups + aligned_first_time)
            logger.info(
                "AS Erlang: WAR*=%.2f n_adapters=%d tmax_range=[%.1f, %.1f]ms",
                self._war_target,
                len(tmax_k),
                min(tmax_k.values()) * 1000 if tmax_k else 0,
                max(tmax_k.values()) * 1000 if tmax_k else 0,
            )

        self._record_overhead(t_start)
        return sched_out

    def _schedule_sort_inner(self) -> "SchedulerOutputs":
        """AS_MODE=sort -- adapter-contiguous batch sort, no deferral (adapter_prefetching v2).

        Architectural motivation:
            Whittle/Erlang modes defer cold adapter requests to future ticks,
            creating queue backlogs when τ_load >> τ_iter (e.g. DP=2 with
            τ_load=96ms and τ_iter=30ms stacks 3+ tick backlogs per cold start).
            The fill_frac penalties rely on a WarmCacheManager that can
            diverge from vLLM's actual GPU LRU state.

        This mode eliminates both problems:
            1. NO deferral: every request selected by super()._schedule() is
               dispatched this tick. Zero extra latency, zero backlog.
            2. Warm-first adapter sort: within the tick, sort decode sequences
               so (a) warm adapters go before cold adapters and (b) within each
               warmth class, adapter IDs are contiguous.
               Effect: warm adapters are processed first, keeping them in the
               GPU LRU; cold adapters follow in a single contiguous run,
               allowing vLLM to pipeline DMA behind the warm-adapter compute.

        PredLFU integration:
            WarmCacheManager.warm_set drives the warm/cold partition for sorting.
            This influences vLLM's LRU INDIRECTLY: warm adapters are served first
            → vLLM updates their LRU timestamps → they are last to be evicted.
            No fill_frac penalty, no T_max deferral -- just priority ordering
            within the tick's already-selected batch.
        """
        t_start = time.perf_counter()

        sched_out: "SchedulerOutputs" = super()._schedule()

        if not _VLLM_AVAILABLE:
            return sched_out

        prefill_groups, decode_groups = self._split_groups(sched_out)

        if not decode_groups:
            self._record_overhead(t_start)
            return sched_out

        # Update EWMA arrival rates and WarmCacheManager for new arrivals only.
        t_now = time.monotonic()
        seen_adapters: set = set()
        for ssg in decode_groups:
            adapter_id = self._adapter_id_of(ssg.seq_group)
            if adapter_id not in seen_adapters:
                seen_adapters.add(adapter_id)
                # Only update estimator for truly new arrivals (not deferred seqs).
                # We use the buffer's buffered check here only for tracking; in sort
                # mode there is no buffer deferral so all are treated as new.
                self._estimator.update(adapter_id, t_now)
        lambda_k_dict = self._estimator.get_all_rates()
        self._update_prefetch_cache(lambda_k_dict, seen_adapters)

        # Warm-first sort: warm adapters → cold adapters, adapter-contiguous within each.
        if self._cache_mgr is not None:
            warm_set = self._cache_mgr.warm_set
            warm_decode = [s for s in decode_groups
                           if self._adapter_id_of(s.seq_group) in warm_set]
            cold_decode = [s for s in decode_groups
                           if self._adapter_id_of(s.seq_group) not in warm_set]
            warm_decode.sort(key=lambda s: self._adapter_id_of(s.seq_group))
            cold_decode.sort(key=lambda s: self._adapter_id_of(s.seq_group))
            sorted_decode = warm_decode + cold_decode
        else:
            # No cache manager: simple adapter-contiguous sort (WAR improvement only).
            sorted_decode = sorted(decode_groups,
                                   key=lambda s: self._adapter_id_of(s.seq_group))

        sched_out.scheduled_seq_groups = prefill_groups + sorted_decode

        self._record_overhead(t_start)
        return sched_out

    def _schedule_whittle_inner(self) -> "SchedulerOutputs":
        """whittle_scheduler Whittle dispatch: Erlang T_max + Whittle-index adapter ranking.

        Adapter selection order is determined by the Whittle index
        W_k(s_k) = p_k * s_k * [1-(1-W*λ_k*Δt)^{W*(1-s_k)}] (Theorem 8.7).
        Dispatch conditions are identical to Erlang mode (full warp OR T_max^(k)*
        timeout), but the highest-index adapter is tried first rather than the
        insertion-order adapter.

        TP-transparency (Proposition 8.8): WhittleDispatcher.rank_adapters() is
        pure Python over scalars -- no GPU calls.  The ranked batch is passed to
        SchedulerOutputs before TP sharding, so WAR(Whittle, TP=d) = WAR(TP=1).
        """
        t_start = time.perf_counter()

        sched_out: "SchedulerOutputs" = super()._schedule()

        if not _VLLM_AVAILABLE:
            return sched_out

        prefill_groups, decode_groups = self._split_groups(sched_out)

        if not decode_groups:
            self._record_overhead(t_start)
            return sched_out

        first_time_groups = []
        post_alignment_groups = []
        for ssg in decode_groups:
            seq_id = self._seq_group_id(ssg.seq_group)
            if seq_id in self._post_alignment_seqs:
                post_alignment_groups.append(ssg)
            else:
                first_time_groups.append(ssg)

        # Update EWMA arrival rates and enqueue into the alignment buffer.
        t_now = time.monotonic()
        adapters_updated_this_tick: set = set()
        arrival_counts: dict = {}
        seq_id_to_sg: dict = {}
        for ssg in first_time_groups:
            sg = ssg.seq_group
            adapter_id = self._adapter_id_of(sg)
            seq_id = self._seq_group_id(sg)
            seq_id_to_sg[seq_id] = ssg
            is_new_arrival = not self._buffer.is_buffered(seq_id)
            self._buffer.enqueue(adapter_id, seq_id)
            if is_new_arrival and adapter_id not in adapters_updated_this_tick:
                self._estimator.update(adapter_id, t_now)
                adapters_updated_this_tick.add(adapter_id)
            if is_new_arrival:
                arrival_counts[adapter_id] = arrival_counts.get(adapter_id, 0) + 1

        # Compute per-adapter arrival rates and Erlang T_max.
        lambda_k_dict = self._estimator.get_all_rates()
        tmax_k = compute_tmax_erlang_batch(
            warp_size=self._warp_size,
            lambda_k_dict=lambda_k_dict,
            war_target=self._war_target,
            ttft_slo_ms=self._ttft_slo_ms,
        )
        # adapter_prefetching: update warm-set tracker with new arrivals
        self._update_prefetch_cache(lambda_k_dict, adapters_updated_this_tick)

        # Lazy-initialise WhittleDispatcher once we know the adapter set.
        # Use the known adapters from the buffer queues.
        known_adapters = list(self._buffer.queues.keys())
        if self._whittle_dispatcher is None or set(known_adapters) != set(self._whittle_dispatcher.adapters):
            self._whittle_dispatcher = WhittleDispatcher(
                adapters=known_adapters,
                warp_size=self._warp_size,
                delta_t=self._whittle_delta_t,
            )

        # Compute fill fractions and rank by Whittle index.
        pending = self._buffer.pending_count()
        fill_fracs = {k: min(v / max(self._warp_size, 1), 1.0) for k, v in pending.items()}
        # adapter_prefetching: lower fill_frac for cold adapters → lower Whittle rank → dispatched later
        # Warm adapters rank higher → served sooner → stay in vLLM GPU LRU
        fill_fracs = self._penalize_cold_fill_fracs(fill_fracs)
        ranked_adapters = self._whittle_dispatcher.rank_adapters(fill_fracs, lambda_k_dict)
        # Update traffic fraction estimates from this tick's new arrivals.
        self._whittle_dispatcher.update_traffic_fractions(arrival_counts)

        # Dispatch: Whittle-ranked order + Erlang per-adapter T_max.
        aligned_pairs = self._buffer.form_batch_whittle(
            ranked_adapters=ranked_adapters,
            tmax_k=tmax_k,
            max_tokens=len(decode_groups),
        )

        aligned_first_time = []
        for adapter_id, seq_id in aligned_pairs:
            ssg = seq_id_to_sg.get(seq_id)
            if ssg is not None:
                aligned_first_time.append(ssg)
                self._post_alignment_seqs.add(seq_id)

        post_alignment_groups.sort(key=lambda s: self._adapter_id_of(s.seq_group))
        sched_out.scheduled_seq_groups = (
            prefill_groups + post_alignment_groups + aligned_first_time
        )

        if self._log_war and decode_groups:
            self._log_war_stats(aligned_first_time, first_time_groups, aligned_pairs,
                                all_decode=post_alignment_groups + aligned_first_time)

        self._record_overhead(t_start)
        return sched_out

    # kernel_promotion: WGKP mode

    def _init_wgkp_components(self, _sys) -> None:
        """Lazy-initialise WGKP components when AS_MODE=wgkp.

        Called once from __init__() after the adapter_prefetching block.
        All components are imported here so they don't load at module level
        (keeps test imports fast when vLLM and Triton are not installed).
        """
        import pathlib as _pathlib

        from adapter_slots.kernel.wgkp_dispatcher import WGKPDispatcher
        from adapter_slots.kernel.merged_weight_cache import MergedWeightCache
        from adapter_slots.kernel.apt import AdaptivePromoThreshold

        # WGKP dispatcher
        self._wgkp_dispatcher = WGKPDispatcher()

        # Merged Weight Cache
        mwc_k_hot = _env_int("AS_MWC_K_HOT", 5)
        mwc_memory_gb = _env_float("AS_MWC_MEMORY_GB", 10.0)
        mwc_projs_raw = os.environ.get("AS_MWC_PROJECTIONS", "q_proj,k_proj,v_proj,o_proj")
        if mwc_projs_raw.strip().lower() == "all":
            mwc_projs = ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"]
        else:
            mwc_projs = [p.strip() for p in mwc_projs_raw.split(",") if p.strip()]

        if mwc_k_hot > 0:
            self._mwc = MergedWeightCache(
                k_hot=mwc_k_hot,
                memory_budget_gb=mwc_memory_gb,
                projections=mwc_projs,
            )
            # Register MWC singleton for AlignmentAwareModelRunner.
            try:
                from adapter_slots.kernel.model_runner import set_mwc
                set_mwc(self._mwc)
            except Exception:
                pass

        # Adaptive Promotion Threshold
        wgkp_hw_profile = os.environ.get("AS_WGKP_HW_PROFILE", "")
        if os.environ.get("AS_WGKP_APT", "0") == "1":
            self._apt = AdaptivePromoThreshold(hw_profile_path=wgkp_hw_profile)
        # Static threshold (default)
        self._wgkp_static_threshold = _env_int("AS_WGKP_THRESHOLD", 8)

        # Macro-batching: AS_MACRO_N_ACCUM > 1 multiplies T_max by N_accum.
        macro_n_accum = _env_int("AS_MACRO_N_ACCUM", 1)
        if macro_n_accum > 1:
            delta_t = _env_float("AS_WHITTLE_DELTA_T", 0.030)
            new_tmax = macro_n_accum * delta_t
            self._buffer.T_max = new_tmax
            print(
                f"[AS] Macro-batching: N_accum={macro_n_accum} τ_iter={delta_t*1000:.1f}ms "
                f"→ T_max={new_tmax*1000:.1f}ms",
                file=_sys.stderr, flush=True,
            )

        print(
            f"[AS] WGKP init: n*={self._wgkp_static_threshold} "
            f"MWC_K_hot={mwc_k_hot} projs={mwc_projs} "
            f"APT={'on' if self._apt else 'off'} "
            f"fused_kernel={os.environ.get('AS_FUSED_KERNEL', '0')}",
            file=_sys.stderr, flush=True,
        )

    def _current_n_star(self) -> int:
        """Return the current promotion threshold n*, from APT or static setting."""
        if self._apt is not None:
            return self._apt.current_threshold()
        return self._wgkp_static_threshold

    def _prefetch_mwc_for_near_full_queues(
        self, lambda_k_dict: dict
    ) -> None:
        """AAP: trigger async MWC merge for queues approaching n*. Disabled: returns.

        It would run once per tick, and for each adapter whose queue depth >= n*/2 and
        whose merged weight is not yet cached, submit merge_async() so that the merge
        completed before the next dispatch tick.

        Nothing consumes a merged weight any more. Level-3 dense weight-merge promotion
        was the only consumer (model_runner.py's execute_model() installed one before a
        forward pass) and it is disabled, because looping per promoted adapter to run a
        full separate transformer forward plus a full per-layer weight clone is O(K)
        against real O(1) kernels (Punica, S-LoRA, vLLM's own BGMV are 6x-105x faster at
        K=16). Merging here would only burn the 2-worker thread pool precomputing dense
        B@A deltas nobody reads: wasted GPU and CPU work on every tick.
        """
        return

    def _get_lora_weights(self, adapter_id: str) -> dict:
        """Retrieve LoRA (A, B, scaling) tensors for adapter_id.

        Returns dict mapping layer_name -> (A, B, scaling) or empty dict if
        the weights are not accessible.

        Returning {} here is not a harmless stub: it makes merge_async() (called from
        _prefetch_mwc_for_near_full_queues) a permanent no-op, leaves MergedWeightCache
        empty, and so keeps the WGKP Condition-2 promotion path from ever firing in a
        real server, paying the scheduling and buffering overhead without the benefit.

        Rather than reaching into vLLM's live in-process LoRA manager (whose internal
        layout is version- and executor-specific), this reads the same PEFT-format
        checkpoint vLLM itself loaded via --lora-modules. It is already on local disk,
        so this stays a plain, robust file read.
        """
        path = self._lora_paths.get(adapter_id)
        if not path or self._mwc is None:
            return {}
        cached = self._lora_weights_cache.get(adapter_id)
        if cached is not None:
            return cached
        try:
            import json as _json
            from pathlib import Path as _Path
            from safetensors import safe_open as _safe_open

            adapter_dir = _Path(path)
            cfg = _json.loads((adapter_dir / "adapter_config.json").read_text())
            rank = cfg.get("r", 16)
            lora_alpha = cfg.get("lora_alpha", rank)
            scaling = lora_alpha / rank
            projections = self._mwc.projections
            weights: dict = {}
            with _safe_open(str(adapter_dir / "adapter_model.safetensors"), framework="pt") as f:
                for key in f.keys():
                    if not key.endswith(".lora_A.weight"):
                        continue
                    layer_name = key[: -len(".lora_A.weight")]
                    if not any(p in layer_name for p in projections):
                        continue
                    b_key = layer_name + ".lora_B.weight"
                    if b_key not in f.keys():
                        continue
                    weights[layer_name] = (f.get_tensor(key), f.get_tensor(b_key), scaling)
            self._lora_weights_cache[adapter_id] = weights
            return weights
        except Exception as e:
            import sys as _sys
            print(f"[AS] _get_lora_weights({adapter_id}) failed: {e}", file=_sys.stderr)
            return {}

    def _schedule_wgkp_inner(self) -> "SchedulerOutputs":
        """kernel_promotion WGKP dispatch: Whittle + CASH + Fused kernel + GEMM promotion.

        Extends _schedule_whittle_inner() with:
        1. form_batch_wgkp() instead of form_batch_whittle() (CASH holdback)
        2. WGKPDispatcher.segment_and_promote() for SegmentDescriptor generation
        3. MWC async prefetch (AAP) for near-full queues
        4. Attach wgkp_segments to sched_out.metadata for ModelRunner
        5. Log promotion_fraction if AS_WGKP_LOG=1

        Existing modes (threshold, erlang, pi_adaptive, whittle, sort) are untouched.
        """
        import sys as _sys
        t_start = time.perf_counter()

        n_prefill_in_running = sum(1 for sg in self.running if sg.is_prefill())
        buffered_total = sum(len(q) for q in self._buffer.queues.values())
        print(
            f"[WGKP] pre-schedule: waiting={len(self.waiting)} "
            f"running={len(self.running)} running_prefill={n_prefill_in_running} "
            f"post_aln={len(self._post_alignment_seqs)} "
            f"buffered={buffered_total}",
            file=_sys.stderr,
            flush=True,
        )

        sched_out: "SchedulerOutputs" = super()._schedule()

        if not _VLLM_AVAILABLE:
            return sched_out

        prefill_groups, decode_groups = self._split_groups(sched_out)

        if not decode_groups:
            self._record_overhead(t_start)
            return sched_out

        # Prune _post_alignment_seqs and _buffer._seq_enqueue to the set of
        # sequences still alive in self.running.  Both grow without bound
        # otherwise: seq_ids are added on dispatch but never removed when a
        # sequence finishes (metrics logging only pops _seq_enqueue when
        # AS_METRICS_PATH is set, leaving silent accumulation in dark runs).
        # Bounding cost: O(|running|) per tick, same order as _split_groups.
        active_ids: set = {
            seq.seq_id
            for sg in self.running
            for seq in sg.get_seqs()
        }
        stale = self._post_alignment_seqs - active_ids
        self._post_alignment_seqs -= stale
        for sid in stale:
            self._buffer._seq_enqueue.pop(sid, None)

        first_time_groups = []
        post_alignment_groups = []
        for ssg in decode_groups:
            seq_id = self._seq_group_id(ssg.seq_group)
            if seq_id in self._post_alignment_seqs:
                post_alignment_groups.append(ssg)
            else:
                first_time_groups.append(ssg)

        # Update EWMA arrival rates and enqueue into alignment buffer.
        t_now = time.monotonic()
        adapters_updated_this_tick: set = set()
        arrival_counts: dict = {}
        seq_id_to_sg: dict = {}
        for ssg in first_time_groups:
            sg = ssg.seq_group
            adapter_id = self._adapter_id_of(sg)
            seq_id = self._seq_group_id(sg)
            seq_id_to_sg[seq_id] = ssg
            is_new_arrival = not self._buffer.is_buffered(seq_id)
            self._buffer.enqueue(adapter_id, seq_id)
            if is_new_arrival and adapter_id not in adapters_updated_this_tick:
                self._estimator.update(adapter_id, t_now)
                adapters_updated_this_tick.add(adapter_id)
            if is_new_arrival:
                arrival_counts[adapter_id] = arrival_counts.get(adapter_id, 0) + 1

        lambda_k_dict = self._estimator.get_all_rates()
        tmax_k = compute_tmax_erlang_batch(
            warp_size=self._warp_size,
            lambda_k_dict=lambda_k_dict,
            war_target=self._war_target,
            ttft_slo_ms=self._ttft_slo_ms,
        )
        # adapter_prefetching: update warm-set tracker and penalize cold fill fracs.
        self._update_prefetch_cache(lambda_k_dict, adapters_updated_this_tick)

        # Lazy-initialise WhittleDispatcher once adapters are known.
        known_adapters = list(self._buffer.queues.keys())
        if self._whittle_dispatcher is None or set(known_adapters) != set(
            self._whittle_dispatcher.adapters
        ):
            if os.environ.get("AS_ADMISSION_LOG") and self._whittle_dispatcher is not None:
                print(
                    f"[ADMISSION] t={time.perf_counter():.4f} WHITTLE_REBUILD "
                    f"old_K={len(self._whittle_dispatcher.adapters)} new_K={len(known_adapters)} "
                    f"old_p_k={self._whittle_dispatcher.p_k} "
                    f"-- learned traffic-fraction priors for all {len(self._whittle_dispatcher.adapters)} "
                    f"previously-known adapters are discarded, reset to uniform 1/{len(known_adapters)}",
                    file=_sys.stderr, flush=True,
                )
            self._whittle_dispatcher = WhittleDispatcher(
                adapters=known_adapters,
                warp_size=self._warp_size,
                delta_t=self._whittle_delta_t,
            )

        pending = self._buffer.pending_count()
        fill_fracs = {k: min(v / max(self._warp_size, 1), 1.0) for k, v in pending.items()}
        fill_fracs = self._penalize_cold_fill_fracs(fill_fracs)
        ranked_adapters = self._whittle_dispatcher.rank_adapters(fill_fracs, lambda_k_dict)
        if os.environ.get("AS_ADMISSION_LOG"):
            print(
                f"[ADMISSION] t={time.perf_counter():.4f} tick "
                f"n_running={len(self.running)} n_known={len(known_adapters)} "
                f"ranked={ranked_adapters} "
                f"p_k={ {k: round(v, 4) for k, v in self._whittle_dispatcher.p_k.items()} } "
                f"fill_fracs={ {k: round(v, 3) for k, v in fill_fracs.items()} } "
                f"lambda={ {k: round(v, 3) for k, v in lambda_k_dict.items()} }",
                file=_sys.stderr, flush=True,
            )
        self._whittle_dispatcher.update_traffic_fractions(arrival_counts)

        # APT update (every tick; APT fires internal update on interval).
        mean_q_depth = sum(pending.values()) / max(len(pending), 1)
        if self._apt is not None:
            self._apt.update(mean_q_depth)

        # AAP: prefetch MWC for near-full queues.
        self._prefetch_mwc_for_near_full_queues(lambda_k_dict)

        # Determine merged adapter set for CASH promotion eligibility.
        merged_ids: set = set()
        if self._mwc is not None:
            merged_ids = {a for a in known_adapters if self._mwc.is_merged(a)}

        n_star = self._current_n_star()

        # CASH dispatch (form_batch_wgkp).
        # Budget is the real per-tick decode capacity (decode_groups), not
        # len(first_time_groups). The dispatch budget caps how many *total*
        # buffered tokens (across all adapters, including ones held over from
        # prior ticks that already hit their TTFT deadline) can leave the
        # buffer this tick. Capping it to only brand-new first-time arrivals
        # meant that once the initial admission burst passed, first_time_groups
        # collapsed toward 0 on most decode ticks (newly-dispatched sequences
        # are flagged post-alignment and bypass the buffer thereafter), so the
        # budget collapsed too -- even fully-expired Condition-4 (TTFT hard-cap)
        # tokens could never force-flush. Real measurement showed the buffer
        # stuck at a constant backlog (e.g. "buffered=24") for 100+ consecutive
        # scheduler ticks, with TTFT_p50 around 9-14s -- a stall, not genuine
        # queuing delay. Sizing the budget to decode_groups (the actual number
        # of decode slots scheduled this tick) removes the artificial cap
        # while still bounding dispatch by real GPU capacity.
        if os.environ.get("AS_DEBUG_WGKP_DISPATCH"):
            _t_dbg = time.perf_counter()
            _dbg = []
            for aid in ranked_adapters:
                q = self._buffer.queues.get(aid)
                if not q:
                    continue
                enq_t = self._buffer.enqueue_time.get(aid)
                age = (_t_dbg - enq_t) if enq_t is not None else -1.0
                eff_dl = min(tmax_k.get(aid, self._buffer.T_max), self._buffer.ttft_slo)
                _dbg.append(f"{aid}:qlen={len(q)},age={age:.3f},eff_dl={eff_dl:.3f},merged={aid in merged_ids}")
            print(f"[ASDBG] n_star={n_star} budget={len(decode_groups)} " + " | ".join(_dbg),
                  file=_sys.stderr, flush=True)
        raw_batch = self._buffer.form_batch_wgkp(
            ranked_adapters=ranked_adapters,
            tmax_k=tmax_k,
            n_star=n_star,
            merged_adapter_ids=merged_ids,
            max_tokens=len(decode_groups),
        )

        if os.environ.get("AS_ADMISSION_LOG"):
            dispatched_per_adapter: dict = {}
            for aid, _sid, _promo in raw_batch:
                dispatched_per_adapter[aid] = dispatched_per_adapter.get(aid, 0) + 1
            starved = [
                aid for aid in ranked_adapters
                if (self._buffer.queues.get(aid) and dispatched_per_adapter.get(aid, 0) == 0)
            ]
            print(
                f"[ADMISSION] t={time.perf_counter():.4f} dispatch "
                f"budget={len(decode_groups)} dispatched_total={len(raw_batch)} "
                f"dispatched_per_adapter={dispatched_per_adapter} "
                f"starved_this_tick={starved}",
                file=_sys.stderr, flush=True,
            )

        # Segment and promote.
        segments = []
        if self._wgkp_dispatcher is not None and raw_batch:
            try:
                segments = self._wgkp_dispatcher.segment_and_promote(raw_batch)
            except ValueError as e:
                import sys as _sys
                print(f"[AS] WGKP segment_and_promote error: {e}", file=_sys.stderr)
                segments = []

        # Build aligned_first_time from raw_batch.
        t_dispatch = time.perf_counter()
        aligned_first_time = []
        dispatched_ids = set()
        for adapter_id, seq_id, is_promoted in raw_batch:
            ssg = seq_id_to_sg.get(seq_id)
            if ssg is not None:
                aligned_first_time.append(ssg)
                self._post_alignment_seqs.add(seq_id)
                dispatched_ids.add(seq_id)
                eff_dl_ms = min(tmax_k.get(adapter_id, self._buffer.T_max), self._buffer.ttft_slo) * 1000.0
                self._record_buffer_wait(adapter_id, seq_id, t_dispatch, t_star_ms=eff_dl_ms)

        # Only force-drain held-back (condition-5) sequences when this tick's
        # schedule would otherwise be completely empty. The original bug:
        # excluding held sequences from scheduled_seq_groups unconditionally
        # meant a held sequence got zero forward passes until WGKP released
        # it. Usually harmless (other post_alignment sequences keep the
        # engine ticking every step regardless), but at cold start -- when
        # *everything* is first-time and nothing is post-alignment yet --
        # scheduled_seq_groups came back fully empty, vLLM's async engine
        # loop had nothing to execute and throttled its own tick rate (one
        # measured gap: 0.94s vs. a normal ~0.04s), which delayed the very
        # T_max/TTFT-hard-cap deadline check meant to release them -- turning
        # a nominal 2s SLO cap into a real ~31s freeze of 33 sequences.
        # An earlier version of this fix drained unconditionally on every
        # tick, which did stop the stall but also meant a request never got
        # more than one tick to accumulate toward n*/full-warp before being
        # forced through unpromoted -- measured promotion_fraction=0.000
        # across an entire run. Gating the drain on "schedule would
        # otherwise be empty" fixes the cold-start starvation feedback loop
        # while leaving CASH's real multi-tick accumulation (conditions 1-4,
        # same n*/W/T_max as the paper) intact whenever other decode work is
        # already keeping the engine loop alive -- which is true almost all
        # of steady-state serving.
        would_be_empty = not (prefill_groups or post_alignment_groups or aligned_first_time)
        if would_be_empty:
            t_drain = time.perf_counter()
            drained = self._drain_buffer_remaining(first_time_groups, dispatched_ids)
            for ssg in drained:
                seq_id = self._seq_group_id(ssg.seq_group)
                self._post_alignment_seqs.add(seq_id)
                drain_adapter_id = self._adapter_id_of(ssg.seq_group)
                eff_dl_ms = min(tmax_k.get(drain_adapter_id, self._buffer.T_max), self._buffer.ttft_slo) * 1000.0
                self._record_buffer_wait(drain_adapter_id, seq_id, t_drain, t_star_ms=eff_dl_ms)
        else:
            drained = []

        if os.environ.get("AS_ADMISSION_LOG"):
            native_decode_budget = len(first_time_groups) + len(post_alignment_groups)
            actual_decode_batch = len(post_alignment_groups) + len(aligned_first_time) + len(drained)
            held_back = len(first_time_groups) - len(aligned_first_time) - len(drained)
            print(
                f"[ADMISSION] t={time.perf_counter():.4f} batch_shrink "
                f"native_decode_budget={native_decode_budget} actual_decode_batch={actual_decode_batch} "
                f"held_back_this_tick={held_back}",
                file=_sys.stderr, flush=True,
            )

        post_alignment_groups.sort(key=lambda s: self._adapter_id_of(s.seq_group))
        sched_out.scheduled_seq_groups = (
            prefill_groups + post_alignment_groups + aligned_first_time + drained
        )

        # Attach wgkp_segments to metadata for AlignmentAwareModelRunner.
        if segments:
            if not hasattr(sched_out, "metadata") or sched_out.metadata is None:
                try:
                    sched_out.metadata = {}
                except (AttributeError, TypeError):
                    pass
            try:
                sched_out.metadata["wgkp_segments"] = segments
            except (AttributeError, TypeError):
                pass

        # Logging.
        if self._wgkp_log and self._wgkp_dispatcher is not None:
            promo_frac = self._wgkp_dispatcher.promotion_fraction()
            print(
                f"[WGKP] n*={n_star} merged={len(merged_ids)} "
                f"segments={len(segments)} promo_frac={promo_frac:.3f}",
                file=_sys.stderr, flush=True,
            )
            if len(self._tick_times_ms) % 50 == 0:
                ov = self.overhead_stats()
                print(
                    f"[ASOVERHEAD] n_ticks={ov.get('n_ticks')} "
                    f"mean_ms={ov.get('mean_ms', 0):.3f} p50_ms={ov.get('p50_ms', 0):.3f} "
                    f"known_adapters={len(known_adapters)}",
                    file=_sys.stderr, flush=True,
                )
                bw = self.buffer_wait_stats()
                for aid, st in bw.items():
                    print(
                        f"[BUFFER_WAIT_AGG] adapter={aid} n={st['n']} "
                        f"mean_ms={st['mean_ms']:.1f} p50_ms={st['p50_ms']:.1f} "
                        f"p99_ms={st['p99_ms']:.1f} max_ms={st['max_ms']:.1f}",
                        file=_sys.stderr, flush=True,
                    )

        if self._log_war and decode_groups:
            raw_pairs = [(a, s) for a, s, _ in raw_batch]
            self._log_war_stats_wgkp(
                aligned_first_time, first_time_groups, raw_pairs,
                all_decode=post_alignment_groups + aligned_first_time,
                n_star=n_star,
                segments=segments,
            )

        self._record_overhead(t_start)
        return sched_out

    def _log_war_stats_wgkp(
        self,
        aligned_decode: list,
        decode_groups: list,
        aligned_pairs: list,
        all_decode: Optional[list] = None,
        n_star: int = 8,
        segments: Optional[list] = None,
    ) -> None:
        """Extended WAR/GWAR logging for WGKP mode."""
        from adapter_slots.metrics.gwar import compute_gwar
        batch_groups = all_decode if all_decode is not None else aligned_decode
        adapter_ids = [
            self._adapter_id_of(ssg.seq_group) for ssg in batch_groups
        ]
        int_ids = [self._adapter_id_to_int(aid) for aid in adapter_ids]
        war = compute_war_from_ids(int_ids, warp_size=self._buffer.W)
        gwar8 = compute_gwar(int_ids, threshold=n_star)
        promo_frac = (
            self._wgkp_dispatcher.promotion_fraction()
            if self._wgkp_dispatcher
            else 0.0
        )
        stats = self._buffer.stats()
        deferred_count = len(decode_groups) - len(aligned_decode)
        logger.info(
            "AS WGKP tick: WAR=%.3f GWAR(%d)=%.3f promo_frac=%.3f "
            "dispatched=%d deferred=%d max_q=%d",
            war, n_star, gwar8, promo_frac,
            len(aligned_decode), deferred_count,
            stats["max_queue_depth"],
        )

        if self._metrics_file is not None:
            t_dispatch_perf = time.perf_counter()
            dispatch_time_ms = t_dispatch_perf * 1000.0 - self._t0_ms
            n_adapters = len(set(adapter_ids))
            n_tokens = len(adapter_ids)
            token_records = []
            for ssg in batch_groups:
                aid = self._adapter_id_of(ssg.seq_group)
                sid = self._seq_group_id(ssg.seq_group)
                t_enq = self._buffer._seq_enqueue.pop(sid, None)
                arrival_ms = (
                    round(t_enq * 1000.0 - self._t0_ms, 3)
                    if t_enq is not None
                    else round(dispatch_time_ms, 3)
                )
                token_records.append({
                    "adapter_id": aid,
                    "arrival_time_ms": arrival_ms,
                    "seq_id": sid,
                })
            record = {
                "batch_id": self._batch_id,
                "dispatch_time_ms": round(dispatch_time_ms, 3),
                "war": round(war, 6),
                "gwar": round(gwar8, 6),
                "promotion_fraction": round(promo_frac, 4),
                "global_intensity": round(n_tokens / max(1, n_adapters), 4),
                "tokens": token_records,
            }
            self._metrics_file.write(json.dumps(record) + "\n")
            self._metrics_file.flush()
            self._batch_id += 1

    # Shared helpers

    def _update_prefetch_cache(
        self,
        lambda_k_dict: dict,
        new_arrivals: set,
    ) -> None:
        """Update WarmCacheManager with new arrivals this tick (adapter_prefetching).

        Called once per tick from any dispatch mode that tracks arrivals.
        Uses the same gate as ArrivalRateEstimator (new_arrivals set) to avoid
        double-counting deferred sequences that re-enter the loop each tick.

        Only active when AS_PREFETCH_POLICY != "none" and AS_K_WARM > 0.
        """
        if self._cache_mgr is None:
            return
        for adapter_id in new_arrivals:
            self._cache_mgr.request(
                adapter_id,
                rate_estimates=lambda_k_dict,
                t_now=time.perf_counter(),
            )

    def _penalize_cold_fill_fracs(self, fill_fracs: dict) -> dict:
        """Lower the effective fill fraction for predicted-cold adapters (adapter_prefetching).

        Two-phase cold-adapter deferral for PCIe hardware:

        Phase 1 -- hard block (0 ≤ age < τ_load): fill_frac = 0.0
          Adapter is loading via PCIe DMA. Setting fill_frac = 0 prevents any
          dispatch until the DMA completes. Implements S-LoRA §4.3 minimum
          deferral window: cold adapter not eligible until t >= t_first_seen + τ_load.
          Active only when AS_PCIE_MIN_DEFERRAL_S > 0.

        Phase 2 -- soft priority reduction (age ≥ τ_load or min-deferral disabled):
          fill_frac = fill_frac / cold_boost (existing cold_boost mechanism).
          DMA complete; adapter is warm in vLLM LRU but cache_mgr hasn't updated yet.
          cold_boost=2.0 (PCIe-calibrated: ceil(τ_load/τ_iter)+1) ensures warm
          adapters still rank higher during the brief warm-set propagation lag.

        When adapter becomes warm: cold_first_seen entry cleared so Phase 1 doesn't
        re-activate if the adapter is later evicted and re-loaded.

        Only active when AS_PREFETCH_POLICY != "none" and AS_K_WARM > 0.
        """
        if self._cache_mgr is None:
            return fill_fracs
        warm_set = self._cache_mgr.warm_set
        t_now = time.perf_counter()
        penalized: dict = {}
        for adapter_id, frac in fill_fracs.items():
            if adapter_id in warm_set:
                # Adapter is warm -- clear any stale cold tracking and dispatch normally
                self._cold_first_seen.pop(adapter_id, None)
                penalized[adapter_id] = frac
            else:
                # Cold adapter -- check PCIe minimum deferral window (Phase 1)
                if self._pcie_min_deferral_s > 0.0:
                    first_seen = self._cold_first_seen.get(adapter_id)
                    if first_seen is None:
                        # First time seeing this cold adapter: record DMA start time
                        self._cold_first_seen[adapter_id] = t_now
                        penalized[adapter_id] = 0.0  # hard block: DMA just initiated
                    elif (t_now - first_seen) < self._pcie_min_deferral_s:
                        # Still within deferral window: apply soft penalty to allow
                        # SLO-forced dispatch if T_max expires, but at lowest priority
                        penalized[adapter_id] = frac / self._cold_boost
                    else:
                        # Deferral window elapsed: DMA complete, allow normal dispatch
                        penalized[adapter_id] = frac
                else:
                    # Min-deferral disabled (non-PCIe or explicit opt-out): Phase 2 only
                    penalized[adapter_id] = frac / self._cold_boost
        return penalized

    def _apply_cold_start_boost(
        self,
        tmax_k: dict,
        lambda_k_dict: dict,
        new_arrivals: set,
    ) -> dict:
        """Apply T_max cold-boost for predicted-cold adapters in threshold/erlang modes.

        Works when T_max < TTFT_SLO (i.e., threshold mode with small AS_TMAX_MS).
        For Whittle mode, use _penalize_cold_fill_fracs() instead (fill_frac
        penalization is effective regardless of T_max magnitude).

        Cold adapters: T_max → min(T_max × cold_boost, SLO)
        Warm adapters: T_max unchanged
        """
        if self._cache_mgr is None:
            return tmax_k
        self._update_prefetch_cache(lambda_k_dict, new_arrivals)
        slo_s = self._ttft_slo_ms / 1000.0
        boosted: dict = {}
        for adapter_id, tmax_s in tmax_k.items():
            if not self._cache_mgr.is_warm(adapter_id):
                boosted[adapter_id] = min(tmax_s * self._cold_boost, slo_s)
            else:
                boosted[adapter_id] = tmax_s
        return boosted

    def _split_groups(self, sched_out: "SchedulerOutputs"):
        """Split scheduled_seq_groups into (prefill_groups, decode_groups)."""
        prefill_groups = []
        decode_groups = []
        for sg in sched_out.scheduled_seq_groups:
            seqs = sg.seq_group.get_seqs(status=SequenceStatus.RUNNING)
            is_decode = all(s.data.get_len() > s.data.get_prompt_len()
                            for s in seqs) if seqs else False
            if is_decode:
                decode_groups.append(sg)
            else:
                prefill_groups.append(sg)
        return prefill_groups, decode_groups

    def _enqueue_decode_groups(self, decode_groups: list) -> dict:
        """Enqueue all decode seq groups into the alignment buffer.

        Returns seq_id → SchedulerSeqGroup mapping.
        """
        seq_id_to_sg = {}
        for ssg in decode_groups:
            sg = ssg.seq_group
            adapter_id = self._adapter_id_of(sg)
            seq_id = self._seq_group_id(sg)
            seq_id_to_sg[seq_id] = ssg
            self._buffer.enqueue(adapter_id, seq_id)
        return seq_id_to_sg

    def _drain_buffer_remaining(self, decode_groups: list, dispatched_ids: set) -> list:
        """Remove buffer-deferred seq_ids and return their SchedulerSeqGroups.

        Sequences that form_batch() did not dispatch are still queued in the
        buffer. We drain them now (removing from the per-adapter deques) so they
        don't accumulate and get double-enqueued on the next tick.  They are
        included at the end of this batch so the scheduled_seq_groups count
        matches the seq_group_metadata_list built by the caller.
        """
        remaining = []
        for ssg in decode_groups:
            seq_id = self._seq_group_id(ssg.seq_group)
            if seq_id not in dispatched_ids:
                remaining.append(ssg)
                adapter_id = self._adapter_id_of(ssg.seq_group)
                q = self._buffer.queues.get(adapter_id)
                if q is not None:
                    self._buffer.queues[adapter_id] = deque(
                        (s, t) for s, t in q if s != seq_id
                    )
                    if not self._buffer.queues[adapter_id]:
                        self._buffer.enqueue_time[adapter_id] = None
        return remaining

    def _adapter_id_to_int(self, adapter_id: str) -> int:
        """Convert string adapter ID to a stable integer for WAR computation."""
        try:
            return int(adapter_id.split("_")[-1])
        except ValueError:
            return hash(adapter_id) & 0x7FFFFFFF

    def _log_war_stats(
        self,
        aligned_decode: list,
        decode_groups: list,
        aligned_pairs: list,
        all_decode: Optional[list] = None,
    ) -> None:
        """Emit WAR and queue stats to the logger and optionally to a JSONL file.

        all_decode: if provided, WAR is computed for the full decode batch
        (aligned + remaining) so that ticks with partial alignment contribute
        non-trivial variance to the EC8a correlation.
        """
        batch_groups = all_decode if all_decode is not None else aligned_decode
        adapter_ids = [
            self._adapter_id_of(ssg.seq_group) for ssg in batch_groups
        ]
        int_ids = [self._adapter_id_to_int(aid) for aid in adapter_ids]
        war = compute_war_from_ids(int_ids, warp_size=self._buffer.W)

        stats = self._buffer.stats()
        deferred_count = len(decode_groups) - len(aligned_decode)
        logger.info(
            "AS tick: WAR=%.3f dispatched=%d deferred=%d "
            "max_q=%d timeout_dispatches=%d",
            war, len(aligned_decode), deferred_count,
            stats["max_queue_depth"], stats["n_timeout_dispatches"],
        )

        if self._metrics_file is not None:
            t_dispatch_perf = time.perf_counter()
            dispatch_time_ms = t_dispatch_perf * 1000.0 - self._t0_ms
            n_adapters = len(set(adapter_ids))
            n_tokens = len(adapter_ids)
            # Build per-token records with arrival_time_ms for WARτ computation.
            # For each seq group in the batch, look up its enqueue time in the
            # alignment buffer.  Tokens not tracked (prefill, force-dispatch) use
            # dispatch_time_ms as arrival so wartau_ms = 0.
            token_records = []
            for ssg in batch_groups:
                aid = self._adapter_id_of(ssg.seq_group)
                sid = self._seq_group_id(ssg.seq_group)
                t_enq = self._buffer._seq_enqueue.pop(sid, None)
                if t_enq is not None:
                    arrival_ms = round(t_enq * 1000.0 - self._t0_ms, 3)
                else:
                    arrival_ms = round(dispatch_time_ms, 3)
                token_records.append({
                    "adapter_id": aid,
                    "arrival_time_ms": arrival_ms,
                    "seq_id": sid,
                })
            cache_stats: dict = {}
            if self._cache_mgr is not None:
                s = self._cache_mgr.stats()
                cache_stats = {
                    "prefetch_policy": s["policy"],
                    "cache_hit_rate": round(s["hit_rate"], 4),
                    "cache_cold_fraction": round(s["cold_fraction"], 4),
                    "cache_n_warm": s["n_warm_current"],
                }
            record = {
                "batch_id": self._batch_id,
                "dispatch_time_ms": round(dispatch_time_ms, 3),
                "war": round(war, 6),
                "global_intensity": round(n_tokens / max(1, n_adapters), 4),
                "tokens": token_records,
                **cache_stats,
            }
            self._metrics_file.write(json.dumps(record) + "\n")
            self._metrics_file.flush()
            self._batch_id += 1

    def _record_overhead(self, t_start: float) -> None:
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self._tick_times_ms.append(elapsed_ms)
        # Keep a rolling window of the last 10 000 ticks to avoid unbounded growth
        if len(self._tick_times_ms) > 10_000:
            self._tick_times_ms = self._tick_times_ms[-10_000:]

    def overhead_stats(self) -> dict:
        """Return scheduler overhead statistics (§5.4 measurement)."""
        if not self._tick_times_ms:
            return {}
        import statistics
        return {
            "n_ticks": len(self._tick_times_ms),
            "mean_ms": statistics.mean(self._tick_times_ms),
            "p50_ms": statistics.median(self._tick_times_ms),
            "p99_ms": sorted(self._tick_times_ms)[int(0.99 * len(self._tick_times_ms))],
            "max_ms": max(self._tick_times_ms),
        }

    def _record_buffer_wait(
        self, adapter_id: str, seq_id: int, t_dispatch: float,
        t_star_ms: Optional[float] = None,
    ) -> None:
        """Record dispatch_time - arrival_time for one seq_id, per adapter.

        Pops the seq_id's enqueue record from AlignmentBuffer (same as
        pop_wartau_ms would have, had anything called it). No-op if seq_id
        was never tracked (e.g. dispatched before this instrumentation was
        added in the same process, or a prefill-only request).

        t_star_ms, when passed, is the CASH/Erlang deadline T*(k) that was
        in effect for this adapter at dispatch time (eff_dl = min(tmax_k,
        ttft_slo), in ms). Comparing actual_wait against it answers: is the
        wait explained by the deadline itself (actual_wait ~= T*(k), i.e.
        Erlang/CASH is "working as designed, just designed to wait this
        long"), or is something else gating release on top of it
        (actual_wait >> T*(k))? Logs the first AS_BUFFER_WAIT_SAMPLE_LIMIT
        (default 20) raw comparisons per adapter rather than every one, per
        request ("for a few requests").
        """
        if seq_id not in self._buffer._seq_enqueue:
            return
        wait_ms = self._buffer.pop_wartau_ms(seq_id, t_dispatch)
        self._buffer_wait_ms.setdefault(adapter_id, []).append(wait_ms)
        if t_star_ms is None or not self._buffer_wait_log:
            return
        n_logged = self._buffer_wait_sample_count.get(adapter_id, 0)
        if n_logged >= self._buffer_wait_sample_limit:
            return
        self._buffer_wait_sample_count[adapter_id] = n_logged + 1
        ratio = wait_ms / t_star_ms if t_star_ms > 0 else float("inf")
        import sys as _sys
        print(
            f"[BUFFER_WAIT] adapter={adapter_id} seq={seq_id} "
            f"actual_wait_ms={wait_ms:.1f} T_star_ms={t_star_ms:.1f} "
            f"ratio={ratio:.2f} "
            f"verdict={'CASH/Erlang deadline (as designed)' if 0.7 <= ratio <= 1.5 else 'EXTRA GATING beyond T*(k)' if ratio > 1.5 else 'released early'}",
            file=_sys.stderr, flush=True,
        )

    def buffer_wait_stats(self) -> dict:
        """Return per-adapter buffer_wait (dispatch_time - arrival_time) stats, ms.

        This is the direct answer to "how much of the missing throughput is
        spent waiting in the alignment buffer before dispatch, vs. anywhere
        else (GPU forward, network, sampling)."
        """
        import statistics
        out = {}
        for adapter_id, samples in self._buffer_wait_ms.items():
            if not samples:
                continue
            s = sorted(samples)
            out[adapter_id] = {
                "n": len(s),
                "mean_ms": statistics.mean(s),
                "p50_ms": statistics.median(s),
                "p99_ms": s[int(0.99 * (len(s) - 1))],
                "max_ms": s[-1],
            }
        return out

    @property
    def buffer(self) -> AlignmentBuffer:
        """Expose the underlying AlignmentBuffer (for monitoring scripts)."""
        return self._buffer
