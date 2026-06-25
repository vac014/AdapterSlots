"""
vllm_scheduler.py -- AlignmentAwareScheduler: vLLM 0.6.3 scheduler subclass.

Each tick vLLM selects the running and newly admitted sequence groups; the
scheduler then passes each sequence's first decode token through the
AlignmentBuffer and hands vLLM the batch in adapter order:

    Scheduler._schedule()
        -> super()._schedule()             vLLM's selection
        -> AlignmentBuffer.enqueue(...)    first decode tokens into Q_k
        -> AlignmentBuffer.form_batch_*()  queues dispatched this tick
        -> scheduled_seq_groups reordered  adapter-contiguous decode batch

The batch is reordered before tensor-parallel sharding, so every TP rank runs
the same aligned order. vLLM 0.6.3 has no flag for a custom scheduler; the
server entry point scripts/vllm_serve_adapter_slots.py builds the engine with
this class (integrations/aligned_engine.py, aligned_mp_engine.py).

Environment variables:
    AS_MODE              dispatch mode: threshold (global T_max), erlang (T*(k)),
                         pi_adaptive (PI-controlled T_max), cash (T*(k) in CASH
                         order), wgkp (CASH holdback with WGKP promotion flags)
    AS_DEADLINE          erlang (default): per-adapter deadlines T*(k); fixed: one
                         deadline T_max = AS_TMAX_MS for every adapter (cash, wgkp)
    AS_TMAX_MS           global deadline T_max (threshold mode, AS_DEADLINE=fixed; default 5)
    AS_TTFT_SLO_MS       T_SLO, the cap on any buffer wait (default 200)
    AS_WARP_SIZE         W (default 32)
    AS_WAR_TARGET        WAR* (default 0.8)
    AS_EWMA_ALPHA        EWMA weight of the rate estimator (default 0.1)
    AS_PI_KP, AS_PI_KI   PI gains (defaults 0.01, 0.001)
    AS_PI_TMAX_INIT_MS   initial T_max under PI control (default 18)
    AS_WGKP_THRESHOLD    promotion threshold n* (default 8)
    AS_WGKP_APT          1: choose n* at run time from the kernel crossover profile
    AS_WGKP_HW_PROFILE   crossover profile JSON (default kernel/hw_profiles/a6000_tp1.json)
    AS_MWC_K_HOT         upper bound on adapters in the L3 merged-weight cache (default 5;
                         0 disables L3); the cache holds as many as fit in free GPU memory
    AS_KV_ADMISSION      1: admit a request only if KV memory covers its whole lifetime
                         on top of the running sequences' remaining growth
    AS_ADMIT_GROUPED     1: admit waiting requests adapter by adapter, longest queue first
    AS_GROUP_MAX_WAIT_S  alignment deadline of grouped admission (default 30)
    AS_ROTATE_AFTER_S    adapter rotation when LoRA slots are fewer than adapters (default 3)
    AS_PREFETCH_POLICY   none (default), lru, topk or predictive (PredictiveLFU)
    AS_K_WARM            warm-set capacity, equal to --max-loras (default 0, off)
    AS_TAU_LOAD_MS       adapter load time tau_load (default 96.3, measured on the A6000)
    AS_COLD_BOOST        deadline multiplier / score divisor for cold adapters (default 2.5)
    AS_PCIE_MIN_DEFERRAL_S  hold a cold adapter this long after first sight (default 0, off)
    AS_LOG_WAR           1: write one JSON line per decode batch to AS_METRICS_PATH
    AS_METRICS_PATH      per-batch log (WAR, GWAR(n*), token order by adapter)
    AS_REQ_METRICS_PATH  per-request engine timestamps (arrival, first scheduled, first token)
"""

import json
import math
import os

import pathlib
import time
import logging
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)

# vLLM is needed only when serving; the unit tests import this module without it.
try:
    from vllm.core.scheduler import Scheduler, SchedulerOutputs, SchedulerRunningOutputs
    from vllm.sequence import SequenceGroup, SequenceStatus
    from vllm.core.interfaces import AllocStatus
    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False
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
from adapter_slots.dispatch.cash import CASHDispatcher
from adapter_slots.control.estimator import ArrivalRateEstimator
from adapter_slots.kernel.l3 import get_mwc



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
    """vLLM scheduler subclass that routes first decode tokens through the
    AlignmentBuffer and emits adapter-contiguous decode batches.

    Each tick:
    1. super()._schedule() makes vLLM's selection.
    2. Decode groups whose sequence has not yet been dispatched through the
       buffer are enqueued in Q_k; the dispatch rule of AS_MODE picks the
       queues that leave this tick.
    3. scheduled_seq_groups becomes prefills, then the already-aligned decode
       groups sorted by adapter, then this tick's dispatched groups.
    Groups held in the buffer stay in vLLM's running queue and are offered
    again on the next tick.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        tmax_ms = _env_float("AS_TMAX_MS", 5.0)
        # T_SLO: held tokens are released at this deadline.
        ttft_slo_ms = _env_float("AS_TTFT_SLO_MS", 200.0)
        warp_size = _env_int("AS_WARP_SIZE", 32)
        self._log_war = os.environ.get("AS_LOG_WAR", "0") == "1"
        # AS_KV_ADMISSION=1: admit a waiting request only when the KV cache can
        # hold its whole lifetime on top of what running sequences still need.
        self._kv_admission = os.environ.get("AS_KV_ADMISSION", "0") == "1"
        # Adapter rotation for a working set smaller than the adapter pool
        # (fewer GPU LoRA slots than adapters): see _rotation_holdout().
        self._rotate_after_s = _env_float("AS_ROTATE_AFTER_S", 3.0)
        self._draining: set = set()
        self._req_metrics_path = os.environ.get("AS_REQ_METRICS_PATH") or None
        self._admit_grouped = os.environ.get("AS_ADMIT_GROUPED", "0") == "1"
        self._group_max_wait_s = _env_float("AS_GROUP_MAX_WAIT_S", 30.0)
        self._mode = os.environ.get("AS_MODE", "threshold")
        self._fixed_deadline = os.environ.get("AS_DEADLINE", "erlang") == "fixed"

        self._war_target = _env_float("AS_WAR_TARGET", 0.8)
        self._ewma_alpha = _env_float("AS_EWMA_ALPHA", 0.1)

        # Under PI control the buffer starts at AS_PI_TMAX_INIT_MS.
        buffer_tmax_ms = (
            _env_float("AS_PI_TMAX_INIT_MS", 18.0)
            if self._mode == "pi_adaptive"
            else tmax_ms
        )
        logger.info(
            "AlignmentAwareScheduler: mode=%s T_max=%.1f ms, TTFT_SLO=%.1f ms, "
            "W=%d, WAR*=%.2f, α=%.3f",
            self._mode, buffer_tmax_ms, ttft_slo_ms, warp_size,
            self._war_target, self._ewma_alpha,
        )
        self._schedule_call_count = 0

        # Adapters are registered as their first requests arrive.
        self._buffer = AlignmentBuffer(
            adapters=[],
            warp_size=warp_size,
            tmax_ms=buffer_tmax_ms,
            ttft_slo_ms=ttft_slo_ms,
        )
        self._ttft_slo_ms = ttft_slo_ms
        self._warp_size = warp_size
        self._batch_id = 0
        self._t0_ms = time.perf_counter() * 1000.0

        metrics_path = os.environ.get("AS_METRICS_PATH", "")
        if metrics_path:
            pathlib.Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
            self._metrics_file = open(metrics_path, "w")
            logger.info("AlignmentAwareScheduler: writing metrics -> %s", metrics_path)
        else:
            self._metrics_file = None

        # One estimator per scheduler; the scheduler lives in the engine process
        # only, so the LOCAL_RANK check is not needed here.
        self._estimator = ArrivalRateEstimator(
            alpha=self._ewma_alpha,
            enforce_rank0=False,
        )

        self._pi_ctrl = None
        if self._mode == "pi_adaptive":
            from adapter_slots.control.pi_controller import PIController
            kp = _env_float("AS_PI_KP", 0.01)
            ki = _env_float("AS_PI_KI", 0.001)
            tmax_init_s = _env_float("AS_PI_TMAX_INIT_MS", 18.0) / 1000.0
            # Dispatch is capped at T_SLO anyway, so T_max ranges up to T_SLO.
            tmax_max_s = ttft_slo_ms / 1000.0
            self._pi_ctrl = PIController(
                kp=kp, ki=ki, war_target=self._war_target,
                tmax_init=tmax_init_s,
                tmax_max=tmax_max_s,
            )
            logger.info(
                f"PI-adaptive: K_p={kp} K_i={ki} "
                f"T_max_init={tmax_init_s*1000:.1f}ms T_max_max={tmax_max_s*1000:.1f}ms "
                f"WAR*={self._war_target}")

        self._cash = CASHDispatcher(warp_size=warp_size)

        # PredictiveLFU: adapters predicted cold get a longer deadline (Erlang
        # mode) or a lower CASH score, so warm adapters fill warps first while
        # cold ones load.
        self._prefetch_policy = os.environ.get("AS_PREFETCH_POLICY", "none")
        self._tau_load_ms = _env_float("AS_TAU_LOAD_MS", 96.3)
        self._k_warm = _env_int("AS_K_WARM", 0)
        self._cold_boost = _env_float("AS_COLD_BOOST", 2.5)
        self._cache_mgr = None
        if self._prefetch_policy != "none" and self._k_warm > 0:
            from adapter_slots.prefetch.cache_manager import WarmCacheManager
            pol = self._prefetch_policy if self._prefetch_policy in (
                "lru", "predictive", "topk") else "lru"
            self._cache_mgr = WarmCacheManager(
                k_warm_max=self._k_warm,
                tau_load_ms=self._tau_load_ms,
                policy=pol,
            )
            logger.info(
                f"PredictiveLFU: policy={pol} K_warm={self._k_warm} "
                f"τ_load={self._tau_load_ms:.1f}ms cold_boost={self._cold_boost:.1f}x")

        # Minimum deferral for cold adapters: over PCIe tau_load is about one
        # decode iteration, so a cold adapter becomes eligible only tau_load
        # after it was first seen, once its weights have loaded (as in S-LoRA
        # and CaraServe).
        self._pcie_min_deferral_s = _env_float("AS_PCIE_MIN_DEFERRAL_S", 0.0)
        # First time each cold adapter was seen; cleared once it is warm.
        self._cold_first_seen: dict = {}

        # Sequences that have received at least one aligned decode token. Their
        # later tokens bypass the alignment buffer and run every tick; holding each
        # decode token for T_max again would multiply the wait by the output length.
        self._post_alignment_seqs: set = set()

        self._wgkp_dispatcher = None
        self._apt = None
        # adapter name -> (lora_int_id, lora_path), for L3 merges
        self._lora_info: dict = {}
        if self._mode == "wgkp":
            self._init_wgkp_components()

    def _adapter_id_of(self, seq_group: "SequenceGroup") -> str:
        """Adapter name of a sequence group (its --lora-modules name), or
        "__base__" without LoRA."""
        lr = getattr(seq_group, "lora_request", None)
        if lr is None:
            return "__base__"
        return getattr(lr, "lora_name", str(id(lr)))

    def _seq_group_id(self, seq_group: "SequenceGroup") -> int:
        """Return a stable integer ID for a SequenceGroup."""
        seqs = seq_group.get_seqs()
        if seqs:
            return seqs[0].seq_id
        return id(seq_group)

    def _schedule_running(self, budget, curr_loras, enable_chunking=False):
        """Schedule the running groups that can decode; hold back the ones
        still in the PREFILL stage for this tick.

        With asynchronous output processing, schedule() for tick N+1 runs
        before tick N's outputs update the sequences, so groups that finished
        prefill in tick N are still in self.running with stage PREFILL. In
        non-chunked mode vLLM requires _schedule_running to return no prefill
        groups, so those groups sit this tick out and go back to the front of
        self.running in their original order. Preempted sequences re-enter
        prefill often under KV pressure, so only the PREFILL groups are held,
        never the whole queue.
        """
        if _VLLM_AVAILABLE and self.running and not enable_chunking:
            pending = [sg for sg in self.running if sg.is_prefill()]
            if pending:
                if len(pending) == len(self.running):
                    return SchedulerRunningOutputs.create_empty()
                self.running = deque(sg for sg in self.running if not sg.is_prefill())
                try:
                    return super()._schedule_running(budget, curr_loras, enable_chunking)
                finally:
                    self.running.extendleft(reversed(pending))
        return super()._schedule_running(budget, curr_loras, enable_chunking)

    def free_finished_seq_groups(self) -> None:
        if self._req_metrics_path:
            rows = []
            for sg in self.running:
                if not sg.is_finished():
                    continue
                m = sg.metrics
                rows.append(json.dumps({
                    "id": sg.request_id, "lora": sg.lora_int_id,
                    "arrival": m.arrival_time, "first_scheduled": m.first_scheduled_time,
                    "first_token": m.first_token_time,
                    "out_tokens": sum(s.get_output_len() for s in sg.get_seqs())}))
            if rows:
                with open(self._req_metrics_path, "a") as f:
                    f.write("\n".join(rows) + "\n")
        super().free_finished_seq_groups()

    def _lifetime_blocks(self, seq, seq_group) -> int:
        """KV blocks a sequence holds once it has produced max_tokens."""
        bs = self.cache_config.block_size
        max_new = seq_group.sampling_params.max_tokens
        if max_new is None:
            max_new = self.scheduler_config.max_model_len - seq.get_prompt_len()
        return -(-(seq.get_prompt_len() + max_new) // bs)

    def _rotation_holdout(self, curr_loras) -> set:
        """Adapters whose waiting requests are held back this tick.

        With fewer GPU LoRA slots than adapters, vLLM admits a request only if
        its adapter is already in the running batch or a slot is free. An
        adapter that keeps receiving requests then never leaves the batch and
        the others starve. When a non-resident adapter's oldest request has
        waited AS_ROTATE_AFTER_S, the resident adapter with the fewest waiting
        requests stops admitting; once its running sequences finish its slot
        goes to the waiting adapter.
        """
        if self.lora_config is None or not curr_loras:
            self._draining.clear()
            return self._draining
        resident = set(curr_loras)
        self._draining &= resident          # a drained adapter has left the batch
        if len(resident) < self.lora_config.max_loras:
            return self._draining
        now = time.time()
        oldest, waiting_n = {}, {}
        for sg in self.waiting:
            lid = sg.lora_int_id
            if lid <= 0:
                continue
            waiting_n[lid] = waiting_n.get(lid, 0) + 1
            if lid not in resident:
                t = sg.metrics.arrival_time
                oldest[lid] = min(oldest.get(lid, t), t)
        if not oldest or self._draining:
            return self._draining
        if now - min(oldest.values()) >= self._rotate_after_s:
            self._draining.add(min(resident, key=lambda lid: waiting_n.get(lid, 0)))
        return self._draining

    def _schedule_prefills(self, budget, curr_loras, enable_chunking=False):
        """Prefill admission. With AS_ADMIT_GROUPED=1 the waiting queue is first
        ordered adapter by adapter (_group_waiting_by_adapter); then adapter
        rotation (_rotation_holdout) and the KV-aware check of
        _schedule_prefills_admit. The queue goes back to arrival order after."""
        if not _VLLM_AVAILABLE:
            return super()._schedule_prefills(budget, curr_loras, enable_chunking)
        if self._admit_grouped and len(self.waiting) > 1:
            self._group_waiting_by_adapter()
            try:
                return self._schedule_prefills_rotated(budget, curr_loras, enable_chunking)
            finally:
                self.waiting = deque(sorted(self.waiting, key=lambda sg: sg.metrics.arrival_time))
        return self._schedule_prefills_rotated(budget, curr_loras, enable_chunking)

    def _group_waiting_by_adapter(self) -> None:
        """Order the waiting queue adapter by adapter, longest queue first.

        Admission takes requests in queue order until the KV check stops it,
        so this fills each wave from the fewest adapters: every decode step
        then reads fewer adapters' LoRA weights in the grouped kernel. Requests
        that have waited AS_GROUP_MAX_WAIT_S keep arrival order at the front.
        """
        now = time.time()
        count: dict = {}
        for sg in self.waiting:
            count[sg.lora_int_id] = count.get(sg.lora_int_id, 0) + 1

        def key(sg):
            aged = now - sg.metrics.arrival_time >= self._group_max_wait_s
            return (0, sg.metrics.arrival_time) if aged else (
                1, -count[sg.lora_int_id], sg.lora_int_id, sg.metrics.arrival_time)
        self.waiting = deque(sorted(self.waiting, key=key))

    def _schedule_prefills_rotated(self, budget, curr_loras, enable_chunking=False):
        held = self._rotation_holdout(curr_loras)
        if held:
            # Held-back requests sit out this tick and keep their queue order.
            keep = deque(sg for sg in self.waiting if sg.lora_int_id not in held)
            out = deque(sg for sg in self.waiting if sg.lora_int_id in held)
            self.waiting = keep
            try:
                return self._schedule_prefills_admit(budget, curr_loras, enable_chunking)
            finally:
                self.waiting.extend(out)
                self.waiting = deque(sorted(self.waiting, key=lambda sg: sg.metrics.arrival_time))
        return self._schedule_prefills_admit(budget, curr_loras, enable_chunking)

    def _schedule_prefills_admit(self, budget, curr_loras, enable_chunking=False):
        """KV-aware admission (AS_KV_ADMISSION=1).

        vLLM admits a request when its prompt fits. Under KV pressure the
        running batch then outgrows the cache and vLLM evicts and recomputes
        sequences from scratch, every one of which repeats its prefill. Here
        a request is admitted only when the free blocks, minus the growth still
        owed to running sequences, cover its whole lifetime, so running
        sequences always have room to finish and nothing is recomputed.
        """
        if not self._kv_admission:
            return super()._schedule_prefills(budget, curr_loras, enable_chunking)
        bm = self.block_manager
        bs = self.cache_config.block_size
        owed = 0
        for sg in self.running:
            for seq in sg.get_seqs(status=SequenceStatus.RUNNING):
                owed += max(0, self._lifetime_blocks(seq, sg) - (-(-seq.get_len() // bs)))
        headroom = [bm.get_num_free_gpu_blocks() - owed]
        nothing_running = [not self.running]
        stock = bm.can_allocate

        def can_allocate(seq_group, num_lookahead_slots=0):
            status = stock(seq_group, num_lookahead_slots=num_lookahead_slots)
            if status != AllocStatus.OK:
                return status
            need = sum(self._lifetime_blocks(seq, seq_group)
                       for seq in seq_group.get_seqs(status=SequenceStatus.WAITING))
            if need > headroom[0] and not nothing_running[0]:
                return AllocStatus.LATER
            headroom[0] -= need
            nothing_running[0] = False  # always let one request in when idle
            return status

        bm.can_allocate = can_allocate
        try:
            return super()._schedule_prefills(budget, curr_loras, enable_chunking)
        finally:
            del bm.can_allocate

    def _schedule(self) -> "SchedulerOutputs":
        """vLLM's schedule() builds the step's metadata from the
        scheduled_seq_groups this returns, so the reordered batch is what the
        model runs."""
        self._schedule_call_count += 1
        if self._mode == "pi_adaptive":
            return self._schedule_pi_adaptive_inner()
        if self._mode == "wgkp":
            return self._schedule_wgkp_inner()
        if self._mode == "erlang":
            return self._schedule_erlang_inner()
        if self._mode == "cash":
            return self._schedule_cash_inner()
        return self._schedule_threshold_inner()

    def _schedule_threshold_inner(self) -> "SchedulerOutputs":
        """Threshold dispatch with one global T_max.

        Only a sequence's first decode token waits in the buffer. Once it has
        been dispatched, the sequence's later tokens run every tick in adapter
        order; holding each token for T_max again would multiply the wait by
        the output length.
        """
        sched_out: "SchedulerOutputs" = super()._schedule()

        if not _VLLM_AVAILABLE:
            return sched_out

        prefill_groups, decode_groups = self._split_groups(sched_out)

        if not decode_groups:
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

        # Already-aligned groups run every tick, sorted by adapter, so the whole
        # decode batch stays adapter-contiguous; first-time groups run when the
        # buffer dispatches them.
        post_alignment_groups.sort(key=lambda s: self._adapter_id_of(s.seq_group))
        sched_out.scheduled_seq_groups = (
            prefill_groups + post_alignment_groups + aligned_first_time
        )

        if self._log_war and decode_groups:
            self._log_war_stats(aligned_first_time, first_time_groups, aligned_pairs,
                                all_decode=post_alignment_groups + aligned_first_time)

        return sched_out

    def _schedule_pi_adaptive_inner(self) -> "SchedulerOutputs":
        """Threshold dispatch with T_max set each tick by the PI controller
        from the WAR of the tick's decode batch."""
        sched_out: "SchedulerOutputs" = super()._schedule()

        if not _VLLM_AVAILABLE:
            return sched_out

        prefill_groups, decode_groups = self._split_groups(sched_out)

        if not decode_groups:
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

        all_decode = post_alignment_groups + aligned_first_time
        if all_decode:
            int_ids = [self._adapter_id_to_int(self._adapter_id_of(ssg.seq_group))
                       for ssg in all_decode]
            war_obs = compute_war_from_ids(int_ids, warp_size=self._buffer.W)
            self._buffer.T_max = self._pi_ctrl.update(war_obs)

        # Drop finished sequences from the aligned set now and then.
        if self._schedule_call_count % 50_000 == 0:
            active_seq_ids = {seq.seq_id for sg in self.running for seq in sg.get_seqs()}
            self._post_alignment_seqs &= active_seq_ids

        if self._log_war and decode_groups:
            self._log_war_stats(aligned_first_time, first_time_groups, aligned_pairs,
                                all_decode=all_decode)

        return sched_out

    def _schedule_erlang_inner(self) -> "SchedulerOutputs":
        """Dispatch with the per-adapter Erlang deadlines T*(k)."""
        sched_out: "SchedulerOutputs" = super()._schedule()

        if not _VLLM_AVAILABLE:
            return sched_out

        prefill_groups, decode_groups = self._split_groups(sched_out)

        if not decode_groups:
            return sched_out

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
        # At most one rate update per adapter per tick: two arrivals with the
        # same timestamp would give a zero inter-arrival time.
        adapters_updated_this_tick: set = set()
        for ssg in first_time_groups:
            sg = ssg.seq_group
            adapter_id = self._adapter_id_of(sg)
            seq_id = self._seq_group_id(sg)
            seq_id_to_sg[seq_id] = ssg
            # Held sequences come back every tick; only new ones are arrivals.
            is_new_arrival = not self._buffer.is_buffered(seq_id)
            self._buffer.enqueue(adapter_id, seq_id)
            if is_new_arrival and adapter_id not in adapters_updated_this_tick:
                self._estimator.update(adapter_id, t_now)
                adapters_updated_this_tick.add(adapter_id)

        lambda_k_dict = self._estimator.get_all_rates()
        tmax_k = self._deadlines(lambda_k_dict)
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

        return sched_out


    def _schedule_cash_inner(self) -> "SchedulerOutputs":
        """Erlang deadlines T*(k) with queues visited in CASH order
        (dispatch/cash.py)."""
        sched_out: "SchedulerOutputs" = super()._schedule()

        if not _VLLM_AVAILABLE:
            return sched_out

        prefill_groups, decode_groups = self._split_groups(sched_out)

        if not decode_groups:
            return sched_out

        first_time_groups = []
        post_alignment_groups = []
        for ssg in decode_groups:
            seq_id = self._seq_group_id(ssg.seq_group)
            if seq_id in self._post_alignment_seqs:
                post_alignment_groups.append(ssg)
            else:
                first_time_groups.append(ssg)

        t_now = time.monotonic()
        adapters_updated_this_tick: set = set()
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

        lambda_k_dict = self._estimator.get_all_rates()
        tmax_k = self._deadlines(lambda_k_dict)
        self._update_prefetch_cache(lambda_k_dict, adapters_updated_this_tick)

        ranked_adapters = self._cash_order(tmax_k, lambda_k_dict)

        aligned_pairs = self._buffer.form_batch_cash(
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

        return sched_out

    def _init_wgkp_components(self) -> None:
        """Build the WGKP dispatcher and the promotion threshold (AS_MODE=wgkp)."""
        from adapter_slots.kernel.apt import AdaptivePromoThreshold
        from adapter_slots.kernel.wgkp_dispatcher import WGKPDispatcher

        self._wgkp_dispatcher = WGKPDispatcher()
        self._k_hot = _env_int("AS_MWC_K_HOT", 5)
        wgkp_hw_profile = os.environ.get("AS_WGKP_HW_PROFILE", "")
        if os.environ.get("AS_WGKP_APT", "0") == "1":
            self._apt = AdaptivePromoThreshold(hw_profile_path=wgkp_hw_profile)
        self._wgkp_static_threshold = _env_int("AS_WGKP_THRESHOLD", 8)

        logger.info(
            f"WGKP init: n*={self._wgkp_static_threshold} AS_MWC_K_HOT={self._k_hot} "
            f"APT={'on' if self._apt else 'off'} "
            f"grouped_kernels={os.environ.get('AS_SEGGEMM', '0')}")

    def _request_merges(self, mwc, pending: dict, lambda_k_dict: dict, n_star: int) -> None:
        """Alignment-aware prefetch: once the queue of one of the K_hot hottest
        adapters holds half of n* tokens, start merging its weights, so they are
        ready by the time the segment reaches n*."""
        hottest = sorted(lambda_k_dict, key=lambda a: lambda_k_dict[a], reverse=True)[:self._k_hot]
        for adapter_id in hottest:
            info = self._lora_info.get(adapter_id)
            if info is not None and pending.get(adapter_id, 0) >= math.ceil(n_star / 2):
                mwc.request(*info)

    def _current_n_star(self) -> int:
        """Return the current promotion threshold n*, from APT or static setting."""
        if self._apt is not None:
            return self._apt.current_threshold()
        return self._wgkp_static_threshold

    def _schedule_wgkp_inner(self) -> "SchedulerOutputs":
        """CASH holdback dispatch (form_batch_wgkp) in CASH order, with the
        dispatched queues grouped into WGKP segments and their promotion flags
        attached to the scheduler output."""
        sched_out: "SchedulerOutputs" = super()._schedule()

        if not _VLLM_AVAILABLE:
            return sched_out

        prefill_groups, decode_groups = self._split_groups(sched_out)

        if not decode_groups:
            return sched_out

        # Forget sequences that have left self.running. O(|running|) per tick.
        active_ids: set = {
            seq.seq_id
            for sg in self.running
            for seq in sg.get_seqs()
        }
        stale = self._post_alignment_seqs - active_ids
        self._post_alignment_seqs -= stale
        for sid in stale:
            self._buffer.release(sid)

        first_time_groups = []
        post_alignment_groups = []
        for ssg in decode_groups:
            seq_id = self._seq_group_id(ssg.seq_group)
            if seq_id in self._post_alignment_seqs:
                post_alignment_groups.append(ssg)
            else:
                first_time_groups.append(ssg)

        t_now = time.monotonic()
        adapters_updated_this_tick: set = set()
        seq_id_to_sg: dict = {}
        for ssg in first_time_groups:
            sg = ssg.seq_group
            adapter_id = self._adapter_id_of(sg)
            seq_id = self._seq_group_id(sg)
            seq_id_to_sg[seq_id] = ssg
            lr = sg.lora_request
            if lr is not None and adapter_id not in self._lora_info:
                self._lora_info[adapter_id] = (lr.lora_int_id, lr.lora_path)
            is_new_arrival = not self._buffer.is_buffered(seq_id)
            self._buffer.enqueue(adapter_id, seq_id)
            if is_new_arrival and adapter_id not in adapters_updated_this_tick:
                self._estimator.update(adapter_id, t_now)
                adapters_updated_this_tick.add(adapter_id)

        lambda_k_dict = self._estimator.get_all_rates()
        tmax_k = self._deadlines(lambda_k_dict)
        self._update_prefetch_cache(lambda_k_dict, adapters_updated_this_tick)

        pending = self._buffer.pending_count()
        ranked_adapters = self._cash_order(tmax_k, lambda_k_dict)

        mean_q_depth = sum(pending.values()) / max(len(pending), 1)
        if self._apt is not None:
            self._apt.update(mean_q_depth)

        n_star = self._current_n_star()
        # L3: request merges and flag the adapters whose merged weights are resident.
        merged_ids: set = set()
        mwc = get_mwc()
        if mwc is not None:
            self._request_merges(mwc, pending, lambda_k_dict, n_star)
            merged_ids = {a for a, (lid, _) in self._lora_info.items() if mwc.is_merged(lid)}

        # The budget is this tick's decode capacity, so every buffered token,
        # including expired ones, can leave the buffer this tick.
        raw_batch = self._buffer.form_batch_wgkp(
            ranked_adapters=ranked_adapters,
            tmax_k=tmax_k,
            n_star=n_star,
            merged_adapter_ids=merged_ids,
            max_tokens=len(decode_groups),
        )

        segments = []
        if self._wgkp_dispatcher is not None and raw_batch:
            try:
                segments = self._wgkp_dispatcher.segment_and_promote(raw_batch)
            except ValueError as e:
                logger.warning(f"WGKP segment_and_promote error: {e}")
                segments = []

        aligned_first_time = []
        dispatched_ids = set()
        for adapter_id, seq_id, is_promoted in raw_batch:
            ssg = seq_id_to_sg.get(seq_id)
            if ssg is not None:
                aligned_first_time.append(ssg)
                self._post_alignment_seqs.add(seq_id)
                dispatched_ids.add(seq_id)
                self._buffer.release(seq_id)

        # Held sequences are released early only when the step would otherwise
        # be empty. An empty step slows vLLM's engine loop, which delays the
        # deadline check that would release them (at cold start every sequence
        # is first-time). Whenever other decode work keeps the loop running,
        # which is nearly always in steady state, the holdback rules apply
        # unchanged.
        would_be_empty = not (prefill_groups or post_alignment_groups or aligned_first_time)
        if would_be_empty:
            drained = self._drain_buffer_remaining(first_time_groups, dispatched_ids)
            for ssg in drained:
                seq_id = self._seq_group_id(ssg.seq_group)
                self._post_alignment_seqs.add(seq_id)
                self._buffer.release(seq_id)
        else:
            drained = []

        post_alignment_groups.sort(key=lambda s: self._adapter_id_of(s.seq_group))
        sched_out.scheduled_seq_groups = (
            prefill_groups + post_alignment_groups + aligned_first_time + drained
        )

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

        if self._log_war and decode_groups:
            raw_pairs = [(a, s) for a, s, _ in raw_batch]
            self._log_war_stats_wgkp(
                aligned_first_time, first_time_groups, raw_pairs,
                all_decode=post_alignment_groups + aligned_first_time,
                n_star=n_star,
                segments=segments,
            )

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
        """Write the decode batch's WAR, GWAR(n*) and token order to the batch log."""
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

    def _deadlines(self, lambda_k_dict: dict) -> dict:
        """Per-adapter deadlines: T*(k) from the Erlang inversion, or the fixed
        T_max for every adapter with AS_DEADLINE=fixed."""
        if self._fixed_deadline:
            return {k: self._buffer.T_max for k in set(self._buffer.queues) | set(lambda_k_dict)}
        return compute_tmax_erlang_batch(
            warp_size=self._warp_size,
            lambda_k_dict=lambda_k_dict,
            war_target=self._war_target,
            ttft_slo_ms=self._ttft_slo_ms,
        )

    def _cash_order(self, tmax_k: dict, lambda_k_dict: dict) -> list:
        """Adapters in decreasing CASH score I_k = lambda_k s_k P_k(s_k), each
        queue's fill probability taken against its deadline min(T_k, T_SLO)."""
        pending = self._buffer.pending_count()
        now = time.perf_counter()
        slo = self._ttft_slo_ms / 1000.0
        ages = {k: now - t for k, t in self._buffer.enqueue_time.items() if t is not None}
        deadlines = {k: min(tmax_k.get(k, slo), slo) for k in pending}
        scores = self._cash.compute_indices(pending, ages, lambda_k_dict, deadlines)
        return self._cash.rank(self._penalize_cold(scores))

    def _update_prefetch_cache(
        self,
        lambda_k_dict: dict,
        new_arrivals: set,
    ) -> None:
        """Record this tick's new arrivals in the warm-set tracker (PredictiveLFU).
        Active when AS_PREFETCH_POLICY is set and AS_K_WARM > 0."""
        if self._cache_mgr is None:
            return
        for adapter_id in new_arrivals:
            self._cache_mgr.request(
                adapter_id,
                rate_estimates=lambda_k_dict,
                t_now=time.perf_counter(),
            )

    def _penalize_cold(self, scores: dict) -> dict:
        """Lower the CASH score of adapters predicted cold, so warm adapters are
        visited first.

        With AS_PCIE_MIN_DEFERRAL_S > 0 a cold adapter scores 0 at the tick it is
        first seen, then score / cold_boost until the deferral window has passed;
        without it, score / cold_boost while it is cold. Warm adapters keep their
        score. Active when AS_PREFETCH_POLICY is set and AS_K_WARM > 0.
        """
        if self._cache_mgr is None:
            return scores
        warm_set = self._cache_mgr.warm_set
        t_now = time.perf_counter()
        penalized: dict = {}
        for adapter_id, frac in scores.items():
            if adapter_id in warm_set:
                self._cold_first_seen.pop(adapter_id, None)
                penalized[adapter_id] = frac
            elif self._pcie_min_deferral_s > 0.0:
                first_seen = self._cold_first_seen.get(adapter_id)
                if first_seen is None:
                    self._cold_first_seen[adapter_id] = t_now
                    penalized[adapter_id] = 0.0
                elif (t_now - first_seen) < self._pcie_min_deferral_s:
                    penalized[adapter_id] = frac / self._cold_boost
                else:
                    penalized[adapter_id] = frac
            else:
                penalized[adapter_id] = frac / self._cold_boost
        return penalized

    def _apply_cold_start_boost(
        self,
        tmax_k: dict,
        lambda_k_dict: dict,
        new_arrivals: set,
    ) -> dict:
        """Erlang mode: lengthen the deadline of adapters predicted cold,
        T*(k) -> min(T*(k) * cold_boost, T_SLO). Warm adapters are unchanged."""
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
        """Split scheduled_seq_groups into (prefill_groups, decode_groups).

        A group vLLM scheduled as a prefill this tick is always a prefill here,
        even when it already holds output tokens: a sequence preempted by
        recompute re-enters through the prefill path with prompt + outputs.
        vLLM has already moved it into self.running, so deferring it in the
        alignment buffer would leave it there in PREFILL stage, never executed.
        """
        prefill_groups = []
        decode_groups = []
        for sg in sched_out.scheduled_seq_groups:
            seqs = sg.seq_group.get_seqs(status=SequenceStatus.RUNNING)
            is_decode = (not sg.seq_group.is_prefill()) and (
                all(s.data.get_len() > s.data.get_prompt_len() for s in seqs)
                if seqs else False)
            if is_decode:
                decode_groups.append(sg)
            else:
                prefill_groups.append(sg)
        return prefill_groups, decode_groups

    def _enqueue_decode_groups(self, decode_groups: list) -> dict:
        """Enqueue all decode seq groups into the alignment buffer.

        Returns seq_id -> SchedulerSeqGroup mapping.
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
        """Remove the sequences the buffer did not dispatch from their queues
        and return their groups, to run in this step."""
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
        """Integer id of an adapter name ("adapter_7" -> 7) for WAR and GWAR."""
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
        """Write the decode batch's WAR and token order to the batch log.

        WAR is computed over the whole decode batch (all_decode) when given.
        """
        batch_groups = all_decode if all_decode is not None else aligned_decode
        adapter_ids = [
            self._adapter_id_of(ssg.seq_group) for ssg in batch_groups
        ]
        int_ids = [self._adapter_id_to_int(aid) for aid in adapter_ids]
        war = compute_war_from_ids(int_ids, warp_size=self._buffer.W)

        if self._metrics_file is not None:
            t_dispatch_perf = time.perf_counter()
            dispatch_time_ms = t_dispatch_perf * 1000.0 - self._t0_ms
            n_adapters = len(set(adapter_ids))
            n_tokens = len(adapter_ids)
            # Per-token records; a token without an enqueue record (already
            # released) is stamped with the dispatch time.
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

    @property
    def buffer(self) -> AlignmentBuffer:
        """The scheduler's AlignmentBuffer."""
        return self._buffer
