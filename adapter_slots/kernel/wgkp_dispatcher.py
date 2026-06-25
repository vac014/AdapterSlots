"""
wgkp_dispatcher.py -- WGKPDispatcher and SegmentDescriptor (kernel_promotion Phase 5).

WGKPDispatcher performs an O(N) single linear scan over a pre-sorted batch
(AlignmentBuffer invariant from kernel_decomposition/4) to produce a list of SegmentDescriptors.
Each descriptor covers a contiguous run of tokens for the same adapter and records
whether that segment is eligible for Level-3 GEMM promotion.

Promotion atomicity invariant:
    Either ALL tokens in a contiguous adapter run are promoted, or NONE are.
    Partial promotion is rejected (raises ValueError) since it would produce
    mismatched output tensors when the ModelRunner routes some tokens through
    cuBLAS GEMM and others through the fused Triton kernel.

The is_promoted flag in each raw_batch tuple is set by form_batch_wgkp() in
buffer.py (Algorithm CASH). This dispatcher validates atomicity and groups
contiguous runs into SegmentDescriptors for the ModelRunner.
"""

from dataclasses import dataclass, field
from typing import List, Set, Tuple


@dataclass
class SegmentDescriptor:
    """Describes a contiguous same-adapter token segment in a dispatch batch.

    Attributes:
        adapter_id:    String identifier of the adapter for this segment.
        seq_ids:       Ordered list of sequence IDs in this segment.
        is_promoted:   True → Level-3 cuBLAS GEMM with merged weight W_k.
                       False → Level-2 Fused Triton kernel or Level-1 SGMV.
        segment_size:  Number of tokens in this segment (== len(seq_ids)).
    """
    adapter_id: str
    seq_ids: List[int] = field(default_factory=list)
    is_promoted: bool = False
    segment_size: int = 0

    def __post_init__(self) -> None:
        if self.segment_size == 0:
            self.segment_size = len(self.seq_ids)


class WGKPDispatcher:
    """Convert a pre-sorted raw batch into a list of SegmentDescriptors.

    The raw batch is produced by AlignmentBuffer.form_batch_wgkp() and has type
    List[Tuple[str, int, bool]] = (adapter_id, seq_id, is_promoted).

    Since AlignmentBuffer always sorts tokens by adapter (kernel_decomposition/4 invariant),
    this is a run-length encoding of the adapter_id sequence. The scan is O(N).

    Usage:
        dispatcher = WGKPDispatcher()
        segments = dispatcher.segment_and_promote(raw_batch)
        for seg in segments:
            if seg.is_promoted:
                # Level-3: install merged weights, run cuBLAS GEMM
            else:
                # Level-2: run fused Triton kernel (or Level-1 SGMV fallback)
    """

    def __init__(self) -> None:
        self._total_batches = 0
        self._total_promoted_tokens = 0
        self._total_tokens = 0

    def segment_and_promote(
        self,
        raw_batch: List[Tuple[str, int, bool]],
    ) -> List[SegmentDescriptor]:
        """Group consecutive (adapter_id, seq_id, is_promoted) triples into SegmentDescriptors.

        Validates promotion atomicity: every token in a contiguous adapter run must
        share the same is_promoted value. Raises ValueError if mixed (should never
        happen given correct CASH implementation in form_batch_wgkp()).

        Args:
            raw_batch: List of (adapter_id, seq_id, is_promoted) triples from
                       AlignmentBuffer.form_batch_wgkp(). Must be adapter-sorted
                       (each adapter's tokens are contiguous).

        Returns:
            List of SegmentDescriptor objects, one per contiguous adapter run.
            Preserves the original dispatch order.

        Raises:
            ValueError: If a segment contains mixed is_promoted values (atomicity
                        violation). This indicates a bug in form_batch_wgkp().
        """
        if not raw_batch:
            return []

        self._total_batches += 1
        self._total_tokens += len(raw_batch)

        segments: List[SegmentDescriptor] = []
        i = 0
        n = len(raw_batch)

        while i < n:
            current_adapter, current_seq, current_promo = raw_batch[i]
            seg_seq_ids = [current_seq]
            j = i + 1

            while j < n and raw_batch[j][0] == current_adapter:
                next_adapter, next_seq, next_promo = raw_batch[j]
                if next_promo != current_promo:
                    raise ValueError(
                        f"WGKP atomicity violation: adapter '{current_adapter}' "
                        f"segment has mixed is_promoted values at index {j}. "
                        f"Expected all {current_promo}, got {next_promo}. "
                        f"This indicates a bug in form_batch_wgkp()."
                    )
                seg_seq_ids.append(next_seq)
                j += 1

            seg = SegmentDescriptor(
                adapter_id=current_adapter,
                seq_ids=seg_seq_ids,
                is_promoted=current_promo,
                segment_size=len(seg_seq_ids),
            )
            segments.append(seg)
            if current_promo:
                self._total_promoted_tokens += len(seg_seq_ids)
            i = j

        return segments

    def promotion_fraction(self) -> float:
        """Return the running fraction of tokens dispatched at Level-3 (promoted)."""
        if self._total_tokens == 0:
            return 0.0
        return self._total_promoted_tokens / self._total_tokens

    def stats(self) -> dict:
        """Return cumulative dispatch statistics."""
        return {
            "total_batches": self._total_batches,
            "total_tokens": self._total_tokens,
            "total_promoted_tokens": self._total_promoted_tokens,
            "promotion_fraction": self.promotion_fraction(),
        }

    def reset_stats(self) -> None:
        """Reset cumulative stats."""
        self._total_batches = 0
        self._total_promoted_tokens = 0
        self._total_tokens = 0
