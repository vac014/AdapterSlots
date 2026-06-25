"""
wgkp_dispatcher.py -- WGKPDispatcher and SegmentDescriptor.

WGKPDispatcher scans the batch that form_batch_wgkp() dispatched once, O(N),
and returns one SegmentDescriptor per run of consecutive same-adapter tokens,
with the run's L3 promotion flag.

All tokens of a run are promoted or none are; a run with mixed flags raises
ValueError, since its tokens would need two different kernels.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class SegmentDescriptor:
    """Describes a contiguous same-adapter token segment in a dispatch batch.

    Attributes:
        adapter_id:    String identifier of the adapter for this segment.
        seq_ids:       Ordered list of sequence IDs in this segment.
        is_promoted:   True: L3-eligible (merged weight W_k resident in the MWC).
                       False: L2 or L1.
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
    """Run-length encoding of a dispatched batch into SegmentDescriptors.

    The batch is a list of (adapter_id, seq_id, is_promoted) triples from
    AlignmentBuffer.form_batch_wgkp(), adapter-contiguous.
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

        Every token of a run must carry the same is_promoted value.

        Args:
            raw_batch: List of (adapter_id, seq_id, is_promoted) triples from
                       AlignmentBuffer.form_batch_wgkp(). Must be adapter-sorted
                       (each adapter's tokens are contiguous).

        Returns:
            List of SegmentDescriptor objects, one per contiguous adapter run.
            Preserves the original dispatch order.

        Raises:
            ValueError: If a run has mixed is_promoted values.
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
                        f"Expected all {current_promo}, got {next_promo}."
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
        """Fraction of dispatched tokens flagged L3-eligible so far."""
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
