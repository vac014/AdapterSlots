"""
test_lora_kernel_fusion.py -- standalone GPU correctness test for
FusedPackedLoRAKernel (adapter_slots/kernel/fused_packed_lora_kernel.py)
against vLLM's own add_lora_packed_nslice loop -- the ground truth this
kernel is supposed to match exactly. No server, no model load; pure kernel
math comparison, fast to run before touching the live serving path.

The bug this guards against: reusing slice 0's lora_b tensor strides for slices
1/2 is wrong whenever a packed layer's slices have different output widths (e.g.
Q vs K/V under GQA). Run this whenever fused_packed_lora_kernel.py changes.

Usage:
    python scripts/test_lora_kernel_fusion.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from adapter_slots.kernel.fused_packed_lora_kernel import FusedPackedLoRAKernel

torch.manual_seed(0)
DEVICE = "cuda"


def make_case(num_slices, batch, K, R, Ns, num_loras, frac_no_lora=0.2):
    x = torch.randn(batch, K, dtype=torch.float16, device=DEVICE)
    lora_a = [torch.randn(num_loras, 1, R, K, dtype=torch.float16, device=DEVICE) * 0.05
              for _ in range(num_slices)]
    lora_b = [torch.randn(num_loras, 1, Ns[s], R, dtype=torch.float16, device=DEVICE) * 0.05
              for s in range(num_slices)]
    indices = torch.randint(0, num_loras, (batch,), dtype=torch.long, device=DEVICE)
    no_lora_mask = torch.rand(batch, device=DEVICE) < frac_no_lora
    indices = torch.where(no_lora_mask, torch.full_like(indices, -1), indices)
    return x, lora_a, lora_b, indices


def reference_via_vllm_loop(x, lora_a, lora_b, scale, output_slices, indices):
    """Exactly the loop add_lora_packed_nslice runs, called once per slice."""
    total_n = sum(output_slices)
    y = torch.zeros(x.shape[0], total_n, dtype=torch.float16, device=x.device)

    class FakeWrapper:
        is_prefill = False
        token_lora_indices = indices

        def shrink_decode(self, yy, xx, w, sc):
            from vllm.lora.ops.bgmv_shrink import bgmv_shrink
            bgmv_shrink(xx, w, yy, self.token_lora_indices, sc)

        def expand_slice_decode(self, yy, xx, w, y_offset, y_slice_size, add_input):
            from vllm.lora.ops.bgmv_expand_slice import bgmv_expand_slice
            bgmv_expand_slice(xx, w, yy, self.token_lora_indices, y_offset, y_slice_size, add_input)

        def add_shrink(self, yy, xx, w, sc):
            self.shrink_decode(yy, xx, w, sc)

        def add_expand_slice(self, yy, xx, w, y_offset, y_slice_size, add_input=True):
            self.expand_slice_decode(yy, xx, w, y_offset, y_slice_size, add_input)

        def add_lora(self, yy, xx, wa, wb, sc, y_offset=None, y_slice_size=None, *, buffer=None):
            r = wb.size(-1)
            if buffer is None:
                buffer = torch.zeros((xx.size(0), r), dtype=torch.float32, device=xx.device)
            self.add_shrink(buffer, xx, wa, sc)
            self.add_expand_slice(yy, buffer, wb, y_offset, y_slice_size, add_input=True)

        def add_lora_packed_nslice(self, yy, xx, la, lb, sc, out_slices):
            offset_left = 0
            for s in range(len(out_slices)):
                self.add_lora(yy, xx, la[s], lb[s], sc, offset_left, out_slices[s])
                offset_left += out_slices[s]

    fw = FakeWrapper()
    fw.add_lora_packed_nslice(y, x, lora_a, lora_b, scale, output_slices)
    return y


def run_case(name, num_slices, batch, K, R, Ns, num_loras) -> bool:
    x, lora_a, lora_b, indices = make_case(num_slices, batch, K, R, Ns, num_loras)
    output_slices = tuple(Ns)
    scale = 1.0

    y_ref = reference_via_vllm_loop(x, lora_a, lora_b, scale, output_slices, indices)

    y_fused = torch.zeros_like(y_ref)
    kernel = FusedPackedLoRAKernel()
    kernel.apply_packed_nslice(y_fused, x, lora_a, lora_b, scale, output_slices, indices)

    diff = (y_ref.float() - y_fused.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    ok = max_diff < 1e-2
    print(f"[{name}] num_slices={num_slices} batch={batch} K={K} R={R} Ns={Ns} "
          f"max_diff={max_diff:.5f} mean_diff={mean_diff:.6f} {'PASS' if ok else 'FAIL'}")
    if not ok:
        worst = diff.argmax()
        row, col = worst // y_ref.shape[1], worst % y_ref.shape[1]
        print(f"   worst at row={row.item()} col={col.item()} "
              f"ref={y_ref[row, col].item():.5f} fused={y_fused[row, col].item():.5f} "
              f"idx={indices[row].item()}")
    return ok


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available -- this kernel requires a GPU.", file=sys.stderr)
        sys.exit(1)

    results = [
        run_case("QKV-like", 3, 23, 5120, 32, [4096, 1024, 1024], 16),
        run_case("gate_up-like", 2, 23, 5120, 32, [13824, 13824], 16),
        run_case("QKV-small-batch", 3, 5, 4096, 16, [4096, 4096, 4096], 8),
        run_case("QKV-all-no-lora", 3, 7, 4096, 16, [2048, 512, 512], 4),
        run_case("gate_up-K2", 2, 16, 5120, 64, [13824, 13824], 2),
    ]

    print("\n=== OVERALL:", "PASS" if all(results) else "FAIL", "===")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
