"""Block-size sweep for the tile-scheduled grouped LoRA kernels at LLaMA-13B
decode shapes (adapter-sorted tokens, Zipf-0.9 segments), each kernel timed
alone under a CUDA graph. segmented_punica_wrapper.py uses the best K=16
configurations (_tiled_shrink_cfg, _tiled_expand_cfg).

Settings (environment): RANK (default 32), OUT, GPU. Writes
<OUT>/kernel_tiles_r<RANK>.json with the best and second-best configuration
per (kernel, K, B, width) and vLLM's BGMV time for the same shape.
"""
import itertools
import json
import os

from common import out_dir, setup

setup()
import numpy as np  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402
from adapter_slots.kernel.tiled_segment_gemm import (_tiled_expand_slice_kernel,  # noqa: E402
                                                     _tiled_shrink_kernel, build_tile_map)
from vllm.lora.ops.bgmv_expand_slice import _bgmv_expand_slice  # noqa: E402
from vllm.lora.ops.bgmv_shrink import _bgmv_shrink  # noqa: E402

dev, dt = "cuda", torch.float16
RANK = int(os.environ.get("RANK", "32"))
SLOTS = 17
OUT = out_dir("kernel")


def segments(B, K, BM):
    w = 1.0 / np.arange(1, K + 1) ** 0.9
    w /= w.sum()
    ids = np.sort(np.random.default_rng(0).choice(K, size=B, p=w))
    uniq, starts, lens = np.unique(ids, return_index=True, return_counts=True)
    st = torch.zeros(SLOTS, dtype=torch.long, device=dev)
    ln = torch.zeros_like(st)
    lo = torch.full((SLOTS,), -1, dtype=torch.long, device=dev)
    st[:len(uniq)] = torch.tensor(starts, device=dev)
    ln[:len(uniq)] = torch.tensor(lens, device=dev)
    lo[:len(uniq)] = torch.tensor(uniq + 1, device=dev)
    need = (8192 + BM - 1) // BM + SLOTS + 1
    tseg = torch.full((need,), -1, dtype=torch.long, device=dev)
    tm = torch.zeros(need, dtype=torch.long, device=dev)
    build_tile_map(ln, BM, need, tseg, tm)
    cap = (B + BM - 1) // BM + SLOTS + 1
    return st, ln, lo, tseg, tm, cap, torch.tensor(ids + 1, device=dev, dtype=torch.long)


def graph_time_us(fn, iters=30):
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(20):
                fn()
        g.replay()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(iters):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        return e0.elapsed_time(e1) / iters / 20 * 1e3
    except Exception:
        return float("inf")


rows = []
for K, B in [(16, 100), (16, 160), (8, 100), (4, 100)]:
    for kin in (5120, 13824):
        x = torch.randn(B, kin, device=dev, dtype=dt)
        w = torch.randn(SLOTS, 1, RANK, kin, device=dev, dtype=dt) * 0.01
        y = torch.zeros(B, RANK, device=dev, dtype=torch.float32)
        best = []
        for BM in (16, 32):
            st, ln, lo, tseg, tm, cap, idx = segments(B, K, BM)
            for BN, BK, SK, nw, ns in itertools.product((16, 32), (32, 64, 128), (4, 8, 16),
                                                        (2, 4), (1, 2)):
                if BN > RANK:
                    continue

                def f():
                    _tiled_shrink_kernel[(cap, triton.cdiv(RANK, BN), SK)](
                        x, w, y, RANK, kin, st, ln, lo, tseg, tm, 1.0,
                        x.stride(0), x.stride(1), w.stride(0), w.stride(2), w.stride(3),
                        y.stride(0), y.stride(1), BM, BN, BK, kin % (BK * SK) == 0, SK,
                        num_warps=nw, num_stages=ns)
                best.append((graph_time_us(f), dict(BM=BM, BN=BN, BK=BK, SPLIT_K=SK,
                                                    num_warps=nw, num_stages=ns)))
        best.sort(key=lambda b: b[0])
        idx = segments(B, K, 32)[-1]
        bgmv = graph_time_us(lambda: _bgmv_shrink(x, w, y, idx, 1.0))
        rows.append(dict(kernel="shrink", K=K, B=B, width=kin, bgmv_us=bgmv,
                         best_us=best[0][0], best=best[0][1], second_us=best[1][0],
                         second=best[1][1]))
        print(f"shrink K={K} B={B} in={kin} bgmv={bgmv:.1f}us best={best[0]}", flush=True)
    for nout in (5120, 13824):
        x = torch.randn(B, RANK, device=dev, dtype=torch.float32)
        w = torch.randn(SLOTS, 1, nout, RANK, device=dev, dtype=dt) * 0.01
        y = torch.zeros(B, nout, device=dev, dtype=dt)
        best = []
        for BM in (16, 32):
            st, ln, lo, tseg, tm, cap, idx = segments(B, K, BM)
            for BN, BK, nw, ns in itertools.product((64, 128, 256), (16, 32), (2, 4, 8), (1, 2)):
                if BK > RANK:
                    continue

                def f():
                    _tiled_expand_slice_kernel[(cap, triton.cdiv(nout, BN))](
                        x, w, y, nout, RANK, st, ln, lo, tseg, tm,
                        x.stride(0), x.stride(1), w.stride(0), w.stride(2), w.stride(3),
                        y.stride(0), y.stride(1), 0, BM, BN, BK, RANK % BK == 0, True, True,
                        num_warps=nw, num_stages=ns)
                best.append((graph_time_us(f), dict(BM=BM, BN=BN, BK=BK, num_warps=nw,
                                                    num_stages=ns)))
        best.sort(key=lambda b: b[0])
        idx = segments(B, K, 32)[-1]
        bgmv = graph_time_us(lambda: _bgmv_expand_slice(x, w, y, idx, 0, nout, True))
        rows.append(dict(kernel="expand", K=K, B=B, width=nout, bgmv_us=bgmv,
                         best_us=best[0][0], best=best[0][1], second_us=best[1][0],
                         second=best[1][1]))
        print(f"expand K={K} B={B} out={nout} bgmv={bgmv:.1f}us best={best[0]}", flush=True)

path = f"{OUT}/kernel_tiles_r{RANK}.json"
json.dump({"rank": RANK, "unit": "us per kernel", "rows": rows}, open(path, "w"), indent=1)
print(f"wrote {path}", flush=True)
