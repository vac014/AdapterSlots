"""LoRA cost of one LLaMA-13B decoder layer per decode step, measured through the
real PunicaWrapper classes under a CUDA graph.

For each adapter count K and batch size B (Zipf-0.9 adapter mix) it times:
  stock_sorted  vLLM's BGMV path on an adapter-sorted batch
  stock_mixed   vLLM's BGMV path on the same tokens in arrival (mixed) order
  as            the AdapterSlots grouped path (AS_SEGGEMM mode) on the sorted batch
and checks the grouped output against vLLM's on the same batch.

QKV and gate_up go through add_lora_packed_nslice, o and down through add_lora,
so one layer covers all seven projections. Times are per layer in microseconds;
multiply by 40 for a whole 13B decode step.

Settings (environment): RANK (default 32), SHAPE (13b, or 13b_tp2 for one TP=2
shard), AS_SEGGEMM (default 3), OUT, GPU.
"""
import json
import os

from common import out_dir, setup

setup()
os.environ.setdefault("AS_SEGGEMM", "3")
os.environ.setdefault("AS_SEGGEMM_MIN_TOKENS", "64")
import numpy as np  # noqa: E402
import torch  # noqa: E402
from vllm.lora.punica import PunicaWrapper  # noqa: E402
import adapter_slots.kernel.segmented_punica_wrapper as SPW  # noqa: E402

dev, dt = "cuda", torch.float16
RANK = int(os.environ.get("RANK", "32"))
SHAPE = os.environ.get("SHAPE", "13b")
H, I = 5120, 13824
HS, IS = (2560, 6912) if SHAPE == "13b_tp2" else (H, I)   # column/row-parallel widths
SLOTS = 17
OUT = out_dir("kernel")


def mk(n, k):
    return torch.randn(SLOTS, 1, n, k, device=dev, dtype=dt) * 0.01


qkvA, qkvB = tuple(mk(RANK, H) for _ in range(3)), tuple(mk(HS, RANK) for _ in range(3))
guA, guB = tuple(mk(RANK, H) for _ in range(2)), tuple(mk(IS, RANK) for _ in range(2))
oA, oB = mk(RANK, HS), mk(H, RANK)
dA, dB = mk(RANK, IS), mk(H, RANK)


def wrapper(grouped):
    pw = PunicaWrapper(8192, 256, dev)
    if grouped:
        pw.__class__ = SPW.SegmentedPunicaWrapper
        pw._seg_cap = -1
    return pw


def prepare(pw, ids):
    n = len(ids)
    pw._token_lora_indices[:n].copy_(torch.tensor(ids, device=dev))
    pw.indices_len[:] = [n] * 5
    pw.is_prefill = False
    if isinstance(pw, SPW.SegmentedPunicaWrapper):
        pw._refresh_segments(SLOTS)
        if SPW._MODE >= 2:
            pw._refresh_tiles(SLOTS)


def layer(pw, xh, xhs, xi, ys):
    pw.add_lora_packed_nslice(ys[0], xh, qkvA, qkvB, 1.0, (HS, HS, HS))
    pw.add_lora(ys[2], xhs, oA, oB, 1.0)
    pw.add_lora_packed_nslice(ys[1], xh, guA, guB, 1.0, (IS, IS))
    pw.add_lora(ys[3], xi, dA, dB, 1.0)


def graph_time_us(fn, iters=30):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(10):
            fn()
    g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(iters):
        g.replay()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters / 10 * 1e3


rows = []
for K in (4, 8, 16):
    for B in (64, 100, 160, 256):
        w = 1.0 / np.arange(1, K + 1) ** 0.9
        w /= w.sum()
        ids = list(np.sort(np.random.default_rng(0).choice(K, size=B, p=w)) + 1)
        mixed = list(np.random.default_rng(1).permutation(ids))
        xh = torch.randn(B, H, device=dev, dtype=dt)
        xhs = torch.randn(B, HS, device=dev, dtype=dt)
        xi = torch.randn(B, IS, device=dev, dtype=dt)
        ys = [torch.zeros(B, n, device=dev, dtype=dt) for n in (3 * HS, 2 * IS, H, H)]
        t = {}
        for name, grouped, order in (("stock_sorted", False, ids), ("stock_mixed", False, mixed),
                                     ("as", True, ids)):
            pw = wrapper(grouped)
            prepare(pw, order)
            t[name] = graph_time_us(lambda: layer(pw, xh, xhs, xi, ys))
        outs = []
        for grouped in (False, True):
            pw = wrapper(grouped)
            prepare(pw, ids)
            yy = [torch.zeros_like(y) for y in ys]
            layer(pw, xh, xhs, xi, yy)
            torch.cuda.synchronize()
            outs.append(yy)
        err = max((a.float() - b.float()).abs().max().item() for a, b in zip(*outs))
        rows.append(dict(K=K, B=B, **{f"{k}_us": v for k, v in t.items()}, max_abs_err=err))
        print(f"r{RANK} {SHAPE} K={K} B={B} stock_sorted={t['stock_sorted']:.1f}us "
              f"stock_mixed={t['stock_mixed']:.1f}us as={t['as']:.1f}us err={err:.4f}", flush=True)

path = f"{OUT}/kernel_layer_r{RANK}_{SHAPE}_mode{SPW._MODE}.json"
json.dump({"rank": RANK, "shape": SHAPE, "mode": SPW._MODE, "unit": "us per layer",
           "rows": rows}, open(path, "w"), indent=1)
print(f"wrote {path}", flush=True)
