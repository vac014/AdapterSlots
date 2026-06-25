"""Punica on the K=10 workload of C1 and Table II, using Punica's own batching
engine in-process (deps/punica).

The request set is the one the other K=10 runs measure: the same 400 ShareGPT
prompts (token lengths) after a 30-request warmup, the same Zipf-0.9 adapter
ids, 128 output tokens each, all requests queued at the start. Weights are
random, as in Punica's own benchmarks; output lengths are fixed, so throughput
does not depend on the weights. The engine loop follows
deps/punica/benchmarks/bench_textgen_lora.py (first come first served, one new
request per step, the batch kept sorted by adapter).

Settings (environment): SEED (default 42), PUNICA_BS (batch cap, default 256),
OUT, GPU. Build Punica first: pip install -e deps/punica.
"""
import os
import sys
import time

from common import REPO, out_dir, record, setup

setup()
sys.path.insert(0, os.environ.get("PUNICA_SRC", str(REPO / "deps" / "punica" / "src")))
sys.path.insert(0, "benchmarks/ablations")
import torch  # noqa: E402
from transformers import AutoTokenizer, LlamaConfig  # noqa: E402
from punica import (BatchedKvCache, BatchedLlamaLoraWeight, BatchLenInfo, KvCache,  # noqa: E402
                    KvPool, LlamaForCausalLMWithLora, LlamaLoraWeight)
import bench_apis as BA  # noqa: E402
from workloads.sharegpt_loader import get_prompts  # noqa: E402

K, RANK, NP, WU, MT = 10, 32, 400, 30, 128
SEED = int(os.environ.get("SEED", "42"))
BS = int(os.environ.get("PUNICA_BS", "256"))
OUT = out_dir("c1_k10/punica")

tok = AutoTokenizer.from_pretrained("./models/llama-7b")
prompts = get_prompts(dataset="sharegpt", n=NP + WU + 50, seed=SEED)
gen = BA.ArrivalPatternGenerator(prompts, max_tokens=MT)
reqs = gen.zipf(300.0, NP + WU, K=K, alpha=BA.ZIPF_ALPHA, seed=SEED)[WU:]
plen = [len(tok(r.prompt).input_ids) for r in reqs]
lidx = [int(r.adapter_id) for r in reqs]

cfg = LlamaConfig.from_pretrained("./models/llama-7b")
dev, dt = torch.device("cuda"), torch.float16
torch.manual_seed(0)
torch.set_default_dtype(dt)
with dev:
    model = LlamaForCausalLMWithLora(cfg).to(dev)
torch.set_default_dtype(torch.float32)
kvpool = KvPool(cfg.num_hidden_layers, cfg.num_key_value_heads,
                cfg.hidden_size // cfg.num_attention_heads, 16, dt, dev)
loras = [LlamaLoraWeight(cfg, RANK, dt, dev) for _ in range(K)]


@torch.inference_mode()
def run():
    nxt, workset, done, out_tok = 0, [], 0, 0
    t0 = time.perf_counter()
    while done < len(reqs):
        newreqs = []
        if len(workset) < BS and nxt < len(reqs):
            p = plen[nxt]
            newreqs.append((nxt, torch.randint(0, 32000, (p,)).tolist(), KvCache(kvpool, p)))
            nxt += 1
        ids, pkv, li, ll = [], [], [], []
        for i, pr, kv in newreqs:
            ids += pr
            pkv.append(kv)
            li.append(lidx[i])
            ll.append(len(pr))
        for w in workset:
            ids.append(w["out"][-1])
            a = lidx[w["i"]]
            if li and li[-1] == a:
                ll[-1] += 1
            else:
                li.append(a)
                ll.append(1)
        blen = BatchLenInfo(prefills=[len(pr) for _, pr, _ in newreqs], decode=len(workset),
                            indptr_device=dev)
        logits, _ = model(torch.tensor(ids, dtype=torch.long, device=dev), blen,
                          BatchedKvCache(pkv) if newreqs else None,
                          BatchedKvCache([w["kv"] for w in workset]) if workset else None,
                          BatchedLlamaLoraWeight([loras[a] for a in li], ll))
        new = []
        if newreqs:
            nt = torch.argmax(logits[blen.indptr[1:] - 1], dim=-1).cpu().numpy()
            for b, (i, pr, kv) in enumerate(newreqs):
                kv.acquire_one()
                new.append({"i": i, "kv": kv, "out": [int(nt[b])]})
            out_tok += len(newreqs)
        if workset:
            nt = torch.argmax(logits[blen.doff:], dim=-1).cpu().numpy()
            for b, w in enumerate(workset):
                w["out"].append(int(nt[b]))
                out_tok += 1
                if len(w["out"]) == MT:
                    w["kv"].release()
                    done += 1
                else:
                    w["kv"].acquire_one()
                    new.append(w)
        new.sort(key=lambda w: lidx[w["i"]])
        workset = new
    return out_tok, time.perf_counter() - t0


n, wall = run()
record(OUT, f"punica_s{SEED}", {"tput": n / wall, "out_tokens": n, "wall": wall, "n": len(reqs),
                                "batch_cap": BS})
print(f"s{SEED} tput={n / wall:.1f} tok/s wall={wall:.1f}s", flush=True)
