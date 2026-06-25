"""
test_l3.py -- WGKP L3 on the CPU: the merged weights W_k = W + alpha B_k A_k,
the cache's sizing and eviction, and the promotion gate.
"""

import json
import time
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from adapter_slots.kernel import l3


D, R, ALPHA = 8, 2, 4.0


def _checkpoint(tmp_path, seed):
    torch.manual_seed(seed)
    tensors = {}
    for i in range(2):
        for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
            base = f"base_model.model.model.layers.{i}.self_attn.{p}"
            tensors[f"{base}.lora_A.weight"] = torch.randn(R, D)
            tensors[f"{base}.lora_B.weight"] = torch.randn(D, R)
    d = tmp_path / f"adapter_{seed}"
    d.mkdir()
    save_file(tensors, str(d / "adapter_model.safetensors"))
    (d / "adapter_config.json").write_text(json.dumps({"r": R, "lora_alpha": ALPHA}))
    return d, tensors


def _layers():
    torch.manual_seed(0)
    layers = {}
    for i in range(2):
        layers[f"model.layers.{i}.self_attn.qkv_proj"] = SimpleNamespace(
            base_layer=SimpleNamespace(weight=torch.randn(3 * D, D)))
        layers[f"model.layers.{i}.self_attn.o_proj"] = SimpleNamespace(
            base_layer=SimpleNamespace(weight=torch.randn(D, D)))
    return layers


def _wait_loaded(mwc, n=1):
    for _ in range(200):
        if len(mwc._loaded) >= n:
            return
        time.sleep(0.01)
    raise AssertionError("LoRA tensors did not load")


@pytest.fixture
def free_memory(monkeypatch):
    def set_free(nbytes):
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (nbytes, nbytes))
    return set_free


def test_merged_weights_match_formula(tmp_path, free_memory):
    free_memory(1 << 40)
    path, t = _checkpoint(tmp_path, 1)
    layers = _layers()
    mwc = l3.MergedWeightCache(k_hot_max=5, margin_gb=0)
    mwc.request(7, str(path))
    _wait_loaded(mwc)
    mwc.process(layers)
    merged = mwc.get(7)
    s = ALPHA / R
    for i in range(2):
        pre = f"base_model.model.model.layers.{i}.self_attn"
        delta_qkv = torch.cat([s * t[f"{pre}.{p}.lora_B.weight"] @ t[f"{pre}.{p}.lora_A.weight"]
                               for p in ("q_proj", "k_proj", "v_proj")])
        w = layers[f"model.layers.{i}.self_attn.qkv_proj"].base_layer.weight
        assert torch.allclose(merged[f"model.layers.{i}.self_attn.qkv_proj"], w + delta_qkv, atol=1e-5)
        delta_o = s * t[f"{pre}.o_proj.lora_B.weight"] @ t[f"{pre}.o_proj.lora_A.weight"]
        w = layers[f"model.layers.{i}.self_attn.o_proj"].base_layer.weight
        assert torch.allclose(merged[f"model.layers.{i}.self_attn.o_proj"], w + delta_o, atol=1e-5)


def test_cache_sized_by_free_memory(tmp_path, free_memory):
    layers = _layers()
    per_adapter = sum(m.base_layer.weight.numel() * 4 for m in layers.values())
    free_memory(2 * per_adapter + per_adapter // 2)
    mwc = l3.MergedWeightCache(k_hot_max=5, margin_gb=0)
    for seed in (1, 2, 3):
        path, _ = _checkpoint(tmp_path, seed)
        mwc.request(seed, str(path))
        _wait_loaded(mwc)
        mwc.process(layers)
    assert mwc.k_hot == 2
    assert not mwc.is_merged(1) and mwc.is_merged(2) and mwc.is_merged(3)


def test_no_room_disables_merging(tmp_path, free_memory):
    free_memory(0)
    path, _ = _checkpoint(tmp_path, 1)
    mwc = l3.MergedWeightCache(k_hot_max=5, margin_gb=0)
    mwc.request(1, str(path))
    _wait_loaded(mwc)
    mwc.process(_layers())
    assert mwc.k_hot == 0 and not mwc.is_merged(1)
    mwc.request(2, str(path))
    assert not mwc._requested


def test_promotion_gate():
    assert l3.promotable_adapter([3] * 8, n_star=8) == 3
    assert l3.promotable_adapter([3] * 7, n_star=8) is None
    assert l3.promotable_adapter([3] * 8 + [4], n_star=8) is None
    assert l3.promotable_adapter([0] * 16, n_star=8) is None


def test_active_weights_only_inside_promoted_step():
    layer = SimpleNamespace(_l3_name="x")
    assert l3._active_weight(layer) is None
    w = torch.ones(2, 2)
    with l3.promoted({"x": w}):
        assert l3._active_weight(layer) is w
    assert l3._active_weight(layer) is None
