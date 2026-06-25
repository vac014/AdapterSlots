"""
fused_lora_kernel.py -- Fused base+LoRA Triton kernel (WGKP L2).

Replaces the two-launch SGMV path (shrink kernel + expand kernel) with a single
Triton kernel that shares the X tensor load between base GEMM and LoRA down-
projection. The intermediate tensor H = X @ A^T is accumulated in registers and
never written to HBM.

Performance properties:
    - X loaded once vs twice in SGMV: ~33% fewer HBM reads on LoRA path
    - 2 kernel launches eliminated per layer per adapter
    - H never materialised in HBM
    - Works for any batch composition (heterogeneous segments, any N)

FusedLoRAKernel.is_available() is False without Triton or with
AS_FUSED_KERNEL=0; callers then use the two-launch path.

scripts/experiments/fused_kernel_crossover_benchmark.py measures it against the
SGMV-style path (psi_fuse). The Triton kernel needs a CUDA GPU; the CPU
reference (torch.matmul) is for unit tests.
"""

import os
from typing import Optional

import torch

# Triton availability check

# Import Triton at module level so `tl` is in the module globals when the JIT
# compiler resolves names inside the @triton.jit kernel.
try:
    import triton
    import triton.language as tl
    from triton.compiler.compiler import CompiledKernel
    _TRITON_IMPORT_OK = True
except ImportError:
    _TRITON_IMPORT_OK = False

_TRITON_AVAILABLE: Optional[bool] = None


def _check_triton() -> bool:
    global _TRITON_AVAILABLE
    if _TRITON_AVAILABLE is not None:
        return _TRITON_AVAILABLE
    if os.environ.get("AS_FUSED_KERNEL", "1") == "0":
        _TRITON_AVAILABLE = False
        return False
    _TRITON_AVAILABLE = _TRITON_IMPORT_OK
    return _TRITON_AVAILABLE


# CPU reference implementation (for testing and fallback)

def _fused_lora_cpu_reference(
    X: torch.Tensor,       # (M, K)
    W: torch.Tensor,       # (N, K), weight matrix (transposed convention)
    A: torch.Tensor,       # (R, K), LoRA down-proj
    B: torch.Tensor,       # (N, R), LoRA up-proj
    alpha: float,
) -> torch.Tensor:
    """CPU reference for fused base+LoRA: Y = X @ W^T + alpha * (X @ A^T) @ B^T.

    This is mathematically equivalent to the Triton kernel but runs on CPU
    using standard torch.matmul. Used for unit testing and as a fallback when
    Triton is unavailable. No performance optimization.
    """
    base_out = torch.matmul(X, W.T)           # (M, N)
    h = torch.matmul(X, A.T)                  # (M, R)
    lora_out = torch.matmul(h, B.T)           # (M, N)
    return base_out + alpha * lora_out


# Triton kernel definition

def _small_m_block(M):
    return 16 if M <= 16 else 32 if M <= 32 else 64


def _autotune_configs():
    """Configs for decode-sized M (BLOCK_M follows M) and for larger M."""
    # Decode-sized M is a skinny, bandwidth-bound GEMM: the grid has to cover the
    # SMs, so split K across CTAs instead of relying on N tiles alone.
    shapes = [
        (32, 128, 1), (32, 128, 2), (32, 128, 4), (32, 256, 1), (32, 256, 2),
        (64, 128, 1), (64, 128, 2), (64, 128, 4), (64, 128, 8), (64, 256, 2), (64, 256, 4),
        (128, 64, 4), (128, 64, 8), (128, 128, 2), (128, 128, 4), (128, 128, 8),
    ]
    small = [(bn, bk, sk, 4, 3) for bn, bk, sk in shapes] + [
        (64, 128, 4, 4, 4), (64, 128, 4, 8, 3), (128, 128, 4, 8, 3), (32, 128, 4, 2, 4)]
    cfgs = [triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "SPLIT_K": sk},
                          num_warps=nw, num_stages=ns)
            for bm in (16, 32, 64) for bn, bk, sk, nw, ns in small]
    # Above 64 rows the GEMM has compute to spare: taller tiles, little or no split.
    large = [(64, 64, 64, 1, 4, 3), (64, 128, 32, 1, 4, 3), (64, 128, 64, 1, 8, 3),
             (64, 64, 64, 2, 4, 3), (128, 64, 32, 1, 4, 3), (128, 64, 64, 1, 8, 3),
             (128, 128, 32, 1, 8, 3), (128, 64, 32, 2, 4, 3)]
    large_cfgs = [triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "SPLIT_K": sk},
                                num_warps=nw, num_stages=ns)
                  for bm, bn, bk, sk, nw, ns in large]
    large_ids = {id(c) for c in large_cfgs}

    def prune(configs, named_args, **kwargs):
        M = named_args["M"]
        if M > 64:
            return [c for c in configs if id(c) in large_ids]
        bm = _small_m_block(M)
        return [c for c in configs if id(c) not in large_ids and c.kwargs["BLOCK_M"] == bm]

    return cfgs + large_cfgs, prune


def _build_triton_kernel():
    """Lazy-build the Triton JIT kernel. Only called when Triton is available."""
    configs, prune = _autotune_configs()

    @triton.autotune(configs=configs, key=["M", "N", "K", "R"],
                     prune_configs_by={"early_config_prune": prune})
    @triton.jit
    def _kernel(
        X_ptr, W_ptr, A_ptr, B_ptr, Y_ptr, WS_ptr, CNT_ptr,
        M, N, K, R: tl.constexpr,
        alpha,
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_an, stride_ak,
        stride_bn, stride_br,
        stride_ym, stride_yn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        SPLIT_K: tl.constexpr,
    ):
        """Fused Y = X @ W^T + alpha * (X @ A^T) @ B^T in one kernel launch.

        X is loaded once per (BLOCK_M, BLOCK_K) tile and reused for the base GEMM
        (X @ W^T) and the LoRA shrink (X @ A^T); H = X @ A^T stays in registers.

        Split-K: program (n, s, m) covers one K slice. The LoRA term is linear
        in H, so (sum_s H_s) @ B^T = sum_s (H_s @ B^T) and each slice adds its
        own share. Slices accumulate into an fp32 workspace; the last slice to
        finish a tile writes Y and zeroes the workspace and its counter, so the
        whole thing stays a single launch with no memset.
        """
        pid_n = tl.program_id(0)
        pid_s = tl.program_id(1)
        pid_m = tl.program_id(2)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_r = tl.arange(0, R)
        rk = tl.arange(0, BLOCK_K)
        m_mask = offs_m < M
        n_mask = offs_n < N

        k_chunk = tl.cdiv(tl.cdiv(K, SPLIT_K), BLOCK_K) * BLOCK_K
        k_lo = pid_s * k_chunk

        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + (k_lo + rk)[None, :] * stride_xk
        w_ptrs = W_ptr + offs_n[:, None] * stride_wn + (k_lo + rk)[None, :] * stride_wk
        a_ptrs = A_ptr + offs_r[:, None] * stride_an + (k_lo + rk)[None, :] * stride_ak

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        h = tl.zeros((BLOCK_M, R), dtype=tl.float32)
        for k in range(0, k_chunk, BLOCK_K):
            k_mask = (k_lo + k + rk) < K
            x = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
            w = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)
            a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)
            acc = tl.dot(x, tl.trans(w), acc)
            h = tl.dot(x, tl.trans(a), h)
            x_ptrs += BLOCK_K * stride_xk
            w_ptrs += BLOCK_K * stride_wk
            a_ptrs += BLOCK_K * stride_ak

        # LoRA expand for this slice, alpha folded into H.
        b = tl.load(B_ptr + offs_n[:, None] * stride_bn + offs_r[None, :] * stride_br,
                    mask=n_mask[:, None], other=0.0)
        acc = tl.dot((h * alpha).to(tl.float16), tl.trans(b), acc)

        y_mask = m_mask[:, None] & n_mask[None, :]
        y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
        if SPLIT_K == 1:
            tl.store(y_ptrs, acc.to(tl.float16), mask=y_mask)
        else:
            ws_ptrs = WS_ptr + offs_m[:, None] * N + offs_n[None, :]
            tl.atomic_add(ws_ptrs, acc, mask=y_mask, sem="relaxed")
            tl.debug_barrier()
            tile = pid_m * tl.num_programs(0) + pid_n
            arrived = tl.atomic_add(CNT_ptr + tile, 1, sem="acq_rel")
            if arrived == SPLIT_K - 1:
                tl.debug_barrier()
                tot = tl.load(ws_ptrs, mask=y_mask, other=0.0, cache_modifier=".cg")
                tl.store(y_ptrs, tot.to(tl.float16), mask=y_mask)
                tl.store(ws_ptrs, tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32), mask=y_mask)
                tl.store(CNT_ptr + tile, 0)

    return _kernel


# Split-K scratch, per device: fp32 partial sums and per-tile arrival counters.
# Both start zeroed and every launch leaves them zeroed again.
_workspace: dict = {}


def _get_workspace(device: torch.device, n_ws: int, n_cnt: int):
    ws, cnt = _workspace.get(device, (None, None))
    if ws is None or ws.numel() < n_ws:
        ws = torch.zeros(max(n_ws, 1 << 20), dtype=torch.float32, device=device)
    if cnt is None or cnt.numel() < n_cnt:
        cnt = torch.zeros(max(n_cnt, 4096), dtype=torch.int32, device=device)
    _workspace[device] = (ws, cnt)
    return ws, cnt


_triton_kernel = None
# Per-shape (compiled kernel, grid) the autotuner chose, for direct relaunch.
_launch_cache: dict = {}


# Public API

class FusedLoRAKernel:
    """Fused base+LoRA kernel: Y = X @ W^T + alpha * (X @ A^T) @ B^T.

    The L2 kernel of the WGKP hierarchy:
        L1 (SGMV):    shrink and expand kernels, two launches, H in HBM
        L2 (fused):   this kernel, one launch, H in registers
        L3 (merged):  cuBLAS GEMM on W_k = W + alpha * B @ A, no LoRA branch

    Usage:
        kernel = FusedLoRAKernel()
        if kernel.is_available():
            Y = kernel.forward(X, W, A, B, alpha)
        else:
            Y = sgmv_fallback(X, W, A, B, alpha)
    """

    def __init__(self) -> None:
        self._available = _check_triton()
        if self._available:
            global _triton_kernel
            if _triton_kernel is None:
                _triton_kernel = _build_triton_kernel()

    @staticmethod
    def is_available() -> bool:
        """Return True if Triton is installed and AS_FUSED_KERNEL=1."""
        return _check_triton()

    def forward(
        self,
        X: torch.Tensor,      # (M, K), input activations
        W: torch.Tensor,      # (N, K), base weight (transposed convention)
        A: torch.Tensor,      # (R, K), LoRA down-proj
        B: torch.Tensor,      # (N, R), LoRA up-proj
        alpha: float = 1.0,
    ) -> torch.Tensor:
        """Compute Y = X @ W^T + alpha * (X @ A^T) @ B^T.

        Without Triton this runs the CPU reference (unit tests only).

        Args:
            X:     Input activations, shape (M, K).
            W:     Base weight matrix, shape (N, K).
            A:     LoRA down-proj matrix, shape (R, K).
            B:     LoRA up-proj matrix, shape (N, R).
            alpha: LoRA scaling factor.

        Returns:
            Output tensor Y, shape (M, N).
        """
        if not self._available or not X.is_cuda:
            return _fused_lora_cpu_reference(X, W, A, B, alpha)
        return self._forward_triton(X, W, A, B, alpha)

    def _forward_triton(
        self,
        X: torch.Tensor,
        W: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        M, K = X.shape
        N = W.shape[0]
        R = A.shape[0]

        # FP16, contiguous (no-ops when the inputs already are).
        X, W, A, B = (t if t.dtype == torch.float16 and t.is_contiguous()
                      else t.contiguous().half() for t in (X, W, A, B))

        # Sized for the smallest tile any config uses (BLOCK_M 16, BLOCK_N 32).
        ws, cnt = _get_workspace(X.device, M * N, triton.cdiv(M, 16) * triton.cdiv(N, 32))
        Y = torch.empty((M, N), dtype=torch.float16, device=X.device)
        strides = (X.stride(0), X.stride(1), W.stride(0), W.stride(1), A.stride(0),
                   A.stride(1), B.stride(0), B.stride(1), Y.stride(0), Y.stride(1))
        # Non-constexpr arguments in signature order (R and the BLOCK sizes are constexpr).
        args = (X, W, A, B, Y, ws, cnt, M, N, K, alpha) + strides

        key = (X.device, M, N, K, R, strides,
               tuple(t.data_ptr() % 16 == 0 for t in (X, W, A, B, Y)))
        hit = _launch_cache.get(key)
        if hit is not None:
            # Decode-sized calls are only tens of microseconds on the GPU, so the
            # autotuner's per-call Python work would dominate; launch the
            # compiled kernel it already picked for this shape directly.
            kernel, grid = hit
            stream = torch.cuda.current_stream(X.device).cuda_stream
            kernel.run(grid[0], grid[1], grid[2], stream, kernel.function,
                       kernel.packed_metadata, kernel.launch_metadata(grid, stream, *args),
                       CompiledKernel.launch_enter_hook, CompiledKernel.launch_exit_hook,
                       *args)
            return Y

        grid_fn = lambda meta: (  # noqa: E731
            triton.cdiv(N, meta["BLOCK_N"]),
            meta["SPLIT_K"],
            triton.cdiv(M, meta["BLOCK_M"]),
        )
        _triton_kernel[grid_fn](X, W, A, B, Y, ws, cnt, M, N, K, R, alpha, *strides)
        cfg = _triton_kernel.best_config
        opts = cfg.all_kwargs() if hasattr(cfg, "all_kwargs") else dict(
            cfg.kwargs, num_warps=cfg.num_warps, num_stages=cfg.num_stages)
        kernel = _triton_kernel.fn.run(X, W, A, B, Y, ws, cnt, M, N, K, R, alpha, *strides,
                                       grid=None, warmup=True, **opts)
        grid = (triton.cdiv(N, cfg.kwargs["BLOCK_N"]), cfg.kwargs["SPLIT_K"],
                triton.cdiv(M, cfg.kwargs["BLOCK_M"]))
        _launch_cache[key] = (kernel, grid)
        return Y
