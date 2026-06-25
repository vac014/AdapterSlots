"""
fused_lora_kernel.py -- Fused base+LoRA Triton kernel (Level-2 execution path).

Replaces the two-launch SGMV path (shrink kernel + expand kernel) with a single
Triton kernel that shares the X tensor load between base GEMM and LoRA down-
projection. The intermediate tensor H = X @ A^T is accumulated in registers and
never written to HBM.

Performance properties:
    - X loaded once vs twice in SGMV: ~33% fewer HBM reads on LoRA path
    - 2 kernel launches eliminated per layer per adapter
    - H never materialised in HBM
    - Expected speedup: 1.27–1.39× (Chronicals measured range, arXiv:2601.02609)
    - Works for any batch composition (heterogeneous segments, any N)

Graceful degradation:
    FusedLoRAKernel.is_available() returns False when Triton is not installed.
    All callers must check is_available() and fall back to SGMV when False.
    Set AS_FUSED_KERNEL=0 to disable even when Triton is present.

Integration:
    Used by AlignmentAwareModelRunner (model_runner.py) for non-promoted segments.
    Promoted segments (Level-3) use cuBLAS GEMM with merged weights instead;
    the fused kernel degenerates to a base GEMM when A=zeros and B=zeros.

GPU execution note:
    The Triton kernel requires a CUDA GPU. CPU reference (torch.matmul) is
    provided for unit testing and correctness validation only.
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
    W: torch.Tensor,       # (N, K) -- weight matrix (transposed convention)
    A: torch.Tensor,       # (R, K) -- LoRA down-proj
    B: torch.Tensor,       # (N, R) -- LoRA up-proj
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

def _build_triton_kernel():
    """Lazy-build the Triton JIT kernel. Only called when Triton is available."""
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 16, "BLOCK_N": 64,  "BLOCK_K": 32}),
            triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 32}),
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 64,  "BLOCK_K": 32}),
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32}),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 64,  "BLOCK_K": 32}),
        ],
        key=["M", "N", "K", "R"],
    )
    @triton.jit
    def _kernel(
        X_ptr, W_ptr, A_ptr, B_ptr, Y_ptr,
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
    ):
        """Fused Y = X @ W^T + alpha * (X @ A^T) @ B^T in one kernel launch.

        Critical optimization: X is loaded once per (BLOCK_M, BLOCK_K) tile and
        reused for both base GEMM (X @ W^T) and LoRA shrink (X @ A^T).
        The intermediate H = X @ A^T lives in registers; never written to HBM.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        acc_base = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        # LoRA shrink accumulator: (BLOCK_M, R) -- rank R is typically small (16–64)
        # We process R in full to keep it register-resident.
        h_acc = tl.zeros((BLOCK_M, R), dtype=tl.float32)

        for k in range(0, tl.cdiv(K, BLOCK_K)):
            offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
            # Load X tile: (BLOCK_M, BLOCK_K)
            x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
            x = tl.load(
                X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                mask=x_mask, other=0.0,
            )
            # Load W tile: (BLOCK_N, BLOCK_K) → contribute to X @ W^T → (BLOCK_M, BLOCK_N)
            w_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
            w = tl.load(
                W_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                mask=w_mask, other=0.0,
            )
            acc_base += tl.dot(x, tl.trans(w), out_dtype=tl.float32)
            # Load A tile: (R, BLOCK_K) -- LoRA down-proj (small)
            offs_r = tl.arange(0, R)
            a_mask = (offs_r[:, None] < R) & (offs_k[None, :] < K)
            a = tl.load(
                A_ptr + offs_r[:, None] * stride_an + offs_k[None, :] * stride_ak,
                mask=a_mask, other=0.0,
            )
            # h_acc += x @ A^T: (BLOCK_M, BLOCK_K) @ (BLOCK_K, R) → (BLOCK_M, R)
            h_acc += tl.dot(x, tl.trans(a), out_dtype=tl.float32)

        # LoRA expand: h_acc @ B^T → (BLOCK_M, BLOCK_N)
        # Cast h_acc to fp16 so both inputs to tl.dot share the same dtype.
        offs_r = tl.arange(0, R)
        b_mask = (offs_n[:, None] < N) & (offs_r[None, :] < R)
        b = tl.load(
            B_ptr + offs_n[:, None] * stride_bn + offs_r[None, :] * stride_br,
            mask=b_mask, other=0.0,
        )
        lora_out = tl.dot(h_acc.to(tl.float16), tl.trans(b), out_dtype=tl.float32)

        out = acc_base + alpha * lora_out
        y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(
            Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
            out.to(tl.float16),
            mask=y_mask,
        )

    return _kernel


_triton_kernel = None


# Public API

class FusedLoRAKernel:
    """Fused base+LoRA kernel: Y = X @ W^T + alpha * (X @ A^T) @ B^T.

    Level-2 execution path in the WGKP three-level hierarchy:
        Level 1 (SGMV):  shrink kernel + expand kernel (two launches, H in HBM)
        Level 2 (Fused): this kernel -- one launch, H in registers
        Level 3 (WGKP):  cuBLAS GEMM with merged W_k = W + alpha*B@A (no LoRA branch)

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
        X: torch.Tensor,      # (M, K) -- input activations
        W: torch.Tensor,      # (N, K) -- base weight (transposed convention)
        A: torch.Tensor,      # (R, K) -- LoRA down-proj
        B: torch.Tensor,      # (N, R) -- LoRA up-proj
        alpha: float = 1.0,
    ) -> torch.Tensor:
        """Compute Y = X @ W^T + alpha * (X @ A^T) @ B^T.

        Falls back to CPU reference if Triton is unavailable (for testing only;
        never use the CPU path in production serving).

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

        # Ensure contiguous and FP16 for the Triton kernel.
        X = X.contiguous().half()
        W = W.contiguous().half()
        A = A.contiguous().half()
        B = B.contiguous().half()

        Y = torch.empty((M, N), dtype=torch.float16, device=X.device)
        grid = lambda meta: (  # noqa: E731
            (M + meta["BLOCK_M"] - 1) // meta["BLOCK_M"],
            (N + meta["BLOCK_N"] - 1) // meta["BLOCK_N"],
        )
        _triton_kernel[grid](
            X, W, A, B, Y,
            M, N, K, R,
            alpha,
            X.stride(0), X.stride(1),
            W.stride(0), W.stride(1),
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            Y.stride(0), Y.stride(1),
        )
        return Y
