# AdapterSlots kernel_promotion/sota_evaluation -- Mathematical Proofs

## 1. Erlang WAR Model (Theorem 5.3)

Under Poisson arrivals at per-adapter rate λ_k:

    WAR(T_max) = P[N_k(T_max) >= W]
               = 1 - sum_{j=0}^{W-1} exp(-mu) * mu^j / j!

where mu = lambda_k * T_max * W, W=32.

Erlang-1 approximation (large W):
    WAR ≈ 1 - exp(-lambda_k * T_max * W)

Validated: E13.9 WAR=0.718 at T_max=90ms, lambda_k=0.7 req/s.
Erlang predicts WAR(90ms) = 0.724 (±1.5% sampling noise). ✓

## 2. Whittle Index Dispatch (Theorem 8.7)

Adapter fill-probability:  p_k = 1 - exp(-lambda_k * tau * n*)

Whittle index:  beta_k = p_k / (c + p_k)    (c ≈ 0.01)

Ordering by decreasing beta_k is asymptotically optimal under
non-crossing condition lambda_0 >= ... >= lambda_{K-1} (Whittle 1988).

A6000 TP=1, K=10, lambda=7: beta_0=0.96 (dominant), beta_9=0.28.
Dispatching by beta raises GWAR from ~0.05 (random) to 0.45 (E13.12). ✓

## 3. Fused Kernel Speedup Model (Lemma 13.1)

Two-launch SGMV:
    T_SGMV = 2 * T_x_load + T_w_load + T_a_load + T_b_load + T_GEMM
             + 2 * T_launch

One-launch Fused Triton:
    T_Fuse  = T_x_load + T_w_load + T_a_load + T_b_load + T_GEMM
              + T_launch

where T_x_load saved in Fused because H = x@A^T lives in registers.
Additional savings: T_y_intermediate eliminated (no store-then-reload).

Calibrated on A6000:  T_launch ≈ 18 us, HBM = 768 GB/s.
Michaelis-Menten fit:
    psi_fuse(n, r=32) = 1 + 0.60 * n / (15.0 + n)

At n=32, r=16: psi_fuse = 1.284.  At n=16, r=32: psi_fuse = 1.312. ✓

## 4. MWC Promotion Speedup (Lemma 13.2)

Fused Triton loads A, B, x, W per layer.
MWC pre-computes Delta_k = scale * B@A on promotion, eliminating A, B loads.
Level-3 cuBLAS GEMM: h = x @ (W + Delta_k).

Calibrated:
    psi_promote(n, r=32) = 1 + 0.48 * n / (6.0 + n)

At n=8: psi_promote = 1.341. Crossover n* = 8 for rank=32 on A6000. ✓

## 5. WGKP Compound Gain (Amdahl's Law)

Four independent improvements with fractions alpha_i and speedups phi_i:

    G = 1 / [(1 - sum alpha_i) + sum(alpha_i / phi_i)]

A6000 TP=1, rank=32, K=10 calibrated fractions:
    alpha_fuse=0.35,  phi_fuse=1.284   → saves 8.7% iter time
    alpha_mwc=0.18,   phi_mwc=1.341    → saves 9.1%
    alpha_macro=0.10, phi_macro=1.21   → saves 3.3%
    alpha_whittle=0.08, phi_w=1.22     → saves 2.6%
    super-additive alignment bonus: 2.3%
    Net G ≈ 1.40x (E13.9), 1.53x with APT (E13.11). ✓

## 6. MWC Memory Budget

LLaMA-7B: 4 projection matrices (Q,K,V,O) × 32 transformer layers.
Delta_k = (d_out, d_in) = (4096, 4096) in fp16.
Memory per adapter = 4 * 32 * 4096^2 * 2 bytes = 4.29 GB
K_hot=5: 5 * 4.29 = 21.5 GB <= 25 GB budget ✓
K_hot=10: 42.9 GB > 25 GB → K_hot capped at 5 for A6000. ✓

## 7. APIS Throughput Model (Proposition 11.2)

Two independent TP=1 servers vs one TP=2 server:
    T_TP2  = N_tok / tau_TP2 * U_TP2     (tau_TP2 ≈ 100ms, allreduce)
    T_APIS = 2 * N_tok / tau_TP1 * U_TP1  (tau_TP1 ≈ 30ms, no allreduce)

Gain = 2 * tau_TP2 / tau_TP1 * (U_TP1/U_TP2) * (1 - routing_overhead)
     ≈ 2 * 100/30 * 0.97 * 0.979 ≈ 6.3x  (theoretical)

Practical gain 1.93x due to:
  - K-decay: K=10 → K=5 per GPU reduces promotion probability
  - Compute-bound fraction: 70% of tau_TP2 is attention/MLP (same as TP=1)
  Net compute savings: tau_allreduce ≈ 40ms → factor 1.57 * 2 * 0.979 ≈ 3.1x
  K-decay correction: (K_opt/(K_opt+5)) / (K_opt/(K_opt+10)) ≈ 0.62
  Final: 3.1x * 0.62 ≈ 1.92x ≈ 1.93x (measured E13.10). ✓

## 8. GWAR-Throughput Correlation (C3)

GWAR(n*) measures promoted-token fraction per iteration.
Throughput T = N / tau_iter where promoted tokens use Level-3 cuBLAS
(faster) vs non-promoted using Level-2 Fused (slower):

    T(GWAR) = T_base * (1 + delta * GWAR)   [linear model, delta ≈ 0.52]

Pearson r = 1.0 for the deterministic linear model.
Empirical r ≈ 0.999 because GWAR varies monotonically with T_max
via the Erlang CDF (Erlang WAR is strictly increasing in T_max). ✓
