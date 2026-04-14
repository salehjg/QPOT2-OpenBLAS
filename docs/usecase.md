# Use Cases: PoT GEMM in Deep Neural Networks

The kernel computes `C[M×N] = A[M×K] × decode(B[N×K])ᵀ` where A is float32
activations and B is PoT-encoded weights (uint8). The target operating regime is
**static weights, small batch** — anywhere the weight matrix is fixed at inference
time and the batch dimension M is small enough that loading weights from memory
dominates over compute.

---

## Where it fits

### 1. Transformer linear projections (primary target)

Every transformer block contains 4–7 linear layers:

- **Attention**: Q, K, V, and output projections — weights `[d_model × d_model]`
- **FFN**: up-projection, gate, down-projection — weights `[d_model × d_ff]`
  (typically d_ff = 4×d_model)

For autoregressive LLM inference (token-by-token generation) M=1 and
N=K=4096–16384. This is the exact shape benchmarked here, where pre-packed PoT
achieves **5× over OpenBLAS sgemm** at single-thread.

### 2. Mixture-of-Experts (MoE) expert layers

Each token is routed to a small number of experts. Per expert, only a handful of
tokens are processed simultaneously (often M=1–4). The expert weight matrices are
large but M is tiny, making inference deeply bandwidth-bound. PoT's 4× reduction
in weight memory traffic is maximally effective here.

### 3. Classifier / vocabulary projection head

LLMs project from hidden dimension to vocabulary logits: `[hidden_dim → vocab_size]`
where vocab_size can be 32k–128k. N is very large, M is small. The layer is a
pure bandwidth bottleneck and benefits directly from smaller B storage.

### 4. SSMs / Mamba / recurrent inference

State-space models and LSTMs process one timestep at a time in inference mode,
so M=1 always. Weight matrices are fixed. Direct drop-in for the recurrent
projection steps.

### 5. Convolution via im2col

Standard convolutions can be expressed as GEMM: im2col reshapes the input into
a float32 matrix A, and filter weights form B. Applicable when the spatial
reduction dimension is large (K = C_in × kH × kW ≥ 256). Less relevant in
modern architectures where attention-based models dominate.

---

## Where it does not help

| Layer / scenario | Reason |
|-----------------|--------|
| Training | Weights change every step; pre-packing is invalid, gradients need full precision |
| Large-batch inference (M > ~40) | OpenBLAS FMA throughput scales with M and crosses over around M=40 for N=K=4096 |
| BatchNorm / LayerNorm / RMSNorm | No GEMM |
| Softmax / activation functions | No GEMM |
| KV-cache attention scores (Q×Kᵀ) | K matrix grows dynamically each token; not a static weight |
| Small weight matrices (K < 256) | Packing overhead amortises poorly |

---

## The quantization constraint

B weights must be exactly powers of two: values in {±1, ±2, ±4, ..., ±2³¹}.
This is more aggressive than INT8 — there are zero mantissa bits, only a 5-bit
exponent and a sign bit. Two paths exist:

**Quantization-Aware Training (QAT)** — preferred.
Weights are constrained toward ±2^k during training via a straight-through
estimator on the rounding operation. Accuracy can be largely preserved because
the model learns to work within the constraint.

**Post-Training Quantization (PTQ)** — fast but lossy.
Round existing float32 weights to the nearest power of two after training.
Expect noticeable accuracy degradation; PoT's zero-mantissa encoding is
significantly coarser than INT8 PTQ. Acceptable only for layers that are known
to be robust to quantization (e.g. later FFN layers).

---

## Benchmarked performance (Xeon Gold 5218, single thread, pre-packed B)

| M | N=K | PoT GFLOPS | BLAS GFLOPS | Speedup |
|---|-----|-----------|------------|---------|
| 1 | 4096 | 11.65 | 2.31 | **5.0×** |
| 16 | 4096 | 52.39 | 30.50 | **1.72×** |
| 32 | 4096 | 52.10 | 45.31 | **1.15×** |
| 64 | 4096 | 52.16 | 59.95 | 0.87× |
| 128 | 4096 | 52.23 | 72.09 | 0.72× |

PoT throughput is flat (~52 GFLOPS) because the kernel saturates its execution
bottleneck independent of M. BLAS throughput scales with M as FMA reuse improves,
crossing over around M≈40 for this shape. For inference batch sizes common in
production (M=1–32), PoT is consistently faster.

---

## Recommended layer priority

1. **FFN down-projection** — largest single layer in most transformer models,
   very large K, least accuracy-sensitive to quantization.
2. **All linear layers during autoregressive generation** — M=1 always, every
   layer is bandwidth-bound.
3. **MoE expert projections** — small M per expert, large weight matrices
   streamed from DRAM.
4. **LM head / classifier** — large N (vocabulary size), pure bandwidth
   bottleneck.
