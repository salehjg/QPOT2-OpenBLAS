# Future Works: PoT (Power-of-Two) GEMM in OpenBLAS

## Context

The PoT approach replaces float32 multiplication with integer exponent addition
via IEEE 754 bit manipulation. B weights are quantized to powers of 2 and stored
as uint8 (4x smaller than float32). The "multiplication" is a shift+add on the
exponent field — no FP multiply unit needed.

**Current state**: standalone tiled kernel (`mm_avx512_f32_1_u8_1_v01`) with
OpenMP threading, benchmarked against OpenBLAS cblas_sgemm on square matrices
1024–4096. The PoT kernel is ~3–4x slower because large square GEMM is
compute-bound and FMA has a ~7:1 µop advantage over the PoT instruction sequence.

**Core thesis**: the PoT advantage is *bandwidth*, not compute. B is 4x smaller,
so when B streams from DRAM (inference workloads, tall-skinny shapes), PoT should
win despite the higher µop count per element.

---

## Phase 1 — Tall-Skinny Benchmarks (standalone)

**Goal**: demonstrate the bandwidth advantage in memory-bound regimes.

### What to benchmark

| Shape | Scenario |
|-------|----------|
| M=1, N=K=4096..16384 | GEMV-like inference (single sample) |
| M=4, N=K=4096..16384 | Small batch inference |
| M=16, N=K=4096..16384 | Medium batch inference |
| M=32, N=K=4096..16384 | Batch inference |
| M=N=K=1024..4096 | Square (compute-bound baseline) |

- Compare: OpenBLAS cblas_sgemm (f32×f32) vs mm_avx512_f32_1_u8_1_v01 (f32×u8)
- For cblas_sgemm, B is float32 (same logical values, just not PoT-encoded)
- Report: time, equivalent GFLOPS, and bytes-read from B per second

### Expected outcome

For small M with large N,K: B dominates memory traffic. PoT reads 4x less B data.
At the DRAM bandwidth wall (~100 GB/s on Xeon 5218), PoT should approach or
exceed cblas_sgemm throughput.

### Deliverable

Extended `bench_gemm.cpp` with rectangular shape sweeps and a summary table
showing the crossover point.

---

## Phase 2 — PoT Micro-Kernel Inside OpenBLAS

**Goal**: apples-to-apples comparison using OpenBLAS's GOTO framework (packing,
threading, cache tiling) with only the micro-kernel and B-packer replaced.

### OpenBLAS GOTO Architecture (for reference)

```
driver/level3/level3.c          — GOTO tile loops (M/N/K tiling, threading)
kernel/x86_64/
  sgemm_kernel_16x4_skylakex.c  — micro-kernel: MR×NR output tile per K step
  sgemm_tcopy_16_skylakex.c     — ICOPY: pack A into MC×KC contiguous panel
  sgemm_ncopy_4_skylakex.c      — OCOPY: pack B into KC×NC contiguous panel
param.h (SKYLAKEX section)      — tile sizes: MR=16, NR=4, P(MC)=448, Q(KC)=448
```

### What to implement

1. **PoT micro-kernel** (`spotgemm_kernel_MRxNR_skylakex.c`)
   - Inputs: packed A (float32, MR×KC), packed B (uint8, KC×NR), output C (float32)
   - Register blocking: MR rows × NR columns of C accumulators
   - Inner loop: load NR uint8 columns from packed B, expand, apply exponent
     addition to MR rows of packed A, accumulate
   - MR/NR to be determined by register pressure analysis (start with MR=16, NR=4
     to match existing sgemm structure, then tune)

2. **B packing routine** (`spotgemm_ncopy_NR_skylakex.c`)
   - Pack uint8 B columns into KC×NR contiguous tiles
   - B is 4x smaller per element, so KC can potentially be 4x larger while
     staying in L2 — sweep KC values

3. **A packing routine**
   - Reuse existing `sgemm_tcopy_16_skylakex.c` (A is still float32)

4. **Tile size tuning**
   - KC (Q): can be larger than 448 since packed B tile is 4x smaller.
     Sweep: 448, 896, 1024, 1792
   - MC (P), NC (R): start with sgemm defaults, tune later

5. **Integration**
   - Wire through `param.h` as new SPOTGEMM parameters
   - Add a C interface callable from the benchmark
   - Reuse OpenBLAS's thread pool (no separate OpenMP needed)

### Key files to create/modify in openblas/

```
kernel/x86_64/spotgemm_kernel_16x4_skylakex.c    NEW — PoT micro-kernel
kernel/x86_64/spotgemm_ncopy_4_skylakex.c         NEW — uint8 B packer
param.h                                            MOD — add SPOTGEMM tile sizes
```

### Expected outcome

With equal packing, threading, and tiling quality, the comparison isolates the
micro-kernel: FMA throughput vs PoT bandwidth savings. For memory-bound shapes
(Phase 1 results), the OpenBLAS-integrated PoT kernel should outperform sgemm.

---

## Phase 3 — Roofline Analysis (paper material)

**Goal**: produce publication-quality roofline plots showing the crossover.

- Measure arithmetic intensity (FLOP/byte) for both kernels at each shape
- Overlay on Xeon 5218 roofline (peak FMA GFLOPS vs DRAM bandwidth)
- Show that PoT shifts the operational intensity boundary, extending the
  compute-bound region to smaller M values
- Sweep M from 1 to 1024 to find the exact crossover

---

## Notes

- OpenBLAS does **not** do runtime autotuning. Tile sizes are hard-coded per
  TARGET in `param.h`. DYNAMIC_ARCH dispatches to pre-compiled kernels at
  runtime based on CPUID, but does not tune parameters.
- The Xeon Gold 5218 (Cascade Lake / SKYLAKEX target) has: L1d=32KB, L2=1MB,
  L3=22MB shared, 16 cores, AVX-512 with VNNI, ~100 GB/s DRAM bandwidth.
- The PoT encoding uses 6 bits per weight (5-bit exponent + 1-bit sign), packed
  into uint8. Future work could explore 4-bit packing for 8x compression.
