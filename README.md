# PoT GEMM — Power-of-Two Quantized Matrix Multiply

GEMM kernels where B weights are quantized to powers of 2 and stored as `uint8`
(4x smaller than `float32`).  Multiplication is replaced by integer exponent
addition on the IEEE 754 bit representation — no FP multiply unit needed.

Targets: AVX-512 (Xeon), CUDA (V100S / RTX), RVV 1.0 (SpacemiT K1).

## What is compared

| | Gold (reference) | Unit under test |
|---|---|---|
| **Operation** | `C = A × Bᵀ` | `C = A × decode(B_u8)ᵀ` |
| **A dtype** | float32 | float32 |
| **B dtype** | float32 (4 B/elem) | uint8 PoT-encoded (1 B/elem) |
| **Implementation** | OpenBLAS `cblas_sgemm` | Custom PoT kernel |
| **Library** | OpenBLAS 0.3.32 | This repo |

## Kernel versions

- **v01** — Tiled, no packing, NR=1.  Simple baseline.  (AVX-512)
- **v02** — GOTO-style 5-loop with A/B packing, NR=16.  (AVX-512)
- **v03** — v02 + ILP improvements: MR=16, KC=1024, k-unroll×2.  (AVX-512)
- **v04** — CUDA tiled PoT GEMM + small-M warp kernel.  (CUDA)
- **v05** — RVV 1.0 for SpacemiT K1.  LMUL parameterized (1/2/4).  Three
  micro-kernel styles: POT (int bit-trick), FMA (decode→vfmacc), MIX
  (alternating FMA+POT for int/float ALU overlap).  (RISC-V)

## Build

### x86 (Xeon, AVX-512) — v01/v02/v03

```bash
git submodule update --init --recursive
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

Produces `bench_v01`, `bench_v02`, `bench_v03` (and `bench_v04` if CUDA found).

### RISC-V (SpacemiT K1, BananaPi F3) — v05

```bash
git submodule update --init --recursive
mkdir -p build && cd build
# Use -DTARGET=x280 to enable RVV 1.0 vectorized OpenBLAS sgemm reference.
# Without it, OpenBLAS falls back to scalar RISCV64_GENERIC.
cmake .. -DCMAKE_BUILD_TYPE=Release -DTARGET=x280
cmake --build . -j$(nproc)
```

Produces `bench_v05`.  K and N must be multiples of 32.

## Run

### Single point (x86)

```bash
# v02, single socket, M=1 N=K=4096
OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 \
numactl --cpunodebind=0 --membind=0 \
./build/src/v02/bench_v02 -M 1 -N 4096 -K 4096

# v02 with pre-packed B
./build/src/v02/bench_v02 -M 1 -N 4096 -K 4096 --packed
```

### Single point (RISC-V / SpacemiT K1)

```bash
# v05, all 8 cores, compare POT/FMA/MIX at LMUL=4
./build/src/v05/bench_v05 -M 4 -N 4096 -K 4096 --packed
```

### Using run_bench.sh

```bash
# Build + sweep all dims on single socket
./run_bench.sh --build --1s --sweep

# Single thread, v02, pre-packed
./run_bench.sh --1t --packed --sweep --kernel v02

# Custom single run
./run_bench.sh --1s -- -M 4 -N 8192 -K 8192
```

### Thread modes

| Flag | Threads | NUMA |
|------|---------|------|
| `--1t` | 1 | — |
| `--1s` | 16 (one socket) | pinned to node 0 |
| `--all` | all cores | unpinned |

## Arguments

| Arg | Description | Default |
|-----|-------------|---------|
| `-M` | Rows of A / C | required |
| `-N` | Columns of C (rows of Bᵀ) | required |
| `-K` | Shared dimension | required |
| `-r, --repeats` | Timing iterations (median taken) | 7 |
| `--no-verify` | Skip correctness check | off |
| `--packed` | Pre-packed B, v02 only | off |

## Timing

All reported times are the **median** of `-r` repetitions (default 7).
A warmup call precedes the timed runs.

## Target hardware

- **Xeon Gold 5218** (Cascade Lake / SKYLAKEX): 2×16 cores, AVX-512,
  L1d=32 KB, L2=1 MB, L3=22 MB, ~100 GB/s DRAM/socket.
- **SpacemiT K1** (BananaPi F3): 8 cores, RVV 1.0, VLEN=256, in-order,
  L1d=32 KB, L2=512 KB/core, LPDDR4X ~34 GB/s.
- **RTX 2000 Ada / V100S** (CUDA): v04 tiled PoT kernel.
