# PoT GEMM — Power-of-Two Quantized Matrix Multiply

AVX-512 GEMM kernels where B weights are quantized to powers of 2 and stored
as `uint8` (4x smaller than `float32`).  Multiplication is replaced by integer
exponent addition on the IEEE 754 bit representation — no FP multiply unit
needed.

## What is compared

| | Gold (reference) | Unit under test |
|---|---|---|
| **Operation** | `C = A × Bᵀ` | `C = A × decode(B_u8)ᵀ` |
| **A dtype** | float32 | float32 |
| **B dtype** | float32 (4 B/elem) | uint8 PoT-encoded (1 B/elem) |
| **Implementation** | OpenBLAS `cblas_sgemm` | Custom AVX-512 PoT kernel |
| **Library** | OpenBLAS 0.3.32 (SKYLAKEX) | This repo (`src/v01`, `src/v02`) |

## Kernel versions

- **v01** — Tiled, no packing, NR=1.  Simple baseline.
- **v02** — GOTO-style 5-loop with A/B packing, NR=16.  Supports pre-packed B
  (`--packed`) for inference where weights are packed once at model load.

## Build

```bash
git submodule update --init --recursive
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

Produces `build/src/v01/bench_v01` and `build/src/v02/bench_v02`.

## Run

### Single point

```bash
# v02, single socket, M=1 N=K=4096
OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 \
numactl --cpunodebind=0 --membind=0 \
./build/src/v02/bench_v02 -M 1 -N 4096 -K 4096

# v02 with pre-packed B
./build/src/v02/bench_v02 -M 1 -N 4096 -K 4096 --packed

# v01
./build/src/v01/bench_v01 -M 1 -N 4096 -K 4096
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

Xeon Gold 5218 (Cascade Lake / SKYLAKEX): 2 sockets × 16 cores, AVX-512,
L1d=32 KB, L2=1 MB, L3=22 MB shared, ~100 GB/s DRAM per socket.
