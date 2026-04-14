#!/usr/bin/env bash
# run_bench.sh — build and benchmark PoT GEMM kernels.
#
# Usage:
#   ./run_bench.sh [OPTIONS]
#
# Thread modes (pick one):
#   --1t           Single thread
#   --1s           Single socket (16 cores, NUMA-pinned)
#   --all          All sockets (default)
#
# Actions:
#   --build        (Re)build before running
#   --sweep        Run predefined M/N/K sweep
#
# Kernel selection:
#   --kernel v01   Benchmark v01 (tiled, no packing)
#   --kernel v02   Benchmark v02 (GOTO-style, packed)  [default]
#   --packed       Use pre-packed B (v02 only)
#
# Custom single run:
#   ./run_bench.sh --1s -- -M 1 -N 4096 -K 4096
#
# Extra args after -- are forwarded to the executable.

set -euo pipefail
cd "$(dirname "$0")"

# ---- defaults ----
THREAD_MODE="all"
DO_BUILD=0
DO_SWEEP=0
KERNEL="v02"
PACKED=""
EXTRA_ARGS=()

# ---- parse args ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --1t)       THREAD_MODE="1t";   shift ;;
        --1s)       THREAD_MODE="1s";   shift ;;
        --all)      THREAD_MODE="all";  shift ;;
        --build)    DO_BUILD=1;         shift ;;
        --sweep)    DO_SWEEP=1;         shift ;;
        --kernel)   KERNEL="$2";        shift 2 ;;
        --packed)   PACKED="--packed";  shift ;;
        --)         shift; EXTRA_ARGS=("$@"); break ;;
        *)          EXTRA_ARGS+=("$1"); shift ;;
    esac
done

EXE="build/src/${KERNEL}/bench_${KERNEL}"

# ---- build ----
if [[ $DO_BUILD -eq 1 ]]; then
    echo "=== Building ==="
    mkdir -p build
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -3
    cmake --build build -j "$(nproc)" --target "bench_${KERNEL}" 2>&1 | tail -5
    echo ""
fi

if [[ ! -x "$EXE" ]]; then
    echo "error: $EXE not found — run with --build first" >&2
    exit 1
fi

# ---- thread environment ----
NCORES_PER_SOCKET=16

case "$THREAD_MODE" in
    1t)
        export OMP_NUM_THREADS=1
        export OPENBLAS_NUM_THREADS=1
        NUMA_PREFIX=""
        MODE_LABEL="single-thread"
        ;;
    1s)
        export OMP_NUM_THREADS=$NCORES_PER_SOCKET
        export OPENBLAS_NUM_THREADS=$NCORES_PER_SOCKET
        export OMP_PROC_BIND=close
        export OMP_PLACES=cores
        NUMA_PREFIX="numactl --cpunodebind=0 --membind=0"
        MODE_LABEL="single-socket (${NCORES_PER_SOCKET} cores)"
        ;;
    all)
        TOTAL_CORES=$(nproc --all)
        export OMP_NUM_THREADS=$TOTAL_CORES
        export OPENBLAS_NUM_THREADS=$TOTAL_CORES
        export OMP_PROC_BIND=close
        export OMP_PLACES=cores
        NUMA_PREFIX=""
        MODE_LABEL="all-sockets (${TOTAL_CORES} cores)"
        ;;
esac

run() {
    $NUMA_PREFIX "./$EXE" "$@"
}

echo "=== Mode: $MODE_LABEL | Kernel: $KERNEL $PACKED ==="
echo ""

# ---- sweep or single run ----
if [[ $DO_SWEEP -eq 1 ]]; then
    M_VALUES=(1 4 16 32 64 128 256)
    NK_VALUES=(4096 8192 16384)

    for nk in "${NK_VALUES[@]}"; do
        echo "--- N=K=$nk ---"
        for m in "${M_VALUES[@]}"; do
            # v01 requires K%64==0; v02 requires K%16==0 and N%16==0 (all NK satisfy both)
            run -M "$m" -N "$nk" -K "$nk" $PACKED --no-verify "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
        done
        echo ""
    done
else
    if [[ ${#EXTRA_ARGS[@]} -eq 0 ]]; then
        echo "error: provide matrix dims, e.g.:  ./run_bench.sh --1s -- -M 1 -N 4096 -K 4096" >&2
        echo "       or use --sweep for a full dimension sweep" >&2
        exit 1
    fi
    run $PACKED "${EXTRA_ARGS[@]}"
fi
