/*
 * v04: CUDA PoT GEMM kernel.
 *
 * Two kernels are provided:
 *
 * pot_gemm_kernel  — general tiled GEMM for M ≥ 1.
 *   Tile: BM=128, BN=128, BK=16, TM=8, TN=8, block=256 threads.
 *   A loaded as float32 into shared memory.
 *   B loaded as uint8 into shared memory (4× smaller than float32).
 *   B decoded to float on the fly inside the register-level compute loop.
 *   Shared memory padding (+1 on A, +4 on B) reduces bank conflicts.
 *
 * pot_gemv_kernel  — specialised M=1 warp-reduction kernel.
 *   Each warp handles one output element via a K-stride loop + shfl_down.
 *   A is broadcast-friendly (same 16 KB vector read by all warps, fits L1).
 *
 * PoT decode: uint8 b → float via IEEE 754 bit construction.
 *   bits [4:0] = log2(|val|) ∈ [0,31], bit 5 = sign.
 *   float = (-1)^s × 2^e  →  ieee_bits = (s << 31) | ((127+e) << 23)
 *   Cost: 3 int ops + __int_as_float (zero-cost register rename).
 */

#include "kernel_v04.cuh"
#include <cuda_runtime.h>
#include <cstdint>

// ---------------------------------------------------------------------------
// PoT decode
// ---------------------------------------------------------------------------

__device__ __forceinline__ float decode_pot(uint8_t b) {
    const int e = b & 0x1F;
    const int s = (b >> 5) & 1;
    return __int_as_float(((127 + e) << 23) | (s << 31));
}

// ---------------------------------------------------------------------------
// General tiled GEMM
// ---------------------------------------------------------------------------

static constexpr int BM         = 128;
static constexpr int BN         = 128;
static constexpr int BK         = 16;
static constexpr int TM         = 8;
static constexpr int TN         = 8;
static constexpr int BLOCK_SIZE = (BM / TM) * (BN / TN);  // 256

__global__ void pot_gemm_kernel(
    const float*   __restrict__ A,
    const uint8_t* __restrict__ B,
    float*         __restrict__ C,
    int M, int N, int K)
{
    const int tid       = threadIdx.x;
    // Thread's position within the BM/TM × BN/TN grid of output tiles
    const int trow      = tid / (BN / TN);
    const int tcol      = tid % (BN / TN);
    // This block's origin in C
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    // Shared memory.
    // +1 column on A_smem: removes bank conflicts when threads in a warp
    //   read the same column (ki) from different rows (2-way period of BK=16
    //   without padding → every 8th row aliases the same bank).
    // +4 bytes on B_smem: same reasoning for uint8 (bank period = 4 bytes).
    __shared__ float   A_smem[BM][BK + 1];
    __shared__ uint8_t B_smem[BN][BK + 4];

    // Per-thread register accumulator for its TM × TN output tile
    float acc[TM][TN] = {};

    for (int k = 0; k < K; k += BK) {

        // --- Load A tile [BM × BK] ---
        // Consecutive threads pick consecutive (r,c) pairs → coalesced reads.
        for (int idx = tid; idx < BM * BK; idx += BLOCK_SIZE) {
            const int r  = idx / BK;
            const int c  = idx % BK;
            const int gr = block_row + r;
            const int gk = k + c;
            A_smem[r][c] = (gr < M && gk < K) ? A[gr * K + gk] : 0.0f;
        }

        // --- Load B tile [BN × BK] as uint8 ---
        for (int idx = tid; idx < BN * BK; idx += BLOCK_SIZE) {
            const int r  = idx / BK;
            const int c  = idx % BK;
            const int gn = block_col + r;
            const int gk = k + c;
            B_smem[r][c] = (gn < N && gk < K) ? B[gn * K + gk] : 0;
        }

        __syncthreads();

        // --- Register-level compute: TM A frags × TN decoded B frags = TM×TN FMAs ---
        #pragma unroll
        for (int ki = 0; ki < BK; ++ki) {
            float a_frag[TM];
            float b_frag[TN];

            #pragma unroll
            for (int m = 0; m < TM; ++m)
                a_frag[m] = A_smem[trow * TM + m][ki];

            // Decode B on the fly: 3 int ops + reinterpret per element
            #pragma unroll
            for (int n = 0; n < TN; ++n)
                b_frag[n] = decode_pot(B_smem[tcol * TN + n][ki]);

            #pragma unroll
            for (int m = 0; m < TM; ++m)
                #pragma unroll
                for (int n = 0; n < TN; ++n)
                    acc[m][n] += a_frag[m] * b_frag[n];
        }

        __syncthreads();
    }

    // --- Write output (overwrite, no accumulate into C) ---
    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        #pragma unroll
        for (int n = 0; n < TN; ++n) {
            const int row = block_row + trow * TM + m;
            const int col = block_col + tcol * TN + n;
            if (row < M && col < N)
                C[row * N + col] = acc[m][n];
        }
    }
}

// ---------------------------------------------------------------------------
// Small-M kernel: one warp per (m, n) output element, K-stride + shfl_down.
//
// Grid: (ceil(N/WARPS_N), ceil(M/WARPS_M))
// Block: WARPS_N × WARPS_M warps = WARPS_N * WARPS_M * 32 threads.
//
// Each warp computes one dot product C[m, n] = sum_k A[m,k] * decode(B[n,k]).
// A rows (K floats each) fit in L2; B rows are independent → coalesced reads.
// Handles M=1 through M=SMALL_M_THRESH efficiently at any N, K.
// ---------------------------------------------------------------------------

static constexpr int WARPS_N = 8;   // warps along N per block
static constexpr int WARPS_M = 4;   // warps along M per block
static constexpr int SMALLM_BLOCK = WARPS_N * WARPS_M * 32;

__global__ void pot_smallm_kernel(
    const float*   __restrict__ A,
    const uint8_t* __restrict__ B,
    float*         __restrict__ C,
    int M, int N, int K)
{
    const int lane    = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;                  // flat warp in block
    const int wn      = warp_id % WARPS_N;                 // warp's N index
    const int wm      = warp_id / WARPS_N;                 // warp's M index
    const int n       = blockIdx.x * WARPS_N + wn;
    const int m       = blockIdx.y * WARPS_M + wm;

    if (n >= N || m >= M) return;

    const float*   a_row = A + (size_t)m * K;
    const uint8_t* b_row = B + (size_t)n * K;

    float sum = 0.0f;
    for (int k = lane; k < K; k += 32)
        sum += a_row[k] * decode_pot(b_row[k]);

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, off);

    if (lane == 0) C[(size_t)m * N + n] = sum;
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

// Use small-M warp-reduction kernel up to this threshold.
// Above it the tiled GEMM provides enough blocks to saturate the GPU.
static constexpr int SMALL_M_THRESH = 16;

void mm_cuda_pot_v04(
    const float*   A_d,
    const uint8_t* B_d,
    float*         C_d,
    int M, int N, int K)
{
    if (M <= SMALL_M_THRESH) {
        const dim3 grid((N + WARPS_N - 1) / WARPS_N,
                        (M + WARPS_M - 1) / WARPS_M);
        pot_smallm_kernel<<<grid, SMALLM_BLOCK>>>(A_d, B_d, C_d, M, N, K);
    } else {
        const dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        pot_gemm_kernel<<<grid, BLOCK_SIZE>>>(A_d, B_d, C_d, M, N, K);
    }
}
