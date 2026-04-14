/*
 * v02: GOTO-style tiled PoT GEMM with packing and AVX-512 micro-kernel.
 *
 * Threading strategy (fixes over original):
 *   - Single parallel region per K-tile (eliminates double fork-join)
 *   - Large M: parallelize over M tiles (classic GOTO)
 *   - Small M: parallelize over N columns (each thread owns an N-stripe)
 *
 * GOTO algorithm (5 loops):
 *   Loop 1 (kk): K tiles of size KC          — sequential
 *   Loop 2     : Pack B for this K tile       — parallel (collaborative)
 *   Loop 3 (ii): M tiles of size MC           — parallel (omp for)
 *   Loop 4     : Pack A for this M×K tile     — per-thread
 *   Loop 5 (jj): N tiles of size NR           — sequential per thread
 *   Loop 6 (ir): MR blocks within MC          — sequential
 *   Loop 7     : Micro-kernel (MR×NR×KB)      — innermost
 *
 * Tile sizes tuned for Xeon Gold 5218 (Cascade Lake):
 *   KC = 512  — B panel (N×512 uint8)  in L3,  A strip (MR×512×4) in L1
 *   MC = 128  — A panel (128×512×4 = 256 KB)   in L2
 *   MR = 8    — 8 accumulators + constants ≤ 32 zmm
 *   NR = 16   — one zmm of uint8 → 16 int32 values
 *
 * Packed formats:
 *   A_packed: MR contiguous floats per K step.
 *             Layout: [k=0: MR floats][k=1: MR floats]...[k=KB-1: MR floats]
 *             Panels are concatenated: panel_0 || panel_1 || ...
 *
 *   B_packed: NR contiguous uint8 per K step.
 *             Layout: [k=0: NR uint8][k=1: NR uint8]...[k=KB-1: NR uint8]
 *             Panels are concatenated: panel_0 || panel_1 || ...
 */

#include "mm_avx512_f32u8_v02.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <immintrin.h>
#include <limits>
#include <omp.h>
#include <vector>

// ---------------------------------------------------------------------------
// Tile sizes
// ---------------------------------------------------------------------------
static constexpr int MR = 8;
static constexpr int NR = 16;
static constexpr int KC = 512;
static constexpr int MC = 128;

// ---------------------------------------------------------------------------
// Pack A: rows [ii, ii+mb) × cols [kk, kk+kb) into MR-wide panels.
// Output: ceil(mb/MR) panels, each MR×kb floats (zero-padded if partial).
// ---------------------------------------------------------------------------
static void pack_a(
    const float* __restrict__ A, float* __restrict__ Ap,
    int K, int ii, int kk, int mb, int kb)
{
    float* dst = Ap;
    for (int ir = 0; ir < mb; ir += MR) {
        const int mr = std::min(MR, mb - ir);
        // Read each A row contiguously (stride-1), write with stride MR.
        // Original had inner loop over rows → stride-K reads = cache miss per row.
        for (int r = 0; r < mr; ++r) {
            const float* src = A + (size_t)(ii + ir + r) * K + kk;
            for (int k = 0; k < kb; ++k)
                dst[k * MR + r] = src[k];
        }
        for (int r = mr; r < MR; ++r) {
            for (int k = 0; k < kb; ++k)
                dst[k * MR + r] = 0.0f;
        }
        dst += MR * kb;
    }
}

// ---------------------------------------------------------------------------
// Pack one NR-wide B panel: columns [jj, jj+NR) × K range [kk, kk+kb).
// Output: NR×kb uint8 in interleaved format.
// ---------------------------------------------------------------------------
static void pack_b_panel(
    const uint8_t* __restrict__ B, uint8_t* __restrict__ Bp,
    int K, int jj, int kk, int kb, int actual_nr)
{
    // Read each B column contiguously (stride-1), write with stride NR.
    // Original had inner loop over columns → stride-K reads = massive cache waste.
    for (int c = 0; c < actual_nr; ++c) {
        const uint8_t* src = B + (size_t)(jj + c) * K + kk;
        for (int k = 0; k < kb; ++k)
            Bp[k * NR + c] = src[k];
    }
    if (actual_nr < NR) {
        for (int k = 0; k < kb; ++k)
            for (int c = actual_nr; c < NR; ++c)
                Bp[k * NR + c] = 0;
    }
}

// ---------------------------------------------------------------------------
// Micro-kernel: MR rows × NR=16 columns, kb K steps (compile-time MR).
//
// Template on MR_CT so the compiler fully unrolls the row loop and keeps
// all accumulators in zmm registers.  The runtime-mr variant is only used
// for the last partial M-block.
// ---------------------------------------------------------------------------
template <int MR_CT>
static inline __attribute__((always_inline)) void pot_microkernel_t(
    int kb,
    const float* __restrict__ AO,
    const uint8_t* __restrict__ BO,
    float* __restrict__ C_ptr, int N)
{
    const __m512i sign_bit_v = _mm512_set1_epi32((int)0x80000000u);
    const __m512i neg_detect = _mm512_set1_epi32(0x1F);
    const __m512i sign_strip = _mm512_set1_epi32(0x20);

    __m512 acc[MR_CT];
    for (int r = 0; r < MR_CT; ++r)
        acc[r] = _mm512_setzero_ps();

    for (int k = 0; k < kb; ++k) {
        // ---- B expansion (done once, reused across MR_CT rows) ----
        const __m128i b_raw = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(BO + k * NR));
        __m512i b32 = _mm512_cvtepi8_epi32(b_raw);
        const __mmask16 neg = _mm512_cmpgt_epi32_mask(b32, neg_detect);
        b32 = _mm512_mask_sub_epi32(b32, neg, b32, sign_strip);
        const __m512i shifted = _mm512_slli_epi32(b32, 23);

        // ---- Per-row PoT: broadcast A[r] → add exponent → flip sign ----
        const int* a_base = reinterpret_cast<const int*>(AO + k * MR);
        for (int r = 0; r < MR_CT; ++r) {
            const __m512i a_int = _mm512_set1_epi32(a_base[r]);
            __m512i res = _mm512_add_epi32(a_int, shifted);
            res = _mm512_mask_xor_epi32(res, neg, res, sign_bit_v);
            acc[r] = _mm512_add_ps(acc[r], _mm512_castsi512_ps(res));
        }
    }

    // ---- SAVE: C[r, 0:16] += acc[r] ----
    for (int r = 0; r < MR_CT; ++r) {
        __m512 c_old = _mm512_loadu_ps(C_ptr + (size_t)r * N);
        _mm512_storeu_ps(C_ptr + (size_t)r * N,
                         _mm512_add_ps(c_old, acc[r]));
    }
}

// Runtime-mr fallback for edge tiles
static inline void pot_microkernel(
    int mr, int kb,
    const float* __restrict__ AO,
    const uint8_t* __restrict__ BO,
    float* __restrict__ C_ptr, int N)
{
    const __m512i sign_bit_v = _mm512_set1_epi32((int)0x80000000u);
    const __m512i neg_detect = _mm512_set1_epi32(0x1F);
    const __m512i sign_strip = _mm512_set1_epi32(0x20);

    __m512 acc[MR];
    for (int r = 0; r < MR; ++r)
        acc[r] = _mm512_setzero_ps();

    for (int k = 0; k < kb; ++k) {
        const __m128i b_raw = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(BO + k * NR));
        __m512i b32 = _mm512_cvtepi8_epi32(b_raw);
        const __mmask16 neg = _mm512_cmpgt_epi32_mask(b32, neg_detect);
        b32 = _mm512_mask_sub_epi32(b32, neg, b32, sign_strip);
        const __m512i shifted = _mm512_slli_epi32(b32, 23);

        const int* a_base = reinterpret_cast<const int*>(AO + k * MR);
        for (int r = 0; r < mr; ++r) {
            const __m512i a_int = _mm512_set1_epi32(a_base[r]);
            __m512i res = _mm512_add_epi32(a_int, shifted);
            res = _mm512_mask_xor_epi32(res, neg, res, sign_bit_v);
            acc[r] = _mm512_add_ps(acc[r], _mm512_castsi512_ps(res));
        }
    }

    for (int r = 0; r < mr; ++r) {
        __m512 c_old = _mm512_loadu_ps(C_ptr + (size_t)r * N);
        _mm512_storeu_ps(C_ptr + (size_t)r * N,
                         _mm512_add_ps(c_old, acc[r]));
    }
}

// Dispatch: compile-time MR for common case, runtime for edge
static inline void microkernel_dispatch(
    int mr, int kb,
    const float* __restrict__ AO,
    const uint8_t* __restrict__ BO,
    float* __restrict__ C_ptr, int N)
{
    if (mr == MR)
        pot_microkernel_t<MR>(kb, AO, BO, C_ptr, N);
    else
        pot_microkernel(mr, kb, AO, BO, C_ptr, N);
}

// ---------------------------------------------------------------------------
// Helper: compute an M-tile's contribution for a range of N columns.
// Called from both the M-parallel and N-parallel paths.
// ---------------------------------------------------------------------------
static inline void compute_mc_tile(
    const float* __restrict__ Ap,
    const uint8_t* __restrict__ Bp,   // packed B, offset to jj_start
    float* __restrict__ C,
    int mb, int kb, int N,
    int ii, int jj_start, int jj_end)
{
    for (int jj = jj_start; jj < jj_end; jj += NR) {
        const uint8_t* bp = Bp + (size_t)(jj - jj_start) * kb;
        const float* ap = Ap;

        for (int ir = 0; ir < mb; ir += MR) {
            const int mr = std::min(MR, mb - ir);
            microkernel_dispatch(mr, kb, ap, bp,
                C + (size_t)(ii + ir) * N + jj, N);
            ap += MR * kb;
        }
    }
}

// ---------------------------------------------------------------------------
// M-parallel path: classic GOTO (parallelize over M tiles, shared packed B).
// Used when there are enough M-tiles to keep all threads busy.
// ---------------------------------------------------------------------------
static void goto_m_parallel(
    const float*   __restrict__ A,
    const uint8_t* __restrict__ B,
    float*         __restrict__ C,
    int M, int N, int K)
{
    const int max_kb = std::min(KC, K);
    const int nthreads = omp_get_max_threads();

    uint8_t* Bp = aligned_new<uint8_t>((size_t)N * max_kb);
    std::vector<float*> Ap_vec(nthreads);
    for (int t = 0; t < nthreads; ++t)
        Ap_vec[t] = aligned_new<float>((size_t)MC * max_kb);

    // ==== Loop 1: K tiles ====
    for (int kk = 0; kk < K; kk += KC) {
        const int kb = std::min(KC, K - kk);

        // ---- Single parallel region per K-tile ----
        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            float* Ap = Ap_vec[tid];

            // ==== Loop 2: Pack B collaboratively ====
            #pragma omp for schedule(static)
            for (int jj = 0; jj < N; jj += NR) {
                const int nr = std::min(NR, N - jj);
                pack_b_panel(B, Bp + (size_t)jj * kb, K, jj, kk, kb, nr);
            }
            // implicit barrier ensures B is fully packed before compute

            // ==== Loop 3: M tiles (parallel) ====
            #pragma omp for schedule(static)
            for (int ii = 0; ii < M; ii += MC) {
                const int mb = std::min(MC, M - ii);

                // ==== Loop 4: Pack A (per-thread) ====
                pack_a(A, Ap, K, ii, kk, mb, kb);

                // ==== Loop 5-7: N tiles → MR blocks → microkernel ====
                compute_mc_tile(Ap, Bp, C, mb, kb, N, ii, 0, N);
            }
        } // end single parallel region
    }

    for (int t = 0; t < nthreads; ++t)
        std::free(Ap_vec[t]);
    std::free(Bp);
}

// ---------------------------------------------------------------------------
// N-parallel path: each thread owns a contiguous stripe of N columns.
// Used when M is too small for M-parallelism (tall-skinny / inference).
//
// Each thread independently:
//   - Packs its own B panels (thread-local, no sharing needed)
//   - Packs A redundantly (small since M is small)
//   - Computes its N-stripe of the output
//
// No barriers within K-tiles — threads are fully independent.
// No false sharing — N-stripes are NR-aligned (= cache-line aligned).
// ---------------------------------------------------------------------------
static void goto_n_parallel(
    const float*   __restrict__ A,
    const uint8_t* __restrict__ B,
    float*         __restrict__ C,
    int M, int N, int K)
{
    const int max_kb = std::min(KC, K);
    const int n_nr_tiles = N / NR;   // N is a multiple of NR (asserted)

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int nt  = omp_get_num_threads();

        // Divide NR-tiles evenly (handle remainder)
        const int base  = n_nr_tiles / nt;
        const int extra = n_nr_tiles % nt;
        const int my_start = (tid < extra) ? tid * (base + 1)
                                           : extra * (base + 1) + (tid - extra) * base;
        const int my_count = base + (tid < extra ? 1 : 0);
        const int jj_start = my_start * NR;
        const int jj_end   = (my_start + my_count) * NR;
        const int local_n  = jj_end - jj_start;

        if (local_n > 0) {
            // Thread-local packing buffers
            float*   Ap = aligned_new<float>((size_t)MC * max_kb);
            uint8_t* Bp = aligned_new<uint8_t>((size_t)local_n * max_kb);

            // Zero this thread's C columns
            for (int i = 0; i < M; ++i)
                std::memset(C + (size_t)i * N + jj_start, 0,
                            local_n * sizeof(float));

            for (int kk = 0; kk < K; kk += KC) {
                const int kb = std::min(KC, K - kk);

                // Pack B for this thread's columns
                for (int jj = jj_start; jj < jj_end; jj += NR) {
                    const int nr = std::min(NR, jj_end - jj);
                    pack_b_panel(B, Bp + (size_t)(jj - jj_start) * kb,
                                 K, jj, kk, kb, nr);
                }

                // Process all M tiles for this thread's N columns
                for (int ii = 0; ii < M; ii += MC) {
                    const int mb = std::min(MC, M - ii);
                    pack_a(A, Ap, K, ii, kk, mb, kb);
                    compute_mc_tile(Ap, Bp, C, mb, kb, N, ii,
                                    jj_start, jj_end);
                }
            }

            std::free(Ap);
            std::free(Bp);
        }
    }
}

// ---------------------------------------------------------------------------
// Driver: GOTO-style tiled PoT GEMM — chooses M-parallel or N-parallel
// based on problem shape and thread count.
// ---------------------------------------------------------------------------
void mm_avx512_f32_1_u8_1_v02(
    const float*   __restrict__ A,
    const uint8_t* __restrict__ B,
    float*         __restrict__ C,
    int M, int N, int K)
{
    assert(K % NR == 0 && "K must be a multiple of 16");
    assert(N % NR == 0 && "N must be a multiple of 16");

    const int nthreads = omp_get_max_threads();
    const int m_tiles = (M + MC - 1) / MC;

    if (m_tiles >= nthreads) {
        // Enough M-tiles: classic GOTO with M-parallelism
        // Zero C here (could also be done in parallel inside, but memset
        // is fast and simple for the large-M case)
        std::memset(C, 0, (size_t)M * N * sizeof(float));
        goto_m_parallel(A, B, C, M, N, K);
    } else {
        // Small M (tall-skinny): parallelize over N columns.
        // C is zeroed per-thread inside goto_n_parallel.
        goto_n_parallel(A, B, C, M, N, K);
    }
}

// ===========================================================================
// Pre-packed B variants (for inference: pack weights once at model load)
// ===========================================================================

// Pre-packed B layout (same total size N × K bytes, reordered):
//   For each K-tile kk (size KB = min(KC, K-kk)):
//     For each N-tile jj (size NR):
//       NR × KB contiguous uint8 at offset kk*N + jj*KB
//   where kk iterates over multiples of KC.
//
// Within each NR×KB block: Bp_block[k * NR + c] for k in [0,KB), c in [0,NR).

void pot_pack_b_v02(
    const uint8_t* __restrict__ B,
    uint8_t*       __restrict__ Bp,
    int N, int K)
{
    #pragma omp parallel for schedule(static) collapse(2)
    for (int kk = 0; kk < K; kk += KC) {
        for (int jj = 0; jj < N; jj += NR) {
            const int kb = std::min(KC, K - kk);
            const int nr = std::min(NR, N - jj);
            pack_b_panel(B, Bp + (size_t)kk * N + (size_t)jj * kb,
                         K, jj, kk, kb, nr);
        }
    }
}

// ---------------------------------------------------------------------------
// M-parallel path using pre-packed B
// ---------------------------------------------------------------------------
static void goto_m_parallel_packed(
    const float*   __restrict__ A,
    const uint8_t* __restrict__ Bp,
    float*         __restrict__ C,
    int M, int N, int K)
{
    const int max_kb = std::min(KC, K);
    const int nthreads = omp_get_max_threads();

    std::vector<float*> Ap_vec(nthreads);
    for (int t = 0; t < nthreads; ++t)
        Ap_vec[t] = aligned_new<float>((size_t)MC * max_kb);

    for (int kk = 0; kk < K; kk += KC) {
        const int kb = std::min(KC, K - kk);
        const uint8_t* bp_tile = Bp + (size_t)kk * N;

        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            float* Ap = Ap_vec[tid];

            #pragma omp for schedule(static)
            for (int ii = 0; ii < M; ii += MC) {
                const int mb = std::min(MC, M - ii);
                pack_a(A, Ap, K, ii, kk, mb, kb);
                compute_mc_tile(Ap, bp_tile, C, mb, kb, N, ii, 0, N);
            }
        }
    }

    for (int t = 0; t < nthreads; ++t)
        std::free(Ap_vec[t]);
}

// ---------------------------------------------------------------------------
// N-parallel path using pre-packed B
// ---------------------------------------------------------------------------
static void goto_n_parallel_packed(
    const float*   __restrict__ A,
    const uint8_t* __restrict__ Bp,
    float*         __restrict__ C,
    int M, int N, int K)
{
    const int max_kb = std::min(KC, K);
    const int n_nr_tiles = N / NR;

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int nt  = omp_get_num_threads();

        const int base  = n_nr_tiles / nt;
        const int extra = n_nr_tiles % nt;
        const int my_start = (tid < extra) ? tid * (base + 1)
                                           : extra * (base + 1) + (tid - extra) * base;
        const int my_count = base + (tid < extra ? 1 : 0);
        const int jj_start = my_start * NR;
        const int jj_end   = (my_start + my_count) * NR;
        const int local_n  = jj_end - jj_start;

        if (local_n > 0) {
            float* Ap = aligned_new<float>((size_t)MC * max_kb);

            // Zero this thread's C columns
            for (int i = 0; i < M; ++i)
                std::memset(C + (size_t)i * N + jj_start, 0,
                            local_n * sizeof(float));

            for (int kk = 0; kk < K; kk += KC) {
                const int kb = std::min(KC, K - kk);
                // Pre-packed B for this K-tile, adjusted to thread's column start
                // so compute_mc_tile's (jj - jj_start) offset resolves correctly
                const uint8_t* bp_base = Bp + (size_t)kk * N
                                            + (size_t)jj_start * kb;

                for (int ii = 0; ii < M; ii += MC) {
                    const int mb = std::min(MC, M - ii);
                    pack_a(A, Ap, K, ii, kk, mb, kb);
                    compute_mc_tile(Ap, bp_base, C, mb, kb, N, ii,
                                    jj_start, jj_end);
                }
            }

            std::free(Ap);
        }
    }
}

// ---------------------------------------------------------------------------
// Driver: pre-packed B variant
// ---------------------------------------------------------------------------
void mm_avx512_f32_1_u8_1_v02_packed(
    const float*   __restrict__ A,
    const uint8_t* __restrict__ Bp,
    float*         __restrict__ C,
    int M, int N, int K)
{
    assert(K % NR == 0 && "K must be a multiple of 16");
    assert(N % NR == 0 && "N must be a multiple of 16");

    const int nthreads = omp_get_max_threads();
    const int m_tiles = (M + MC - 1) / MC;

    if (m_tiles >= nthreads) {
        std::memset(C, 0, (size_t)M * N * sizeof(float));
        goto_m_parallel_packed(A, Bp, C, M, N, K);
    } else {
        goto_n_parallel_packed(A, Bp, C, M, N, K);
    }
}
