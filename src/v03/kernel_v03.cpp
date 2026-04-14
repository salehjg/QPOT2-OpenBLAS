/*
 * v03: V02 + ILP improvements applied to the micro-kernel and tiling.
 *
 * Changes vs V02
 * ──────────────
 * #1  MR = 16 (was 8)
 *       Doubles the rows processed per micro-kernel call, amortising the
 *       fixed B-decode cost over twice as many accumulate operations.
 *
 * #2  KC = 1024 (was 512)
 *       Packed B is uint8 (4× smaller than float32), so doubling KC keeps
 *       the packed-B working set at the same effective byte footprint.
 *       Larger KC reduces the number of K-tile iterations and associated
 *       pack/barrier overhead.
 *
 * #3  K-loop unrolled by 2 (in pot_microkernel_t only)
 *       Decoding B[k] and B[k+1] back-to-back lets the out-of-order engine
 *       overlap the 5-cycle decode chain of k+1 with the per-row accumulate
 *       work for k, hiding decode latency.
 *
 * #4  Explicit B prefetch (_mm_prefetch T0, B_PREFETCH_DIST iterations ahead)
 *       B is accessed sequentially at 16 bytes/k; prefetching 8 iterations
 *       ahead (128 bytes) hides L2→L1 latency for the next two cache lines.
 *
 * #6  Explicit A prefetch (_mm_prefetch T1, two MR-blocks ahead)
 *       With KC=1024 and MR=16 the A panel is 64 KB — twice L1. Prefetching
 *       the next micro-kernel's A rows into L2 reduces stalls.
 *
 * #7  Faster B decode: cmpgt_mask + and (independent) vs cmpgt_mask → masked_sub (serial)
 *       Replacing the masked subtraction with an unconditional AND decouples
 *       the exponent-strip from the sign detection. The critical path to
 *       'shifted' drops from 8 → 5 cycles:
 *         old: movsxbd(3) → cmpgt(3) → mask_sub(1) → slld(1) = 8 cy
 *         new: movsxbd(3) → and(1)   → slld(1)               = 5 cy
 *       The sign mask from cmpgt runs in parallel and feeds only the cheap
 *       maskz_mov (1 cy), which is precomputed once per k outside the row loop.
 *
 * NOT included: non-temporal C stores (#5)
 *   With K > KC there are multiple kk passes; each pass reads back C written
 *   by the previous one. NT stores bypass L3 and would force those reads from
 *   DRAM, increasing total traffic. NT stores are only net-positive for single-
 *   pass (K ≤ KC) or when C exceeds L3, neither of which is the common case.
 */

#include "kernel_v03.hpp"
#include "pot_common.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <immintrin.h>
#include <omp.h>
#include <vector>

static constexpr int MR = 16;
static constexpr int NR = 16;
static constexpr int KC = 1024;
static constexpr int MC = 128;
static constexpr int B_PREFETCH_DIST = 8;  // k-iterations ahead for B prefetch
static constexpr int A_PREFETCH_DIST = 2;  // MR-blocks ahead for A prefetch (T1)

// ---------------------------------------------------------------------------
// Packing (logic identical to v02; MR constant drives the stride)
// ---------------------------------------------------------------------------

static void pack_a(
    const float* __restrict__ A, float* __restrict__ Ap,
    int K, int ii, int kk, int mb, int kb)
{
    float* dst = Ap;
    for (int ir = 0; ir < mb; ir += MR) {
        const int mr = std::min(MR, mb - ir);
        for (int r = 0; r < mr; ++r) {
            const float* src = A + (size_t)(ii + ir + r) * K + kk;
            for (int k = 0; k < kb; ++k)
                dst[k * MR + r] = src[k];
        }
        for (int r = mr; r < MR; ++r)
            for (int k = 0; k < kb; ++k)
                dst[k * MR + r] = 0.0f;
        dst += MR * kb;
    }
}

static void pack_b_panel(
    const uint8_t* __restrict__ B, uint8_t* __restrict__ Bp,
    int K, int jj, int kk, int kb, int actual_nr)
{
    for (int c = 0; c < actual_nr; ++c) {
        const uint8_t* src = B + (size_t)(jj + c) * K + kk;
        for (int k = 0; k < kb; ++k)
            Bp[k * NR + c] = src[k];
    }
    if (actual_nr < NR)
        for (int k = 0; k < kb; ++k)
            for (int c = actual_nr; c < NR; ++c)
                Bp[k * NR + c] = 0;
}

// ---------------------------------------------------------------------------
// Micro-kernel — full MR rows (template, k-unrolled by 2, all improvements)
// ---------------------------------------------------------------------------

template <int MR_CT>
static inline __attribute__((always_inline)) void pot_microkernel_t(
    int kb, const float* __restrict__ AO, const uint8_t* __restrict__ BO,
    float* __restrict__ C_ptr, int N)
{
    const __m512i sign_bit_v = _mm512_set1_epi32((int)0x80000000u);
    // neg_detect doubles as the AND mask for exponent extraction:
    //   cmpgt(b32, 0x1F) detects bit-5 (sign) set for values in [0,63]
    //   and(b32, 0x1F)   strips bit-5 and above, keeping exponent [4:0]
    const __m512i neg_detect = _mm512_set1_epi32(0x1F);

    __m512 acc[MR_CT];
    for (int r = 0; r < MR_CT; ++r)
        acc[r] = _mm512_setzero_ps();

    // Main loop: unrolled by 2.
    // Placing both B-decodes before both row-accumulate blocks lets the
    // out-of-order engine begin the 5-cycle decode chain for k+1 while
    // the per-row work for k is in flight.
    int k = 0;
    for (; k + 1 < kb; k += 2) {

        // Prefetch B two cache-lines ahead (each k step = 16 bytes = 1 line)
        _mm_prefetch(reinterpret_cast<const char*>(
            BO + (k + B_PREFETCH_DIST) * NR), _MM_HINT_T0);

        // --- Decode B[k] ---
        // cmpgt and and are independent: issue in the same cycle.
        // Critical path to shifted0: movsxbd(3cy)→and(1cy)→slld(1cy) = 5cy
        const __m128i b_raw0  = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(BO + k * NR));
        __m512i b32_0         = _mm512_cvtepi8_epi32(b_raw0);
        const __mmask16 neg0  = _mm512_cmpgt_epi32_mask(b32_0, neg_detect);
        b32_0                 = _mm512_and_si512(b32_0, neg_detect);
        const __m512i sh0     = _mm512_slli_epi32(b32_0, 23);
        // Precompute sign contribution once — used by all MR_CT rows below.
        const __m512i sc0     = _mm512_maskz_mov_epi32(neg0, sign_bit_v);

        // --- Decode B[k+1] ---
        const __m128i b_raw1  = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(BO + (k + 1) * NR));
        __m512i b32_1         = _mm512_cvtepi8_epi32(b_raw1);
        const __mmask16 neg1  = _mm512_cmpgt_epi32_mask(b32_1, neg_detect);
        b32_1                 = _mm512_and_si512(b32_1, neg_detect);
        const __m512i sh1     = _mm512_slli_epi32(b32_1, 23);
        const __m512i sc1     = _mm512_maskz_mov_epi32(neg1, sign_bit_v);

        // Prefetch A for upcoming k iterations (T0 = into L1)
        const int* a0 = reinterpret_cast<const int*>(AO + k * MR);
        _mm_prefetch(reinterpret_cast<const char*>(
            a0 + A_PREFETCH_DIST * MR * 2), _MM_HINT_T0);

        // --- Accumulate k into all MR_CT rows ---
        for (int r = 0; r < MR_CT; ++r) {
            const __m512i a_int = _mm512_set1_epi32(a0[r]);
            __m512i res = _mm512_add_epi32(a_int, sh0);
            res = _mm512_xor_si512(res, sc0);
            acc[r] = _mm512_add_ps(acc[r], _mm512_castsi512_ps(res));
        }

        // --- Accumulate k+1 into all MR_CT rows ---
        const int* a1 = reinterpret_cast<const int*>(AO + (k + 1) * MR);
        for (int r = 0; r < MR_CT; ++r) {
            const __m512i a_int = _mm512_set1_epi32(a1[r]);
            __m512i res = _mm512_add_epi32(a_int, sh1);
            res = _mm512_xor_si512(res, sc1);
            acc[r] = _mm512_add_ps(acc[r], _mm512_castsi512_ps(res));
        }
    }

    // Safety tail for odd kb (kb is always even when KC%2==0 and K%NR==0,
    // but included for correctness on arbitrary edge tiles).
    for (; k < kb; ++k) {
        const __m128i b_raw  = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(BO + k * NR));
        __m512i b32          = _mm512_cvtepi8_epi32(b_raw);
        const __mmask16 neg  = _mm512_cmpgt_epi32_mask(b32, neg_detect);
        b32                  = _mm512_and_si512(b32, neg_detect);
        const __m512i sh     = _mm512_slli_epi32(b32, 23);
        const __m512i sc     = _mm512_maskz_mov_epi32(neg, sign_bit_v);
        const int* a_base    = reinterpret_cast<const int*>(AO + k * MR);
        for (int r = 0; r < MR_CT; ++r) {
            const __m512i a_int = _mm512_set1_epi32(a_base[r]);
            __m512i res = _mm512_add_epi32(a_int, sh);
            res = _mm512_xor_si512(res, sc);
            acc[r] = _mm512_add_ps(acc[r], _mm512_castsi512_ps(res));
        }
    }

    for (int r = 0; r < MR_CT; ++r) {
        __m512 c_old = _mm512_loadu_ps(C_ptr + (size_t)r * N);
        _mm512_storeu_ps(C_ptr + (size_t)r * N,
                         _mm512_add_ps(c_old, acc[r]));
    }
}

// ---------------------------------------------------------------------------
// Micro-kernel — edge rows (mr < MR); no k-unroll, improved decode only
// ---------------------------------------------------------------------------

static inline void pot_microkernel(
    int mr, int kb, const float* __restrict__ AO,
    const uint8_t* __restrict__ BO, float* __restrict__ C_ptr, int N)
{
    const __m512i sign_bit_v = _mm512_set1_epi32((int)0x80000000u);
    const __m512i neg_detect = _mm512_set1_epi32(0x1F);

    __m512 acc[MR];
    for (int r = 0; r < MR; ++r) acc[r] = _mm512_setzero_ps();

    for (int k = 0; k < kb; ++k) {
        const __m128i b_raw = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(BO + k * NR));
        __m512i b32         = _mm512_cvtepi8_epi32(b_raw);
        const __mmask16 neg = _mm512_cmpgt_epi32_mask(b32, neg_detect);
        b32                 = _mm512_and_si512(b32, neg_detect);
        const __m512i sh    = _mm512_slli_epi32(b32, 23);
        const __m512i sc    = _mm512_maskz_mov_epi32(neg, sign_bit_v);
        const int* a_base   = reinterpret_cast<const int*>(AO + k * MR);
        for (int r = 0; r < mr; ++r) {
            const __m512i a_int = _mm512_set1_epi32(a_base[r]);
            __m512i res = _mm512_add_epi32(a_int, sh);
            res = _mm512_xor_si512(res, sc);
            acc[r] = _mm512_add_ps(acc[r], _mm512_castsi512_ps(res));
        }
    }

    for (int r = 0; r < mr; ++r) {
        __m512 c_old = _mm512_loadu_ps(C_ptr + (size_t)r * N);
        _mm512_storeu_ps(C_ptr + (size_t)r * N,
                         _mm512_add_ps(c_old, acc[r]));
    }
}

static inline void microkernel_dispatch(
    int mr, int kb, const float* __restrict__ AO,
    const uint8_t* __restrict__ BO, float* __restrict__ C_ptr, int N)
{
    if (mr == MR) pot_microkernel_t<MR>(kb, AO, BO, C_ptr, N);
    else          pot_microkernel(mr, kb, AO, BO, C_ptr, N);
}

// ---------------------------------------------------------------------------
// Shared compute helper — with A prefetch for upcoming micro-kernels
// ---------------------------------------------------------------------------

static inline void compute_mc_tile(
    const float* __restrict__ Ap, const uint8_t* __restrict__ Bp,
    float* __restrict__ C, int mb, int kb, int N,
    int ii, int jj_start, int jj_end)
{
    for (int jj = jj_start; jj < jj_end; jj += NR) {
        const uint8_t* bp = Bp + (size_t)(jj - jj_start) * kb;
        const float* ap = Ap;
        for (int ir = 0; ir < mb; ir += MR) {
            const int mr = std::min(MR, mb - ir);
            // Prefetch A for A_PREFETCH_DIST micro-kernels ahead into L2.
            // With KC=1024 and MR=16 the A panel is 64 KB (> L1), so A lives
            // in L2; T1 prefetch brings the next block before it is needed.
            if (ir + A_PREFETCH_DIST * MR < mb)
                _mm_prefetch(reinterpret_cast<const char*>(
                    ap + (size_t)A_PREFETCH_DIST * MR * kb), _MM_HINT_T1);
            microkernel_dispatch(mr, kb, ap, bp,
                C + (size_t)(ii + ir) * N + jj, N);
            ap += MR * kb;
        }
    }
}

// ---------------------------------------------------------------------------
// M-parallel path
// ---------------------------------------------------------------------------

static void goto_m_parallel(
    const float* __restrict__ A, const uint8_t* __restrict__ B,
    float* __restrict__ C, int M, int N, int K)
{
    const int max_kb  = std::min(KC, K);
    const int nthreads = omp_get_max_threads();

    uint8_t* Bp = aligned_new<uint8_t>((size_t)N * max_kb);
    std::vector<float*> Ap_vec(nthreads);
    for (int t = 0; t < nthreads; ++t)
        Ap_vec[t] = aligned_new<float>((size_t)MC * max_kb);

    for (int kk = 0; kk < K; kk += KC) {
        const int kb = std::min(KC, K - kk);
        #pragma omp parallel
        {
            float* Ap = Ap_vec[omp_get_thread_num()];
            #pragma omp for schedule(static)
            for (int jj = 0; jj < N; jj += NR)
                pack_b_panel(B, Bp + (size_t)jj * kb, K, jj, kk, kb,
                             std::min(NR, N - jj));
            #pragma omp for schedule(static)
            for (int ii = 0; ii < M; ii += MC) {
                const int mb = std::min(MC, M - ii);
                pack_a(A, Ap, K, ii, kk, mb, kb);
                compute_mc_tile(Ap, Bp, C, mb, kb, N, ii, 0, N);
            }
        }
    }

    for (int t = 0; t < nthreads; ++t) std::free(Ap_vec[t]);
    std::free(Bp);
}

// ---------------------------------------------------------------------------
// N-parallel path
// ---------------------------------------------------------------------------

static void goto_n_parallel(
    const float* __restrict__ A, const uint8_t* __restrict__ B,
    float* __restrict__ C, int M, int N, int K)
{
    const int max_kb   = std::min(KC, K);
    const int n_nr_tiles = N / NR;

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int nt  = omp_get_num_threads();
        const int base  = n_nr_tiles / nt, extra = n_nr_tiles % nt;
        const int my_start = (tid < extra) ? tid * (base + 1)
                                           : extra * (base + 1) + (tid - extra) * base;
        const int my_count = base + (tid < extra ? 1 : 0);
        const int jj_start = my_start * NR;
        const int jj_end   = (my_start + my_count) * NR;
        const int local_n  = jj_end - jj_start;

        if (local_n > 0) {
            float*   Ap = aligned_new<float>((size_t)MC * max_kb);
            uint8_t* Bp = aligned_new<uint8_t>((size_t)local_n * max_kb);

            for (int i = 0; i < M; ++i)
                std::memset(C + (size_t)i * N + jj_start, 0, local_n * sizeof(float));

            for (int kk = 0; kk < K; kk += KC) {
                const int kb = std::min(KC, K - kk);
                for (int jj = jj_start; jj < jj_end; jj += NR)
                    pack_b_panel(B, Bp + (size_t)(jj - jj_start) * kb,
                                 K, jj, kk, kb, std::min(NR, jj_end - jj));
                for (int ii = 0; ii < M; ii += MC) {
                    const int mb = std::min(MC, M - ii);
                    pack_a(A, Ap, K, ii, kk, mb, kb);
                    compute_mc_tile(Ap, Bp, C, mb, kb, N, ii, jj_start, jj_end);
                }
            }
            std::free(Ap);
            std::free(Bp);
        }
    }
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

void mm_avx512_f32_1_u8_1_v03(
    const float* __restrict__ A, const uint8_t* __restrict__ B,
    float* __restrict__ C, int M, int N, int K)
{
    assert(K % NR == 0 && "K must be a multiple of 16");
    assert(N % NR == 0 && "N must be a multiple of 16");

    const int nthreads = omp_get_max_threads();
    const int m_tiles  = (M + MC - 1) / MC;

    if (m_tiles >= nthreads) {
        std::memset(C, 0, (size_t)M * N * sizeof(float));
        goto_m_parallel(A, B, C, M, N, K);
    } else {
        goto_n_parallel(A, B, C, M, N, K);
    }
}

// ===========================================================================
// Pre-packed B variants
// ===========================================================================

void pot_pack_b_v03(
    const uint8_t* __restrict__ B, uint8_t* __restrict__ Bp,
    int N, int K)
{
    #pragma omp parallel for schedule(static) collapse(2)
    for (int kk = 0; kk < K; kk += KC)
        for (int jj = 0; jj < N; jj += NR) {
            const int kb = std::min(KC, K - kk);
            pack_b_panel(B, Bp + (size_t)kk * N + (size_t)jj * kb,
                         K, jj, kk, kb, std::min(NR, N - jj));
        }
}

static void goto_m_parallel_packed(
    const float* __restrict__ A, const uint8_t* __restrict__ Bp,
    float* __restrict__ C, int M, int N, int K)
{
    const int max_kb   = std::min(KC, K);
    const int nthreads = omp_get_max_threads();

    std::vector<float*> Ap_vec(nthreads);
    for (int t = 0; t < nthreads; ++t)
        Ap_vec[t] = aligned_new<float>((size_t)MC * max_kb);

    for (int kk = 0; kk < K; kk += KC) {
        const int kb = std::min(KC, K - kk);
        const uint8_t* bp_tile = Bp + (size_t)kk * N;
        #pragma omp parallel
        {
            float* Ap = Ap_vec[omp_get_thread_num()];
            #pragma omp for schedule(static)
            for (int ii = 0; ii < M; ii += MC) {
                const int mb = std::min(MC, M - ii);
                pack_a(A, Ap, K, ii, kk, mb, kb);
                compute_mc_tile(Ap, bp_tile, C, mb, kb, N, ii, 0, N);
            }
        }
    }

    for (int t = 0; t < nthreads; ++t) std::free(Ap_vec[t]);
}

static void goto_n_parallel_packed(
    const float* __restrict__ A, const uint8_t* __restrict__ Bp,
    float* __restrict__ C, int M, int N, int K)
{
    const int max_kb   = std::min(KC, K);
    const int n_nr_tiles = N / NR;

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int nt  = omp_get_num_threads();
        const int base  = n_nr_tiles / nt, extra = n_nr_tiles % nt;
        const int my_start = (tid < extra) ? tid * (base + 1)
                                           : extra * (base + 1) + (tid - extra) * base;
        const int my_count = base + (tid < extra ? 1 : 0);
        const int jj_start = my_start * NR;
        const int jj_end   = (my_start + my_count) * NR;
        const int local_n  = jj_end - jj_start;

        if (local_n > 0) {
            float* Ap = aligned_new<float>((size_t)MC * max_kb);

            for (int i = 0; i < M; ++i)
                std::memset(C + (size_t)i * N + jj_start, 0, local_n * sizeof(float));

            for (int kk = 0; kk < K; kk += KC) {
                const int kb = std::min(KC, K - kk);
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

void mm_avx512_f32_1_u8_1_v03_packed(
    const float* __restrict__ A, const uint8_t* __restrict__ Bp,
    float* __restrict__ C, int M, int N, int K)
{
    assert(K % NR == 0 && "K must be a multiple of 16");
    assert(N % NR == 0 && "N must be a multiple of 16");

    const int nthreads = omp_get_max_threads();
    const int m_tiles  = (M + MC - 1) / MC;

    if (m_tiles >= nthreads) {
        std::memset(C, 0, (size_t)M * N * sizeof(float));
        goto_m_parallel_packed(A, Bp, C, M, N, K);
    } else {
        goto_n_parallel_packed(A, Bp, C, M, N, K);
    }
}
