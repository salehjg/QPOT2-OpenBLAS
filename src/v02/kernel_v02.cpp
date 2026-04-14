/*
 * v02: GOTO-style tiled PoT GEMM with packing and AVX-512 micro-kernel.
 *
 * Threading strategy:
 *   - Single parallel region per K-tile (eliminates double fork-join)
 *   - Large M: parallelize over M tiles (classic GOTO)
 *   - Small M: parallelize over N columns (each thread owns an N-stripe)
 *
 * Tile sizes tuned for Xeon Gold 5218 (Cascade Lake):
 *   KC = 512, MC = 128, MR = 8, NR = 16
 */

#include "kernel_v02.hpp"
#include "pot_common.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <immintrin.h>
#include <omp.h>
#include <vector>

static constexpr int MR = 8;
static constexpr int NR = 16;
static constexpr int KC = 512;
static constexpr int MC = 128;

// ---------------------------------------------------------------------------
// Packing
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
// Micro-kernel (compile-time MR for unrolling, runtime fallback for edges)
// ---------------------------------------------------------------------------

template <int MR_CT>
static inline __attribute__((always_inline)) void pot_microkernel_t(
    int kb, const float* __restrict__ AO, const uint8_t* __restrict__ BO,
    float* __restrict__ C_ptr, int N)
{
    const __m512i sign_bit_v = _mm512_set1_epi32((int)0x80000000u);
    const __m512i neg_detect = _mm512_set1_epi32(0x1F);
    const __m512i sign_strip = _mm512_set1_epi32(0x20);

    __m512 acc[MR_CT];
    for (int r = 0; r < MR_CT; ++r)
        acc[r] = _mm512_setzero_ps();

    for (int k = 0; k < kb; ++k) {
        const __m128i b_raw = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(BO + k * NR));
        __m512i b32 = _mm512_cvtepi8_epi32(b_raw);
        const __mmask16 neg = _mm512_cmpgt_epi32_mask(b32, neg_detect);
        b32 = _mm512_mask_sub_epi32(b32, neg, b32, sign_strip);
        const __m512i shifted = _mm512_slli_epi32(b32, 23);

        const int* a_base = reinterpret_cast<const int*>(AO + k * MR);
        for (int r = 0; r < MR_CT; ++r) {
            const __m512i a_int = _mm512_set1_epi32(a_base[r]);
            __m512i res = _mm512_add_epi32(a_int, shifted);
            res = _mm512_mask_xor_epi32(res, neg, res, sign_bit_v);
            acc[r] = _mm512_add_ps(acc[r], _mm512_castsi512_ps(res));
        }
    }

    for (int r = 0; r < MR_CT; ++r) {
        __m512 c_old = _mm512_loadu_ps(C_ptr + (size_t)r * N);
        _mm512_storeu_ps(C_ptr + (size_t)r * N,
                         _mm512_add_ps(c_old, acc[r]));
    }
}

static inline void pot_microkernel(
    int mr, int kb, const float* __restrict__ AO,
    const uint8_t* __restrict__ BO, float* __restrict__ C_ptr, int N)
{
    const __m512i sign_bit_v = _mm512_set1_epi32((int)0x80000000u);
    const __m512i neg_detect = _mm512_set1_epi32(0x1F);
    const __m512i sign_strip = _mm512_set1_epi32(0x20);

    __m512 acc[MR];
    for (int r = 0; r < MR; ++r) acc[r] = _mm512_setzero_ps();

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

static inline void microkernel_dispatch(
    int mr, int kb, const float* __restrict__ AO,
    const uint8_t* __restrict__ BO, float* __restrict__ C_ptr, int N)
{
    if (mr == MR) pot_microkernel_t<MR>(kb, AO, BO, C_ptr, N);
    else           pot_microkernel(mr, kb, AO, BO, C_ptr, N);
}

// ---------------------------------------------------------------------------
// Shared compute helper
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
    const int max_kb = std::min(KC, K);
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
            for (int jj = 0; jj < N; jj += NR) {
                pack_b_panel(B, Bp + (size_t)jj * kb, K, jj, kk, kb,
                             std::min(NR, N - jj));
            }
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
    const int max_kb = std::min(KC, K);
    const int n_nr_tiles = N / NR;

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int nt  = omp_get_num_threads();
        const int base = n_nr_tiles / nt, extra = n_nr_tiles % nt;
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

void mm_avx512_f32_1_u8_1_v02(
    const float* __restrict__ A, const uint8_t* __restrict__ B,
    float* __restrict__ C, int M, int N, int K)
{
    assert(K % NR == 0 && "K must be a multiple of 16");
    assert(N % NR == 0 && "N must be a multiple of 16");

    const int nthreads = omp_get_max_threads();
    const int m_tiles = (M + MC - 1) / MC;

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

void pot_pack_b_v02(
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
    const int max_kb = std::min(KC, K);
    const int n_nr_tiles = N / NR;

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int nt  = omp_get_num_threads();
        const int base = n_nr_tiles / nt, extra = n_nr_tiles % nt;
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

void mm_avx512_f32_1_u8_1_v02_packed(
    const float* __restrict__ A, const uint8_t* __restrict__ Bp,
    float* __restrict__ C, int M, int N, int K)
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
