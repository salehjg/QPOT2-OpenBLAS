/*
 * v01: Tiled, multithreaded AVX-512 PoT matmul kernel.
 *
 * Tiling: K tiles (KC=256), M blocks (MR=8), N columns (NR=1).
 * OpenMP parallelises M-blocks (large M) or N-columns (small M).
 * No packing — reads A and B directly from input layout.
 */

#include "kernel_v01.hpp"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <immintrin.h>
#include <limits>
#include <omp.h>

static constexpr int KC = 256;
static constexpr int MR = 8;

// ---------------------------------------------------------------------------
// One 16-element pass of a 64-byte B chunk.
// ---------------------------------------------------------------------------
#define POT_PASS(LANE, OFFSET)                                              \
    do {                                                                     \
        __m512i b1 = _mm512_cvtepi8_epi32(                                  \
            _mm512_extracti64x2_epi64(vec_b, (LANE)));                      \
        const __mmask16 nmask = _mm512_cmpgt_epi32_mask(b1, neg_detect);   \
        b1 = _mm512_mask_sub_epi32(b1, nmask, b1, sign_strip);             \
        const __m512i shifted = _mm512_slli_epi32(b1, 23);                 \
        for (int r = 0; r < mlen; ++r) {                                    \
            const __m512 va = _mm512_loadu_ps(                              \
                A + (size_t)(ii + r) * K + k + (OFFSET));                   \
            const __mmask16 imask = _mm512_cmp_ps_mask(                    \
                _mm512_and_ps(va, abs_mask), inf_val, _CMP_NEQ_UQ);       \
            __m512i vai = _mm512_castps_si512(va);                          \
            __m512i res = _mm512_mask_add_epi32(vai, imask, vai, shifted);  \
            res = _mm512_mask_xor_epi32(res, nmask, res, sign_bit);        \
            acc[r] = _mm512_add_ps(acc[r], _mm512_castsi512_ps(res));      \
        }                                                                    \
    } while (0)

void mm_avx512_f32_1_u8_1_v01(
    const float*   __restrict__ A,
    const uint8_t* __restrict__ B,
    float*         __restrict__ C,
    int M, int N, int K)
{
    assert(K % 64 == 0 && "K must be a multiple of 64");

    const __m512i neg_detect = _mm512_set1_epi32(0x1F);
    const __m512i sign_strip = _mm512_set1_epi32(0x20);
    const __m512i sign_bit   = _mm512_set1_epi32(0x80000000u);
    const __m512  abs_mask   = _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF));
    const __m512  inf_val    = _mm512_set1_ps(std::numeric_limits<float>::infinity());

    std::memset(C, 0, (size_t)M * N * sizeof(float));

    const int m_blocks = (M + MR - 1) / MR;
    const int nthreads = omp_get_max_threads();

    for (int kk = 0; kk < K; kk += KC) {
        const int klen = std::min(KC, K - kk);

        if (m_blocks >= nthreads) {
            // ---- M-parallel (large M) ----
            #pragma omp parallel for schedule(static)
            for (int ii = 0; ii < M; ii += MR) {
                const int mlen = std::min(MR, M - ii);
                for (int j = 0; j < N; ++j) {
                    __m512 acc[MR];
                    for (int r = 0; r < mlen; ++r) acc[r] = _mm512_setzero_ps();

                    for (int k = kk; k < kk + klen; k += 64) {
                        const __m512i vec_b = _mm512_loadu_si512(
                            reinterpret_cast<const __m512i*>(B + (size_t)j * K + k));
                        POT_PASS(0,  0);
                        POT_PASS(1, 16);
                        POT_PASS(2, 32);
                        POT_PASS(3, 48);
                    }
                    for (int r = 0; r < mlen; ++r)
                        C[(size_t)(ii + r) * N + j] += _mm512_reduce_add_ps(acc[r]);
                }
            }
        } else {
            // ---- N-parallel (small M / tall-skinny) ----
            for (int ii = 0; ii < M; ii += MR) {
                const int mlen = std::min(MR, M - ii);
                #pragma omp parallel for schedule(static)
                for (int j = 0; j < N; ++j) {
                    __m512 acc[MR];
                    for (int r = 0; r < mlen; ++r) acc[r] = _mm512_setzero_ps();

                    for (int k = kk; k < kk + klen; k += 64) {
                        const __m512i vec_b = _mm512_loadu_si512(
                            reinterpret_cast<const __m512i*>(B + (size_t)j * K + k));
                        POT_PASS(0,  0);
                        POT_PASS(1, 16);
                        POT_PASS(2, 32);
                        POT_PASS(3, 48);
                    }
                    for (int r = 0; r < mlen; ++r)
                        C[(size_t)(ii + r) * N + j] += _mm512_reduce_add_ps(acc[r]);
                }
            }
        }
    }
}

#undef POT_PASS
