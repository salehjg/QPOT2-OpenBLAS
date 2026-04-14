/*
 * Tiled, multithreaded AVX-512 PoT matmul kernel.
 *
 * Tiling strategy (modeled after OpenBLAS GOTO-style blocking):
 *
 *   Outer loop : K tiles of size KC  — keeps A & B working sets in L2
 *   Middle loop: M rows in blocks of MR — reuses B expansion across rows
 *   Inner loop : N columns one at a time (each B column strip is KC bytes)
 *
 * Register micro-kernel: MR rows × 1 column.  B uint8→int32 expansion
 * (extract + sign detect + shift) is done once per 64-element K step and
 * reused across all MR rows.
 *
 * OpenMP parallelises the M-block loop (each thread owns a contiguous
 * chunk of output rows).
 *
 * Tile sizes tuned for Xeon Gold 5218 (Cascade Lake):
 *   L1d = 32 KB, L2 = 1 MB, L3 = 22 MB shared.
 *
 *   KC = 256  →  A strip per row   = 1 KB
 *                B column strip     = 256 B
 *                B panel (all cols) = N×256 B  (1 MB for N=4096, fits L2)
 *   MR = 8   →  8 zmm accumulators + 5 constants + temps ≤ 32 zmm regs
 */

#include "mm_avx512_f32u8.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <immintrin.h>
#include <limits>
#include <omp.h>

// ---------------------------------------------------------------------------
// Tile sizes
// ---------------------------------------------------------------------------
static constexpr int KC = 256;   // K-tile: must be a multiple of 64
static constexpr int MR = 8;     // row micro-tile

// ---------------------------------------------------------------------------
// Macro: process one 16-element "pass" of the 64-byte B chunk.
//
// LANE = compile-time index (0-3) for _mm512_extracti64x2_epi64
// OFFSET = element offset within the 64-element group (0, 16, 32, 48)
//
// Precomputes B expansion once, then applies it to `mlen` rows of A.
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

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
void mm_avx512_f32_1_u8_1_v01(
    const float*   __restrict__ A,
    const uint8_t* __restrict__ B,
    float*         __restrict__ C,
    int M, int N, int K)
{
    assert(K % 64 == 0 && "K must be a multiple of 64");

    // AVX-512 constants (broadcast once, live in zmm registers)
    const __m512i neg_detect = _mm512_set1_epi32(0x1F);        // 31
    const __m512i sign_strip = _mm512_set1_epi32(0x20);        // 32
    const __m512i sign_bit   = _mm512_set1_epi32(0x80000000u); // IEEE sign
    const __m512  abs_mask   = _mm512_castsi512_ps(
                                   _mm512_set1_epi32(0x7FFFFFFF));
    const __m512  inf_val    = _mm512_set1_ps(
                                   std::numeric_limits<float>::infinity());

    // Zero the output matrix
    std::memset(C, 0, (size_t)M * N * sizeof(float));

    // ---- K tile loop (outermost, sequential) ----
    for (int kk = 0; kk < K; kk += KC) {
        const int klen = std::min(KC, K - kk);

        // ---- M row-block loop (parallel) ----
        #pragma omp parallel for schedule(static)
        for (int ii = 0; ii < M; ii += MR) {
            const int mlen = std::min(MR, M - ii);

            // ---- N column loop ----
            for (int j = 0; j < N; ++j) {

                // One zmm accumulator per row in the micro-tile
                __m512 acc[MR];
                for (int r = 0; r < mlen; ++r)
                    acc[r] = _mm512_setzero_ps();

                // ---- inner K loop (step 64 uint8 = 512 bits) ----
                for (int k = kk; k < kk + klen; k += 64) {
                    const __m512i vec_b = _mm512_loadu_si512(
                        reinterpret_cast<const __m512i*>(
                            B + (size_t)j * K + k));

                    POT_PASS(0,  0);   // elements  0-15
                    POT_PASS(1, 16);   // elements 16-31
                    POT_PASS(2, 32);   // elements 32-47
                    POT_PASS(3, 48);   // elements 48-63
                }

                // Horizontal reduction + accumulate into C
                for (int r = 0; r < mlen; ++r) {
                    C[(size_t)(ii + r) * N + j] +=
                        _mm512_reduce_add_ps(acc[r]);
                }
            }
        }
    }
}

#undef POT_PASS
