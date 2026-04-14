/*
 * v02: GOTO-style tiled PoT matmul with packing.
 *
 * Key difference from v01: A and B are packed into cache-optimal contiguous
 * buffers before the micro-kernel, matching the OpenBLAS GOTO algorithm.
 *
 * Micro-kernel orientation (optimized for row-major C):
 *   MR = 8  rows  — A values are broadcast to all 16 lanes
 *   NR = 16 cols  — B loads 16 uint8, expanded to 16 int32
 *   → C stores are contiguous 16-wide zmm writes
 *
 * This is the transpose of OpenBLAS's MR=16,NR=4 convention because
 * OpenBLAS stores C column-major while we use row-major.
 *
 * Assumptions (for benchmark sizes):
 *   - K is a multiple of 16
 *   - N is a multiple of 16
 *   - A contains no Inf/NaN values (common in inference)
 */

#pragma once

#include "mm_avx512_f32u8.hpp"   // pot_encode, pot_decode, aligned_new

/// GOTO-style tiled, packed, multithreaded AVX-512 PoT matmul.
void mm_avx512_f32_1_u8_1_v02(
    const float*   __restrict__ A,   // [M × K]  row-major
    const uint8_t* __restrict__ B,   // [N × K]  transposed, PoT-encoded
    float*         __restrict__ C,   // [M × N]  row-major
    int M, int N, int K);

/// Pre-pack B into tile-friendly layout (call once at model load).
/// Bp must be at least N × K bytes, 64-byte aligned.
void pot_pack_b_v02(
    const uint8_t* __restrict__ B,   // [N × K]  transposed, PoT-encoded
    uint8_t*       __restrict__ Bp,  // [N × K]  pre-packed output
    int N, int K);

/// GEMM using pre-packed B (skips B packing — for inference hot path).
void mm_avx512_f32_1_u8_1_v02_packed(
    const float*   __restrict__ A,   // [M × K]  row-major
    const uint8_t* __restrict__ Bp,  // [N × K]  pre-packed by pot_pack_b_v02
    float*         __restrict__ C,   // [M × N]  row-major
    int M, int N, int K);
