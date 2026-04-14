#pragma once
// kernel_v03.hpp — V02 + ILP improvements.
//
// Matrix layout:
//   A : [M x K] row-major, float32
//   B : [N x K] pre-transposed, PoT-encoded, uint8
//   C : [M x N] row-major, float32
//
// K and N must be multiples of 16.

#include <cstdint>

/// GOTO-style tiled, packed, multithreaded AVX-512 PoT matmul (v03).
void mm_avx512_f32_1_u8_1_v03(
    const float*   __restrict__ A,
    const uint8_t* __restrict__ B,
    float*         __restrict__ C,
    int M, int N, int K);

/// Pre-pack B into tile-friendly layout (call once at model load).
/// Bp must be at least N * K bytes, 64-byte aligned.
void pot_pack_b_v03(
    const uint8_t* __restrict__ B,
    uint8_t*       __restrict__ Bp,
    int N, int K);

/// GEMM using pre-packed B (skips B packing — for inference hot path).
void mm_avx512_f32_1_u8_1_v03_packed(
    const float*   __restrict__ A,
    const uint8_t* __restrict__ Bp,
    float*         __restrict__ C,
    int M, int N, int K);
