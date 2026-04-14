#pragma once
// kernel_v01.hpp — Tiled AVX-512 PoT GEMM (no packing, NR=1).
//
// Matrix layout:
//   A : [M x K] row-major, float32
//   B : [N x K] pre-transposed, PoT-encoded, uint8
//   C : [M x N] row-major, float32
//
// K must be a multiple of 64.

#include <cstdint>

void mm_avx512_f32_1_u8_1_v01(
    const float*   __restrict__ A,
    const uint8_t* __restrict__ B,
    float*         __restrict__ C,
    int M, int N, int K);
