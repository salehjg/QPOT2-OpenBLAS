#pragma once
// kernel_v04.cuh — CUDA PoT GEMM kernel.
//
// Matrix layout (same as v01-v03):
//   A : [M × K] row-major, float32
//   B : [N × K] pre-transposed, PoT-encoded, uint8
//   C : [M × N] row-major, float32
//
// All pointers passed to mm_cuda_pot_v04 must be device pointers.

#include <cstdint>

/// CUDA PoT GEMM: C = A × decode(B)^T, all device pointers.
/// Selects M=1 warp-reduction kernel or general tiled GEMM automatically.
void mm_cuda_pot_v04(
    const float*   A_d,
    const uint8_t* B_d,
    float*         C_d,
    int M, int N, int K);
