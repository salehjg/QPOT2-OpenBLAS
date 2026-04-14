#pragma once
#include <cstddef>
#include <cstdint>

// POT style: integer bit-trick (vadd + vxor + vfadd)
void mm_rvv_pot_lmul1(const float* A, const uint8_t* B, float* C, int M, int N, int K);
void mm_rvv_pot_lmul2(const float* A, const uint8_t* B, float* C, int M, int N, int K);
void mm_rvv_pot_lmul4(const float* A, const uint8_t* B, float* C, int M, int N, int K);
void mm_rvv_pot_lmul1_packed(const float* A, const uint8_t* Bp, float* C, int M, int N, int K);
void mm_rvv_pot_lmul2_packed(const float* A, const uint8_t* Bp, float* C, int M, int N, int K);
void mm_rvv_pot_lmul4_packed(const float* A, const uint8_t* Bp, float* C, int M, int N, int K);

// FMA style: decode u8→float, vfmacc
void mm_rvv_fma_lmul1(const float* A, const uint8_t* B, float* C, int M, int N, int K);
void mm_rvv_fma_lmul2(const float* A, const uint8_t* B, float* C, int M, int N, int K);
void mm_rvv_fma_lmul4(const float* A, const uint8_t* B, float* C, int M, int N, int K);
void mm_rvv_fma_lmul1_packed(const float* A, const uint8_t* Bp, float* C, int M, int N, int K);
void mm_rvv_fma_lmul2_packed(const float* A, const uint8_t* Bp, float* C, int M, int N, int K);
void mm_rvv_fma_lmul4_packed(const float* A, const uint8_t* Bp, float* C, int M, int N, int K);

// MIX style: k+0 FMA, k+1 POT, unrolled ×2 (overlap int/float ALU)
void mm_rvv_mix_lmul1(const float* A, const uint8_t* B, float* C, int M, int N, int K);
void mm_rvv_mix_lmul2(const float* A, const uint8_t* B, float* C, int M, int N, int K);
void mm_rvv_mix_lmul4(const float* A, const uint8_t* B, float* C, int M, int N, int K);
void mm_rvv_mix_lmul1_packed(const float* A, const uint8_t* Bp, float* C, int M, int N, int K);
void mm_rvv_mix_lmul2_packed(const float* A, const uint8_t* Bp, float* C, int M, int N, int K);
void mm_rvv_mix_lmul4_packed(const float* A, const uint8_t* Bp, float* C, int M, int N, int K);

// Pack B (shared by all styles — same layout)
void pot_pack_b_v05_lmul1(const uint8_t* B, uint8_t* Bp, int N, int K);
void pot_pack_b_v05_lmul2(const uint8_t* B, uint8_t* Bp, int N, int K);
void pot_pack_b_v05_lmul4(const uint8_t* B, uint8_t* Bp, int N, int K);
