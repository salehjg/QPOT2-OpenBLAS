/*
 * AVX-512 tiled matrix multiplication: float32 × uint8 (Power-of-Two encoded)
 *
 * Instead of FP multiplication, the kernel adds the PoT exponent to the
 * IEEE 754 exponent field of A, which is equivalent to multiplying by 2^exp.
 *
 * B encoding (uint8):
 *   bits [4:0] = log2(|value|)   (exponent, 0-31)
 *   bit  5     = sign            (1 = negative)
 *
 * Matrix layout:
 *   A : [M × K], row-major, float32
 *   B : [N × K], pre-transposed & PoT-encoded, uint8
 *   C : [M × N], row-major, float32
 *
 * Computes C = A × B_original, where B_original[K×N] is the decoded,
 * non-transposed form of B.
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

// ---------------------------------------------------------------------------
// PoT encoding / decoding helpers
// ---------------------------------------------------------------------------

/// Encode a power-of-2 float to the uint8 PoT format.
/// The absolute value MUST be an exact power of 2.
inline uint8_t pot_encode(float val) {
    int exp = static_cast<int>(std::log2f(std::fabs(val)));
    uint8_t enc = static_cast<uint8_t>(exp);
    if (val < 0.0f) enc |= 0x20;  // bit 5 = sign
    return enc;
}

/// Decode a uint8 PoT value back to float32.
inline float pot_decode(uint8_t enc) {
    int exp = enc & 0x1F;
    float val = static_cast<float>(1u << exp);
    if (enc & 0x20) val = -val;
    return val;
}

// ---------------------------------------------------------------------------
// 64-byte aligned allocation helper
// ---------------------------------------------------------------------------

inline void* aligned_alloc_64(size_t bytes) {
    size_t aligned_bytes = bytes;
    if (aligned_bytes % 64 != 0)
        aligned_bytes += 64 - (aligned_bytes % 64);
    void* p = std::aligned_alloc(64, aligned_bytes);
    if (!p) throw std::bad_alloc();
    return p;
}

template <typename T>
T* aligned_new(size_t count) {
    return static_cast<T*>(aligned_alloc_64(count * sizeof(T)));
}

// ---------------------------------------------------------------------------
// Kernel declaration
// ---------------------------------------------------------------------------

/// Tiled, multithreaded AVX-512 PoT matmul.
/// K MUST be a multiple of 64.
void mm_avx512_f32_1_u8_1_v01(
    const float*   __restrict__ A,   // [M × K]  row-major
    const uint8_t* __restrict__ B,   // [N × K]  transposed, PoT-encoded
    float*         __restrict__ C,   // [M × N]  row-major
    int M, int N, int K);
