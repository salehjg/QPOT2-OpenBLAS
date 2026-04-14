#pragma once
// pot_common.hpp — shared PoT encoding/decoding and aligned allocation.

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

// ---------------------------------------------------------------------------
// PoT encoding / decoding
// ---------------------------------------------------------------------------

/// Encode a power-of-2 float to uint8 PoT format.
///   bits [4:0] = log2(|value|)   (exponent, 0-31)
///   bit  5     = sign            (1 = negative)
inline uint8_t pot_encode(float val) {
    int exp = static_cast<int>(std::log2f(std::fabs(val)));
    uint8_t enc = static_cast<uint8_t>(exp);
    if (val < 0.0f) enc |= 0x20;
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
// 64-byte aligned allocation
// ---------------------------------------------------------------------------

inline void* aligned_alloc_64(size_t bytes) {
    size_t aligned = (bytes + 63) & ~size_t(63);
    void* p = std::aligned_alloc(64, aligned);
    if (!p) throw std::bad_alloc();
    return p;
}

template <typename T>
T* aligned_new(size_t count) {
    return static_cast<T*>(aligned_alloc_64(count * sizeof(T)));
}
