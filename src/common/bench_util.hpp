#pragma once
// bench_util.hpp — benchmark scaffolding: data generation, median timing,
//                  verification against OpenBLAS cblas_sgemm, result printing.

#include "pot_common.hpp"

#include <omp.h>       // must precede cblas.h (OpenBLAS header conflict)
#include <cblas.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

// ---------------------------------------------------------------------------
// Test data: A [M x K] float32, B [N x K] uint8 (transposed PoT-encoded),
//            B_float [N x K] float32 (decoded, for BLAS reference).
// ---------------------------------------------------------------------------
struct PotTestData {
    float*   A;
    float*   B_float;
    uint8_t* B_u8;
    float*   C;
    float*   C_ref;
    int      M, N, K;

    PotTestData(int M_, int N_, int K_, unsigned seed = 123)
        : M(M_), N(N_), K(K_)
    {
        A       = aligned_new<float>((size_t)M * K);
        B_float = aligned_new<float>((size_t)N * K);
        B_u8    = aligned_new<uint8_t>((size_t)N * K);
        C       = aligned_new<float>((size_t)M * N);
        C_ref   = aligned_new<float>((size_t)M * N);

        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> adist(-10.0f, 10.0f);

        for (size_t i = 0; i < (size_t)M * K; ++i)
            A[i] = adist(rng);

        for (size_t i = 0; i < (size_t)N * K; ++i) {
            int exp = rng() % 7;
            float val = static_cast<float>(1 << exp);
            if (rng() % 2) val = -val;
            B_float[i] = val;
            B_u8[i]    = pot_encode(val);
        }

        std::memset(C,     0, (size_t)M * N * sizeof(float));
        std::memset(C_ref, 0, (size_t)M * N * sizeof(float));
    }

    void reset_outputs() {
        std::memset(C,     0, (size_t)M * N * sizeof(float));
        std::memset(C_ref, 0, (size_t)M * N * sizeof(float));
    }

    ~PotTestData() {
        std::free(A); std::free(B_float); std::free(B_u8);
        std::free(C); std::free(C_ref);
    }
    PotTestData(const PotTestData&) = delete;
    PotTestData& operator=(const PotTestData&) = delete;
};

// ---------------------------------------------------------------------------
// Median timing — runs fn() `repeats` times and returns the median seconds.
// ---------------------------------------------------------------------------
template <typename Fn>
double median_time(Fn&& fn, int repeats) {
    std::vector<double> times(repeats);
    for (int i = 0; i < repeats; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        fn();
        auto t1 = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration<double>(t1 - t0).count();
    }
    std::sort(times.begin(), times.end());
    return times[repeats / 2];
}

// ---------------------------------------------------------------------------
// BLAS reference timing (median).
// ---------------------------------------------------------------------------
inline double bench_blas(int M, int N, int K, int repeats) {
    PotTestData d(M, N, K);
    // warmup
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K, 1.0f, d.A, K, d.B_float, K, 0.0f, d.C_ref, N);

    return median_time([&]{
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    M, N, K, 1.0f, d.A, K, d.B_float, K, 0.0f, d.C_ref, N);
    }, repeats);
}

// ---------------------------------------------------------------------------
// Kernel function type.
// ---------------------------------------------------------------------------
using pot_kernel_fn = void(*)(const float*, const uint8_t*, float*, int, int, int);

// ---------------------------------------------------------------------------
// Verification against cblas_sgemm.  Returns true on PASS.
// ---------------------------------------------------------------------------
inline bool verify(pot_kernel_fn kernel, int M, int N, int K) {
    PotTestData d(M, N, K);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K, 1.0f, d.A, K, d.B_float, K, 0.0f, d.C_ref, N);
    kernel(d.A, d.B_u8, d.C, M, N, K);

    const size_t mn = (size_t)M * N;
    float max_abs = 0.0f;
    for (size_t i = 0; i < mn; ++i)
        max_abs = std::max(max_abs, std::fabs(d.C_ref[i]));

    float max_diff = 0.0f;
    size_t bad = 0;
    for (size_t i = 0; i < mn; ++i) {
        float diff = std::fabs(d.C[i] - d.C_ref[i]);
        max_diff = std::max(max_diff, diff);
        if (diff > max_abs * 1e-4f + 1e-2f) ++bad;
    }
    float rel = (max_abs > 0.0f) ? max_diff / max_abs : max_diff;

    if (bad == 0) {
        std::printf("  PASS  %4dx%4dx%4d  max_diff=%.2e  rel=%.2e\n",
                    M, N, K, max_diff, rel);
        return true;
    }
    std::printf("  FAIL  %4dx%4dx%4d  max_diff=%.2e  rel=%.2e  bad=%zu/%zu\n",
                M, N, K, max_diff, rel, bad, mn);
    return false;
}

// ---------------------------------------------------------------------------
// Kernel timing (median).
// ---------------------------------------------------------------------------
inline double bench_kernel(pot_kernel_fn kernel, int M, int N, int K, int repeats) {
    PotTestData d(M, N, K);
    kernel(d.A, d.B_u8, d.C, M, N, K);  // warmup
    return median_time([&]{
        kernel(d.A, d.B_u8, d.C, M, N, K);
    }, repeats);
}

// ---------------------------------------------------------------------------
// Print one benchmark row.
// ---------------------------------------------------------------------------
inline void print_row(int M, int N, int K,
                      double t_blas, double t_kernel, const char* extra = "") {
    double flops = 2.0 * (double)M * N * K;
    double g_blas = flops / t_blas / 1e9;
    double g_kern = flops / t_kernel / 1e9;
    double ratio  = t_blas / t_kernel;
    std::printf("%-6d %-6d %-6d  %10.6f %8.2f  %10.6f %8.2f  %6.2fx %s\n",
                M, N, K, t_blas, g_blas, t_kernel, g_kern, ratio, extra);
}

inline void print_header() {
    std::printf("%-6s %-6s %-6s  %10s %8s  %10s %8s  %7s\n",
                "M", "N", "K", "BLAS(s)", "GFLOPS", "PoT(s)", "GFLOPS", "speedup");
    std::printf("------ ------ ------  ---------- --------"
                "  ---------- --------  -------\n");
}

inline void print_config() {
    std::printf("OMP threads    : %d\n", omp_get_max_threads());
    std::printf("OpenBLAS config: %s\n", openblas_get_config());
    std::printf("OpenBLAS cores : %d\n", openblas_get_num_threads());
}
