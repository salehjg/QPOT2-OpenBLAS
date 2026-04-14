#include <omp.h>          // must come before cblas.h (OpenBLAS common.h conflict)
#include <cblas.h>
#include "mm_avx512_f32u8.hpp"
#include "mm_avx512_f32u8_v02.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

// ---------------------------------------------------------------------------
// Fill a float vector with uniform random values in [-1, 1].
// ---------------------------------------------------------------------------
static void fill_random(std::vector<float>& v, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : v) x = dist(rng);
}

// ---------------------------------------------------------------------------
// Benchmark cblas_sgemm for a given M×N×K.
// Returns elapsed wall-clock seconds for a single call.
// ---------------------------------------------------------------------------
static double bench_sgemm(int M, int N, int K, int warmup_iters, int bench_iters) {
    std::vector<float> A((size_t)M * K), B((size_t)K * N), C((size_t)M * N);

    std::mt19937 rng(42);
    fill_random(A, rng);
    fill_random(B, rng);

    for (int i = 0; i < warmup_iters; ++i) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K,
                    1.0f, A.data(), K, B.data(), N,
                    0.0f, C.data(), N);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K,
                    1.0f, A.data(), K, B.data(), N,
                    0.0f, C.data(), N);
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double>(t1 - t0).count() / bench_iters;
}

// ---------------------------------------------------------------------------
// Generate test data for the PoT kernel (rectangular M×N×K).
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

        std::memset(C, 0, (size_t)M * N * sizeof(float));
        std::memset(C_ref, 0, (size_t)M * N * sizeof(float));
    }

    ~PotTestData() {
        std::free(A);
        std::free(B_float);
        std::free(B_u8);
        std::free(C);
        std::free(C_ref);
    }

    PotTestData(const PotTestData&) = delete;
    PotTestData& operator=(const PotTestData&) = delete;
};

// ---------------------------------------------------------------------------
// Verify a PoT kernel against cblas_sgemm.
// ---------------------------------------------------------------------------
typedef void (*pot_fn)(const float*, const uint8_t*, float*, int, int, int);

static bool verify_pot_gemm(const char* name, pot_fn kernel, int M, int N, int K) {
    PotTestData d(M, N, K);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K,
                1.0f, d.A, K, d.B_float, K,
                0.0f, d.C_ref, N);

    kernel(d.A, d.B_u8, d.C, M, N, K);

    const size_t mn = (size_t)M * N;
    float max_abs_ref = 0.0f;
    for (size_t i = 0; i < mn; ++i) {
        float a = std::fabs(d.C_ref[i]);
        if (a > max_abs_ref) max_abs_ref = a;
    }

    float max_diff = 0.0f;
    size_t diff_count = 0;
    for (size_t i = 0; i < mn; ++i) {
        float diff = std::fabs(d.C[i] - d.C_ref[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > max_abs_ref * 1e-4f + 1e-2f)
            ++diff_count;
    }

    float rel_err = (max_abs_ref > 0.0f) ? max_diff / max_abs_ref : max_diff;

    if (diff_count == 0) {
        std::printf("  %-6s %4dx%4dx%4d PASS  (max_diff=%.4e  rel_err=%.4e)\n",
                    name, M, N, K, max_diff, rel_err);
        return true;
    } else {
        std::printf("  %-6s %4dx%4dx%4d FAIL  (max_diff=%.4e  rel_err=%.4e  bad=%zu/%zu)\n",
                    name, M, N, K, max_diff, rel_err, diff_count, mn);
        return false;
    }
}

// ---------------------------------------------------------------------------
// Benchmark a PoT kernel for a given M×N×K.
// ---------------------------------------------------------------------------
static double bench_pot_gemm(pot_fn kernel, int M, int N, int K,
                             int warmup_iters, int bench_iters) {
    PotTestData d(M, N, K);

    for (int i = 0; i < warmup_iters; ++i)
        kernel(d.A, d.B_u8, d.C, M, N, K);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < bench_iters; ++i)
        kernel(d.A, d.B_u8, d.C, M, N, K);
    auto t1 = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double>(t1 - t0).count() / bench_iters;
}

// ---------------------------------------------------------------------------
// Benchmark v02_packed (pre-packed B — inference hot path).
// B is packed once outside the timed loop.
// ---------------------------------------------------------------------------
static double bench_pot_gemm_packed(int M, int N, int K,
                                    int warmup_iters, int bench_iters) {
    PotTestData d(M, N, K);

    // Pre-pack B (not timed — done once at model load)
    uint8_t* Bp = aligned_new<uint8_t>((size_t)N * K);
    pot_pack_b_v02(d.B_u8, Bp, N, K);

    for (int i = 0; i < warmup_iters; ++i)
        mm_avx512_f32_1_u8_1_v02_packed(d.A, Bp, d.C, M, N, K);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < bench_iters; ++i)
        mm_avx512_f32_1_u8_1_v02_packed(d.A, Bp, d.C, M, N, K);
    auto t1 = std::chrono::high_resolution_clock::now();

    std::free(Bp);
    return std::chrono::duration<double>(t1 - t0).count() / bench_iters;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    std::printf("=== Configuration ===\n");
    std::printf("OMP threads  : %d\n", omp_get_max_threads());
    std::printf("OpenBLAS cfg : %s\n", openblas_get_config());
    std::printf("OpenBLAS cores: %d\n", openblas_get_num_threads());
    std::printf("\n");

    // -----------------------------------------------------------------------
    // Verification (both kernels, square + rectangular)
    // -----------------------------------------------------------------------
    std::printf("=== Verification ===\n");

    struct Shape { int M, N, K; };
    const Shape verify_shapes[] = {
        {1024, 1024, 1024},
        {2048, 2048, 2048},
        {  16, 4096, 4096},
        {   1, 4096, 4096},
    };

    bool all_pass = true;
    for (auto& s : verify_shapes) {
        // v01 requires K % 64 == 0
        if (s.K % 64 == 0)
            all_pass &= verify_pot_gemm("v01", mm_avx512_f32_1_u8_1_v01, s.M, s.N, s.K);
        // v02 requires K % 16 == 0 and N % 16 == 0
        if (s.K % 16 == 0 && s.N % 16 == 0) {
            all_pass &= verify_pot_gemm("v02", mm_avx512_f32_1_u8_1_v02, s.M, s.N, s.K);

            // Verify pre-packed variant
            PotTestData vd(s.M, s.N, s.K);
            uint8_t* Bp = aligned_new<uint8_t>((size_t)s.N * s.K);
            pot_pack_b_v02(vd.B_u8, Bp, s.N, s.K);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        s.M, s.N, s.K,
                        1.0f, vd.A, s.K, vd.B_float, s.K,
                        0.0f, vd.C_ref, s.N);
            mm_avx512_f32_1_u8_1_v02_packed(vd.A, Bp, vd.C, s.M, s.N, s.K);

            float max_abs_ref = 0.0f;
            const size_t mn = (size_t)s.M * s.N;
            for (size_t i = 0; i < mn; ++i) {
                float a = std::fabs(vd.C_ref[i]);
                if (a > max_abs_ref) max_abs_ref = a;
            }
            float max_diff = 0.0f;
            size_t diff_count = 0;
            for (size_t i = 0; i < mn; ++i) {
                float diff = std::fabs(vd.C[i] - vd.C_ref[i]);
                if (diff > max_diff) max_diff = diff;
                if (diff > max_abs_ref * 1e-4f + 1e-2f) ++diff_count;
            }
            float rel_err = (max_abs_ref > 0.0f) ? max_diff / max_abs_ref : max_diff;
            if (diff_count == 0) {
                std::printf("  %-6s %4dx%4dx%4d PASS  (max_diff=%.4e  rel_err=%.4e)\n",
                            "pkd", s.M, s.N, s.K, max_diff, rel_err);
            } else {
                std::printf("  %-6s %4dx%4dx%4d FAIL  (max_diff=%.4e  rel_err=%.4e  bad=%zu/%zu)\n",
                            "pkd", s.M, s.N, s.K, max_diff, rel_err, diff_count, mn);
                all_pass = false;
            }
            std::free(Bp);
        }
    }

    if (!all_pass) {
        std::printf("\n*** VERIFICATION FAILED ***\n");
        return 1;
    }

    // -----------------------------------------------------------------------
    // Square benchmarks (compute-bound baseline)
    // -----------------------------------------------------------------------
    std::printf("\n=== Square matrices (compute-bound) ===\n");
    std::printf("%-6s %-6s %-6s  %10s %8s  %10s %8s  %10s %8s  %7s\n",
                "M", "N", "K", "BLAS(s)", "GFLOPS",
                "v01(s)", "GFLOPS", "v02(s)", "GFLOPS", "v02/BL");
    std::printf("------ ------ ------  ---------- --------"
                "  ---------- --------  ---------- --------  -------\n");

    const int sq_sizes[] = {1024, 2048, 4096};
    for (int N : sq_sizes) {
        int warmup = (N <= 1024) ? 5 : 2;
        int iters  = (N <= 1024) ? 10 : (N <= 2048) ? 5 : 3;

        double t_blas = bench_sgemm(N, N, N, warmup, iters);
        double t_v01  = bench_pot_gemm(mm_avx512_f32_1_u8_1_v01, N, N, N, warmup, iters);
        double t_v02  = bench_pot_gemm(mm_avx512_f32_1_u8_1_v02, N, N, N, warmup, iters);
        double flops  = 2.0 * (double)N * N * N;

        std::printf("%-6d %-6d %-6d  %10.6f %8.2f  %10.6f %8.2f  %10.6f %8.2f  %6.2fx\n",
                    N, N, N,
                    t_blas, flops / t_blas / 1e9,
                    t_v01,  flops / t_v01  / 1e9,
                    t_v02,  flops / t_v02  / 1e9,
                    t_blas / t_v02);
    }

    // -----------------------------------------------------------------------
    // Tall-skinny benchmarks (bandwidth-bound — where PoT should win)
    // -----------------------------------------------------------------------
    std::printf("\n=== Tall-skinny matrices (bandwidth-bound) ===\n");
    std::printf("%-6s %-6s %-6s  %10s %8s  %10s %8s %7s  %10s %8s %7s\n",
                "M", "N", "K", "BLAS(s)", "GFLOPS",
                "v02(s)", "GFLOPS", "v02/BL",
                "packed(s)", "GFLOPS", "pkd/BL");
    std::printf("------ ------ ------  ---------- --------"
                "  ---------- -------- -------"
                "  ---------- -------- -------\n");

    const Shape rect_shapes[] = {
        // GEMV-like (single sample inference)
        {   1, 4096, 4096},
        {   1, 8192, 8192},
        {   1,16384,16384},
        // Small batch inference
        {   4, 4096, 4096},
        {   4, 8192, 8192},
        {   4,16384,16384},
        // Medium batch
        {  16, 4096, 4096},
        {  16, 8192, 8192},
        {  16,16384,16384},
        // Larger batch
        {  32, 4096, 4096},
        {  32, 8192, 8192},
        // Transition zone
        {  64, 4096, 4096},
        { 128, 4096, 4096},
        { 256, 4096, 4096},
    };

    for (auto& s : rect_shapes) {
        double flops = 2.0 * (double)s.M * s.N * s.K;

        int warmup, iters;
        double est_elements = (double)s.M * s.N * s.K;
        if (est_elements <= 1e8)       { warmup = 10; iters = 50; }
        else if (est_elements <= 1e9)  { warmup = 5;  iters = 20; }
        else if (est_elements <= 1e10) { warmup = 3;  iters = 10; }
        else                           { warmup = 2;  iters = 5;  }

        double t_blas = bench_sgemm(s.M, s.N, s.K, warmup, iters);
        double t_v02  = bench_pot_gemm(mm_avx512_f32_1_u8_1_v02,
                                       s.M, s.N, s.K, warmup, iters);
        double t_pkd  = bench_pot_gemm_packed(s.M, s.N, s.K, warmup, iters);

        std::printf("%-6d %-6d %-6d  %10.6f %8.2f  %10.6f %8.2f %6.2fx  %10.6f %8.2f %6.2fx\n",
                    s.M, s.N, s.K,
                    t_blas, flops / t_blas / 1e9,
                    t_v02,  flops / t_v02  / 1e9,
                    t_blas / t_v02,
                    t_pkd,  flops / t_pkd  / 1e9,
                    t_blas / t_pkd);
    }

    return 0;
}
