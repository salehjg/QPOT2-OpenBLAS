// bench_v03 — benchmark the v03 PoT kernel against OpenBLAS sgemm.

#include "kernel_v03.hpp"
#include "bench_util.hpp"
#include <CLI/CLI.hpp>
#include <cstdio>

int main(int argc, char** argv) {
    CLI::App app{"bench_v03 — PoT v03 (MR=16, KC=1024, k-unroll, prefetch) vs OpenBLAS sgemm"};

    int M = 0, N = 0, K = 0;
    int repeats = 7;
    bool skip_verify = false;
    bool packed = false;

    app.add_option("-M", M, "Rows of A / rows of C")->required();
    app.add_option("-N", N, "Columns of C (= rows of transposed B)")->required();
    app.add_option("-K", K, "Shared dimension")->required();
    app.add_option("-r,--repeats", repeats, "Timing repeats (median taken)");
    app.add_flag("--no-verify", skip_verify, "Skip correctness check");
    app.add_flag("--packed", packed, "Use pre-packed B (inference mode)");

    CLI11_PARSE(app, argc, argv);

    if (K % 16 != 0) {
        std::fprintf(stderr, "error: K must be a multiple of 16 (got %d)\n", K);
        return 1;
    }
    if (N % 16 != 0) {
        std::fprintf(stderr, "error: N must be a multiple of 16 (got %d)\n", N);
        return 1;
    }

    print_config();
    std::printf("Repeats: %d (median)\n", repeats);
    std::printf("Mode   : %s\n\n", packed ? "pre-packed B" : "dynamic packing");

    // Verify
    if (!skip_verify) {
        std::printf("=== Verification ===\n");
        std::printf("  dynamic: ");
        if (!verify(mm_avx512_f32_1_u8_1_v03, M, N, K)) return 1;

        if (packed) {
            PotTestData d(M, N, K);
            uint8_t* Bp = aligned_new<uint8_t>((size_t)N * K);
            pot_pack_b_v03(d.B_u8, Bp, N, K);

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        M, N, K, 1.0f, d.A, K, d.B_float, K, 0.0f, d.C_ref, N);
            mm_avx512_f32_1_u8_1_v03_packed(d.A, Bp, d.C, M, N, K);

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
            if (bad == 0)
                std::printf("  packed:  PASS  %4dx%4dx%4d  max_diff=%.2e  rel=%.2e\n",
                            M, N, K, max_diff, rel);
            else {
                std::printf("  packed:  FAIL  %4dx%4dx%4d  bad=%zu/%zu\n",
                            M, N, K, bad, mn);
                std::free(Bp);
                return 1;
            }
            std::free(Bp);
        }
        std::printf("\n");
    }

    // Benchmark
    std::printf("=== Benchmark ===\n");
    print_header();

    double t_blas = bench_blas(M, N, K, repeats);

    if (!packed) {
        double t_v03 = bench_kernel(mm_avx512_f32_1_u8_1_v03, M, N, K, repeats);
        print_row(M, N, K, t_blas, t_v03);
    } else {
        PotTestData d(M, N, K);
        uint8_t* Bp = aligned_new<uint8_t>((size_t)N * K);
        pot_pack_b_v03(d.B_u8, Bp, N, K);

        mm_avx512_f32_1_u8_1_v03_packed(d.A, Bp, d.C, M, N, K);  // warmup

        double t_pkd = median_time([&]{
            mm_avx512_f32_1_u8_1_v03_packed(d.A, Bp, d.C, M, N, K);
        }, repeats);

        print_row(M, N, K, t_blas, t_pkd, "(packed)");
        std::free(Bp);
    }

    return 0;
}
