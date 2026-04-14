// bench_v01 — benchmark the v01 tiled PoT kernel against OpenBLAS cblas_sgemm.

#include "kernel_v01.hpp"
#include "bench_util.hpp"
#include <CLI/CLI.hpp>
#include <cstdio>

int main(int argc, char** argv) {
    CLI::App app{"bench_v01 — PoT v01 (tiled, no packing) vs OpenBLAS sgemm"};

    int M = 0, N = 0, K = 0;
    int repeats = 7;
    bool skip_verify = false;

    app.add_option("-M", M, "Rows of A / rows of C")->required();
    app.add_option("-N", N, "Columns of C (= rows of transposed B)")->required();
    app.add_option("-K", K, "Shared dimension")->required();
    app.add_option("-r,--repeats", repeats, "Timing repeats (median taken)");
    app.add_flag("--no-verify", skip_verify, "Skip correctness check");

    CLI11_PARSE(app, argc, argv);

    if (K % 64 != 0) {
        std::fprintf(stderr, "error: K must be a multiple of 64 (got %d)\n", K);
        return 1;
    }

    print_config();
    std::printf("Repeats: %d (median)\n\n", repeats);

    // Verify
    if (!skip_verify) {
        std::printf("=== Verification (v01) ===\n");
        if (!verify(mm_avx512_f32_1_u8_1_v01, M, N, K)) return 1;
        std::printf("\n");
    }

    // Benchmark
    std::printf("=== Benchmark ===\n");
    print_header();

    double t_blas = bench_blas(M, N, K, repeats);
    double t_v01  = bench_kernel(mm_avx512_f32_1_u8_1_v01, M, N, K, repeats);
    print_row(M, N, K, t_blas, t_v01);

    return 0;
}
