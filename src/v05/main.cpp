// bench_v05 — Compare POT / FMA / MIX kernel styles at LMUL=4 vs OpenBLAS.

#include "kernel_v05.hpp"
#include "bench_util.hpp"
#include <CLI/CLI.hpp>
#include <cstdio>

int main(int argc, char** argv) {
    CLI::App app{"bench_v05 — PoT v05 (POT/FMA/MIX × LMUL4) vs OpenBLAS sgemm"};

    int M = 0, N = 0, K = 0, repeats = 7;
    bool skip_verify = false, packed = false;

    app.add_option("-M", M, "Rows")->required();
    app.add_option("-N", N, "Columns")->required();
    app.add_option("-K", K, "Shared dim")->required();
    app.add_option("-r,--repeats", repeats, "Timing repeats");
    app.add_flag("--no-verify", skip_verify, "Skip verification");
    app.add_flag("--packed", packed, "Also bench pre-packed B");
    CLI11_PARSE(app, argc, argv);

    if (K % 32 || N % 32) {
        std::fprintf(stderr, "error: K and N must be multiples of 32\n");
        return 1;
    }

    print_config();
    std::printf("Repeats: %d (median)\n\n", repeats);

    // Verify all styles at LMUL=4
    if (!skip_verify) {
        std::printf("=== Verification ===\n");
        std::printf("  pot4:  "); if (!verify(mm_rvv_pot_lmul4, M, N, K)) return 1;
        std::printf("  fma4:  "); if (!verify(mm_rvv_fma_lmul4, M, N, K)) return 1;
        std::printf("  mix4:  "); if (!verify(mm_rvv_mix_lmul4, M, N, K)) return 1;
        std::printf("\n");
    }

    // Benchmark
    std::printf("=== Benchmark ===\n");
    print_header();

    double t_blas = bench_blas(M, N, K, repeats);

    // Dynamic packing — all 3 styles at LMUL=4
    double t_pot = bench_kernel(mm_rvv_pot_lmul4, M, N, K, repeats);
    print_row(M, N, K, t_blas, t_pot, "pot4");
    double t_fma = bench_kernel(mm_rvv_fma_lmul4, M, N, K, repeats);
    print_row(M, N, K, t_blas, t_fma, "fma4");
    double t_mix = bench_kernel(mm_rvv_mix_lmul4, M, N, K, repeats);
    print_row(M, N, K, t_blas, t_mix, "mix4");

    if (packed) {
        auto bench_pk = [&](const char* tag,
                            void(*fn)(const float*, const uint8_t*, float*, int, int, int),
                            void(*pk)(const uint8_t*, uint8_t*, int, int)) {
            PotTestData d(M, N, K);
            uint8_t* Bp = aligned_new<uint8_t>((size_t)N * K);
            pk(d.B_u8, Bp, N, K);
            fn(d.A, Bp, d.C, M, N, K);
            double t = median_time([&]{ fn(d.A, Bp, d.C, M, N, K); }, repeats);
            print_row(M, N, K, t_blas, t, tag);
            std::free(Bp);
        };
        bench_pk("pot4_pk", mm_rvv_pot_lmul4_packed, pot_pack_b_v05_lmul4);
        bench_pk("fma4_pk", mm_rvv_fma_lmul4_packed, pot_pack_b_v05_lmul4);
        bench_pk("mix4_pk", mm_rvv_mix_lmul4_packed, pot_pack_b_v05_lmul4);
    }

    return 0;
}
