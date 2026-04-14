// bench_v04 — PoT CUDA kernel vs cuBLAS sgemm.

#include "kernel_v04.cuh"
#include "pot_common.hpp"

#include <CLI/CLI.hpp>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

// ---------------------------------------------------------------------------
// Error checking
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            std::fprintf(stderr, "CUDA error %s:%d — %s\n",                    \
                         __FILE__, __LINE__, cudaGetErrorString(_e));           \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t _s = (call);                                             \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                      \
            std::fprintf(stderr, "cuBLAS error %s:%d — status %d\n",           \
                         __FILE__, __LINE__, (int)_s);                          \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// CUDA-event median timer
// ---------------------------------------------------------------------------

template <typename Fn>
float median_time_ms(Fn&& fn, int repeats) {
    // warmup
    fn();
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    std::vector<float> ms(repeats);
    for (int i = 0; i < repeats; ++i) {
        CUDA_CHECK(cudaEventRecord(t0));
        fn();
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        CUDA_CHECK(cudaEventElapsedTime(&ms[i], t0, t1));
    }
    std::sort(ms.begin(), ms.end());

    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return ms[repeats / 2];
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    CLI::App app{"bench_v04 — PoT CUDA (v04) vs cuBLAS sgemm"};

    int  M = 0, N = 0, K = 0;
    int  repeats    = 7;
    bool skip_verify = false;

    app.add_option("-M", M, "Rows of A / C")->required();
    app.add_option("-N", N, "Columns of C")->required();
    app.add_option("-K", K, "Shared dimension")->required();
    app.add_option("-r,--repeats", repeats, "Timing repeats (median)");
    app.add_flag("--no-verify", skip_verify, "Skip correctness check");
    CLI11_PARSE(app, argc, argv);

    // --- Device info ---
    int dev;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    std::printf("Device : %s  (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    std::printf("Memory : %.1f GB global\n", prop.totalGlobalMem / 1e9);
    std::printf("M=%-5d N=%-5d K=%-5d  repeats=%d\n\n", M, N, K, repeats);

    // --- Host data generation ---
    const size_t sz_A  = (size_t)M * K;
    const size_t sz_B  = (size_t)N * K;
    const size_t sz_C  = (size_t)M * N;

    std::vector<float>   h_A(sz_A), h_B_f(sz_B);
    std::vector<uint8_t> h_B_u8(sz_B);
    std::vector<float>   h_C(sz_C), h_C_ref(sz_C);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> adist(-10.0f, 10.0f);
    for (float& v : h_A) v = adist(rng);
    for (size_t i = 0; i < sz_B; ++i) {
        float val = static_cast<float>(1 << (rng() % 7));
        if (rng() % 2) val = -val;
        h_B_f[i]   = val;
        h_B_u8[i]  = pot_encode(val);
    }

    // --- Device allocations ---
    float   *d_A, *d_Bf, *d_C, *d_Cref;
    uint8_t *d_Bu8;

    CUDA_CHECK(cudaMalloc(&d_A,    sz_A  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Bf,   sz_B  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Bu8,  sz_B  * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_C,    sz_C  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Cref, sz_C  * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A,   h_A.data(),    sz_A * sizeof(float),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Bf,  h_B_f.data(),  sz_B * sizeof(float),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Bu8, h_B_u8.data(), sz_B * sizeof(uint8_t), cudaMemcpyHostToDevice));

    // --- cuBLAS setup ---
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // C_rm[M×N] = A_rm[M×K] × B_rm[N×K]^T
    // cuBLAS (col-major trick): swap A↔B args and M↔N dims, transpose first arg
    //   cublasSgemm(handle, OP_T, OP_N, N, M, K, &1, B, K, A, K, &0, C, N)
    const float one = 1.0f, zero = 0.0f;
    auto run_cublas = [&]{
        CUBLAS_CHECK(cublasSgemm(handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, K, &one,
            d_Bf, K, d_A, K,
            &zero, d_Cref, N));
    };

    auto run_pot = [&]{
        mm_cuda_pot_v04(d_A, d_Bu8, d_C, M, N, K);
    };

    // --- Verification ---
    if (!skip_verify) {
        run_cublas();
        run_pot();
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_C_ref.data(), d_Cref, sz_C * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_C.data(),     d_C,    sz_C * sizeof(float), cudaMemcpyDeviceToHost));

        float max_abs = 0.0f, max_diff = 0.0f;
        size_t bad = 0;
        for (size_t i = 0; i < sz_C; ++i)
            max_abs = std::fmax(max_abs, std::fabs(h_C_ref[i]));
        for (size_t i = 0; i < sz_C; ++i) {
            float d = std::fabs(h_C[i] - h_C_ref[i]);
            max_diff = std::fmax(max_diff, d);
            if (d > max_abs * 1e-4f + 1e-2f) ++bad;
        }
        float rel = (max_abs > 0.0f) ? max_diff / max_abs : max_diff;
        if (bad == 0)
            std::printf("Verification: PASS  max_diff=%.2e  rel=%.2e\n\n", max_diff, rel);
        else {
            std::printf("Verification: FAIL  bad=%zu/%zu  max_diff=%.2e\n", bad, sz_C, max_diff);
            goto cleanup;
        }
    }

    {
        // --- Benchmark ---
        // The PoT kernel overwrites C (no += into existing C), so no memset needed.
        const double flops = 2.0 * M * N * K;

        float t_cublas = median_time_ms(run_cublas, repeats);
        float t_pot    = median_time_ms(run_pot,    repeats);

        double tf_cublas = flops / (t_cublas * 1e-3) / 1e12;
        double tf_pot    = flops / (t_pot    * 1e-3) / 1e12;
        double speedup   = t_cublas / t_pot;

        std::printf("%-6s %-6s %-6s  %10s %8s  %10s %8s  %7s\n",
                    "M", "N", "K", "cuBLAS(ms)", "TFLOPS", "PoT(ms)", "TFLOPS", "speedup");
        std::printf("%-6d %-6d %-6d  %10.4f %8.4f  %10.4f %8.4f  %6.2fx\n",
                    M, N, K, t_cublas, tf_cublas, t_pot, tf_pot, speedup);

        // Also report effective bandwidth for PoT
        // BW = (A bytes read + B bytes read + C bytes written) / time
        double bw_bytes = (double)sz_A * sizeof(float)
                        + (double)sz_B * sizeof(uint8_t)   // uint8 B (not float!)
                        + (double)sz_C * sizeof(float);
        double bw_pot   = bw_bytes / (t_pot * 1e-3) / 1e9;
        std::printf("  PoT effective BW: %.1f GB/s  (B loaded as uint8, %.0f MB)\n",
                    bw_pot, (double)sz_B / 1e6);
    }

cleanup:
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_Bf));
    CUDA_CHECK(cudaFree(d_Bu8));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_Cref));
    return 0;
}
