/*
 * v05: RVV 1.0 PoT GEMM for SpacemiT K1 (VLEN=256, in-order).
 *
 * Three micro-kernel styles sharing the same GOTO-style tiling:
 *
 *   POT  — Integer bit-trick: vadd(a_bits, sh) ^ sc → vfadd.
 *           Pro: fewest decode ops.  Con: 3 ops/FLOP (vadd+vxor+vfadd).
 *
 *   FMA  — Decode uint8→IEEE float, then vfmacc(acc, a, b_float).
 *           Pro: uses FMA (2 FLOPs/op).  Con: heavier decode (vadd+127, vsll).
 *
 *   MIX  — Unroll ×2: k+0 via FMA (float ALU), k+1 via POT (int ALU).
 *           Hypothesis: int and float ops overlap on dual-issue hardware.
 */

#include "kernel_v05.hpp"
#include "pot_common.hpp"

#include <riscv_vector.h>
#include <algorithm>
#include <cassert>
#include <cstring>
#include <omp.h>

static constexpr int KC = 512;
static constexpr int MC = 128;
static constexpr int B_PREFETCH_BYTES = 256;

enum class KS { POT, MIX, FMA };

// ---------------------------------------------------------------------------
// LMUL traits
// ---------------------------------------------------------------------------

template<int L> struct Tr;

template<> struct Tr<1> {
    static constexpr int NR = 8, MR = 16;
    using u32 = vuint32m1_t; using f32 = vfloat32m1_t;
    using u8n = vuint8mf4_t; using bm  = vbool32_t;
    static size_t vl() { return __riscv_vsetvlmax_e32m1(); }
    static u8n  ld8 (const uint8_t* p, size_t n) { return __riscv_vle8_v_u8mf4(p, n); }
    static u32  ext (u8n v, size_t n)             { return __riscv_vzext_vf4_u32m1(v, n); }
    static bm   gtu (u32 v, uint32_t x, size_t n) { return __riscv_vmsgtu_vx_u32m1_b32(v, x, n); }
    static u32  andu(u32 v, uint32_t x, size_t n) { return __riscv_vand_vx_u32m1(v, x, n); }
    static u32  sllu(u32 v, uint32_t s, size_t n) { return __riscv_vsll_vx_u32m1(v, s, n); }
    static u32  bcat(uint32_t x, size_t n)         { return __riscv_vmv_v_x_u32m1(x, n); }
    static u32  mrgx(u32 vs2, uint32_t x, bm m, size_t n) { return __riscv_vmerge_vxm_u32m1(vs2, x, m, n); }
    static u32  addx(u32 v, uint32_t x, size_t n) { return __riscv_vadd_vx_u32m1(v, x, n); }
    static u32  xorv(u32 a, u32 b, size_t n)       { return __riscv_vxor_vv_u32m1(a, b, n); }
    static f32  cast(u32 v)                         { return __riscv_vreinterpret_v_u32m1_f32m1(v); }
    static f32  fzer(size_t n)                      { return __riscv_vfmv_v_f_f32m1(0.f, n); }
    static f32  fadd(f32 a, f32 b, size_t n)        { return __riscv_vfadd_vv_f32m1(a, b, n); }
    static f32  fld (const float* p, size_t n)      { return __riscv_vle32_v_f32m1(p, n); }
    static void fst (float* p, f32 v, size_t n)     { __riscv_vse32_v_f32m1(p, v, n); }
    static f32  fmac(f32 acc, f32 a, f32 b, size_t n) { return __riscv_vfmacc_vv_f32m1(acc, a, b, n); }
    static f32  fbcast(float x, size_t n)              { return __riscv_vfmv_v_f_f32m1(x, n); }
};

template<> struct Tr<2> {
    static constexpr int NR = 16, MR = 8;
    using u32 = vuint32m2_t; using f32 = vfloat32m2_t;
    using u8n = vuint8mf2_t; using bm  = vbool16_t;
    static size_t vl() { return __riscv_vsetvlmax_e32m2(); }
    static u8n  ld8 (const uint8_t* p, size_t n) { return __riscv_vle8_v_u8mf2(p, n); }
    static u32  ext (u8n v, size_t n)             { return __riscv_vzext_vf4_u32m2(v, n); }
    static bm   gtu (u32 v, uint32_t x, size_t n) { return __riscv_vmsgtu_vx_u32m2_b16(v, x, n); }
    static u32  andu(u32 v, uint32_t x, size_t n) { return __riscv_vand_vx_u32m2(v, x, n); }
    static u32  sllu(u32 v, uint32_t s, size_t n) { return __riscv_vsll_vx_u32m2(v, s, n); }
    static u32  bcat(uint32_t x, size_t n)         { return __riscv_vmv_v_x_u32m2(x, n); }
    static u32  mrgx(u32 vs2, uint32_t x, bm m, size_t n) { return __riscv_vmerge_vxm_u32m2(vs2, x, m, n); }
    static u32  addx(u32 v, uint32_t x, size_t n) { return __riscv_vadd_vx_u32m2(v, x, n); }
    static u32  xorv(u32 a, u32 b, size_t n)       { return __riscv_vxor_vv_u32m2(a, b, n); }
    static f32  cast(u32 v)                         { return __riscv_vreinterpret_v_u32m2_f32m2(v); }
    static f32  fzer(size_t n)                      { return __riscv_vfmv_v_f_f32m2(0.f, n); }
    static f32  fadd(f32 a, f32 b, size_t n)        { return __riscv_vfadd_vv_f32m2(a, b, n); }
    static f32  fld (const float* p, size_t n)      { return __riscv_vle32_v_f32m2(p, n); }
    static void fst (float* p, f32 v, size_t n)     { __riscv_vse32_v_f32m2(p, v, n); }
    static f32  fmac(f32 acc, f32 a, f32 b, size_t n) { return __riscv_vfmacc_vv_f32m2(acc, a, b, n); }
    static f32  fbcast(float x, size_t n)              { return __riscv_vfmv_v_f_f32m2(x, n); }
};

template<> struct Tr<4> {
    static constexpr int NR = 32, MR = 4;
    using u32 = vuint32m4_t; using f32 = vfloat32m4_t;
    using u8n = vuint8m1_t;  using bm  = vbool8_t;
    static size_t vl() { return __riscv_vsetvlmax_e32m4(); }
    static u8n  ld8 (const uint8_t* p, size_t n) { return __riscv_vle8_v_u8m1(p, n); }
    static u32  ext (u8n v, size_t n)             { return __riscv_vzext_vf4_u32m4(v, n); }
    static bm   gtu (u32 v, uint32_t x, size_t n) { return __riscv_vmsgtu_vx_u32m4_b8(v, x, n); }
    static u32  andu(u32 v, uint32_t x, size_t n) { return __riscv_vand_vx_u32m4(v, x, n); }
    static u32  sllu(u32 v, uint32_t s, size_t n) { return __riscv_vsll_vx_u32m4(v, s, n); }
    static u32  bcat(uint32_t x, size_t n)         { return __riscv_vmv_v_x_u32m4(x, n); }
    static u32  mrgx(u32 vs2, uint32_t x, bm m, size_t n) { return __riscv_vmerge_vxm_u32m4(vs2, x, m, n); }
    static u32  addx(u32 v, uint32_t x, size_t n) { return __riscv_vadd_vx_u32m4(v, x, n); }
    static u32  xorv(u32 a, u32 b, size_t n)       { return __riscv_vxor_vv_u32m4(a, b, n); }
    static f32  cast(u32 v)                         { return __riscv_vreinterpret_v_u32m4_f32m4(v); }
    static f32  fzer(size_t n)                      { return __riscv_vfmv_v_f_f32m4(0.f, n); }
    static f32  fadd(f32 a, f32 b, size_t n)        { return __riscv_vfadd_vv_f32m4(a, b, n); }
    static f32  fld (const float* p, size_t n)      { return __riscv_vle32_v_f32m4(p, n); }
    static void fst (float* p, f32 v, size_t n)     { __riscv_vse32_v_f32m4(p, v, n); }
    static f32  fmac(f32 acc, f32 a, f32 b, size_t n) { return __riscv_vfmacc_vv_f32m4(acc, a, b, n); }
    static f32  fbcast(float x, size_t n)              { return __riscv_vfmv_v_f_f32m4(x, n); }
};

// ---------------------------------------------------------------------------
// Packing (unchanged — shared by all kernel styles)
// ---------------------------------------------------------------------------

template<int L>
static void pack_a(const float* __restrict__ A, float* __restrict__ Ap,
                   int K, int ii, int kk, int mb, int kb)
{
    constexpr int MR = Tr<L>::MR;
    float* dst = Ap;
    for (int ir = 0; ir < mb; ir += MR) {
        const int mr = std::min(MR, mb - ir);
        for (int r = 0; r < mr; ++r) {
            const float* src = A + (size_t)(ii + ir + r) * K + kk;
            for (int k = 0; k < kb; ++k) dst[k * MR + r] = src[k];
        }
        for (int r = mr; r < MR; ++r)
            for (int k = 0; k < kb; ++k) dst[k * MR + r] = 0.0f;
        dst += MR * kb;
    }
}

template<int L>
static void pack_b_panel(const uint8_t* __restrict__ B, uint8_t* __restrict__ Bp,
                         int K, int jj, int kk, int kb, int actual_nr)
{
    constexpr int NR = Tr<L>::NR;
    for (int c = 0; c < actual_nr; ++c) {
        const uint8_t* src = B + (size_t)(jj + c) * K + kk;
        for (int k = 0; k < kb; ++k) Bp[k * NR + c] = src[k];
    }
    if (actual_nr < NR)
        for (int k = 0; k < kb; ++k)
            for (int c = actual_nr; c < NR; ++c) Bp[k * NR + c] = 0;
}

// ---------------------------------------------------------------------------
// Decode helpers (shared across styles)
// ---------------------------------------------------------------------------

// PoT decode: returns (sh, sc) for integer bit-trick path.
//   sh = exponent << 23,  sc = sign contribution (0 or 0x80000000)
template<int L>
static inline void decode_pot(const uint8_t* BO, int k, size_t vl,
    typename Tr<L>::u32& sh, typename Tr<L>::u32& sc)
{
    constexpr int NR = Tr<L>::NR;
    auto b_u8 = Tr<L>::ld8(BO + k * NR, vl);
    auto b32  = Tr<L>::ext(b_u8, vl);
    auto neg  = Tr<L>::gtu(b32, 0x1Fu, vl);
    auto bexp = Tr<L>::andu(b32, 0x1Fu, vl);
    sh = Tr<L>::sllu(bexp, 23u, vl);
    sc = Tr<L>::mrgx(Tr<L>::bcat(0u, vl), 0x80000000u, neg, vl);
}

// Float decode: returns IEEE float32 vector from uint8 PoT encoding.
//   ±2^e  →  ((e+127)<<23) | (sign<<31)
template<int L>
static inline typename Tr<L>::f32 decode_float(const uint8_t* BO, int k, size_t vl)
{
    constexpr int NR = Tr<L>::NR;
    auto b_u8  = Tr<L>::ld8(BO + k * NR, vl);
    auto b32   = Tr<L>::ext(b_u8, vl);
    auto neg   = Tr<L>::gtu(b32, 0x1Fu, vl);
    auto bexp  = Tr<L>::andu(b32, 0x1Fu, vl);
    auto biased = Tr<L>::addx(bexp, 127u, vl);
    auto ieee_e = Tr<L>::sllu(biased, 23u, vl);
    auto sign   = Tr<L>::mrgx(Tr<L>::bcat(0u, vl), 0x80000000u, neg, vl);
    return Tr<L>::cast(Tr<L>::xorv(ieee_e, sign, vl));
}

// ---------------------------------------------------------------------------
// Micro-kernel: full MR rows (compile-time MR_CT), style selected by KS.
// ---------------------------------------------------------------------------

template<int L, KS S, int MR_CT>
static inline __attribute__((always_inline)) void microkernel_full(
    int kb, const float* __restrict__ AO, const uint8_t* __restrict__ BO,
    float* __restrict__ C_ptr, int N)
{
    constexpr int NR = Tr<L>::NR;
    const size_t vl = Tr<L>::vl();

    alignas(64) float acc_buf[MR_CT * NR];
    __builtin_memset(acc_buf, 0, sizeof(acc_buf));

    if constexpr (S == KS::POT) {
        // ---- POT: vadd(a_bits, sh) ^ sc → vfadd ----
        for (int k = 0; k < kb; ++k) {
            __builtin_prefetch(BO + (k + B_PREFETCH_BYTES / NR) * NR, 0, 2);
            typename Tr<L>::u32 sh, sc;
            decode_pot<L>(BO, k, vl, sh, sc);
            const float* a_base = AO + k * MR_CT;
            for (int r = 0; r < MR_CT; ++r) {
                uint32_t ai; __builtin_memcpy(&ai, a_base + r, 4);
                auto acc_r = Tr<L>::fld(acc_buf + r * NR, vl);
                auto res   = Tr<L>::xorv(Tr<L>::addx(sh, ai, vl), sc, vl);
                acc_r      = Tr<L>::fadd(acc_r, Tr<L>::cast(res), vl);
                Tr<L>::fst(acc_buf + r * NR, acc_r, vl);
            }
        }
    } else if constexpr (S == KS::FMA) {
        // ---- FMA: decode B→float, vfmacc(acc, a, b) ----
        for (int k = 0; k < kb; ++k) {
            __builtin_prefetch(BO + (k + B_PREFETCH_BYTES / NR) * NR, 0, 2);
            auto b_float = decode_float<L>(BO, k, vl);
            const float* a_base = AO + k * MR_CT;
            for (int r = 0; r < MR_CT; ++r) {
                auto acc_r = Tr<L>::fld(acc_buf + r * NR, vl);
                auto a_vec = Tr<L>::fbcast(a_base[r], vl);
                acc_r      = Tr<L>::fmac(acc_r, a_vec, b_float, vl);
                Tr<L>::fst(acc_buf + r * NR, acc_r, vl);
            }
        }
    } else {
        // ---- MIX: k+0 via FMA (float ALU), k+1 via POT (int ALU) ----
        int k = 0;
        for (; k + 1 < kb; k += 2) {
            __builtin_prefetch(BO + (k + B_PREFETCH_BYTES / NR) * NR, 0, 2);

            // Decode both k-steps up front
            auto b0_float = decode_float<L>(BO, k, vl);
            typename Tr<L>::u32 sh1, sc1;
            decode_pot<L>(BO, k + 1, vl, sh1, sc1);

            const float* a0 = AO + k * MR_CT;
            const float* a1 = AO + (k + 1) * MR_CT;
            for (int r = 0; r < MR_CT; ++r) {
                auto acc_r = Tr<L>::fld(acc_buf + r * NR, vl);
                // k+0: FMA path (float pipeline)
                acc_r = Tr<L>::fmac(acc_r, Tr<L>::fbcast(a0[r], vl), b0_float, vl);
                // k+1: POT path (int pipeline → float add)
                uint32_t ai; __builtin_memcpy(&ai, a1 + r, 4);
                auto res = Tr<L>::xorv(Tr<L>::addx(sh1, ai, vl), sc1, vl);
                acc_r    = Tr<L>::fadd(acc_r, Tr<L>::cast(res), vl);
                Tr<L>::fst(acc_buf + r * NR, acc_r, vl);
            }
        }
        // Odd-k tail: use POT path
        for (; k < kb; ++k) {
            typename Tr<L>::u32 sh, sc;
            decode_pot<L>(BO, k, vl, sh, sc);
            const float* a_base = AO + k * MR_CT;
            for (int r = 0; r < MR_CT; ++r) {
                uint32_t ai; __builtin_memcpy(&ai, a_base + r, 4);
                auto acc_r = Tr<L>::fld(acc_buf + r * NR, vl);
                acc_r = Tr<L>::fadd(acc_r, Tr<L>::cast(
                    Tr<L>::xorv(Tr<L>::addx(sh, ai, vl), sc, vl)), vl);
                Tr<L>::fst(acc_buf + r * NR, acc_r, vl);
            }
        }
    }

    for (int r = 0; r < MR_CT; ++r) {
        auto c_old = Tr<L>::fld(C_ptr + (size_t)r * N, vl);
        auto acc_r = Tr<L>::fld(acc_buf + r * NR, vl);
        Tr<L>::fst(C_ptr + (size_t)r * N, Tr<L>::fadd(c_old, acc_r, vl), vl);
    }
}

// Edge micro-kernel: runtime mr < MR.
template<int L, KS S>
static inline void microkernel_edge(
    int mr, int kb, const float* __restrict__ AO, const uint8_t* __restrict__ BO,
    float* __restrict__ C_ptr, int N)
{
    constexpr int NR = Tr<L>::NR;
    constexpr int MR = Tr<L>::MR;
    const size_t vl = Tr<L>::vl();

    alignas(64) float acc_buf[MR * NR];
    __builtin_memset(acc_buf, 0, sizeof(acc_buf));

    // Edge tiles use POT path for all styles (simplicity, edge is rare).
    for (int k = 0; k < kb; ++k) {
        __builtin_prefetch(BO + (k + B_PREFETCH_BYTES / NR) * NR, 0, 2);
        typename Tr<L>::u32 sh, sc;
        decode_pot<L>(BO, k, vl, sh, sc);
        const float* a_base = AO + k * MR;
        for (int r = 0; r < mr; ++r) {
            uint32_t ai; __builtin_memcpy(&ai, a_base + r, 4);
            auto acc_r = Tr<L>::fld(acc_buf + r * NR, vl);
            acc_r = Tr<L>::fadd(acc_r, Tr<L>::cast(
                Tr<L>::xorv(Tr<L>::addx(sh, ai, vl), sc, vl)), vl);
            Tr<L>::fst(acc_buf + r * NR, acc_r, vl);
        }
    }

    for (int r = 0; r < mr; ++r) {
        auto c_old = Tr<L>::fld(C_ptr + (size_t)r * N, vl);
        auto acc_r = Tr<L>::fld(acc_buf + r * NR, vl);
        Tr<L>::fst(C_ptr + (size_t)r * N, Tr<L>::fadd(c_old, acc_r, vl), vl);
    }
}

template<int L, KS S>
static inline void microkernel_dispatch(
    int mr, int kb, const float* __restrict__ AO,
    const uint8_t* __restrict__ BO, float* __restrict__ C_ptr, int N)
{
    if (mr == Tr<L>::MR)
        microkernel_full<L, S, Tr<L>::MR>(kb, AO, BO, C_ptr, N);
    else
        microkernel_edge<L, S>(mr, kb, AO, BO, C_ptr, N);
}

// ---------------------------------------------------------------------------
// GOTO drivers — templated on L and KS, otherwise identical.
// ---------------------------------------------------------------------------

template<int L, KS S>
static inline void compute_mc_tile(
    const float* __restrict__ Ap, const uint8_t* __restrict__ Bp,
    float* __restrict__ C, int mb, int kb, int N,
    int ii, int jj_start, int jj_end)
{
    constexpr int MR = Tr<L>::MR, NR = Tr<L>::NR;
    for (int jj = jj_start; jj < jj_end; jj += NR) {
        const uint8_t* bp = Bp + (size_t)(jj - jj_start) * kb;
        const float* ap = Ap;
        for (int ir = 0; ir < mb; ir += MR) {
            microkernel_dispatch<L, S>(std::min(MR, mb - ir), kb, ap, bp,
                C + (size_t)(ii + ir) * N + jj, N);
            ap += MR * kb;
        }
    }
}

template<int L, KS S>
static void goto_m_parallel(
    const float* __restrict__ A, const uint8_t* __restrict__ B,
    float* __restrict__ C, int M, int N, int K)
{
    constexpr int NR = Tr<L>::NR;
    const int max_kb = std::min(KC, K), nth = omp_get_max_threads();
    uint8_t* Bp = aligned_new<uint8_t>((size_t)N * max_kb);
    float** Ap = new float*[nth];
    for (int t = 0; t < nth; ++t) Ap[t] = aligned_new<float>((size_t)MC * max_kb);

    for (int kk = 0; kk < K; kk += KC) {
        const int kb = std::min(KC, K - kk);
        #pragma omp parallel
        {
            float* ap = Ap[omp_get_thread_num()];
            #pragma omp for schedule(static)
            for (int jj = 0; jj < N; jj += NR)
                pack_b_panel<L>(B, Bp + (size_t)jj * kb, K, jj, kk, kb, std::min(NR, N - jj));
            #pragma omp for schedule(static)
            for (int ii = 0; ii < M; ii += MC) {
                const int mb = std::min(MC, M - ii);
                pack_a<L>(A, ap, K, ii, kk, mb, kb);
                compute_mc_tile<L, S>(ap, Bp, C, mb, kb, N, ii, 0, N);
            }
        }
    }
    for (int t = 0; t < nth; ++t) std::free(Ap[t]);
    delete[] Ap; std::free(Bp);
}

template<int L, KS S>
static void goto_n_parallel(
    const float* __restrict__ A, const uint8_t* __restrict__ B,
    float* __restrict__ C, int M, int N, int K)
{
    constexpr int NR = Tr<L>::NR;
    const int max_kb = std::min(KC, K), n_tiles = N / NR;
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num(), nt = omp_get_num_threads();
        const int base = n_tiles / nt, extra = n_tiles % nt;
        const int s = (tid < extra) ? tid*(base+1) : extra*(base+1)+(tid-extra)*base;
        const int c = base + (tid < extra ? 1 : 0);
        const int j0 = s*NR, j1 = (s+c)*NR, ln = j1-j0;
        if (ln > 0) {
            float* ap = aligned_new<float>((size_t)MC*max_kb);
            uint8_t* bp = aligned_new<uint8_t>((size_t)ln*max_kb);
            for (int i = 0; i < M; ++i) std::memset(C+(size_t)i*N+j0, 0, ln*sizeof(float));
            for (int kk = 0; kk < K; kk += KC) {
                const int kb = std::min(KC, K-kk);
                for (int jj = j0; jj < j1; jj += NR)
                    pack_b_panel<L>(B, bp+(size_t)(jj-j0)*kb, K, jj, kk, kb, std::min(NR, j1-jj));
                for (int ii = 0; ii < M; ii += MC) {
                    const int mb = std::min(MC, M-ii);
                    pack_a<L>(A, ap, K, ii, kk, mb, kb);
                    compute_mc_tile<L, S>(ap, bp, C, mb, kb, N, ii, j0, j1);
                }
            }
            std::free(ap); std::free(bp);
        }
    }
}

template<int L, KS S>
static void mm_rvv(const float* A, const uint8_t* B, float* C, int M, int N, int K)
{
    constexpr int NR = Tr<L>::NR;
    assert(Tr<L>::vl() == (size_t)NR && "VLEN != 256");
    assert(K % NR == 0); assert(N % NR == 0);
    const int nth = omp_get_max_threads();
    if ((M+MC-1)/MC >= nth) {
        std::memset(C, 0, (size_t)M*N*sizeof(float));
        goto_m_parallel<L, S>(A, B, C, M, N, K);
    } else {
        goto_n_parallel<L, S>(A, B, C, M, N, K);
    }
}

// ---------------------------------------------------------------------------
// Pre-packed B drivers
// ---------------------------------------------------------------------------

template<int L>
static void pot_pack_b(const uint8_t* B, uint8_t* Bp, int N, int K)
{
    constexpr int NR = Tr<L>::NR;
    #pragma omp parallel for schedule(static) collapse(2)
    for (int kk = 0; kk < K; kk += KC)
        for (int jj = 0; jj < N; jj += NR) {
            const int kb = std::min(KC, K-kk);
            pack_b_panel<L>(B, Bp+(size_t)kk*N+(size_t)jj*kb, K, jj, kk, kb, std::min(NR, N-jj));
        }
}

template<int L, KS S>
static void goto_m_packed(const float* A, const uint8_t* Bp, float* C, int M, int N, int K)
{
    const int max_kb = std::min(KC, K), nth = omp_get_max_threads();
    float** Ap = new float*[nth];
    for (int t = 0; t < nth; ++t) Ap[t] = aligned_new<float>((size_t)MC*max_kb);
    for (int kk = 0; kk < K; kk += KC) {
        const int kb = std::min(KC, K-kk);
        const uint8_t* bp = Bp + (size_t)kk*N;
        #pragma omp parallel
        {
            float* ap = Ap[omp_get_thread_num()];
            #pragma omp for schedule(static)
            for (int ii = 0; ii < M; ii += MC) {
                const int mb = std::min(MC, M-ii);
                pack_a<L>(A, ap, K, ii, kk, mb, kb);
                compute_mc_tile<L, S>(ap, bp, C, mb, kb, N, ii, 0, N);
            }
        }
    }
    for (int t = 0; t < nth; ++t) std::free(Ap[t]);
    delete[] Ap;
}

template<int L, KS S>
static void goto_n_packed(const float* A, const uint8_t* Bp, float* C, int M, int N, int K)
{
    constexpr int NR = Tr<L>::NR;
    const int max_kb = std::min(KC, K), n_tiles = N / NR;
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num(), nt = omp_get_num_threads();
        const int base = n_tiles/nt, extra = n_tiles%nt;
        const int s = (tid<extra) ? tid*(base+1) : extra*(base+1)+(tid-extra)*base;
        const int c = base + (tid<extra?1:0);
        const int j0 = s*NR, j1 = (s+c)*NR, ln = j1-j0;
        if (ln > 0) {
            float* ap = aligned_new<float>((size_t)MC*max_kb);
            for (int i = 0; i < M; ++i) std::memset(C+(size_t)i*N+j0, 0, ln*sizeof(float));
            for (int kk = 0; kk < K; kk += KC) {
                const int kb = std::min(KC, K-kk);
                const uint8_t* bp = Bp + (size_t)kk*N + (size_t)j0*kb;
                for (int ii = 0; ii < M; ii += MC) {
                    const int mb = std::min(MC, M-ii);
                    pack_a<L>(A, ap, K, ii, kk, mb, kb);
                    compute_mc_tile<L, S>(ap, bp, C, mb, kb, N, ii, j0, j1);
                }
            }
            std::free(ap);
        }
    }
}

template<int L, KS S>
static void mm_rvv_packed(const float* A, const uint8_t* Bp, float* C, int M, int N, int K)
{
    constexpr int NR = Tr<L>::NR;
    assert(Tr<L>::vl() == (size_t)NR); assert(K%NR==0); assert(N%NR==0);
    const int nth = omp_get_max_threads();
    std::memset(C, 0, (size_t)M*N*sizeof(float));
    if ((M+MC-1)/MC >= nth) goto_m_packed<L, S>(A, Bp, C, M, N, K);
    else                     goto_n_packed<L, S>(A, Bp, C, M, N, K);
}

// ---------------------------------------------------------------------------
// Exported functions: POT style
// ---------------------------------------------------------------------------
void mm_rvv_pot_lmul1(const float* A, const uint8_t* B, float* C, int M, int N, int K) { mm_rvv<1,KS::POT>(A,B,C,M,N,K); }
void mm_rvv_pot_lmul2(const float* A, const uint8_t* B, float* C, int M, int N, int K) { mm_rvv<2,KS::POT>(A,B,C,M,N,K); }
void mm_rvv_pot_lmul4(const float* A, const uint8_t* B, float* C, int M, int N, int K) { mm_rvv<4,KS::POT>(A,B,C,M,N,K); }

void pot_pack_b_v05_lmul1(const uint8_t* B, uint8_t* Bp, int N, int K) { pot_pack_b<1>(B,Bp,N,K); }
void pot_pack_b_v05_lmul2(const uint8_t* B, uint8_t* Bp, int N, int K) { pot_pack_b<2>(B,Bp,N,K); }
void pot_pack_b_v05_lmul4(const uint8_t* B, uint8_t* Bp, int N, int K) { pot_pack_b<4>(B,Bp,N,K); }

void mm_rvv_pot_lmul1_packed(const float* A, const uint8_t* Bp, float* C, int M, int N, int K) { mm_rvv_packed<1,KS::POT>(A,Bp,C,M,N,K); }
void mm_rvv_pot_lmul2_packed(const float* A, const uint8_t* Bp, float* C, int M, int N, int K) { mm_rvv_packed<2,KS::POT>(A,Bp,C,M,N,K); }
void mm_rvv_pot_lmul4_packed(const float* A, const uint8_t* Bp, float* C, int M, int N, int K) { mm_rvv_packed<4,KS::POT>(A,Bp,C,M,N,K); }

// ---------------------------------------------------------------------------
// Exported functions: FMA style (decode u8→float, vfmacc)
// ---------------------------------------------------------------------------
void mm_rvv_fma_lmul1(const float* A, const uint8_t* B, float* C, int M, int N, int K) { mm_rvv<1,KS::FMA>(A,B,C,M,N,K); }
void mm_rvv_fma_lmul2(const float* A, const uint8_t* B, float* C, int M, int N, int K) { mm_rvv<2,KS::FMA>(A,B,C,M,N,K); }
void mm_rvv_fma_lmul4(const float* A, const uint8_t* B, float* C, int M, int N, int K) { mm_rvv<4,KS::FMA>(A,B,C,M,N,K); }

void mm_rvv_fma_lmul1_packed(const float* A, const uint8_t* Bp, float* C, int M, int N, int K) { mm_rvv_packed<1,KS::FMA>(A,Bp,C,M,N,K); }
void mm_rvv_fma_lmul2_packed(const float* A, const uint8_t* Bp, float* C, int M, int N, int K) { mm_rvv_packed<2,KS::FMA>(A,Bp,C,M,N,K); }
void mm_rvv_fma_lmul4_packed(const float* A, const uint8_t* Bp, float* C, int M, int N, int K) { mm_rvv_packed<4,KS::FMA>(A,Bp,C,M,N,K); }

// ---------------------------------------------------------------------------
// Exported functions: MIX style (k+0 FMA, k+1 POT, unrolled ×2)
// ---------------------------------------------------------------------------
void mm_rvv_mix_lmul1(const float* A, const uint8_t* B, float* C, int M, int N, int K) { mm_rvv<1,KS::MIX>(A,B,C,M,N,K); }
void mm_rvv_mix_lmul2(const float* A, const uint8_t* B, float* C, int M, int N, int K) { mm_rvv<2,KS::MIX>(A,B,C,M,N,K); }
void mm_rvv_mix_lmul4(const float* A, const uint8_t* B, float* C, int M, int N, int K) { mm_rvv<4,KS::MIX>(A,B,C,M,N,K); }

void mm_rvv_mix_lmul1_packed(const float* A, const uint8_t* Bp, float* C, int M, int N, int K) { mm_rvv_packed<1,KS::MIX>(A,Bp,C,M,N,K); }
void mm_rvv_mix_lmul2_packed(const float* A, const uint8_t* Bp, float* C, int M, int N, int K) { mm_rvv_packed<2,KS::MIX>(A,Bp,C,M,N,K); }
void mm_rvv_mix_lmul4_packed(const float* A, const uint8_t* Bp, float* C, int M, int N, int K) { mm_rvv_packed<4,KS::MIX>(A,Bp,C,M,N,K); }
