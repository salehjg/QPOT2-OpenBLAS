# v05: RVV 1.0 PoT GEMM for SpacemiT K1 (VLEN=256, in-order).
#
# Cross-compile with:
#   cmake -DCMAKE_TOOLCHAIN_FILE=<path>/toolchain-riscv64.cmake \
#         -DCMAKE_BUILD_TYPE=Release ..
#
# SpacemiT K1 ISA string: rv64gc_zve64x_zvl256b (or rv64gcv for full V 1.0).
# The -march flag below targets the K1 vector extension.

add_executable(bench_v05
    main.cpp
    kernel_v05.cpp
)

target_link_libraries(bench_v05 PRIVATE
    pot_common
    openblas_static
    OpenMP::OpenMP_CXX
    CLI11::CLI11
)

target_include_directories(bench_v05 PRIVATE
    ${OPENBLAS_INCLUDE_DIRS}
)

# RVV 1.0: use the most portable march that enables V extension.
# Prefer the explicit K1 string if your toolchain supports it:
#   -march=rv64gc_zve64d_zvl256b
# Fall back to the full V 1.0 string:
#   -march=rv64gcv
# Override at cmake configure time:
#   cmake -DPOT_RVV_MARCH="rv64gc_zve64d_zvl256b" ..
set(POT_RVV_MARCH "rv64gcv" CACHE STRING
    "RISC-V march string for v05 (e.g. \"rv64gcv\" or \"rv64gc_zve64d_zvl256b\")")

target_compile_options(bench_v05 PRIVATE
    -O3
    -march=${POT_RVV_MARCH}
    # Aggressive math optimisations — safe for our use case (no NaN/Inf in A).
    -ffast-math
    # Omit frame pointer to free up a GPR on in-order cores.
    -fomit-frame-pointer
)
