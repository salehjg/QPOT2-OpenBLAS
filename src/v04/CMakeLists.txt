find_package(CUDAToolkit REQUIRED)

add_executable(bench_v04
    main.cu
    kernel_v04.cu
)

# Default to native GPU only.
# Override: cmake -DPOT_CUDA_ARCH="70;89" ..
# Note: sm_70 (V100) requires CUDA ≤ 12.x — dropped in CUDA 13+.
set(POT_CUDA_ARCH "native" CACHE STRING "CUDA architectures (e.g. \"89\" or \"70;89\")")

set_target_properties(bench_v04 PROPERTIES
    CUDA_ARCHITECTURES "${POT_CUDA_ARCH}"
    CUDA_SEPARABLE_COMPILATION OFF
)

target_link_libraries(bench_v04 PRIVATE
    pot_common
    CLI11::CLI11
    CUDA::cublas
    CUDA::cudart
)

target_include_directories(bench_v04 PRIVATE
    ${CMAKE_SOURCE_DIR}/src/common
)

target_compile_options(bench_v04 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3>
)
