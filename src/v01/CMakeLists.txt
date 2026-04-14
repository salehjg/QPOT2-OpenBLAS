add_executable(bench_v01
    main.cpp
    kernel_v01.cpp
)

target_link_libraries(bench_v01 PRIVATE
    pot_common
    openblas_static
    OpenMP::OpenMP_CXX
    CLI11::CLI11
)

target_include_directories(bench_v01 PRIVATE
    ${OPENBLAS_INCLUDE_DIRS}
)

target_compile_options(bench_v01 PRIVATE -O3 -march=skylake-avx512)
