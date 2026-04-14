# Common utility library (header-only interface + compiled bench_util bits).
# Provides: PotTestData, median timing, BLAS verification, print helpers.

add_library(pot_common INTERFACE)

target_include_directories(pot_common INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Consumers of pot_common must link OpenBLAS and OpenMP themselves.
