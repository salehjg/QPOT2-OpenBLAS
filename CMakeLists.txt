cmake_minimum_required(VERSION 3.20)
project(openblas_qpot LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# OpenBLAS (extern/openblas submodule)
# Build: static, CBLAS only, OpenMP threading, SKYLAKEX target.
# ---------------------------------------------------------------------------
set(TARGET       "SKYLAKEX" CACHE STRING "" FORCE)
set(BUILD_STATIC_LIBS    ON  CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS    OFF CACHE BOOL "" FORCE)
set(BUILD_WITHOUT_LAPACK ON  CACHE BOOL "" FORCE)
set(BUILD_TESTING        OFF CACHE BOOL "" FORCE)
set(BUILD_BENCHMARKS     OFF CACHE BOOL "" FORCE)
set(NO_FORTRAN           ON  CACHE BOOL "" FORCE)
set(C_LAPACK             OFF CACHE BOOL "" FORCE)
set(USE_OPENMP           ON  CACHE BOOL "" FORCE)

add_subdirectory(extern/openblas)

# Collect OpenBLAS include paths for consumers.
set(OPENBLAS_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/extern/openblas
    ${CMAKE_BINARY_DIR}/extern/openblas
)

# ---------------------------------------------------------------------------
# CLI11 (extern/cli11 submodule — header-only)
# ---------------------------------------------------------------------------
add_subdirectory(extern/cli11)

# ---------------------------------------------------------------------------
# OpenMP
# ---------------------------------------------------------------------------
find_package(OpenMP REQUIRED)

# ---------------------------------------------------------------------------
# Sub-projects
# ---------------------------------------------------------------------------
add_subdirectory(src/common)
add_subdirectory(src/v01)
add_subdirectory(src/v02)
