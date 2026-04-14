cmake_minimum_required(VERSION 3.20)
project(openblas_qpot LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# OpenBLAS (extern/openblas submodule)
# Build: static, CBLAS only, OpenMP threading.
# TARGET is architecture-dependent: SKYLAKEX on x86, auto-detect on RISC-V.
# ---------------------------------------------------------------------------
if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "riscv")
    set(TARGET "SKYLAKEX" CACHE STRING "" FORCE)
endif()
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
# pot_common (header-only interface — all platforms)
# ---------------------------------------------------------------------------
add_subdirectory(src/common)

# ---------------------------------------------------------------------------
# x86 sub-projects: v01/v02/v03 (AVX-512 intrinsics)
# ---------------------------------------------------------------------------
if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "riscv")
    add_subdirectory(src/v01)
    add_subdirectory(src/v02)
    add_subdirectory(src/v03)
endif()

# ---------------------------------------------------------------------------
# v05: RVV 1.0 PoT kernel (optional — only when cross-compiling for RISC-V)
# ---------------------------------------------------------------------------
if(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv")
    add_subdirectory(src/v05)
    message(STATUS "RISC-V target detected — building v05 (RVV 1.0)")
else()
    message(STATUS "Non-RISC-V host — skipping v05")
endif()

# ---------------------------------------------------------------------------
# v04: CUDA PoT kernel (optional — requires CUDA toolkit)
# ---------------------------------------------------------------------------
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    add_subdirectory(src/v04)
    message(STATUS "CUDA found (${CMAKE_CUDA_COMPILER}) — building v04")
else()
    message(STATUS "CUDA compiler not found — skipping v04")
endif()
