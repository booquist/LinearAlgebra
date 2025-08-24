cmake_minimum_required(VERSION 3.14)

project(MyLinAlg VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(MYLIB_BUILD_TESTS "Build tests" ON)
option(MYLIB_BUILD_EXAMPLES "Build examples" ON)
option(MYLIB_BUILD_BENCHMARKS "Build benchmarks" ON)
option(MYLIB_USE_BLAS "Enable BLAS/LAPACK backend" OFF)
option(MYLIB_USE_SIMD "Enable SIMD optimizations" OFF)

# Optimization toggles
option(MYLIB_ENABLE_LTO "Enable link-time optimization (IPO) for Release/RelWithDebInfo" OFF)
option(MYLIB_FAST_MATH "Enable fast-math (may change numerical behavior)" OFF)
option(MYLIB_MARCH_NATIVE "Enable -march=native (or /arch:AVX2 on MSVC)" ON)

# Default build type for single-config generators
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

# Header-only interface target
add_library(MyLinAlg INTERFACE)
add_library(MyLinAlg::MyLinAlg ALIAS MyLinAlg)

target_include_directories(MyLinAlg
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

if(MYLIB_USE_BLAS)
    find_package(BLAS QUIET)
    if(BLAS_FOUND)
        target_compile_definitions(MyLinAlg INTERFACE MYLINALG_USE_BLAS)
        target_link_libraries(MyLinAlg INTERFACE BLAS::BLAS)
    else()
        message(WARNING "MYLIB_USE_BLAS=ON but BLAS not found. Falling back to built-in kernels.")
    endif()
endif()

# Position-independent code is generally a good default for consumers
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Warnings + optimizations, cross-compiler and config-gated
target_compile_options(MyLinAlg INTERFACE
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /permissive->
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>

    # Optimize only in Release/RelWithDebInfo
    $<$<AND:$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:-O3>
    $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:/O2>

    # Optional native tuning
    $<$<AND:$<BOOL:${MYLIB_MARCH_NATIVE}>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-march=native>
    $<$<AND:$<BOOL:${MYLIB_MARCH_NATIVE}>,$<CXX_COMPILER_ID:MSVC>>:/arch:AVX2>

    # Optional fast math (be careful)
    $<$<AND:$<BOOL:${MYLIB_FAST_MATH}>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-ffast-math -fno-math-errno>
    $<$<AND:$<BOOL:${MYLIB_FAST_MATH}>,$<CXX_COMPILER_ID:MSVC>>:/fp:fast>
)

# Interprocedural optimization (LTO) for Release-like configs if requested
if(MYLIB_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_supported OUTPUT ipo_msg)
    if(ipo_supported)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO TRUE)
    else()
        message(WARNING "LTO requested but not supported: ${ipo_msg}")
    endif()
endif()

# Examples
if(MYLIB_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Tests
if(MYLIB_BUILD_TESTS)
    include(CTest)
    if(BUILD_TESTING)
        add_subdirectory(tests)
    endif()
endif()

# Benchmarks
if(MYLIB_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Install rules
include(GNUInstallDirs)
install(TARGETS MyLinAlg
    EXPORT MyLinAlgTargets
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT MyLinAlgTargets
    NAMESPACE MyLinAlg::
    FILE MyLinAlgTargets.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MyLinAlg
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/MyLinAlgConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/MyLinAlgConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/MyLinAlgConfig.cmake
    @ONLY
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/MyLinAlgConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/MyLinAlgConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MyLinAlg
)
