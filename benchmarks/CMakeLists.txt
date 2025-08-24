cmake_minimum_required(VERSION 3.14)

find_package(Eigen3 REQUIRED)

add_executable(bench_basic bench_basic.cpp)
target_link_libraries(bench_basic PRIVATE MyLinAlg::MyLinAlg Eigen3::Eigen)

target_compile_features(bench_basic PRIVATE cxx_std_17)
target_compile_options(bench_basic PRIVATE -O3 -march=native)