include(FetchContent)

# Prevent GoogleTest from adding install rules
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(INSTALL_GMOCK OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(test_matrix test_matrix.cpp test_lu.cpp)
target_link_libraries(test_matrix PRIVATE MyLinAlg::MyLinAlg GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(test_matrix)

