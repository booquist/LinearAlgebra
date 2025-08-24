#include <gtest/gtest.h>
#include "MyLinAlg/Matrix.hpp"
#include "MyLinAlg/Decompositions.hpp"

using namespace myla;

TEST(LU, SolveAndDet) {
    Matrix<double, Dynamic, Dynamic> A(2, 2);
    A(0,0)=4.0; A(0,1)=3.0; A(1,0)=6.0; A(1,1)=3.0;
    auto lu = lu_decompose(A);
    auto detA = determinant(lu);
    EXPECT_NEAR(detA, -6.0, 1e-12);
    Matrix<double, Dynamic, 1> b(2, 1);
    b(0,0)=10.0; b(1,0)=12.0;
    auto x = lu_solve(lu, b);
    EXPECT_NEAR(x(0,0), 1.0, 1e-12);
    EXPECT_NEAR(x(1,0), 2.0, 1e-12);
}

