#include <gtest/gtest.h>
#include "MyLinAlg/Types.hpp"

using namespace myla;

TEST(MatrixBasics, StaticConstructionAndAccess) {
    Matrix<double, 2, 2> A{{ {1.0, 2.0}, {3.0, 4.0} }};
    EXPECT_NEAR(A(0,0), 1.0, 1e-12);
    EXPECT_NEAR(A(1,1), 4.0, 1e-12);
}

TEST(MatrixBasics, DynamicConstructionAndAssign) {
    DynamicMatrixXd B(2, 2);
    B = Matrix<double, Dynamic, Dynamic>{{ {1.0, 2.0}, {3.0, 4.0} }};
    EXPECT_NEAR(B(0,1), 2.0, 1e-12);
}

TEST(MatrixOps, MatMul) {
    Matrix<double, 2, 2> A{{ {1.0, 2.0}, {3.0, 4.0} }};
    Matrix<double, 2, 2> B{{ {5.0, 6.0}, {7.0, 8.0} }};
    auto C = A.matmul(B);
    EXPECT_NEAR(C(0,0), 19.0, 1e-12);
    EXPECT_NEAR(C(0,1), 22.0, 1e-12);
    EXPECT_NEAR(C(1,0), 43.0, 1e-12);
    EXPECT_NEAR(C(1,1), 50.0, 1e-12);
}

TEST(VectorOps, DotCross) {
    Vector3d a; a(0,0)=1.0; a(1,0)=2.0; a(2,0)=3.0;
    Vector3d b; b(0,0)=4.0; b(1,0)=5.0; b(2,0)=6.0;
    double d = dot(a, b);
    EXPECT_NEAR(d, 32.0, 1e-12);
    auto c = cross(a, b);
    EXPECT_NEAR(c(0,0), -3.0, 1e-12);
    EXPECT_NEAR(c(1,0), 6.0, 1e-12);
    EXPECT_NEAR(c(2,0), -3.0, 1e-12);
}

