#include <iostream>
#include "MyLinAlg/Matrix.hpp"
#include "MyLinAlg/Types.hpp"

using namespace myla;

int main() {
    Matrix<double, 2, 2> A{{ {1.0, 2.0}, {3.0, 4.0} }};
    Matrix<double, 2, 2> B{{ {5.0, 6.0}, {7.0, 8.0} }};

    auto C = A.matmul(B);
    std::cout << C(0,0) << ", " << C(0,1) << "\n";
    std::cout << C(1,0) << ", " << C(1,1) << "\n";
    return 0;
}

