#pragma once

#include "Matrix.hpp"
#include "Decompositions.hpp"

namespace myla {

// High-level solve: choose LU for square systems, simple least-squares via normal equations otherwise
template <typename T, std::size_t R, std::size_t C>
auto solve(const Matrix<T, R, C>& A, const Matrix<T, Dynamic, 1>& b)
    -> Matrix<T, Dynamic, 1>
{
    if (A.rows() == A.cols()) {
        auto A_dyn = Matrix<T, Dynamic, Dynamic>(A.rows(), A.cols());
        for (std::size_t i = 0; i < A.rows(); ++i)
            for (std::size_t j = 0; j < A.cols(); ++j) A_dyn(i,j) = A(i,j);
        auto lu = lu_decompose(A_dyn);
        return lu_solve(lu, b);
    }
    // Least squares: x = (A^T A)^{-1} A^T b (naive)
    Matrix<T, Dynamic, Dynamic> AtA(A.cols(), A.cols());
    for (std::size_t i = 0; i < A.cols(); ++i)
        for (std::size_t j = 0; j < A.cols(); ++j) {
            T s{}; for (std::size_t k = 0; k < A.rows(); ++k) s += A(k,i) * A(k,j); AtA(i,j) = s;
        }
    Matrix<T, Dynamic, 1> Atb(A.cols(), 1);
    for (std::size_t i = 0; i < A.cols(); ++i) {
        T s{}; for (std::size_t k = 0; k < A.rows(); ++k) s += A(k,i) * b(k,0); Atb(i,0) = s;
    }
    auto lu = lu_decompose(AtA);
    return lu_solve(lu, Atb);
}

} // namespace myla


