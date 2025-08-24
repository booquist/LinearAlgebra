#pragma once

#include <vector>
#include <stdexcept>
#include <cmath>
#include "Matrix.hpp"

namespace myla {

// Simple LU decomposition with partial pivoting for square matrices
template <typename T>
struct LUResult {
    Matrix<T, Dynamic, Dynamic> L;
    Matrix<T, Dynamic, Dynamic> U;
    std::vector<std::size_t> piv;
    int sign = 1;
};

template <typename T>
LUResult<T> lu_decompose(const Matrix<T, Dynamic, Dynamic>& A) {
    const std::size_t n = A.rows();
    if (n != A.cols()) throw std::invalid_argument("LU requires square matrix");
    Matrix<T, Dynamic, Dynamic> U = A;
    Matrix<T, Dynamic, Dynamic> L(n, n);
    for (std::size_t i = 0; i < n; ++i) 
    {
        L(i,i) = 1;
    }
    std::vector<std::size_t> piv(n);
    for (std::size_t i = 0; i < n; ++i) 
    {
        piv[i] = i;
    }
    int sign = 1;

    for (std::size_t k = 0; k < n; ++k) {
        // Pivot
        std::size_t maxRow = k;
        T maxVal = std::abs(U(k, k));
        for (std::size_t i = k + 1; i < n; ++i) 
        {
            T val = std::abs(U(i, k));
            if (val > maxVal) { maxVal = val; maxRow = i; }
        }
        if (maxVal == T{}) throw std::runtime_error("Singular matrix in LU");
        if (maxRow != k) 
        {
            for (std::size_t j = 0; j < n; ++j) std::swap(U(k,j), U(maxRow,j));
            for (std::size_t j = 0; j < k; ++j) std::swap(L(k,j), L(maxRow,j));
            std::swap(piv[k], piv[maxRow]);
            sign = -sign;
        }
        // Elimination
        for (std::size_t i = k + 1; i < n; ++i) 
        {
            L(i, k) = U(i, k) / U(k, k);
            for (std::size_t j = k; j < n; ++j) 
            {
                U(i, j) -= L(i, k) * U(k, j);
            }
        }
    }
    return {L, U, piv, sign};
}

template <typename T>
Matrix<T, Dynamic, 1> lu_solve(const LUResult<T>& lu,
                                                       const Matrix<T, Dynamic, 1>& b) {
    const std::size_t n = b.rows();
    Matrix<T, Dynamic, 1> x(n, 1);
    Matrix<T, Dynamic, 1> y(n, 1);

    // Apply permutation to b
    for (std::size_t i = 0; i < n; ++i) y(i,0) = b(lu.piv[i], 0);

    // Forward substitution Ly = Pb
    for (std::size_t i = 0; i < n; ++i) {
        T sum{};
        for (std::size_t j = 0; j < i; ++j) sum += lu.L(i,j) * y(j,0);
        y(i,0) -= sum;
    }

    // Backward substitution Ux = y
    for (std::size_t i = n; i-- > 0;) {
        T sum{};
        for (std::size_t j = i + 1; j < n; ++j) sum += lu.U(i,j) * x(j,0);
        x(i,0) = (y(i,0) - sum) / lu.U(i,i);
    }
    return x;
}

template <typename T>
T determinant(const LUResult<T>& lu) {
    T det = static_cast<T>(lu.sign);
    for (std::size_t i = 0; i < lu.U.rows(); ++i) det *= lu.U(i,i);
    return det;
}

template <typename T>
Matrix<T, Dynamic, Dynamic> inverse(const LUResult<T>& lu) {
    const std::size_t n = lu.U.rows();
    Matrix<T, Dynamic, Dynamic> inv(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        Matrix<T, Dynamic, 1> e(n, 1);
        e(i,0) = 1;
        auto col = lu_solve(lu, e);
        for (std::size_t r = 0; r < n; ++r) inv(r, i) = col(r,0);
    }
    return inv;
}

} // namespace myla


