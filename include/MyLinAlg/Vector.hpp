#pragma once

#include <cmath>
#include "MyLinAlg/Types.hpp"

namespace myla {

// Dot products
template <typename T, std::size_t N>
inline T dot(const Matrix<T, N, 1>& a, const Matrix<T, N, 1>& b) {
    T s{};
    for (std::size_t i = 0; i < N; ++i) s += a(i,0) * b(i,0);
    return s;
}

template <typename T>
inline T dot(const Matrix<T, Dynamic, 1>& a, const Matrix<T, Dynamic, 1>& b) {
    if (a.rows() != b.rows()) throw std::invalid_argument("dot size mismatch");
    T s{};
    for (std::size_t i = 0; i < a.rows(); ++i) s += a(i,0) * b(i,0);
    return s;
}

// 2D scalar cross product (z-component of 3D cross with z=0)
template <typename T>
inline T cross2(const Matrix<T, 2, 1>& a, const Matrix<T, 2, 1>& b) {
    return a(0,0) * b(1,0) - a(1,0) * b(0,0);
}

// 3D vector cross product
template <typename T>
inline Matrix<T, 3, 1> cross(const Matrix<T, 3, 1>& a, const Matrix<T, 3, 1>& b) {
    Matrix<T, 3, 1> c;
    c(0,0) = a(1,0)*b(2,0) - a(2,0)*b(1,0);
    c(1,0) = a(2,0)*b(0,0) - a(0,0)*b(2,0);
    c(2,0) = a(0,0)*b(1,0) - a(1,0)*b(0,0);
    return c;
}

// Norms
template <typename T, std::size_t N>
inline T norm(const Matrix<T, N, 1>& a) {
    long double s = 0.0L;
    for (std::size_t i = 0; i < N; ++i) s += static_cast<long double>(a(i,0)) * a(i,0);
    return static_cast<T>(std::sqrt(s));
}

template <typename T>
inline T norm(const Matrix<T, Dynamic, 1>& a) {
    long double s = 0.0L;
    for (std::size_t i = 0; i < a.rows(); ++i) s += static_cast<long double>(a(i,0)) * a(i,0);
    return static_cast<T>(std::sqrt(s));
}

} // namespace myla


