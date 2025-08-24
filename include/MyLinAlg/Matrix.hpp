#pragma once

#include <array>
#include <vector>
#include <cassert>
#include <initializer_list>
#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "Config.hpp"
#include "Expressions.hpp"
#include "Decompositions.hpp"
#include "Assertions.hpp"

namespace myla {

// Helper to compute index (row-major only)
inline std::size_t linear_index(std::size_t r, std::size_t c, std::size_t cols) noexcept {
    return r * cols + c;
}

// Traits for static extent
template <std::size_t Extent>
struct is_dynamic_extent : std::integral_constant<bool, (Extent == Dynamic)> {};

// Matrix class (row-major only)
template <typename T, std::size_t Rows, std::size_t Cols>
class Matrix : public Expr<Matrix<T, Rows, Cols>, T> {
public:
    using value_type = T;
    static constexpr std::size_t rows_at_compile_time = Rows;
    static constexpr std::size_t cols_at_compile_time = Cols;

    // Storage
    using StaticArray = std::array<T, (Rows == Dynamic || Cols == Dynamic) ? 1 : Rows * Cols>;

    Matrix() noexcept {
        if constexpr (Rows == Dynamic || Cols == Dynamic) {
            dyn_rows_ = 0; dyn_cols_ = 0;
        } else {
            data_static_.fill(T{});
        }
    }

    Matrix(std::size_t rows, std::size_t cols) {
        L_ASSERT( std::is_arithmetic<T>::value, "Matrix type must be arithmetic", AssertLevel::CRITICAL );
        
        dyn_rows_ = rows;
        dyn_cols_ = cols;
        data_dynamic_.assign(rows * cols, T{});
    }

    // Initializer list constructor (row-major input)
    Matrix(std::initializer_list<std::initializer_list<T>> init) {
        L_ASSERT( std::is_arithmetic<T>::value, "Matrix type must be arithmetic", AssertLevel::CRITICAL );

        std::size_t r = init.size();
        std::size_t c = (r > 0) ? init.begin()->size() : 0;

        if constexpr (Rows == Dynamic || Cols == Dynamic) {
            dyn_rows_ = r; dyn_cols_ = c; data_dynamic_.assign(r * c, T{});
        } else {
            data_static_.fill(T{});
        }
        std::size_t rr = 0;
        for (const auto& row : init) {
            L_ASSERT( row.size() == c, "Jagged initializer list", AssertLevel::CRITICAL );

            std::size_t cc = 0;
            for (const auto& v : row) {
                (*this)(rr, cc) = v;
                ++cc;
            }
            ++rr;
        }
    }

    // Copy/move default
    Matrix(const Matrix&) = default;
    Matrix(Matrix&&) noexcept = default;
    Matrix& operator=(const Matrix&) = default;
    Matrix& operator=(Matrix&&) noexcept = default;

    // Expression assignment (evaluate into this)
    template <typename ExprType>
    Matrix& operator=(const Expr<ExprType, T>& expr_base) {
        const ExprType& expr = static_cast<const ExprType&>(expr_base);
        resize_if_needed(expr.rows(), expr.cols());
        for (std::size_t r = 0; r < rows(); ++r) {
            for (std::size_t c = 0; c < cols(); ++c) {
                (*this)(r, c) = expr.eval(r, c);
            }
        }
        return *this;
    }

    // Accessors
    constexpr std::size_t rows() const noexcept {
        if constexpr (Rows == Dynamic || Cols == Dynamic) return dyn_rows_;
        else return Rows;
    }
    constexpr std::size_t cols() const noexcept {
        if constexpr (Rows == Dynamic || Cols == Dynamic) return dyn_cols_;
        else return Cols;
    }
    constexpr std::size_t size() const noexcept { return rows() * cols(); }

    T* data() noexcept { return is_dynamic() ? data_dynamic_.data() : data_static_.data(); }
    const T* data() const noexcept { return is_dynamic() ? data_dynamic_.data() : data_static_.data(); }

    // Fast row access (contiguous in row-major layout)
    inline T* row_data(std::size_t r) noexcept { return data() + r * cols(); }
    inline const T* row_data(std::size_t r) const noexcept { return data() + r * cols(); }

    // Element access
    inline T& operator()(std::size_t r, std::size_t c) noexcept {
        return data()[linear_index(r, c, cols())];
    }
    inline const T& operator()(std::size_t r, std::size_t c) const noexcept {
        return data()[linear_index(r, c, cols())];
    }

    // Expr API
    T eval(std::size_t r, std::size_t c) const noexcept { return (*this)(r, c); }

    // Utilities
    bool is_dynamic() const noexcept { return Rows == Dynamic || Cols == Dynamic; }

    void resize_if_needed(std::size_t r, std::size_t c) {
        if constexpr (Rows == Dynamic || Cols == Dynamic) {
            if (dyn_rows_ != r || dyn_cols_ != c) {
                dyn_rows_ = r; dyn_cols_ = c; data_dynamic_.assign(r * c, T{});
            }
        } else {
            L_ASSERT( r == Rows && c == Cols, "Static matrix dimension mismatch in assignment", AssertLevel::CRITICAL );
        }
    }

    // Basic arithmetic creating expressions
    template <typename Other>
    auto operator+(const Other& other) const {
        return BinaryExpr<Matrix, Other, OpAdd>(*this, other);
    }
    template <typename Other>
    auto operator-(const Other& other) const {
        return BinaryExpr<Matrix, Other, OpSub>(*this, other);
    }
    template <typename Scalar>
    auto operator*(const Scalar& s) const {
        return ScalarBinaryExpr<Matrix, Scalar, OpMul>(*this, s);
    }
    template <typename Scalar>
    auto operator/(const Scalar& s) const {
        return ScalarBinaryExpr<Matrix, Scalar, OpDiv>(*this, s);
    }

    // Transpose
    Matrix<T, Cols, Rows> transpose() const {
        Matrix<T, Cols, Rows> out;
        if constexpr (Cols == Dynamic || Rows == Dynamic) {
            out.resize_if_needed(cols(), rows());
        }
        for (std::size_t r = 0; r < rows(); ++r) {
            for (std::size_t c = 0; c < cols(); ++c) {
                out(c, r) = (*this)(r, c);
            }
        }
        return out;
    }

    // Norm (L2)
    T norm() const {
        long double sum = 0.0L;
        for (std::size_t i = 0; i < size(); ++i) sum += static_cast<long double>(data()[i]) * static_cast<long double>(data()[i]);
        return static_cast<T>(std::sqrt(sum));
    }

    // Matrix-matrix multiplication (naive; BLAS if enabled)
    template <std::size_t OtherCols>
    Matrix<T, Rows, OtherCols> matmul(const Matrix<T, Cols, OtherCols>& rhs) const {
        L_ASSERT( cols() == rhs.rows(), "matmul dimension mismatch", AssertLevel::CRITICAL );

        Matrix<T, Rows, OtherCols> out;
        if constexpr (Rows == Dynamic || OtherCols == Dynamic) {
            out.resize_if_needed(rows(), rhs.cols());
        }
        for (std::size_t i = 0; i < rows(); ++i) 
        {
            for (std::size_t k = 0; k < cols(); ++k) 
            {
                T aik = (*this)(i, k);
                for (std::size_t j = 0; j < rhs.cols(); ++j) 
                {
                    out(i, j) += aik * rhs(k, j);
                }
            }
        }
        return out;
    }

    // Determinant and inverse via LU (dynamic path)
    T determinant() const {
        L_ASSERT( rows() == cols(), "determinant requires square matrix", AssertLevel::CRITICAL );

        Matrix<T, Dynamic, Dynamic> A(rows(), cols());
        for (std::size_t i = 0; i < rows(); ++i)
        {
            for (std::size_t j = 0; j < cols(); ++j) A(i,j) = (*this)(i,j);
        }
        auto lu = lu_decompose(A);
        return myla::determinant(lu);
    }

    Matrix inverse() const {
        L_ASSERT( rows() == cols(), "inverse requires square matrix", AssertLevel::CRITICAL );

        Matrix<T, Dynamic, Dynamic> A(rows(), cols());
        for (std::size_t i = 0; i < rows(); ++i)
        {
            for (std::size_t j = 0; j < cols(); ++j)
            {
                A(i,j) = (*this)(i,j);
            }
        }
        auto lu = lu_decompose(A);
        auto invA = myla::inverse(lu);
        Matrix out(rows(), cols());
        for (std::size_t i = 0; i < rows(); ++i)
        {
            for (std::size_t j = 0; j < cols(); ++j)
            {
                out(i,j) = invA(i,j);
            }
        }
        return out;
    }

private:
    // For dynamic
    std::size_t dyn_rows_ { Rows == Dynamic || Cols == Dynamic ? 0 : Rows };
    std::size_t dyn_cols_ { Rows == Dynamic || Cols == Dynamic ? 0 : Cols };
    std::vector<T> data_dynamic_;
    // For static
    StaticArray data_static_{};
};

// dot product for vectors (Nx1 or 1xN)
template <typename T, std::size_t N>
T dot(const Matrix<T, N, 1>& a, const Matrix<T, N, 1>& b) {
    T s{};
    for (std::size_t i = 0; i < N; ++i) s += a(i, 0) * b(i, 0);
    return s;
}

// cross product for 3D vectors (3x1)
template <typename T>
Matrix<T, 3, 1> cross(const Matrix<T, 3, 1>& a,
                                              const Matrix<T, 3, 1>& b) {
    Matrix<T, 3, 1> c;
    c(0,0) = a(1,0)*b(2,0) - a(2,0)*b(1,0);
    c(1,0) = a(2,0)*b(0,0) - a(0,0)*b(2,0);
    c(2,0) = a(0,0)*b(1,0) - a(1,0)*b(0,0);
    return c;
}

} // namespace myla


