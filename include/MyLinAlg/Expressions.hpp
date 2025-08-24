#pragma once

#include <type_traits>
#include <cstddef>
#include "Config.hpp"

namespace myla {

// Forward declaration: row-major only
template <typename T, std::size_t Rows, std::size_t Cols>
class Matrix;

// CRTP base for expression templates
template <typename Derived, typename T>
class Expr {
public:
    using value_type = T;

    const Derived& derived() const noexcept { return static_cast<const Derived&>(*this); }
};

// Matrix binary expression for + and -
template <typename Lhs, typename Rhs, typename Op>
class BinaryExpr : public Expr<BinaryExpr<Lhs, Rhs, Op>, typename Lhs::value_type> {
public:
    using value_type = typename Lhs::value_type;
    BinaryExpr(const Lhs& lhs, const Rhs& rhs) : lhs_(lhs), rhs_(rhs) {}

    std::size_t rows() const noexcept { return lhs_.rows(); }
    std::size_t cols() const noexcept { return lhs_.cols(); }

    value_type eval(std::size_t r, std::size_t c) const noexcept {
        return Op::apply(lhs_.eval(r, c), rhs_.eval(r, c));
    }

private:
    const Lhs& lhs_;
    const Rhs& rhs_;
};

struct OpAdd {
    template <typename T>
    static T apply(const T& a, const T& b) noexcept { return a + b; }
};

struct OpSub {
    template <typename T>
    static T apply(const T& a, const T& b) noexcept { return a - b; }
};

// Scalar binary expression for matrix +/-/* scalar
template <typename Lhs, typename TScalar, typename Op>
class ScalarBinaryExpr : public Expr<ScalarBinaryExpr<Lhs, TScalar, Op>, typename Lhs::value_type> {
public:
    using value_type = typename Lhs::value_type;
    ScalarBinaryExpr(const Lhs& lhs, const TScalar& scalar) : lhs_(lhs), scalar_(scalar) {}

    std::size_t rows() const noexcept { return lhs_.rows(); }
    std::size_t cols() const noexcept { return lhs_.cols(); }

    value_type eval(std::size_t r, std::size_t c) const noexcept {
        return Op::apply(lhs_.eval(r, c), static_cast<value_type>(scalar_));
    }

private:
    const Lhs& lhs_;
    TScalar scalar_;
};

struct OpMul {
    template <typename T>
    static T apply(const T& a, const T& b) noexcept { return a * b; }
};

struct OpDiv {
    template <typename T>
    static T apply(const T& a, const T& b) noexcept { return a / b; }
};

// Matrix multiplication expression
template <typename Lhs, typename Rhs>
class MatMulExpr : public Expr<MatMulExpr<Lhs, Rhs>, typename Lhs::value_type> {
public:
    using value_type = typename Lhs::value_type;
    MatMulExpr(const Lhs& lhs, const Rhs& rhs) : lhs_(lhs), rhs_(rhs) {}

    std::size_t rows() const noexcept { return lhs_.rows(); }
    std::size_t cols() const noexcept { return rhs_.cols(); }

    value_type eval(std::size_t r, std::size_t c) const noexcept {
        value_type sum{};
        const std::size_t K = lhs_.cols();
        for (std::size_t k = 0; k < K; ++k) {
            sum += lhs_.eval(r, k) * rhs_.eval(k, c);
        }
        return sum;
    }

private:
    const Lhs& lhs_;
    const Rhs& rhs_;
};

} // namespace myla


