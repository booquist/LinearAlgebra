#pragma once

#include <vector>
#include <initializer_list>

namespace linear_algebra
{
    template <typename T>
    class IMatrix
    {
    public:
        virtual ~IMatrix() = default;

        // Element access operators
        virtual std::vector<T>& operator[](size_t index) = 0;
        virtual T& operator()(size_t row, size_t column) = 0;

        // Data access methods
        virtual std::vector<std::vector<T>>& getData() = 0;
        virtual std::vector<T> getRow(size_t index) const = 0;
        virtual T get(size_t row, size_t column) const = 0;
        virtual std::vector<T> getColumn(size_t index) const = 0;

        // Data modification methods
        virtual void setColumn(size_t index, const std::vector<T>& column) = 0;
        virtual void setRow(size_t index, const std::vector<T>& row) = 0;

        // Scalar operations
        virtual void operator*(T&& scalar) = 0;

        // Output operations
        virtual void print() = 0;
    };
}