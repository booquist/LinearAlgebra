
#pragma once

#include "Assert.hpp"
#include "IMatrix.hpp"

// std
#include <type_traits>
#include <vector>
#include <algorithm>
#include <initializer_list>
#include <string> 
#include <sstream>
#include <iostream>

namespace linear_algebra 
{

template <typename T>
class Matrix
    : public IMatrix<T>
{
    public: 
        Matrix(size_t m, size_t n)
        {
            L_ASSERT( std::is_arithmetic<T>::value, "Matrix type must be arithmetic", AssertLevel::CRITICAL );

            // Initialize the Matrix with zeros
            data.resize(m);

            for ( std::vector<T> & row : data )
            {
                row.resize(n);
            }
        }

        Matrix(std::initializer_list<std::initializer_list<T>> init) : data(init)
        {
            L_ASSERT( std::is_arithmetic<T>::value, "Matrix type must be arithmetic", AssertLevel::CRITICAL );

            data.resize(init.size());
            size_t i = 0;
            for (const auto& row : init)
            {
                data[i].resize(row.size());
                std::move(row.begin(), row.end(), data[i].begin());
                ++i;
            }
        }

        ~Matrix() = default;

        // Get a row of data
        inline virtual std::vector<T>& operator[](size_t index) override { return data[index]; };
        virtual T& operator()(size_t row, size_t column) override
        {
            if (row < data.size() && column < data[row].size())
            {
                return data[row][column];
            }
            throw std::out_of_range("Index out of range");
        };

        inline std::vector<std::vector<T>>& getData() override
        {
            return data;
        }

        inline virtual std::vector<T> getRow(size_t index) const override
        {
            if (index < data.size())
            {
                return data[index];
            }
            throw std::out_of_range("Index out of range");
        };

        inline virtual T get(size_t row, size_t column) const override
        {
            if (row < data.size() && column < data[row].size())
            {
                return data[row][column];
            }
            throw std::out_of_range("Index out of range");
        };

        inline virtual std::vector<T> getColumn(size_t index) const override
        {
            if (index < data[0].size())
            {
                std::vector<T> column(data.size());
                for (size_t i = 0; i < data.size(); ++i)
                {
                    column[i] = data[i][index];
                }
                return column;
            }
            throw std::out_of_range("Index out of range");
        };

        inline virtual void setColumn(size_t index, const std::vector<T>& column) override
        {
            if (index < data[0].size() && column.size() == data.size())
            {
                for (size_t i = 0; i < data.size(); ++i)
                {
                    data[i][index] = column[i];
                }
            }
            else
            {
                throw std::out_of_range("Index out of range or column size mismatch");
            }
        };

        inline virtual void setRow(size_t index, const std::vector<T>& row) override
        {
            if (index < data.size() && row.size() == data[index].size())
            {
                data[index] = row;
            }
            else
            {
                throw std::out_of_range("Index out of range or row size mismatch");
            }
        };

        void multiply( IMatrix<T>& matrix )
        {
            if (data[0].size() != matrix.getData().size())
            {
                throw std::invalid_argument("Matrix dimensions do not match for multiplication");
            }

            std::vector<std::vector<T>> result(data.size(), std::vector<T>(matrix.getData()[0].size(), 0));

            for (size_t i = 0; i < data.size(); ++i)
            {
                for (size_t j = 0; j < matrix.getData()[0].size(); ++j)
                {
                    for (size_t k = 0; k < data[0].size(); ++k)
                    {
                        result[i][j] += data[i][k] * matrix.getData()[k][j];
                    }
                }
            }

            data = result;
        }

        void operator *(T&& scalar) override
        {
            for (size_t i = 0; i < data.size(); ++i)
            {
                for (size_t j = 0; j < data[i].size(); ++j)
                {
                    data[i][j] *= scalar;
                }
            }
        }

        virtual void print() override
        {
            std::ostringstream s("");
            for (size_t i = 0; i < data.size(); ++i)
            {
                for (size_t j = 0; j < data[i].size(); ++j)
                {
                    s << data[i][j] << " ";
                }
                s << std::endl;
            }
            std::cout << s.str();
        }

    private: 
        std::vector<std::vector<T>> data;
};

}