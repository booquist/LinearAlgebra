
#pragma once

#include "Assert.hpp"
#include "IVector.hpp"

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
class Vector
    : public IVector<T>
{
    public: 
        Vector(size_t size) : data(size)
        {
            L_ASSERT( std::is_arithmetic<T>::value, "Vector type must be arithmetic", AssertLevel::CRITICAL );

            // Initialize the vector with zeros
            data.resize(size);
        }

        Vector(std::initializer_list<T> init) : data(init)
        {
            L_ASSERT( std::is_arithmetic<T>::value, "Vector type must be arithmetic", AssertLevel::CRITICAL );

            // Initialize the vector with the provided initializer list
            data = std::vector<T>(init);
        }

        ~Vector() = default;

        virtual int operator[](size_t index) const override
        {
            return data[index];
        };

        // Multiplication by a scalar
        void operator *(T&& scalar) override
        {
            for (size_t i = 0; i < data.size(); ++i)
            {
                data[i] *= scalar;
            }
        }

        Vector<T> operator *(T&& scalar) const
        {
            Vector<T> result(data.size());
            for (size_t i = 0; i < data.size(); ++i)
            {
                result.data[i] = data[i] * scalar;
            }

            return result;
        }

        inline void multiply(T&& scalar) override { 
            for (size_t i = 0; i < data.size(); ++i)
            {
                data[i] *= scalar;
            }
        }

        // Dot product of two vectors
        int operator *( IVector<T>& vector ) override
        {
            std::vector<T>& vector_data = vector.getData();
            L_ASSERT( data.size() == vector_data.size(), "Vectors must be of the same size", AssertLevel::CRITICAL );

            int result = 0;
            for (size_t i = 0; i < data.size(); ++i)
            {
                int product = data[i] * vector_data[i];
                result += product;
            }

            return result;
        }

        void print() override
        {
            std::ostringstream s("");
            for (size_t i = 0; i < data.size() - 1; ++i)
            {
                s << data[i] << ", ";
            }
            s << data[data.size() - 1] << std::endl;
            std::cout << s.str();
        }

        inline int dot( IVector<T>&& vector ) override { return (*this * vector); }
            
        void operator /(T&& scalar) override
        {
            if ( scalar == 0 )
            {
                throw std::invalid_argument("Division by zero");
            }
            for (size_t i = 0; i < data.size(); ++i)
            {
                data[i] /= scalar;
            }
        }

        int at(size_t index) const override
        {
            if (index < data.size())
            {
                return data[index];
            }
            throw std::out_of_range("Index out of range");
            return -1;
        };

        void set( size_t index, T value ) override
        {
            if (index < data.size())
            {
                data[index] = value;
            }
            else
            {
                throw std::out_of_range("Index out of range");
            }
        };

        std::vector<T>& getData() override
        {
            return data;
        }

        // TODO: Implement the cross product 
        //      when Matrices are implemented
        // Vector<T> cross(Vector<T>&& vector);

    private: 
        std::vector<T> data;
};

}