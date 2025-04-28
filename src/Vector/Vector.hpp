
#pragma once

#include "Assert.hpp"

// std
#include <type_traits>
#include <vector>
#include <algorithm>
#include <initializer_list>
#include <string> 
#include <sstream>

namespace linear_algebra 
{

template <typename T>
class Vector
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

        int operator [](size_t index) const
        {
            return data[index];
        };

        // Multiplication by a scalar
        void operator *(T&& scalar)
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

        inline void multiply(T&& scalar) { 
            for (size_t i = 0; i < data.size(); ++i)
            {
                data[i] *= scalar;
            }
        }

        // Dot product of two vectors
        int operator *( Vector<T>& vector )
        {
            L_ASSERT( data.size() == vector.data.size(), "Vectors must be of the same size", AssertLevel::CRITICAL );

            int result = 0;
            for (size_t i = 0; i < data.size(); ++i)
            {
                int product = data[i] * vector.data[i];
                result += product;
            }

            return result;
        }

        void print()
        {
            std::ostringstream s("");
            for (size_t i = 0; i < data.size() - 1; ++i)
            {
                s << data[i] << ", ";
            }
            s << data[data.size() - 1] << std::endl;
            std::cout << s.str();
        }

        inline int dot( Vector<T>&& vector ) { return (*this * vector); }
            
        void operator /(T&& scalar)
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

        int at(size_t index) const
        {
            if (index < data.size())
            {
                return data[index];
            }
            throw std::out_of_range("Index out of range");
            return -1;
        };

        void set( size_t index, T value )
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

        // TODO: Implement the cross product 
        //      when Matrices are implemented
        // Vector<T> cross(Vector<T>&& vector);

    private: 
        std::vector<T> data;
};

}