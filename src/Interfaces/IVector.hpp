#pragma once

#include <vector>

namespace linear_algebra
{
    template <typename T>
    class IVector 
    {
    public:
        // Virtual destructor for proper cleanup in derived classes
        virtual ~IVector() = default;
        
        // Basic access operations
        virtual int operator[](size_t index) const = 0;
        virtual int at(size_t index) const = 0;
        virtual void set(size_t index, T value) = 0;
        
        // Scalar operations
        virtual void operator*(T&& scalar) = 0;
        virtual void multiply(T&& scalar) = 0;
        virtual void operator/(T&& scalar) = 0;
        
        // Vector operations
        virtual int operator*(IVector<T>& vector) = 0;
        virtual int dot(IVector<T>&& vector) = 0;

        // Getters
        virtual std::vector<T>& getData() = 0;
        
        // Utility
        virtual void print() = 0;
    };
}