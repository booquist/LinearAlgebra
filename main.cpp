#include "Vector.hpp"
#include "Matrix.hpp"

// std
#include <iostream>
#include <chrono>

int main() 
{
    // Create a vector
    linear_algebra::Vector<double> vec1({1, 1, 1});

    std::cout << "Before multiplication:" << std::endl;
    vec1.print();

    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    for ( int i = 0; i < 500000; ++i )
    {
        vec1.multiply( 1.000001 );
    }
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

    std::cout << "After multiplication:" << std::endl;
    vec1.print();
    std::cout << "Time taken: " 
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() 
              << " microseconds" << std::endl;

    linear_algebra::Matrix<double> mat1(3, 3);
    mat1.setRow(0, {1, 2, 3});
    mat1.setRow(1, {4, 5, 6});
    mat1.setRow(2, {7, 8, 9});

    linear_algebra::Matrix<double> mat2(3, 3);
    mat2.setRow(0, {2, 0, 0});
    mat2.setRow(1, {0, 2, 0});
    mat2.setRow(2, {0, 0, 2});

    mat1.print();

    mat1.multiply(mat2);

    mat1.print();

    return 0;
}