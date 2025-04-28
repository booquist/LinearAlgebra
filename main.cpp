#include "Vector.hpp"
#include <iostream>
#include <chrono>

int main() 
{
    // Create a vector
    linear_algebra::Vector<double> vec1({1, 1, 1});

    std::cout << "Before multiplication:" << std::endl;
    vec1.print();

    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    for ( int i = 0; i < 50000; ++i )
    {
        vec1.multiply( 1.0001 );
    }
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

    std::cout << "After multiplication:" << std::endl;
    vec1.print();
    std::cout << "Time taken: " 
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() 
              << " microseconds" << std::endl;

    return 0;
}