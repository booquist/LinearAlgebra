cmake_minimum_required( VERSION 3.10 )

project( "LinearAlg" VERSION 1.0 LANGUAGES CXX )

# Add our src
add_subdirectory( src )

set( CMAKE_CXX_STANDARD 17 )

set( target LinearAlg )

set( sources main.cpp )

add_executable( ${target} ${sources} )

target_link_libraries( ${target} 
    PUBLIC
        Vector
        Matrix
        Utilities
    )

target_compile_options( ${target} PUBLIC -O3 )

target_include_directories(LinearAlg PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Interfaces
)

