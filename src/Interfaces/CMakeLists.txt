set( target LinAlgInterfaces )

set( headers 
    IVector.hpp
    IMatrix.hpp
)

add_library( ${target} INTERFACE ${headers} )

target_include_directories( ${target} INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)