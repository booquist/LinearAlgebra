set( target Vector )

add_library( ${target} INTERFACE )

target_include_directories( ${target} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR} )

target_link_libraries( ${target} INTERFACE 
    Utilities
    )