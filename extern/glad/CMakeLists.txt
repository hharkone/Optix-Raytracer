add_library(glad STATIC
    src/glad.c
)
target_include_directories(glad PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_options(glad PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W0>
)
