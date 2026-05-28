add_executable(OptixRaytracer
    main.cpp
    Application.h
    Application.cpp
    Camera.h
    Scene.h
    Scene.cpp
    Mesh.h
    Texture.h
    SceneLoader.h
    SceneLoader.cpp
)

target_include_directories(OptixRaytracer PRIVATE
    ${CMAKE_SOURCE_DIR}/shaders
    ${OptiX_INCLUDE_DIR}
)

target_link_libraries(OptixRaytracer PRIVATE
    CUDA::cudart
    CUDA::cuda_driver
    OpenGL::GL
    glad
    glfw
    imgui
    tinygltf
    nfd
)

# Ensure PTX is (re)built whenever source shaders change
add_dependencies(OptixRaytracer device_programs)

# Copy all .ptx files to the executable output directory after each build
add_custom_command(TARGET OptixRaytracer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_OBJECTS:device_programs>
            "$<TARGET_FILE_DIR:OptixRaytracer>/"
    COMMENT "Copying PTX shaders to output directory"
    COMMAND_EXPAND_LISTS
)

# Suppress LNK4098 — GLFW is built with release CRT; /NODEFAULTLIB suppresses
# the "defaultlib LIBCMT conflicts" warning in Debug builds without breaking linkage
target_link_options(OptixRaytracer PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/NODEFAULTLIB:LIBCMT>
)

target_compile_definitions(OptixRaytracer PRIVATE NOMINMAX)

# Visual Studio source groups for cleaner Solution Explorer layout
source_group("Source Files" FILES main.cpp Application.cpp Scene.cpp SceneLoader.cpp)
source_group("Header Files" FILES Application.h Camera.h Scene.h SceneLoader.h Mesh.h Texture.h)
