#pragma once

#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
// GLFW_INCLUDE_NONE prevents glfw3.h from pulling in GL headers — glad owns that
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "LaunchParams.h"

#include <string>

class Application {
public:
    Application(int width, int height, const std::string& title);
    ~Application();

    // Returns false when the window should close.
    bool tick();

private:
    GLFWwindow*        m_window        = nullptr;
    int                m_width         = 0;
    int                m_height        = 0;

    OptixDeviceContext m_optixContext   = nullptr;

    // Framebuffer — filled by device programs, displayed via OpenGL texture
    uchar4*            d_colorBuffer   = nullptr;
    uchar4*            h_colorBuffer   = nullptr;

    void initWindow(const std::string& title);
    void initOpenGL();
    void initImGui();
    void initCuda();
    void initOptix();

    static void optixLogCallback(unsigned int level,
                                 const char*  tag,
                                 const char*  message,
                                 void*        cbdata);
};
