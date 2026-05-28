#ifndef OPTIX_RAYTRACER_APPLICATION_H
#define OPTIX_RAYTRACER_APPLICATION_H

#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
// GLFW_INCLUDE_NONE prevents glfw3.h from pulling in GL headers — glad owns that
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "LaunchParams.h"
#include "Scene.h"

#include <memory>
#include <string>

class Application
{
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

    std::unique_ptr<Scene> m_scene;

    // Framebuffer — filled by device programs, displayed via OpenGL texture
    uchar4*      d_colorBuffer   = nullptr;  // CUDA device buffer
    uchar4*      h_colorBuffer   = nullptr;  // host staging buffer
    unsigned int m_displayTexture = 0;       // GL_RGBA8 texture (GLuint)
    int          m_viewportWidth  = 0;       // current framebuffer dimensions
    int          m_viewportHeight = 0;       // driven by the Viewport panel size

    void initWindow(const std::string& title);
    void initOpenGL();
    void initImGui();
    void initCuda();
    void initOptix();
    void resizeFramebuffer(int w, int h);

    void loadScene(const std::string& path);

    static void optixLogCallback(unsigned int level,
                                 const char*  tag,
                                 const char*  message,
                                 void*        cbdata);

    std::string m_sceneFilePath;  // empty = no scene loaded
    std::string m_loadError;      // empty = no error
};

#endif // OPTIX_RAYTRACER_APPLICATION_H
