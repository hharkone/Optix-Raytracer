#ifndef OPTIX_RAYTRACER_APPLICATION_H
#define OPTIX_RAYTRACER_APPLICATION_H

#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
// GLFW_INCLUDE_NONE prevents glfw3.h from pulling in GL headers — glad owns that
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "Accel.h"
#include "LaunchParams.h"
#include "Scene.h"

#include <memory>
#include <string>

class Application
{
public:
    // ptxDir: directory containing the compiled .ptx shader files (usually the
    // same directory as the executable — pass std::filesystem::path(argv[0]).parent_path()).
    Application(int width, int height, const std::string& title,
                const std::string& ptxDir = ".");
    ~Application();

    // Returns false when the window should close.
    bool tick();

private:
    GLFWwindow*        m_window        = nullptr;
    int                m_width         = 0;
    int                m_height        = 0;

    OptixDeviceContext m_optixContext   = nullptr;

    std::unique_ptr<Scene> m_scene;
    std::unique_ptr<Accel> m_accel;  // null = no scene loaded or AS not yet built

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
    void buildPipeline(const std::string& ptxDir);
    void buildSbt();
    void resizeFramebuffer(int w, int h);

    void loadScene(const std::string& path);

    static void optixLogCallback(unsigned int level,
                                 const char*  tag,
                                 const char*  message,
                                 void*        cbdata);

    // OptiX pipeline (built once at startup)
    OptixModule       m_module      = nullptr;
    OptixProgramGroup m_pgRaygen    = nullptr;
    OptixProgramGroup m_pgMiss      = nullptr;
    OptixProgramGroup m_pgHitgroup  = nullptr;
    OptixPipeline     m_pipeline    = nullptr;

    // Shader binding table (rebuilt whenever the scene changes)
    CUdeviceptr             m_sbtRaygenBuffer   = 0;
    CUdeviceptr             m_sbtMissBuffer     = 0;
    CUdeviceptr             m_sbtHitgroupBuffer = 0;
    OptixShaderBindingTable m_sbt               = {};

    // Launch parameters — host struct updated each frame, device copy passed to optixLaunch
    LaunchParams m_launchParams       = {};
    CUdeviceptr  m_launchParamsBuffer = 0;

    std::string m_sceneFilePath;  // empty = no scene loaded
    std::string m_loadError;      // empty = no error
};

#endif // OPTIX_RAYTRACER_APPLICATION_H
