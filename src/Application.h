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
#include "Texture.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
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

    // Hot-reload: rebuild the pipeline from the PTX file whenever it changes on disk.
    // reloadPipeline() swaps in the new pipeline atomically — the old one stays alive
    // until the new one is confirmed working, so a broken shader never kills the render.
    void reloadPipeline();
    void checkShaderHotReload();

    void loadScene(const std::string& path);
    void loadEnvMap(const std::string& path);
    void uploadMaterials();  // (re)upload host materials to the GPU buffer

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

    // Scene materials on device (uploaded when a scene is loaded)
    CUdeviceptr m_materialsBuffer = 0;

    // Sample accumulation — reset whenever the camera, scene, or env map changes
    CUdeviceptr m_accumBuffer  = 0;   // float4, width * height
    uint32_t    m_sampleCount  = 0;
    bool        m_accumDirty   = true; // true = clear before the next launch

    // Launch parameters — host struct updated each frame, device copy passed to optixLaunch
    LaunchParams m_launchParams       = {};
    CUdeviceptr  m_launchParamsBuffer = 0;

    std::string m_sceneFilePath;  // empty = no scene loaded
    std::string m_loadError;      // empty = no error

    // Environment map (lat-long EXR)
    Texture     m_envMap;          // RGBA32F; gpuTex == 0 = not loaded
    std::string m_envMapPath;      // display name
    std::string m_envMapError;     // non-empty = last load failed

    // Hot-reload state
    std::string                      m_ptxDir;
    std::filesystem::file_time_type  m_ptxWriteTime = {};
    std::string                      m_shaderError;   // non-empty = last reload failed

    // GPU device info — queried once in initCuda()
    std::string   m_deviceName;
    int           m_deviceComputeMajor = 0;
    int           m_deviceComputeMinor = 0;
    std::uint64_t m_deviceMemoryMB     = 0;

    // Frame timing — updated at the top of each tick()
    std::chrono::steady_clock::time_point m_frameStart;
    float m_frameTimeMs = 0.0f;  // exponential moving average of per-frame duration

    // Free-fly camera state — the scene camera matrix is rebuilt from these each frame
    float3 m_camPos    = { 0.0f, 0.0f, 3.0f };
    float  m_camYaw    = 0.0f;     // radians, rotation around world Y; 0 = facing -Z
    float  m_camPitch  = 0.0f;     // radians, tilt around camera X; positive = look up
    float  m_moveSpeed = 5.0f;     // world units / second (WASD)
    float  m_rotSpeed  = 0.003f;  // radians / pixel (RMB drag)

    // Raw input state carried across frames
    double m_prevMouseX      = 0.0;
    double m_prevMouseY      = 0.0;
    bool   m_prevRmb         = false;  // right mouse button state last frame
    bool   m_viewportHovered  = false;  // ImGui hover flag (set during Viewport panel)
    int    m_selectedNodeIdx  = -1;     // Scene Graph selection; -1 = nothing selected

    void updateCamera();  // process input, rebuild scene camera matrix
};

#endif // OPTIX_RAYTRACER_APPLICATION_H
