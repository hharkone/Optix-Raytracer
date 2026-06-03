// Application.cpp — host-side application: window, CUDA/OptiX init, render loop.
//
// IMPORTANT: optix_function_table_definition.h must appear in exactly ONE
// translation unit. This file is that unit — do not include it elsewhere.
//
// Include glad before Application.h so OpenGL functions are loaded before any
// GLFW or OptiX headers try to reference them.
#include <glad/glad.h>

#include "Application.h"

#include <optix_function_table_definition.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <nfd.h>
#include "SceneLoader.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// ─── Error macros ─────────────────────────────────────────────────────────────

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t rc = (call);                                               \
        if (rc != cudaSuccess) {                                               \
            throw std::runtime_error(std::string("CUDA error in " __FILE__     \
                ":" + std::to_string(__LINE__) + " — ")                        \
                + cudaGetErrorString(rc));                                     \
        }                                                                      \
    } while (0)

#define OPTIX_CHECK(call)                                                      \
    do {                                                                       \
        OptixResult rc = (call);                                               \
        if (rc != OPTIX_SUCCESS) {                                             \
            throw std::runtime_error(std::string("OptiX error in " __FILE__    \
                ":" + std::to_string(__LINE__) + " — ")                        \
                + optixGetErrorString(rc));                                    \
        }                                                                      \
    } while (0)

// ─── Construction / Destruction ───────────────────────────────────────────────

Application::Application(int width, int height, const std::string& title,
                         const std::string& ptxDir)
    : m_width(width), m_height(height)
{
    initWindow(title);
    initOpenGL();
    initImGui();
    initCuda();
    initOptix();
    buildPipeline(ptxDir);

    // Hot-reload — remember where the PTX lives and when it was last written
    m_ptxDir = ptxDir;
    {
        std::error_code ec;
        m_ptxWriteTime = std::filesystem::last_write_time(std::filesystem::path(ptxDir) / "devicePrograms.ptx", ec);
    }

    m_scene = std::make_unique<Scene>();
    buildSbt();  // empty SBT — no meshes yet

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_launchParamsBuffer),
                           sizeof(LaunchParams)));
    NFD_Init();
}

Application::~Application()
{
    if (d_colorBuffer)
    {
        cudaFree(d_colorBuffer);
        d_colorBuffer = nullptr;
    }
    delete[] h_colorBuffer;

    if (m_launchParamsBuffer)
    {
        cudaFree(reinterpret_cast<void*>(m_launchParamsBuffer));
        m_launchParamsBuffer = 0;
    }
    if (m_sbtRaygenBuffer)
    {
        cudaFree(reinterpret_cast<void*>(m_sbtRaygenBuffer));
        m_sbtRaygenBuffer = 0;
    }
    if (m_sbtMissBuffer)
    {
        cudaFree(reinterpret_cast<void*>(m_sbtMissBuffer));
        m_sbtMissBuffer = 0;
    }
    if (m_sbtHitgroupBuffer)
    {
        cudaFree(reinterpret_cast<void*>(m_sbtHitgroupBuffer));
        m_sbtHitgroupBuffer = 0;
    }

    freeTexture(m_envMap);

    if (m_accumBuffer)
    {
        cudaFree(reinterpret_cast<void*>(m_accumBuffer));
        m_accumBuffer = 0;
    }
    if (m_materialsBuffer)
    {
        cudaFree(reinterpret_cast<void*>(m_materialsBuffer));
        m_materialsBuffer = 0;
    }

    m_accel.reset();  // free AS device memory before destroying OptiX context

    if (m_pipeline)   { optixPipelineDestroy(m_pipeline);       m_pipeline   = nullptr; }
    if (m_pgHitgroup) { optixProgramGroupDestroy(m_pgHitgroup); m_pgHitgroup = nullptr; }
    if (m_pgMiss)     { optixProgramGroupDestroy(m_pgMiss);     m_pgMiss     = nullptr; }
    if (m_pgRaygen)   { optixProgramGroupDestroy(m_pgRaygen);   m_pgRaygen   = nullptr; }
    if (m_module)     { optixModuleDestroy(m_module);           m_module     = nullptr; }

    if (m_optixContext)
    {
        optixDeviceContextDestroy(m_optixContext);
        m_optixContext = nullptr;
    }

    NFD_Quit();

    if (m_displayTexture)
    {
        glDeleteTextures(1, &m_displayTexture);
        m_displayTexture = 0;
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    if (m_window)
    {
        glfwDestroyWindow(m_window);
        m_window = nullptr;
    }
    glfwTerminate();
}

// ─── Initialisation ───────────────────────────────────────────────────────────

void Application::initWindow(const std::string& title)
{
    if (!glfwInit())
    {
        throw std::runtime_error("Failed to initialise GLFW");
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    m_window = glfwCreateWindow(m_width, m_height, title.c_str(), nullptr, nullptr);
    if (!m_window)
    {
        throw std::runtime_error("Failed to create GLFW window");
    }

    glfwMakeContextCurrent(m_window);
    glfwSwapInterval(0);  // 0 = no VSync — present immediately for maximum throughput
}

void Application::initOpenGL()
{
    if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress)))
    {
        throw std::runtime_error("Failed to initialise GLAD");
    }
}

void Application::initImGui()
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(m_window, true);
    ImGui_ImplOpenGL3_Init("#version 330 core");
}

void Application::initCuda()
{
    // Force CUDA runtime initialisation on device 0
    CUDA_CHECK(cudaFree(nullptr));

    // Query device properties for the performance stats display
    cudaDeviceProp prop = {};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    m_deviceName         = prop.name;
    m_deviceComputeMajor = prop.major;
    m_deviceComputeMinor = prop.minor;
    m_deviceMemoryMB     = static_cast<std::uint64_t>(prop.totalGlobalMem)
                           / (1024ULL * 1024ULL);
}

void Application::optixLogCallback(unsigned int level,
                                   const char*  tag,
                                   const char*  message,
                                   void* /*cbdata*/)
{
    std::cerr << "[OptiX][" << level << "][" << tag << "] " << message << '\n';
}

void Application::initOptix()
{
    OPTIX_CHECK(optixInit());

    OptixDeviceContextOptions opts = {};
    opts.logCallbackFunction       = &Application::optixLogCallback;
    opts.logCallbackLevel          = 3; // warnings and errors

    CUcontext cuCtx = 0; // 0 = use the CUDA runtime's current context
    OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &opts, &m_optixContext));
}

// ─── SBT record types ────────────────────────────────────────────────────────
// Each record = 32-byte opaque header (packed by optixSbtRecordPackHeader) + user data.
// alignas(OPTIX_SBT_RECORD_ALIGNMENT) ensures the compiler pads sizeof() to a multiple
// of the required 16-byte SBT stride.

namespace
{

struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord
{
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // no extra raygen data
};

struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord
{
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // no extra miss data
};

struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) HitGroupRecord
{
    char     header[OPTIX_SBT_RECORD_HEADER_SIZE];
    MeshData data;   // device pointers to this mesh's geometry
};

} // anonymous namespace

// ─── Pipeline ─────────────────────────────────────────────────────────────────

void Application::buildPipeline(const std::string& ptxDir)
{
    // Load PTX source from the directory next to the executable
    const std::string ptxPath =
        (std::filesystem::path(ptxDir) / "devicePrograms.ptx").string();

    std::ifstream ptxFile(ptxPath, std::ios::binary | std::ios::ate);
    if (!ptxFile)
    {
        throw std::runtime_error("Cannot open PTX file: " + ptxPath);
    }
    const std::streamsize ptxSize = ptxFile.tellg();
    ptxFile.seekg(0, std::ios::beg);
    std::string ptxSource(static_cast<size_t>(ptxSize), '\0');
    ptxFile.read(ptxSource.data(), ptxSize);

    // ── Module ────────────────────────────────────────────────────────────────
    OptixModuleCompileOptions moduleOpts = {};
    moduleOpts.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    moduleOpts.optLevel         = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleOpts.debugLevel       = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

    OptixPipelineCompileOptions pipelineOpts = {};
    pipelineOpts.usesMotionBlur                   = 0;
    pipelineOpts.traversableGraphFlags            =
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipelineOpts.numPayloadValues                 = 2;  // p0/p1 = packed PathVertex pointer
    pipelineOpts.numAttributeValues               = 2;  // barycentrics (built-in triangle)
    pipelineOpts.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineOpts.pipelineLaunchParamsVariableName = "optixLaunchParams";
    pipelineOpts.usesPrimitiveTypeFlags           =
        static_cast<unsigned int>(OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);

    OPTIX_CHECK(optixModuleCreate(
        m_optixContext,
        &moduleOpts,
        &pipelineOpts,
        ptxSource.c_str(), ptxSource.size(),
        nullptr, nullptr,
        &m_module));

    // ── Program groups ────────────────────────────────────────────────────────
    OptixProgramGroupOptions pgOpts = {};
    OptixProgramGroupDesc    pgDesc = {};

    pgDesc.kind                    = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgDesc.raygen.module            = m_module;
    pgDesc.raygen.entryFunctionName = "__raygen__renderFrame";
    OPTIX_CHECK(optixProgramGroupCreate(
        m_optixContext, &pgDesc, 1, &pgOpts, nullptr, nullptr, &m_pgRaygen));

    pgDesc                         = {};
    pgDesc.kind                    = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgDesc.miss.module              = m_module;
    pgDesc.miss.entryFunctionName   = "__miss__radiance";
    OPTIX_CHECK(optixProgramGroupCreate(
        m_optixContext, &pgDesc, 1, &pgOpts, nullptr, nullptr, &m_pgMiss));

    pgDesc                                   = {};
    pgDesc.kind                              = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pgDesc.hitgroup.moduleCH                  = m_module;
    pgDesc.hitgroup.entryFunctionNameCH       = "__closesthit__radiance";
    pgDesc.hitgroup.moduleAH                  = nullptr;
    pgDesc.hitgroup.entryFunctionNameAH       = nullptr;
    pgDesc.hitgroup.moduleIS                  = nullptr;  // built-in triangle IS
    pgDesc.hitgroup.entryFunctionNameIS       = nullptr;
    OPTIX_CHECK(optixProgramGroupCreate(
        m_optixContext, &pgDesc, 1, &pgOpts, nullptr, nullptr, &m_pgHitgroup));

    // ── Pipeline ──────────────────────────────────────────────────────────────
    const OptixProgramGroup pgs[] = { m_pgRaygen, m_pgMiss, m_pgHitgroup };

    OptixPipelineLinkOptions linkOpts = {};
    // Depth 1: the raygen calls optixTrace in a loop — no CH/miss ever calls
    // optixTrace, so the trace chain never exceeds depth 1.
    linkOpts.maxTraceDepth = 1;

    OPTIX_CHECK(optixPipelineCreate(
        m_optixContext,
        &pipelineOpts, &linkOpts,
        pgs, 3,
        nullptr, nullptr,
        &m_pipeline));

    // Stack size — 2 KB continuation stack, max traversal depth 2 (TLAS → BLAS)
    OPTIX_CHECK(optixPipelineSetStackSize(m_pipeline, 0, 0, 2048, 2));
}

// ─── Hot reload ───────────────────────────────────────────────────────────────

void Application::reloadPipeline()
{
    // Drain the GPU before touching any pipeline objects
    CUDA_CHECK(cudaDeviceSynchronize());

    // Save the current handles — we restore them if the new PTX fails to compile,
    // which keeps the last working shader running instead of going black.
    const OptixModule       oldModule     = m_module;
    const OptixProgramGroup oldPgRaygen   = m_pgRaygen;
    const OptixProgramGroup oldPgMiss     = m_pgMiss;
    const OptixProgramGroup oldPgHitgroup = m_pgHitgroup;
    const OptixPipeline     oldPipeline   = m_pipeline;

    m_module     = nullptr;
    m_pgRaygen   = nullptr;
    m_pgMiss     = nullptr;
    m_pgHitgroup = nullptr;
    m_pipeline   = nullptr;

    try
    {
        buildPipeline(m_ptxDir);
    }
    catch (...)
    {
        if (m_pipeline)   { optixPipelineDestroy(m_pipeline);       m_pipeline   = nullptr; }
        if (m_pgHitgroup) { optixProgramGroupDestroy(m_pgHitgroup); m_pgHitgroup = nullptr; }
        if (m_pgMiss)     { optixProgramGroupDestroy(m_pgMiss);     m_pgMiss     = nullptr; }
        if (m_pgRaygen)   { optixProgramGroupDestroy(m_pgRaygen);   m_pgRaygen   = nullptr; }
        if (m_module)     { optixModuleDestroy(m_module);           m_module     = nullptr; }

        m_module     = oldModule;
        m_pgRaygen   = oldPgRaygen;
        m_pgMiss     = oldPgMiss;
        m_pgHitgroup = oldPgHitgroup;
        m_pipeline   = oldPipeline;
        throw;
    }

    buildSbt();
    m_accumDirty = true;  // new shader = new result; clear accumulation

    if (oldPipeline)   { optixPipelineDestroy(oldPipeline);       }
    if (oldPgHitgroup) { optixProgramGroupDestroy(oldPgHitgroup); }
    if (oldPgMiss)     { optixProgramGroupDestroy(oldPgMiss);     }
    if (oldPgRaygen)   { optixProgramGroupDestroy(oldPgRaygen);   }
    if (oldModule)     { optixModuleDestroy(oldModule);           }
}

void Application::checkShaderHotReload()
{
    const auto ptxPath = std::filesystem::path(m_ptxDir) / "devicePrograms.ptx";

    std::error_code ec;
    const auto newWriteTime = std::filesystem::last_write_time(ptxPath, ec);
    if (ec || newWriteTime == m_ptxWriteTime)
    {
        return;
    }

    // Stamp first — prevents hammering reloadPipeline every frame if the PTX
    // stays broken (the timestamp will have moved but won't keep changing).
    m_ptxWriteTime = newWriteTime;

    try
    {
        reloadPipeline();
        m_shaderError.clear();
    }
    catch (const std::exception& e)
    {
        m_shaderError = e.what();
    }
}

// ─── Shader binding table ────────────────────────────────────────────────────

void Application::buildSbt()
{
    // Free any previously allocated SBT device buffers
    if (m_sbtRaygenBuffer)
    {
        cudaFree(reinterpret_cast<void*>(m_sbtRaygenBuffer));
        m_sbtRaygenBuffer = 0;
    }
    if (m_sbtMissBuffer)
    {
        cudaFree(reinterpret_cast<void*>(m_sbtMissBuffer));
        m_sbtMissBuffer = 0;
    }
    if (m_sbtHitgroupBuffer)
    {
        cudaFree(reinterpret_cast<void*>(m_sbtHitgroupBuffer));
        m_sbtHitgroupBuffer = 0;
    }
    m_sbt = {};

    // ── Raygen record ─────────────────────────────────────────────────────────
    RaygenRecord raygenRec = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(m_pgRaygen, &raygenRec));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_sbtRaygenBuffer),
                           sizeof(RaygenRecord)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_sbtRaygenBuffer),
                           &raygenRec, sizeof(RaygenRecord), cudaMemcpyHostToDevice));

    // ── Miss record ───────────────────────────────────────────────────────────
    MissRecord missRec = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(m_pgMiss, &missRec));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_sbtMissBuffer), sizeof(MissRecord)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_sbtMissBuffer),
                           &missRec, sizeof(MissRecord), cudaMemcpyHostToDevice));

    // ── Hit group records — one per mesh ──────────────────────────────────────
    const auto& meshes = m_scene->meshes();
    std::vector<HitGroupRecord> hitRecs(meshes.size());

    for (size_t i = 0; i < meshes.size(); ++i)
    {
        OPTIX_CHECK(optixSbtRecordPackHeader(m_pgHitgroup, &hitRecs[i]));

        if (m_accel && m_accel->valid())
        {
            const auto ptrs            = m_accel->meshDevicePtrs(i);
            hitRecs[i].data.positions  = reinterpret_cast<const float3*>(ptrs.positions);
            hitRecs[i].data.normals    = reinterpret_cast<const float3*>(ptrs.normals);
            hitRecs[i].data.indices    = reinterpret_cast<const uint3*>(ptrs.indices);
            hitRecs[i].data.uvs        = nullptr;
            hitRecs[i].data.materialIndex = meshes[i].materialIndex;
        }
    }

    if (!hitRecs.empty())
    {
        const size_t hitByteSize = hitRecs.size() * sizeof(HitGroupRecord);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_sbtHitgroupBuffer), hitByteSize));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_sbtHitgroupBuffer),
                               hitRecs.data(), hitByteSize, cudaMemcpyHostToDevice));
    }

    // ── Fill the SBT descriptor ───────────────────────────────────────────────
    m_sbt.raygenRecord                = m_sbtRaygenBuffer;

    m_sbt.missRecordBase              = m_sbtMissBuffer;
    m_sbt.missRecordStrideInBytes     = sizeof(MissRecord);
    m_sbt.missRecordCount             = 1;

    m_sbt.hitgroupRecordBase          = m_sbtHitgroupBuffer;
    m_sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupRecord);
    m_sbt.hitgroupRecordCount         = static_cast<unsigned int>(hitRecs.size());
}

// ─── Framebuffer ─────────────────────────────────────────────────────────────

void Application::resizeFramebuffer(int w, int h)
{
    if (d_colorBuffer)
    {
        cudaFree(d_colorBuffer);
        d_colorBuffer = nullptr;
    }
    delete[] h_colorBuffer;
    h_colorBuffer = nullptr;

    if (m_displayTexture)
    {
        glDeleteTextures(1, &m_displayTexture);
        m_displayTexture = 0;
    }

    m_viewportWidth  = w;
    m_viewportHeight = h;

    const size_t pixelCount = static_cast<size_t>(w) * h;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_colorBuffer), pixelCount * sizeof(uchar4)));
    h_colorBuffer = new uchar4[pixelCount];

    // Accumulation buffer: float4 per pixel (w component unused)
    if (m_accumBuffer)
    {
        cudaFree(reinterpret_cast<void*>(m_accumBuffer));
        m_accumBuffer = 0;
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_accumBuffer), pixelCount * sizeof(float4)));
    m_accumDirty = true;

    glGenTextures(1, &m_displayTexture);
    glBindTexture(GL_TEXTURE_2D, m_displayTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);
}

// ─── Material upload ─────────────────────────────────────────────────────────

void Application::uploadMaterials()
{
    const auto& mats = m_scene->materials();
    if (mats.empty())
    {
        return;
    }
    const size_t matBytes = mats.size() * sizeof(MaterialData);

    // Allocate on first call or if the buffer is gone; otherwise reuse.
    if (!m_materialsBuffer)
    {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_materialsBuffer), matBytes));
    }
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_materialsBuffer),
                           mats.data(), matBytes, cudaMemcpyHostToDevice));
}

// ─── Scene loading ────────────────────────────────────────────────────────────

void Application::loadScene(const std::string& path)
{
    m_scene->clear();
    m_accel.reset();
    m_loadError.clear();
    m_selectedNodeIdx = -1;
    m_sceneFilePath.clear();

    if (loadGltfFile(path, *m_scene, m_loadError))
    {
        m_sceneFilePath = path;

        if (!m_scene->empty())
        {
            try
            {
                m_accel = std::make_unique<Accel>();
                m_accel->build(m_optixContext, *m_scene);
            }
            catch (const std::exception& e)
            {
                m_loadError = std::string("AS build failed: ") + e.what();
                m_accel.reset();
            }
        }
    }
    else
    {
        m_scene->clear();  // discard any partial data from a failed load
    }

    // Upload materials to device so the closest-hit shader can look up properties.
    if (m_materialsBuffer)
    {
        cudaFree(reinterpret_cast<void*>(m_materialsBuffer));
        m_materialsBuffer = 0;
    }
    uploadMaterials();

    m_accumDirty = true;  // scene changed — clear accumulated samples
    buildSbt();  // rebuild with new mesh count (0 if load failed or no geometry)

    // Sync fly-camera state from the newly loaded (or default) scene camera so
    // movement immediately continues from the correct position and orientation.
    {
        const Camera& cam = m_scene->camera();
        m_camPos.x = cam.transform.m[0][3];
        m_camPos.y = cam.transform.m[1][3];
        m_camPos.z = cam.transform.m[2][3];

        // Forward = -column2 of the camera-to-world matrix
        const float fx = -cam.transform.m[0][2];
        const float fy = -cam.transform.m[1][2];
        const float fz = -cam.transform.m[2][2];
        const float fLen = std::max(1e-6f, sqrtf(fx*fx + fy*fy + fz*fz));
        m_camPitch = asinf(std::max(-1.0f, std::min(1.0f, fy / fLen)));
        m_camYaw   = atan2f(fx / fLen, -(fz / fLen));
    }
}

// ─── Environment map ─────────────────────────────────────────────────────────

void Application::loadEnvMap(const std::string& path)
{
    m_envMapError.clear();
    freeTexture(m_envMap);
    m_envMapPath.clear();

    if (loadEXR(path, m_envMap, m_envMapError))
    {
        uploadToGpu(m_envMap);
        m_envMapPath = std::filesystem::path(path).filename().string();
        m_accumDirty = true;  // new env map = new lighting; clear accumulated samples
    }
}

// ─── Camera controller ───────────────────────────────────────────────────────

void Application::updateCamera()
{
    ImGuiIO& io = ImGui::GetIO();

    // ── Mouse delta ───────────────────────────────────────────────────────────
    double mouseX, mouseY;
    glfwGetCursorPos(m_window, &mouseX, &mouseY);
    const float dx = static_cast<float>(mouseX - m_prevMouseX);
    const float dy = static_cast<float>(mouseY - m_prevMouseY);
    m_prevMouseX = mouseX;
    m_prevMouseY = mouseY;

    const bool rmb          = glfwGetMouseButton(m_window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
    const bool rmbFirstFrame = rmb && !m_prevRmb;  // true only on the press event
    m_prevRmb = rmb;

    // ── Rotation / Orbit — right-drag while the Viewport panel is under the cursor
    // Ctrl+RMB: orbit the camera position around the world origin at a fixed radius.
    //           Camera orientation (yaw/pitch) is left unchanged — no snap.
    // Plain RMB: free-look (rotate orientation in place, position fixed).
    // rmbFirstFrame is skipped to avoid a position-jump on the first drag frame.
    if (rmb && !rmbFirstFrame && m_viewportHovered)
    {
        const bool ctrlHeld = glfwGetKey(m_window, GLFW_KEY_LEFT_CONTROL)  == GLFW_PRESS
                           || glfwGetKey(m_window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS;

        if (ctrlHeld)
        {
            // Orbit: move the position along a sphere of the same radius and
            // derive yaw/pitch from the new position so the camera always looks
            // at the origin.  Rotation is driven by the mouse delta on the
            // position — not imposed directly on angles — so there is no snap.
            const float r = sqrtf(m_camPos.x*m_camPos.x
                                + m_camPos.y*m_camPos.y
                                + m_camPos.z*m_camPos.z);
            if (r > 1e-4f)
            {
                float azimuth   = atan2f(m_camPos.x, m_camPos.z);
                float elevation = asinf(std::max(-1.0f, std::min(1.0f, m_camPos.y / r)));

                azimuth   -= dx * m_rotSpeed;
                elevation += dy * m_rotSpeed;

                const float kPoleLimit = 1.5533430f;  // 89°
                elevation = std::max(-kPoleLimit, std::min(kPoleLimit, elevation));

                m_camPos.x = r * cosf(elevation) * sinf(azimuth);
                m_camPos.y = r * sinf(elevation);
                m_camPos.z = r * cosf(elevation) * cosf(azimuth);

                // Apply the equivalent delta to yaw/pitch (yaw = -azimuth,
                // pitch = -elevation) so they track the orbit continuously
                // without snapping to an absolute value.
                m_camYaw   += dx * m_rotSpeed;
                m_camPitch -= dy * m_rotSpeed;
                m_camPitch  = std::max(-kPoleLimit, std::min(kPoleLimit, m_camPitch));
            }
        }
        else
        {
            // Free-look: update view direction, position stays fixed.
            m_camYaw   += dx * m_rotSpeed;
            m_camPitch -= dy * m_rotSpeed;

            // Clamp pitch to just under ±90° to avoid gimbal singularity
            const float kPitchLimit = 1.5533430f;  // 89 degrees in radians
            m_camPitch = std::max(-kPitchLimit, std::min(kPitchLimit, m_camPitch));
        }
    }

    // ── Translation — WASD in camera space ───────────────────────────────────
    if (!io.WantCaptureKeyboard)
    {
        // dt clamped so an initial stall frame doesn't teleport the camera
        const float dt   = std::max(0.001f, std::min(m_frameTimeMs * 0.001f, 0.1f));
        const float dist = m_moveSpeed * dt;

        const float sy = sinf(m_camYaw),   cy = cosf(m_camYaw);
        const float sp = sinf(m_camPitch),  cp = cosf(m_camPitch);

        // forward = direction the camera looks, right = camera's local +X, up = camera's local +Y
        const float3 forward = {  sy*cp,  sp, -cy*cp };
        const float3 right   = {  cy,    0.0f,  sy    };
        const float3 up      = { -sy*sp,  cp,  cy*sp };

        const bool wKey = glfwGetKey(m_window, GLFW_KEY_W) == GLFW_PRESS;
        const bool sKey = glfwGetKey(m_window, GLFW_KEY_S) == GLFW_PRESS;
        const bool aKey = glfwGetKey(m_window, GLFW_KEY_A) == GLFW_PRESS;
        const bool dKey = glfwGetKey(m_window, GLFW_KEY_D) == GLFW_PRESS;
        const bool eKey = glfwGetKey(m_window, GLFW_KEY_E) == GLFW_PRESS;
        const bool qKey = glfwGetKey(m_window, GLFW_KEY_Q) == GLFW_PRESS;

        if (wKey || sKey)
        {
            const float fwd = wKey ? dist : -dist;
            m_camPos.x += forward.x * fwd;
            m_camPos.y += forward.y * fwd;
            m_camPos.z += forward.z * fwd;
        }
        if (aKey || dKey)
        {
            const float strafe = dKey ? dist : -dist;
            m_camPos.x += right.x * strafe;
            m_camPos.z += right.z * strafe;
        }
        if (eKey || qKey)
        {
            const float lift = eKey ? dist : -dist;
            m_camPos.x += up.x * lift;
            m_camPos.y += up.y * lift;
            m_camPos.z += up.z * lift;
        }
    }

    // ── Rebuild camera-to-world matrix ────────────────────────────────────────
    // Row-major Matrix4x4, columns are world-space camera axes:
    //   col 0 = right   = {cy,      0,     sy    }
    //   col 1 = up      = {-sy*sp,  cp,    cy*sp }
    //   col 2 = +Z cam  = {-sy*cp, -sp,    cy*cp } (camera looks down -Z)
    //   col 3 = pos
    const float sy = sinf(m_camYaw),   cy = cosf(m_camYaw);
    const float sp = sinf(m_camPitch),  cp = cosf(m_camPitch);

    Camera cam = m_scene->camera();
    cam.transform.m[0][0] =  cy;     cam.transform.m[0][1] = -sy*sp;  cam.transform.m[0][2] = -sy*cp;  cam.transform.m[0][3] = m_camPos.x;
    cam.transform.m[1][0] = 0.0f;     cam.transform.m[1][1] =  cp;     cam.transform.m[1][2] = -sp;     cam.transform.m[1][3] = m_camPos.y;
    cam.transform.m[2][0] =  sy;     cam.transform.m[2][1] =  cy*sp;  cam.transform.m[2][2] =  cy*cp;  cam.transform.m[2][3] = m_camPos.z;
    cam.transform.m[3][0] = 0.0f;     cam.transform.m[3][1] = 0.0f;     cam.transform.m[3][2] = 0.0f;     cam.transform.m[3][3] = 1.0f;
    m_scene->setCamera(std::move(cam));
}

// ─── Scene Graph helpers ──────────────────────────────────────────────────────

static void drawNode3D(const Scene& scene, int nodeIdx, int& selectedNodeIdx)
{
    const Node3D& node = *scene.nodes()[nodeIdx];

    std::string label = node.name.empty()
        ? (std::string(node.typeName()) + " " + std::to_string(nodeIdx))
        : node.name;
    label += std::string("  <") + node.typeName() + ">";

    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow
                             | ImGuiTreeNodeFlags_SpanAvailWidth;
    if (nodeIdx == selectedNodeIdx)
    {
        flags |= ImGuiTreeNodeFlags_Selected;
    }
    if (node.children.empty())
    {
        flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
    }

    ImGui::PushID(nodeIdx);
    const bool open = ImGui::TreeNodeEx(label.c_str(), flags);

    if (ImGui::IsItemClicked())
    {
        selectedNodeIdx = nodeIdx;
    }

    if (open && !node.children.empty())
    {
        for (int childIdx : node.children)
        {
            drawNode3D(scene, childIdx, selectedNodeIdx);
        }
        ImGui::TreePop();
    }

    ImGui::PopID();
}

// ─── Per-frame ────────────────────────────────────────────────────────────────

bool Application::tick()
{
    if (glfwWindowShouldClose(m_window))
    {
        return false;
    }

    // ── Shader hot-reload ─────────────────────────────────────────────────────
    checkShaderHotReload();

    // ── Camera-change detection ────────────────────────────────────────────────
    // Save camera state before updateCamera() so we can detect any change.
    const float3 prevPos   = m_camPos;
    const float  prevYaw   = m_camYaw;
    const float  prevPitch = m_camPitch;

    // ── Frame timing ──────────────────────────────────────────────────────────
    {
        const auto now = std::chrono::steady_clock::now();
        if (m_frameTimeMs > 0.0f)
        {
            const float deltaMs = std::chrono::duration<float, std::milli>(
                now - m_frameStart).count();
            // Exponential moving average — α=0.1 keeps the display readable
            m_frameTimeMs = 0.1f * deltaMs + 0.9f * m_frameTimeMs;
        }
        else if (m_frameStart != std::chrono::steady_clock::time_point{})
        {
            // Second frame: initialise with the first real measurement
            m_frameTimeMs = std::chrono::duration<float, std::milli>(
                now - m_frameStart).count();
        }
        m_frameStart = now;
    }

    glfwPollEvents();

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // Camera input is processed here — after NewFrame() so WantCaptureMouse /
    // WantCaptureKeyboard are current, but before the GPU launch so this frame
    // renders with the updated camera.
    updateCamera();

    if (m_camPos.x != prevPos.x || m_camPos.y != prevPos.y || m_camPos.z != prevPos.z ||
        m_camYaw   != prevYaw   || m_camPitch  != prevPitch)
    {
        m_accumDirty = true;
    }

    // Enable a full-window dockspace so panels can be docked to it
    ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport(),
                                 ImGuiDockNodeFlags_PassthruCentralNode);

    // ── Viewport panel ────────────────────────────────────────────────────────
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::Begin("Viewport", nullptr,
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    ImGui::PopStyleVar();

    // Record hover state for use in updateCamera() next frame
    m_viewportHovered = ImGui::IsWindowHovered();

    {
        const ImVec2 regionSize = ImGui::GetContentRegionAvail();
        const int vpW = std::max(1, static_cast<int>(regionSize.x));
        const int vpH = std::max(1, static_cast<int>(regionSize.y));

        if (vpW != m_viewportWidth || vpH != m_viewportHeight)
        {
            resizeFramebuffer(vpW, vpH);
        }

        // ── Update launch parameters ──────────────────────────────────────────
        // ── Reset accumulation buffer if anything changed ─────────────────────
        if (m_accumDirty && m_accumBuffer)
        {
            CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(m_accumBuffer), 0,
                static_cast<size_t>(m_viewportWidth) * m_viewportHeight * sizeof(float4)));
            m_sampleCount = 0;
            m_accumDirty  = false;
        }

        m_launchParams.colorBuffer   = d_colorBuffer;
        m_launchParams.fbSize        = make_uint2(
            static_cast<unsigned int>(m_viewportWidth),
            static_cast<unsigned int>(m_viewportHeight));
        m_launchParams.traversable   =
            (m_accel && m_accel->valid()) ? m_accel->traversable() : 0;
        m_launchParams.envMap        = m_envMap.gpuTex;
        m_launchParams.accumBuffer   = m_accumBuffer
            ? reinterpret_cast<float4*>(m_accumBuffer) : nullptr;
        m_launchParams.sampleIndex   = m_sampleCount;
        m_launchParams.materials     = m_materialsBuffer
            ? reinterpret_cast<const MaterialData*>(m_materialsBuffer) : nullptr;

        // Camera basis vectors derived from the scene camera each frame
        {
            const Camera&    cam = m_scene->camera();
            const Matrix4x4& t   = cam.transform;

            // Camera-to-world columns: right=col0, up=col1, +Z=col2, eye=col3
            m_launchParams.eye =
                make_float3(t.m[0][3], t.m[1][3], t.m[2][3]);
            const float3 right   =
                make_float3(t.m[0][0], t.m[1][0], t.m[2][0]);
            const float3 up      =
                make_float3(t.m[0][1], t.m[1][1], t.m[2][1]);
            // Camera looks down -Z, so forward = -column2
            const float3 forward =
                make_float3(-t.m[0][2], -t.m[1][2], -t.m[2][2]);

            const float tanHalfFovV =
                std::tan(cam.yFov * 0.5f);
            const float tanHalfFovH =
                tanHalfFovV * (static_cast<float>(m_viewportWidth)
                               / static_cast<float>(m_viewportHeight));

            m_launchParams.U =
                make_float3(right.x * tanHalfFovH,
                            right.y * tanHalfFovH,
                            right.z * tanHalfFovH);
            m_launchParams.V =
                make_float3(up.x * tanHalfFovV,
                            up.y * tanHalfFovV,
                            up.z * tanHalfFovV);
            m_launchParams.W = forward;
        }

        // ── GPU launch ────────────────────────────────────────────────────────
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(m_launchParamsBuffer),
            &m_launchParams, sizeof(LaunchParams),
            cudaMemcpyHostToDevice));

        OPTIX_CHECK(optixLaunch(
            m_pipeline, nullptr,
            m_launchParamsBuffer, sizeof(LaunchParams),
            &m_sbt,
            static_cast<unsigned int>(m_viewportWidth),
            static_cast<unsigned int>(m_viewportHeight),
            1));

        CUDA_CHECK(cudaDeviceSynchronize());
        ++m_sampleCount;

        // Copy rendered result from device to host, then upload to GL texture
        CUDA_CHECK(cudaMemcpy(
            h_colorBuffer, d_colorBuffer,
            static_cast<size_t>(m_viewportWidth) * m_viewportHeight * sizeof(uchar4),
            cudaMemcpyDeviceToHost));

        glBindTexture(GL_TEXTURE_2D, m_displayTexture);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
            m_viewportWidth, m_viewportHeight,
            GL_RGBA, GL_UNSIGNED_BYTE, h_colorBuffer);
        glBindTexture(GL_TEXTURE_2D, 0);

        // UV0=(0,1) UV1=(1,0): flip vertically so OptiX row-0=bottom matches
        // OpenGL's bottom-up texture convention when displayed top-down by ImGui.
        ImGui::Image(
            (ImTextureID)(intptr_t)m_displayTexture,
            ImVec2(static_cast<float>(m_viewportWidth),
                   static_cast<float>(m_viewportHeight)),
            ImVec2(0.0f, 1.0f),
            ImVec2(1.0f, 0.0f));
    }

    ImGui::End();

    // ── Raytracer panel ───────────────────────────────────────────────────────
    ImGui::Begin("Raytracer");

    // ── Performance stats ─────────────────────────────────────────────────────
    ImGui::Text("GPU: %s", m_deviceName.c_str());
    ImGui::Text("     SM %d.%d  |  %llu MB VRAM",
        m_deviceComputeMajor, m_deviceComputeMinor,
        static_cast<unsigned long long>(m_deviceMemoryMB));
    ImGui::Text("Resolution: %d x %d", m_viewportWidth, m_viewportHeight);

    if (m_frameTimeMs > 0.0f)
    {
        const float    fps      = 1000.0f / m_frameTimeMs;
        const double   raysPerS = static_cast<double>(m_viewportWidth)
                                * static_cast<double>(m_viewportHeight) * fps;

        ImGui::Text("Frame: %.2f ms  (%.0f fps)", m_frameTimeMs, fps);

        if (raysPerS >= 1.0e9)
        {
            ImGui::Text("Rays: %.2f Grays/s", raysPerS * 1.0e-9);
        }
        else if (raysPerS >= 1.0e6)
        {
            ImGui::Text("Rays: %.2f Mrays/s", raysPerS * 1.0e-6);
        }
        else
        {
            ImGui::Text("Rays: %.2f Krays/s", raysPerS * 1.0e-3);
        }
    }
    else
    {
        ImGui::TextDisabled("Frame: --");
    }

    ImGui::Separator();

    if (ImGui::Button("Open glTF..."))
    {
        nfdu8char_t*    outPath = nullptr;
        nfdfilteritem_t filters[] = { { "glTF Scene", "gltf,glb" } };
        if (NFD_OpenDialogU8(&outPath, filters, 1, nullptr) == NFD_OKAY)
        {
            loadScene(reinterpret_cast<const char*>(outPath));
            NFD_FreePathU8(outPath);
        }
    }

    ImGui::SameLine();
    if (ImGui::Button("Open EXR..."))
    {
        nfdu8char_t*    outPath = nullptr;
        nfdfilteritem_t filters[] = { { "EXR Image", "exr" } };
        if (NFD_OpenDialogU8(&outPath, filters, 1, nullptr) == NFD_OKAY)
        {
            loadEnvMap(reinterpret_cast<const char*>(outPath));
            NFD_FreePathU8(outPath);
        }
    }
    if (m_envMap.gpuTex != 0)
    {
        ImGui::SameLine();
        if (ImGui::Button("Clear EXR"))
        {
            freeTexture(m_envMap);
            m_envMapPath.clear();
            m_envMapError.clear();
        }
    }

    if (!m_sceneFilePath.empty())
    {
        const std::string filename =
            std::filesystem::path(m_sceneFilePath).filename().string();
        ImGui::Text("Scene: %s", filename.c_str());
        if (ImGui::IsItemHovered())
        {
            ImGui::SetTooltip("%s", m_sceneFilePath.c_str());
        }
    }
    else
    {
        ImGui::TextDisabled("No scene loaded");
    }

    if (!m_loadError.empty())
    {
        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f),
                           "Error: %s", m_loadError.c_str());
    }

    if (!m_envMapPath.empty())
    {
        ImGui::Text("Env: %s  (%d x %d)", m_envMapPath.c_str(),
                    m_envMap.width, m_envMap.height);
    }
    else if (!m_envMapError.empty())
    {
        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f),
                           "EXR error: %s", m_envMapError.c_str());
    }
    else
    {
        ImGui::TextDisabled("No environment map");
    }

    ImGui::Separator();
    ImGui::Text("Meshes: %d  Materials: %d  Textures: %d",
        static_cast<int>(m_scene->meshes().size()),
        static_cast<int>(m_scene->materials().size()),
        static_cast<int>(m_scene->textures().size()));

    ImGui::Separator();
    const Camera& cam = m_scene->camera();
    ImGui::Text("Camera: %s", cam.name.c_str());
    ImGui::Text("Position: (%.2f, %.2f, %.2f)",
        cam.transform.m[0][3],
        cam.transform.m[1][3],
        cam.transform.m[2][3]);
    ImGui::Text("FOV: %.1f deg", cam.yFov * (180.0f / 3.14159265f));

    ImGui::Separator();
    if (m_accel && m_accel->valid())
    {
        ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.4f, 1.0f), "AS: ready");
    }
    else if (!m_scene->empty())
    {
        ImGui::TextDisabled("AS: build failed");
    }
    else
    {
        ImGui::TextDisabled("AS: no geometry");
    }

    // ── Path tracing progress ─────────────────────────────────────────────────
    ImGui::Separator();
    ImGui::Text("Samples: %u", m_sampleCount);
    if (ImGui::Button("Reset Accumulation"))
    {
        m_accumDirty = true;
    }

    ImGui::Separator();
    if (m_shaderError.empty())
    {
        ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.4f, 1.0f), "Shader: OK");
        ImGui::TextDisabled("(auto-reloads on PTX change)");
    }
    else
    {
        ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.2f, 1.0f), "Shader error — last good pipeline active");
        ImGui::TextWrapped("%s", m_shaderError.c_str());
    }

    ImGui::End();

    // ── Materials panel ───────────────────────────────────────────────────────
    ImGui::Begin("Materials");

    auto& mats = m_scene->materials();
    bool anyMatChanged = false;

    if (mats.empty())
    {
        ImGui::TextDisabled("No materials loaded");
    }
    else
    {
        for (int i = 0; i < static_cast<int>(mats.size()); ++i)
        {
            ImGui::PushID(i);

            const std::string& rawName = m_scene->materialName(i);
            const std::string  header  = rawName.empty()
                                       ? ("Material " + std::to_string(i))
                                       : rawName;

            if (ImGui::CollapsingHeader(header.c_str()))
            {
                if (ImGui::ColorEdit3("Albedo",   &mats[i].albedo.x))
                {
                    anyMatChanged = true;
                }
                if (ImGui::SliderFloat("Roughness", &mats[i].roughness, 0.0f, 1.0f))
                {
                    anyMatChanged = true;
                }
                if (ImGui::SliderFloat("Metallic",  &mats[i].metallic,  0.0f, 1.0f))
                {
                    anyMatChanged = true;
                }
                if (ImGui::ColorEdit3("Emission",  &mats[i].emission.x))
                {
                    anyMatChanged = true;
                }
                if (ImGui::DragFloat("Emission Scale", &mats[i].emissionScale,
                                     0.1f, 0.0f, 1000.0f, "%.2f"))
                {
                    anyMatChanged = true;
                }
                if (ImGui::SliderFloat("Transmission", &mats[i].transmission, 0.0f, 1.0f, "%.3f"))
                {
                    anyMatChanged = true;
                }
                if (ImGui::SliderFloat("IOR",          &mats[i].ior,          1.0f, 3.0f, "%.3f"))
                {
                    anyMatChanged = true;
                }
            }

            ImGui::PopID();
        }
    }

    if (anyMatChanged)
    {
        uploadMaterials();
        m_accumDirty = true;
    }

    ImGui::End();

    // ── Scene Graph panel ─────────────────────────────────────────────────────
    ImGui::Begin("Scene Graph");

    if (m_scene->rootNodes().empty())
    {
        ImGui::TextDisabled("No scene loaded");
    }
    else
    {
        if (!m_sceneFilePath.empty())
        {
            ImGui::TextDisabled("%s",
                std::filesystem::path(m_sceneFilePath).filename().string().c_str());
            ImGui::Separator();
        }
        for (int rootIdx : m_scene->rootNodes())
        {
            drawNode3D(*m_scene, rootIdx, m_selectedNodeIdx);
        }
    }

    ImGui::End();

    // ── Node Properties panel ─────────────────────────────────────────────────
    ImGui::Begin("Node Properties");

    if (m_selectedNodeIdx < 0 || m_selectedNodeIdx >= static_cast<int>(m_scene->nodes().size()))
    {
        ImGui::TextDisabled("No node selected");
    }
    else
    {
        Node3D& node = m_scene->nodeAt(m_selectedNodeIdx);

        // ── Identity ──────────────────────────────────────────────────────────
        ImGui::Text("Type: %s", node.typeName());
        ImGui::Text("Name: %s", node.name.empty() ? "(unnamed)" : node.name.c_str());
        ImGui::Separator();

        // ── Local transform (4×4 matrix, row by row) ──────────────────────────
        if (ImGui::CollapsingHeader("Local Transform"))
        {
            ImGui::PushItemWidth(-1.0f);
            for (int row = 0; row < 4; ++row)
            {
                ImGui::PushID(row);
                ImGui::DragFloat4("", node.localTransform.m[row], 0.01f);
                ImGui::PopID();
            }
            ImGui::PopItemWidth();
        }

        // ── Type-specific content ─────────────────────────────────────────────
        if (MeshNode* meshNode = dynamic_cast<MeshNode*>(&node))
        {
            ImGui::Separator();
            ImGui::Text("Meshes: %d primitive(s)", static_cast<int>(meshNode->meshIndices.size()));

            auto& mats      = m_scene->materials();
            bool  anyChanged = false;

            for (int meshIdx : meshNode->meshIndices)
            {
                const int matIdx = m_scene->meshes()[meshIdx].materialIndex;
                if (matIdx < 0 || matIdx >= static_cast<int>(mats.size()))
                {
                    continue;
                }

                const std::string& matName = m_scene->materialName(matIdx);
                const std::string  header  = matName.empty()
                    ? ("Material " + std::to_string(matIdx))
                    : matName;

                ImGui::PushID(meshIdx);
                if (ImGui::CollapsingHeader(header.c_str()))
                {
                    if (ImGui::ColorEdit3("Albedo",   &mats[matIdx].albedo.x))
                    {
                        anyChanged = true;
                    }
                    if (ImGui::SliderFloat("Roughness",    &mats[matIdx].roughness,     0.0f, 1.0f))
                    {
                        anyChanged = true;
                    }
                    if (ImGui::SliderFloat("Metallic",     &mats[matIdx].metallic,      0.0f, 1.0f))
                    {
                        anyChanged = true;
                    }
                    if (ImGui::ColorEdit3("Emission",  &mats[matIdx].emission.x))
                    {
                        anyChanged = true;
                    }
                    if (ImGui::DragFloat("Emission Scale", &mats[matIdx].emissionScale,
                                         0.1f, 0.0f, 1000.0f, "%.2f"))
                    {
                        anyChanged = true;
                    }
                    if (ImGui::SliderFloat("Transmission", &mats[matIdx].transmission, 0.0f, 1.0f, "%.3f"))
                    {
                        anyChanged = true;
                    }
                    if (ImGui::SliderFloat("IOR",          &mats[matIdx].ior,          1.0f, 3.0f, "%.3f"))
                    {
                        anyChanged = true;
                    }
                }
                ImGui::PopID();
            }

            if (anyChanged)
            {
                uploadMaterials();
                m_accumDirty = true;
            }
        }
        else if (dynamic_cast<CameraNode*>(&node))
        {
            ImGui::Separator();
            const Camera& cam = m_scene->camera();
            ImGui::Text("FOV:        %.1f deg", cam.yFov * (180.0f / 3.14159265f));
            ImGui::Text("Aspect:     %.3f",     cam.aspectRatio);
            ImGui::Text("Near / Far: %.4f / %.1f", cam.zNear, cam.zFar);
            ImGui::TextDisabled("(camera driven by fly-cam controller)");
        }
    }

    ImGui::End();

    ImGui::Render();

    int w, h;
    glfwGetFramebufferSize(m_window, &w, &h);
    glViewport(0, 0, w, h);
    glClearColor(0.10f, 0.10f, 0.15f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    // Multi-viewport support (ImGuiConfigFlags_ViewportsEnable)
    if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
    {
        GLFWwindow* backup = glfwGetCurrentContext();
        ImGui::UpdatePlatformWindows();
        ImGui::RenderPlatformWindowsDefault();
        glfwMakeContextCurrent(backup);
    }

    glfwSwapBuffers(m_window);
    return true;
}
