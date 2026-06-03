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
    initDenoiser();
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

    freeTexture(m_envMap);  // also frees CDF buffers via Texture::cdfMarginal/cdfConditional

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

    // ── Denoiser resources ────────────────────────────────────────────────────
    if (m_denoiser)
    {
        optixDenoiserDestroy(m_denoiser);
        m_denoiser = nullptr;
    }
    if (m_denoiserIntensity) { cudaFree(reinterpret_cast<void*>(m_denoiserIntensity)); m_denoiserIntensity = 0; }
    if (m_normalBuffer)      { cudaFree(reinterpret_cast<void*>(m_normalBuffer));      m_normalBuffer      = 0; }
    if (m_albedoBuffer)      { cudaFree(reinterpret_cast<void*>(m_albedoBuffer));      m_albedoBuffer      = 0; }
    if (m_hdrBuffer)         { cudaFree(reinterpret_cast<void*>(m_hdrBuffer));         m_hdrBuffer         = 0; }
    if (m_denoisedBuffer)    { cudaFree(reinterpret_cast<void*>(m_denoisedBuffer));    m_denoisedBuffer    = 0; }
    if (m_denoiserState)     { cudaFree(reinterpret_cast<void*>(m_denoiserState));     m_denoiserState     = 0; }
    if (m_denoiserScratch)   { cudaFree(reinterpret_cast<void*>(m_denoiserScratch));   m_denoiserScratch   = 0; }
    delete[] h_hdrBuffer;    h_hdrBuffer = nullptr;

    if (m_pipeline)      { optixPipelineDestroy(m_pipeline);           m_pipeline      = nullptr; }
    if (m_pgHitgroup)   { optixProgramGroupDestroy(m_pgHitgroup);    m_pgHitgroup    = nullptr; }
    if (m_pgMissShadow) { optixProgramGroupDestroy(m_pgMissShadow);  m_pgMissShadow  = nullptr; }
    if (m_pgMiss)       { optixProgramGroupDestroy(m_pgMiss);        m_pgMiss        = nullptr; }
    if (m_pgRaygen)     { optixProgramGroupDestroy(m_pgRaygen);      m_pgRaygen      = nullptr; }
    if (m_module)       { optixModuleDestroy(m_module);              m_module        = nullptr; }

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

void Application::initDenoiser()
{
    OptixDenoiserOptions denoiserOpts = {};
    denoiserOpts.guideNormal = 1;
    denoiserOpts.guideAlbedo = 1;
    OPTIX_CHECK(optixDenoiserCreate(
        m_optixContext,
        OPTIX_DENOISER_MODEL_KIND_HDR,
        &denoiserOpts,
        &m_denoiser));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_denoiserIntensity), sizeof(float)));
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

    pgDesc                         = {};
    pgDesc.kind                    = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgDesc.miss.module              = m_module;
    pgDesc.miss.entryFunctionName   = "__miss__shadow";
    OPTIX_CHECK(optixProgramGroupCreate(
        m_optixContext, &pgDesc, 1, &pgOpts, nullptr, nullptr, &m_pgMissShadow));

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
    const OptixProgramGroup pgs[] = {
        m_pgRaygen, m_pgMiss, m_pgMissShadow, m_pgHitgroup };

    OptixPipelineLinkOptions linkOpts = {};
    // Depth 1: path rays and NEE shadow rays are both called from raygen —
    // no CH/miss ever calls optixTrace, so the chain never exceeds depth 1.
    linkOpts.maxTraceDepth = 1;

    OPTIX_CHECK(optixPipelineCreate(
        m_optixContext,
        &pipelineOpts, &linkOpts,
        pgs, 4,
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
    const OptixModule       oldModule        = m_module;
    const OptixProgramGroup oldPgRaygen      = m_pgRaygen;
    const OptixProgramGroup oldPgMiss        = m_pgMiss;
    const OptixProgramGroup oldPgMissShadow  = m_pgMissShadow;
    const OptixProgramGroup oldPgHitgroup    = m_pgHitgroup;
    const OptixPipeline     oldPipeline      = m_pipeline;

    m_module        = nullptr;
    m_pgRaygen      = nullptr;
    m_pgMiss        = nullptr;
    m_pgMissShadow  = nullptr;
    m_pgHitgroup    = nullptr;
    m_pipeline      = nullptr;

    try
    {
        buildPipeline(m_ptxDir);
    }
    catch (...)
    {
        if (m_pipeline)      { optixPipelineDestroy(m_pipeline);          m_pipeline      = nullptr; }
        if (m_pgHitgroup)    { optixProgramGroupDestroy(m_pgHitgroup);    m_pgHitgroup    = nullptr; }
        if (m_pgMissShadow)  { optixProgramGroupDestroy(m_pgMissShadow);  m_pgMissShadow  = nullptr; }
        if (m_pgMiss)        { optixProgramGroupDestroy(m_pgMiss);        m_pgMiss        = nullptr; }
        if (m_pgRaygen)      { optixProgramGroupDestroy(m_pgRaygen);      m_pgRaygen      = nullptr; }
        if (m_module)        { optixModuleDestroy(m_module);              m_module        = nullptr; }

        m_module        = oldModule;
        m_pgRaygen      = oldPgRaygen;
        m_pgMiss        = oldPgMiss;
        m_pgMissShadow  = oldPgMissShadow;
        m_pgHitgroup    = oldPgHitgroup;
        m_pipeline      = oldPipeline;
        throw;
    }

    buildSbt();
    m_accumDirty = true;  // new shader = new result; clear accumulation

    if (oldPipeline)      { optixPipelineDestroy(oldPipeline);          }
    if (oldPgHitgroup)    { optixProgramGroupDestroy(oldPgHitgroup);    }
    if (oldPgMissShadow)  { optixProgramGroupDestroy(oldPgMissShadow);  }
    if (oldPgMiss)        { optixProgramGroupDestroy(oldPgMiss);        }
    if (oldPgRaygen)      { optixProgramGroupDestroy(oldPgRaygen);      }
    if (oldModule)        { optixModuleDestroy(oldModule);              }
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

    // ── Miss records — index 0 = radiance, index 1 = NEE shadow ─────────────
    MissRecord missRecs[2] = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(m_pgMiss,       &missRecs[0]));
    OPTIX_CHECK(optixSbtRecordPackHeader(m_pgMissShadow, &missRecs[1]));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_sbtMissBuffer), sizeof(missRecs)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_sbtMissBuffer),
                           missRecs, sizeof(missRecs), cudaMemcpyHostToDevice));

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
    m_sbt.missRecordCount             = 2;  // [0]=radiance, [1]=shadow

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

    // ── Denoiser guide + working buffers ──────────────────────────────────────
    if (m_normalBuffer)    { cudaFree(reinterpret_cast<void*>(m_normalBuffer));    m_normalBuffer    = 0; }
    if (m_albedoBuffer)    { cudaFree(reinterpret_cast<void*>(m_albedoBuffer));    m_albedoBuffer    = 0; }
    if (m_hdrBuffer)       { cudaFree(reinterpret_cast<void*>(m_hdrBuffer));       m_hdrBuffer       = 0; }
    if (m_denoisedBuffer)  { cudaFree(reinterpret_cast<void*>(m_denoisedBuffer));  m_denoisedBuffer  = 0; }
    if (m_denoiserState)   { cudaFree(reinterpret_cast<void*>(m_denoiserState));   m_denoiserState   = 0; }
    if (m_denoiserScratch) { cudaFree(reinterpret_cast<void*>(m_denoiserScratch)); m_denoiserScratch = 0; }
    delete[] h_hdrBuffer;  h_hdrBuffer = nullptr;

    const size_t float4Bytes = pixelCount * sizeof(float4);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_normalBuffer),   float4Bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_albedoBuffer),   float4Bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_hdrBuffer),      float4Bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_denoisedBuffer), float4Bytes));
    h_hdrBuffer = new float4[pixelCount];

    if (m_denoiser)
    {
        OptixDenoiserSizes sizes = {};
        OPTIX_CHECK(optixDenoiserComputeMemoryResources(
            m_denoiser,
            static_cast<unsigned int>(w),
            static_cast<unsigned int>(h),
            &sizes));
        m_denoiserStateSize   = sizes.stateSizeInBytes;
        m_denoiserScratchSize = sizes.withoutOverlapScratchSizeInBytes;

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_denoiserState),   m_denoiserStateSize));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_denoiserScratch), m_denoiserScratchSize));

        OPTIX_CHECK(optixDenoiserSetup(
            m_denoiser, nullptr,
            static_cast<unsigned int>(w),
            static_cast<unsigned int>(h),
            m_denoiserState,   m_denoiserStateSize,
            m_denoiserScratch, m_denoiserScratchSize));
    }

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

// ─── TLAS rebuild ────────────────────────────────────────────────────────────

void Application::rebuildTlas()
{
    if (!m_accel || !m_accel->valid())
    {
        return;
    }
    try
    {
        m_accel->rebuildTlas(m_optixContext, *m_scene);
    }
    catch (const std::exception& e)
    {
        m_loadError = std::string("TLAS rebuild failed: ") + e.what();
    }
    m_accumDirty = true;
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
    freeTexture(m_envMap);  // releases texture + any previously built CDF
    m_envMapPath.clear();

    if (loadEXR(path, m_envMap, m_envMapError))
    {
        uploadToGpu(m_envMap);
        buildEnvMapCdf();
        m_envMapPath = std::filesystem::path(path).filename().string();
        m_accumDirty = true;  // new env map = new lighting; clear accumulated samples
    }
}

// ─── HDRI importance-sampling CDF ────────────────────────────────────────────

void Application::buildEnvMapCdf()
{
    if (!m_envMap.isHdr() || m_envMap.pixels.empty())
        return;

    const int   W       = m_envMap.width;
    const int   H       = m_envMap.height;
    const float* src    = m_envMap.floatPixels();  // RGBA32F, row-major
    const float  kInvPi = 0.31830988618f;

    // ── Per-pixel weight = luminance(RGB) × sin(θ) ───────────────────────────
    std::vector<float> weights(static_cast<size_t>(H) * W);
    std::vector<float> rowSums(H, 0.f);

    for (int j = 0; j < H; ++j)
    {
        const float theta    = (j + 0.5f) / static_cast<float>(H) * 3.14159265358979f;
        const float sinTheta = sinf(theta);

        for (int i = 0; i < W; ++i)
        {
            const float* p = src + (j * W + i) * 4;
            const float  lum = 0.2126f * p[0] + 0.7152f * p[1] + 0.0722f * p[2];
            const float  w   = fmaxf(0.f, lum) * sinTheta;
            weights[j * W + i] = w;
            rowSums[j] += w;
        }
    }

    // ── Conditional CDF per row (prefix-sum, normalised to [0,1]) ────────────
    std::vector<float> conditionalCdf(static_cast<size_t>(H) * W);

    for (int j = 0; j < H; ++j)
    {
        const float invRow = (rowSums[j] > 0.f) ? 1.f / rowSums[j] : 0.f;
        float running = 0.f;
        for (int i = 0; i < W; ++i)
        {
            running += weights[j * W + i];
            conditionalCdf[j * W + i] = (rowSums[j] > 0.f)
                ? running * invRow
                : (i + 1.f) / static_cast<float>(W);
        }
        // Force last entry to exactly 1 to prevent binary-search overrun
        conditionalCdf[j * W + (W - 1)] = 1.f;
    }

    // ── Marginal CDF (prefix-sum over rows, normalised to [0,1]) ─────────────
    std::vector<float> marginalCdf(H);
    float totalWeight = 0.f;
    for (int j = 0; j < H; ++j)
        totalWeight += rowSums[j];

    {
        float running = 0.f;
        for (int j = 0; j < H; ++j)
        {
            running += rowSums[j];
            marginalCdf[j] = (totalWeight > 0.f)
                ? running / totalWeight
                : (j + 1.f) / static_cast<float>(H);
        }
        marginalCdf[H - 1] = 1.f;  // force last entry exactly to 1
    }

    // ── Upload to device — stored on the Texture itself ──────────────────────
    // Any previous CDF buffers are already freed by freeTexture() in loadEnvMap().
    const size_t marginalBytes    = static_cast<size_t>(H) * sizeof(float);
    const size_t conditionalBytes = static_cast<size_t>(H) * W * sizeof(float);

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_envMap.cdfMarginal),    marginalBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_envMap.cdfConditional), conditionalBytes));

    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_envMap.cdfMarginal),
                           marginalCdf.data(),    marginalBytes,    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_envMap.cdfConditional),
                           conditionalCdf.data(), conditionalBytes, cudaMemcpyHostToDevice));
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
        const bool shiftHeld = glfwGetKey(m_window, GLFW_KEY_LEFT_SHIFT)   == GLFW_PRESS
                            || glfwGetKey(m_window, GLFW_KEY_RIGHT_SHIFT)  == GLFW_PRESS;
        const bool ctrlHeld  = glfwGetKey(m_window, GLFW_KEY_LEFT_CONTROL)  == GLFW_PRESS
                            || glfwGetKey(m_window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS;

        if (shiftHeld)
        {
            // Rotate environment map azimuthally — same sensitivity as camera rotation
            m_envMapRotation += dx * m_rotSpeed;
            m_accumDirty = true;
        }
        else if (ctrlHeld)
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
    cam.transform.m[0][0] =  cy;  cam.transform.m[0][1] = -sy*sp; cam.transform.m[0][2] = -sy*cp; cam.transform.m[0][3] = m_camPos.x;
    cam.transform.m[1][0] = 0.0f; cam.transform.m[1][1] =  cp;    cam.transform.m[1][2] = -sp;    cam.transform.m[1][3] = m_camPos.y;
    cam.transform.m[2][0] =  sy;  cam.transform.m[2][1] =  cy*sp; cam.transform.m[2][2] =  cy*cp; cam.transform.m[2][3] = m_camPos.z;
    cam.transform.m[3][0] = 0.0f; cam.transform.m[3][1] =  0.0f;  cam.transform.m[3][2] =  0.0f;  cam.transform.m[3][3] = 1.0f;
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
            // Exponential moving average — α=0.02 ≈ 50-frame window for stable stats
            m_frameTimeMs = 0.02f * deltaMs + 0.98f * m_frameTimeMs;
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
    ImGuizmo::BeginFrame();  // must be called once per frame, right after ImGui::NewFrame()

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
            m_sampleCount             = 0;
            m_accumDirty              = false;
            m_hasValidDenoisedFrame   = false;  // stale denoised frame is now invalid
        }

        m_launchParams.colorBuffer    = d_colorBuffer;
        m_launchParams.fbSize         = make_uint2(static_cast<unsigned int>(m_viewportWidth), static_cast<unsigned int>(m_viewportHeight));
        m_launchParams.traversable    = (m_accel && m_accel->valid()) ? m_accel->traversable() : 0;
        m_launchParams.envMap            = m_envMap.gpuTex;
        m_launchParams.envMapRotation    = m_envMapRotation;
        m_launchParams.envExposure       = m_envExposure;
        m_launchParams.envMarginalCdf    = m_envMap.cdfMarginal
            ? reinterpret_cast<const float*>(m_envMap.cdfMarginal)    : nullptr;
        m_launchParams.envConditionalCdf = m_envMap.cdfConditional
            ? reinterpret_cast<const float*>(m_envMap.cdfConditional) : nullptr;
        m_launchParams.envCdfW           = m_envMap.width;
        m_launchParams.envCdfH           = m_envMap.height;
        m_launchParams.accumBuffer    = m_accumBuffer ? reinterpret_cast<float4*>(m_accumBuffer) : nullptr;
        m_launchParams.sampleIndex    = m_sampleCount;
        m_launchParams.materials      = m_materialsBuffer ? reinterpret_cast<const MaterialData*>(m_materialsBuffer) : nullptr;
        m_launchParams.normalBuffer   = m_normalBuffer ? reinterpret_cast<float4*>(m_normalBuffer) : nullptr;
        m_launchParams.albedoBuffer   = m_albedoBuffer ? reinterpret_cast<float4*>(m_albedoBuffer) : nullptr;
        m_launchParams.hdrBuffer      = m_hdrBuffer    ? reinterpret_cast<float4*>(m_hdrBuffer)    : nullptr;

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

            // FOV from physical lens + sensor parameters
            const float aspect       = static_cast<float>(m_viewportWidth)
                                      / static_cast<float>(m_viewportHeight);
            const float sensorHeight = cam.sensorSize / aspect;  // mm
            const float tanHalfFovV  = sensorHeight / (2.0f * cam.focalLength);
            const float tanHalfFovH  = tanHalfFovV * aspect;

            m_launchParams.U =
                make_float3(right.x * tanHalfFovH,
                            right.y * tanHalfFovH,
                            right.z * tanHalfFovH);
            m_launchParams.V =
                make_float3(up.x * tanHalfFovV,
                            up.y * tanHalfFovV,
                            up.z * tanHalfFovV);
            m_launchParams.W = forward;

            // Thin-lens DoF parameters
            m_launchParams.lensRadius    = (cam.focalLength / (2.0f * cam.fStop)) / 1000.0f;
            m_launchParams.focusDistance = cam.focusDistance;
            m_launchParams.bokehEdgeBias = cam.bokehEdgeBias;
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

        // ── Denoiser post-process ─────────────────────────────────────────────
        const bool runDenoiser = m_denoiserEnabled
                              && m_denoiser
                              && m_hdrBuffer
                              && m_denoiserState
                              && m_denoiserInterval > 0
                              && (m_sampleCount % m_denoiserInterval == 0 || m_sampleCount == 25);
        if (runDenoiser)
        {
            const auto makeImage = [&](CUdeviceptr ptr) -> OptixImage2D
            {
                OptixImage2D img          = {};
                img.data                  = ptr;
                img.width                 = static_cast<unsigned int>(m_viewportWidth);
                img.height                = static_cast<unsigned int>(m_viewportHeight);
                img.rowStrideInBytes      = img.width * static_cast<unsigned int>(sizeof(float4));
                img.pixelStrideInBytes    = sizeof(float4);
                img.format                = OPTIX_PIXEL_FORMAT_FLOAT4;
                return img;
            };

            OptixDenoiserLayer layer = {};
            layer.input              = makeImage(m_hdrBuffer);
            layer.output             = makeImage(m_denoisedBuffer);

            OptixDenoiserGuideLayer guide = {};
            guide.normal                  = makeImage(m_normalBuffer);
            guide.albedo                  = makeImage(m_albedoBuffer);

            OPTIX_CHECK(optixDenoiserComputeIntensity(
                m_denoiser, nullptr,
                &layer.input,
                m_denoiserIntensity,
                m_denoiserScratch, m_denoiserScratchSize));

            OptixDenoiserParams denoiserParams = {};
            denoiserParams.hdrIntensity        = m_denoiserIntensity;
            denoiserParams.blendFactor         = 0.0f;

            OPTIX_CHECK(optixDenoiserInvoke(
                m_denoiser, nullptr,
                &denoiserParams,
                m_denoiserState,   m_denoiserStateSize,
                &guide,
                &layer, 1,
                0, 0,
                m_denoiserScratch, m_denoiserScratchSize));

            CUDA_CHECK(cudaDeviceSynchronize());

            // Copy denoised float4 to host and CPU-tone-map into h_colorBuffer
            const size_t pixelCount = static_cast<size_t>(m_viewportWidth) * m_viewportHeight;
            CUDA_CHECK(cudaMemcpy(h_hdrBuffer,
                reinterpret_cast<void*>(m_denoisedBuffer),
                pixelCount * sizeof(float4),
                cudaMemcpyDeviceToHost));

            for (size_t i = 0; i < pixelCount; ++i)
            {
                float r = h_hdrBuffer[i].x / (h_hdrBuffer[i].x + 1.0f);
                float g = h_hdrBuffer[i].y / (h_hdrBuffer[i].y + 1.0f);
                float b = h_hdrBuffer[i].z / (h_hdrBuffer[i].z + 1.0f);
                r = powf(std::max(0.0f, r), 1.0f / 2.2f);
                g = powf(std::max(0.0f, g), 1.0f / 2.2f);
                b = powf(std::max(0.0f, b), 1.0f / 2.2f);
                h_colorBuffer[i] = make_uchar4(
                    static_cast<unsigned char>(r * 255.0f),
                    static_cast<unsigned char>(g * 255.0f),
                    static_cast<unsigned char>(b * 255.0f),
                    255u);
            }
            m_hasValidDenoisedFrame = true;
        }
        else if (!m_denoiserEnabled || !m_hasValidDenoisedFrame)
        {
            // Show the live raygen output when:
            //  • denoiser is off, OR
            //  • denoiser is on but hasn't fired yet since the last accum reset
            //    (e.g. camera just moved) — keeps the viewport responsive.
            CUDA_CHECK(cudaMemcpy(
                h_colorBuffer, d_colorBuffer,
                static_cast<size_t>(m_viewportWidth) * m_viewportHeight * sizeof(uchar4),
                cudaMemcpyDeviceToHost));
        }
        // else: denoiser enabled, valid denoised frame exists, but interval not reached —
        //        h_colorBuffer already holds the last denoised result; leave it untouched.

        glBindTexture(GL_TEXTURE_2D, m_displayTexture);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
            m_viewportWidth, m_viewportHeight,
            GL_RGBA, GL_UNSIGNED_BYTE, h_colorBuffer);
        glBindTexture(GL_TEXTURE_2D, 0);

        // UV0=(0,1) UV1=(1,0): flip vertically so OptiX row-0=bottom matches
        // OpenGL's bottom-up texture convention when displayed top-down by ImGui.
        const ImVec2 imageScreenPos = ImGui::GetCursorScreenPos();
        ImGui::Image(
            (ImTextureID)(intptr_t)m_displayTexture,
            ImVec2(static_cast<float>(m_viewportWidth),
                   static_cast<float>(m_viewportHeight)),
            ImVec2(0.0f, 1.0f),
            ImVec2(1.0f, 0.0f));

        // ── 3D gizmo overlay ──────────────────────────────────────────────────
        // Render the ImGuizmo gizmo on top of the viewport image for the
        // currently selected scene-graph node.
        if (m_selectedNodeIdx >= 0
            && m_selectedNodeIdx < static_cast<int>(m_scene->nodes().size()))
        {
            const float vpW = static_cast<float>(m_viewportWidth);
            const float vpH = static_cast<float>(m_viewportHeight);

            ImGuizmo::SetOrthographic(false);
            ImGuizmo::SetDrawlist();
            ImGuizmo::SetRect(imageScreenPos.x, imageScreenPos.y, vpW, vpH);

            // ── View matrix (column-major) ────────────────────────────────────
            const float3 eye = m_launchParams.eye;
            const float3 U   = m_launchParams.U;
            const float3 V   = m_launchParams.V;
            const float3 W   = m_launchParams.W;

            // Normalise U and V to recover unit right/up directions
            auto len3  = [](float3 v){ return sqrtf(v.x*v.x + v.y*v.y + v.z*v.z); };
            auto dot3  = [](float3 a, float3 b){ return a.x*b.x + a.y*b.y + a.z*b.z; };
            const float3 R  = { U.x / len3(U), U.y / len3(U), U.z / len3(U) };
            const float3 Up = { V.x / len3(V), V.y / len3(V), V.z / len3(V) };

            const float view[16] = {
                R.x,         Up.x,        -W.x,       0.f,
                R.y,         Up.y,        -W.y,       0.f,
                R.z,         Up.z,        -W.z,       0.f,
                -dot3(R, eye), -dot3(Up, eye), dot3(W, eye), 1.f
            };

            // ── Projection matrix (column-major) ──────────────────────────────
            const float tanHalfFovV = len3(V);
            const float f           = 1.0f / tanHalfFovV;
            const float aspect      = vpW / vpH;
            const float zNear = 0.01f, zFar = 10000.0f;
            const float proj[16] = {
                f / aspect, 0.f, 0.f,  0.f,
                0.f,        f,   0.f,  0.f,
                0.f, 0.f, (zFar + zNear) / (zNear - zFar), -1.f,
                0.f, 0.f, 2.0f * zFar * zNear / (zNear - zFar), 0.f
            };

            // ── World transform → column-major float[16] ─────────────────────
            Matrix4x4 worldTx = m_scene->computeWorldTransform(m_selectedNodeIdx);
            float gizmoMatrix[16];
            mat4ToColMajor(worldTx, gizmoMatrix);

            ImGuizmo::Manipulate(view, proj, m_gizmoOp, m_gizmoMode, gizmoMatrix);

            if (ImGuizmo::IsUsing())
            {
                // Convert result back to row-major world transform
                Matrix4x4 newWorldTx = mat4FromColMajor(gizmoMatrix);

                // Compute the new local transform relative to the parent
                Node3D& node = m_scene->nodeAt(m_selectedNodeIdx);
                if (node.parent >= 0)
                {
                    Matrix4x4 parentWorld = m_scene->computeWorldTransform(node.parent);
                    node.localTransform   = mat4Multiply(mat4Inverse(parentWorld), newWorldTx);
                }
                else
                {
                    node.localTransform = newWorldTx;
                }

                rebuildTlas();
            }
        }
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
            freeTexture(m_envMap);  // frees texture + CDF buffers
            m_envMapPath.clear();
            m_envMapError.clear();
            m_accumDirty = true;
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

    if (ImGui::SliderFloat("Env Exposure (EV)", &m_envExposure, -10.0f, 10.0f, "%.1f"))
    {
        m_accumDirty = true;
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
    {
        const float aspect      = static_cast<float>(m_viewportWidth)
                                 / static_cast<float>(m_viewportHeight);
        const float sH          = cam.sensorSize / aspect;
        const float derivedFov  = 2.0f * std::atan(sH / (2.0f * cam.focalLength));
        ImGui::Text("FOV: %.1f deg (%.0f mm / %.0f mm sensor)",
            derivedFov * (180.0f / 3.14159265f), cam.focalLength, cam.sensorSize);
    }

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
    ImGui::SameLine();
    if (ImGui::Checkbox("Denoiser", &m_denoiserEnabled))
    {
        m_accumDirty = true;
    }
    if (m_denoiserEnabled)
    {
        ImGui::DragInt("Denoise every N samples", &m_denoiserInterval, 1, 1, 10000);
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
                if (ImGui::ColorEdit3("Albedo",   &mats[i].albedo.x, ImGuiColorEditFlags_Float))
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
                if (ImGui::DragFloat("Emission Scale", &mats[i].emissionScale, 0.1f, 0.0f, 1000.0f, "%.2f"))
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
                if (ImGui::DragFloat("Absorption Dist.", &mats[i].absorptionDistance, 0.002f, 0.0001f, 1000.0f, "%.4f"))
                {
                    anyMatChanged = true;
                }
                if (ImGui::SliderFloat("Clearcoat",      &mats[i].clearcoat,          0.0f, 1.0f, "%.3f"))
                {
                    anyMatChanged = true;
                }
                if (ImGui::SliderFloat("Coat Roughness", &mats[i].clearcoatRoughness,  0.0f, 1.0f, "%.3f"))
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

        // ── Gizmo operation selector ──────────────────────────────────────────
        ImGui::Text("Gizmo:");
        ImGui::SameLine();
        if (ImGui::RadioButton("Translate", m_gizmoOp == ImGuizmo::TRANSLATE))
            m_gizmoOp = ImGuizmo::TRANSLATE;
        ImGui::SameLine();
        if (ImGui::RadioButton("Rotate", m_gizmoOp == ImGuizmo::ROTATE))
            m_gizmoOp = ImGuizmo::ROTATE;
        ImGui::SameLine();
        if (ImGui::RadioButton("Scale", m_gizmoOp == ImGuizmo::SCALE))
            m_gizmoOp = ImGuizmo::SCALE;
        ImGui::SameLine();
        ImGui::Spacing();
        ImGui::SameLine();
        if (ImGui::RadioButton("Local", m_gizmoMode == ImGuizmo::LOCAL))
            m_gizmoMode = ImGuizmo::LOCAL;
        ImGui::SameLine();
        if (ImGui::RadioButton("World", m_gizmoMode == ImGuizmo::WORLD))
            m_gizmoMode = ImGuizmo::WORLD;

        ImGui::Separator();

        // ── Local transform (raw matrix, for fine-grained editing) ────────────
        if (ImGui::CollapsingHeader("Local Transform"))
        {
            bool transformChanged = false;
            ImGui::PushItemWidth(-1.0f);
            for (int row = 0; row < 4; ++row)
            {
                ImGui::PushID(row);
                if (ImGui::DragFloat4("", node.localTransform.m[row], 0.01f))
                    transformChanged = true;
                ImGui::PopID();
            }
            ImGui::PopItemWidth();
            if (transformChanged)
                rebuildTlas();
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
                    if (ImGui::ColorEdit3("Albedo",   &mats[matIdx].albedo.x, ImGuiColorEditFlags_Float))
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
                    if (ImGui::DragFloat("Absorption Dist.", &mats[matIdx].absorptionDistance,
                                         0.01f, 0.001f, 1000.0f, "%.3f"))
                    {
                        anyChanged = true;
                    }
                    if (ImGui::SliderFloat("Clearcoat",      &mats[matIdx].clearcoat,          0.0f, 1.0f, "%.3f"))
                    {
                        anyChanged = true;
                    }
                    if (ImGui::SliderFloat("Coat Roughness", &mats[matIdx].clearcoatRoughness,  0.0f, 1.0f, "%.3f"))
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
            {
                {
                    Camera cam = m_scene->camera();

                    // Focal length — drives FOV together with sensor size
                    float fl = cam.focalLength;
                    if (ImGui::SliderFloat("Focal Length (mm)", &fl, 8.0f, 800.0f, "%.1f mm"))
                    {
                        cam.focalLength = fl;
                        m_scene->setCamera(cam);
                        m_accumDirty = true;
                    }

                    // Sensor size (horizontal width)
                    float ss = cam.sensorSize;
                    if (ImGui::SliderFloat("Sensor Size (mm)", &ss, 1.0f, 100.0f, "%.1f mm"))
                    {
                        cam.sensorSize = ss;
                        m_scene->setCamera(cam);
                        m_accumDirty = true;
                    }

                    // F-stop / aperture
                    float fs = cam.fStop;
                    if (ImGui::SliderFloat("F-Stop", &fs, 0.5f, 64.0f, "f/%.1f"))
                    {
                        cam.fStop = fs;
                        m_scene->setCamera(cam);
                        m_accumDirty = true;
                    }

                    // Focus distance
                    float fd = cam.focusDistance;
                    if (ImGui::DragFloat("Focus Distance", &fd, 0.1f, 0.1f, 10000.0f, "%.2f m"))
                    {
                        cam.focusDistance = std::max(0.001f, fd);
                        m_scene->setCamera(cam);
                        m_accumDirty = true;
                    }

                    // Bokeh edge bias
                    float eb = cam.bokehEdgeBias;
                    if (ImGui::SliderFloat("Bokeh Edge Bias", &eb, 0.0f, 1.0f, "%.2f"))
                    {
                        cam.bokehEdgeBias = eb;
                        m_scene->setCamera(cam);
                        m_accumDirty = true;
                    }

                    // Derived FOV display
                    const float aspect     = static_cast<float>(m_viewportWidth)
                                            / static_cast<float>(m_viewportHeight);
                    const float sH         = cam.sensorSize / aspect;
                    const float derivedFov = 2.0f * std::atan(sH / (2.0f * cam.focalLength));
                    ImGui::Text("FOV: %.1f deg  |  Aperture: %.1f mm",
                        derivedFov * (180.0f / 3.14159265f),
                        cam.focalLength / cam.fStop);
                    ImGui::TextDisabled("(transform driven by fly-cam controller)");
                }
            }
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
