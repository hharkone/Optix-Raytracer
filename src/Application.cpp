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

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t rc = (call);                                                \
        if (rc != cudaSuccess) {                                                \
            throw std::runtime_error(std::string("CUDA error in " __FILE__     \
                ":" + std::to_string(__LINE__) + " — ")                        \
                + cudaGetErrorString(rc));                                      \
        }                                                                       \
    } while (0)

#define OPTIX_CHECK(call)                                                       \
    do {                                                                        \
        OptixResult rc = (call);                                                \
        if (rc != OPTIX_SUCCESS) {                                              \
            throw std::runtime_error(std::string("OptiX error in " __FILE__    \
                ":" + std::to_string(__LINE__) + " — ")                        \
                + optixGetErrorString(rc));                                     \
        }                                                                       \
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

    m_accel.reset();  // free AS device memory before destroying OptiX context

    if (m_pipeline)   { optixPipelineDestroy(m_pipeline);          m_pipeline   = nullptr; }
    if (m_pgHitgroup) { optixProgramGroupDestroy(m_pgHitgroup);    m_pgHitgroup = nullptr; }
    if (m_pgMiss)     { optixProgramGroupDestroy(m_pgMiss);        m_pgMiss     = nullptr; }
    if (m_pgRaygen)   { optixProgramGroupDestroy(m_pgRaygen);      m_pgRaygen   = nullptr; }
    if (m_module)     { optixModuleDestroy(m_module);              m_module     = nullptr; }

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
    glfwSwapInterval(1);
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
    pipelineOpts.numPayloadValues                 = 3;  // p0=R, p1=G, p2=B
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
    linkOpts.maxTraceDepth            = 1;  // primary rays only, no recursion

    OPTIX_CHECK(optixPipelineCreate(
        m_optixContext,
        &pipelineOpts, &linkOpts,
        pgs, 3,
        nullptr, nullptr,
        &m_pipeline));

    // Stack size — 2 KB continuation stack, max traversal depth 2 (TLAS → BLAS)
    OPTIX_CHECK(optixPipelineSetStackSize(m_pipeline, 0, 0, 2048, 2));
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
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_sbtMissBuffer),
                           sizeof(MissRecord)));
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
            hitRecs[i].data.indices    = reinterpret_cast<const uint3*>(ptrs.indices);
            hitRecs[i].data.normals    = nullptr;
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

    const size_t byteCount = static_cast<size_t>(w) * h * sizeof(uchar4);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_colorBuffer), byteCount));
    h_colorBuffer = new uchar4[static_cast<size_t>(w) * h];

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

// ─── Scene loading ────────────────────────────────────────────────────────────

void Application::loadScene(const std::string& path)
{
    m_scene->clear();
    m_accel.reset();
    m_loadError.clear();
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

    buildSbt();  // rebuild with new mesh count (0 if load failed or no geometry)
}

// ─── Per-frame ────────────────────────────────────────────────────────────────

bool Application::tick()
{
    if (glfwWindowShouldClose(m_window))
    {
        return false;
    }

    glfwPollEvents();

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // Enable a full-window dockspace so panels can be docked to it
    ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport(),
                                 ImGuiDockNodeFlags_PassthruCentralNode);

    // ── Viewport panel ────────────────────────────────────────────────────────
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.f, 0.f));
    ImGui::Begin("Viewport", nullptr,
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    ImGui::PopStyleVar();

    {
        const ImVec2 regionSize = ImGui::GetContentRegionAvail();
        const int vpW = std::max(1, static_cast<int>(regionSize.x));
        const int vpH = std::max(1, static_cast<int>(regionSize.y));

        if (vpW != m_viewportWidth || vpH != m_viewportHeight)
        {
            resizeFramebuffer(vpW, vpH);
        }

        // ── Update launch parameters ──────────────────────────────────────────
        m_launchParams.colorBuffer = d_colorBuffer;
        m_launchParams.fbSize      = make_uint2(
            static_cast<unsigned int>(m_viewportWidth),
            static_cast<unsigned int>(m_viewportHeight));
        m_launchParams.traversable =
            (m_accel && m_accel->valid()) ? m_accel->traversable() : 0;

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
            ImVec2(0.f, 1.f),
            ImVec2(1.f, 0.f));
    }

    ImGui::End();

    // ── Raytracer panel ───────────────────────────────────────────────────────
    ImGui::Begin("Raytracer");

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
        ImGui::TextColored(ImVec4(1.f, 0.3f, 0.3f, 1.f),
                           "Error: %s", m_loadError.c_str());
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
    ImGui::Text("FOV: %.1f deg", cam.yFov * (180.f / 3.14159265f));

    ImGui::Separator();
    if (m_accel && m_accel->valid())
    {
        ImGui::TextColored(ImVec4(0.3f, 1.f, 0.4f, 1.f), "AS: ready");
    }
    else if (!m_scene->empty())
    {
        ImGui::TextDisabled("AS: build failed");
    }
    else
    {
        ImGui::TextDisabled("AS: no geometry");
    }

    ImGui::End();

    ImGui::Render();

    int w, h;
    glfwGetFramebufferSize(m_window, &w, &h);
    glViewport(0, 0, w, h);
    glClearColor(0.10f, 0.10f, 0.15f, 1.f);
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
