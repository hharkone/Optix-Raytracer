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

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

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

Application::Application(int width, int height, const std::string& title)
    : m_width(width), m_height(height)
{
    initWindow(title);
    initOpenGL();
    initImGui();
    initCuda();
    initOptix();

    const size_t fbBytes = static_cast<size_t>(width) * height * sizeof(uchar4);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_colorBuffer), fbBytes));
    h_colorBuffer = new uchar4[static_cast<size_t>(width) * height];

    m_scene = std::make_unique<Scene>();
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

    if (m_optixContext)
    {
        optixDeviceContextDestroy(m_optixContext);
        m_optixContext = nullptr;
    }

    NFD_Quit();

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
        throw std::runtime_error("Failed to initialise GLFW");

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    m_window = glfwCreateWindow(m_width, m_height, title.c_str(), nullptr, nullptr);
    if (!m_window)
        throw std::runtime_error("Failed to create GLFW window");

    glfwMakeContextCurrent(m_window);
    glfwSwapInterval(1);
}

void Application::initOpenGL()
{
    if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress)))
        throw std::runtime_error("Failed to initialise GLAD");
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

// ─── Scene loading ────────────────────────────────────────────────────────────

void Application::loadScene(const std::string& path)
{
    m_scene->clear();
    m_loadError.clear();
    m_sceneFilePath.clear();
    if (loadGltfFile(path, *m_scene, m_loadError))
        m_sceneFilePath = path;
    else
        m_scene->clear();  // discard any partial data from a failed load
}

// ─── Per-frame ────────────────────────────────────────────────────────────────

bool Application::tick()
{
    if (glfwWindowShouldClose(m_window))
        return false;

    glfwPollEvents();

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // Enable a full-window dockspace so panels can be docked to it
    ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport(),
                                 ImGuiDockNodeFlags_PassthruCentralNode);

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
            ImGui::SetTooltip("%s", m_sceneFilePath.c_str());
    }
    else
    {
        ImGui::TextDisabled("No scene loaded");
    }

    if (!m_loadError.empty())
        ImGui::TextColored(ImVec4(1.f, 0.3f, 0.3f, 1.f),
                           "Error: %s", m_loadError.c_str());

    ImGui::Separator();
    ImGui::Text("Meshes: %d  Materials: %d  Textures: %d",
        static_cast<int>(m_scene->meshes().size()),
        static_cast<int>(m_scene->materials().size()),
        static_cast<int>(m_scene->textures().size()));

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
