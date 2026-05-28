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
    m_loadError.clear();
    m_sceneFilePath.clear();
    if (loadGltfFile(path, *m_scene, m_loadError))
    {
        m_sceneFilePath = path;
    }
    else
    {
        m_scene->clear();  // discard any partial data from a failed load
    }
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

        // Placeholder: CPU UV gradient — replace with OptiX launch + cudaMemcpy later
        for (int y = 0; y < m_viewportHeight; ++y)
        {
            for (int x = 0; x < m_viewportWidth; ++x)
            {
                h_colorBuffer[y * m_viewportWidth + x] = {
                    static_cast<unsigned char>(255 * x / m_viewportWidth),
                    static_cast<unsigned char>(255 * y / m_viewportHeight),
                    128,
                    255
                };
            }
        }

        glBindTexture(GL_TEXTURE_2D, m_displayTexture);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
            m_viewportWidth, m_viewportHeight,
            GL_RGBA, GL_UNSIGNED_BYTE, h_colorBuffer);
        glBindTexture(GL_TEXTURE_2D, 0);

        ImGui::Image(
            (ImTextureID)(intptr_t)m_displayTexture,
            ImVec2(static_cast<float>(m_viewportWidth),
                   static_cast<float>(m_viewportHeight)));
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
