# OptiX Raytracer

A GPU path tracer built on NVIDIA OptiX 9.x, C++17, and Dear ImGui with docking support.

## Prerequisites

Before configuring the project, ensure the following are installed:

| Requirement | Version | Notes |
|---|---|---|
| NVIDIA GPU | Compute capability ≥ 7.5 | RTX 20xx / 30xx / 40xx series |
| NVIDIA Driver | ≥ 570.x | Required by OptiX 9.1 |
| [NVIDIA OptiX SDK](https://developer.nvidia.com/optix) | 9.1.0 | Free download; requires NVIDIA developer account |
| [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) | 12.x or 13.x | Installs `nvcc` and CUDA runtime |
| [Visual Studio 2022](https://visualstudio.microsoft.com/) | 17.x | With **Desktop development with C++** and **CUDA** workloads |
| [CMake](https://cmake.org/download/) | ≥ 3.20 | Add to PATH during install |
| Python + pip | any recent | Only needed for the one-time GLAD generation step |

> **Driver check**: Run `nvidia-smi` in a terminal. The driver version appears top-right. If it is below 570, download the latest from [nvidia.com/drivers](https://www.nvidia.com/drivers).

## First-time Setup — Generate the OpenGL Loader

GLAD (the OpenGL function loader) is committed as pre-generated source files and does **not** need to be regenerated on every build. If you are setting up the repo for the first time and the `extern/glad/src/glad.c` file is already present, skip this step.

If the file is missing (e.g., after a fresh clone that lost it), regenerate with:

```powershell
pip install glad==0.1.36
python -m glad --generator=c --profile=core --out-path=extern/glad --api="gl=3.3"
```

## Building

Two batch scripts are provided in the repository root:

| Script | Purpose |
|---|---|
| `configure.bat` | Generate (or refresh) the Visual Studio solution without building |
| `build.bat` | Configure if needed, then compile from the command line |

### Building from the command line

```bat
build.bat          :: Debug build (configures automatically on first run)
build.bat Release  :: Release build
```

On first run the script fetches GLFW, ImGui, tinygltf, and nativefiledialog-extended from GitHub — internet access is required. To force a full reconfigure, delete `build\CMakeCache.txt` and run again.

### Building in Visual Studio

Run `configure.bat` once to generate the solution:

```bat
configure.bat          :: Generate (or refresh) build\OptixRaytracer.sln
configure.bat --clean  :: Wipe the CMake cache first, then regenerate
```

Then open `build\OptixRaytracer.sln` in Visual Studio 2022. **OptixRaytracer** is set as the startup project automatically, so pressing **F5** or **Ctrl+F5** runs it immediately.

Re-run `configure.bat` whenever you add or remove source files, or change `CMakeLists.txt`. There is no need to run it before every build — Visual Studio detects when the project files are stale and prompts to reload them.

### Manual CMake commands

If you prefer to drive CMake directly:

```powershell
# Configure
cmake -S . -B build -G "Visual Studio 17 2022" -A x64

# If OptiX is not detected automatically (e.g. multiple SDK versions installed):
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
    -DOptiX_INSTALL_DIR="C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.1.0"

# Build
cmake --build build --config Debug   --parallel
cmake --build build --config Release --parallel
```

### Run

```powershell
.\build\bin\Debug\OptixRaytracer.exe
```

You should see a 1280×720 window with a dark background and a floating **Raytracer** ImGui panel. Click **Open glTF...** to browse for a `.gltf` or `.glb` file; the panel updates with mesh, material, and texture counts once the file loads.

## Project Structure

```
Optix-Raytracer/
├── CMakeLists.txt               Root build file: project settings, FetchContent, subdirs
├── build.bat                    Command-line build script (configures on first run)
├── configure.bat                Generates / refreshes the Visual Studio solution
├── cmake/
│   ├── FindOptiX.cmake          Locates the OptiX SDK; creates the OptiX::OptiX target
│   └── cuda_intellisense.props.in  VS property sheet template: adds OptiX to IntelliSense
├── extern/
│   └── glad/                    Pre-generated OpenGL 3.3 core function loader
├── shaders/
│   ├── LaunchParams.h           LaunchParams struct shared between host and device code
│   ├── SceneData.h              MeshData and MaterialData structs (host + device, no STL)
│   └── devicePrograms.cu        OptiX device programs (raygen, miss) — compiled to PTX
└── src/
    ├── main.cpp                 Entry point
    ├── Application.h/.cpp       Window, CUDA/OptiX init, ImGui UI, per-frame loop
    ├── Scene.h/.cpp             Scene container: owns meshes, materials, and textures
    ├── Mesh.h                   Host-side mesh: separate vertex attribute arrays
    ├── Texture.h                Host-side texture: raw RGBA pixels + dimensions
    ├── SceneLoader.h/.cpp       glTF 2.0 loader (tinygltf); populates a Scene from file
    └── CMakeLists.txt           Executable target, include paths, link libraries
```

## Troubleshooting

**`OptiX SDK not found`**
Add `-DOptiX_INSTALL_DIR=...` to the CMake configure command (see above). The SDK defaults to `C:/ProgramData/NVIDIA Corporation/OptiX SDK <version>`.

**`optixInit() failed` or crash on startup**
Your NVIDIA driver is too old. Update to ≥ 570.x from [nvidia.com/drivers](https://www.nvidia.com/drivers).

**`nvcc` not found during configure**
CUDA Toolkit is not on the system PATH. Reinstall CUDA Toolkit and ensure it is added to PATH, or open the project from a **Visual Studio Developer Command Prompt**.

**`CUDA : error : Cannot find compiler 'cl.exe'`**
Your Visual Studio installation is missing the C++ workload. Open the VS Installer, modify the 2022 installation, and add **Desktop development with C++**.

**ImGui windows cannot be docked**
Docking is enabled but requires dragging a window's title bar over another panel or to the edges of the main viewport. Right-click a panel's title bar to access docking options.

## License

GNU General Public License v3 — see [LICENSE](LICENSE).
