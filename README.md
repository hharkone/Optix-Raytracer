# OptiX Path Tracer

A physically based GPU path tracer built on NVIDIA OptiX 9.x, CUDA, C++17, and Dear ImGui.

## Features

### Rendering
- **Monte Carlo path tracing** with up to 16 bounces and per-pixel progressive accumulation
- **PBR materials** (GGX-VNDF microfacet BRDF) — albedo, roughness, metallic, emission, transmission, IOR, absorption distance
- **Probabilistic path selection** — diffuse/specular split driven by Fresnel probability for energy conservation
- **Thin-lens depth of field** — focal length, sensor size, f-stop, focus distance, and adjustable bokeh edge bias
- **Beer-Lambert volumetric absorption** — coloured glass with physically accurate thickness falloff
- **Environment lighting** — equirectangular EXR maps or procedural sky gradient, with rotation and exposure (EV) controls
- **Reinhard tone mapping** with sRGB gamma encoding
- **OptiX AI denoiser** — normal + albedo guide layers, configurable denoise interval, keeps the last denoised frame while accumulating

### Scene
- **glTF 2.0 / GLB loading** — meshes, PBR materials, base-colour textures, cameras, scene hierarchy
- **Scene graph** — full glTF node hierarchy preserved as a `Node3D` tree (`MeshNode`, `CameraNode`, `GroupNode`)
- **Node transforms applied to TLAS** — mesh instances positioned using accumulated world-space transforms from the node hierarchy
- **Live transform editing** — drag node transform values in the Node Properties panel; TLAS-only rebuild keeps BLASes intact

### Camera
- **Free-fly camera** — WASD (move), EQ (up/down), right-drag (look), Ctrl+drag (orbit origin), Shift+drag (rotate environment)
- **Physical camera parameters** — focal length (mm), sensor size (mm), f-stop, and focus distance drive the FOV and depth of field
- **glTF camera import** — imported yFov converted to focal length at load time

### UI (Dear ImGui with docking)
| Panel | Contents |
|---|---|
| **Viewport** | Live rendered image, resizes dynamically |
| **Raytracer** | GPU stats, sample count, denoiser toggle, environment controls |
| **Materials** | Per-material editor with all PBR parameters |
| **Scene Graph** | Hierarchy tree of all scene nodes (click to select) |
| **Node Properties** | Transform matrix, material editor, camera parameters for the selected node |

### Performance
- **PTX hot-reload** — edit `devicePrograms.cu`, rebuild the PTX, and the shader reloads without restarting
- **BLAS compaction** — per-mesh bottom-level AS built with size compaction
- **Frame-time EMA** — smoothed frame time and Mrays/s display

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| NVIDIA GPU | Compute capability ≥ 7.5 | RTX 20xx / 30xx / 40xx |
| NVIDIA Driver | ≥ 570.x | Required by OptiX 9.1 |
| [NVIDIA OptiX SDK](https://developer.nvidia.com/optix) | 9.1.0 | Free download; requires NVIDIA developer account |
| [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) | 12.x or 13.x | Installs `nvcc` and CUDA runtime |
| [Visual Studio 2022](https://visualstudio.microsoft.com/) | 17.x | With **Desktop development with C++** and **CUDA** workloads |
| [CMake](https://cmake.org/download/) | ≥ 3.20 | Add to PATH during install |

> **Driver check**: Run `nvidia-smi`. The driver version appears top-right. If below 570, download the latest from [nvidia.com/drivers](https://www.nvidia.com/drivers).

---

## Building

Two batch scripts are provided in the repository root:

| Script | Purpose |
|---|---|
| `configure.bat` | Generate (or refresh) the Visual Studio solution |
| `build.bat` | Configure if needed, then compile |

### Command line

```bat
build.bat          :: Debug build (configures automatically on first run)
build.bat Release  :: Release build
```

On first run, CMake fetches GLFW, ImGui, tinygltf, tinyexr, and nativefiledialog-extended from GitHub — internet access is required. Delete `build\CMakeCache.txt` to force a full reconfigure.

### Visual Studio

```bat
configure.bat          :: Generate build\OptixRaytracer.sln
configure.bat --clean  :: Wipe CMake cache first, then regenerate
```

Open `build\OptixRaytracer.sln`. **OptixRaytracer** is the startup project — press **F5** to run. Re-run `configure.bat` after adding/removing source files or changing `CMakeLists.txt`.

### Manual CMake

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64

# If OptiX is not detected automatically:
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
    -DOptiX_INSTALL_DIR="C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.1.0"

cmake --build build --config Debug   --parallel
cmake --build build --config Release --parallel
```

### Run

```powershell
.\build\bin\Debug\OptixRaytracer.exe
```

---

## Controls

| Input | Action |
|---|---|
| **RMB drag** | Free-look (rotate camera orientation) |
| **Ctrl + RMB drag** | Orbit camera around world origin |
| **Shift + RMB drag** | Rotate environment map azimuthally |
| **W / S** | Move forward / backward |
| **A / D** | Strafe left / right |
| **E / Q** | Move up / down |
| **Open glTF…** | Browse for `.gltf` or `.glb` scene file |
| **Open EXR…** | Browse for an equirectangular HDR environment map |
| **Clear EXR** | Remove the environment map (falls back to procedural sky) |

---

## Project Structure

```
Optix-Raytracer/
├── CMakeLists.txt              Root build: project settings, FetchContent, subdirs
├── build.bat / configure.bat   Convenience build scripts
├── cmake/
│   ├── FindOptiX.cmake         Locates the OptiX SDK; creates the OptiX::OptiX target
│   └── cuda_intellisense.props.in  VS property sheet: adds OptiX to IntelliSense
├── extern/
│   └── glad/                   Pre-generated OpenGL 3.3 core function loader
├── shaders/
│   ├── device_math.h           float3 operator overloads (+ − * / for device and host)
│   ├── LaunchParams.h          GPU parameter struct shared between host and device
│   ├── SceneData.h             MeshData and MaterialData (no STL; host + device)
│   └── devicePrograms.cu       OptiX device programs (path tracer, denoiser guides)
└── src/
    ├── main.cpp                Entry point
    ├── Application.h/.cpp      Window, CUDA/OptiX init, ImGui UI, per-frame render loop
    ├── Math.h                  Matrix4x4 + inline mat4Identity / mat4Multiply
    ├── Camera.h                Camera struct: transform, FOV, DoF parameters
    ├── Node3D.h                Node3D base + MeshNode, CameraNode, GroupNode
    ├── Scene.h/.cpp            Scene container: meshes, materials, textures, node tree
    ├── Mesh.h                  Host-side mesh: separate vertex attribute arrays
    ├── Texture.h/.cpp          Host+GPU texture: RGBA8 / RGBA32F, EXR loading, GPU upload
    ├── Accel.h/.cpp            OptiX acceleration structure: BLAS per mesh + TLAS
    ├── SceneLoader.h/.cpp      glTF 2.0 loader (tinygltf); populates Scene from file
    └── CMakeLists.txt          Executable target, include paths, link libraries
```

---

## Troubleshooting

**`OptiX SDK not found`**  
Add `-DOptiX_INSTALL_DIR=...` to the CMake configure command. The SDK defaults to `C:/ProgramData/NVIDIA Corporation/OptiX SDK <version>`.

**`optixInit() failed` or crash on startup**  
Your NVIDIA driver is too old. Update to ≥ 570.x from [nvidia.com/drivers](https://www.nvidia.com/drivers).

**`nvcc` not found during configure**  
CUDA Toolkit is not on PATH. Reinstall CUDA Toolkit and ensure it is added to PATH, or open the project from a **Visual Studio Developer Command Prompt**.

**`CUDA : error : Cannot find compiler 'cl.exe'`**  
Visual Studio C++ workload is missing. Open the VS Installer, modify the 2022 installation, and add **Desktop development with C++**.

**Image is very dark or very bright**  
Adjust the **Env Exposure** slider in the Raytracer panel. For scenes with emissive materials adjust the emissive scale on the material.

**Depth of field has no visible effect**  
Ensure f-stop is low (try f/2 or f/1.4) and that objects in the scene are at a different distance from the **Focus Distance** setting in the Node Properties camera panel.

---

## License

GNU General Public License v3 — see [LICENSE](LICENSE).
