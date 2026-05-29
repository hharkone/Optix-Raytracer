#ifndef OPTIX_RAYTRACER_ACCEL_H
#define OPTIX_RAYTRACER_ACCEL_H

#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>

#include <vector>

// Forward declaration — Accel.cpp includes Scene.h
class Scene;

// Manages the OptiX acceleration structure for a loaded scene:
//   - one BLAS (bottom-level AS) per mesh, built with compaction
//   - one TLAS (top-level AS) instancing all BLASes with identity transforms
//
// The traversable handle exposed here is stored in LaunchParams and passed to
// optixLaunch so device programs can call optixTrace() against the geometry.
class Accel
{
public:
    Accel()  = default;
    ~Accel() { destroy(); }

    Accel(const Accel&)            = delete;
    Accel& operator=(const Accel&) = delete;
    Accel(Accel&&)                 = delete;
    Accel& operator=(Accel&&)      = delete;

    // Build or rebuild from scene geometry. Destroys any previous state first.
    // Throws std::runtime_error on CUDA / OptiX failure.
    void build(OptixDeviceContext ctx, const Scene& scene);

    // Free all device memory and reset to empty state. Safe to call multiple times.
    void destroy();

    OptixTraversableHandle traversable() const { return m_tlas; }
    bool                   valid()       const { return m_tlas != 0; }
    size_t                 meshCount()   const { return m_meshBuffers.size(); }

    // Per-mesh device pointers needed to fill SBT hit group records.
    struct MeshDevicePtrs
    {
        CUdeviceptr positions;  // device float3 array
        CUdeviceptr normals;    // device float3 array
        CUdeviceptr indices;    // device uint3  array
    };
    MeshDevicePtrs meshDevicePtrs(size_t idx) const
    {
        return { m_meshBuffers[idx].positions,
                 m_meshBuffers[idx].normals,
                 m_meshBuffers[idx].indices };
    }

private:
    // Per-mesh device buffers kept alive for the lifetime of the AS.
    struct MeshBuffers
    {
        CUdeviceptr            positions = 0;  // device copy of Mesh::positions
        CUdeviceptr            normals   = 0;  // device copy of Mesh::normals
        CUdeviceptr            indices   = 0;  // device copy of Mesh::indices
        CUdeviceptr            outputAS  = 0;  // compacted BLAS output buffer
        OptixTraversableHandle blas      = 0;
    };

    std::vector<MeshBuffers> m_meshBuffers;
    CUdeviceptr              m_tlasOutputBuffer = 0;
    CUdeviceptr              m_instanceBuffer   = 0;  // device OptixInstance array
    OptixTraversableHandle   m_tlas             = 0;

    // Uploads vertex/index data and builds one BLAS with compaction.
    // Writes to buffers.positions, buffers.indices, buffers.outputAS, buffers.blas.
    static void buildBlas(
        OptixDeviceContext ctx,
        MeshBuffers&       buffers,
        unsigned int       vertexCount,
        unsigned int       triangleCount);
};

#endif // OPTIX_RAYTRACER_ACCEL_H
