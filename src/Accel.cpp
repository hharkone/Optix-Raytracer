// Accel.cpp — OptiX acceleration structure builder.
//
// Builds one BLAS per mesh (with compaction) and one TLAS instancing all
// BLASes with identity transforms. The compacted BLAS output buffers and TLAS
// output buffer are kept alive in MeshBuffers / m_tlasOutputBuffer so OptiX
// can continue to traverse them during rendering.
#include "Accel.h"
#include "Scene.h"

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

// ─── Error macros ─────────────────────────────────────────────────────────────

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t rc = (call);                                                \
        if (rc != cudaSuccess)                                                  \
        {                                                                       \
            throw std::runtime_error(std::string("CUDA error in " __FILE__     \
                ":" + std::to_string(__LINE__) + " — ")                        \
                + cudaGetErrorString(rc));                                      \
        }                                                                       \
    } while (0)

#define OPTIX_CHECK(call)                                                       \
    do {                                                                        \
        OptixResult rc = (call);                                                \
        if (rc != OPTIX_SUCCESS)                                                \
        {                                                                       \
            throw std::runtime_error(std::string("OptiX error in " __FILE__    \
                ":" + std::to_string(__LINE__) + " — ")                        \
                + optixGetErrorString(rc));                                     \
        }                                                                       \
    } while (0)

// ─── Helpers ──────────────────────────────────────────────────────────────────

// Free a CUdeviceptr and zero it. Safe to call with ptr == 0.
static void cudaFreePtr(CUdeviceptr& ptr)
{
    if (ptr)
    {
        cudaFree(reinterpret_cast<void*>(ptr));
        ptr = 0;
    }
}

// ─── Accel::destroy ───────────────────────────────────────────────────────────

void Accel::destroy()
{
    for (MeshBuffers& mb : m_meshBuffers)
    {
        cudaFreePtr(mb.positions);
        cudaFreePtr(mb.indices);
        cudaFreePtr(mb.outputAS);
        mb.blas = 0;
    }
    m_meshBuffers.clear();

    cudaFreePtr(m_tlasOutputBuffer);
    cudaFreePtr(m_instanceBuffer);
    m_tlas = 0;
}

// ─── Accel::buildBlas ─────────────────────────────────────────────────────────

void Accel::buildBlas(
    OptixDeviceContext ctx,
    MeshBuffers&       buffers,
    unsigned int       vertexCount,
    unsigned int       triangleCount)
{
    // One geometry flag entry (one SBT record)
    const uint32_t buildFlags[] = { OPTIX_GEOMETRY_FLAG_NONE };

    OptixBuildInput buildInput       = {};
    buildInput.type                  = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    auto& tri                        = buildInput.triangleArray;
    tri.vertexFormat                 = OPTIX_VERTEX_FORMAT_FLOAT3;
    tri.vertexStrideInBytes          = sizeof(float3);
    tri.numVertices                  = vertexCount;
    tri.vertexBuffers                = &buffers.positions;

    tri.indexFormat                  = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    tri.indexStrideInBytes           = sizeof(uint3);
    tri.numIndexTriplets             = triangleCount;
    tri.indexBuffer                  = buffers.indices;

    tri.flags                        = buildFlags;
    tri.numSbtRecords                = 1;
    tri.sbtIndexOffsetBuffer         = 0;

    OptixAccelBuildOptions opts = {};
    opts.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION
                    | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    opts.operation  = OPTIX_BUILD_OPERATION_BUILD;

    // Query memory requirements
    OptixAccelBufferSizes sizes = {};
    OPTIX_CHECK(optixAccelComputeMemoryUsage(ctx, &opts, &buildInput, 1, &sizes));

    CUdeviceptr tempBuffer = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&tempBuffer),
                           sizes.tempSizeInBytes));

    // Device slot to receive the compacted AS size
    CUdeviceptr compactedSizeSlot = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&compactedSizeSlot),
                           sizeof(uint64_t)));

    OptixAccelEmitDesc emitDesc = {};
    emitDesc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = compactedSizeSlot;

    // Uncompacted build
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&buffers.outputAS),
                           sizes.outputSizeInBytes));

    OPTIX_CHECK(optixAccelBuild(
        ctx, nullptr,
        &opts, &buildInput, 1,
        tempBuffer,       sizes.tempSizeInBytes,
        buffers.outputAS, sizes.outputSizeInBytes,
        &buffers.blas,
        &emitDesc, 1));

    CUDA_CHECK(cudaDeviceSynchronize());

    // Read back compacted size and compact if it is smaller
    uint64_t compactedSize = 0;
    CUDA_CHECK(cudaMemcpy(&compactedSize,
                           reinterpret_cast<void*>(compactedSizeSlot),
                           sizeof(uint64_t), cudaMemcpyDeviceToHost));

    if (compactedSize < sizes.outputSizeInBytes)
    {
        CUdeviceptr compactedAS = 0;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&compactedAS),
                               compactedSize));

        OPTIX_CHECK(optixAccelCompact(ctx, nullptr,
                                      buffers.blas,
                                      compactedAS, compactedSize,
                                      &buffers.blas));

        CUDA_CHECK(cudaDeviceSynchronize());

        cudaFree(reinterpret_cast<void*>(buffers.outputAS));
        buffers.outputAS = compactedAS;
    }

    cudaFree(reinterpret_cast<void*>(tempBuffer));
    cudaFree(reinterpret_cast<void*>(compactedSizeSlot));
}

// ─── Accel::build ─────────────────────────────────────────────────────────────

void Accel::build(OptixDeviceContext ctx, const Scene& scene)
{
    destroy();

    const auto& meshes = scene.meshes();
    if (meshes.empty())
    {
        return;
    }

    m_meshBuffers.resize(meshes.size());

    // ── BLAS per mesh ─────────────────────────────────────────────────────────
    for (size_t i = 0; i < meshes.size(); ++i)
    {
        const Mesh& mesh = meshes[i];
        MeshBuffers& mb  = m_meshBuffers[i];

        // Upload vertex positions to device
        const size_t posByteSize = mesh.positions.size() * sizeof(float3);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mb.positions), posByteSize));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(mb.positions),
                               mesh.positions.data(),
                               posByteSize, cudaMemcpyHostToDevice));

        // Upload triangle indices to device
        const size_t idxByteSize = mesh.indices.size() * sizeof(uint3);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mb.indices), idxByteSize));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(mb.indices),
                               mesh.indices.data(),
                               idxByteSize, cudaMemcpyHostToDevice));

        buildBlas(ctx, mb,
                  static_cast<unsigned int>(mesh.positions.size()),
                  static_cast<unsigned int>(mesh.indices.size()));
    }

    // ── TLAS — one instance per mesh, identity transforms ─────────────────────
    std::vector<OptixInstance> instances(meshes.size());

    for (size_t i = 0; i < meshes.size(); ++i)
    {
        OptixInstance& inst = instances[i];
        std::memset(&inst, 0, sizeof(inst));

        // Row-major 3×4 identity transform (last row [0,0,0,1] is implicit)
        inst.transform[0]  = 1.f;
        inst.transform[5]  = 1.f;
        inst.transform[10] = 1.f;

        inst.instanceId        = static_cast<unsigned int>(i);
        inst.sbtOffset         = static_cast<unsigned int>(i);  // one hit record per mesh
        inst.visibilityMask    = 0xFF;
        inst.flags             = OPTIX_INSTANCE_FLAG_NONE;
        inst.traversableHandle = m_meshBuffers[i].blas;
    }

    const size_t instByteSize = instances.size() * sizeof(OptixInstance);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_instanceBuffer), instByteSize));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_instanceBuffer),
                           instances.data(), instByteSize, cudaMemcpyHostToDevice));

    OptixBuildInput tlasInput                          = {};
    tlasInput.type                                     = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    tlasInput.instanceArray.instances                  = m_instanceBuffer;
    tlasInput.instanceArray.numInstances               =
        static_cast<unsigned int>(instances.size());

    OptixAccelBuildOptions tlasOpts = {};
    tlasOpts.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    tlasOpts.operation  = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes tlasSizes = {};
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        ctx, &tlasOpts, &tlasInput, 1, &tlasSizes));

    CUdeviceptr tlasTempBuffer = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&tlasTempBuffer),
                           tlasSizes.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_tlasOutputBuffer),
                           tlasSizes.outputSizeInBytes));

    OPTIX_CHECK(optixAccelBuild(
        ctx, nullptr,
        &tlasOpts, &tlasInput, 1,
        tlasTempBuffer,   tlasSizes.tempSizeInBytes,
        m_tlasOutputBuffer, tlasSizes.outputSizeInBytes,
        &m_tlas, nullptr, 0));

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(reinterpret_cast<void*>(tlasTempBuffer));
}
