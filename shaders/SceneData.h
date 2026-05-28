#ifndef OPTIX_RAYTRACER_SCENE_DATA_H
#define OPTIX_RAYTRACER_SCENE_DATA_H

#include <cuda_runtime.h>  // float3, float2, uint3, cudaTextureObject_t

// GPU view of one mesh — placed in the SBT hit group record per mesh instance.
// In a hit shader: const MeshData* mesh = (const MeshData*)optixGetSbtDataPointer();
struct MeshData
{
    const float3* positions;
    const float3* normals;
    const float2* uvs;
    const uint3*  indices;
    int           materialIndex;  // into the per-launch MaterialData array
};

// Flat POD material — used on the host side and copied directly into SBT records
// or a device-side MaterialData array. No std::string; names live in Scene on the host.
struct MaterialData
{
    float3 albedo        = { 1.f, 1.f, 1.f };
    int    albedoTexture = -1;    // index into device texture array; -1 = no texture

    float  roughness     = 0.5f;
    float  metallic      = 0.f;

    float3 emission      = { 0.f, 0.f, 0.f };
    float  emissionScale = 1.f;
};

#endif // OPTIX_RAYTRACER_SCENE_DATA_H
