// LaunchParams.h — shared between host (C++) and device (CUDA) code.
// Included by both Application.cpp and devicePrograms.cu.
// Must use only types valid in both C++17 and CUDA device code:
// no STL, no windows.h, no host-only headers.
#ifndef OPTIX_RAYTRACER_LAUNCH_PARAMS_H
#define OPTIX_RAYTRACER_LAUNCH_PARAMS_H

#include <optix.h>         // OptixTraversableHandle
#include <cuda_runtime.h>  // uchar4, uint2
#include "SceneData.h"     // MaterialData

struct LaunchParams
{
    uchar4*                colorBuffer;  // device pointer — RGBA8 output
    uint2                  fbSize;       // { width, height } in pixels
    OptixTraversableHandle traversable;  // top-level IAS; 0 = no scene loaded

    // Pinhole camera basis vectors computed on the host each frame.
    // dir = normalize(ndc.x * U + ndc.y * V + W)  where ndc in [-1,1]
    float3 eye;  // camera position in world space
    float3 U;    // right direction * tan(halfFovH)
    float3 V;    // up direction   * tan(halfFovV)
    float3 W;    // forward direction (unit length, pointing into scene)

    // Equirectangular (lat-long) environment map. 0 = not loaded; raygen falls
    // back to the procedural sky gradient when this is 0.
    cudaTextureObject_t envMap;
    float               envMapRotation;  // azimuth offset in radians (Shift+RMB drag)
    float               envExposure;     // exposure in EV stops; applied as 2^n to env radiance

    // Thin-lens depth of field. lensRadius = 0 → pinhole (no DoF).
    // lensRadius is in world units (1 unit = 1 m); focusDistance is in world units.
    float lensRadius;     // half aperture = focalLength_mm / (2 × fStop × 1000)
    float focusDistance;  // distance from eye to focal plane along W
    float bokehEdgeBias;  // [0, 1]: 0 = uniform disk, 1 = pure rim ring

    // Per-pixel HDR accumulation buffer (float4, w unused).  The raygen adds
    // one sample per launch; the display value is accumBuffer[i] / sampleIndex.
    float4*  accumBuffer;
    uint32_t sampleIndex;  // number of samples already accumulated

    // Denoiser guide layers — null when denoiser is disabled.
    float4* normalBuffer;  // first-hit world-space shading normal (written every frame)
    float4* albedoBuffer;  // first-hit material albedo          (written every frame)
    float4* hdrBuffer;     // running HDR average = accumBuffer / sampleIndex

    // Scene materials — device pointer to a MaterialData array, indexed by
    // MeshData::materialIndex.  Null when no scene is loaded.
    const MaterialData* materials;
};

#endif // OPTIX_RAYTRACER_LAUNCH_PARAMS_H
