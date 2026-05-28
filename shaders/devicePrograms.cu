// devicePrograms.cu — OptiX device programs compiled to PTX.
//
// Implements a minimal primary-ray renderer:
//   raygen  — fires one ray per pixel from the pinhole camera
//   miss    — sky gradient (white horizon → blue zenith)
//   closesthit — world-space face normal visualization
//
// RGB is packed through three unsigned int payloads (values 0-255).

#include <optix.h>
#include "LaunchParams.h"
#include "SceneData.h"   // MeshData — SBT hit record data

extern "C"
{
    __constant__ LaunchParams optixLaunchParams;
}

// ─── Device math helpers ──────────────────────────────────────────────────────
// The public CUDA SDK headers do not expose cross / normalize / dot for float3
// in device code — we define them here as file-local device functions.

static __forceinline__ __device__
float devDot(float3 a, float3 b)
{
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

static __forceinline__ __device__
float3 devNormalize(float3 v)
{
    const float ilen = rsqrtf(devDot(v, v));
    return make_float3(v.x*ilen, v.y*ilen, v.z*ilen);
}

static __forceinline__ __device__
float3 devCross(float3 a, float3 b)
{
    return make_float3(
        a.y*b.z - a.z*b.y,
        a.z*b.x - a.x*b.z,
        a.x*b.y - a.y*b.x);
}

static __forceinline__ __device__
float devClamp01(float x)
{
    return x < 0.f ? 0.f : (x > 1.f ? 1.f : x);
}

// ─── Ray-generation program ───────────────────────────────────────────────────

extern "C" __global__ void __raygen__renderFrame()
{
    const uint3         idx    = optixGetLaunchIndex();
    const uint3         dim    = optixGetLaunchDimensions();
    const unsigned int  fbIdx  = idx.y * dim.x + idx.x;

    if (optixLaunchParams.traversable == 0)
    {
        // No scene loaded: UV gradient placeholder
        const unsigned char r = (unsigned char)(255u * idx.x / dim.x);
        const unsigned char g = (unsigned char)(255u * idx.y / dim.y);
        optixLaunchParams.colorBuffer[fbIdx] = make_uchar4(r, g, 128u, 255u);
        return;
    }

    // Map pixel centre to NDC [-1, 1]
    const float nx = ((float)idx.x + 0.5f) / (float)dim.x * 2.f - 1.f;
    const float ny = ((float)idx.y + 0.5f) / (float)dim.y * 2.f - 1.f;

    // Ray direction: linear combination of camera basis vectors
    const float3 rawDir = make_float3(
        nx * optixLaunchParams.U.x + ny * optixLaunchParams.V.x + optixLaunchParams.W.x,
        nx * optixLaunchParams.U.y + ny * optixLaunchParams.V.y + optixLaunchParams.W.y,
        nx * optixLaunchParams.U.z + ny * optixLaunchParams.V.z + optixLaunchParams.W.z);
    const float3 dir = devNormalize(rawDir);

    unsigned int p0 = 0, p1 = 0, p2 = 0;
    optixTrace(
        optixLaunchParams.traversable,
        optixLaunchParams.eye, dir,
        1e-4f,  // tmin
        1e20f,  // tmax
        0.f,    // ray time (not used)
        OptixVisibilityMask(0xFF),
        OPTIX_RAY_FLAG_NONE,
        0,      // SBT offset
        1,      // SBT stride
        0,      // miss SBT index
        p0, p1, p2);

    optixLaunchParams.colorBuffer[fbIdx] =
        make_uchar4((unsigned char)p0, (unsigned char)p1, (unsigned char)p2, 255u);
}

// ─── Miss program ─────────────────────────────────────────────────────────────

extern "C" __global__ void __miss__radiance()
{
    // Sky gradient: warm white at the horizon, sky blue at the zenith.
    const float3 dir = optixGetWorldRayDirection();
    const float  t   = devClamp01(0.5f * (devNormalize(dir).y + 1.f));

    // Blend: t=0 → (1, 1, 1) white; t=1 → (0.3, 0.5, 1.0) sky blue
    const float r = 1.f - 0.7f * t;
    const float g = 1.f - 0.5f * t;
    const float b = 1.f;

    optixSetPayload_0((unsigned int)(r * 255.f));
    optixSetPayload_1((unsigned int)(g * 255.f));
    optixSetPayload_2((unsigned int)(b * 255.f));
}

// ─── Closest-hit program ──────────────────────────────────────────────────────

extern "C" __global__ void __closesthit__radiance()
{
    // Retrieve mesh data from the SBT hit record
    const MeshData& mesh = *reinterpret_cast<const MeshData*>(optixGetSbtDataPointer());

    const uint3  tri = mesh.indices[optixGetPrimitiveIndex()];
    const float3 v0  = mesh.positions[tri.x];
    const float3 v1  = mesh.positions[tri.y];
    const float3 v2  = mesh.positions[tri.z];

    // Compute face normal in object space from the two edge vectors
    const float3 e0 = make_float3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
    const float3 e1 = make_float3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);
    const float3 objNormal = devNormalize(devCross(e0, e1));

    // Transform to world space using the instance's inverse-transpose transform
    const float3 worldNormal = devNormalize(
        optixTransformNormalFromObjectToWorldSpace(objNormal));

    // Remap [-1, 1] → [0, 255] for display as an RGB colour
    optixSetPayload_0((unsigned int)((worldNormal.x * 0.5f + 0.5f) * 255.f));
    optixSetPayload_1((unsigned int)((worldNormal.y * 0.5f + 0.5f) * 255.f));
    optixSetPayload_2((unsigned int)((worldNormal.z * 0.5f + 0.5f) * 255.f));
}
