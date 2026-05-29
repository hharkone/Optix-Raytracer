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

static __forceinline__ __device__
float devMix(float a, float b, float x)
{
    return b * x + a * (1.0f - x);
}

static __forceinline__ __device__
float3 devMix(float3 a, float3 b, float x)
{
    return make_float3(
        devMix(a.x, b.x, x),
        devMix(a.y, b.y, x),
        devMix(a.z, b.z, x)
        );
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
    const float3 dir = devNormalize(optixGetWorldRayDirection());

    if (optixLaunchParams.envMap != 0)
    {
        // Equirectangular (lat-long) lookup.
        //
        // theta: polar angle from +Y.  dir.y=1 → theta=0 → v=0 (zenith, top
        //        of texture); dir.y=-1 → theta=π → v=1 (nadir, bottom).
        // phi:   azimuth. atan2(x,-z) so the default -Z camera forward maps to
        //        phi=0 → u=0.5, placing the front of the scene at image centre.
        const float kInvPi  = 0.31830988618f;
        const float kInv2Pi = 0.15915494309f;

        const float theta = acosf(fmaxf(-1.f, fminf(1.f, dir.y)));
        const float phi   = atan2f(dir.x, -dir.z);

        const float u = phi * kInv2Pi + 0.5f;   // [-π, π] → [0, 1] wrapping
        const float v = theta * kInvPi;         // [ 0, π] → [0, 1]

        const float4 s = tex2D<float4>(optixLaunchParams.envMap, u, v);

        // Reinhard tone-map: HDR [0, ∞) → [0, 1) — prevents hard clipping on
        // bright light sources while keeping the overall exposure natural.
        optixSetPayload_0((unsigned int)(devClamp01(s.x / (s.x + 1.f)) * 255.f));
        optixSetPayload_1((unsigned int)(devClamp01(s.y / (s.y + 1.f)) * 255.f));
        optixSetPayload_2((unsigned int)(devClamp01(s.z / (s.z + 1.f)) * 255.f));
    }
    else
    {
        // Procedural sky: warm white at the horizon, blue at the zenith.
        const float t = devClamp01(0.5f * (dir.y + 1.f));

        const float r = 1.f - 0.7f * t;
        const float g = 1.f - 0.5f * t;
        const float b = 1.f;

        optixSetPayload_0((unsigned int)(r * 255.f));
        optixSetPayload_1((unsigned int)(g * 255.f));
        optixSetPayload_2((unsigned int)(b * 255.f));
    }
}

// ─── Closest-hit program ──────────────────────────────────────────────────────

extern "C" __global__ void __closesthit__radiance()
{
    // Retrieve mesh data from the SBT hit record
    const MeshData& mesh = *reinterpret_cast<const MeshData*>(optixGetSbtDataPointer());

    const uint3  tri = mesh.indices[optixGetPrimitiveIndex()];
    const float3 v0  = mesh.normals[tri.x];
    const float3 v1  = mesh.normals[tri.y];
    const float3 v2  = mesh.normals[tri.z];
    //const float2 v3  = mesh.uvs[tri.z];

    const float2 coord = optixGetTriangleBarycentrics();

    const float3 n1 = devMix(v0, v1, coord.x);
    const float3 n2 = devMix(n1, v2, coord.y);
    const float3 objNormal = devNormalize(n2);

    // Transform to world space using the instance's inverse-transpose transform
    const float3 worldNormal = devNormalize(optixTransformNormalFromObjectToWorldSpace(objNormal));

    // Remap [-1, 1] → [0, 255] for display as an RGB colour
    optixSetPayload_0((unsigned int)((worldNormal.x * 0.5f + 0.5f) * 255.f));
    optixSetPayload_1((unsigned int)((worldNormal.y * 0.5f + 0.5f) * 255.f));
    optixSetPayload_2((unsigned int)((worldNormal.z * 0.5f + 0.5f) * 255.f));
}
