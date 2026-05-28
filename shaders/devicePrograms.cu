// devicePrograms.cu — OptiX device programs compiled to PTX.
// Phase 1 stub: raygen writes a UV gradient, miss is a no-op.
// No actual ray tracing yet — just enough to validate the pipeline compiles.

#include <optix.h>
#include "LaunchParams.h"

// Launch parameters uploaded by the host via cudaMemcpyToSymbol.
// Must have C linkage and be declared __constant__.
extern "C"
{
    __constant__ LaunchParams optixLaunchParams;
}

// ─── Ray-generation program ───────────────────────────────────────────────────
extern "C" __global__ void __raygen__renderFrame()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    // UV gradient — proves the pipeline runs; replaced by actual shading later
    const float u = static_cast<float>(idx.x) / static_cast<float>(dim.x);
    const float v = static_cast<float>(idx.y) / static_cast<float>(dim.y);

    const uchar4 color = {
        static_cast<unsigned char>(u * 255.f),
        static_cast<unsigned char>(v * 255.f),
        static_cast<unsigned char>(128),
        255
    };

    const unsigned int fbIndex = idx.x + idx.y * optixLaunchParams.fbSize.x;
    optixLaunchParams.colorBuffer[fbIndex] = color;
}

// ─── Miss program ─────────────────────────────────────────────────────────────
extern "C" __global__ void __miss__radiance()
{
    // Stub — no rays are traced in phase 1
}
