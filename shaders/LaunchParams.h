// LaunchParams.h — shared between host (C++) and device (CUDA) code.
// Included by both Application.cpp and devicePrograms.cu.
// Must use only types valid in both C++17 and CUDA device code:
// no STL, no windows.h, no host-only headers.
#ifndef OPTIX_RAYTRACER_LAUNCH_PARAMS_H
#define OPTIX_RAYTRACER_LAUNCH_PARAMS_H

#include <optix.h>         // OptixTraversableHandle
#include <cuda_runtime.h>  // uchar4, uint2

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
};

#endif // OPTIX_RAYTRACER_LAUNCH_PARAMS_H
