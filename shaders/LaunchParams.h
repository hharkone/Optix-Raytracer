// LaunchParams.h — shared between host (C++) and device (CUDA) code.
// Included by both Application.cpp and devicePrograms.cu.
// Must use only types valid in both C++17 and CUDA device code:
// no STL, no windows.h, no host-only headers.
#pragma once

#include <optix.h>         // OptixTraversableHandle
#include <cuda_runtime.h>  // uchar4, uint2

struct LaunchParams {
    uchar4*               colorBuffer;   // device pointer — RGBA8 output
    uint2                 fbSize;        // { width, height } in pixels
    OptixTraversableHandle traversable;  // top-level IAS (populated later)
};
