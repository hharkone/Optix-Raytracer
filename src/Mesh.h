#pragma once

#include <cuda_runtime.h>  // float3, float2, uint3
#include <vector>
#include <string>

// Host-side mesh — owns vertex attribute arrays on the CPU.
// Separate arrays (not interleaved) so positions.data() passes directly to
// OptixBuildInputTriangleArray::vertexBuffers without stride arithmetic.
struct Mesh
{
    std::vector<float3> positions;
    std::vector<float3> normals;
    std::vector<float2> uvs;
    std::vector<uint3>  indices;    // one uint3 per triangle

    int         materialIndex = -1;
    std::string name;
};
