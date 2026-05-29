#ifndef OPTIX_RAYTRACER_ENVMAP_H
#define OPTIX_RAYTRACER_ENVMAP_H

#include <cuda_runtime.h>
#include <string>

// Loads an EXR from disk and uploads it as a CUDA float4 2D texture
// suitable for equirectangular (lat-long) environment map sampling.
// Throws std::runtime_error on failure.
struct EnvMapData
{
    cudaArray_t         array  = nullptr;
    cudaTextureObject_t tex    = 0;
    int                 width  = 0;
    int                 height = 0;
};

EnvMapData loadEnvMapEXR(const std::string& path);
void       freeEnvMap(EnvMapData& data);

#endif // OPTIX_RAYTRACER_ENVMAP_H
