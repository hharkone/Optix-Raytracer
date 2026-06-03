// Texture.cpp — unified LDR/HDR texture: GPU upload and EXR loading.
//
// TINYEXR_IMPLEMENTATION must appear in exactly ONE translation unit.
// This file is that unit — do not define it elsewhere.
//
// TINYEXR_USE_MINIZ pulls in the bundled miniz single-header for zlib
// decompression so no external zlib install is required.
// TINYEXR_USE_THREAD enables parallel scanline decoding via std::thread.
#define TINYEXR_IMPLEMENTATION
#define TINYEXR_USE_MINIZ  1
#define TINYEXR_USE_THREAD 1
#include <tinyexr.h>

#include "Texture.h"

#include <cstring>   // std::memcpy
#include <stdexcept>
#include <string>

// ─── Helpers ─────────────────────────────────────────────────────────────────

#define CUDA_CHECK_TEXTURE(call)                                                 \
    do {                                                                         \
        cudaError_t rc = (call);                                                 \
        if (rc != cudaSuccess)                                                   \
        {                                                                        \
            throw std::runtime_error(                                            \
                std::string("CUDA error in Texture: ")                          \
                + cudaGetErrorString(rc));                                       \
        }                                                                        \
    } while (0)

// ─── uploadToGpu ─────────────────────────────────────────────────────────────

void uploadToGpu(Texture& tex)
{
    cudaChannelFormatDesc fmt;
    cudaTextureDesc       texDesc = {};

    if (tex.format == PixelFormat::RGBA32F)
    {
        // HDR environment map: raw float4 read, wrap longitude / clamp latitude
        fmt = cudaCreateChannelDesc<float4>();
        texDesc.addressMode[0]   = cudaAddressModeWrap;   // U — longitude wraps
        texDesc.addressMode[1]   = cudaAddressModeClamp;  // V — clamp at poles
        texDesc.filterMode       = cudaFilterModeLinear;
        texDesc.readMode         = cudaReadModeElementType;  // return raw float4
        texDesc.normalizedCoords = 1;
    }
    else  // RGBA8
    {
        // LDR albedo texture: read as normalised [0, 1] float4
        fmt = cudaCreateChannelDesc<uchar4>();
        texDesc.addressMode[0]   = cudaAddressModeWrap;
        texDesc.addressMode[1]   = cudaAddressModeWrap;
        texDesc.filterMode       = cudaFilterModeLinear;
        texDesc.readMode         = cudaReadModeNormalizedFloat;
        texDesc.normalizedCoords = 1;
    }

    const size_t rowBytes = static_cast<size_t>(tex.width)
                          * (tex.format == PixelFormat::RGBA32F
                                 ? sizeof(float4)
                                 : sizeof(uchar4));

    CUDA_CHECK_TEXTURE(cudaMallocArray(&tex.gpuArray, &fmt, tex.width, tex.height));
    CUDA_CHECK_TEXTURE(cudaMemcpy2DToArray(
        tex.gpuArray, 0, 0,
        tex.pixels.data(), rowBytes, rowBytes, tex.height,
        cudaMemcpyHostToDevice));

    cudaResourceDesc resDesc        = {};
    resDesc.resType                 = cudaResourceTypeArray;
    resDesc.res.array.array         = tex.gpuArray;

    CUDA_CHECK_TEXTURE(cudaCreateTextureObject(&tex.gpuTex, &resDesc, &texDesc, nullptr));
}

// ─── freeTexture ─────────────────────────────────────────────────────────────

void freeTexture(Texture& tex)
{
    if (tex.cdfConditional) { cudaFree(reinterpret_cast<void*>(tex.cdfConditional)); tex.cdfConditional = 0; }
    if (tex.cdfMarginal)    { cudaFree(reinterpret_cast<void*>(tex.cdfMarginal));    tex.cdfMarginal    = 0; }
    if (tex.gpuTex)         { cudaDestroyTextureObject(tex.gpuTex);                  tex.gpuTex         = 0; }
    if (tex.gpuArray)       { cudaFreeArray(tex.gpuArray);                           tex.gpuArray       = nullptr; }
}

// ─── loadEXR ─────────────────────────────────────────────────────────────────

bool loadEXR(const std::string& path, Texture& tex, std::string& outError)
{
    float*      rgba   = nullptr;
    int         width  = 0;
    int         height = 0;
    const char* err    = nullptr;

    const int ret = LoadEXR(&rgba, &width, &height, path.c_str(), &err);
    if (ret != TINYEXR_SUCCESS)
    {
        outError = err ? std::string(err) : "unknown EXR error";
        FreeEXRErrorMessage(err);
        return false;
    }

    // LoadEXR always returns 4 floats per pixel (RGBA); copy into raw byte storage.
    const size_t byteCount = static_cast<size_t>(width) * height * sizeof(float4);
    tex.pixels.resize(byteCount);
    std::memcpy(tex.pixels.data(), rgba, byteCount);
    free(rgba);  // tinyexr uses malloc

    tex.format = PixelFormat::RGBA32F;
    tex.width  = width;
    tex.height = height;

    return true;
}
