// EnvMap.cpp — EXR environment map loading.
//
// TINYEXR_IMPLEMENTATION must appear in exactly ONE translation unit.
// This file is that unit — do not define it elsewhere.
//
// TINYEXR_USE_MINIZ pulls in the bundled miniz single-header for zlib
// decompression so no external zlib install is required.
#define TINYEXR_IMPLEMENTATION
#define TINYEXR_USE_MINIZ  1
#define TINYEXR_USE_THREAD 1  // decode scanlines/tiles in parallel with std::thread
#include <tinyexr.h>

#include "EnvMap.h"

#include <stdexcept>
#include <string>

// ─── Helpers ─────────────────────────────────────────────────────────────────

#define CUDA_CHECK_ENVMAP(call)                                                  \
    do {                                                                         \
        cudaError_t rc = (call);                                                 \
        if (rc != cudaSuccess)                                                   \
            throw std::runtime_error(                                            \
                std::string("CUDA error in EnvMap: ")                           \
                + cudaGetErrorString(rc));                                       \
    } while (0)

// ─── loadEnvMapEXR ───────────────────────────────────────────────────────────

EnvMapData loadEnvMapEXR(const std::string& path)
{
    // ── Load EXR ──────────────────────────────────────────────────────────────
    // LoadEXR always returns 4 floats per pixel (RGBA); missing channels are
    // filled with 0 (RGB) or 1 (A).  The caller must free() the pointer.
    float*      rgba   = nullptr;
    int         width  = 0;
    int         height = 0;
    const char* err    = nullptr;

    const int ret = LoadEXR(&rgba, &width, &height, path.c_str(), &err);
    if (ret != TINYEXR_SUCCESS)
    {
        const std::string msg = err ? err : "unknown error";
        FreeEXRErrorMessage(err);
        throw std::runtime_error("Failed to load EXR '" + path + "': " + msg);
    }

    // ── Upload to CUDA 2D array (float4) ──────────────────────────────────────
    // Image rows are stored top-to-bottom; CUDA maps row 0 to v=0, which
    // paired with v = theta/pi gives v=0 at the zenith (dir.y=1). ✓
    const cudaChannelFormatDesc fmt = cudaCreateChannelDesc<float4>();
    cudaArray_t arr = nullptr;
    CUDA_CHECK_ENVMAP(cudaMallocArray(&arr, &fmt, width, height));

    const size_t rowBytes = static_cast<size_t>(width) * sizeof(float4);
    CUDA_CHECK_ENVMAP(cudaMemcpy2DToArray(
        arr, 0, 0,
        rgba, rowBytes, rowBytes, height,
        cudaMemcpyHostToDevice));

    free(rgba);  // tinyexr uses malloc

    // ── Create texture object ─────────────────────────────────────────────────
    cudaResourceDesc resDesc = {};
    resDesc.resType          = cudaResourceTypeArray;
    resDesc.res.array.array  = arr;

    cudaTextureDesc texDesc   = {};
    texDesc.addressMode[0]    = cudaAddressModeWrap;   // longitude — wraps
    texDesc.addressMode[1]    = cudaAddressModeClamp;  // latitude  — clamp at poles
    texDesc.filterMode        = cudaFilterModeLinear;
    texDesc.readMode          = cudaReadModeElementType;  // return raw float4
    texDesc.normalizedCoords  = 1;

    cudaTextureObject_t tex = 0;
    CUDA_CHECK_ENVMAP(cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr));

    return { arr, tex, width, height };
}

// ─── freeEnvMap ──────────────────────────────────────────────────────────────

void freeEnvMap(EnvMapData& data)
{
    if (data.tex)   { cudaDestroyTextureObject(data.tex); data.tex   = 0; }
    if (data.array) { cudaFreeArray(data.array);          data.array = nullptr; }
    data.width = data.height = 0;
}
