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

// stb_image: the implementation is compiled in SceneLoader.cpp via
// #define STB_IMAGE_IMPLEMENTATION before #include <tiny_gltf.h>.
// We only need the function declarations here.
#include <stb_image.h>

#include "Texture.h"

#include <cmath>     // sinf, fmaxf
#include <cstring>   // std::memcpy
#include <stdexcept>
#include <string>
#include <vector>

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

// ─── Move semantics ───────────────────────────────────────────────────────────

Texture::Texture(Texture&& other) noexcept
    : format         (other.format)
    , pixels         (std::move(other.pixels))
    , width          (other.width)
    , height         (other.height)
    , name           (std::move(other.name))
    , gpuTex         (other.gpuTex)
    , cdfMarginal    (other.cdfMarginal)
    , cdfConditional (other.cdfConditional)
    , gpuArray       (other.gpuArray)
{
    // Null the source so its destructor doesn't free the GPU resources we just took.
    other.gpuArray       = nullptr;
    other.gpuTex         = 0;
    other.cdfMarginal    = 0;
    other.cdfConditional = 0;
}

Texture& Texture::operator=(Texture&& other) noexcept
{
    if (this != &other)
    {
        free();  // release whatever we currently own
        format         = other.format;
        pixels         = std::move(other.pixels);
        width          = other.width;
        height         = other.height;
        name           = std::move(other.name);
        gpuArray       = other.gpuArray;
        gpuTex         = other.gpuTex;
        cdfMarginal    = other.cdfMarginal;
        cdfConditional = other.cdfConditional;
        other.gpuArray       = nullptr;
        other.gpuTex         = 0;
        other.cdfMarginal    = 0;
        other.cdfConditional = 0;
    }
    return *this;
}

// ─── Texture::uploadToGpu ────────────────────────────────────────────────────

void Texture::uploadToGpu()
{
    cudaChannelFormatDesc fmt;
    cudaTextureDesc       texDesc = {};

    if (format == PixelFormat::RGBA32F)
    {
        fmt = cudaCreateChannelDesc<float4>();
        texDesc.addressMode[0]   = cudaAddressModeWrap;
        texDesc.addressMode[1]   = cudaAddressModeClamp;
        texDesc.filterMode       = cudaFilterModeLinear;
        texDesc.readMode         = cudaReadModeElementType;
        texDesc.normalizedCoords = 1;
    }
    else  // RGBA8
    {
        fmt = cudaCreateChannelDesc<uchar4>();
        texDesc.addressMode[0]   = cudaAddressModeWrap;
        texDesc.addressMode[1]   = cudaAddressModeWrap;
        texDesc.filterMode       = cudaFilterModeLinear;
        texDesc.readMode         = cudaReadModeNormalizedFloat;
        texDesc.normalizedCoords = 1;
    }

    const size_t rowBytes = static_cast<size_t>(width)
                          * (format == PixelFormat::RGBA32F ? sizeof(float4) : sizeof(uchar4));

    CUDA_CHECK_TEXTURE(cudaMallocArray(&gpuArray, &fmt, width, height));
    CUDA_CHECK_TEXTURE(cudaMemcpy2DToArray(
        gpuArray, 0, 0,
        pixels.data(), rowBytes, rowBytes, height,
        cudaMemcpyHostToDevice));

    cudaResourceDesc resDesc       = {};
    resDesc.resType                = cudaResourceTypeArray;
    resDesc.res.array.array        = gpuArray;

    CUDA_CHECK_TEXTURE(cudaCreateTextureObject(&gpuTex, &resDesc, &texDesc, nullptr));
}

// ─── Texture::free ───────────────────────────────────────────────────────────

void Texture::free()
{
    if (cdfConditional) { cudaFree(reinterpret_cast<void*>(cdfConditional)); cdfConditional = 0; }
    if (cdfMarginal)    { cudaFree(reinterpret_cast<void*>(cdfMarginal));    cdfMarginal    = 0; }
    if (gpuTex)         { cudaDestroyTextureObject(gpuTex);                  gpuTex         = 0; }
    if (gpuArray)       { cudaFreeArray(gpuArray);                           gpuArray       = nullptr; }
}

// ─── Texture::loadEXR ────────────────────────────────────────────────────────

bool Texture::loadEXR(const std::string& path, std::string& outError)
{
    float*      rgba   = nullptr;
    int         w      = 0;
    int         h      = 0;
    const char* err    = nullptr;

    const int ret = LoadEXR(&rgba, &w, &h, path.c_str(), &err);
    if (ret != TINYEXR_SUCCESS)
    {
        outError = err ? std::string(err) : "unknown EXR error";
        FreeEXRErrorMessage(err);
        return false;
    }

    const size_t byteCount = static_cast<size_t>(w) * h * sizeof(float4);
    pixels.resize(byteCount);
    std::memcpy(pixels.data(), rgba, byteCount);
    ::free(rgba);  // tinyexr uses malloc; :: avoids ambiguity with Texture::free

    format = PixelFormat::RGBA32F;
    width  = w;
    height = h;

    return true;
}

// ─── Texture::loadImage ──────────────────────────────────────────────────────

bool Texture::loadImage(const std::string& path, std::string& outError)
{
    int w = 0, h = 0, channels = 0;
    // Request 4 channels (RGBA) regardless of source format.
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &channels, 4);
    if (!data)
    {
        outError = stbi_failure_reason() ? stbi_failure_reason() : "unknown stb_image error";
        return false;
    }

    const size_t byteCount = static_cast<size_t>(w) * h * 4;
    pixels.assign(data, data + byteCount);
    stbi_image_free(data);

    format = PixelFormat::RGBA8;
    width  = w;
    height = h;
    return true;
}

// ─── buildCdf ─────────────────────────────────────────────────────────────────

void Texture::buildCdf()
{
    if (!isHdr() || pixels.empty())
        return;

    const int    W   = width;
    const int    H   = height;
    const float* src = floatPixels();  // RGBA32F, row-major

    // ── Per-pixel weight = luminance(RGB) × sin(θ) ───────────────────────────
    std::vector<float> weights(static_cast<size_t>(H) * W);
    std::vector<float> rowSums(H, 0.f);

    for (int j = 0; j < H; ++j)
    {
        const float theta    = (j + 0.5f) / static_cast<float>(H) * 3.14159265358979f;
        const float sinTheta = sinf(theta);

        for (int i = 0; i < W; ++i)
        {
            const float* p   = src + (j * W + i) * 4;
            const float  lum = 0.2126f * p[0] + 0.7152f * p[1] + 0.0722f * p[2];
            const float  w   = fmaxf(0.f, lum) * sinTheta;
            weights[j * W + i] = w;
            rowSums[j] += w;
        }
    }

    // ── Conditional CDF per row (prefix-sum, normalised to [0,1]) ────────────
    std::vector<float> conditionalCdf(static_cast<size_t>(H) * W);

    for (int j = 0; j < H; ++j)
    {
        const float invRow = (rowSums[j] > 0.f) ? 1.f / rowSums[j] : 0.f;
        float running = 0.f;
        for (int i = 0; i < W; ++i)
        {
            running += weights[j * W + i];
            conditionalCdf[j * W + i] = (rowSums[j] > 0.f)
                ? running * invRow
                : (i + 1.f) / static_cast<float>(W);
        }
        conditionalCdf[j * W + (W - 1)] = 1.f;  // prevent binary-search overrun
    }

    // ── Marginal CDF (prefix-sum over rows, normalised to [0,1]) ─────────────
    std::vector<float> marginalCdf(H);
    float totalWeight = 0.f;
    for (int j = 0; j < H; ++j)
        totalWeight += rowSums[j];

    {
        float running = 0.f;
        for (int j = 0; j < H; ++j)
        {
            running += rowSums[j];
            marginalCdf[j] = (totalWeight > 0.f)
                ? running / totalWeight
                : (j + 1.f) / static_cast<float>(H);
        }
        marginalCdf[H - 1] = 1.f;
    }

    // ── Upload to device ──────────────────────────────────────────────────────
    const size_t marginalBytes    = static_cast<size_t>(H) * sizeof(float);
    const size_t conditionalBytes = static_cast<size_t>(H) * W * sizeof(float);

    CUDA_CHECK_TEXTURE(cudaMalloc(reinterpret_cast<void**>(&cdfMarginal),    marginalBytes));
    CUDA_CHECK_TEXTURE(cudaMalloc(reinterpret_cast<void**>(&cdfConditional), conditionalBytes));

    CUDA_CHECK_TEXTURE(cudaMemcpy(reinterpret_cast<void*>(cdfMarginal),
                                   marginalCdf.data(),    marginalBytes,    cudaMemcpyHostToDevice));
    CUDA_CHECK_TEXTURE(cudaMemcpy(reinterpret_cast<void*>(cdfConditional),
                                   conditionalCdf.data(), conditionalBytes, cudaMemcpyHostToDevice));
}
