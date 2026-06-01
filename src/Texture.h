#ifndef OPTIX_RAYTRACER_TEXTURE_H
#define OPTIX_RAYTRACER_TEXTURE_H

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <string>

enum class PixelFormat { RGBA8, RGBA32F };

// Unified host+GPU texture — covers both LDR (uint8_t) and HDR (float) images.
//
// Host pixels are stored as raw bytes regardless of format:
//   RGBA8   — pixels.size() == width * height * 4        (1 byte  per channel)
//   RGBA32F — pixels.size() == width * height * 4 * 4   (4 bytes per float channel)
//
// GPU state (gpuArray, gpuTex) is zero/null until uploadToGpu() is called.
struct Texture
{
    PixelFormat          format   = PixelFormat::RGBA8;
    std::vector<uint8_t> pixels;
    int                  width    = 0;
    int                  height   = 0;
    std::string          name;

    bool         isHdr()       const { return format == PixelFormat::RGBA32F; }
    const float* floatPixels() const
    {
        return reinterpret_cast<const float*>(pixels.data());
    }

    // GPU state — managed by uploadToGpu() / freeTexture() below.
    cudaArray_t         gpuArray = nullptr;
    cudaTextureObject_t gpuTex   = 0;
};

// ─── GPU lifecycle ────────────────────────────────────────────────────────────

// Upload tex.pixels to a CUDA 2D array and create a texture object.
// Addressing and read mode are chosen from tex.format:
//   RGBA8   — uchar4, wrap+wrap,  normalised float read  (LDR albedo style)
//   RGBA32F — float4, wrap+clamp, element-type read      (HDR envmap  style)
// Throws std::runtime_error on CUDA failure.
void uploadToGpu(Texture& tex);

// Destroy GPU resources; resets gpuArray and gpuTex to zero/null.
// Safe to call on a Texture that was never uploaded.
void freeTexture(Texture& tex);

// ─── EXR loader ───────────────────────────────────────────────────────────────

// Load an EXR from disk into tex.pixels (format = RGBA32F, host-side only).
// Does NOT upload to GPU — call uploadToGpu() afterwards.
// Returns false and writes outError on failure.
bool loadEXR(const std::string& path, Texture& tex, std::string& outError);

#endif // OPTIX_RAYTRACER_TEXTURE_H
