#ifndef OPTIX_RAYTRACER_TEXTURE_H
#define OPTIX_RAYTRACER_TEXTURE_H

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <string>

// CUdeviceptr is unsigned long long on all 64-bit CUDA targets; use the
// same underlying type here so Texture.h stays free of the CUDA driver API.
using CdfDevicePtr = std::uint64_t;

enum class PixelFormat { RGBA8, RGBA32F };

// Unified host+GPU texture — covers both LDR (uint8_t) and HDR (float) images.
//
// Host pixels are stored as raw bytes regardless of format:
//   RGBA8   — pixels.size() == width * height * 4        (1 byte  per channel)
//   RGBA32F — pixels.size() == width * height * 4 * 4   (4 bytes per float channel)
//
// GPU state is managed by the class; the destructor releases it automatically.
// Texture is non-copyable (GPU handles cannot be shared) but movable, so it
// can be stored in std::vector and passed by std::move.
class Texture
{
public:
    // ── Host-side data (public for direct construction in SceneLoader) ────────
    PixelFormat          format   = PixelFormat::RGBA8;
    std::vector<uint8_t> pixels;
    int                  width    = 0;
    int                  height   = 0;
    std::string          name;

    // ── GPU state (public for use in LaunchParams / UI checks) ───────────────
    cudaTextureObject_t gpuTex           = 0;        // 0 = not uploaded
    CdfDevicePtr        cdfMarginal      = 0;        // device float[height]
    CdfDevicePtr        cdfConditional   = 0;        // device float[height × width]

    // ── Lifetime ──────────────────────────────────────────────────────────────
    Texture()  = default;
    ~Texture() { free(); }

    Texture(const Texture&)            = delete;
    Texture& operator=(const Texture&) = delete;

    Texture(Texture&& other) noexcept;
    Texture& operator=(Texture&& other) noexcept;

    // ── Host helpers ──────────────────────────────────────────────────────────
    bool         isHdr()       const { return format == PixelFormat::RGBA32F; }
    const float* floatPixels() const { return reinterpret_cast<const float*>(pixels.data()); }

    // ── GPU operations ────────────────────────────────────────────────────────

    // Upload pixels to a CUDA 2D array and create a texture object.
    // Addressing and read mode are chosen from format:
    //   RGBA8   — uchar4, wrap+wrap,  normalised float read  (LDR albedo style)
    //   RGBA32F — float4, wrap+clamp, element-type read      (HDR envmap  style)
    // Throws std::runtime_error on CUDA failure.
    void uploadToGpu();

    // Release GPU resources (array, texture object, CDF buffers).
    // Safe to call on a Texture that was never uploaded or already freed.
    void free();

    // Build the 2-D luminance CDF for HDRI importance sampling.
    // Must be called after loadEXR() + uploadToGpu().
    // No-op if the texture is not HDR or has no pixel data.
    void buildCdf();

    // ── EXR loader ────────────────────────────────────────────────────────────

    // Load an EXR from disk into pixels (format = RGBA32F, host-side only).
    // Does NOT upload to GPU — call uploadToGpu() afterwards.
    // Returns false and writes outError on failure.
    bool loadEXR(const std::string& path, std::string& outError);

private:
    cudaArray_t gpuArray = nullptr;
};

#endif // OPTIX_RAYTRACER_TEXTURE_H
