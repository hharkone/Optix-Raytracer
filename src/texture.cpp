// Texture.cpp — unified LDR/HDR/EXR texture: GPU upload and image loading.
//
// EXR loading uses OpenEXR — all compression types are supported:
//   NONE, RLE, ZIPS, ZIP, PIZ, PXR24, B44, B44A, DWAA, DWAB.
//
// HDR (Radiance RGBE .hdr) loading uses stb_image (stbi_loadf), which
// decodes the RGBE encoding to linear float values directly.
//
// stb_image: the implementation is compiled in SceneLoader.cpp via
// #define STB_IMAGE_IMPLEMENTATION before #include <tiny_gltf.h>.
// We only need the function declarations here.
#include <stb_image.h>

#include <ImfRgbaFile.h>
#include <ImfArray.h>
#include <ImfIO.h>

#include "texture.h"

#include <cmath>     // sinf, fmaxf
#include <cstdint>   // uint64_t
#include <cstring>   // std::memcpy
#include <stdexcept>
#include <string>
#include <vector>

// ─── Windows: UTF-8 file-open helpers ────────────────────────────────────────
//
// stbi_loadf() and Imf::RgbaInputFile(const char*) both call fopen() under the
// hood, which on MSVC uses the ANSI code page.  Paths that arrive as UTF-8
// (from NFD or std::filesystem::u8string()) are therefore misread when they
// contain characters outside that code page — Scandinavian ä/ö/å, etc.
//
// Fixes:
//  - loadHDR  → open with _wfopen, then hand the FILE* to stbi_loadf_from_file
//  - loadEXR  → use a custom Imf::IStream (WideFileStream) backed by _wfopen
#ifdef _WIN32
#   ifndef WIN32_LEAN_AND_MEAN
#       define WIN32_LEAN_AND_MEAN
#   endif
#   include <windows.h>

static std::wstring utf8ToWide(const std::string& utf8)
{
    if (utf8.empty())
    {
        return {};
    }
    const int n = MultiByteToWideChar(
        CP_UTF8, 0, utf8.c_str(), static_cast<int>(utf8.size()), nullptr, 0);
    std::wstring wide(n, L'\0');
    MultiByteToWideChar(
        CP_UTF8, 0, utf8.c_str(), static_cast<int>(utf8.size()), wide.data(), n);
    return wide;
}

// Thin Imf::IStream wrapper around a _wfopen handle.
// Enables OpenEXR to open files whose paths contain non-ANSI characters.
class WideFileStream final : public Imf::IStream
{
public:
    explicit WideFileStream(const std::string& utf8Path)
        : Imf::IStream(utf8Path.c_str())
        , m_file(_wfopen(utf8ToWide(utf8Path).c_str(), L"rb"))
    {
        if (!m_file)
        {
            throw std::runtime_error("Cannot open file: " + utf8Path);
        }
    }
    ~WideFileStream() override
    {
        if (m_file)
        {
            fclose(m_file);
        }
    }

    // Returns true while the stream is not at EOF, false on the last chunk.
    // Throws on a short read (genuine error or unexpected EOF mid-stream).
    bool read(char c[], int n) override
    {
        if (fread(c, 1, static_cast<size_t>(n), m_file) < static_cast<size_t>(n))
        {
            if (feof(m_file))
            {
                throw std::runtime_error("Unexpected end of file");
            }
            throw std::runtime_error("File read error");
        }
        return feof(m_file) == 0;
    }
    uint64_t tellg() override
    {
        return static_cast<uint64_t>(_ftelli64(m_file));
    }
    void seekg(uint64_t pos) override
    {
        _fseeki64(m_file, static_cast<__int64>(pos), SEEK_SET);
    }
    void clear() override { clearerr(m_file); }

private:
    FILE* m_file;
};
#endif // _WIN32

// ─── Helpers ─────────────────────────────────────────────────────────────────

// Sanitise every channel in an RGBA32F pixel buffer so all values are finite.
// Called once after loading; all downstream consumers (thumbnail generation,
// CDF building, CUDA texture upload) then see clean data automatically.
//
// EXR stores radiance in 16-bit half floats; values that exceed the maximum
// representable half (65504) are written as +inf.  Setting those to 0 makes
// bright areas (e.g. the sun) render black, so we clamp to 65504 instead —
// the value is still very bright and physically meaningful.
// NaN values have no sensible interpretation and are zeroed out.
static void sanitizeHdrPixels(std::vector<uint8_t>& pixels)
{
    static constexpr float kMaxHalf = 65504.0f;  // largest finite half-float

    float* p           = reinterpret_cast<float*>(pixels.data());
    const size_t count = pixels.size() / sizeof(float);
    for (size_t i = 0; i < count; ++i)
    {
        if (std::isnan(p[i]))
        {
            p[i] = 0.0f;          // truly invalid — no meaningful value
        }
        else if (p[i] > kMaxHalf) // +inf or any overflow beyond EXR's ceiling
        {
            p[i] = kMaxHalf;
        }
        else if (p[i] < 0.0f)    // negative radiance is non-physical
        {
            p[i] = 0.0f;
        }
    }
}

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
    try
    {
#ifdef _WIN32
        WideFileStream    stream(path);
        Imf::RgbaInputFile file(stream);
#else
        Imf::RgbaInputFile file(path.c_str());
#endif
        const Imath::Box2i dw = file.dataWindow();
        const int w = dw.max.x - dw.min.x + 1;
        const int h = dw.max.y - dw.min.y + 1;

        // Read scanlines into a temporary half-float RGBA buffer.
        Imf::Array2D<Imf::Rgba> buf(h, w);
        file.setFrameBuffer(&buf[0][0] - dw.min.x - dw.min.y * w, 1, w);
        file.readPixels(dw.min.y, dw.max.y);

        // Convert half4 → float4 and store as raw bytes in pixels.
        const size_t pixelCount = static_cast<size_t>(w) * h;
        pixels.resize(pixelCount * sizeof(float4));
        float* dst = reinterpret_cast<float*>(pixels.data());

        for (int j = 0; j < h; ++j)
        {
            for (int i = 0; i < w; ++i)
            {
                const Imf::Rgba& px = buf[j][i];
                *dst++ = static_cast<float>(px.r);
                *dst++ = static_cast<float>(px.g);
                *dst++ = static_cast<float>(px.b);
                *dst++ = static_cast<float>(px.a);
            }
        }

        format = PixelFormat::RGBA32F;
        width  = w;
        height = h;
        sanitizeHdrPixels(pixels);
        return true;
    }
    catch (const std::exception& e)
    {
        outError = e.what();
        return false;
    }
}

// ─── Texture::loadHDR ────────────────────────────────────────────────────────

bool Texture::loadHDR(const std::string& path, std::string& outError)
{
    int w = 0, h = 0, channels = 0;
    // stbi_loadf decodes RGBE encoding → linear float; requesting 4 channels
    // always produces RGBA (alpha is filled with 1.0 since .hdr has no alpha).
    // On Windows use _wfopen so non-ANSI characters in the path work correctly.
#ifdef _WIN32
    FILE* f = _wfopen(utf8ToWide(path).c_str(), L"rb");
    if (!f)
    {
        outError = "Cannot open file: " + path;
        return false;
    }
    float* data = stbi_loadf_from_file(f, &w, &h, &channels, 4);
    fclose(f);
#else
    float* data = stbi_loadf(path.c_str(), &w, &h, &channels, 4);
#endif
    if (!data)
    {
        outError = stbi_failure_reason()
                 ? stbi_failure_reason()
                 : "unknown stb_image error";
        return false;
    }

    const size_t byteCount = static_cast<size_t>(w) * h * sizeof(float4);
    pixels.resize(byteCount);
    std::memcpy(pixels.data(), data, byteCount);
    stbi_image_free(data);

    format = PixelFormat::RGBA32F;
    width  = w;
    height = h;
    sanitizeHdrPixels(pixels);
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
    {
        return;
    }

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
    {
        totalWeight += rowSums[j];
    }

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
