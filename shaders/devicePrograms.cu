// devicePrograms.cu — OptiX Monte Carlo path tracer.
//
// Architecture:
//   • The raygen shader drives the entire bounce loop iteratively via a for-loop.
//     No closest-hit program ever calls optixTrace — trace depth is always 1.
//   • Hit data (position, normal, albedo) is returned to the raygen by packing a
//     pointer into the two uint32 payload values.  The closest-hit program writes
//     through that pointer; the miss program leaves it untouched (hit==0 default).
//   • Each launch adds one sample to a per-pixel float4 accumulation buffer.
//     The raygen reads the running average and writes a tone-mapped uchar4 to the
//     display buffer every frame.

#include <optix.h>
#include "LaunchParams.h"
#include "SceneData.h"

extern "C" { __constant__ LaunchParams optixLaunchParams; }

// ─── Tuning ───────────────────────────────────────────────────────────────────

static constexpr int MAX_BOUNCES = 6;

// ─── Device math helpers ──────────────────────────────────────────────────────

static __forceinline__ __device__ float  devDot(float3 a, float3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
static __forceinline__ __device__ float3 devAdd(float3 a, float3 b) { return make_float3(a.x+b.x, a.y+b.y, a.z+b.z); }
static __forceinline__ __device__ float3 devMul(float3 a, float3 b) { return make_float3(a.x*b.x, a.y*b.y, a.z*b.z); }
static __forceinline__ __device__ float3 devScale(float3 v, float s) { return make_float3(v.x*s, v.y*s, v.z*s); }
static __forceinline__ __device__ float  devClamp01(float x)        { return x < 0.f ? 0.f : (x > 1.f ? 1.f : x); }

static __forceinline__ __device__
float3 devNormalize(float3 v)
{
    const float ilen = rsqrtf(devDot(v, v));
    return make_float3(v.x*ilen, v.y*ilen, v.z*ilen);
}

static __forceinline__ __device__
float3 devMix(float3 a, float3 b, float t)
{
    return make_float3(
        b.x*t + a.x*(1.f-t),
        b.y*t + a.y*(1.f-t),
        b.z*t + a.z*(1.f-t));
}

// ─── Path payload ─────────────────────────────────────────────────────────────
// The raygen stack-allocates one of these and passes a packed pointer to each
// optixTrace call.  The closest-hit program fills in the fields and sets hit=1.

struct PathVertex
{
    float3 pos;     // world-space surface point
    float3 N;       // world-space shading normal (unit)
    float3 albedo;  // Lambertian surface colour
    int    hit;     // 1 = geometry hit, 0 = ray escaped to background
};

static __forceinline__ __device__
void packPointer(void* ptr, uint32_t& p0, uint32_t& p1)
{
    const uint64_t u = reinterpret_cast<uint64_t>(ptr);
    p0 = static_cast<uint32_t>(u >> 32);
    p1 = static_cast<uint32_t>(u & 0xFFFFFFFFu);
}

static __forceinline__ __device__
PathVertex* unpackVertex(uint32_t p0, uint32_t p1)
{
    return reinterpret_cast<PathVertex*>(
        (static_cast<uint64_t>(p0) << 32) | static_cast<uint64_t>(p1));
}

// ─── PCG pseudo-random number generator ──────────────────────────────────────
// O'Neill 2014 — one 32-bit state, good statistical quality, cheap on GPU.

static __forceinline__ __device__
uint32_t pcgNext(uint32_t& state)
{
    uint32_t prev = state * 747796405u + 2891336453u;
    state = prev;
    uint32_t w = ((prev >> ((prev >> 28u) + 4u)) ^ prev) * 277803737u;
    return (w >> 22u) ^ w;
}

static __forceinline__ __device__
float rnd(uint32_t& seed) { return (float)pcgNext(seed) * (1.f / 4294967296.f); }

// ─── Cosine-weighted hemisphere sampling ─────────────────────────────────────
// Returns a direction in a Z-up local frame.
// For a Lambertian surface: BRDF = albedo/π, PDF = cosθ/π  →  BRDF/PDF = albedo.
// The throughput update is therefore simply: throughput *= albedo.

static __forceinline__ __device__
float3 cosineSampleHemisphere(float r1, float r2)
{
    const float phi    = 2.f * 3.14159265358979f * r1;
    const float sqrtR2 = sqrtf(r2);
    return make_float3(cosf(phi)*sqrtR2, sinf(phi)*sqrtR2, sqrtf(1.f - r2));
}

// Builds a tangent frame around N using the Duff et al. 2017 method.
static __forceinline__ __device__
void buildONB(float3 N, float3& T, float3& B)
{
    const float s = copysignf(1.f, N.z);
    const float a = -1.f / (s + N.z);
    const float b =  N.x * N.y * a;
    T = make_float3(1.f + s*N.x*N.x*a, s*b, -s*N.x);
    B = make_float3(b, s + N.y*N.y*a, -N.y);
}

// ─── Background / environment map ────────────────────────────────────────────
// Returns raw HDR radiance for the given direction.  Tone mapping is applied
// later, once per pixel, on the accumulated average.

static __forceinline__ __device__
float3 sampleBackground(float3 dir)
{
    if (optixLaunchParams.envMap != 0)
    {
        const float kInvPi  = 0.31830988618f;
        const float kInv2Pi = 0.15915494309f;
        const float theta   = acosf(fmaxf(-1.f, fminf(1.f, dir.y)));
        const float phi     = atan2f(dir.x, -dir.z);
        const float4 s = tex2D<float4>(optixLaunchParams.envMap,
                                       phi * kInv2Pi + 0.5f,
                                       theta * kInvPi);
        return make_float3(s.x, s.y, s.z);  // raw HDR
    }
    // Procedural sky: warm white horizon → sky blue zenith
    const float t = devClamp01(0.5f * (dir.y + 1.f));
    return devMix(make_float3(1.f, 1.f, 1.f), make_float3(0.3f, 0.5f, 1.0f), t);
}

// ─── Raygen — iterative path loop ────────────────────────────────────────────

extern "C" __global__ void __raygen__renderFrame()
{
    const uint3        idx   = optixGetLaunchIndex();
    const uint3        dim   = optixGetLaunchDimensions();
    const unsigned int fbIdx = idx.y * dim.x + idx.x;

    // ── No scene: show env map preview or UV placeholder ──────────────────────
    if (optixLaunchParams.traversable == 0)
    {
        if (optixLaunchParams.envMap != 0)
        {
            const float nx  = ((float)idx.x + 0.5f) / (float)dim.x * 2.f - 1.f;
            const float ny  = ((float)idx.y + 0.5f) / (float)dim.y * 2.f - 1.f;
            const float3 dir = devNormalize(devAdd(devAdd(
                devScale(optixLaunchParams.U, nx),
                devScale(optixLaunchParams.V, ny)),
                optixLaunchParams.W));
            float3 c = sampleBackground(dir);
            c.x = c.x / (c.x + 1.f);  // Reinhard
            c.y = c.y / (c.y + 1.f);
            c.z = c.z / (c.z + 1.f);
            optixLaunchParams.colorBuffer[fbIdx] = make_uchar4(
                (unsigned char)(devClamp01(c.x) * 255.f),
                (unsigned char)(devClamp01(c.y) * 255.f),
                (unsigned char)(devClamp01(c.z) * 255.f), 255u);
        }
        else
        {
            optixLaunchParams.colorBuffer[fbIdx] = make_uchar4(
                (unsigned char)(255u * idx.x / dim.x),
                (unsigned char)(255u * idx.y / dim.y),
                128u, 255u);
        }
        return;
    }

    // ── Path tracing ──────────────────────────────────────────────────────────

    // Seed: mix pixel index and sample index for uncorrelated sequences
    uint32_t seed = fbIdx * 1973u + optixLaunchParams.sampleIndex * 9277u + 127u;
    pcgNext(seed);  // warm up

    // Sub-pixel jitter for anti-aliasing
    const float nx = ((float)idx.x + rnd(seed)) / (float)dim.x * 2.f - 1.f;
    const float ny = ((float)idx.y + rnd(seed)) / (float)dim.y * 2.f - 1.f;

    float3 rayOrig = optixLaunchParams.eye;
    float3 rayDir  = devNormalize(devAdd(devAdd(
        devScale(optixLaunchParams.U, nx),
        devScale(optixLaunchParams.V, ny)),
        optixLaunchParams.W));

    float3 throughput = make_float3(1.f, 1.f, 1.f);
    float3 radiance   = make_float3(0.f, 0.f, 0.f);

    for (int bounce = 0; bounce < MAX_BOUNCES; ++bounce)
    {
        // Allocate hit record on the stack and pass it as a pointer payload
        PathVertex vtx;
        vtx.hit = 0;

        uint32_t p0, p1;
        packPointer(&vtx, p0, p1);

        optixTrace(
            optixLaunchParams.traversable,
            rayOrig, rayDir,
            1e-3f, 1e30f, 0.f,
            OptixVisibilityMask(0xFF),
            OPTIX_RAY_FLAG_NONE,
            0, 1, 0,   // SBT offset / stride / miss index
            p0, p1);

        if (!vtx.hit)
        {
            // Ray escaped — environment light terminates the path
            radiance = devAdd(radiance, devMul(throughput, sampleBackground(rayDir)));
            break;
        }

        // Multiply throughput by albedo (BRDF/PDF cancels for cosine sampling)
        throughput = devMul(throughput, vtx.albedo);

        // Russian roulette: randomly terminate dim paths to keep variance bounded
        if (bounce >= 3)
        {
            const float maxThr = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
            if (rnd(seed) > maxThr) break;
            throughput = devScale(throughput, 1.f / fmaxf(maxThr, 1e-6f));
        }

        // Sample next direction: cosine-weighted hemisphere around N
        float3 T, B;
        buildONB(vtx.N, T, B);
        const float3 localDir = cosineSampleHemisphere(rnd(seed), rnd(seed));
        rayDir = devNormalize(devAdd(devAdd(
            devScale(T,     localDir.x),
            devScale(B,     localDir.y)),
            devScale(vtx.N, localDir.z)));

        // Offset ray origin from the surface to prevent self-intersection
        rayOrig = devAdd(vtx.pos, devScale(vtx.N, 1e-3f));
    }

    // ── Accumulate ────────────────────────────────────────────────────────────
    float4& acc = optixLaunchParams.accumBuffer[fbIdx];
    acc.x += radiance.x;
    acc.y += radiance.y;
    acc.z += radiance.z;

    // ── Tone-map the running average and write to the display buffer ──────────
    const float  inv = 1.f / (float)(optixLaunchParams.sampleIndex + 1);
    float3 avg = make_float3(acc.x * inv, acc.y * inv, acc.z * inv);

    // Reinhard: maps [0, ∞) → [0, 1)
    avg.x = avg.x / (avg.x + 1.f);
    avg.y = avg.y / (avg.y + 1.f);
    avg.z = avg.z / (avg.z + 1.f);

    optixLaunchParams.colorBuffer[fbIdx] = make_uchar4(
        (unsigned char)(devClamp01(avg.x) * 255.f),
        (unsigned char)(devClamp01(avg.y) * 255.f),
        (unsigned char)(devClamp01(avg.z) * 255.f),
        255u);
}

// ─── Miss — radiance ray ──────────────────────────────────────────────────────
// The ray escaped to the background. The PathVertex already has hit=0 (set by the
// raygen before the trace call), so nothing needs to change here.

extern "C" __global__ void __miss__radiance()
{
    // intentionally empty
}

// ─── Closest-hit — radiance ray ──────────────────────────────────────────────

extern "C" __global__ void __closesthit__radiance()
{
    PathVertex* vtx = unpackVertex(optixGetPayload_0(), optixGetPayload_1());

    const MeshData& mesh = *reinterpret_cast<const MeshData*>(optixGetSbtDataPointer());

    // ── Interpolated shading normal ────────────────────────────────────────────
    const uint3  tri = mesh.indices[optixGetPrimitiveIndex()];
    const float2 bc  = optixGetTriangleBarycentrics();
    const float  w0  = 1.f - bc.x - bc.y;

    const float3 n_obj = devNormalize(devAdd(devAdd(
        devScale(mesh.normals[tri.x], w0),
        devScale(mesh.normals[tri.y], bc.x)),
        devScale(mesh.normals[tri.z], bc.y)));

    vtx->N = devNormalize(optixTransformNormalFromObjectToWorldSpace(n_obj));

    // ── World-space hit point ──────────────────────────────────────────────────
    const float3 d   = optixGetWorldRayDirection();
    const float3 o   = optixGetWorldRayOrigin();
    const float  t   = optixGetRayTmax();
    vtx->pos = make_float3(o.x + t*d.x, o.y + t*d.y, o.z + t*d.z);

    // ── Material albedo ────────────────────────────────────────────────────────
    vtx->albedo = make_float3(0.8f, 0.8f, 0.8f);
    if (optixLaunchParams.materials && mesh.materialIndex >= 0)
        vtx->albedo = optixLaunchParams.materials[mesh.materialIndex].albedo;

    vtx->hit = 1;
}
