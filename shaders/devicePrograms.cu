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

static constexpr int MAX_BOUNCES = 16;

// ─── Device math helpers ──────────────────────────────────────────────────────

static __forceinline__ __device__ float  devDot(float3 a, float3 b)   { return a.x*b.x + a.y*b.y + a.z*b.z; }
static __forceinline__ __device__ float3 devAdd(float3 a, float3 b)   { return make_float3(a.x+b.x, a.y+b.y, a.z+b.z); }
static __forceinline__ __device__ float3 devSub(float3 a, float3 b)   { return make_float3(a.x-b.x, a.y-b.y, a.z-b.z); }
static __forceinline__ __device__ float3 devMul(float3 a, float3 b)   { return make_float3(a.x*b.x, a.y*b.y, a.z*b.z); }
static __forceinline__ __device__ float3 devScale(float3 v, float s)  { return make_float3(v.x*s, v.y*s, v.z*s); }
static __forceinline__ __device__ float3 devNeg(float3 v)             { return make_float3(-v.x, -v.y, -v.z); }
static __forceinline__ __device__ float  devClamp01(float x)          { return x < 0.f ? 0.f : (x > 1.f ? 1.f : x); }
static __forceinline__ __device__ float  devLuminance(float3 c)       { return 0.2126f*c.x + 0.7152f*c.y + 0.0722f*c.z; }

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
    float3 pos;       // world-space surface point
    float3 N;         // world-space shading normal (unit)
    float3 albedo;    // base colour
    float3 emission;  // pre-scaled emissive radiance (emission * emissionScale)
    float  roughness; // perceptual roughness [0, 1]
    float  metallic;  // metallic factor [0, 1]
    int    hit;       // 1 = geometry hit, 0 = ray escaped to background
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

// ─── PBR helpers ─────────────────────────────────────────────────────────────

// Schlick Fresnel — F0 is the reflectance at normal incidence.
static __forceinline__ __device__
float3 devFresnel(float cosTheta, float3 F0)
{
    const float t  = 1.f - devClamp01(cosTheta);
    const float t5 = t * t * t * t * t;
    return devAdd(F0, devScale(devSub(make_float3(1.f, 1.f, 1.f), F0), t5));
}

// Sample GGX visible normals (Heitz 2018).
// Standard hemisphere sampling produces many below-horizon reflections on rough
// surfaces (causing energy loss).  VNDF conditions on the view direction so the
// sampled half-vector is always visible, making sub-horizon outcomes nearly
// impossible and simplifying the MC weight to just F·G1(L).
//
// V_local: view direction in tangent space (z = N component, must be > 0).
// Returns: half-vector in tangent space.
static __forceinline__ __device__
float3 devSampleGGX_VNDF(float3 V_local, float alpha, float u1, float u2)
{
    // Stretch view vector into the hemisphere configuration
    const float3 Vh = devNormalize(
        make_float3(alpha * V_local.x, alpha * V_local.y, V_local.z));

    // Orthonormal basis around Vh (T1 ⊥ Vh in the XY plane)
    const float len2 = Vh.x*Vh.x + Vh.y*Vh.y;
    const float3 T1  = (len2 > 1e-7f)
        ? devScale(make_float3(-Vh.y, Vh.x, 0.f), rsqrtf(len2))
        : make_float3(1.f, 0.f, 0.f);
    // T2 = cross(Vh, T1) — inline to avoid needing devCross
    const float3 T2  = make_float3(Vh.y*T1.z - Vh.z*T1.y,
                                    Vh.z*T1.x - Vh.x*T1.z,
                                    Vh.x*T1.y - Vh.y*T1.x);

    // Sample a point on the projected area disk
    const float r   = sqrtf(u1);
    const float phi = 2.f * 3.14159265358979f * u2;
    float t1 = r * cosf(phi);
    float t2 = r * sinf(phi);
    // Blend between uniform disk (s=0) and projected hemisphere (s=1)
    const float s = 0.5f * (1.f + Vh.z);
    t2 = (1.f - s) * sqrtf(fmaxf(0.f, 1.f - t1*t1)) + s * t2;

    // Lift onto hemisphere, stretch back to GGX ellipsoid
    const float3 Nh = devAdd(devAdd(devScale(T1, t1), devScale(T2, t2)),
                             devScale(Vh, sqrtf(fmaxf(0.f, 1.f - t1*t1 - t2*t2))));
    return devNormalize(
        make_float3(alpha * Nh.x, alpha * Nh.y, fmaxf(1e-6f, Nh.z)));
}

// Smith G1 shadowing-masking term for GGX.
static __forceinline__ __device__
float devSmithG1(float cosTheta, float alpha)
{
    const float a2 = alpha * alpha;
    const float c2 = cosTheta * cosTheta;
    return 2.f * cosTheta / (cosTheta + sqrtf(a2 + (1.f - a2) * c2));
}

// Reflect incident direction v around surface normal n.
static __forceinline__ __device__
float3 devReflect(float3 v, float3 n)
{
    return devSub(v, devScale(n, 2.f * devDot(v, n)));
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
            c.x = powf(devClamp01(c.x / (c.x + 1.f)), 1.f / 2.2f);  // Reinhard + sRGB
            c.y = powf(devClamp01(c.y / (c.y + 1.f)), 1.f / 2.2f);
            c.z = powf(devClamp01(c.z / (c.z + 1.f)), 1.f / 2.2f);
            optixLaunchParams.colorBuffer[fbIdx] = make_uchar4(
                (unsigned char)(c.x * 255.f),
                (unsigned char)(c.y * 255.f),
                (unsigned char)(c.z * 255.f), 255u);
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

        // ── Emission ──────────────────────────────────────────────────────────
        radiance = devAdd(radiance, devMul(throughput, vtx.emission));

        // ── PBR material properties ───────────────────────────────────────────
        const float3 albedo    = vtx.albedo;
        const float  roughness = vtx.roughness;
        const float  metallic  = vtx.metallic;
        // Remap perceptual roughness → GGX alpha; clamp to avoid singularities.
        const float  alpha     = fmaxf(roughness * roughness, 1e-3f);

        // F0: dielectrics reflect ~4 % at normal incidence; metals reflect albedo.
        const float3 F0  = devMix(make_float3(0.04f, 0.04f, 0.04f), albedo, metallic);
        const float  cosV = fmaxf(0.f, devDot(vtx.N, devNeg(rayDir)));
        const float3 F    = devFresnel(cosV, F0);

        // Probability of taking the specular path.
        // Metals are always specular (lerp to 1) so 1–p_spec never underflows.
        const float p_spec = devClamp01(
            (1.f - metallic) * devLuminance(F) + metallic);

        if (rnd(seed) < p_spec)
        {
            // ── Specular path (GGX-VNDF) ──────────────────────────────────────
            // Build tangent frame and express the view direction in it.
            float3 T, B;
            buildONB(vtx.N, T, B);
            const float3 V       = devNeg(rayDir);  // outgoing (toward camera)
            const float3 V_local = make_float3(
                devDot(V, T), devDot(V, B), fmaxf(1e-4f, devDot(V, vtx.N)));

            // Sample a visible half-vector and transform back to world space.
            const float3 H_local = devSampleGGX_VNDF(V_local, alpha, rnd(seed), rnd(seed));
            const float3 H = devNormalize(devAdd(devAdd(
                devScale(T,      H_local.x),
                devScale(B,      H_local.y)),
                devScale(vtx.N,  H_local.z)));

            const float3 L = devReflect(rayDir, H);

            // VNDF guarantees L above the surface almost always; the check guards
            // rare numerical-precision edge cases and is NOT the main energy sink
            // that standard GGX hemisphere sampling suffered from.
            if (devDot(L, vtx.N) <= 0.f)
            {
                break;
            }

            // VNDF MC weight: BRDF/PDF / p_spec = F * G1(L) / p_spec.
            // G1(V) cancels between BRDF and the VNDF sampling PDF.
            const float cosNL = fmaxf(1e-4f, devDot(vtx.N, L));
            throughput = devMul(throughput,
                devScale(F, devSmithG1(cosNL, alpha) / fmaxf(1e-4f, p_spec)));
            rayDir = L;
        }
        else
        {
            // ── Diffuse path (Lambertian, cosine-weighted hemisphere) ──────────
            // Weight = (1-F) * albedo / (1 - p_spec)
            // BRDF/PDF cancels for cosine sampling → weight = albedo.
            float3 T, B;
            buildONB(vtx.N, T, B);
            const float3 localDir = cosineSampleHemisphere(rnd(seed), rnd(seed));
            rayDir = devNormalize(devAdd(devAdd(
                devScale(T,      localDir.x),
                devScale(B,      localDir.y)),
                devScale(vtx.N,  localDir.z)));

            const float3 kD = devMul(devSub(make_float3(1.f, 1.f, 1.f), F), albedo);
            throughput = devMul(throughput,
                devScale(kD, 1.f / fmaxf(1e-4f, 1.f - p_spec)));
        }

        // Russian roulette: randomly terminate dim paths to keep variance bounded
        if (bounce >= 3)
        {
            const float maxThr = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
            if (rnd(seed) > maxThr)
            {
                break;
            }
            throughput = devScale(throughput, 1.f / fmaxf(maxThr, 1e-6f));
        }

        // Offset ray origin along normal to prevent self-intersection
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

    // Reinhard: maps [0, ∞) → [0, 1)  (result is still linear)
    avg.x = avg.x / (avg.x + 1.f);
    avg.y = avg.y / (avg.y + 1.f);
    avg.z = avg.z / (avg.z + 1.f);

    // Linear → sRGB (γ ≈ 1/2.2): the display buffer is read by the monitor as sRGB,
    // so we must gamma-encode before writing.  powf(x, 1/2.2) approximates the
    // standard sRGB OETF well enough for a path tracer output.
    avg.x = powf(devClamp01(avg.x), 1.f / 2.2f);
    avg.y = powf(devClamp01(avg.y), 1.f / 2.2f);
    avg.z = powf(devClamp01(avg.z), 1.f / 2.2f);

    optixLaunchParams.colorBuffer[fbIdx] = make_uchar4(
        (unsigned char)(avg.x * 255.f),
        (unsigned char)(avg.y * 255.f),
        (unsigned char)(avg.z * 255.f),
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
    vtx->pos = devAdd(o, devScale(d, t));

    // ── Material properties ────────────────────────────────────────────────────
    if (optixLaunchParams.materials && mesh.materialIndex >= 0)
    {
        const MaterialData& mat = optixLaunchParams.materials[mesh.materialIndex];
        vtx->albedo    = mat.albedo;
        vtx->roughness = mat.roughness;
        vtx->metallic  = mat.metallic;
        vtx->emission  = devScale(mat.emission, mat.emissionScale);
    }
    else
    {
        vtx->albedo    = make_float3(0.8f, 0.8f, 0.8f);
        vtx->roughness = 0.5f;
        vtx->metallic  = 0.f;
        vtx->emission  = make_float3(0.f, 0.f, 0.f);
    }

    vtx->hit = 1;
}
