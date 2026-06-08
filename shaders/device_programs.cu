// device_programs.cu — OptiX Monte Carlo path tracer.
//
// Architecture:
//   • The raygen shader drives the entire bounce loop iteratively via a for-loop.
//     No closest-hit program ever calls optixTrace — trace depth is always 1.
//   • Hit data (position, normal, material) is returned to the raygen by packing a
//     pointer into the two uint32 payload values.  The closest-hit program writes
//     through that pointer; the miss program leaves it untouched (hit==0 default).
//   • Each launch adds one sample to a per-pixel float4 accumulation buffer.
//     The raygen reads the running average and writes a tone-mapped uchar4 to the
//     display buffer every frame.

#include <optix.h>
#include "device_math.h"   // float3 operators (+, -, *, /, unary -)
#include "launch_params.h"
#include "scene_data.h"

extern "C" { __constant__ LaunchParams optixLaunchParams; }

// ─── Tuning ───────────────────────────────────────────────────────────────────

static constexpr int MAX_BOUNCES = 16;

// ─── Scalar / float3 utilities ───────────────────────────────────────────────

static __forceinline__ __device__ float devClamp01(float x) { return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x); }
static __forceinline__ __device__ float devDot(float3 a, float3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
static __forceinline__ __device__ float devLuminance(float3 c) { return 0.2126f*c.x + 0.7152f*c.y + 0.0722f*c.z; }

static __forceinline__ __device__
float3 devNormalize(float3 v)
{
    return v * rsqrtf(devDot(v, v));
}

// Linear interpolation: a + (b-a)*t
static __forceinline__ __device__
float3 devMix(float3 a, float3 b, float t)
{
    return a + (b - a) * t;
}

// ─── Path payload ─────────────────────────────────────────────────────────────
// The raygen stack-allocates one of these and passes a packed pointer to each
// optixTrace call.  The closest-hit program fills in the fields and sets hit=1.

struct PathVertex
{
    float3 pos;          // world-space surface point
    float3 N;            // world-space shading normal (unit)
    float3 albedo;       // base colour
    float3 emission;     // pre-scaled emissive radiance (emission * emissionScale)
    float  roughness;    // perceptual roughness [0, 1]
    float  metallic;     // metallic factor [0, 1]
    float  transmission;        // 0 = opaque, 1 = fully transmissive
    float  ior;                 // index of refraction
    float  absorptionDistance;  // world-space units for full albedo absorption
    float  clearcoat;           // clearcoat layer intensity [0, 1]
    float  clearcoatRoughness;  // clearcoat layer roughness [0, 1]
    float  t;                   // ray travel distance to this hit (Beer-Lambert)
    int    hit;          // 1 = geometry hit, 0 = ray escaped to background
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
float rnd(uint32_t& seed) { return (float)pcgNext(seed) * (1.0f / 4294967296.0f); }

// ─── Sampling helpers ─────────────────────────────────────────────────────────

// Cosine-weighted hemisphere direction in a Z-up local frame.
// BRDF = albedo/π, PDF = cosθ/π → BRDF/PDF = albedo (weight is just albedo).
static __forceinline__ __device__
float3 cosineSampleHemisphere(float r1, float r2)
{
    const float phi     = 2.0f * 3.14159265358979f * r1;
    const float sqrtR2  = sqrtf(r2);
    return make_float3(cosf(phi)*sqrtR2, sinf(phi)*sqrtR2, sqrtf(1.0f - r2));
}

// Orthonormal tangent frame around N (Duff et al. 2017).
static __forceinline__ __device__
void buildONB(float3 N, float3& T, float3& B)
{
    const float s = copysignf(1.0f, N.z);
    const float a = -1.0f / (s + N.z);
    const float b =  N.x * N.y * a;
    T = make_float3(1.0f + s*N.x*N.x*a, s*b, -s*N.x);
    B = make_float3(b, s + N.y*N.y*a, -N.y);
}

// ─── PBR helpers ─────────────────────────────────────────────────────────────

// Schlick Fresnel approximation.
static __forceinline__ __device__
float3 devFresnel(float cosTheta, float3 F0)
{
    const float t  = 1.0f - devClamp01(cosTheta);
    const float t5 = t * t * t * t * t;
    return F0 + (make_float3(1.0f, 1.0f, 1.0f) - F0) * t5;
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
    const float  len2 = Vh.x*Vh.x + Vh.y*Vh.y;
    const float3 T1   = (len2 > 1e-7f)
        ? make_float3(-Vh.y, Vh.x, 0.0f) * rsqrtf(len2)
        : make_float3(1.0f, 0.0f, 0.0f);
    // T2 = cross(Vh, T1)
    const float3 T2 = make_float3(Vh.y*T1.z - Vh.z*T1.y,
                                   Vh.z*T1.x - Vh.x*T1.z,
                                   Vh.x*T1.y - Vh.y*T1.x);

    // Sample a point on the projected area disk
    const float r   = sqrtf(u1);
    const float phi = 2.0f * 3.14159265358979f * u2;
    float t1 = r * cosf(phi);
    float t2 = r * sinf(phi);
    // Blend between uniform disk (s=0) and projected hemisphere (s=1)
    const float s = 0.5f * (1.0f + Vh.z);
    t2 = (1.0f - s) * sqrtf(fmaxf(0.0f, 1.0f - t1*t1)) + s * t2;

    // Lift onto hemisphere, stretch back to GGX ellipsoid
    const float3 Nh = T1 * t1 + T2 * t2
                    + Vh * sqrtf(fmaxf(0.0f, 1.0f - t1*t1 - t2*t2));
    return devNormalize(
        make_float3(alpha * Nh.x, alpha * Nh.y, fmaxf(1e-6f, Nh.z)));
}

// Smith G1 shadowing-masking term for GGX.
static __forceinline__ __device__
float devSmithG1(float cosTheta, float alpha)
{
    const float a2 = alpha * alpha;
    const float c2 = cosTheta * cosTheta;
    return 2.0f * cosTheta / (cosTheta + sqrtf(a2 + (1.0f - a2) * c2));
}

// Reflect incident direction v around normal n.
static __forceinline__ __device__
float3 devReflect(float3 v, float3 n)
{
    return v - n * (2.0f * devDot(v, n));
}

// Refract incident direction v through a surface with normal n (opposing v) and
// relative IOR eta = n1/n2.  Returns true on successful refraction and writes the
// transmitted direction to `out`.  Returns false on total internal reflection and
// writes the mirrored direction to `out` so the caller can continue the path.
static __forceinline__ __device__
bool devRefract(float3 v, float3 n, float eta, float3& out)
{
    const float cosI  = devDot(-v, n);                        // positive: n opposes v
    const float sinT2 = eta * eta * fmaxf(0.0f, 1.0f - cosI * cosI);
    if (sinT2 >= 1.0f)                                         // total internal reflection
    {
        out = devReflect(v, n);
        return false;
    }
    const float cosT = sqrtf(1.0f - sinT2);
    out = devNormalize(v * eta + n * (eta * cosI - cosT));    // Snell's law, vector form
    return true;
}

// ─── Background / environment map ────────────────────────────────────────────

static __forceinline__ __device__
float3 sampleBackground(float3 dir)
{
    const float exposure = exp2f(optixLaunchParams.envExposure);  // EV stops → linear scale

    if (optixLaunchParams.envMap != 0)
    {
        const float kInvPi  = 0.31830988618f;
        const float kInv2Pi = 0.15915494309f;
        const float theta   = acosf(fmaxf(-1.0f, fminf(1.0f, dir.y)));
        const float phi     = atan2f(dir.x, -dir.z) + optixLaunchParams.envMapRotation;
        const float4 s = tex2D<float4>(optixLaunchParams.envMap,
                                       phi * kInv2Pi + 0.5f,
                                       theta * kInvPi);
        return make_float3(s.x, s.y, s.z) * exposure;
    }
    // Procedural sky:
    return devMix(make_float3(0.25f, 0.3f, 0.6f), make_float3(0.8f, 1.1f, 3.0f),
                  devClamp01(dir.y)) * exposure;
}

// ─── HDRI importance-sampling helpers ────────────────────────────────────────

// Binary search: first index in cdf[0..n-1] where cdf[idx] >= u.
static __forceinline__ __device__
int upperBound(const float* cdf, int n, float u)
{
    int lo = 0, hi = n - 1;
    while (lo < hi)
    {
        const int mid = (lo + hi) >> 1;
        if (cdf[mid] < u)
        {
            lo = mid + 1;
        }
        else
        {
            hi = mid;
        }
    }
    return lo;
}

// Sample the env map by importance.
// r1, r2: uniform [0,1).  Writes the sampled world-space direction to outDir
// and its solid-angle PDF to outPdf.  Returns the HDR radiance at that direction.
static __forceinline__ __device__
float3 sampleEnvMapIS(float r1, float r2, float3& outDir, float& outPdf)
{
    const int    W           = optixLaunchParams.envCdfW;
    const int    H           = optixLaunchParams.envCdfH;
    const float* marginal    = optixLaunchParams.envMarginalCdf;
    const float* conditional = optixLaunchParams.envConditionalCdf;

    // Sample row j, then column i
    const int j = upperBound(marginal,          H, r1);
    const int i = upperBound(conditional + j*W, W, r2);

    // Continuous UV at texel centre
    const float u = (i + 0.5f) / static_cast<float>(W);
    const float v = (j + 0.5f) / static_cast<float>(H);

    // UV → spherical: undo the envMapRotation sampleBackground would apply
    const float theta    = v * 3.14159265358979f;
    const float phiWorld = (u - 0.5f) * 6.28318530717959f - optixLaunchParams.envMapRotation;
    const float sinT     = sinf(theta);
    outDir = make_float3(sinf(phiWorld) * sinT, cosf(theta), -cosf(phiWorld) * sinT);

    // PDF in solid-angle measure
    const float mPrev = (j > 0) ? marginal[j - 1]             : 0.0f;
    const float cPrev = (i > 0) ? conditional[j * W + i - 1] : 0.0f;
    const float pUV   = (marginal[j] - mPrev) * static_cast<float>(H)
                      * (conditional[j * W + i] - cPrev) * static_cast<float>(W);
    outPdf = pUV / fmaxf(1e-5f, 2.0f * 3.14159265358979f * 3.14159265358979f * sinT);

    return sampleBackground(outDir);  // applies exposure + rotation + fallback
}

// Evaluate the solid-angle PDF for a world-space direction under env map IS.
// Returns 0 when no CDF is loaded.
static __forceinline__ __device__
float evalEnvMapPdf(float3 dir)
{
    if (!optixLaunchParams.envMarginalCdf)
    {
        return 0.0f;
    }

    const int    W           = optixLaunchParams.envCdfW;
    const int    H           = optixLaunchParams.envCdfH;
    const float* marginal    = optixLaunchParams.envMarginalCdf;
    const float* conditional = optixLaunchParams.envConditionalCdf;

    // Direction → (u, v) — mirror exactly what sampleBackground computes
    const float kInvPi  = 0.31830988618f;
    const float kInv2Pi = 0.15915494309f;
    const float theta   = acosf(fmaxf(-1.0f, fminf(1.0f, dir.y)));
    const float phi     = atan2f(dir.x, -dir.z) + optixLaunchParams.envMapRotation;
    float u = phi * kInv2Pi + 0.5f;
    u = u - floorf(u);  // wrap to [0,1]
    const float v = theta * kInvPi;

    const int i = min(static_cast<int>(u * static_cast<float>(W)), W - 1);
    const int j = min(static_cast<int>(v * static_cast<float>(H)), H - 1);

    const float mPrev = (j > 0) ? marginal[j - 1]             : 0.0f;
    const float cPrev = (i > 0) ? conditional[j * W + i - 1] : 0.0f;
    const float pUV   = (marginal[j] - mPrev) * static_cast<float>(H)
                      * (conditional[j * W + i] - cPrev) * static_cast<float>(W);
    const float sinT  = sinf(theta);
    return (sinT > 1e-5f)
        ? pUV / (2.0f * 3.14159265358979f * 3.14159265358979f * sinT)
        : 0.0f;
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
            const float nx  = ((float)idx.x + 0.5f) / (float)dim.x * 2.0f - 1.0f;
            const float ny  = ((float)idx.y + 0.5f) / (float)dim.y * 2.0f - 1.0f;
            const float3 dir = devNormalize(
                optixLaunchParams.U * nx +
                optixLaunchParams.V * ny +
                optixLaunchParams.W);
            float3 c = sampleBackground(dir);
            c.x = powf(devClamp01(c.x / (c.x + 1.0f)), 1.0f / 2.2f);
            c.y = powf(devClamp01(c.y / (c.y + 1.0f)), 1.0f / 2.2f);
            c.z = powf(devClamp01(c.z / (c.z + 1.0f)), 1.0f / 2.2f);
            optixLaunchParams.colorBuffer[fbIdx] = make_uchar4(
                (unsigned char)(c.x * 255.0f),
                (unsigned char)(c.y * 255.0f),
                (unsigned char)(c.z * 255.0f), 255u);
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
    const float nx = ((float)idx.x + rnd(seed)) / (float)dim.x * 2.0f - 1.0f;
    const float ny = ((float)idx.y + rnd(seed)) / (float)dim.y * 2.0f - 1.0f;

    float3 rayOrig = optixLaunchParams.eye;
    float3 rayDir  = devNormalize(
        optixLaunchParams.U * nx +
        optixLaunchParams.V * ny +
        optixLaunchParams.W);

    // ── Thin-lens depth of field ──────────────────────────────────────────────
    // When lensRadius > 0, offset the ray origin over the aperture disk and
    // redirect the ray to pass through the focal point on the focal plane.
    if (optixLaunchParams.lensRadius > 0.0f)
    {
        // Focal point: intersection of the pinhole ray with the focal plane
        // (plane perpendicular to W at distance focusDistance along W)
        const float  t       = optixLaunchParams.focusDistance
                              / devDot(rayDir, optixLaunchParams.W);
        const float3 focalPt = optixLaunchParams.eye + rayDir * t;

        // Sample the lens disk with optional edge bias.
        // alpha = 0.5 * (1 - bias): alpha=0.5 → sqrt (uniform area), alpha→0 → rim ring.
        // r = R * u^alpha concentrates samples toward the edge as bias → 1.
        const float alpha = 0.5f * (1.0f - devClamp01(optixLaunchParams.bokehEdgeBias));
        const float r     = powf(rnd(seed), fmaxf(alpha, 1e-4f)) * optixLaunchParams.lensRadius;
        const float phi = 2.0f * 3.14159265358979f * rnd(seed);

        // Lens plane axes = normalised camera right (U) and up (V)
        const float3 lensU = devNormalize(optixLaunchParams.U);
        const float3 lensV = devNormalize(optixLaunchParams.V);

        rayOrig = optixLaunchParams.eye
                + lensU * (r * cosf(phi))
                + lensV * (r * sinf(phi));
        rayDir  = devNormalize(focalPt - rayOrig);
    }

    float3 throughput    = make_float3(1.0f, 1.0f, 1.0f);
    float3 radiance      = make_float3(0.0f, 0.0f, 0.0f);
    // PDF of the last BSDF-sampled direction (diffuse only); 0 = specular/glass/first ray.
    // Carried into the next iteration so the miss block can MIS-weight env map samples.
    float  bsdfPdfForMis = 0.0f;

    // Beer-Lambert absorption coefficient: -log(albedo) per unit distance.
    // Zero until a transmissive surface is entered; cleared again on exit.
    float3 absorb = make_float3(0.0f, 0.0f, 0.0f);

    for (int bounce = 0; bounce < MAX_BOUNCES; ++bounce)
    {
        PathVertex vtx;
        vtx.hit = 0;

        uint32_t p0, p1;
        packPointer(&vtx, p0, p1);

        optixTrace(
            optixLaunchParams.traversable,
            rayOrig, rayDir,
            1e-3f, 1e30f, 0.0f,
            OptixVisibilityMask(0xFF),
            OPTIX_RAY_FLAG_NONE,
            0, 1, 0,
            p0, p1);

        // ── Denoiser guide layers: write on first bounce only ────────────────
        if (bounce == 0)
        {
            if (optixLaunchParams.normalBuffer)
            {
                optixLaunchParams.normalBuffer[fbIdx] = vtx.hit
                    ? make_float4(vtx.N.x, vtx.N.y, vtx.N.z, 0.0f)
                    : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
            if (optixLaunchParams.albedoBuffer)
            {
                optixLaunchParams.albedoBuffer[fbIdx] = vtx.hit
                    ? make_float4(vtx.albedo.x, vtx.albedo.y, vtx.albedo.z, 1.0f)
                    : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }

        if (!vtx.hit)
        {
            float3 bg = sampleBackground(rayDir);
            if (optixLaunchParams.envMarginalCdf && bsdfPdfForMis > 0.0f)
            {
                // MIS power heuristic: down-weight the BSDF path when the env map
                // IS would have sampled this direction with a higher density.
                const float pEnv  = evalEnvMapPdf(rayDir);
                const float wBsdf = (pEnv > 0.0f)
                    ? (bsdfPdfForMis * bsdfPdfForMis)
                      / (bsdfPdfForMis * bsdfPdfForMis + pEnv * pEnv)
                    : 1.0f;
                radiance += throughput * bg * wBsdf;
            }
            else
            {
                radiance += throughput * bg;
            }
            break;
        }

        // ── Beer-Lambert absorption ───────────────────────────────────────────
        // Apply exp(-σ·t) for each channel; σ = -log(albedo) set when entering glass.
        throughput.x *= expf(-absorb.x * vtx.t);
        throughput.y *= expf(-absorb.y * vtx.t);
        throughput.z *= expf(-absorb.z * vtx.t);

        // ── Emission ──────────────────────────────────────────────────────────
        radiance += throughput * vtx.emission;

        // ── Lobe selection: clearcoat → specular → diffuse / refraction ─────────
        //
        // Clearcoat is sampled first: it's a dielectric layer (IOR 1.5, F0 = 0.04)
        // on top of the base material.  If not chosen, the base material receives
        // attenuated energy: throughput *= (1 − clearcoat · F_cc).
        //
        // Within the base material:
        //   1. specular   — p_spec (metallic / Fresnel)
        //   2. diffuse    — (1 − p_spec) · (1 − transmission)
        //   3. refraction — (1 − p_spec) ·      transmission

        const float alpha = fmaxf(vtx.roughness * vtx.roughness, 1e-3f);

        // Front-facing normal: always faces the incoming ray.
        const float3 Nf  = (devDot(vtx.N, -rayDir) >= 0.0f) ? vtx.N : -vtx.N;
        const float  cosV = devDot(Nf, -rayDir);  // shared by clearcoat + base Fresnel

        // ── Clearcoat Fresnel (fixed IOR 1.5 → F0 = 0.04) ────────────────────
        const float  cc_alpha = fmaxf(vtx.clearcoatRoughness * vtx.clearcoatRoughness, 1e-3f);
        const float3 cc_F     = devFresnel(cosV, make_float3(0.04f, 0.04f, 0.04f));
        const float  p_coat   = vtx.clearcoat * devLuminance(cc_F);

        if (p_coat > 0.0f && rnd(seed) < p_coat)
        {
            // ── Clearcoat specular (GGX-VNDF) ────────────────────────────────
            float3 Tc, Bc;
            buildONB(Nf, Tc, Bc);
            const float3 Vc       = -rayDir;
            const float3 Vc_local = make_float3(
                devDot(Vc, Tc), devDot(Vc, Bc), fmaxf(1e-4f, devDot(Vc, Nf)));
            const float3 Hc_local = devSampleGGX_VNDF(Vc_local, cc_alpha, rnd(seed), rnd(seed));
            const float3 Hc       = devNormalize(Tc * Hc_local.x + Bc * Hc_local.y + Nf * Hc_local.z);
            const float3 Lc       = devReflect(rayDir, Hc);

            if (devDot(Lc, Nf) <= 0.0f)
            {
                break;
            }

            // cc_F is achromatic (F0 = 0.04 is grey → Schlick stays grey), so
            // cc_F * clearcoat / p_coat = cc_F * clearcoat / (clearcoat * luminance(cc_F)) = 1.
            // The probability selection already carries the energy split; only G1 remains.
            const float cosNLc = fmaxf(1e-4f, devDot(Nf, Lc));
            throughput *= devSmithG1(cosNLc, cc_alpha);
            rayDir        = Lc;
            rayOrig       = vtx.pos + Nf * 1e-3f;
            bsdfPdfForMis = 0.0f;  // clearcoat specular — MIS not applied on miss
        }
        else
        {
            // ── Base material ─────────────────────────────────────────────────
            // (1 - cc_F * clearcoat) / (1 - p_coat) = 1 for achromatic cc_F,
            // so no throughput change is needed here either.

            // Base Fresnel — shared by specular + non-specular lobes
            const float  r0_d  = (1.0f - vtx.ior) / (1.0f + vtx.ior);
            const float3 F0    = devMix(make_float3(r0_d*r0_d, r0_d*r0_d, r0_d*r0_d),
                                        vtx.albedo, vtx.metallic);
            const float3 F     = devFresnel(cosV, F0);

            // Specular probability — metals always specular
            const float p_spec = devClamp01((1.0f - vtx.metallic) * devLuminance(F) + vtx.metallic);

            if (rnd(seed) < p_spec)
            {
                // ── 1. Specular reflection (GGX-VNDF) ─────────────────────────
                float3 T, B;
                buildONB(Nf, T, B);

                const float3 V       = -rayDir;
                const float3 V_local = make_float3(
                    devDot(V, T), devDot(V, B), fmaxf(1e-4f, devDot(V, Nf)));

                const float3 H_local = devSampleGGX_VNDF(V_local, alpha, rnd(seed), rnd(seed));
                const float3 H       = devNormalize(T * H_local.x + B * H_local.y + Nf * H_local.z);
                const float3 L       = devReflect(rayDir, H);

                if (devDot(L, Nf) <= 0.0f)
                {
                    break;
                }

                const float cosNL = fmaxf(1e-4f, devDot(Nf, L));
                throughput *= F * (devSmithG1(cosNL, alpha) / fmaxf(1e-4f, p_spec));
                rayDir        = L;
                rayOrig       = vtx.pos + Nf * 1e-3f;
                bsdfPdfForMis = 0.0f;  // GGX specular — no diffuse NEE on miss
            }
            else
            {
                // Non-specular weight — identical for diffuse and refraction sub-lobes
                const float3 kNS = (make_float3(1.0f, 1.0f, 1.0f) - F) * vtx.albedo
                                   * (1.0f / fmaxf(1e-4f, 1.0f - p_spec));
                throughput *= kNS;

                const float p_trans = devClamp01(vtx.transmission);

                if (rnd(seed) > p_trans)
                {
                    // ── 2. Diffuse (Lambertian, cosine-weighted) ───────────────
                    float3 T, B;
                    buildONB(Nf, T, B);

                    // ── NEE: direct env-map lighting ──────────────────────────
                    // Sample the env map IS distribution, fire a shadow ray to
                    // check visibility, and add a MIS-weighted direct contribution.
                    if (optixLaunchParams.envMarginalCdf)
                    {
                        float3 neeDir; float neePdf;
                        float3 neeL = sampleEnvMapIS(rnd(seed), rnd(seed), neeDir, neePdf);

                        const float cosL = devDot(neeDir, Nf);
                        if (cosL > 0.0f && neePdf > 0.0f)
                        {
                            // Shadow ray: SBT miss index 1 → __miss__shadow sets p0=1
                            uint32_t vis = 0u, dummy = 0u;
                            optixTrace(
                                optixLaunchParams.traversable,
                                vtx.pos + Nf * 1e-3f, neeDir,
                                1e-3f, 1e30f, 0.0f,
                                OptixVisibilityMask(0xFF),
                                OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT
                                | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                                0, 1, 1,   // offset=0, stride=1, missIndex=1
                                vis, dummy);

                            if (vis)
                            {
                                // throughput already carries kNS = (1-F)*albedo/(1-p_spec)
                                // Lambertian: f×cosθ = kNS * (cosθ/π) evaluated at neeDir
                                const float kInvPi = 0.31830988618f;
                                const float pBsdf  = (1.0f - p_coat) * (1.0f - p_spec)
                                                   * (1.0f - p_trans) * cosL * kInvPi;
                                const float wNee   = (neePdf * neePdf)
                                                   / (neePdf * neePdf + pBsdf * pBsdf);
                                radiance += throughput * (cosL * kInvPi / neePdf) * neeL * wNee;
                            }
                        }
                    }

                    // ── Continue: cosine-sample the hemisphere for the path ────
                    const float3 d = cosineSampleHemisphere(rnd(seed), rnd(seed));
                    rayDir  = devNormalize(T * d.x + B * d.y + Nf * d.z);
                    rayOrig = vtx.pos + Nf * 1e-3f;

                    // Store BSDF PDF for MIS in next-iteration miss block
                    const float kInvPi = 0.31830988618f;
                    bsdfPdfForMis = devDot(rayDir, Nf) * kInvPi;
                }
                else
                {
                    bsdfPdfForMis = 0.0f;  // refraction — no diffuse NEE on next hit
                // ── 3. Refraction (rough Snell's law via GGX microfacet) ─────────
                // vtx.N is the true outward geometry normal — used only for the
                // entering/exiting test, not for the microfacet direction.
                const bool   entering = (devDot(rayDir, vtx.N) < 0.0f);
                const float3 faceN    = entering ? vtx.N : -vtx.N;
                const float  eta      = entering ? (1.0f / vtx.ior) : vtx.ior;

                // Sample a GGX microfacet normal (same VNDF as the specular lobe).
                // At alpha → 0 this converges to the macro normal → smooth glass.
                // At higher alpha the half-vector is scattered → frosted glass.
                float3 Tt, Bt;
                buildONB(Nf, Tt, Bt);
                const float3 Vt       = -rayDir;
                const float3 Vt_local = make_float3(
                    devDot(Vt, Tt), devDot(Vt, Bt), fmaxf(1e-4f, devDot(Vt, Nf)));
                const float3 Ht_local = devSampleGGX_VNDF(Vt_local, alpha, rnd(seed), rnd(seed));
                const float3 Ht       = devNormalize(Tt * Ht_local.x + Bt * Ht_local.y + Nf * Ht_local.z);

                // Refract through the microfacet normal; TIR falls back to reflection.
                const bool refracted = devRefract(rayDir, Ht, eta, rayDir);

                // Smith G1 on the transmitted direction — mirrors the specular weight.
                if (refracted)
                {
                    // cosine against the far-side normal (-faceN)
                    const float cosT = fmaxf(1e-4f, devDot(rayDir, -faceN));
                    throughput *= devSmithG1(cosT, alpha);
                }

                // Update Beer-Lambert absorption state.
                // On true refraction: entering glass starts absorbing, exiting stops.
                // On TIR: stay in the same medium — absorb is unchanged.
                if (refracted)
                {
                    if (entering)
                    {
                        // σ = -log(albedo) / absorptionDistance
                        // → exp(-σ·t) = albedo^(t / absorptionDistance)
                        const float invDist = 1.0f / vtx.absorptionDistance;
                        absorb.x = -logf(fmaxf(vtx.albedo.x, 1e-6f)) * invDist;
                        absorb.y = -logf(fmaxf(vtx.albedo.y, 1e-6f)) * invDist;
                        absorb.z = -logf(fmaxf(vtx.albedo.z, 1e-6f)) * invDist;
                    }
                    else
                    {
                        absorb = make_float3(0.0f, 0.0f, 0.0f);
                    }
                }

                // On TIR devRefract writes the reflected direction; offset stays on
                // the same side.  On true refraction offset to the far side.
                rayOrig = vtx.pos + faceN * (refracted ? -1e-3f : 1e-3f);
                }   // end refraction else
            }   // end non-specular else (diffuse / refraction)
        }   // end base material else (specular / diffuse / refraction) = clearcoat else

        // Russian roulette: stochastically terminate dim paths (all branches)
        if (bounce >= 3)
        {
            const float maxThr = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
            if (rnd(seed) > maxThr)
            {
                break;
            }
            throughput *= 1.0f / fmaxf(maxThr, 1e-6f);
        }
    }

    // ── Accumulate ────────────────────────────────────────────────────────────
    float4& acc = optixLaunchParams.accumBuffer[fbIdx];
    acc.x += radiance.x;
    acc.y += radiance.y;
    acc.z += radiance.z;

    // ── HDR average for denoiser ──────────────────────────────────────────────
    const float  inv = 1.0f / (float)(optixLaunchParams.sampleIndex + 1);
    if (optixLaunchParams.hdrBuffer)
    {
        optixLaunchParams.hdrBuffer[fbIdx] =
            make_float4(acc.x * inv, acc.y * inv, acc.z * inv, 1.0f);
    }

    // ── Tone-map and gamma-encode ─────────────────────────────────────────────
    float3 avg = make_float3(acc.x * inv, acc.y * inv, acc.z * inv);

    // Reinhard: [0, ∞) → [0, 1)
    avg.x = avg.x / (avg.x + 1.0f);
    avg.y = avg.y / (avg.y + 1.0f);
    avg.z = avg.z / (avg.z + 1.0f);

    // Linear → sRGB (γ ≈ 1/2.2)
    avg.x = powf(devClamp01(avg.x), 1.0f / 2.2f);
    avg.y = powf(devClamp01(avg.y), 1.0f / 2.2f);
    avg.z = powf(devClamp01(avg.z), 1.0f / 2.2f);

    optixLaunchParams.colorBuffer[fbIdx] = make_uchar4(
        (unsigned char)(avg.x * 255.0f),
        (unsigned char)(avg.y * 255.0f),
        (unsigned char)(avg.z * 255.0f),
        255u);
}

// ─── Miss ─────────────────────────────────────────────────────────────────────

extern "C" __global__ void __miss__radiance()
{
    // intentionally empty — hit==0 default in raygen is sufficient
}

// ─── Miss — NEE shadow ray ────────────────────────────────────────────────────
// Fires when a shadow ray reaches the background unoccluded.
// Sets p0 = 1 (visible); the raygen reads this to confirm the light path is clear.

extern "C" __global__ void __miss__shadow()
{
    optixSetPayload_0(1u);
}

// ─── Scene texture sampling helper ───────────────────────────────────────────

// Sample a scene texture at the barycentric-interpolated UV for the current hit.
// Returns true and writes the sampled float4 to `out` when all of the following hold:
//   • index >= 0 (a texture is assigned)
//   • optixLaunchParams.sceneTextures is non-null
//   • uvs is non-null (the mesh has UV data)
// Returns false and leaves `out` unchanged otherwise.
// uvTransform: xy = tiling (scale), zw = offset.
static __forceinline__ __device__
bool sampleSceneTex(int index, const float2* uvs, const uint3& tri,
                    float w0, float2 bc, float4 uvTransform, float4& out)
{
    if (index < 0 || !optixLaunchParams.sceneTextures || !uvs)
        return false;

    // Interpolate raw mesh UV, then apply tiling and offset.
    const float2 uvRaw = make_float2(
        uvs[tri.x].x * w0 + uvs[tri.y].x * bc.x + uvs[tri.z].x * bc.y,
        uvs[tri.x].y * w0 + uvs[tri.y].y * bc.x + uvs[tri.z].y * bc.y);
    const float2 uv = make_float2(
        uvRaw.x * uvTransform.x + uvTransform.z,
        uvRaw.y * uvTransform.y + uvTransform.w);

    out = tex2D<float4>(optixLaunchParams.sceneTextures[index], uv.x, uv.y);
    return true;
}

// ─── Closest-hit ─────────────────────────────────────────────────────────────

extern "C" __global__ void __closesthit__radiance()
{
    PathVertex* vtx = unpackVertex(optixGetPayload_0(), optixGetPayload_1());

    const MeshData& mesh = *reinterpret_cast<const MeshData*>(optixGetSbtDataPointer());

    // ── Interpolated shading normal ───────────────────────────────────────────
    const uint3  tri = mesh.indices[optixGetPrimitiveIndex()];
    const float2 bc  = optixGetTriangleBarycentrics();
    const float  w0  = 1.0f - bc.x - bc.y;

    const float3 n_obj = devNormalize(
        mesh.normals[tri.x] * w0  +
        mesh.normals[tri.y] * bc.x +
        mesh.normals[tri.z] * bc.y);

    vtx->N = devNormalize(optixTransformNormalFromObjectToWorldSpace(n_obj));

    // ── World-space hit point and travel distance ─────────────────────────────
    vtx->t   = optixGetRayTmax();
    vtx->pos = optixGetWorldRayOrigin()
             + optixGetWorldRayDirection() * vtx->t;

    // ── Material ──────────────────────────────────────────────────────────────
    if (optixLaunchParams.materials && mesh.materialIndex >= 0)
    {
        const MaterialData& mat = optixLaunchParams.materials[mesh.materialIndex];
        // Albedo — base colour × texture tint (glTF: baseColorFactor × baseColorTexture)
        vtx->albedo = mat.albedo;
        float4 texSample;
        if (sampleSceneTex(mat.albedoTexture, mesh.uvs, tri, w0, bc, mat.uvTransform, texSample))
            vtx->albedo = make_float3(vtx->albedo.x * texSample.x,
                                      vtx->albedo.y * texSample.y,
                                      vtx->albedo.z * texSample.z);

        // Roughness — red channel × roughness factor
        vtx->roughness = mat.roughness;
        if (sampleSceneTex(mat.roughnessTexture, mesh.uvs, tri, w0, bc, mat.uvTransform, texSample))
            vtx->roughness *= texSample.x;

        vtx->metallic      = mat.metallic;
        vtx->emission      = mat.emission * mat.emissionScale;
        vtx->transmission       = mat.transmission;
        vtx->ior                = mat.ior;
        vtx->absorptionDistance = fmaxf(mat.absorptionDistance, 1e-4f);
        vtx->clearcoat          = mat.clearcoat;
        vtx->clearcoatRoughness = mat.clearcoatRoughness;
    }
    else
    {
        vtx->albedo             = make_float3(0.8f, 0.8f, 0.8f);
        vtx->roughness          = 0.5f;
        vtx->metallic           = 0.0f;
        vtx->emission           = make_float3(0.0f, 0.0f, 0.0f);
        vtx->transmission       = 0.0f;
        vtx->ior                = 1.5f;
        vtx->absorptionDistance = 1.0f;
        vtx->clearcoat          = 0.0f;
        vtx->clearcoatRoughness = 0.0f;
    }

    vtx->hit = 1;
}
