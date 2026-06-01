#ifndef OPTIX_RAYTRACER_DEVICE_MATH_H
#define OPTIX_RAYTRACER_DEVICE_MATH_H

// float3 operator overloads.
//
// float3 is a plain struct — C++ allows operator overloads for any struct type.
// The DMATH_FN macro resolves to __forceinline__ __host__ __device__ when
// compiled by nvcc (device code) and to plain inline otherwise (host code),
// so this header is safe to include from both .cu and .cpp translation units.

#include <cuda_runtime.h>

#ifdef __CUDACC__
#  define DMATH_FN __forceinline__ __host__ __device__
#else
#  define DMATH_FN inline
#endif

// ─── Binary arithmetic ────────────────────────────────────────────────────────

DMATH_FN float3 operator+(float3 a, float3 b) { return make_float3(a.x+b.x, a.y+b.y, a.z+b.z); }
DMATH_FN float3 operator-(float3 a, float3 b) { return make_float3(a.x-b.x, a.y-b.y, a.z-b.z); }
DMATH_FN float3 operator*(float3 a, float3 b) { return make_float3(a.x*b.x, a.y*b.y, a.z*b.z); }
DMATH_FN float3 operator*(float3 a, float  s) { return make_float3(a.x*s,   a.y*s,   a.z*s  ); }
DMATH_FN float3 operator*(float  s, float3 a) { return make_float3(a.x*s,   a.y*s,   a.z*s  ); }
DMATH_FN float3 operator/(float3 a, float  s) { return make_float3(a.x/s,   a.y/s,   a.z/s  ); }

// ─── Unary negation ───────────────────────────────────────────────────────────

DMATH_FN float3 operator-(float3 a) { return make_float3(-a.x, -a.y, -a.z); }

// ─── Compound assignment ─────────────────────────────────────────────────────

DMATH_FN float3& operator+=(float3& a, float3 b) { a.x += b.x; a.y += b.y; a.z += b.z; return a; }
DMATH_FN float3& operator-=(float3& a, float3 b) { a.x -= b.x; a.y -= b.y; a.z -= b.z; return a; }
DMATH_FN float3& operator*=(float3& a, float3 b) { a.x *= b.x; a.y *= b.y; a.z *= b.z; return a; }
DMATH_FN float3& operator*=(float3& a, float  s) { a.x *= s;   a.y *= s;   a.z *= s;   return a; }

#undef DMATH_FN

#endif // OPTIX_RAYTRACER_DEVICE_MATH_H
