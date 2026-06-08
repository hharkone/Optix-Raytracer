#ifndef OPTIX_RAYTRACER_MATRIX4X4_H
#define OPTIX_RAYTRACER_MATRIX4X4_H

#include <cmath>      // fabsf
#include <algorithm>  // std::swap

// Row-major 4×4 matrix. Element access: m[row][col].
struct Matrix4x4
{
    float m[4][4] = {};
};

inline Matrix4x4 mat4Identity()
{
    Matrix4x4 r;
    r.m[0][0] = r.m[1][1] = r.m[2][2] = r.m[3][3] = 1.0f;
    return r;
}

inline Matrix4x4 mat4Multiply(const Matrix4x4& a, const Matrix4x4& b)
{
    Matrix4x4 r;
    for (int row = 0; row < 4; ++row)
        for (int col = 0; col < 4; ++col)
            for (int k = 0; k < 4; ++k)
                r.m[row][col] += a.m[row][k] * b.m[k][col];
    return r;
}

// General 4×4 matrix inverse via Gauss-Jordan elimination.
// Returns the identity matrix if the input is singular (determinant ≈ 0).
inline Matrix4x4 mat4Inverse(const Matrix4x4& in)
{
    float A[4][4], I[4][4];
    for (int r = 0; r < 4; ++r)
    {
        for (int c = 0; c < 4; ++c)
        {
            A[r][c] = in.m[r][c];
            I[r][c] = (r == c) ? 1.f : 0.f;
        }
    }

    for (int col = 0; col < 4; ++col)
    {
        int pivot = col;
        for (int row = col + 1; row < 4; ++row)
        {
            if (fabsf(A[row][col]) > fabsf(A[pivot][col]))
            {
                pivot = row;
            }
        }

        if (pivot != col)
        {
            for (int k = 0; k < 4; ++k)
            {
                std::swap(A[col][k], A[pivot][k]);
                std::swap(I[col][k], I[pivot][k]);
            }
        }

        const float diag = A[col][col];
        if (fabsf(diag) < 1e-8f)
        {
            return mat4Identity();
        }

        const float inv = 1.f / diag;
        for (int k = 0; k < 4; ++k)
        {
            A[col][k] *= inv;
            I[col][k] *= inv;
        }

        for (int row = 0; row < 4; ++row)
        {
            if (row == col)
            {
                continue;
            }
            const float f = A[row][col];
            for (int k = 0; k < 4; ++k)
            {
                A[row][k] -= f * A[col][k];
                I[row][k] -= f * I[col][k];
            }
        }
    }

    Matrix4x4 result;
    for (int r = 0; r < 4; ++r)
    {
        for (int c = 0; c < 4; ++c)
        {
            result.m[r][c] = I[r][c];
        }
    }
    return result;
}

// Fast inverse for pure rigid-body transforms (rotation + translation, no scale).
// Uses the analytical shortcut: R^-1 = R^T, t^-1 = -R^T * t.
// Faster than mat4Inverse() and exact for camera / node transforms.
inline Matrix4x4 mat4RigidInverse(const Matrix4x4& m)
{
    Matrix4x4 inv;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            inv.m[r][c] = m.m[c][r];  // transpose the 3×3 rotation block

    const float tx = m.m[0][3], ty = m.m[1][3], tz = m.m[2][3];
    inv.m[0][3] = -(inv.m[0][0]*tx + inv.m[0][1]*ty + inv.m[0][2]*tz);
    inv.m[1][3] = -(inv.m[1][0]*tx + inv.m[1][1]*ty + inv.m[1][2]*tz);
    inv.m[2][3] = -(inv.m[2][0]*tx + inv.m[2][1]*ty + inv.m[2][2]*tz);
    inv.m[3][3] = 1.0f;
    return inv;
}

// Convert our row-major Matrix4x4 to a column-major float[16] array as expected
// by ImGuizmo (and OpenGL / GLM).  Mathematically this is a transpose.
inline void mat4ToColMajor(const Matrix4x4& m, float out[16])
{
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            out[c * 4 + r] = m.m[r][c];
}

// Convert a column-major float[16] (from ImGuizmo) back to our row-major Matrix4x4.
inline Matrix4x4 mat4FromColMajor(const float in[16])
{
    Matrix4x4 m;
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            m.m[r][c] = in[c * 4 + r];
    return m;
}

#endif // OPTIX_RAYTRACER_MATRIX4X4_H
