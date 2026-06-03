#ifndef OPTIX_RAYTRACER_MATH_H
#define OPTIX_RAYTRACER_MATH_H

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
    {
        for (int col = 0; col < 4; ++col)
        {
            for (int k = 0; k < 4; ++k)
            {
                r.m[row][col] += a.m[row][k] * b.m[k][col];
            }
        }
    }
    return r;
}

#endif // OPTIX_RAYTRACER_MATH_H
