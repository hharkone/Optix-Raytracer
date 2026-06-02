#ifndef OPTIX_RAYTRACER_MATH_H
#define OPTIX_RAYTRACER_MATH_H

// Row-major 4×4 matrix. Element access: m[row][col].
struct Matrix4x4
{
    float m[4][4] = {};
};

#endif // OPTIX_RAYTRACER_MATH_H
