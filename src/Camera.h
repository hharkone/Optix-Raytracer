#ifndef OPTIX_RAYTRACER_CAMERA_H
#define OPTIX_RAYTRACER_CAMERA_H

#include "Math.h"
#include <cmath>
#include <string>

struct Camera
{
    Matrix4x4   transform;                // camera-to-world (position + orientation in world space)
    Matrix4x4   view;                     // world-to-camera (inverse of transform; OpenGL "view matrix")
    float       yFov        = 0.872665f;  // vertical field of view, radians (~50 deg)
    float       aspectRatio = 16.f / 9.f;
    float       zNear       = 0.01f;
    float       zFar        = 10000.f;
    std::string name;

    // Returns a default camera at (0, 0, 3) looking toward the origin along -Z.
    static Camera makeDefault();
};

inline Camera Camera::makeDefault()
{
    Camera cam;
    cam.name = "Default";

    // Camera-to-world: identity rotation + translation (0, 0, 3).
    cam.transform.m[0][0] = 1.f;
    cam.transform.m[1][1] = 1.f;
    cam.transform.m[2][2] = 1.f;
    cam.transform.m[2][3] = 3.f;
    cam.transform.m[3][3] = 1.f;

    // View (world-to-camera) = T(0, 0, -3)
    cam.view.m[0][0] = 1.f;
    cam.view.m[1][1] = 1.f;
    cam.view.m[2][2] = 1.f;
    cam.view.m[2][3] = -3.f;
    cam.view.m[3][3] = 1.f;

    return cam;
}

#endif // OPTIX_RAYTRACER_CAMERA_H
