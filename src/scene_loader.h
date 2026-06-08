#ifndef OPTIX_RAYTRACER_SCENE_LOADER_H
#define OPTIX_RAYTRACER_SCENE_LOADER_H

#include "scene.h"

#include <string>

// Loads a glTF 2.0 file (.gltf or .glb) and appends its content to outScene.
// Returns true on success. On failure, outError receives a human-readable message.
// Appends rather than replaces so multiple files can be accumulated into one scene.
bool loadGltfFile(const std::string& path, Scene& outScene, std::string& outError);

#endif // OPTIX_RAYTRACER_SCENE_LOADER_H
