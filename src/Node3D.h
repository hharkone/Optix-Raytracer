#ifndef OPTIX_RAYTRACER_NODE3D_H
#define OPTIX_RAYTRACER_NODE3D_H

#include "Matrix4x4.h"
#include <memory>
#include <string>
#include <vector>

// ─── Base class ───────────────────────────────────────────────────────────────
// Every node in the scene graph carries a local transform, a name, and
// parent/child links (by integer index into Scene::m_nodes).

class Node3D
{
public:
    std::string      name;
    Matrix4x4        localTransform;  // node-to-parent space
    int              parent   = -1;   // index in Scene::m_nodes; -1 = root
    std::vector<int> children;        // indices in Scene::m_nodes

    virtual ~Node3D() = default;

    // Short type label shown in the Scene Graph UI.
    virtual const char* typeName() const = 0;
};

// ─── Concrete node types ──────────────────────────────────────────────────────

// A node that owns one or more mesh primitives (one per glTF primitive).
class MeshNode : public Node3D
{
public:
    std::vector<int> meshIndices;  // indices into Scene::m_meshes
    const char* typeName() const override { return "Mesh"; }
};

// A node that positions the scene camera.
class CameraNode : public Node3D
{
public:
    const char* typeName() const override { return "Camera"; }
};

// A transform-only group node (no geometry or camera attached).
class GroupNode : public Node3D
{
public:
    const char* typeName() const override { return "Group"; }
};

#endif // OPTIX_RAYTRACER_NODE3D_H
