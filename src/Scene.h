#ifndef OPTIX_RAYTRACER_SCENE_H
#define OPTIX_RAYTRACER_SCENE_H

#include "Accel.h"
#include "Camera.h"
#include "Mesh.h"
#include "Node3D.h"
#include "Texture.h"
#include "SceneData.h"  // MaterialData — shared with device code; on include path via CMake

#include <memory>
#include <vector>
#include <string>

class Scene
{
public:
    Scene()  = default;
    ~Scene() = default;

    Scene(const Scene&)            = delete;
    Scene& operator=(const Scene&) = delete;
    Scene(Scene&&)                 = default;
    Scene& operator=(Scene&&)      = default;

    // Returns the index of the added item.
    int addMesh(Mesh mesh);
    int addMaterial(MaterialData material, std::string name = {});
    int addTexture(Texture texture);

    const std::vector<Mesh>&         meshes()                   const;
    const std::vector<MaterialData>& materials()                const;
    std::vector<MaterialData>&       materials();               // mutable for in-place editing
    const std::vector<Texture>&      textures()                 const;
    const std::string&               materialName(int index)    const;

    const Camera& camera()             const;
    void          setCamera(Camera camera);

    // Scene graph — glTF node hierarchy preserved at load time.
    int     addNode(std::unique_ptr<Node3D> node);  // takes ownership; returns index
    void    addRootNode(int index);
    Node3D& nodeAt(int index);                       // mutable access for child-link wiring

    const std::vector<std::unique_ptr<Node3D>>& nodes()     const;
    const std::vector<int>&                     rootNodes() const;

    // Accumulate transforms from the root down to nodeIdx (world = parent × … × local).
    Matrix4x4 computeWorldTransform(int nodeIdx) const;

    // ── Acceleration structure ────────────────────────────────────────────────
    // The Accel is derived entirely from the scene's geometry and node hierarchy,
    // so Scene owns it.  All methods require an OptiX device context.

    // Build (or rebuild) all BLASes and the TLAS from the current scene data.
    // Destroys any previous acceleration structure first.
    // Throws std::runtime_error on CUDA / OptiX failure.
    void buildAccel(OptixDeviceContext ctx);

    // Rebuild only the TLAS from the current node world-space transforms.
    // BLASes and device geometry buffers are reused unchanged.
    // Much cheaper than buildAccel() — use after live node transform edits.
    void rebuildTlas(OptixDeviceContext ctx);

    // Free all GPU acceleration-structure memory.
    // Called automatically by clear() and in Application's destructor
    // (before the OptiX context is destroyed).
    void destroyAccel();

    bool                   hasAccel()    const;  // true when built and valid
    OptixTraversableHandle traversable() const;  // TLAS handle; 0 when not built

    // Per-mesh device pointers for filling SBT hit group records.
    Accel::MeshDevicePtrs meshDevicePtrs(size_t meshIdx) const;

    void clear();        // remove all scene data and free the acceleration structure
    bool empty() const;  // true when there are no meshes

private:
    std::vector<Mesh>         m_meshes;
    std::vector<MaterialData> m_materials;
    std::vector<std::string>  m_materialNames;  // parallel to m_materials
    std::vector<Texture>      m_textures;
    Camera                    m_camera = Camera::makeDefault();

    std::vector<std::unique_ptr<Node3D>> m_nodes;
    std::vector<int>                     m_rootNodes;

    std::unique_ptr<Accel> m_accel;  // null until buildAccel() is called
};

#endif // OPTIX_RAYTRACER_SCENE_H
