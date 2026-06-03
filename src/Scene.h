#ifndef OPTIX_RAYTRACER_SCENE_H
#define OPTIX_RAYTRACER_SCENE_H

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

    void clear();        // remove all meshes, materials, textures, nodes, and reset camera
    bool empty() const;  // true when there are no meshes

private:
    std::vector<Mesh>         m_meshes;
    std::vector<MaterialData> m_materials;
    std::vector<std::string>  m_materialNames;  // parallel to m_materials
    std::vector<Texture>      m_textures;
    Camera                    m_camera = Camera::makeDefault();

    std::vector<std::unique_ptr<Node3D>> m_nodes;
    std::vector<int>                     m_rootNodes;
};

#endif // OPTIX_RAYTRACER_SCENE_H
