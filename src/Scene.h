#ifndef OPTIX_RAYTRACER_SCENE_H
#define OPTIX_RAYTRACER_SCENE_H

#include "Mesh.h"
#include "Texture.h"
#include "SceneData.h"  // MaterialData — shared with device code; on include path via CMake

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
    const std::vector<Texture>&      textures()                 const;
    const std::string&               materialName(int index)    const;

    bool empty() const;  // true when there are no meshes

private:
    std::vector<Mesh>         m_meshes;
    std::vector<MaterialData> m_materials;
    std::vector<std::string>  m_materialNames;  // parallel to m_materials
    std::vector<Texture>      m_textures;
};

#endif // OPTIX_RAYTRACER_SCENE_H
