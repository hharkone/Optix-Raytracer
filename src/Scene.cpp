#include "Scene.h"

#include <stdexcept>
#include <utility>

int Scene::addMesh(Mesh mesh)
{
    m_meshes.push_back(std::move(mesh));
    return static_cast<int>(m_meshes.size()) - 1;
}

int Scene::addMaterial(MaterialData material, std::string name)
{
    m_materials.push_back(material);
    m_materialNames.push_back(std::move(name));
    return static_cast<int>(m_materials.size()) - 1;
}

int Scene::addTexture(Texture texture)
{
    m_textures.push_back(std::move(texture));
    return static_cast<int>(m_textures.size()) - 1;
}

const std::vector<Mesh>& Scene::meshes() const
{
    return m_meshes;
}

const std::vector<MaterialData>& Scene::materials() const
{
    return m_materials;
}

const std::vector<Texture>& Scene::textures() const
{
    return m_textures;
}

const std::string& Scene::materialName(int index) const
{
    if (index < 0 || index >= static_cast<int>(m_materialNames.size()))
        throw std::out_of_range("Scene::materialName index out of range");
    return m_materialNames[index];
}

void Scene::clear()
{
    m_meshes.clear();
    m_materials.clear();
    m_materialNames.clear();
    m_textures.clear();
}

bool Scene::empty() const
{
    return m_meshes.empty();
}
