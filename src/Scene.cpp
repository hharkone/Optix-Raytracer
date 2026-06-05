#include "Scene.h"
#include "Matrix4x4.h"

#include <stdexcept>
#include <utility>
#include <vector>

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

std::vector<MaterialData>& Scene::materials()
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
    {
        throw std::out_of_range("Scene::materialName index out of range");
    }
    return m_materialNames[index];
}

const Camera& Scene::camera() const
{
    return m_camera;
}

void Scene::setCamera(Camera camera)
{
    m_camera = std::move(camera);
}

int Scene::addNode(std::unique_ptr<Node3D> node)
{
    m_nodes.push_back(std::move(node));
    return static_cast<int>(m_nodes.size()) - 1;
}

void Scene::addRootNode(int index)
{
    m_rootNodes.push_back(index);
}

Node3D& Scene::nodeAt(int index)
{
    return *m_nodes[index];
}

const std::vector<std::unique_ptr<Node3D>>& Scene::nodes() const
{
    return m_nodes;
}

const std::vector<int>& Scene::rootNodes() const
{
    return m_rootNodes;
}

void Scene::clear()
{
    m_meshes.clear();
    m_materials.clear();
    m_materialNames.clear();
    m_textures.clear();
    m_nodes.clear();
    m_rootNodes.clear();
    m_camera = Camera::makeDefault();
}

bool Scene::empty() const
{
    return m_meshes.empty();
}

Matrix4x4 Scene::computeWorldTransform(int nodeIdx) const
{
    // Build the ancestor chain from nodeIdx up to the root.
    std::vector<int> chain;
    for (int i = nodeIdx; i >= 0; i = m_nodes[i]->parent)
        chain.push_back(i);

    // Multiply from root → node (reverse order).
    Matrix4x4 world = mat4Identity();
    for (int i = static_cast<int>(chain.size()) - 1; i >= 0; --i)
        world = mat4Multiply(world, m_nodes[chain[i]]->localTransform);
    return world;
}
