#include "application.h"

#include <filesystem>
#include <iostream>
#include <stdexcept>

int main(int /*argc*/, char** argv)
{
    try
    {
        // Resolve the directory that contains the compiled PTX shaders.
        // The build system copies device_programs.ptx next to the executable.
        const std::string ptxDir =
            std::filesystem::path(argv[0]).parent_path().string();

        Application app(1920, 1080, "OptiX Raytracer", ptxDir);
        while (app.tick())
        {
            // All per-frame work happens inside tick().
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Fatal: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
