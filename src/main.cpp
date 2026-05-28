#include "Application.h"

#include <iostream>
#include <stdexcept>

int main(int /*argc*/, char** /*argv*/)
{
    try {
        Application app(1280, 720, "OptiX Raytracer");
        while (app.tick()) {}
    }
    catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
