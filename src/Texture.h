#pragma once

#include <cstdint>
#include <vector>
#include <string>

struct Texture
{
    std::vector<uint8_t> pixels;
    int                  width    = 0;
    int                  height   = 0;
    int                  channels = 4;  // RGBA
    std::string          name;
};
