// UI fragment shader for the scRGB ImGui pipeline.
//
// ImGui authors all its colours (style, widgets, the colour-picker gradients)
// as display-referred sRGB.  On a linear scRGB (FP16) swapchain those values
// must be linearised and scaled to paper white or the whole UI renders washed
// out and dim.  Doing the conversion here — after vertex-attribute
// interpolation — also makes gradients interpolate in sRGB space, exactly
// matching how they look on a regular sRGB swapchain.
//
// Textures are multiplied in unconverted: the renderer's viewport image is
// already linear (tinted by white -> just scaled to paper white), and sRGB
// UI textures use VK_FORMAT_*_SRGB views so the sampler linearises them.
//
// uUiScale = paperWhiteNits / 80 (scRGB: 1.0 = 80 nits), baked in as a
// specialization constant — the pipeline is recreated when the slider moves.
//
// Compiled to SPIR-V embedded in src/ui_scrgb_spv.h; regenerate with:
//   glslc -O -o ui_scrgb.frag.spv ui_scrgb.frag   (+ array conversion, see header)
#version 450 core
layout(location = 0) out vec4 fColor;
layout(set = 0, binding = 0) uniform texture2D _Texture;
layout(set = 1, binding = 0) uniform sampler   _Sampler;
layout(location = 0) in struct { vec4 Color; vec2 UV; } In;
layout(constant_id = 0) const float uUiScale = 1.0;

void main()
{
    vec4 c = In.Color;
    c.rgb  = pow(c.rgb, vec3(2.2)) * uUiScale;
    fColor = c * texture(sampler2D(_Texture, _Sampler), In.UV.st);
}
