// UI vertex shader for the scRGB ImGui pipeline — identical interface to the
// stock imgui_impl_vulkan vertex shader (backends/vulkan/glsl_shader.vert).
// Compiled to SPIR-V embedded in src/ui_scrgb_spv.h; regenerate with:
//   glslc -O -o ui_scrgb.vert.spv ui_scrgb.vert   (+ array conversion, see header)
#version 450 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aUV;
layout(location = 2) in vec4 aColor;
layout(push_constant) uniform uPushConstant { vec2 uScale; vec2 uTranslate; } pc;

out gl_PerVertex { vec4 gl_Position; };
layout(location = 0) out struct { vec4 Color; vec2 UV; } Out;

void main()
{
    Out.Color = aColor;
    Out.UV = aUV;
    gl_Position = vec4(aPos * pc.uScale + pc.uTranslate, 0, 1);
}
