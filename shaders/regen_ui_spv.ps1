# Recompile the scRGB ImGui UI shaders and regenerate src/ui_scrgb_spv.h.
# Run from the repo root:  .\shaders\regen_ui_spv.ps1
# Requires the Vulkan SDK to be installed (glslc is found via the registry path below).

param(
    [string]$GlslcPath = ""
)

$ErrorActionPreference = "Stop"

# ── Locate glslc ──────────────────────────────────────────────────────────────

if (-not $GlslcPath)
{
    # Try PATH first, then fall back to the newest VulkanSDK install.
    $found = Get-Command glslc -ErrorAction SilentlyContinue
    if ($found)
    {
        $GlslcPath = $found.Source
    }
    else
    {
        $sdkBin = Get-ChildItem "C:\VulkanSDK" -Filter "glslc.exe" -Recurse -ErrorAction SilentlyContinue |
                  Sort-Object FullName -Descending |
                  Select-Object -First 1
        if (-not $sdkBin)
        {
            Write-Error "glslc not found. Install the Vulkan SDK or pass -GlslcPath."
        }
        $GlslcPath = $sdkBin.FullName
    }
}
Write-Host "glslc: $GlslcPath"

# ── Paths ─────────────────────────────────────────────────────────────────────

$repo      = Split-Path $PSScriptRoot -Parent
$shaderDir = Join-Path $repo "shaders"
$outFile   = Join-Path $repo "src\ui_scrgb_spv.h"

$vert      = Join-Path $shaderDir "ui_scrgb.vert"
$frag      = Join-Path $shaderDir "ui_scrgb.frag"
$vertSpv   = Join-Path $shaderDir "ui_scrgb.vert.spv"
$fragSpv   = Join-Path $shaderDir "ui_scrgb.frag.spv"

# ── Compile ───────────────────────────────────────────────────────────────────

Write-Host "Compiling $vert ..."
& $GlslcPath -O $vert -o $vertSpv
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "Compiling $frag ..."
& $GlslcPath -O $frag -o $fragSpv
if ($LASTEXITCODE -ne 0) { exit 1 }

# ── Convert SPV → C uint32_t array ───────────────────────────────────────────

function ConvertTo-CArray
{
    param([string]$spvPath, [string]$arrayName)

    $bytes = [System.IO.File]::ReadAllBytes($spvPath)
    if ($bytes.Length % 4 -ne 0)
    {
        Write-Error "SPIR-V file size is not a multiple of 4: $spvPath"
    }

    $wordCount = $bytes.Length / 4
    $lines     = [System.Collections.Generic.List[string]]::new()
    $chunk     = [System.Collections.Generic.List[string]]::new()

    for ($i = 0; $i -lt $wordCount; $i++)
    {
        $word = [System.BitConverter]::ToUInt32($bytes, $i * 4)
        $chunk.Add("0x{0:x8}" -f $word)

        if ($chunk.Count -eq 8 -or $i -eq ($wordCount - 1))
        {
            # trailing comma on every line — last entry has it too (valid C++)
            $lines.Add("    " + ($chunk -join ",") + ",")
            $chunk.Clear()
        }
    }

    return "static const uint32_t ${arrayName}[] =`r`n{`r`n" +
           ($lines -join "`r`n") +
           "`r`n};"
}

Write-Host "Converting SPIR-V to C arrays ..."
$vertArray = ConvertTo-CArray -spvPath $vertSpv -arrayName "kUiScrgbVertSpv"
$fragArray = ConvertTo-CArray -spvPath $fragSpv -arrayName "kUiScrgbFragSpv"

# ── Write header ──────────────────────────────────────────────────────────────

$header = @"
// Generated SPIR-V for the scRGB ImGui UI pipeline. Do not edit by hand.
// Sources: shaders/ui_scrgb.vert, shaders/ui_scrgb.frag
// Regenerate with: .\shaders\regen_ui_spv.ps1  (from the repo root)
#ifndef OPTIX_RAYTRACER_UI_SCRGB_SPV_H
#define OPTIX_RAYTRACER_UI_SCRGB_SPV_H

#include <cstdint>

$vertArray

$fragArray

#endif // OPTIX_RAYTRACER_UI_SCRGB_SPV_H
"@

[System.IO.File]::WriteAllText($outFile, $header, [System.Text.Encoding]::UTF8)
Write-Host "Written: $outFile"

# Clean up intermediate .spv files
Remove-Item $vertSpv, $fragSpv -ErrorAction SilentlyContinue
Write-Host "Done."
