#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./build.sh          -- Release build (configure once if needed)
#   ./build.sh Debug    -- Debug build
#   ./build.sh --clean  -- Delete build cache, then Release build
#   ./build.sh --clean Debug -- Delete build cache, then Debug build
#
# OptiX SDK location (pick one):
#   export OptiX_INSTALL_DIR=~/NVIDIA-OptiX-SDK-9.1.0
#   ./build.sh -DOptiX_INSTALL_DIR=~/NVIDIA-OptiX-SDK-9.1.0

CONFIG=Release
CLEAN=0
EXTRA_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --clean) CLEAN=1 ;;
        Debug|Release|RelWithDebInfo|MinSizeRel) CONFIG="$arg" ;;
        *) EXTRA_ARGS+=("$arg") ;;
    esac
done

# -- Generator -----------------------------------------------------------------

if command -v ninja &>/dev/null; then
    GENERATOR="Ninja"
else
    GENERATOR="Unix Makefiles"
fi

# -- Compiler ------------------------------------------------------------------
# CUDA 13.x does not support GCC 16+. Prefer gcc-15 when available.

if command -v g++-15 &>/dev/null; then
    CXX_COMPILER="g++-15"
    CUDA_HOST_COMPILER="g++-15"
else
    CXX_COMPILER="g++"
    CUDA_HOST_COMPILER="g++"
fi

# -- Clean ---------------------------------------------------------------------

if [[ $CLEAN -eq 1 ]]; then
    echo "[build] Cleaning build cache..."
    rm -f  build/CMakeCache.txt
    rm -rf build/CMakeFiles
fi

# -- Configure -----------------------------------------------------------------

echo "[build] Configuring ($GENERATOR, $CONFIG, CXX=$CXX_COMPILER)..."
cmake -S . -B build \
    -G "$GENERATOR" \
    -DCMAKE_BUILD_TYPE="$CONFIG" \
    -DCMAKE_CXX_COMPILER="$CXX_COMPILER" \
    -DCMAKE_CUDA_HOST_COMPILER="$CUDA_HOST_COMPILER" \
    "${EXTRA_ARGS[@]}"

# -- Build ---------------------------------------------------------------------

echo "[build] Building $CONFIG..."
cmake --build build --parallel "$(nproc)"

echo "[build] Done -- build/bin/$CONFIG/OptixRaytracer"
