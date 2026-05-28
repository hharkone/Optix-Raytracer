# FindOptiX.cmake
# ───────────────
# Locates NVIDIA OptiX SDK (7.x / 8.x / 9.x) on Windows.
#
# Variables that influence the search (in priority order):
#   OptiX_INSTALL_DIR  — CMake cache variable  (-DOptiX_INSTALL_DIR=...)
#   OptiX_INSTALL_DIR  — Environment variable
#
# Provides:
#   OptiX_FOUND          — TRUE if found
#   OptiX_INCLUDE_DIR    — Path to directory containing optix.h
#   OptiX::OptiX         — INTERFACE imported target (header-only; no .lib)
#
# Usage:
#   find_package(OptiX REQUIRED)
#   target_link_libraries(myapp PRIVATE OptiX::OptiX)

# ── 1. Determine search roots ─────────────────────────────────────────────────

if(DEFINED OptiX_INSTALL_DIR)
    set(_optix_search_roots "${OptiX_INSTALL_DIR}")
elseif(DEFINED ENV{OptiX_INSTALL_DIR})
    set(_optix_search_roots "$ENV{OptiX_INSTALL_DIR}")
else()
    # Auto-discover: glob all "OptiX SDK *" directories under ProgramData
    file(GLOB _optix_candidates
        LIST_DIRECTORIES TRUE
        "C:/ProgramData/NVIDIA Corporation/OptiX SDK *"
    )
    # Sort descending so the newest version wins
    list(SORT _optix_candidates ORDER DESCENDING)
    set(_optix_search_roots ${_optix_candidates})
endif()

# ── 2. Find optix.h ───────────────────────────────────────────────────────────

set(_optix_include_search_paths "")
foreach(root IN LISTS _optix_search_roots)
    list(APPEND _optix_include_search_paths "${root}/include")
endforeach()

find_path(OptiX_INCLUDE_DIR
    NAMES optix.h
    PATHS ${_optix_include_search_paths}
    NO_DEFAULT_PATH
    DOC "Path to the directory containing optix.h"
)

# ── 3. Extract version from optix.h ──────────────────────────────────────────

if(OptiX_INCLUDE_DIR AND EXISTS "${OptiX_INCLUDE_DIR}/optix.h")
    file(STRINGS "${OptiX_INCLUDE_DIR}/optix.h" _optix_ver_line
        REGEX "#define OPTIX_VERSION [0-9]+"
        LIMIT_COUNT 1
    )
    if(_optix_ver_line MATCHES "#define OPTIX_VERSION ([0-9]+)")
        set(_v "${CMAKE_MATCH_1}")
        math(EXPR OptiX_VERSION_MAJOR "${_v} / 10000")
        math(EXPR OptiX_VERSION_MINOR "(${_v} % 10000) / 100")
        math(EXPR OptiX_VERSION_PATCH "${_v} % 100")
        set(OptiX_VERSION "${OptiX_VERSION_MAJOR}.${OptiX_VERSION_MINOR}.${OptiX_VERSION_PATCH}")
    endif()
endif()

# ── 4. Standard find_package handling ────────────────────────────────────────

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OptiX
    REQUIRED_VARS OptiX_INCLUDE_DIR
    VERSION_VAR   OptiX_VERSION
    FAIL_MESSAGE  "OptiX SDK not found. Set OptiX_INSTALL_DIR to the SDK root \
(e.g. -DOptiX_INSTALL_DIR=\"C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.1.0\")"
)

# ── 5. Create imported target ─────────────────────────────────────────────────

if(OptiX_FOUND AND NOT TARGET OptiX::OptiX)
    add_library(OptiX::OptiX INTERFACE IMPORTED GLOBAL)
    set_target_properties(OptiX::OptiX PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${OptiX_INCLUDE_DIR}"
    )
    message(STATUS "Found OptiX ${OptiX_VERSION}: ${OptiX_INCLUDE_DIR}")
endif()

mark_as_advanced(OptiX_INCLUDE_DIR)
