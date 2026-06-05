# InstallDefaultIni.cmake
# Copies the versioned default Dear ImGui layout next to the executable the
# first time it is missing.  Skipped when DST already exists so layouts that
# the user has customised during a session are never silently overwritten.
#
# Expected -D arguments (supplied by the POST_BUILD command in src/CMakeLists.txt):
#   SRC  — path to the source imgui.ini in the repository root
#   DST  — path to the desired destination next to the executable

if(NOT EXISTS "${DST}")
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E copy "${SRC}" "${DST}"
    )
    message(STATUS "Installed default imgui.ini → ${DST}")
endif()
