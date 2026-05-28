@echo off
setlocal

:: Usage:
::   build.bat          -- Debug build (configure once if needed)
::   build.bat Release  -- Release build

set CONFIG=%~1
if "%CONFIG%"=="" set CONFIG=Debug

:: -- Configure (runs only when no build cache exists) ---------------------

if not exist "build\CMakeCache.txt" (
    echo [build] Configuring...
    cmake -S . -B build -G "Visual Studio 17 2022" -A x64
    if errorlevel 1 (
        echo [build] Configure FAILED.
        exit /b 1
    )
)

:: -- Build ----------------------------------------------------------------

echo [build] Building %CONFIG%...
cmake --build build --config %CONFIG% --parallel
if errorlevel 1 (
    echo [build] Build FAILED.
    exit /b 1
)

echo [build] Done -- build\bin\%CONFIG%\OptixRaytracer.exe
