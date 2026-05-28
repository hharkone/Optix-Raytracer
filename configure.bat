@echo off
setlocal

:: Usage:
::   configure.bat          -- (re)generate the Visual Studio solution
::   configure.bat --clean  -- delete the build cache first, then regenerate

if /i "%~1"=="--clean" (
    echo [configure] Cleaning build cache...
    if exist "build\CMakeCache.txt" del /f /q "build\CMakeCache.txt"
    if exist "build\CMakeFiles"     rmdir /s /q "build\CMakeFiles"
)

echo [configure] Generating Visual Studio solution...
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
if errorlevel 1 (
    echo [configure] FAILED.
    exit /b 1
)

echo [configure] Done -- open build\OptixRaytracer.sln in Visual Studio.
