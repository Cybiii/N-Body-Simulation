@echo off
echo ===============================================
echo    Building N-Body Simulation
echo ===============================================

set BUILD_DIR=build
set CMAKE_GENERATOR="Visual Studio 17 2022"

echo Configuring with CMake...
cmake -S . -B %BUILD_DIR% -G %CMAKE_GENERATOR% -A x64

echo Building project...
cmake --build %BUILD_DIR% --config Release

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ===============================================
    echo Build failed! Check errors above.
    echo ===============================================
) else (
    echo.
    echo ===============================================
    echo Build successful! Executable is in %BUILD_DIR%/bin/Release
    echo ===============================================
)

echo.
pause 