@echo off
echo ===============================================
echo    Building N-Body Simulation - Phase 1
echo ===============================================

REM Create build directory
if not exist "build" mkdir build
cd build

REM Configure with CMake
echo Configuring with CMake...
cmake .. -G "Visual Studio 17 2022" -A x64

REM Build the project
echo Building project...
cmake --build . --config Release

REM Check if build was successful
if exist "bin\Release\nbody.exe" (
    echo.
    echo ===============================================
    echo Build completed successfully!
    echo Executables:
    echo   - bin\Release\nbody.exe
    echo   - bin\Release\benchmark.exe
    echo ===============================================
) else (
    echo.
    echo ===============================================
    echo Build failed! Check errors above.
    echo ===============================================
)

cd ..
pause 