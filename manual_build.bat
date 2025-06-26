@echo off
echo ===============================================
echo    Building and Running N-Body Simulation
echo    (Must be run from VS Developer Command Prompt)
echo ===============================================

set BUILD_DIR=build

REM Step 1: Configure with CMake
echo.
echo [1/3] Configuring with CMake...
cmake -S . -B %BUILD_DIR% -G "Visual Studio 17 2022" -A x64
if %ERRORLEVEL% NEQ 0 (
    echo CMake configuration failed. Exiting.
    pause
    exit /b 1
)

REM Step 2: Build the project
echo.
echo [2/3] Building project...
cmake --build %BUILD_DIR% --config Release
if %ERRORLEVEL% NEQ 0 (
    echo Build failed. Exiting.
    pause
    exit /b 1
)

REM Step 3: Run the benchmark executable
echo.
echo [3/3] Running benchmark...
echo ===============================================
%BUILD_DIR%\\bin\\Release\\benchmark.exe
echo ===============================================

echo.
echo Script finished successfully.
pause 