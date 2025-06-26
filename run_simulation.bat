@echo off
REM This script launches the simulation inside the VS 2022 Developer Command Prompt
REM to ensure all environment variables for MSVC and CUDA are correctly set.

set VS_DEV_CMD_PATH="C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set PROJECT_DIR=%~dp0

if not exist %VS_DEV_CMD_PATH% (
    echo Error: Visual Studio Developer Command Prompt not found at %VS_DEV_CMD_PATH%
    pause
    exit /b 1
)

echo =================================================================
echo Launching in VS 2022 Developer Command Prompt...
echo =================================================================

rem Start a new cmd instance, call vcvars64.bat to set up the environment,
rem then execute our build and run commands.
cmd /k "%VS_DEV_CMD_PATH% && cd /d %PROJECT_DIR% && echo. && echo Running simulation... && build\\bin\\Release\\nbody_simulation.exe" 