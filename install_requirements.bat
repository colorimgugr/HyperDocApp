@echo off
echo ===========================================
echo HyperTool Installation - Administrator Mode
echo ===========================================

:: Check if the script is run as administrator
net session >nul 2>&1
if %errorLevel% NEQ 0 (
    echo.
    echo [ERROR] This script must be run as administrator.
    echo Right-click this file and select "Run as administrator".
    pause
    exit /b
)

:: Change to the directory of this script
cd /d "%~dp0"

:: Run the Python installation script
echo.
echo Installing dependencies...
python install_requirements.py

echo.
echo Installation completed.
pause
