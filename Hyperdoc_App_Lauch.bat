@echo off
echo ===========================================
echo Launching HyperTool Main Window
echo ===========================================

:: Change to the directory of this script
cd /d "%~dp0"

:: Run the Python script
echo.
echo Starting application...
python MainWindow.py

:: Check for error and pause only if there was a problem
if errorlevel 1 (
    echo.
    echo [ERROR] The application encountered a problem.
    pause
)
