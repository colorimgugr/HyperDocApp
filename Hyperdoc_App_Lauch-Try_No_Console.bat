@echo off
cd /d "%~dp0"
start "" pythonw.exe MainWindow.py >nul 2>&1


:: Check if pythonw.exe is available
where pythonw.exe >nul 2>&1
if %errorlevel%==0 (
    echo Launching with pythonw (no console)...
    start "" pythonw.exe MainWindow.py
    exit /b
)

:: If pythonw is not found, fallback to python
where python.exe >nul 2>&1
if %errorlevel%==0 (
    echo pythonw.exe not found. Launching with python (console will stay open)...
    python MainWindow.py
    exit /b
)

:: If neither is found, show error
echo [ERROR] Neither pythonw.exe nor python.exe was found in PATH.
echo Please install Python and make sure it is added to the system PATH.
pause
