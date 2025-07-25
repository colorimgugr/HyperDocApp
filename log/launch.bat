@echo off

:: MATLAB path
set MATLABROOT=C:\Program Files\MATLAB\R2024b
set PATH=%MATLABROOT%\bin\win64;%MATLABROOT%\extern\bin\win64;%PATH%

:: lauch programme
start "" "%~dp0dist\MainWindow\MainWindow.exe"