@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "QT_QPA_PLATFORM_PLUGIN_PATH="
set "QT_PLUGIN_PATH="
set "QT_QPA_PLATFORM="
echo ==========================================
echo   PCB Defect Detection System - YOLOv8
echo ==========================================
echo Starting GUI, please wait...
python main.py
if errorlevel 1 (
  echo.
  echo Failed to start. Please check Python and dependencies.
  pause
)
