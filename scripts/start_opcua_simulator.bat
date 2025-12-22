@echo off
REM Quick start script for OPC UA Simulator on Windows
echo ========================================
echo OPC UA Server Simulator - Quick Start
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Check if opcua library is installed
python -c "import opcua" >nul 2>&1
if errorlevel 1 (
    echo OPC UA library not found. Installing...
    pip install opcua
    if errorlevel 1 (
        echo ERROR: Failed to install opcua library
        echo Please run: pip install opcua
        pause
        exit /b 1
    )
)

echo Starting OPC UA Simulator...
echo.
python scripts\opcua_server_simulator.py

pause

