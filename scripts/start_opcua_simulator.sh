#!/bin/bash
# Quick start script for OPC UA Simulator on Linux/Mac

echo "========================================"
echo "OPC UA Server Simulator - Quick Start"
echo "========================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8+ and try again"
    exit 1
fi

# Check if opcua library is installed
if ! python3 -c "import opcua" 2>/dev/null; then
    echo "OPC UA library not found. Installing..."
    pip3 install opcua
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install opcua library"
        echo "Please run: pip3 install opcua"
        exit 1
    fi
fi

echo "Starting OPC UA Simulator..."
echo ""
python3 scripts/opcua_server_simulator.py

