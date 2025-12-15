@echo off
REM Start script for dashboard-usine service using Python 3.11
cd /d %~dp0
echo Starting dashboard-usine service on port 8091...
echo Using Python 3.11
py -3.11 start_service.py
pause

