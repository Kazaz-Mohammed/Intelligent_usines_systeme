# PowerShell script to start dashboard-usine service with Python 3.11
Write-Host "Starting dashboard-usine service on port 8091..." -ForegroundColor Cyan
Write-Host "Using Python 3.11" -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host ""

Set-Location $PSScriptRoot
py -3.11 start_service.py

