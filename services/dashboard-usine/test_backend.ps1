# Script to test if the dashboard-usine backend is running
Write-Host "Testing dashboard-usine backend connection..." -ForegroundColor Cyan

$backendUrl = "http://localhost:8091"
$healthUrl = "$backendUrl/health"
$wsUrl = "ws://localhost:8091/ws/dashboard"

Write-Host "`n1. Testing HTTP health endpoint..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri $healthUrl -Method GET -TimeoutSec 3 -ErrorAction Stop
    Write-Host "✓ Backend is running!" -ForegroundColor Green
    Write-Host "  Status: $($response.StatusCode)" -ForegroundColor Green
    Write-Host "  Response: $($response.Content)" -ForegroundColor Gray
} catch {
    Write-Host "✗ Backend is NOT running or not accessible" -ForegroundColor Red
    Write-Host "  Error: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "`n  Please start the backend with:" -ForegroundColor Yellow
    Write-Host "    cd services/dashboard-usine/backend" -ForegroundColor Yellow
    Write-Host "    py -3.11 start_service.py" -ForegroundColor Yellow
    exit 1
}

Write-Host "`n2. Testing WebSocket endpoint..." -ForegroundColor Yellow
Write-Host "  WebSocket URL: $wsUrl" -ForegroundColor Gray
Write-Host "  Note: WebSocket connection will be tested by the frontend" -ForegroundColor Gray

Write-Host "`n✓ Backend appears to be running correctly!" -ForegroundColor Green
Write-Host "  API Base URL: $backendUrl/api/v1" -ForegroundColor Gray
Write-Host "  WebSocket URL: $wsUrl" -ForegroundColor Gray

