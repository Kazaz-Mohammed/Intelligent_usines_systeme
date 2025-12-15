# Script simple pour tester le service
$url = "http://localhost:8081/api/v1/ingestion/health"
try {
    $response = Invoke-RestMethod -Uri $url -Method GET -TimeoutSec 5
    Write-Host "Service OK: $($response.status)" -ForegroundColor Green
    Write-Host "Service: $($response.service)" -ForegroundColor Green
    exit 0
} catch {
    Write-Host "Service non accessible: $_" -ForegroundColor Red
    exit 1
}

