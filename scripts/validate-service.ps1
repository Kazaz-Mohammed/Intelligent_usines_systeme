# Script pour valider que le service IngestionIIoT est opérationnel
# Usage: .\scripts\validate-service.ps1

$SERVICE_PORT = 8081
$BASE_URL = "http://localhost:$SERVICE_PORT"
$MAX_ATTEMPTS = 40
$DELAY_SECONDS = 3

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "VALIDATION SERVICE INGESTIONIIOT" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Attendre que le service soit prêt
Write-Host "Attente du démarrage du service..." -ForegroundColor Yellow
$serviceReady = $false
$attempts = 0

while ($serviceReady -eq $false -and $attempts -lt $MAX_ATTEMPTS) {
    try {
        $response = Invoke-WebRequest -Uri "$BASE_URL/api/v1/ingestion/health" -Method GET -TimeoutSec 2 -ErrorAction SilentlyContinue
        if ($null -ne $response -and $response.StatusCode -eq 200) {
            $serviceReady = $true
            Write-Host "✓ Service demarre et accessible" -ForegroundColor Green
        } else {
            $attempts = $attempts + 1
            if ($attempts % 5 -eq 0) {
                Write-Host "  Tentative $attempts/$MAX_ATTEMPTS..." -ForegroundColor Gray
            }
            Start-Sleep -Seconds $DELAY_SECONDS
        }
    } catch {
        $attempts = $attempts + 1
        if ($attempts % 5 -eq 0) {
            Write-Host "  Tentative $attempts/$MAX_ATTEMPTS..." -ForegroundColor Gray
        }
        Start-Sleep -Seconds $DELAY_SECONDS
    }
}

if ($serviceReady -eq $false) {
    Write-Host "✗ Le service n'est pas accessible apres $($MAX_ATTEMPTS * $DELAY_SECONDS) secondes" -ForegroundColor Red
    Write-Host "Verifiez les logs du service pour plus d'informations" -ForegroundColor Yellow
    exit 1
}

Write-Host ""

# Test 1: Health Check
Write-Host "[1/3] Test Health Endpoint" -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$BASE_URL/api/v1/ingestion/health" -Method GET
    Write-Host "  ✓ Status: $($response.status)" -ForegroundColor Green
    Write-Host "  ✓ Service: $($response.service)" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Echec: $_" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Test 2: Status Endpoint
Write-Host "[2/3] Test Status Endpoint" -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$BASE_URL/api/v1/ingestion/status" -Method GET
    Write-Host "  ✓ Service: $($response.service)" -ForegroundColor Green
    Write-Host "  ✓ Status: $($response.status)" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Echec: $_" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Test 3: Ingestion de donnees
Write-Host "[3/3] Test Ingestion de Donnees" -ForegroundColor Yellow
try {
    $testData = @{
        timestamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
        assetId = "ASSET001"
        sensorId = "SENSOR001"
        value = 25.5
        unit = "°C"
        quality = 2
        sourceType = "TEST"
    } | ConvertTo-Json
    
    $headers = @{ "Content-Type" = "application/json" }
    $response = Invoke-RestMethod -Uri "$BASE_URL/api/v1/ingestion/data" -Method POST -Body $testData -Headers $headers
    
    Write-Host "  ✓ Ingestion reussie" -ForegroundColor Green
    Write-Host "  ✓ Status: $($response.status)" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Echec: $_" -ForegroundColor Red
    if ($null -ne $_.Exception.Response) {
        $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
        $responseBody = $reader.ReadToEnd()
        Write-Host "  Reponse: $responseBody" -ForegroundColor Gray
    }
    exit 1
}
Write-Host ""

# Resume
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "✓ TOUS LES TESTS REUSSIS" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Service URL: $BASE_URL" -ForegroundColor Gray
Write-Host "Health: $BASE_URL/api/v1/ingestion/health" -ForegroundColor Gray
Write-Host "Status: $BASE_URL/api/v1/ingestion/status" -ForegroundColor Gray
Write-Host ""
