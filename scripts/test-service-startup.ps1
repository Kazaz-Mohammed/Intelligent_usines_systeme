# Script simplifié pour tester le démarrage du service IngestionIIoT
# Usage: .\scripts\test-service-startup.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "TEST DÉMARRAGE SERVICE INGESTIONIIOT" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$ErrorActionPreference = "Continue"

# Variables
$SERVICE_DIR = "services\ingestion-iiot"
$SERVICE_PORT = 8081
$BASE_URL = "http://localhost:$SERVICE_PORT"
$MAX_WAIT_TIME = 120 # secondes

# Test 1: Vérification infrastructure
Write-Host "[1/5] Vérification Infrastructure Docker" -ForegroundColor Yellow
Write-Host "----------------------------------------" -ForegroundColor Gray

$services = @{
    postgresql = "PostgreSQL/TimescaleDB"
    kafka = "Kafka"
    minio = "MinIO"
}

$allRunning = $true
foreach ($service in $services.Keys) {
    $container = docker ps --filter "name=$service" --format "{{.Names}}" 2>$null
    if ($container) {
        $status = docker ps --filter "name=$service" --format "{{.Status}}" 2>$null
        Write-Host "  ✓ $($services[$service]) : $status" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $($services[$service]) : Non démarré" -ForegroundColor Red
        $allRunning = $false
    }
}

if (-not $allRunning) {
    Write-Host ""
    Write-Host "Démarrage de l'infrastructure..." -ForegroundColor Yellow
    docker-compose -f infrastructure/docker-compose.yml up -d postgresql kafka minio redis
    Write-Host "Attente 30 secondes pour le démarrage..." -ForegroundColor Yellow
    Start-Sleep -Seconds 30
}
Write-Host ""

# Test 2: Compilation
Write-Host "[2/5] Compilation Maven" -ForegroundColor Yellow
Write-Host "----------------------------------------" -ForegroundColor Gray
try {
    Push-Location $SERVICE_DIR
    Write-Host "  Compilation en cours..." -ForegroundColor Gray
    mvn clean compile -DskipTests 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ Compilation réussie" -ForegroundColor Green
    } else {
        Write-Host "  ✗ Échec de la compilation" -ForegroundColor Red
        Pop-Location
        exit 1
    }
} catch {
    Write-Host "  ✗ Erreur: $_" -ForegroundColor Red
    Pop-Location
    exit 1
} finally {
    Pop-Location
}
Write-Host ""

# Test 3: Variables d'environnement
Write-Host "[3/5] Configuration Variables d'Environnement" -ForegroundColor Yellow
Write-Host "----------------------------------------" -ForegroundColor Gray

$env:DATABASE_HOST = "localhost"
$env:DATABASE_PORT = "5432"
$env:DATABASE_NAME = "predictive_maintenance"
$env:DATABASE_USER = "pmuser"
$env:DATABASE_PASSWORD = "pmpassword"
$env:KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
$env:MINIO_ENDPOINT = "http://localhost:9000"
$env:MINIO_ACCESS_KEY = "minioadmin"
$env:MINIO_SECRET_KEY = "minioadmin"
$env:OPCUA_ENABLED = "false"  # Désactiver OPC UA pour éviter erreurs
$env:SERVER_PORT = "8081"

Write-Host "  ✓ Variables d'environnement configurées" -ForegroundColor Green
Write-Host ""

# Test 4: Démarrage du service
Write-Host "[4/5] Démarrage du Service Spring Boot" -ForegroundColor Yellow
Write-Host "----------------------------------------" -ForegroundColor Gray
Write-Host "  Démarrage en cours (peut prendre 30-60 secondes)..." -ForegroundColor Gray
Write-Host "  Profil: local (OPC UA désactivé)" -ForegroundColor Gray
Write-Host ""

$serviceProcess = $null
try {
    Push-Location $SERVICE_DIR
    
    # Démarrer le service en arrière-plan
    $job = Start-Job -ScriptBlock {
        param($dir, $envVars)
        Set-Location $dir
        foreach ($key in $envVars.Keys) {
            [Environment]::SetEnvironmentVariable($key, $envVars[$key], "Process")
        }
        mvn spring-boot:run -Dspring-boot.run.profiles=local 2>&1
    } -ArgumentList (Resolve-Path $SERVICE_DIR), @{
        DATABASE_HOST = "localhost"
        DATABASE_PORT = "5432"
        DATABASE_NAME = "predictive_maintenance"
        DATABASE_USER = "pmuser"
        DATABASE_PASSWORD = "pmpassword"
        KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
        MINIO_ENDPOINT = "http://localhost:9000"
        MINIO_ACCESS_KEY = "minioadmin"
        MINIO_SECRET_KEY = "minioadmin"
        OPCUA_ENABLED = "false"
        SERVER_PORT = "8081"
    }
    
    Write-Host "  Job démarré (ID: $($job.Id))" -ForegroundColor Gray
    Write-Host "  Attente du démarrage du service..." -ForegroundColor Gray
    
    # Attendre que le service soit prêt
    $serviceReady = $false
    $attempts = 0
    $maxAttempts = $MAX_WAIT_TIME / 3
    
    while (-not $serviceReady -and $attempts -lt $maxAttempts) {
        Start-Sleep -Seconds 3
        $attempts++
        
        try {
            $response = Invoke-WebRequest -Uri "$BASE_URL/api/v1/ingestion/health" `
                -Method GET -TimeoutSec 2 -ErrorAction SilentlyContinue 2>$null
            if ($response.StatusCode -eq 200) {
                $serviceReady = $true
                Write-Host "  ✓ Service démarré avec succès (après $($attempts * 3) secondes)" -ForegroundColor Green
            }
        } catch {
            Write-Host "  . Tentative $attempts/$maxAttempts..." -ForegroundColor DarkGray
        }
    }
    
    if (-not $serviceReady) {
        Write-Host "  ✗ Le service n'a pas démarré dans le délai imparti" -ForegroundColor Red
        Write-Host "  Arrêt du job..." -ForegroundColor Yellow
        Stop-Job $job -ErrorAction SilentlyContinue
        Remove-Job $job -ErrorAction SilentlyContinue
        Pop-Location
        exit 1
    }
    
    $serviceProcess = $job
    
} catch {
    Write-Host "  ✗ Erreur lors du démarrage: $_" -ForegroundColor Red
    if ($serviceProcess) {
        Stop-Job $serviceProcess -ErrorAction SilentlyContinue
        Remove-Job $serviceProcess -ErrorAction SilentlyContinue
    }
    Pop-Location
    exit 1
} finally {
    Pop-Location
}
Write-Host ""

# Test 5: Tests des endpoints
Write-Host "[5/5] Tests des Endpoints REST" -ForegroundColor Yellow
Write-Host "----------------------------------------" -ForegroundColor Gray

# Health
Write-Host "  Test Health endpoint..." -ForegroundColor Gray
try {
    $response = Invoke-RestMethod -Uri "$BASE_URL/api/v1/ingestion/health" -Method GET
    Write-Host "    ✓ Health: $($response.status) - $($response.service)" -ForegroundColor Green
} catch {
    Write-Host "    ✗ Health check échoué: $_" -ForegroundColor Red
}

# Status
Write-Host "  Test Status endpoint..." -ForegroundColor Gray
try {
    $response = Invoke-RestMethod -Uri "$BASE_URL/api/v1/ingestion/status" -Method GET
    Write-Host "    ✓ Status: $($response.status)" -ForegroundColor Green
} catch {
    Write-Host "    ✗ Status check échoué: $_" -ForegroundColor Red
}

# Ingestion de données
Write-Host "  Test Ingestion endpoint..." -ForegroundColor Gray
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
    $response = Invoke-RestMethod -Uri "$BASE_URL/api/v1/ingestion/data" `
        -Method POST -Body $testData -Headers $headers
    Write-Host "    ✓ Ingestion: $($response.status)" -ForegroundColor Green
} catch {
    Write-Host "    ✗ Ingestion échouée: $_" -ForegroundColor Red
}
Write-Host ""

# Résumé
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "RÉSUMÉ" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Service URL: $BASE_URL" -ForegroundColor Gray
Write-Host "Health: $BASE_URL/api/v1/ingestion/health" -ForegroundColor Gray
Write-Host "Status: $BASE_URL/api/v1/ingestion/status" -ForegroundColor Gray
Write-Host ""
Write-Host "Service en cours d'exécution (Job ID: $($serviceProcess.Id))" -ForegroundColor Green
Write-Host ""
Write-Host "Pour arrêter le service:" -ForegroundColor Yellow
Write-Host "  Stop-Job -Id $($serviceProcess.Id)" -ForegroundColor Gray
Write-Host "  Remove-Job -Id $($serviceProcess.Id)" -ForegroundColor Gray
Write-Host ""

