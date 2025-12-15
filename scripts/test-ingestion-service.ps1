# Script de test pour le service IngestionIIoT
# Teste le démarrage, les connexions et les endpoints

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "TEST DU SERVICE INGESTIONIIOT" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$ErrorActionPreference = "Stop"

# Variables
$SERVICE_DIR = "services\ingestion-iiot"
$SERVICE_PORT = 8081
$BASE_URL = "http://localhost:$SERVICE_PORT"

# Fonction pour attendre que le service soit prêt
function Wait-ForService {
    param([int]$MaxAttempts = 30, [int]$DelaySeconds = 2)
    
    Write-Host "Attente du démarrage du service..." -ForegroundColor Yellow
    for ($i = 1; $i -le $MaxAttempts; $i++) {
        try {
            $response = Invoke-WebRequest -Uri "$BASE_URL/api/v1/ingestion/health" -Method GET -TimeoutSec 2 -ErrorAction SilentlyContinue
            if ($response.StatusCode -eq 200) {
                Write-Host "Service démarré avec succès !" -ForegroundColor Green
                return $true
            }
        } catch {
            Write-Host "Tentative $i/$MaxAttempts..." -ForegroundColor Gray
            Start-Sleep -Seconds $DelaySeconds
        }
    }
    return $false
}

# Test 1: Compilation Maven
Write-Host "TEST 1: Compilation Maven" -ForegroundColor Cyan
Write-Host "----------------------------------------" -ForegroundColor Cyan
try {
    Push-Location $SERVICE_DIR
    mvn clean compile -DskipTests
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Compilation réussie" -ForegroundColor Green
    } else {
        Write-Host "✗ Échec de la compilation" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "✗ Erreur lors de la compilation: $_" -ForegroundColor Red
    exit 1
} finally {
    Pop-Location
}
Write-Host ""

# Test 2: Vérification de l'infrastructure Docker
Write-Host "TEST 2: Vérification Infrastructure Docker" -ForegroundColor Cyan
Write-Host "----------------------------------------" -ForegroundColor Cyan
$services = @("postgres", "kafka", "minio", "redis")
$allRunning = $true

foreach ($service in $services) {
    $container = docker ps --filter "name=$service" --format "{{.Names}}" 2>$null
    if ($container) {
        Write-Host "✓ $service est en cours d'exécution" -ForegroundColor Green
    } else {
        Write-Host "✗ $service n'est pas en cours d'exécution" -ForegroundColor Red
        $allRunning = $false
    }
}

if (-not $allRunning) {
    Write-Host "Démarrage de l'infrastructure..." -ForegroundColor Yellow
    docker-compose -f infrastructure/docker-compose.yml up -d
    Start-Sleep -Seconds 10
}
Write-Host ""

# Test 3: Démarrage du service
Write-Host "TEST 3: Démarrage du Service Spring Boot" -ForegroundColor Cyan
Write-Host "----------------------------------------" -ForegroundColor Cyan
Write-Host "Démarrage du service en arrière-plan..." -ForegroundColor Yellow

try {
    Push-Location $SERVICE_DIR
    
    # Démarrer le service en arrière-plan
    $job = Start-Job -ScriptBlock {
        param($dir)
        Set-Location $dir
        mvn spring-boot:run
    } -ArgumentList (Resolve-Path $SERVICE_DIR)
    
    Write-Host "Service en cours de démarrage (Job ID: $($job.Id))..." -ForegroundColor Yellow
    
    # Attendre que le service soit prêt
    if (Wait-ForService -MaxAttempts 60 -DelaySeconds 3) {
        Write-Host "✓ Service démarré avec succès" -ForegroundColor Green
    } else {
        Write-Host "✗ Le service n'a pas démarré dans le délai imparti" -ForegroundColor Red
        Stop-Job $job
        Remove-Job $job
        exit 1
    }
} catch {
    Write-Host "✗ Erreur lors du démarrage: $_" -ForegroundColor Red
    if ($job) {
        Stop-Job $job
        Remove-Job $job
    }
    exit 1
} finally {
    Pop-Location
}
Write-Host ""

# Test 4: Health Check
Write-Host "TEST 4: Health Check Endpoint" -ForegroundColor Cyan
Write-Host "----------------------------------------" -ForegroundColor Cyan
try {
    $response = Invoke-RestMethod -Uri "$BASE_URL/api/v1/ingestion/health" -Method GET
    Write-Host "✓ Health check réussi" -ForegroundColor Green
    Write-Host "  Status: $($response.status)" -ForegroundColor Gray
    Write-Host "  Service: $($response.service)" -ForegroundColor Gray
} catch {
    Write-Host "✗ Health check échoué: $_" -ForegroundColor Red
}
Write-Host ""

# Test 5: Status Endpoint
Write-Host "TEST 5: Status Endpoint" -ForegroundColor Cyan
Write-Host "----------------------------------------" -ForegroundColor Cyan
try {
    $response = Invoke-RestMethod -Uri "$BASE_URL/api/v1/ingestion/status" -Method GET
    Write-Host "✓ Status check réussi" -ForegroundColor Green
    Write-Host "  Service: $($response.service)" -ForegroundColor Gray
    Write-Host "  Status: $($response.status)" -ForegroundColor Gray
} catch {
    Write-Host "✗ Status check échoué: $_" -ForegroundColor Red
}
Write-Host ""

# Test 6: Ingestion de données de test
Write-Host "TEST 6: Ingestion de Données de Test" -ForegroundColor Cyan
Write-Host "----------------------------------------" -ForegroundColor Cyan
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
    
    $headers = @{
        "Content-Type" = "application/json"
    }
    
    $response = Invoke-RestMethod -Uri "$BASE_URL/api/v1/ingestion/data" -Method POST -Body $testData -Headers $headers
    Write-Host "✓ Ingestion de données réussie" -ForegroundColor Green
    Write-Host "  Status: $($response.status)" -ForegroundColor Gray
} catch {
    Write-Host "✗ Ingestion de données échouée: $_" -ForegroundColor Red
    if ($_.Exception.Response) {
        $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
        $responseBody = $reader.ReadToEnd()
        Write-Host "  Réponse: $responseBody" -ForegroundColor Gray
    }
}
Write-Host ""

# Test 7: Vérification des connexions
Write-Host "TEST 7: Vérification des Connexions" -ForegroundColor Cyan
Write-Host "----------------------------------------" -ForegroundColor Cyan

# Kafka
Write-Host "Vérification Kafka..." -ForegroundColor Yellow
try {
    $kafkaTopics = docker exec kafka kafka-topics --list --bootstrap-server localhost:9092 2>$null
    if ($kafkaTopics -match "sensor-data") {
        Write-Host "✓ Kafka connecté et topic 'sensor-data' existe" -ForegroundColor Green
    } else {
        Write-Host "⚠ Kafka connecté mais topic 'sensor-data' non trouvé" -ForegroundColor Yellow
    }
} catch {
    Write-Host "✗ Erreur vérification Kafka: $_" -ForegroundColor Red
}

# PostgreSQL
Write-Host "Vérification PostgreSQL..." -ForegroundColor Yellow
try {
    $pgCheck = docker exec postgres psql -U postgres -d predictive_maintenance -c "SELECT COUNT(*) FROM assets;" 2>$null
    if ($pgCheck -match "\d+") {
        Write-Host "✓ PostgreSQL connecté et accessible" -ForegroundColor Green
    } else {
        Write-Host "⚠ PostgreSQL connecté mais tables non vérifiées" -ForegroundColor Yellow
    }
} catch {
    Write-Host "✗ Erreur vérification PostgreSQL: $_" -ForegroundColor Red
}

# MinIO
Write-Host "Vérification MinIO..." -ForegroundColor Yellow
try {
    $minioCheck = docker exec minio mc ls local/raw-sensor-data 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ MinIO connecté et bucket accessible" -ForegroundColor Green
    } else {
        Write-Host "⚠ MinIO connecté mais bucket non vérifié" -ForegroundColor Yellow
    }
} catch {
    Write-Host "✗ Erreur vérification MinIO: $_" -ForegroundColor Red
}
Write-Host ""

# Résumé
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "RÉSUMÉ DES TESTS" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Service URL: $BASE_URL" -ForegroundColor Gray
Write-Host "Health: $BASE_URL/api/v1/ingestion/health" -ForegroundColor Gray
Write-Host "Status: $BASE_URL/api/v1/ingestion/status" -ForegroundColor Gray
Write-Host ""
Write-Host "Pour arrêter le service, utilisez:" -ForegroundColor Yellow
Write-Host "  Stop-Job -Id $($job.Id)" -ForegroundColor Gray
Write-Host "  Remove-Job -Id $($job.Id)" -ForegroundColor Gray
Write-Host ""


