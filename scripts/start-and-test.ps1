# Script pour démarrer l'infrastructure et tester le service
# Usage: .\scripts\start-and-test.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "DEMARRAGE ET TEST SERVICE INGESTIONIIOT" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Étape 1: Vérifier Docker
Write-Host "[1/4] Verification Docker Desktop" -ForegroundColor Yellow
try {
    docker ps > $null 2>&1
    Write-Host "  ✓ Docker Desktop est demarre" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Docker Desktop n'est pas demarre" -ForegroundColor Red
    Write-Host "  Veuillez demarrer Docker Desktop et relancer ce script" -ForegroundColor Yellow
    exit 1
}
Write-Host ""

# Étape 2: Démarrer l'infrastructure
Write-Host "[2/4] Demarrage Infrastructure Docker" -ForegroundColor Yellow
Write-Host "  Demarrage des services..." -ForegroundColor Gray
docker-compose -f infrastructure/docker-compose.yml up -d postgresql kafka minio redis zookeeper
Write-Host "  Attente 30 secondes pour le demarrage..." -ForegroundColor Gray
Start-Sleep -Seconds 30

# Vérifier les services
$services = @("postgresql", "kafka", "minio")
$allRunning = $true
foreach ($service in $services) {
    $container = docker ps --filter "name=$service" --format "{{.Names}}" 2>$null
    if ($container) {
        Write-Host "  ✓ $service est en cours d'execution" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $service n'est pas demarre" -ForegroundColor Red
        $allRunning = $false
    }
}

if (-not $allRunning) {
    Write-Host "  Certains services ne sont pas demarres. Attente supplementaire..." -ForegroundColor Yellow
    Start-Sleep -Seconds 30
}
Write-Host ""

# Étape 3: Configurer les variables d'environnement
Write-Host "[3/4] Configuration Variables d'Environnement" -ForegroundColor Yellow
$env:OPCUA_ENABLED = "false"
$env:DATABASE_HOST = "localhost"
$env:DATABASE_PORT = "5432"
$env:DATABASE_NAME = "predictive_maintenance"
$env:DATABASE_USER = "pmuser"
$env:DATABASE_PASSWORD = "pmpassword"
$env:KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
$env:MINIO_ENDPOINT = "http://localhost:9000"
$env:MINIO_ACCESS_KEY = "minioadmin"
$env:MINIO_SECRET_KEY = "minioadmin"
$env:SERVER_PORT = "8081"
Write-Host "  ✓ Variables configurees" -ForegroundColor Green
Write-Host ""

# Étape 4: Instructions pour démarrer le service
Write-Host "[4/4] Instructions pour Demarrer le Service" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Pour demarrer le service, executez dans un NOUVEAU terminal:" -ForegroundColor Yellow
Write-Host ""
Write-Host "  cd `"C:\Users\DELL\Desktop\Predictive Maintenance Projet\services\ingestion-iiot`"" -ForegroundColor White
Write-Host "  `$env:OPCUA_ENABLED = `"false`"" -ForegroundColor White
Write-Host "  `$env:DATABASE_HOST = `"localhost`"" -ForegroundColor White
Write-Host "  `$env:KAFKA_BOOTSTRAP_SERVERS = `"localhost:9092`"" -ForegroundColor White
Write-Host "  `$env:MINIO_ENDPOINT = `"http://localhost:9000`"" -ForegroundColor White
Write-Host "  mvn spring-boot:run -Dspring-boot.run.profiles=local" -ForegroundColor White
Write-Host ""
Write-Host "Attendez 30-60 secondes que le service demarre." -ForegroundColor Gray
Write-Host ""
Write-Host "Ensuite, dans un autre terminal, testez:" -ForegroundColor Yellow
Write-Host ""
Write-Host "  Invoke-RestMethod -Uri `"http://localhost:8081/api/v1/ingestion/health`" -Method GET" -ForegroundColor White
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan

