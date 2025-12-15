# Script PowerShell pour exécuter les tests avec Docker

param(
    [string]$TestType = "all",
    [switch]$Coverage = $false,
    [switch]$KeepServices = $false
)

Write-Host "=== Exécution des tests avec Docker ===" -ForegroundColor Cyan
Write-Host ""

# Vérifier que Docker est disponible
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] Docker n'est pas installé ou non disponible dans le PATH" -ForegroundColor Red
    exit 1
}

Write-Host "[INFO] Docker est disponible" -ForegroundColor Green

# Vérifier que docker-compose est disponible
if (-not (Get-Command docker-compose -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] docker-compose n'est pas installé ou non disponible dans le PATH" -ForegroundColor Red
    exit 1
}

Write-Host "[INFO] docker-compose est disponible" -ForegroundColor Green

Write-Host "[INFO] Type de test: $TestType" -ForegroundColor Green
Write-Host "[INFO] Couverture: $Coverage" -ForegroundColor Green

# Construire l'image de test
Write-Host "[INFO] Construction de l'image de test..." -ForegroundColor Green
docker build -f Dockerfile.test -t preprocessing-test:latest .

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Échec de la construction de l'image" -ForegroundColor Red
    exit 1
}

Write-Host "[INFO] Image construite avec succès" -ForegroundColor Green

# Créer le réseau si nécessaire
Write-Host "[INFO] Création du réseau Docker..." -ForegroundColor Green
docker network create predictive-maintenance 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARN] Réseau déjà existant" -ForegroundColor Yellow
}

# Démarrer les services dépendants
Write-Host "[INFO] Démarrage des services dépendants (Kafka, PostgreSQL)..." -ForegroundColor Green
docker-compose -f docker-compose.test.yml up -d kafka-test zookeeper-test postgresql-test

# Attendre que les services soient prêts
Write-Host "[INFO] Attente que les services soient prêts..." -ForegroundColor Green
Start-Sleep -Seconds 30

# Vérifier que Kafka est prêt
Write-Host "[INFO] Vérification de Kafka..." -ForegroundColor Green
$timeout = 60
$elapsed = 0
$kafkaReady = $false

while (-not $kafkaReady -and $elapsed -lt $timeout) {
    $result = docker exec kafka-test nc -z localhost 9092 2>$null
    if ($LASTEXITCODE -eq 0) {
        $kafkaReady = $true
        Write-Host "[INFO] Kafka est prêt" -ForegroundColor Green
        break
    }
    Start-Sleep -Seconds 2
    $elapsed += 2
}

if (-not $kafkaReady) {
    Write-Host "[ERROR] Kafka n'est pas prêt après $timeout secondes" -ForegroundColor Red
    exit 1
}

# Vérifier que PostgreSQL est prêt
Write-Host "[INFO] Vérification de PostgreSQL..." -ForegroundColor Green
$timeout = 60
$elapsed = 0
$postgresReady = $false

while (-not $postgresReady -and $elapsed -lt $timeout) {
    $result = docker exec postgresql-test pg_isready -U pmuser -d predictive_maintenance 2>$null
    if ($LASTEXITCODE -eq 0) {
        $postgresReady = $true
        Write-Host "[INFO] PostgreSQL est prêt" -ForegroundColor Green
        break
    }
    Start-Sleep -Seconds 2
    $elapsed += 2
}

if (-not $postgresReady) {
    Write-Host "[ERROR] PostgreSQL n'est pas prêt après $timeout secondes" -ForegroundColor Red
    exit 1
}

# Exécuter les tests
Write-Host "[INFO] Exécution des tests..." -ForegroundColor Green

$testCommand = ""

switch ($TestType) {
    "unit" {
        Write-Host "[INFO] Tests unitaires uniquement" -ForegroundColor Green
        $testCommand = "pytest tests/ -v --tb=short -m 'not integration'"
    }
    "integration" {
        Write-Host "[INFO] Tests d'intégration uniquement" -ForegroundColor Green
        $testCommand = "pytest tests/ -v --tb=short -m integration"
    }
    "all" {
        Write-Host "[INFO] Tous les tests" -ForegroundColor Green
        if ($Coverage) {
            $testCommand = "pytest tests/ -v --tb=short --cov=app --cov-report=html --cov-report=term-missing"
        } else {
            $testCommand = "pytest tests/ -v --tb=short"
        }
    }
    default {
        Write-Host "[ERROR] Type de test inconnu: $TestType" -ForegroundColor Red
        Write-Host "Usage: .\run-tests-docker.ps1 [-TestType unit|integration|all] [-Coverage] [-KeepServices]"
        exit 1
    }
}

docker-compose -f docker-compose.test.yml run --rm preprocessing-test bash -c $testCommand

$testExitCode = $LASTEXITCODE

# Afficher les résultats
if ($testExitCode -eq 0) {
    Write-Host "[INFO] Tests réussis!" -ForegroundColor Green
} else {
    Write-Host "[ERROR] Tests échoués avec le code: $testExitCode" -ForegroundColor Red
}

# Arrêter les services (optionnel)
if (-not $KeepServices) {
    $response = Read-Host "Arrêter les services? (y/n)"
    if ($response -eq "y" -or $response -eq "Y") {
        Write-Host "[INFO] Arrêt des services..." -ForegroundColor Green
        docker-compose -f docker-compose.test.yml down
    }
} else {
    Write-Host "[INFO] Services maintenus en cours d'exécution" -ForegroundColor Green
}

exit $testExitCode

