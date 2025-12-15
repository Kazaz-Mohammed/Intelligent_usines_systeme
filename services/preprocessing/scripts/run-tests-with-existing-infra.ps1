# Script PowerShell pour exécuter les tests avec l'infrastructure existante
# Utilise l'infrastructure Docker existante (Kafka, PostgreSQL)

param(
    [string]$TestType = "all",
    [switch]$Coverage = $false,
    [switch]$BuildImage = $false
)

Write-Host "=== Exécution des tests avec infrastructure existante ===" -ForegroundColor Cyan
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
Write-Host "[INFO] Construction image: $BuildImage" -ForegroundColor Green

# Vérifier que l'infrastructure est démarrée
Write-Host "[INFO] Vérification de l'infrastructure..." -ForegroundColor Green

$infrastructureDir = Join-Path $PSScriptRoot "..\..\infrastructure"
if (-not (Test-Path $infrastructureDir)) {
    Write-Host "[ERROR] Répertoire infrastructure non trouvé: $infrastructureDir" -ForegroundColor Red
    exit 1
}

# Vérifier que Kafka est démarré
$kafkaRunning = docker ps --filter "name=kafka" --format "{{.Names}}" | Select-String -Pattern "kafka"
if (-not $kafkaRunning) {
    Write-Host "[WARN] Kafka n'est pas démarré" -ForegroundColor Yellow
    Write-Host "[INFO] Démarrage de l'infrastructure..." -ForegroundColor Yellow
    Push-Location $infrastructureDir
    docker-compose up -d
    Pop-Location
    
    # Attendre que Kafka soit prêt
    Write-Host "[INFO] Attente que Kafka soit prêt..." -ForegroundColor Yellow
    Start-Sleep -Seconds 30
    
    # Vérifier à nouveau
    $kafkaRunning = docker ps --filter "name=kafka" --format "{{.Names}}" | Select-String -Pattern "kafka"
    if (-not $kafkaRunning) {
        Write-Host "[ERROR] Kafka n'est toujours pas démarré" -ForegroundColor Red
        exit 1
    }
}

Write-Host "[INFO] Kafka est démarré" -ForegroundColor Green

# Vérifier que PostgreSQL est démarré
$postgresRunning = docker ps --filter "name=postgresql" --format "{{.Names}}" | Select-String -Pattern "postgresql"
if (-not $postgresRunning) {
    Write-Host "[WARN] PostgreSQL n'est pas démarré" -ForegroundColor Yellow
    Write-Host "[INFO] Démarrage de l'infrastructure..." -ForegroundColor Yellow
    Push-Location $infrastructureDir
    docker-compose up -d postgresql
    Pop-Location
    
    # Attendre que PostgreSQL soit prêt
    Write-Host "[INFO] Attente que PostgreSQL soit prêt..." -ForegroundColor Yellow
    Start-Sleep -Seconds 20
    
    # Vérifier à nouveau
    $postgresRunning = docker ps --filter "name=postgresql" --format "{{.Names}}" | Select-String -Pattern "postgresql"
    if (-not $postgresRunning) {
        Write-Host "[ERROR] PostgreSQL n'est toujours pas démarré" -ForegroundColor Red
        exit 1
    }
}

Write-Host "[INFO] PostgreSQL est démarré" -ForegroundColor Green

# Vérifier que le réseau existe
Write-Host "[INFO] Vérification du réseau Docker..." -ForegroundColor Green
$networkExists = docker network ls --format "{{.Name}}" | Select-String -Pattern "predictive-maintenance-network"
if (-not $networkExists) {
    Write-Host "[WARN] Réseau predictive-maintenance-network non trouvé" -ForegroundColor Yellow
    Write-Host "[INFO] Le réseau doit être créé par l'infrastructure Docker" -ForegroundColor Yellow
    Write-Host "[INFO] Démarrer l'infrastructure : cd infrastructure && docker-compose up -d" -ForegroundColor Cyan
} else {
    Write-Host "[INFO] Réseau predictive-maintenance-network existe" -ForegroundColor Green
}

# Construire l'image si nécessaire
if ($BuildImage) {
    Write-Host "[INFO] Construction de l'image de test..." -ForegroundColor Green
    
    # Essayer d'abord avec python:3.11 (sans slim)
    Write-Host "[INFO] Essai avec python:3.11..." -ForegroundColor Yellow
    $dockerfileContent = Get-Content "Dockerfile.test" -Raw
    $dockerfileContent = $dockerfileContent -replace "FROM python:3.11-slim", "FROM python:3.11"
    Set-Content "Dockerfile.test.tmp" -Value $dockerfileContent
    
    docker build -f Dockerfile.test.tmp -t preprocessing-test:latest . 2>&1 | Out-Null
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[INFO] Image construite avec succès (python:3.11)" -ForegroundColor Green
        Remove-Item "Dockerfile.test.tmp" -ErrorAction SilentlyContinue
    } else {
        Write-Host "[WARN] Impossible de construire l'image avec python:3.11" -ForegroundColor Yellow
        Write-Host "[INFO] Utilisation des tests locaux à la place..." -ForegroundColor Yellow
        Remove-Item "Dockerfile.test.tmp" -ErrorAction SilentlyContinue
        
        # Utiliser les tests locaux
        & "$PSScriptRoot\run-tests-local.ps1" -TestType $TestType -Coverage:$Coverage
        exit $LASTEXITCODE
    }
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
        Write-Host "Usage: .\run-tests-with-existing-infra.ps1 [-TestType unit|integration|all] [-Coverage] [-BuildImage]"
        exit 1
    }
}

# Vérifier si l'image existe
$imageExists = docker images --format "{{.Repository}}:{{.Tag}}" | Select-String -Pattern "preprocessing-test:latest"
if (-not $imageExists) {
    Write-Host "[WARN] Image preprocessing-test:latest n'existe pas" -ForegroundColor Yellow
    Write-Host "[INFO] Utilisation des tests locaux à la place..." -ForegroundColor Yellow
    
    # Utiliser les tests locaux
    & "$PSScriptRoot\run-tests-local.ps1" -TestType $TestType -Coverage:$Coverage
    exit $LASTEXITCODE
}

# Exécuter les tests dans Docker
docker-compose -f docker-compose.test.yml run --rm preprocessing-test bash -c $testCommand

$testExitCode = $LASTEXITCODE

# Afficher les résultats
if ($testExitCode -eq 0) {
    Write-Host "[INFO] Tests réussis!" -ForegroundColor Green
} else {
    Write-Host "[ERROR] Tests échoués avec le code: $testExitCode" -ForegroundColor Red
}

# Afficher le rapport de couverture (si généré)
if ($Coverage -and (Test-Path "htmlcov/index.html")) {
    Write-Host "[INFO] Rapport de couverture généré: htmlcov/index.html" -ForegroundColor Green
    Write-Host "[INFO] Ouvrir le rapport avec: start htmlcov/index.html" -ForegroundColor Cyan
}

exit $testExitCode

