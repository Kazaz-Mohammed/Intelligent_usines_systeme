# Script PowerShell pour exécuter les tests localement (sans Docker)
# Utiliser cette version si Docker Hub n'est pas accessible

param(
    [string]$TestType = "all",
    [switch]$Coverage = $false
)

Write-Host "=== Exécution des tests localement (sans Docker) ===" -ForegroundColor Cyan
Write-Host ""

# Vérifier que Python est installé
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] Python n'est pas installé ou non disponible dans le PATH" -ForegroundColor Red
    exit 1
}

Write-Host "[INFO] Python est disponible" -ForegroundColor Green

# Vérifier que pip est installé
if (-not (Get-Command pip -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] pip n'est pas installé ou non disponible dans le PATH" -ForegroundColor Red
    exit 1
}

Write-Host "[INFO] pip est disponible" -ForegroundColor Green

Write-Host "[INFO] Type de test: $TestType" -ForegroundColor Green
Write-Host "[INFO] Couverture: $Coverage" -ForegroundColor Green

# Installer les dépendances
Write-Host "[INFO] Installation des dépendances..." -ForegroundColor Green
pip install -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Échec de l'installation des dépendances" -ForegroundColor Red
    exit 1
}

Write-Host "[INFO] Dépendances installées avec succès" -ForegroundColor Green

# Vérifier que l'infrastructure est démarrée (optionnel)
Write-Host "[INFO] Vérification de l'infrastructure..." -ForegroundColor Green

$kafkaRunning = docker ps --filter "name=kafka" --format "{{.Names}}" | Select-String -Pattern "kafka"
$postgresRunning = docker ps --filter "name=postgresql" --format "{{.Names}}" | Select-String -Pattern "postgresql"

if ($kafkaRunning -and $postgresRunning) {
    Write-Host "[INFO] Infrastructure Docker démarrée" -ForegroundColor Green
    
    # Configurer les variables d'environnement pour les tests d'intégration
    $env:KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
    $env:DATABASE_HOST = "localhost"
    $env:DATABASE_PORT = "5432"
    $env:DATABASE_NAME = "predictive_maintenance"
    $env:DATABASE_USER = "pmuser"
    $env:DATABASE_PASSWORD = "pmpassword"
    
    Write-Host "[INFO] Variables d'environnement configurées pour les tests d'intégration" -ForegroundColor Green
} else {
    Write-Host "[WARN] Infrastructure Docker non démarrée" -ForegroundColor Yellow
    Write-Host "[INFO] Les tests d'intégration nécessitent Kafka et PostgreSQL" -ForegroundColor Yellow
    Write-Host "[INFO] Pour démarrer l'infrastructure :" -ForegroundColor Yellow
    Write-Host "  cd infrastructure" -ForegroundColor Cyan
    Write-Host "  docker-compose up -d" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "[INFO] Continuation avec les tests unitaires uniquement..." -ForegroundColor Yellow
    
    # Forcer les tests unitaires uniquement
    if ($TestType -eq "all" -or $TestType -eq "integration") {
        Write-Host "[INFO] Passage aux tests unitaires uniquement (infrastructure non disponible)" -ForegroundColor Yellow
        $TestType = "unit"
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
        Write-Host "Usage: .\run-tests-local.ps1 [-TestType unit|integration|all] [-Coverage]"
        exit 1
    }
}

# Exécuter les tests
Invoke-Expression $testCommand

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

